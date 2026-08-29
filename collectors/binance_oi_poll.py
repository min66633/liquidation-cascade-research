# -*- coding: utf-8 -*-
"""Binance USD-M 선물 파생지표 5분봉 적재기 — 30일 보관 한계 구제용.

/futures/data/* 계열(openInterestHist, topLongShortPositionRatio,
globalLongShortAccountRatio, takerlongshortRatio 등)은 전부 '최근 30일'만
제공한다. 오늘 적재를 시작하지 않으면 그 이전 구간은 무료로는 영구 복원 불가다.

역할: 시장 전체 레버리지 축적 상태(Hyperliquid는 BTC 글로벌 OI의 약 16%에 불과).
청산 식별 4분면 — OI감소+가격하락=롱청산 / OI감소+가격상승=숏청산 /
OI증가+가격급변=신규진입(캐스케이드 아님).

실행:
    python collectors/binance_oi_poll.py           # 상시 루프
    python collectors/binance_oi_poll.py --once    # 1회만 (점검용)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402


def series_path(series_name: str, symbol: str) -> str:
    return os.path.join(C.BINANCE_DIR, series_name, "%s.parquet" % symbol)


def fetch_series(session, endpoint: str, symbol: str, value_fields: list[str],
                 ts_field: str, stats: dict) -> pd.DataFrame | None:
    """단일 (엔드포인트, 심볼) 조회 → 정규화된 DataFrame. 실패 시 None."""
    url = C.BINANCE_FAPI + endpoint
    params = {"symbol": symbol, "period": C.BINANCE_PERIOD, "limit": C.BINANCE_LIMIT}
    try:
        data = U.get_json(session, url, params, timeout=C.BINANCE_HTTP_TIMEOUT_S,
                          max_retry=C.BINANCE_MAX_RETRY,
                          backoff_base=C.BINANCE_BACKOFF_BASE_S, stats=stats)
    except U.FetchError as e:
        U.log("fetch failed %s %s: %s" % (endpoint, symbol, e))
        return None

    # 에러 응답은 리스트가 아니라 {"code":-1121,"msg":"..."} 형태로 온다.
    if isinstance(data, dict):
        U.log("api error %s %s: %s" % (endpoint, symbol, str(data)[:160]))
        return None
    if not isinstance(data, list) or not data:
        return None

    out = {"timestamp": [], "symbol": []}
    for f in value_fields:
        out[f] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        ts = U.to_float(row.get(ts_field))
        if ts != ts:                                  # NaN 검사
            continue
        out["timestamp"].append(int(ts))
        # takerlongshortRatio 응답에는 symbol 필드가 없다 → 요청값으로 채운다.
        out["symbol"].append(str(row.get("symbol") or symbol))
        for f in value_fields:
            out[f].append(U.to_float(row.get(f)))

    if not out["timestamp"]:
        return None
    df = pd.DataFrame(out)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["symbol"] = df["symbol"].astype("string")
    for f in value_fields:
        df[f] = df[f].astype("float64")
    df["ingested_ms"] = U.utc_now_ms()
    return df


def merge_write(df: pd.DataFrame, path: str) -> tuple[int, int]:
    """기존 parquet과 병합(timestamp 기준 중복 제거) 후 원자적 저장.

    읽기 실패 시 절대 덮어쓰지 않는다. Binance는 30일만 서빙하므로 이 파일이
    유일본이고, 한 번의 일시적 읽기 오류(Defender 스캔, 분석 노트북이 잡고 있는
    핸들, 클라우드 동기화)로 덮어쓰면 수개월치가 영구 소멸한다. 대신 격리하고
    새 파일로 시작하되, 격리조차 실패하면 예외를 올려 쓰기를 포기한다.

    반환: (총 행수, 이번에 새로 늘어난 행수)
    """
    old = U.read_parquet_or_quarantine(path)
    before = 0 if old is None else len(old)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    # 같은 타임스탬프는 최신 조회분을 남긴다(미마감 봉이 나중에 확정되므로).
    # concat이 old→new 순서라 keep='last'가 곧 '새로 받은 값'이다.
    df = (df.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True))
    U.atomic_write_parquet(df, path)
    return len(df), len(df) - before


def poll_once(session, symbols: list[str]) -> dict:
    stats: dict = {}
    n_req = n_ok = 0
    added_total = 0
    for series_name, endpoint, ts_field, value_fields in C.BINANCE_SERIES:
        for symbol in symbols:
            n_req += 1
            time.sleep(C.BINANCE_REQ_INTERVAL_S)
            df = fetch_series(session, endpoint, symbol, value_fields, ts_field, stats)
            if df is None:
                continue
            try:
                total, added = merge_write(df, series_path(series_name, symbol))
            except Exception as e:
                U.log("write failed %s %s: %s" % (series_name, symbol, e))
                continue
            n_ok += 1
            added_total += added
    U.log("poll: %d/%d ok, +%d new rows, 429=%d neterr=%d"
          % (n_ok, n_req, added_total, stats.get("n_429", 0), stats.get("n_neterr", 0)))
    return {"n_req": n_req, "n_ok": n_ok, "added": added_total}


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance futures 5m derivatives-data collector")
    ap.add_argument("--once", action="store_true", help="single poll then exit")
    ap.add_argument("--symbols", nargs="*", default=None, help="override symbol list")
    ap.add_argument("--interval", type=float, default=C.BINANCE_POLL_INTERVAL_S)
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    lock = U.acquire_single_instance(C.LOGS, "binance_oi_poll")
    if lock is None:
        U.log("another binance_oi_poll instance is already running -> exit")
        return 0
    symbols = a.symbols if a.symbols else C.BINANCE_SYMBOLS
    session = U.make_session(C.USER_AGENT)

    U.log("binance_oi start: %d symbols x %d series, period=%s interval=%ds"
          % (len(symbols), len(C.BINANCE_SERIES), C.BINANCE_PERIOD, a.interval))

    try:
        while True:
            slot = time.monotonic()
            try:
                poll_once(session, symbols)
            except Exception as e:
                U.log("poll aborted: %s: %s" % (type(e).__name__, e))
            if a.once:
                return 0
            elapsed = time.monotonic() - slot
            if elapsed < a.interval:
                time.sleep(a.interval - elapsed)
            else:
                U.log("overrun: poll took %.0fs (interval %.0fs)" % (elapsed, a.interval))
    except KeyboardInterrupt:
        U.log("interrupted -> exit")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
