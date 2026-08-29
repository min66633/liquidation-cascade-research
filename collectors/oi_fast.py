# -*- coding: utf-8 -*-
"""고빈도 미결제약정(OI) 수집기 — 설계의 방아쇠를 실시간으로 만든다.

왜 이것이 필요한가 (2026-08-03 분석 결과)
  캐스케이드 되돌림을 실제로 예측하게 만드는 유일한 부품이 확인됐다:
  **디레버리징의 크기** = 짧은 시간에 OI 가 몇 % 사라지는가.

    가격 급변만으로 방아쇠      -> 우위 없음 (전부 <= +0.3bp)  realtime_trigger.py
    + 청산 프린트 확인          -> 우위 없음 (더미 t=-0.1)     liq_trigger.py
    + **OI 급감 확인**          -> 모형 지정가 **+90.4bp (t=3.2)**  intra_event.py

  청산 프린트가 왜 안 되는가: OI 감소의 약 1% 만 태그된 표본이라 **크기를 못 잰다**
  (scale_check.py 1절). '청산이 있었나' 는 급변 사건의 68.6% 에서 참이라 정보가 0이다.

  그런데 내 분석의 '5분 지연' 은 **데이터 아티팩트**였다. `/futures/data/openInterestHist`
  가 5분봉으로만 보관되기 때문이지, 실시간 제약이 아니다.
  **`/fapi/v1/openInterest` 는 현재 OI 를 즉시 준다.**

가중치 예산 (상한 2400/분)
  openInterest : 심볼당 weight 1  -> 21종 x (60/5초) = 252/분
  premiumIndex : 전종목 1회 weight 10 -> 120/분   (markPrice 로 명목가 환산)
  합계 약 372/분 = 상한의 15%. 여유 있다.

저장
  data/oi_fast/<날짜>/oi_<ts>.parquet
    ts_ms(수신), exch_ms(거래소), symbol, oi(계약), mark, oi_usd

주의
  과거 백테스트는 불가능하다(5분봉밖에 없다). **순방향 수집만**이 답이며,
  며칠 쌓이면 intra_event.py 의 +90.4bp 를 룩어헤드 없이 재검정할 수 있다.

실행:
    python collectors/oi_fast.py                 # 상시 루프
    python collectors/oi_fast.py --interval 5    # 폴링 주기(초)
    python collectors/oi_fast.py --once          # 1회 점검
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

OUT = os.path.join(C.DATA, "oi_fast")
URL_OI = C.BINANCE_FAPI + "/fapi/v1/openInterest"
URL_PREM = C.BINANCE_FAPI + "/fapi/v1/premiumIndex"
FLUSH_SEC = 60
WORKERS = 7


def fetch_marks(session, stats) -> dict:
    """전종목 markPrice 1회 조회(weight 10). 실패 시 빈 dict."""
    try:
        data = U.get_json(session, URL_PREM, {}, timeout=8.0, max_retry=2,
                          backoff_base=0.5, stats=stats)
    except U.FetchError:
        return {}
    if not isinstance(data, list):
        return {}
    return {r.get("symbol"): U.to_float(r.get("markPrice")) for r in data}


def fetch_oi(session, symbol: str, stats) -> tuple | None:
    """단일 심볼 현재 OI. (exch_ms, symbol, oi) 또는 None."""
    try:
        d = U.get_json(session, URL_OI, {"symbol": symbol}, timeout=8.0,
                       max_retry=2, backoff_base=0.4, stats=stats)
    except U.FetchError:
        return None
    if not isinstance(d, dict) or "openInterest" not in d:
        return None
    return (int(d.get("time") or 0), symbol, U.to_float(d.get("openInterest")))


def flush(rows: list) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=["ts_ms", "exch_ms", "symbol", "oi",
                                     "mark", "oi_usd"])
    day = pd.to_datetime(df["ts_ms"].iloc[0], unit="ms").strftime("%Y-%m-%d")
    d = os.path.join(OUT, day)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "oi_%d.parquet" % int(df["ts_ms"].iloc[-1]))
    df.to_parquet(p, index=False)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser(description="high-frequency open interest poller")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--interval", type=float, default=5.0, help="폴링 주기(초)")
    ap.add_argument("--hours", type=float, default=0.0, help="0이면 무한")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    U.init_stdout()
    C.ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    # 중복 실행 방지 — 둘 뜨면 API 가중치가 배가 되고 같은 스냅샷이 두 번 저장된다.
    lock = U.acquire_single_instance(C.LOGS, "oi_fast")
    if lock is None:
        U.log("another oi_fast instance is already running -> exit")
        return 0
    syms = a.symbols if a.symbols else C.BINANCE_SYMBOLS

    session = U.make_session(C.USER_AGENT)
    stats: dict = {}
    stop_at = time.time() + (a.hours * 3600 if a.hours > 0 else 10 ** 9)

    U.log("oi_fast 시작 | %d종 | 주기 %.1f초 | 예상 가중치 %.0f/분"
          % (len(syms), a.interval, (len(syms) + 10) * (60.0 / a.interval)))

    rows: list = []
    last_flush = time.time()
    n_cycle = n_ok = n_fail = 0
    marks: dict = {}
    last_mark = 0.0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while time.time() < stop_at:
            t_cycle = time.time()
            # markPrice 는 20초에 한 번이면 충분하다(명목가 환산용)
            if t_cycle - last_mark > 20.0:
                m = fetch_marks(session, stats)
                if m:
                    marks = m
                last_mark = t_cycle

            res = list(pool.map(lambda s: fetch_oi(session, s, stats), syms))
            ts = U.now_ms() if hasattr(U, "now_ms") else int(time.time() * 1000)
            for r in res:
                if r is None:
                    n_fail += 1
                    continue
                exch_ms, sym, oi = r
                mk = marks.get(sym, float("nan"))
                rows.append((ts, exch_ms, sym, oi, mk, oi * mk))
                n_ok += 1
            n_cycle += 1

            if a.once:
                break
            if time.time() - last_flush >= FLUSH_SEC:
                n = flush(rows)
                rows = []
                last_flush = time.time()
                U.log("flush %d행 | 사이클 %d | 성공 %d 실패 %d | 최근주기 %.2f초"
                      % (n, n_cycle, n_ok, n_fail, time.time() - t_cycle))
                n_ok = n_fail = 0
            sleep = a.interval - (time.time() - t_cycle)
            if sleep > 0:
                time.sleep(sleep)

    n = flush(rows)
    if a.once:
        df = pd.DataFrame(rows, columns=["ts_ms", "exch_ms", "symbol", "oi",
                                         "mark", "oi_usd"])
        print(df.to_string())
        print("\n총 OI 명목가 $%.4g" % df["oi_usd"].sum())
    U.log("종료 | 마지막 flush %d행" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
