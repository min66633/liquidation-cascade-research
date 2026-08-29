# -*- coding: utf-8 -*-
"""Bybit allLiquidation 실시간 수집기 — L(p) 보정 신호 + 실현 청산 관측치.

왜 Bybit인가 (2026-07-31 실측)
  Binance !forceOrder@arr 를 90초간 전 종목 구독 -> **0건**.
  Bybit allLiquidation 을 같은 90초간 4종목만 구독 -> 109메시지 / **380 청산 레코드**.
  Binance 스트림은 문서상 초당 심볼별 1건 스냅샷인데 실측으로는 사실상 무용이었다.
  Bybit는 2025-02-25부터 전건을 500ms 주기로 내보내며 **가격과 크기를 모두** 준다.

이 데이터의 두 가지 용도
  1) Y변수 — 시장 전체 실현 청산. 캐스케이드 이벤트의 정답 레이블.
  2) **L(p) 온라인 보정** — 청산이 실제로 터진 가격은 잠재 청산분포 L(p)에서 뽑힌
     관측치다. 예측한 연료가 두꺼운 구간을 가격이 통과했는데 청산이 적게 나왔다면
     그 구간의 추정이 틀린 것이다. 이를 되먹여 L(p) 추정을 갱신할 수 있다.

필드 의미 (공식 문서 확인)
  T = 갱신 타임스탬프(ms)
  s = 심볼
  S = **청산된 포지션의 방향**. "Buy" 이면 롱 포지션이 청산된 것(시장에는 매도가 나감).
      직관과 반대라 주의 — 구 liquidation 토픽과 의미가 다르다.
  v = 체결 수량
  p = **파산가(bankruptcy price)**. 체결가가 아니다. L(p) 보정에는 오히려 이쪽이 맞다.

히스토리가 없다 — 지금부터 쌓는 수밖에 없다(Tardis.dev 유료로 2025-02-25 이후 백필 가능).

실행:
    python collectors/bybit_liq_ws.py
    python collectors/bybit_liq_ws.py --symbols BTCUSDT ETHUSDT --seconds 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

WS_URL = "wss://stream.bybit.com/v5/public/linear"
SUB_BATCH = 10                 # 한 subscribe 메시지당 토픽 수
FLUSH_SEC = 60
FLUSH_ROWS = 5000
APP_PING_SEC = 20              # Bybit는 앱 레벨 ping을 요구한다

DTYPES = {
    "exch_ms": "int64", "recv_ms": "int64", "symbol": "string",
    "pos_side": "string", "size": "float64", "bankruptcy_px": "float64",
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in DTYPES.items()})


def to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    for col, dt in DTYPES.items():
        if col not in df.columns:
            df[col] = pd.NA
        try:
            df[col] = df[col].astype(dt)
        except (TypeError, ValueError):
            df[col] = (pd.to_numeric(df[col], errors="coerce")
                       if dt != "string" else df[col].astype("string"))
    return df[list(DTYPES.keys())]


def flush(rows: list[dict]) -> int:
    """버퍼를 date 파티션 parquet으로 원자적 저장. 저장한 행 수 반환."""
    if not rows:
        return 0
    df = to_frame(rows)
    ts = U.utc_now_ms()
    day = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000.0))
    path = os.path.join(C.BYBIT_LIQ_DIR, day, "liq_%d.parquet" % ts)
    U.atomic_write_parquet(df, path)
    return len(df)


def pick_symbols(top_n: int) -> list[str]:
    """거래대금 상위 USDT 무기한 종목을 고른다. 실패하면 폴백 목록."""
    import requests
    try:
        s = U.make_session(C.USER_AGENT)
        d = U.get_json(s, C.BYBIT_TICKERS_URL, {}, timeout=30, max_retry=2,
                       backoff_base=2.0)
        s.close()
        rows = (d.get("result") or {}).get("list") or []
        rows = [r for r in rows if str(r.get("symbol", "")).endswith("USDT")]
        rows.sort(key=lambda r: -U.to_float(r.get("turnover24h")))
        syms = [r["symbol"] for r in rows[:top_n]]
        if syms:
            U.log("symbol pick: %d of %d USDT perps by 24h turnover" % (len(syms), len(rows)))
            return syms
    except (requests.RequestException, U.FetchError, KeyError, TypeError) as e:
        U.log("symbol pick failed (%s) -> fallback list" % e)
    return list(C.BYBIT_SYMBOLS)


def parse(msg: dict, recv_ms: int) -> list[dict]:
    out = []
    for d in msg.get("data") or []:
        if not isinstance(d, dict):
            continue
        side = str(d.get("S", ""))
        # S="Buy" -> 롱 포지션이 청산됨 (공식 문서). 헷갈리기 쉬워 여기서 명시 변환한다.
        pos_side = "long" if side == "Buy" else ("short" if side == "Sell" else "")
        out.append({
            "exch_ms": int(U.to_float(d.get("T")) or 0),
            "recv_ms": recv_ms,
            "symbol": str(d.get("s", "")),
            "pos_side": pos_side,
            "size": U.to_float(d.get("v")),
            "bankruptcy_px": U.to_float(d.get("p")),
        })
    return out


async def consume(symbols: list[str], run_seconds: float | None) -> None:
    import websockets

    topics = ["allLiquidation.%s" % s for s in symbols]
    buf: list[dict] = []
    last_flush = time.monotonic()
    started = time.monotonic()
    total = 0
    backoff = 1.0

    while True:
        if run_seconds is not None and time.monotonic() - started >= run_seconds:
            break
        try:
            async with websockets.connect(WS_URL, ping_interval=None,
                                          open_timeout=20, close_timeout=5) as ws:
                for i in range(0, len(topics), SUB_BATCH):
                    await ws.send(json.dumps({"op": "subscribe",
                                              "args": topics[i:i + SUB_BATCH]}))
                U.log("connected: %d topics" % len(topics))
                backoff = 1.0
                last_ping = time.monotonic()

                while True:
                    if run_seconds is not None and time.monotonic() - started >= run_seconds:
                        break
                    now = time.monotonic()
                    if now - last_ping >= APP_PING_SEC:
                        await ws.send(json.dumps({"op": "ping"}))
                        last_ping = now
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=APP_PING_SEC)
                    except asyncio.TimeoutError:
                        raw = None
                    if raw is not None:
                        try:
                            msg = json.loads(raw)
                        except ValueError:
                            msg = None
                        if msg and str(msg.get("topic", "")).startswith("allLiquidation"):
                            buf.extend(parse(msg, U.utc_now_ms()))

                    # 타임아웃 경로에서도 반드시 flush를 확인한다. 청산이 뜸한 구간에서
                    # continue로 건너뛰면 버퍼가 무한정 남아 크래시 시 통째로 유실된다.
                    if len(buf) >= FLUSH_ROWS or (time.monotonic() - last_flush) >= FLUSH_SEC:
                        n = flush(buf)
                        total += n
                        if n:
                            U.log("flushed %d rows (total %d)" % (n, total))
                        buf.clear()
                        last_flush = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            U.log("ws error: %s: %s -> reconnect in %.0fs" % (type(e).__name__, e, backoff))
            # 재접속 전에 버퍼를 비워 데이터 유실을 막는다.
            n = flush(buf)
            if n:
                total += n
                U.log("flushed %d rows before reconnect (total %d)" % (n, total))
            buf.clear()
            last_flush = time.monotonic()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    n = flush(buf)
    total += n
    U.log("stopped. total rows %d" % total)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bybit allLiquidation collector")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--top-n", type=int, default=C.BYBIT_TOP_N,
                    help="subscribe to the top N USDT perps by 24h turnover")
    ap.add_argument("--seconds", type=float, default=None, help="run for N seconds then exit (test)")
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    lock = U.acquire_single_instance(C.LOGS, "bybit_liq_ws")
    if lock is None:
        U.log("another bybit_liq_ws instance is already running -> exit")
        return 0

    symbols = a.symbols if a.symbols else pick_symbols(a.top_n)
    U.log("bybit_liq start: %d symbols" % len(symbols))
    try:
        asyncio.run(consume(symbols, a.seconds))
    except KeyboardInterrupt:
        U.log("interrupted -> exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
