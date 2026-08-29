# -*- coding: utf-8 -*-
"""호가 깊이 실시간 수집기 — 충격의 분모 D(u).

왜 필요한가
  모델 충격항이 임계형 max(0, V/D - c)^beta 가 되었으므로(MODEL.md v2 §3),
  D 없이는 임계 초과 여부를 판정할 수 없다. 과거분은 data.binance.vision/bookDepth 로
  T-1 까지 받지만, 극단 이벤트(연 수 회)는 실시간으로 잡지 않으면 영구 결손이다.

관측 한계 — 실측으로 확인한 것 (2026-07-31)
  WS depth20 : ±0.1% 정도만 덮는다. -0.5% 이상은 전부 NaN. 사용 불가.
  REST limit=1000 (최대) 도달 범위:
      BTCUSDT  0.18%    ETHUSDT  0.56%    SOLUSDT  13.7%    WLDUSDT  99.9%
  틱 사이즈가 가격 대비 작을수록 1000레벨이 좁은 폭만 덮는다.
  => **BTC/ETH의 -1% 깊이는 공개 API로 실시간 관측이 불가능하다.**

  따라서 이 수집기는 '도달 가능한 만큼'을 기록하고, 도달하지 못한 구간은 0이 아니라
  NaN 으로 남긴다. reach_pct 를 같이 저장해 커버리지를 항상 알 수 있게 한다.
  더 깊은 구간이 필요하면 bookDepth 일별 파일(T-1)이나 유료 L2 피드를 써야 한다.

  다만 청산 시장가는 최우선호가부터 바깥으로 먹어 들어가므로, 근접 깊이(0.05~0.2%)가
  즉각적 충격의 분모로는 오히려 더 적절하다. 과거 분석에서 -0.2%와 -1% 어느 쪽으로
  나눠도 R2가 0.000이었다는 점도 이 선택을 지지한다.

실행:
    python collectors/depth_poll.py
    python collectors/depth_poll.py --symbols BTCUSDT --seconds 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

DEPTH_URL = C.BINANCE_FAPI + "/fapi/v1/depth"
LIMIT = 1000                 # 최대. weight 20
BANDS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]      # 현재가 대비 %
POLL_SEC = 30
FLUSH_SEC = 120
REQ_INTERVAL_S = 0.30        # 21종 x weight20 = 420/라운드, 30초 간격이면 840/분 (상한 2400)

DTYPES: dict[str, str] = {
    "ts_ms": "int64", "symbol": "string", "mid": "float64",
    "bid_reach_pct": "float64", "ask_reach_pct": "float64",
    "bid_total": "float64", "ask_total": "float64",
    "n_bids": "int64", "n_asks": "int64",
}
for _b in BANDS:
    tag = str(_b).replace(".", "_")
    DTYPES["bid_%s" % tag] = "float64"
    DTYPES["ask_%s" % tag] = "float64"


def _empty() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in DTYPES.items()})


def to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    for col, dt in DTYPES.items():
        if col not in df.columns:
            df[col] = np.nan
        try:
            df[col] = df[col].astype(dt)
        except (TypeError, ValueError):
            df[col] = (df[col].astype("string") if dt == "string"
                       else pd.to_numeric(df[col], errors="coerce"))
    return df[list(DTYPES.keys())]


def summarize(symbol: str, book: dict, ts: int) -> dict | None:
    try:
        b = [(float(p), float(q)) for p, q in book.get("bids", []) if float(q) > 0]
        a = [(float(p), float(q)) for p, q in book.get("asks", []) if float(q) > 0]
    except (TypeError, ValueError):
        return None
    if not b or not a:
        return None
    mid = 0.5 * (b[0][0] + a[0][0])
    if mid <= 0:
        return None
    b_reach = (mid - b[-1][0]) / mid * 100.0
    a_reach = (a[-1][0] - mid) / mid * 100.0

    rec = {"ts_ms": ts, "symbol": symbol, "mid": mid,
           "bid_reach_pct": b_reach, "ask_reach_pct": a_reach,
           "bid_total": sum(p * q for p, q in b),
           "ask_total": sum(p * q for p, q in a),
           "n_bids": len(b), "n_asks": len(a)}
    for band in BANDS:
        tag = str(band).replace(".", "_")
        # 도달 못 한 구간은 0이 아니라 NaN. 0으로 채우면 '깊이가 없다'로 오독된다.
        rec["bid_%s" % tag] = (sum(p * q for p, q in b if p >= mid * (1 - band / 100))
                               if band <= b_reach else np.nan)
        rec["ask_%s" % tag] = (sum(p * q for p, q in a if p <= mid * (1 + band / 100))
                               if band <= a_reach else np.nan)
    return rec


def flush(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = to_frame(rows)
    ts = U.utc_now_ms()
    day = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000.0))
    U.atomic_write_parquet(df, os.path.join(C.DEPTH_DIR, day, "depth_%d.parquet" % ts))
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance order-book depth poller")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--interval", type=float, default=POLL_SEC)
    ap.add_argument("--seconds", type=float, default=None, help="run N seconds then exit")
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    lock = U.acquire_single_instance(C.LOGS, "depth_poll")
    if lock is None:
        U.log("another depth_poll instance is already running -> exit")
        return 0

    symbols = a.symbols if a.symbols else C.MAJORS
    session = U.make_session(C.USER_AGENT)
    buf: list[dict] = []
    last_flush = time.monotonic()
    started = time.monotonic()
    total = 0
    U.log("depth_poll start: %d symbols, limit=%d, every %.0fs"
          % (len(symbols), LIMIT, a.interval))

    try:
        while True:
            if a.seconds is not None and time.monotonic() - started >= a.seconds:
                break
            slot = time.monotonic()
            stats: dict = {}
            n_ok = 0
            for s in symbols:
                time.sleep(REQ_INTERVAL_S)
                try:
                    book = U.get_json(session, DEPTH_URL, {"symbol": s, "limit": LIMIT},
                                      timeout=C.BINANCE_HTTP_TIMEOUT_S,
                                      max_retry=C.BINANCE_MAX_RETRY,
                                      backoff_base=C.BINANCE_BACKOFF_BASE_S, stats=stats)
                except U.FetchError:
                    continue
                rec = summarize(s, book, U.utc_now_ms())
                if rec:
                    buf.append(rec)
                    n_ok += 1

            if (time.monotonic() - last_flush) >= FLUSH_SEC or len(buf) >= 5000:
                n = flush(buf)
                buf.clear()
                last_flush = time.monotonic()
                if n:
                    total += n
                    U.log("flushed %d rows (total %d) | ok=%d/%d 429=%d"
                          % (n, total, n_ok, len(symbols), stats.get("n_429", 0)))

            elapsed = time.monotonic() - slot
            if elapsed < a.interval:
                time.sleep(a.interval - elapsed)
            else:
                U.log("overrun: round took %.0fs (interval %.0fs)" % (elapsed, a.interval))
    except KeyboardInterrupt:
        U.log("interrupted -> exit")
    finally:
        total += flush(buf)
        session.close()
        U.log("stopped. total rows %d" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
