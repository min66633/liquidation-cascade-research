# -*- coding: utf-8 -*-
"""Tardis.dev 무료 샘플 청산 데이터 다운로더 (매월 1일).

Tardis는 '매월 1일' 데이터셋을 API 키 없이 공개한다(실측 확인: 1일=200, 2일=401).
Bybit 청산은 2020-12-18부터 제공되므로, 매월 1일씩 모으면 약 5년치에 걸친
67일 분량의 **청산 원본(마이크로초 타임스탬프 + 가격 + 수량)** 을 무료로 얻는다.

이 데이터가 왜 필요한가
  선생님 질문 = "큰 청산 물량 V가 터지면 가격이 얼마나 더 밀리나" -> 가격충격 함수.
  이건 시간축이 아니라 (청산량, 가격변위) 쌍의 문제이고, 5분 OI로는 청산 규모를
  정확히 못 잰다(OI 변화에는 자발적 청산·신규진입이 섞인다). Tardis 청산 원본은
  강제청산만 따로, 가격과 수량까지 준다.

중요한 주의 — 2025-02-25 이전 데이터의 편향
  Bybit는 2025-02-25에 allLiquidation(전건, 500ms) 토픽을 열었다. 그 이전 데이터는
  구 liquidation 토픽 기반이고 **심볼당 초당 1건**으로 스로틀되어 있었다. 따라서
  2025-02-25 이전 구간은 청산 '건수와 규모가 과소'집계되어 있고, 격렬할수록 누락이
  커지는 체계적 편향을 갖는다. 분석 시 시기를 나눠서 볼 것.

실행:
    python downloaders/tardis_liq.py
    python downloaders/tardis_liq.py --symbols BTCUSDT ETHUSDT --start 2024-01
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

BASE = "https://datasets.tardis.dev/v1/bybit/liquidations"
RAW = os.path.join(C.DATA, "tardis_liq", "_raw")
OUT = os.path.join(C.DATA, "tardis_liq")
FIRST_MONTH = (2020, 12)          # Bybit 청산 제공 시작
# 전건 피드(allLiquidation) 전환일. 이 이전은 초당 1건 스로틀 -> 과소집계.
FULL_FEED_FROM_MS = 1740441600000  # 2025-02-25T00:00:00Z


def months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    out, y, m = [], start[0], start[1]
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch(session: requests.Session, symbol: str, y: int, m: int) -> str | None:
    dest = os.path.join(RAW, symbol, "%04d-%02d-01.csv.gz" % (y, m))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = "%s/%04d/%02d/01/%s.csv.gz" % (BASE, y, m, symbol)
    try:
        r = session.get(url, timeout=120)
    except requests.RequestException as e:
        U.log("fetch error %s %04d-%02d: %s" % (symbol, y, m, e))
        return None
    if r.status_code in (401, 403, 404):
        return None                      # 무료 구간이 아니거나 데이터 없음 — 정상
    if r.status_code != 200:
        U.log("HTTP %d %s %04d-%02d" % (r.status_code, symbol, y, m))
        return None
    try:                                  # gzip 유효성 확인 후 저장
        gzip.GzipFile(fileobj=io.BytesIO(r.content)).read(64)
    except OSError:
        U.log("bad gzip %s %04d-%02d" % (symbol, y, m))
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = "%s.%d.tmp" % (dest, os.getpid())
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def parse(symbol: str, paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, compression="gzip")
        except Exception as e:
            U.log("unreadable %s (%s)" % (os.path.basename(p), e))
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"timestamp": "ts_us", "local_timestamp": "recv_us"})
    for c in ("ts_us", "recv_us", "price", "amount"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["ts_us", "price", "amount"])
    d["ts_ms"] = (d["ts_us"] // 1000).astype("int64")
    d["symbol"] = symbol
    # Tardis의 side는 '청산 주문의 방향'이다. sell = 롱 포지션이 강제 매도된 것.
    d["pos_side"] = d["side"].astype(str).str.lower().map(
        {"sell": "long", "buy": "short"}).fillna("")
    d["notional"] = d["price"] * d["amount"]
    d["full_feed"] = d["ts_ms"] >= FULL_FEED_FROM_MS
    return (d[["ts_ms", "symbol", "pos_side", "price", "amount", "notional", "full_feed"]]
            .sort_values("ts_ms").reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Tardis free monthly Bybit liquidation samples")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--start", default="2020-12", help="YYYY-MM")
    ap.add_argument("--end", default=None, help="YYYY-MM (default: last month)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    os.makedirs(RAW, exist_ok=True)
    symbols = a.symbols if a.symbols else C.BINANCE_SYMBOLS
    sy, sm = (int(x) for x in a.start.split("-"))
    if a.end:
        ey, em = (int(x) for x in a.end.split("-"))
    else:
        t = date.today()
        ey, em = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    ms = months((max(sy, FIRST_MONTH[0]), sm), (ey, em))

    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})
    U.log("tardis liq: %d symbols x %d months (free 1st-of-month samples)"
          % (len(symbols), len(ms)))
    try:
        for s in symbols:
            paths = []
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(fetch, session, s, y, m) for y, m in ms]
                for fu in as_completed(futs):
                    p = fu.result()
                    if p:
                        paths.append(p)
            d = parse(s, sorted(paths))
            if d.empty:
                U.log("%s: no data" % s)
                continue
            U.atomic_write_parquet(d, os.path.join(OUT, "%s.parquet" % s))
            full = d[d.full_feed]
            U.log("%s: %d days, %d liquidations (≥2025-02-25 full-feed: %d), $%.1fM total"
                  % (s, len(paths), len(d), len(full), d.notional.sum() / 1e6))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
