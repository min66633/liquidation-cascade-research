# -*- coding: utf-8 -*-
"""특정 날짜만 골라 Binance 1분봉을 받는다 (디스크 절약용).

왜 필요한가
  Tardis 무료 청산 샘플은 '매월 1일'만 존재한다(약 81일). 그 81일의 가격만 있으면
  impact/continuity 분석이 되는데, binance_bulk.py 로 21종 전체 이력을 받으면
  1분봉만 12GB 가까이 된다. 필요한 날짜만 받으면 100MB 수준이다.

산출물은 binance_bulk 와 같은 위치/형식(klines_1m/<SYMBOL>.parquet)이라
기존 분석 코드가 그대로 읽는다. 이미 전체 이력을 받아둔 심볼은 건드리지 않는다.

실행:
    python downloaders/binance_days.py                 # MAJORS x Tardis 보유일
    python downloaders/binance_days.py --symbols AVAXUSDT --interval 1m
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from downloaders.binance_bulk import KLINE_COLS, read_zip_csv   # noqa: E402

BASE = "https://data.binance.vision/data/futures/um/daily/klines"
BULK = os.path.join(C.DATA, "binance_bulk")
RAW = os.path.join(BULK, "_raw")


def tardis_days() -> list[str]:
    """Tardis 청산 데이터가 존재하는 날짜(YYYY-MM-DD) 목록."""
    p = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError("run downloaders/tardis_multi.py first")
    d = pd.read_parquet(p, columns=["ts_ms"])
    days = pd.to_datetime(d["ts_ms"], unit="ms").dt.strftime("%Y-%m-%d").unique()
    return sorted(days)


def fetch_day(session: requests.Session, symbol: str, interval: str, day: str) -> str | None:
    dest = os.path.join(RAW, symbol, "klines_" + interval, "%s.zip" % day)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = "%s/%s/%s/%s-%s-%s.zip" % (BASE, symbol, interval, symbol, interval, day)
    try:
        r = session.get(url, timeout=90)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        zipfile.ZipFile(io.BytesIO(r.content)).testzip()
    except zipfile.BadZipFile:
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = "%s.%d.tmp" % (dest, os.getpid())
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def build(symbol: str, interval: str, paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_zip_csv(p, KLINE_COLS)
        if df is None or df.empty:
            continue
        keep = ["open_time", "open", "high", "low", "close", "volume",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
        frames.append(df[[c for c in keep if c in df.columns]].copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open_time", "close"])
    big = out["open_time"] > 1e14              # 일부 zip이 마이크로초 단위
    out.loc[big, "open_time"] = out.loc[big, "open_time"] // 1000
    out["open_time"] = out["open_time"].astype("int64")
    out["symbol"] = symbol
    return (out.drop_duplicates(subset=["open_time"], keep="last")
               .sort_values("open_time").reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch Binance 1m klines for selected days only")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    symbols = a.symbols if a.symbols else C.MAJORS
    days = tardis_days()
    U.log("binance days: %d symbols x %d days (%s)" % (len(symbols), len(days), a.interval))

    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})
    outdir = os.path.join(BULK, "klines_" + a.interval)
    try:
        for s in symbols:
            dest = os.path.join(outdir, "%s.parquet" % s)
            paths = []
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(fetch_day, session, s, a.interval, d) for d in days]
                for fu in as_completed(futs):
                    p = fu.result()
                    if p:
                        paths.append(p)
            new = build(s, a.interval, sorted(paths))
            if new.empty:
                U.log("%s: no data" % s)
                continue
            # 이미 전체 이력을 받아둔 심볼이면 합쳐서 보존한다(덮어쓰지 않는다).
            old = U.read_parquet_or_quarantine(dest)
            if old is not None and not old.empty:
                new = (pd.concat([old, new], ignore_index=True)
                         .drop_duplicates(subset=["open_time"], keep="last")
                         .sort_values("open_time").reset_index(drop=True))
            U.atomic_write_parquet(new, dest)
            U.log("%s: %d days fetched, %d rows total (%s~%s)"
                  % (s, len(paths), len(new),
                     pd.to_datetime(new.open_time.min(), unit="ms").date(),
                     pd.to_datetime(new.open_time.max(), unit="ms").date()))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
