# -*- coding: utf-8 -*-
"""Binance 펀딩비 다운로더 — pi(u) 의 대리변수.

왜 필요한가
  pi(u) = U/(U+I) 에서 U 는 '정보 없는 강제 흐름'이다. U 가 크다는 것은 포지션이
  한쪽으로 몰려 있다는 뜻이고, 펀딩비가 그것을 직접 가격으로 표시한다.
      펀딩비 > 0  ->  롱이 숏에게 지불  ->  롱이 붐빔  ->  하락 시 강제 롱청산이 많다
  지금까지 pi 의 대리변수로 쓴 것은 dOI <= -2% 이진값 하나뿐이었다.

데이터 (data.binance.vision, 무료)
  월별 파일: calc_time, funding_interval_hours, last_funding_rate
  8시간 간격(하루 3회). 일부 종목/시기는 4시간 간격이라 interval 컬럼을 같이 저장한다.

실행:
    python downloaders/binance_funding.py
    python downloaders/binance_funding.py --symbols BTCUSDT --since 2021-01
"""
from __future__ import annotations

import argparse
import datetime as dt
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

BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
RAW = os.path.join(C.DATA, "binance_bulk", "_raw")
OUT = os.path.join(C.DATA, "binance_bulk", "funding")
FIRST = "2020-09"


def months(since: str) -> list[str]:
    y, m = (int(x) for x in since.split("-"))
    end = dt.date.today().replace(day=1)
    out = []
    cur = dt.date(y, m, 1)
    while cur <= end:
        out.append("%04d-%02d" % (cur.year, cur.month))
        cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return out


def fetch(session: requests.Session, symbol: str, mon: str) -> str | None:
    dest = os.path.join(RAW, symbol, "fundingRate", "%s.zip" % mon)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = "%s/%s/%s-fundingRate-%s.zip" % (BASE, symbol, symbol, mon)
    try:
        r = session.get(url, timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        if zipfile.ZipFile(io.BytesIO(r.content)).testzip() is not None:
            return None
    except zipfile.BadZipFile:
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = "%s.%d.tmp" % (dest, os.getpid())
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def build(symbol: str, paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            z = zipfile.ZipFile(p)
            d = pd.read_csv(z.open(z.namelist()[0]))
        except Exception:                     # noqa: BLE001
            continue
        if d.empty or "last_funding_rate" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    for c in ("calc_time", "funding_interval_hours", "last_funding_rate"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["calc_time", "last_funding_rate"])
    d = d.rename(columns={"calc_time": "ts_ms", "last_funding_rate": "funding",
                          "funding_interval_hours": "interval_h"})
    d["ts_ms"] = d["ts_ms"].astype("int64")
    d["symbol"] = symbol
    return (d[["ts_ms", "funding", "interval_h", "symbol"]]
            .drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance funding rate downloader")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--since", default=FIRST)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    syms = a.symbols if a.symbols else C.MAJORS
    mons = months(a.since)
    U.log("funding: %d symbols x %d months (%s ~ %s)"
          % (len(syms), len(mons), mons[0], mons[-1]))

    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})
    try:
        for s in syms:
            got = []
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(fetch, session, s, m) for m in mons]
                for fu in as_completed(futs):
                    p = fu.result()
                    if p:
                        got.append(p)
            d = build(s, sorted(got))
            if d.empty:
                U.log("%s: no data" % s)
                continue
            U.atomic_write_parquet(d, os.path.join(OUT, "%s.parquet" % s))
            U.log("%s: %d행 (%s ~ %s), 간격 %s시간"
                  % (s, len(d),
                     pd.to_datetime(d.ts_ms.min(), unit="ms", utc=True).date(),
                     pd.to_datetime(d.ts_ms.max(), unit="ms", utc=True).date(),
                     sorted(d.interval_h.dropna().unique().tolist())))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
