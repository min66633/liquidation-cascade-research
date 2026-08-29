# -*- coding: utf-8 -*-
"""data.binance.vision 벌크 다운로더 — H2 검정용 과거 데이터 확보.

받는 것
  klines 5m  : 월별 zip (2020-01~), 현재 달은 일별 zip으로 보충
  metrics    : 일별 zip만 존재 (2020-09-01~). 5분 해상도 OI + 롱숏비율 4종.
               /futures/data/* 실시간 API가 최근 30일만 주는 그 데이터의 전체 이력이다.

주의 — USD-M(um)의 liquidationSnapshot은 이 버킷에서 제거되었다(코인마진 cm만 잔존).
따라서 '실현 청산' 원본은 무료로 백필되지 않는다. 청산 이벤트는 OI 급감 + 가격 급변의
4분면으로 식별한다(README의 청산 식별 4분면 참조).

실행:
    python downloaders/binance_bulk.py                     # 전체 심볼 전체 기간
    python downloaders/binance_bulk.py --symbols BTCUSDT   # 특정 심볼만
    python downloaders/binance_bulk.py --parse-only        # 다운로드 생략, 재파싱만
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

BASE = "https://data.binance.vision/data/futures/um"
RAW = os.path.join(C.DATA, "binance_bulk", "_raw")
OUT = os.path.join(C.DATA, "binance_bulk")

METRICS_START = date(2020, 9, 1)      # 이 날짜 이전 metrics 없음
KLINE_INTERVAL = "5m"                 # main()에서 --interval로 덮어쓴다

# klines zip은 시기에 따라 헤더가 있기도 없기도 하다. 컬럼은 고정.
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _month_range(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append("%04d-%02d" % (y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _day_range(start: date, end: date) -> list[str]:
    out, d = [], start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def download(session: requests.Session, url: str, dest: str) -> str | None:
    """이미 있으면 건너뛴다. 404는 정상(해당 기간 데이터 없음)이라 조용히 None."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        r = session.get(url, timeout=120)
    except requests.RequestException as e:
        U.log("download error %s: %s" % (os.path.basename(url), e))
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        U.log("download HTTP %d %s" % (r.status_code, os.path.basename(url)))
        return None
    # zip 유효성을 먼저 확인하고 저장한다 — 깨진 파일이 캐시에 남으면 매번 재파싱에 실패한다.
    try:
        zipfile.ZipFile(io.BytesIO(r.content)).testzip()
    except zipfile.BadZipFile:
        U.log("bad zip %s" % os.path.basename(url))
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = "%s.%d.tmp" % (dest, os.getpid())
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def read_zip_csv(path: str, cols: list[str] | None) -> pd.DataFrame | None:
    """zip 안의 단일 CSV를 읽는다. 헤더 유무를 자동 판별한다."""
    try:
        z = zipfile.ZipFile(path)
        name = z.namelist()[0]
        raw = z.open(name).read()
    except (zipfile.BadZipFile, IndexError, KeyError, OSError) as e:
        U.log("unreadable zip %s (%s) -> removing" % (os.path.basename(path), e))
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    first = raw.split(b"\n", 1)[0].decode("utf-8", "replace")
    if cols is None:                       # metrics: 항상 헤더 있음
        return pd.read_csv(io.BytesIO(raw))
    # klines: 첫 필드가 숫자면 헤더 없음
    has_header = not first.split(",")[0].strip().replace(".", "", 1).isdigit()
    if has_header:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip() for c in df.columns]
        return df
    return pd.read_csv(io.BytesIO(raw), header=None, names=cols)


def build_klines(symbol: str, paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_zip_csv(p, KLINE_COLS)
        if df is None or df.empty:
            continue
        keep = ["open_time", "open", "high", "low", "close", "volume",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
        df = df[[c for c in keep if c in df.columns]].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open_time", "close"])
    # 2025년 일부 zip이 마이크로초 단위 open_time을 쓴다 — 밀리초로 통일한다.
    big = out["open_time"] > 1e14
    out.loc[big, "open_time"] = out.loc[big, "open_time"] // 1000
    out["open_time"] = out["open_time"].astype("int64")
    out = (out.drop_duplicates(subset=["open_time"], keep="last")
              .sort_values("open_time").reset_index(drop=True))
    out["symbol"] = symbol
    return out


def build_metrics(symbol: str, paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = read_zip_csv(p, None)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["create_time"] = pd.to_datetime(out["create_time"], errors="coerce", utc=True)
    out = out.dropna(subset=["create_time"])
    out["open_time"] = (out["create_time"].astype("int64") // 10 ** 6)   # ms
    for c in out.columns:
        if c not in ("create_time", "symbol", "open_time"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = (out.drop_duplicates(subset=["open_time"], keep="last")
              .sort_values("open_time").reset_index(drop=True))
    out = out.drop(columns=["create_time"])
    out["symbol"] = symbol
    return out


def fetch_symbol(session: requests.Session, symbol: str, end: date,
                 workers: int, parse_only: bool, with_metrics: bool = True) -> dict:
    """한 심볼의 klines(+선택적으로 metrics)를 받아 parquet으로 저장.

    with_metrics=False면 klines만 받는다. 1분봉을 추가로 받을 때 metrics를 다시 받을
    이유가 없으므로(5분 해상도 고정) 중복 다운로드를 피한다.
    """
    months = _month_range(METRICS_START, end)
    days = _day_range(METRICS_START, end)
    cur_month = "%04d-%02d" % (end.year, end.month)

    jobs = []   # (url, dest, kind)
    for mo in months:
        if mo == cur_month:
            continue                       # 현재 달은 월별 zip이 아직 없다 → 일별로
        jobs.append(("%s/monthly/klines/%s/%s/%s-%s-%s.zip" % (BASE, symbol, KLINE_INTERVAL, symbol, KLINE_INTERVAL, mo),
                     os.path.join(RAW, symbol, "klines_" + KLINE_INTERVAL, "%s.zip" % mo), "k"))
    for d in days:
        if d[:7] == cur_month:
            jobs.append(("%s/daily/klines/%s/%s/%s-%s-%s.zip" % (BASE, symbol, KLINE_INTERVAL, symbol, KLINE_INTERVAL, d),
                         os.path.join(RAW, symbol, "klines_" + KLINE_INTERVAL, "%s.zip" % d), "k"))
        if with_metrics:
            jobs.append(("%s/daily/metrics/%s/%s-metrics-%s.zip" % (BASE, symbol, symbol, d),
                         os.path.join(RAW, symbol, "metrics", "%s.zip" % d), "m"))

    if not parse_only:
        done = failed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(download, session, u, p): (u, p) for u, p, _ in jobs}
            for fu in as_completed(futs):
                if fu.result():
                    done += 1
                else:
                    failed += 1
        U.log("%s: fetched %d/%d files (%d missing/failed)" % (symbol, done, len(jobs), failed))

    kpaths = sorted(p for _, p, k in jobs if k == "k" and os.path.exists(p))
    mpaths = sorted(p for _, p, k in jobs if k == "m" and os.path.exists(p))

    res = {"symbol": symbol}
    kl = build_klines(symbol, kpaths)
    if not kl.empty:
        U.atomic_write_parquet(kl, os.path.join(OUT, "klines_%s" % KLINE_INTERVAL, "%s.parquet" % symbol))
    res["klines_rows"] = len(kl)

    if with_metrics:
        me = build_metrics(symbol, mpaths)
        if not me.empty:
            U.atomic_write_parquet(me, os.path.join(OUT, "metrics", "%s.parquet" % symbol))
        res["metrics_rows"] = len(me)

    if not kl.empty:
        U.log("%s: klines_%s %d rows (%s~%s)%s"
              % (symbol, KLINE_INTERVAL, len(kl),
                 pd.to_datetime(kl.open_time.min(), unit="ms").date(),
                 pd.to_datetime(kl.open_time.max(), unit="ms").date(),
                 (", metrics %d rows" % res["metrics_rows"]) if with_metrics else ""))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance vision bulk downloader")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--interval", default="5m", help="kline interval, e.g. 1m / 5m")
    ap.add_argument("--no-metrics", action="store_true", help="klines only")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: yesterday UTC)")
    a = ap.parse_args()

    global KLINE_INTERVAL
    KLINE_INTERVAL = a.interval
    U.init_stdout()
    C.ensure_dirs()
    os.makedirs(RAW, exist_ok=True)
    symbols = a.symbols if a.symbols else C.BINANCE_SYMBOLS
    end = date.fromisoformat(a.end) if a.end else (
        pd.Timestamp.utcnow().date() - timedelta(days=1))

    U.log("bulk download: %d symbols, %s klines, %s ~ %s, workers=%d"
          % (len(symbols), KLINE_INTERVAL, METRICS_START, end, a.workers))
    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})
    try:
        for s in symbols:
            fetch_symbol(session, s, end, a.workers, a.parse_only, not a.no_metrics)
    finally:
        session.close()
    U.log("bulk download done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
