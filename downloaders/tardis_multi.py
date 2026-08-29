# -*- coding: utf-8 -*-
"""Tardis 무료 샘플(매월 1일) 다중 거래소 · 다중 심볼 청산 다운로더.

왜 확장하는가
  기존 분석은 Bybit 5심볼로만 했다. Bybit은 BTC perp OI의 24.5%이므로, 가격을 실제로
  민 물량의 일부만 설명변수로 넣은 셈이고 계수가 희석된다(impact.py의 한계 #1).
  실측 확인: Tardis 무료 구간이 binance-futures / bybit / okex-swap 청산 모두에
  적용된다. 셋을 합치면 BTC perp OI의 약 85%를 덮는다(Binance 47 + Bybit 24.5 + OKX 13.5).

거래소별 피드 특성 (중요)
  bybit           : 2025-02-25부터 allLiquidation 전건. 그 이전은 초당 1건 스로틀.
  binance-futures : forceOrder. 초당 심볼별 1건 스냅샷 -> 상시 과소집계.
                    실측(2026-07-31): 전 종목 90초 구독에 0건이었다.
  okex-swap       : 별도 확인 필요.
  따라서 거래소를 합칠 때 '건수'는 비교 불가이고 '가격 위치'는 비교 가능하다.
  같은 날·같은 심볼로 bybit vs binance를 대조하면 스로틀 편향을 직접 측정할 수 있다.

심볼 표기
  bybit / binance-futures : BTCUSDT
  okex-swap               : BTC-USDT-SWAP

실행:
    python downloaders/tardis_multi.py --top 40
    python downloaders/tardis_multi.py --exchanges bybit binance-futures --top 20
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from downloaders.tardis_liq import months, FIRST_MONTH, FULL_FEED_FROM_MS   # noqa: E402

BASE = "https://datasets.tardis.dev/v1"
RAW = os.path.join(C.DATA, "tardis_multi", "_raw")
OUT = os.path.join(C.DATA, "tardis_multi")
EXCHANGES = ["bybit", "binance-futures", "okex-swap"]


def to_symbol(exchange: str, base_sym: str) -> str:
    """BTCUSDT -> 거래소별 표기."""
    if exchange == "okex-swap":
        if not base_sym.endswith("USDT"):
            return base_sym
        return "%s-USDT-SWAP" % base_sym[:-4]
    return base_sym


def pick_symbols(top: int) -> list[str]:
    """Bybit 24h 거래대금 상위 USDT 무기한. 실패 시 config 폴백."""
    try:
        r = urllib.request.Request(C.BYBIT_TICKERS_URL, headers={"User-Agent": C.USER_AGENT})
        d = json.load(urllib.request.urlopen(r, timeout=30))
        rows = [x for x in d["result"]["list"] if str(x["symbol"]).endswith("USDT")]
        rows.sort(key=lambda x: -U.to_float(x.get("turnover24h")))
        return [x["symbol"] for x in rows[:top]]
    except Exception as e:
        U.log("symbol pick failed (%s) -> config fallback" % e)
        return list(C.BINANCE_SYMBOLS)


def fetch(session: requests.Session, exchange: str, base_sym: str,
          y: int, m: int) -> str | None:
    sym = to_symbol(exchange, base_sym)
    dest = os.path.join(RAW, exchange, base_sym, "%04d-%02d-01.csv.gz" % (y, m))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = "%s/%s/liquidations/%04d/%02d/01/%s.csv.gz" % (BASE, exchange, y, m, sym)
    try:
        r = session.get(url, timeout=120)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None                       # 400/401/404 = 무료 구간 밖이거나 미상장 (정상)
    try:
        gzip.GzipFile(fileobj=io.BytesIO(r.content)).read(64)
    except OSError:
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = "%s.%d.tmp" % (dest, os.getpid())
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


_CTVAL_PATH = os.path.join(OUT, "okx_ctval.json")


def okx_ctval() -> dict:
    """OKX 심볼별 계약 크기(ctVal). 없으면 공개 API에서 받아 캐시한다.

    OKX의 amount는 코인이 아니라 **계약 수**다. 실측(2026-07-31): BTC 중앙 amount가
    Bybit 0.027 / Binance 0.020 인데 OKX는 5.0 이었고, 그대로 명목가를 계산하면
    OKX 하나가 $127.6B로 나머지 합($4.2B)의 30배가 된다. ctVal(BTC 0.01, ETH 0.1,
    XRP 100, DOGE 1000 ...)을 곱해야 코인 수량이 된다.
    """
    if os.path.exists(_CTVAL_PATH):
        try:
            with open(_CTVAL_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    try:
        req = urllib.request.Request(
            "https://www.okx.com/api/v5/public/instruments?instType=SWAP",
            headers={"User-Agent": C.USER_AGENT})
        rows = json.load(urllib.request.urlopen(req, timeout=30))["data"]
        m = {x["instId"]: float(x["ctVal"]) for x in rows
             if str(x["instId"]).endswith("-USDT-SWAP") and x.get("ctVal")}
        os.makedirs(OUT, exist_ok=True)
        with open(_CTVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(m, f)
        U.log("okx ctVal: fetched %d instruments" % len(m))
        return m
    except Exception as e:
        U.log("okx ctVal fetch failed (%s) -> OKX rows will be dropped" % e)
        return {}


def parse(exchange: str, base_sym: str, paths: list[str],
          ctval: dict | None = None) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, compression="gzip")
        except Exception:
            continue
        if df.empty or "timestamp" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    for c in ("timestamp", "price", "amount"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "price", "amount"])

    if exchange == "okex-swap":
        cv = (ctval or {}).get(to_symbol(exchange, base_sym))
        if not cv or cv <= 0:
            return pd.DataFrame()          # 계약 크기를 모르면 쓰지 않는다
        d["amount"] = d["amount"] * cv     # 계약 수 -> 코인 수량

    d["ts_ms"] = (d["timestamp"] // 1000).astype("int64")
    # Tardis의 side는 청산 '주문'의 방향. sell = 롱 포지션이 강제 매도된 것.
    d["pos_side"] = d["side"].astype(str).str.lower().map(
        {"sell": "long", "buy": "short"}).fillna("")
    d["notional"] = d["price"] * d["amount"]
    d["exchange"] = exchange
    d["symbol"] = base_sym
    d["full_feed"] = (d["ts_ms"] >= FULL_FEED_FROM_MS) if exchange == "bybit" else False
    return (d[["ts_ms", "exchange", "symbol", "pos_side", "price", "amount",
               "notional", "full_feed"]]
            .sort_values("ts_ms").reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="multi-exchange Tardis free liquidation samples")
    ap.add_argument("--exchanges", nargs="*", default=EXCHANGES)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    os.makedirs(RAW, exist_ok=True)
    syms = a.symbols if a.symbols else pick_symbols(a.top)
    sy, sm = (int(x) for x in a.start.split("-"))
    if a.end:
        ey, em = (int(x) for x in a.end.split("-"))
    else:
        t = date.today()
        ey, em = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    ms = months((max(sy, FIRST_MONTH[0]), sm), (ey, em))

    U.log("tardis multi: %d exchanges x %d symbols x %d months"
          % (len(a.exchanges), len(syms), len(ms)))
    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})

    jobs = [(ex, s, y, m) for ex in a.exchanges for s in syms for (y, m) in ms]
    got: dict[tuple[str, str], list[str]] = {}
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(fetch, session, ex, s, y, m): (ex, s)
                    for ex, s, y, m in jobs}
            done = 0
            for fu in as_completed(futs):
                p = fu.result()
                done += 1
                if p:
                    got.setdefault(futs[fu], []).append(p)
                if done % 2000 == 0:
                    U.log("  %d/%d requests done, %d files"
                          % (done, len(jobs), sum(len(v) for v in got.values())))
    finally:
        session.close()

    # 이번에 받은 것만 파싱하면 이전 다운로드분이 덮어써진다. 항상 캐시 전체를 훑는다.
    ctval = okx_ctval()
    frames = []
    for ex in sorted(os.listdir(RAW)):
        exdir = os.path.join(RAW, ex)
        if not os.path.isdir(exdir):
            continue
        for sym in sorted(os.listdir(exdir)):
            paths = sorted(glob.glob(os.path.join(exdir, sym, "*.csv.gz")))
            if not paths:
                continue
            d = parse(ex, sym, paths, ctval)
            if not d.empty:
                frames.append(d)
    if not frames:
        U.log("nothing parsed")
        return 1
    allq = pd.concat(frames, ignore_index=True)
    U.atomic_write_parquet(allq, os.path.join(OUT, "liquidations.parquet"))

    U.log("total %d liquidations, $%.1fM" % (len(allq), allq.notional.sum() / 1e6))
    g = allq.groupby("exchange").agg(n=("notional", "size"),
                                     musd=("notional", lambda s: s.sum() / 1e6),
                                     symbols=("symbol", "nunique"))
    print()
    print(g.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
