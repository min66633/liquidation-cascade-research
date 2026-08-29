# -*- coding: utf-8 -*-
"""충격함수 재추정 — 분모를 호가 깊이로 바꾼다.

문제
  기존 추정은 변위를 ADV(일평균 거래대금)로 정규화해 b = 0.19 를 얻었다.
  표준 제곱근 법칙 0.5의 절반도 안 되는 값인데, ADV 는 '순간 호가 깊이'의 대용치로
  형편없다. 강제청산은 시장가로 나가 **아래쪽에 대기 중인 지정매수를 먹어치우므로**,
  얼마나 밀리는지는 청산액 자체가 아니라 청산액 대비 그 구간 호가 명목가가 결정한다.

    변위 = g( V_liq / Depth(u) )      <- 물리적으로 맞는 형태
    변위 = g( V_liq / ADV )           <- 기존(잘못된 분모)

데이터
  청산: Tardis Bybit 전건(2025-02-25~), 메이저 21종
  깊이: Binance bookDepth (2023-01-01~), 현재가 대비 %구간별 누적 호가 명목가, 2.5초 간격
  가격: Binance 1분봉

주의
  깊이는 Binance, 청산은 Bybit이다. 거래소가 다르므로 절대 배율에는 편의가 있다.
  다만 (a) 두 시장의 가격은 거의 동일하고 (b) 우리가 보는 것은 지수 b(로그-로그 기울기)라
  상수 배율은 절편에만 들어간다. 기울기 추정에는 영향이 없다.

실행:
    python analysis/impact_depth.py
    python analysis/impact_depth.py --depth-col dm1_0 --min-usd 50000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from analysis.impact import load_1m_with_volume, load_liq, fit_power   # noqa: E402
from analysis.impact2d import ols                                       # noqa: E402

DEPTH_DIR = os.path.join(C.DATA, "binance_bulk", "book_depth")
MIN_MS = 60_000
# bookDepth 컬럼: dm0_2 = -0.2%, dm1_0 = -1%, dm2_0 = -2% ... (누적 명목가)
DEPTH_COLS = ["dm0_2", "dm0_5", "dm1_0", "dm2_0", "dm3_0", "dm5_0"]


def load_depth(symbol: str) -> pd.DataFrame:
    """2026-08-01: 일자 분할 저장 + 2025년 고정 필드 필터를 공용 로더로 통일.

    이전에는 심볼당 단일 parquet 을 그대로 읽었다. 고정 필드를 안 거르면 분모가
    상수라 V/D 가 날조된다(analysis/bookdepth.py 참조).
    """
    cols = [c for c in DEPTH_COLS]
    d, st = BD.load_clean(symbol, cols)
    if d.empty:
        raise FileNotFoundError("no usable bookDepth for %s (missing=%s) — run "
                                "downloaders/binance_depth.py" % (symbol, st["missing"]))
    return d.sort_values("ts_ms").reset_index(drop=True)


def build(symbol: str, bucket_sec: int, horizon_min: int, min_usd: float) -> pd.DataFrame:
    liq = load_liq(symbol, True)
    if liq.empty:
        return pd.DataFrame()
    m1 = load_1m_with_volume(symbol)
    dep = load_depth(symbol)
    if m1.empty or dep.empty:
        return pd.DataFrame()

    step = bucket_sec * 1000
    liq = liq.copy()
    liq["bucket"] = (liq["ts_ms"] // step) * step

    ot = m1["open_time"].to_numpy()
    close = m1["close"].to_numpy()
    high = m1["high"].to_numpy()
    low = m1["low"].to_numpy()
    vol = m1["volume"].to_numpy()
    n1 = len(ot)
    s = pd.Series(close)
    sigma = (s / s.shift(1) - 1.0).shift(1).rolling(1440, min_periods=360).std().to_numpy()
    adv = pd.Series(vol * close).shift(1).rolling(1440, min_periods=360).mean().to_numpy()

    dts = dep["ts_ms"].to_numpy()
    dcols = [c for c in DEPTH_COLS if c in dep.columns]
    dvals = {c: dep[c].to_numpy() for c in dcols}

    out = []
    for (bkt, side), g in liq.groupby(["bucket", "pos_side"]):
        if side not in ("long", "short"):
            continue
        v = float(g["notional"].sum())
        if v < min_usd:
            continue
        end_ms = int(bkt) + step
        j = int(np.searchsorted(ot, end_ms, side="left"))
        if j <= 0 or j + horizon_min >= n1:
            continue
        ref = close[j - 1]
        if not (np.isfinite(ref) and ref > 0 and np.isfinite(sigma[j - 1]) and sigma[j - 1] > 0):
            continue
        # 버스트 '직전'의 깊이 스냅샷 (룩어헤드 차단)
        di = int(np.searchsorted(dts, int(bkt), side="right")) - 1
        if di < 0:
            continue
        rec = {"symbol": symbol, "bucket": int(bkt), "side": side,
               "v_liq": v, "sigma": float(sigma[j - 1]),
               "adv": float(adv[j - 1]) if np.isfinite(adv[j - 1]) else np.nan}
        ok = True
        for c in dcols:
            val = dvals[c][di]
            if not np.isfinite(val) or val <= 0:
                ok = False
                break
            rec[c] = float(val)
        if not ok:
            continue
        push = ((ref / low[j:j + horizon_min].min() - 1.0) if side == "long"
                else (high[j:j + horizon_min].max() / ref - 1.0))
        rec["push"] = push
        rec["push_sig"] = push / sigma[j - 1]
        out.append(rec)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="impact function with order-book depth as denominator")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--bucket-sec", type=int, default=30)
    ap.add_argument("--horizon-min", type=int, default=5)
    ap.add_argument("--min-usd", type=float, default=20000.0)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 210)
    symbols = a.symbols if a.symbols else C.MAJORS

    frames = []
    for s in symbols:
        try:
            d = build(s, a.bucket_sec, a.horizon_min, a.min_usd)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
    if not frames:
        U.log("no bursts")
        return 1
    d = pd.concat(frames, ignore_index=True)
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "impact_depth.parquet"))

    print("\n=== 표본 ===")
    print("버스트 n=%d | 청산 $%.0fM | 심볼 %d | 기간 %s ~ %s"
          % (len(d), d.v_liq.sum() / 1e6, d.symbol.nunique(),
             pd.to_datetime(d.bucket.min(), unit="ms").date(),
             pd.to_datetime(d.bucket.max(), unit="ms").date()))
    dcols = [c for c in DEPTH_COLS if c in d.columns]
    print("\n호가 깊이 중앙값($M):",
          {c: round(d[c].median() / 1e6, 1) for c in dcols})
    print("청산액 중앙값: $%.0f" % d.v_liq.median())

    print("\n=== 분모 비교: 무엇으로 나눠야 관계가 깨끗한가 ===")
    print("%-14s %8s %10s %10s" % ("분모", "지수 b", "R2", "n"))
    y = d["push_sig"].to_numpy()
    rows = []
    if "adv" in d.columns and d["adv"].notna().any():
        dd = d[np.isfinite(d["adv"]) & (d["adv"] > 0)]
        _, b, r2, n = fit_power((dd["v_liq"] / dd["adv"]).to_numpy(), dd["push_sig"].to_numpy())
        rows.append(("ADV (기존)", b, r2, n))
    for c in dcols:
        _, b, r2, n = fit_power((d["v_liq"] / d[c]).to_numpy(), y)
        lab = "Depth %s" % c.replace("dm", "-").replace("_", ".") + "%"
        rows.append((lab, b, r2, n))
    for lab, b, r2, n in rows:
        print("%-14s %8.3f %10.3f %10d" % (lab, b, r2, n))
    print("  b=0.5 이면 표준 제곱근 법칙. R2 가 높은 분모가 물리적으로 맞는 분모다.")

    print("\n=== 청산액 / 호가깊이 비율 구간별 실제 변위 ===")
    best = "dm1_0" if "dm1_0" in dcols else dcols[0]
    d["ratio"] = d["v_liq"] / d[best]
    d["q"] = pd.qcut(d["ratio"].rank(method="first"), 8, labels=False)
    t = d.groupby("q").agg(n=("ratio", "size"),
                           ratio_med=("ratio", "median"),
                           v_liq_med=("v_liq", "median"),
                           push_med=("push_sig", "median"),
                           push_p90=("push_sig", lambda s: s.quantile(0.9)))
    print("  (분모 = %s 구간 누적 호가 명목가)" % best)
    print(t.round(3).to_string())

    print("\n=== 깊이가 ADV보다 나은가 — 같은 회귀에 둘 다 ===")
    dd = d[np.isfinite(d["adv"]) & (d["adv"] > 0) & (d[best] > 0)].copy()
    if len(dd) > 200:
        X = np.column_stack([np.ones(len(dd)),
                             np.log(dd["v_liq"]),
                             np.log(dd[best]),
                             np.log(dd["adv"])])
        yy = np.log(dd["push_sig"].clip(lower=1e-6)).to_numpy()
        c, se, n = ols(X, yy)
        for i, nm in enumerate(["절편", "log V_liq", "log Depth", "log ADV"]):
            if i:
                print("  %-12s %8.4f (t=%6.2f)" % (nm, c[i], c[i] / se[i]))
        print("  n=%d" % n)
        print("  예측: log Depth 계수는 음수여야 한다(깊으면 덜 밀림).")
        print("        V_liq 와 Depth 계수의 크기가 비슷하면 비율(V/Depth)이 옳은 형태다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
