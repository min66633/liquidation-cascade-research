# -*- coding: utf-8 -*-
"""H4 검정 — 캐스케이드는 연료의 '연속성'이 끌고 가고 '공백'에서 멈춘다.

가설 수정 경위
  당초 가설: 두꺼운 청산 군집이 터지면 크게 슈팅한다.
  impact2d.py 실측: 같은 규모라도 **한 점에 몰려 터진 것이 덜 밀렸다**
  (conc 계수 -0.67 t=-2.29, span% +0.09 t=2.23, 4개 규모구간 전부 단조).
  버스트 구간 가격움직임을 통제해도 계수가 그대로였다(교락 아님).

  -> 수정: 두께가 아니라 **연속성**이다. 한 점에 몰려 있다는 것은 그 아래에 연료가
     없다는 뜻이고, 흩어져 있다는 것은 계단식으로 다음 연료가 이어진다는 뜻이다.
     연쇄가 되려면 연료가 이어져 있어야 한다.

측정량 (가격축 위 연료 분포의 형태)
  coverage   = 연료가 있는 가격빈 / 전체 가격빈      (연속성. 1이면 빈틈없음)
  max_gap    = 분포 안 최대 연속 공백 폭(%)          (가장 큰 구멍)
  n_clusters = 분리된 연료 덩어리 개수                (파편화 정도)
  gini       = 명목가의 불균등도                      (한 곳 쏠림)

예측
  coverage 높을수록 / max_gap 작을수록 -> 더 멀리 밀린다.
  고립된 한 덩이(coverage 낮고 max_gap 큼) -> 거기서 멈춘다.

순환논법이 아닌 이유
  형태는 버스트 구간 [t0, t0+w] 안의 실현 청산으로만 재고, 변위는 그 이후 [t0+w, +h]
  에서 잰다. 예측변수가 완전히 관측된 뒤의 변위만 종속변수다.

두 번째 모드 (--hl)
  같은 측정량을 Hyperliquid 실측 지도에 적용해 '현재가에서 첫 공백까지의 거리'를 뽑는다.
  이게 실전에서 지정가를 어디 깔지 정하는 운용량이다.

실행:
    python analysis/continuity.py
    python analysis/continuity.py --hl
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from analysis.impact import load_1m_with_volume, load_liq   # noqa: E402
from analysis.impact2d import ols                            # noqa: E402


def shape_features(prices: np.ndarray, weights: np.ndarray, ref: float,
                   bin_bps: float, min_frac: float = 0.001) -> dict:
    """가격축 분포의 형태 특징. min_frac 미만인 빈은 '비어있음'으로 본다."""
    binw = ref * bin_bps / 1e4
    if binw <= 0 or len(prices) == 0:
        return {}
    lo, hi = prices.min(), prices.max()
    n_bins = int(np.floor((hi - lo) / binw)) + 1
    if n_bins < 3 or n_bins > 20000:
        return {}
    idx = np.floor((prices - lo) / binw).astype("int64")
    h = np.bincount(idx, weights=weights, minlength=n_bins).astype("float64")
    tot = h.sum()
    if tot <= 0:
        return {}
    filled = h >= tot * min_frac

    # 최대 연속 공백
    max_gap_bins = 0
    run = 0
    for f in filled:
        run = 0 if f else run + 1
        max_gap_bins = max(max_gap_bins, run)
    # 덩어리 개수
    n_clusters = int(np.sum(filled & ~np.r_[False, filled[:-1]]))
    # 지니
    x = np.sort(h[h > 0])
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if n > 1 else 0.0

    return {
        "coverage": float(filled.mean()),
        "max_gap_pct": float(max_gap_bins * binw / ref * 100.0),
        "n_clusters": n_clusters,
        "gini": gini,
        "span_pct": float((hi - lo) / ref * 100.0),
        "n_bins": n_bins,
    }


def build(symbol: str, bucket_sec: int, bin_bps: float, min_usd: float,
          horizons: list[int], full_only: bool) -> pd.DataFrame:
    liq = load_liq(symbol, full_only)
    m1 = load_1m_with_volume(symbol)
    if liq.empty or m1.empty:
        return pd.DataFrame()

    step = bucket_sec * 1000
    liq = liq.copy()
    liq["bucket"] = (liq["ts_ms"] // step) * step

    ot = m1["open_time"].to_numpy()
    close = m1["close"].to_numpy()
    high = m1["high"].to_numpy()
    low = m1["low"].to_numpy()
    n1 = len(ot)
    s = pd.Series(close)
    sigma = (s / s.shift(1) - 1.0).shift(1).rolling(1440, min_periods=360).std().to_numpy()

    out = []
    for (bkt, side), g in liq.groupby(["bucket", "pos_side"]):
        if side not in ("long", "short"):
            continue
        v_total = float(g["notional"].sum())
        if v_total < min_usd:
            continue
        j = int(np.searchsorted(ot, int(bkt) + step, side="left"))
        if j <= 0 or j + max(horizons) >= n1:
            continue
        ref = close[j - 1]
        if not (np.isfinite(ref) and ref > 0 and np.isfinite(sigma[j - 1]) and sigma[j - 1] > 0):
            continue
        f = shape_features(g["price"].to_numpy(), g["notional"].to_numpy(), ref, bin_bps)
        if not f:
            continue
        rec = {"symbol": symbol, "bucket": int(bkt), "side": side,
               "v_total": v_total, "n_liq": int(len(g)),
               "sigma": float(sigma[j - 1])}
        rec.update(f)
        for h in horizons:
            push = ((ref / low[j:j + h].min() - 1.0) if side == "long"
                    else (high[j:j + h].max() / ref - 1.0))
            rec["pushsig%d" % h] = push / sigma[j - 1]
        out.append(rec)
    return pd.DataFrame(out)


def hl_gaps(bin_bps: float, band_pct: float, min_frac: float,
            gap_pct: float = 1.0) -> pd.DataFrame:
    """Hyperliquid 실측 지도에서 '현재가부터 첫 공백까지의 거리'를 코인별로 계산."""
    fs = sorted(glob.glob(os.path.join(C.HL_DIR_POSITIONS, "*", "positions_*.parquet")),
                key=os.path.getmtime)
    if not fs:
        return pd.DataFrame()
    f = fs[-1]
    sid = os.path.basename(f).split("_")[-1].split(".")[0]
    mf = [g for g in glob.glob(os.path.join(C.HL_DIR_MIDS, "*", "mids_*.parquet")) if sid in g]
    if not mf:
        return pd.DataFrame()
    p = pd.read_parquet(f)
    m = pd.read_parquet(mf[0])
    m = m[m["phase"] == "start"].set_index("coin")["mid_px"]
    p["mark"] = p["coin"].map(m)
    p = p[p["mark"].notna() & (p["mark"] > 0) & p["liquidation_px"].notna()].copy()
    p["notional"] = p["szi"].abs() * p["mark"]
    p["pos_side"] = np.where(p["szi"] > 0, "long", "short")

    rows = []
    for (coin, side), g in p.groupby(["coin", "pos_side"]):
        mark = float(g["mark"].iloc[0])
        binw = mark * bin_bps / 1e4
        # 롱청산은 현재가 아래, 숏청산은 위
        if side == "long":
            sel = g[(g.liquidation_px < mark) & (g.liquidation_px >= mark * (1 - band_pct / 100))]
        else:
            sel = g[(g.liquidation_px > mark) & (g.liquidation_px <= mark * (1 + band_pct / 100))]
        if sel.empty:
            continue
        n_bins = int(band_pct / 100 * mark / binw) + 1
        dist = ((mark - sel.liquidation_px) / mark if side == "long"
                else (sel.liquidation_px - mark) / mark)
        idx = np.floor(dist * mark / binw).astype("int64")
        idx = idx[(idx >= 0) & (idx < n_bins)]
        w = sel["notional"].to_numpy()[:len(idx)]
        h = np.bincount(idx, weights=w, minlength=n_bins).astype("float64")
        tot = h.sum()
        if tot <= 0:
            continue
        filled = h >= tot * min_frac
        # 현재가에서 출발해 첫 '연속 공백'까지의 거리
        gap_run = 0
        reach_bins = n_bins
        need = max(1, int(round(gap_pct / (bin_bps / 100))))   # gap_pct 이상 비면 공백
        for i, fl in enumerate(filled):
            gap_run = 0 if fl else gap_run + 1
            if gap_run >= need:
                reach_bins = i - need + 1
                break
        rows.append({"coin": coin, "side": side, "mark": mark,
                     "fuel_musd": tot / 1e6,
                     "reach_pct": reach_bins * binw / mark * 100.0,
                     "coverage": float(filled.mean()),
                     "n_pos": len(sel)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="H4: fuel continuity and gaps drive cascade distance")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--bucket-sec", type=int, default=30)
    ap.add_argument("--bin-bps", type=float, default=5.0)
    ap.add_argument("--min-usd", type=float, default=50000.0)
    ap.add_argument("--min-frac", type=float, default=0.001,
                    help="a bin below this share of total is treated as empty")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 15])
    ap.add_argument("--all-period", action="store_true")
    ap.add_argument("--hl", action="store_true", help="also report gaps on the live HL map")
    ap.add_argument("--hl-bin-bps", type=float, default=25.0)
    ap.add_argument("--hl-band", type=float, default=25.0)
    ap.add_argument("--gap-pct", type=float, default=1.0,
                    help="an empty stretch of at least this %% counts as a gap")
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 220)

    if a.hl:
        g = hl_gaps(a.hl_bin_bps, a.hl_band, a.min_frac, a.gap_pct)
        print("\n=== Hyperliquid 실측 지도: 현재가 -> 첫 공백까지 거리 ===")
        if g.empty:
            print("no HL data")
        else:
            g = g[g.n_pos >= 3].sort_values("fuel_musd", ascending=False)
            print(g.head(15).round(3).to_string(index=False))
            print("\nreach_pct = 이 거리까지는 연료가 이어져 있고 그 다음이 공백")
            print("=> 지정가는 두꺼운 곳이 아니라 이 거리 '너머'에 깔아야 한다는 것이 H4의 함의")
        return 0

    symbols = a.symbols if a.symbols else C.MAJORS
    frames = []
    for s in symbols:
        try:
            d = build(s, a.bucket_sec, a.bin_bps, a.min_usd, a.horizons, not a.all_period)
        except FileNotFoundError as e:
            U.log(str(e)); continue
        if not d.empty:
            frames.append(d); U.log("%s: %d bursts" % (s, len(d)))
    if not frames:
        U.log("no bursts"); return 1
    d = pd.concat(frames, ignore_index=True)
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "continuity_bursts.parquet"))

    h = a.horizons[1] if len(a.horizons) > 1 else a.horizons[0]
    y = np.log(d["pushsig%d" % h].clip(lower=1e-4)).to_numpy()
    lv = np.log(d["v_total"].clip(lower=1.0)).to_numpy()

    print("\n=== 표본 ===")
    print("n=%d | 청산 $%.0fM" % (len(d), d.v_total.sum() / 1e6))
    print(d[["coverage", "max_gap_pct", "n_clusters", "gini", "span_pct"]]
          .describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())

    print("\n=== H4 회귀 (h=%d분, 종속=log(변위/sigma)) ===" % h)
    X = np.column_stack([np.ones(len(d)), lv, d["coverage"].to_numpy(),
                         np.log1p(d["max_gap_pct"].to_numpy()),
                         np.log1p(d["n_clusters"].to_numpy())])
    c, se, n = ols(X, y)
    names = ["절편", "log V_total", "coverage(연속성)", "log(1+max_gap%)", "log(1+덩어리수)"]
    for i in range(1, len(c)):
        print("  %-18s %8.4f  (t=%6.2f)" % (names[i], c[i], c[i] / se[i]))
    print("  n=%d" % n)
    print("  예측: coverage(+), max_gap(-)  <- 연속적일수록 멀리 간다")

    print("\n=== 규모 x 연속성 이중 정렬 (변위 중앙값, sigma 배수) ===")
    d2 = d.copy()
    d2["vq"] = pd.qcut(d2["v_total"].rank(method="first"), 4, labels=["V1", "V2", "V3", "V4 큼"])
    d2["cq"] = pd.qcut(d2["coverage"].rank(method="first"), 3, labels=["끊김", "중간", "연속"])
    print(d2.pivot_table(index="vq", columns="cq", values="pushsig%d" % h,
                         aggfunc="median", observed=True).round(2).to_string())
    print("  예측: 같은 행에서 오른쪽(연속)이 커야 한다")
    print("\n  (셀별 표본 수)")
    print(d2.pivot_table(index="vq", columns="cq", values="v_total",
                         aggfunc="size", observed=True).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
