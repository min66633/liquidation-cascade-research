# -*- coding: utf-8 -*-
"""앞서 성급하게 '기각'한 두 항목의 재검정 + 격리/교차 구조 측정.

재검정 1 — "강제청산 $1 = 평범한 거래량 $1" 이 정말인가
  기존 회귀의 결함 두 가지:
    (i)  1분 칸 전수를 썼다. 칸당 청산 명목가 중앙값이 $59라 잡음이 표본을 지배했다.
         같은 데이터에 $50k 임계를 걸자 규모 효과가 0.057 -> 0.19 로 살아났는데,
         이 회귀는 임계 적용 후에 다시 돌린 적이 없다.
    (ii) 청산은 Bybit(시장의 24.5%), 거래량은 Binance(전량)를 썼다. 부분표본 변수를
         전량 변수와 함께 넣으면 부분표본 계수가 0으로 끌린다. 명세 오류다.
  -> 임계를 걸고, 청산을 Bybit 점유율로 스케일업한 버전도 함께 본다.

구조 측정 — 격리 vs 교차
  교차마진은 청산가가 진입가보다 한참 아래(실측 중앙 -38%)라 강제청산이 드물다.
  근접 연료(현재가 근처에서 청산되는 물량)는 격리·고레버리지에서 나와야 한다.
  강제 청산은 격리에서 나오고, 교차 보유자는 그걸 보고 '재량으로' 손절한다면
  L(p)는 강제 부분만 담고 재량 손절 층은 통째로 빠져 있다는 뜻이 된다.

실행:
    python analysis/recheck.py
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

MIN_MS = 60_000
BYBIT_SHARE = 0.245          # BTC perp OI 기준 실측 점유율


def build(symbol: str, bucket_min: int, horizon: int) -> pd.DataFrame:
    liq = load_liq(symbol, True)
    m1 = load_1m_with_volume(symbol)
    if liq.empty or m1.empty:
        return pd.DataFrame()
    step = bucket_min * MIN_MS
    liq = liq.copy()
    liq["bucket"] = (liq["ts_ms"] // step) * step
    g = (liq.groupby(["bucket", "pos_side"])["notional"].sum()
            .unstack(fill_value=0.0))
    for c in ("long", "short"):
        if c not in g.columns:
            g[c] = 0.0
    g = g.rename(columns={"long": "liq_long", "short": "liq_short"}).reset_index()

    k = m1.copy()
    k["bucket"] = (k["open_time"] // step) * step
    agg = k.groupby("bucket").agg(high=("high", "max"), low=("low", "min"),
                                  close=("close", "last"),
                                  volume=("volume", "sum")).reset_index()
    df = agg.merge(g, on="bucket", how="left").fillna({"liq_long": 0.0, "liq_short": 0.0})
    df = df.sort_values("bucket").reset_index(drop=True)
    df["contig"] = df["bucket"].diff().eq(step)
    win = max(int(1440 / bucket_min), 30)
    ret = df["close"] / df["close"].shift(1) - 1.0
    df["sigma"] = ret.shift(1).rolling(win, min_periods=win // 4).std()
    df["dollar_vol"] = df["volume"] * df["close"]
    fwd_low = df["low"].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    fwd_high = df["high"].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    out = []
    for side, lcol in (("long", "liq_long"), ("short", "liq_short")):
        t = df[["bucket", "sigma", "dollar_vol", "contig"]].copy()
        t["symbol"] = symbol
        t["side"] = side
        t["liq_usd"] = df[lcol]
        t["push"] = (df["close"] / fwd_low - 1.0) if side == "long" else (fwd_high / df["close"] - 1.0)
        out.append(t)
    p = pd.concat(out, ignore_index=True)
    return p[p["contig"] & np.isfinite(p["push"]) & (p["sigma"] > 0) & (p["dollar_vol"] > 0)]


def recheck_volume(symbols: list[str], thresholds: list[float]) -> None:
    frames = []
    for s in symbols:
        try:
            d = build(s, 1, 5)
        except FileNotFoundError:
            continue
        if not d.empty:
            frames.append(d)
    if not frames:
        U.log("no data")
        return
    p = pd.concat(frames, ignore_index=True)
    p = p[p["liq_usd"] > 0].copy()
    p["push_sig"] = p["push"] / p["sigma"]
    print("=== 재검정 1: 강제청산 $1 vs 일반 거래량 $1 ===")
    print("종속=log(변위/sigma), 설명=log(청산액), log(기타거래량). h=5분")
    print()
    print("%-12s %8s %14s %14s   %s" % ("임계", "n", "log(청산)", "log(기타량)", "판정"))
    for thr in thresholds:
        d = p[p["liq_usd"] >= thr].copy()
        if len(d) < 100:
            print("%-12s %8d  (표본 부족)" % ("$%.0f" % thr, len(d)))
            continue
        other = (d["dollar_vol"] - d["liq_usd"]).clip(lower=1.0)
        X = np.column_stack([np.ones(len(d)), np.log(d["liq_usd"]), np.log(other)])
        y = np.log(d["push_sig"].clip(lower=1e-6)).to_numpy()
        c, se, n = ols(X, y)
        t1, t2 = c[1] / se[1], c[2] / se[2]
        verdict = "청산 유의" if abs(t1) > 2 else ("경계" if abs(t1) > 1.6 else "무의미")
        print("%-12s %8d  %7.4f(t=%5.2f) %7.4f(t=%5.2f)   %s"
              % ("$%.0f" % thr, n, c[1], t1, c[2], t2, verdict))

    print()
    print("--- 점유율 보정: 청산액을 Bybit 점유율(%.1f%%)로 나눠 시장 전체로 환산 ---"
          % (100 * BYBIT_SHARE))
    d = p[p["liq_usd"] >= 50000].copy()
    d["liq_mkt"] = d["liq_usd"] / BYBIT_SHARE
    other = (d["dollar_vol"] - d["liq_mkt"]).clip(lower=1.0)
    X = np.column_stack([np.ones(len(d)), np.log(d["liq_mkt"]), np.log(other)])
    y = np.log(d["push_sig"].clip(lower=1e-6)).to_numpy()
    c, se, n = ols(X, y)
    print("  log(청산_시장환산) %.4f (t=%.2f)   log(기타거래량) %.4f (t=%.2f)   n=%d"
          % (c[1], c[1] / se[1], c[2], c[2] / se[2], n))
    print("  주: 상수배 스케일링은 로그에서 절편만 바꾸므로 계수 자체는 불변이다.")
    print("      명세 오류의 본질은 '부분표본 vs 전량' 대조이며, 이는 Bybit 거래량이")
    print("      있어야 완전히 교정된다(현재 미보유).")


def isolated_vs_cross() -> None:
    """근접 연료가 격리에서 나오는가 — 사용자 구조 가설의 직접 검정."""
    print()
    print("=== 구조 측정: 근접 연료는 격리(isolated)에서 나오는가 ===")
    fs = sorted(glob.glob(os.path.join(C.HL_DIR_POSITIONS, "*", "positions_*.parquet")),
                key=os.path.getmtime)
    mids = {os.path.basename(f).split("_")[-1].split(".")[0]: f
            for f in glob.glob(os.path.join(C.HL_DIR_MIDS, "*", "mids_*.parquet"))}
    rows = []
    for f in fs[-12:]:
        sid = os.path.basename(f).split("_")[-1].split(".")[0]
        if sid not in mids:
            continue
        try:
            p = pd.read_parquet(f)
            m = pd.read_parquet(mids[sid])
        except Exception:
            continue
        m = m[m["phase"] == "start"].set_index("coin")["mid_px"]
        p["mark"] = p["coin"].map(m)
        p = p[p["mark"].notna() & (p["mark"] > 0) & p["liquidation_px"].notna()].copy()
        if p.empty:
            continue
        p["ntl"] = p["szi"].abs() * p["mark"]
        p["dist"] = (p["liquidation_px"] / p["mark"] - 1.0).abs() * 100.0
        rows.append(p[["lev_type", "ntl", "dist", "lev_value"]])
    if not rows:
        print("  HL 스냅샷 없음")
        return
    d = pd.concat(rows, ignore_index=True)
    print("  표본: 청산가 보유 포지션 %d건 (최근 12스윕)" % len(d))
    print()
    print("  %-10s %10s %14s %10s" % ("거리대", "격리 건수", "격리 명목가$M", "격리 비중"))
    for lo, hi, lab in ((0, 5, "0-5%"), (5, 10, "5-10%"), (10, 20, "10-20%"),
                        (20, 50, "20-50%"), (50, 1e9, "50%+")):
        q = d[(d["dist"] >= lo) & (d["dist"] < hi)]
        if q.empty:
            continue
        iso = q[q["lev_type"] == "isolated"]
        print("  %-10s %10d %14.1f %9.1f%%"
              % (lab, len(iso), iso["ntl"].sum() / 1e6,
                 100 * iso["ntl"].sum() / max(q["ntl"].sum(), 1e-9)))
    print()
    print("  전체 격리 비중(명목가): %.1f%%" %
          (100 * d[d.lev_type == "isolated"]["ntl"].sum() / d["ntl"].sum()))
    print("  예측: 근접할수록 격리 비중이 높아야 한다(교차는 청산가가 멀다)")


def main() -> int:
    ap = argparse.ArgumentParser(description="recheck premature rejections")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0, 1000, 10000, 50000, 200000])
    a = ap.parse_args()
    U.init_stdout()
    pd.set_option("display.width", 200)
    recheck_volume(a.symbols if a.symbols else C.MAJORS, a.thresholds)
    isolated_vs_cross()
    return 0


if __name__ == "__main__":
    sys.exit(main())
