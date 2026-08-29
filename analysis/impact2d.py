# -*- coding: utf-8 -*-
"""2차원(시간 x 가격) 청산면 기반 충격 분석 — "쌓인 게 터질 때" 를 제대로 재기.

앞선 impact.py의 결함
  1분 칸마다 청산을 스칼라 하나로 합쳤다. 그러면 가격축이 사라져서
  "$10M이 0.2% 폭에 몰려 터진 것"과 "$10M이 5% 폭에 흩어져 터진 것"이 같은 값이 된다.
  게다가 표본이 사소한 이벤트에 지배됐다 — 칸당 청산 명목가 중앙값이 $59였다.
  가설은 '쌓인 물량이 한 번에 터질 때'에 관한 것인데 잡음으로 평균을 낸 셈이다.

이 스크립트가 만드는 것
  (시간 버킷 x 가격 빈) 격자 위의 청산 명목가 = 청산면 L(t, p).
  각 버스트에 대해 스칼라가 아니라 **분포의 형태**를 특징화한다:
    V_total    총 청산 명목가
    V_peak     단일 가격빈 최대 명목가 (= 가장 두꺼운 군집이 터진 규모)
    conc       V_peak / V_total  (집중도. 1에 가까우면 한 점에 몰림)
    span_pct   청산이 걸친 가격 폭(%)
    density    V_total / span_pct  (가격 1% 폭당 청산 명목가)
  가설이 옳다면 같은 V_total이라도 **집중/고밀도일수록 더 밀려야** 한다.

동시성 처리
  청산은 가격 움직임의 결과이기도 하다. 따라서 예측변수는 버스트 구간 안에서만 관측하고,
  변위는 **그 구간이 끝난 뒤부터** 잰다. 예측변수가 완전히 관측된 뒤의 변위만 종속변수다.

실행:
    python analysis/impact2d.py
    python analysis/impact2d.py --bucket-sec 30 --min-usd 100000
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
from analysis.impact import load_1m_with_volume, load_liq   # noqa: E402

MIN_MS = 60_000


def bursts(symbol: str, bucket_sec: int, bin_bps: float, min_usd: float,
           horizons: list[int], full_only: bool) -> pd.DataFrame:
    """버스트(청산 집중 구간)를 찾아 분포 형태 + 이후 변위를 기록."""
    liq = load_liq(symbol, full_only)
    if liq.empty:
        return pd.DataFrame()
    m1 = load_1m_with_volume(symbol)
    if m1.empty:
        return pd.DataFrame()

    step = bucket_sec * 1000
    liq = liq.copy()
    liq["bucket"] = (liq["ts_ms"] // step) * step

    ot = m1["open_time"].to_numpy()
    close = m1["close"].to_numpy()
    high = m1["high"].to_numpy()
    low = m1["low"].to_numpy()
    vol = m1["volume"].to_numpy() if "volume" in m1.columns else np.full(len(ot), np.nan)
    n1 = len(ot)

    # 사전 변동성/거래량 (1분봉 기준, 직전 1일 창, 현재 제외)
    s = pd.Series(close)
    ret = s / s.shift(1) - 1.0
    sigma = ret.shift(1).rolling(1440, min_periods=360).std().to_numpy()
    dv = pd.Series(vol * close)
    adv = dv.shift(1).rolling(1440, min_periods=360).mean().to_numpy()

    out = []
    for (bkt, side), g in liq.groupby(["bucket", "pos_side"]):
        if side not in ("long", "short"):
            continue
        v_total = float(g["notional"].sum())
        if v_total < min_usd:
            continue
        # 버스트가 끝난 시각 이후의 첫 1분봉부터 변위를 잰다
        end_ms = int(bkt) + step
        j = int(np.searchsorted(ot, end_ms, side="left"))
        if j <= 0 or j + max(horizons) >= n1:
            continue
        ref = close[j - 1]
        if not (np.isfinite(ref) and ref > 0) or not np.isfinite(sigma[j - 1]) or sigma[j - 1] <= 0:
            continue
        if not np.isfinite(adv[j - 1]) or adv[j - 1] <= 0:
            continue

        # 가격축 분포 형태
        pxs = g["price"].to_numpy()
        w = g["notional"].to_numpy()
        binw = ref * bin_bps / 1e4
        idx = np.floor(pxs / binw).astype("int64")
        agg = pd.Series(w).groupby(idx).sum()
        v_peak = float(agg.max())
        conc = v_peak / v_total
        span_pct = float((pxs.max() - pxs.min()) / ref * 100.0)
        density = v_total / max(span_pct, bin_bps / 100.0)

        # 버스트 구간 동안 가격이 이미 얼마나 움직였나 — 이걸 통제하지 않으면
        # 'span이 넓다'가 '이미 빠르게 움직이는 중'을 대리해 가짜 효과를 만든다.
        j0 = int(np.searchsorted(ot, int(bkt), side="left"))
        pre = close[j0 - 1] if j0 >= 1 else np.nan
        burst_ret = abs(ref / pre - 1.0) if (np.isfinite(pre) and pre > 0) else np.nan

        rec = {"symbol": symbol, "bucket": int(bkt), "side": side,
               "burst_ret": float(burst_ret) if np.isfinite(burst_ret) else np.nan,
               "v_total": v_total, "v_peak": v_peak, "conc": conc,
               "span_pct": span_pct, "density": density,
               "n_liq": int(len(g)),
               "sigma": float(sigma[j - 1]), "adv": float(adv[j - 1]),
               "ref": float(ref)}
        for h in horizons:
            seg_lo = low[j:j + h].min()
            seg_hi = high[j:j + h].max()
            # 청산 방향으로 더 밀린 폭 (양수 = 계속 밀림)
            push = (ref / seg_lo - 1.0) if side == "long" else (seg_hi / ref - 1.0)
            rec["push%d" % h] = push
            rec["pushsig%d" % h] = push / sigma[j - 1]
        out.append(rec)
    return pd.DataFrame(out)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(y) < 40:
        return (np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan), len(y))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    s2 = float(r @ r) / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return coef, se, len(y)


def main() -> int:
    ap = argparse.ArgumentParser(description="2D (time x price) liquidation impact")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--bucket-sec", type=int, default=30)
    ap.add_argument("--bin-bps", type=float, default=10.0, help="price bin width in bps")
    ap.add_argument("--min-usd", type=float, default=50000.0,
                    help="only bursts above this notional (noise floor)")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 15])
    ap.add_argument("--all-period", action="store_true")
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.MAJORS
    full_only = not a.all_period
    U.log("impact2d: bucket=%ds bin=%.0fbps min=$%.0f full_feed=%s"
          % (a.bucket_sec, a.bin_bps, a.min_usd, full_only))

    frames = []
    for s in symbols:
        try:
            d = bursts(s, a.bucket_sec, a.bin_bps, a.min_usd, a.horizons, full_only)
        except FileNotFoundError as e:
            U.log(str(e)); continue
        if not d.empty:
            frames.append(d)
            U.log("%s: %d bursts" % (s, len(d)))
    if not frames:
        U.log("no bursts above threshold")
        return 1
    d = pd.concat(frames, ignore_index=True)
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "impact2d_bursts.parquet"))

    pd.set_option("display.width", 220)
    print("\n=== 버스트 표본 (임계 $%.0f 이상) ===" % a.min_usd)
    print("n=%d | 청산 총액 $%.0fM | 중앙 버스트 $%.0f"
          % (len(d), d.v_total.sum() / 1e6, d.v_total.median()))
    print(d[["v_total", "v_peak", "conc", "span_pct", "density", "n_liq"]]
          .describe(percentiles=[0.25, 0.5, 0.75, 0.95]).round(3).to_string())

    h = a.horizons[1] if len(a.horizons) > 1 else a.horizons[0]
    y = np.log(d["pushsig%d" % h].clip(lower=1e-4))

    print("\n=== 규모만 vs 규모+집중도 (h=%d분, 종속=log(변위/sigma)) ===" % h)
    lv = np.log(d["v_total"].clip(lower=1.0))
    X1 = np.column_stack([np.ones(len(d)), lv])
    c1, s1, n1 = ols(X1, y.to_numpy())
    print("  [규모만]      log V_total  %.4f (t=%.2f)   n=%d" % (c1[1], c1[1] / s1[1], n1))

    X2 = np.column_stack([np.ones(len(d)), lv, d["conc"].to_numpy(),
                          np.log(d["span_pct"].clip(lower=1e-3).to_numpy())])
    c2, s2, n2 = ols(X2, y.to_numpy())
    print("  [+집중도]     log V_total  %.4f (t=%.2f)" % (c2[1], c2[1] / s2[1]))
    print("                conc         %.4f (t=%.2f)   <- 한 가격빈에 몰릴수록" % (c2[2], c2[2] / s2[2]))
    print("                log span%%    %.4f (t=%.2f)   <- 넓게 흩어질수록" % (c2[3], c2[3] / s2[3]))
    print("                n=%d" % n2)

    # 버스트 구간 자체의 가격 움직임을 통제한 뒤에도 집중도가 남는가
    dd = d[np.isfinite(d["burst_ret"])].copy()
    if len(dd) > 100:
        yd = np.log(dd["pushsig%d" % h].clip(lower=1e-4)).to_numpy()
        X4 = np.column_stack([np.ones(len(dd)),
                              np.log(dd["v_total"].clip(lower=1.0).to_numpy()),
                              dd["conc"].to_numpy(),
                              np.log(dd["span_pct"].clip(lower=1e-3).to_numpy()),
                              np.log(dd["burst_ret"].clip(lower=1e-6).to_numpy() /
                                     dd["sigma"].to_numpy())])
        c4, s4, n4 = ols(X4, yd)
        print("  [+버스트 움직임 통제]")
        print("                log V_total  %.4f (t=%.2f)" % (c4[1], c4[1] / s4[1]))
        print("                conc         %.4f (t=%.2f)" % (c4[2], c4[2] / s4[2]))
        print("                log span%%    %.4f (t=%.2f)" % (c4[3], c4[3] / s4[3]))
        print("                log |burst move|/sigma  %.4f (t=%.2f)  <- 통제변수" % (c4[4], c4[4] / s4[4]))
        print("                n=%d" % n4)

    X3 = np.column_stack([np.ones(len(d)), np.log(d["density"].clip(lower=1.0).to_numpy())])
    c3, s3, n3 = ols(X3, y.to_numpy())
    print("  [밀도만]      log density  %.4f (t=%.2f)   n=%d" % (c3[1], c3[1] / s3[1], n3))

    print("\n=== 규모 x 집중도 이중 정렬 (변위 중앙값, sigma 배수) ===")
    d2 = d.copy()
    d2["vq"] = pd.qcut(d2["v_total"].rank(method="first"), 4, labels=["V1 작음", "V2", "V3", "V4 큼"])
    d2["cq"] = pd.qcut(d2["conc"].rank(method="first"), 3, labels=["흩어짐", "중간", "집중"])
    t = d2.pivot_table(index="vq", columns="cq", values="pushsig%d" % h,
                       aggfunc="median", observed=True)
    print(t.round(2).to_string())
    print("  가설: 같은 행(같은 규모) 안에서 오른쪽(집중)이 더 커야 한다")
    cnt = d2.pivot_table(index="vq", columns="cq", values="v_total", aggfunc="size", observed=True)
    print("\n  (셀별 표본 수)")
    print(cnt.to_string())

    print("\n=== 규모 구간별 충격 (상위 꼬리 집중 확인) ===")
    d2["vq8"] = pd.qcut(d2["v_total"].rank(method="first"), 8, labels=False)
    g = d2.groupby("vq8").agg(n=("v_total", "size"), v_med=("v_total", "median"),
                              push_med=("pushsig%d" % h, "median"),
                              push_p90=("pushsig%d" % h, lambda s: s.quantile(0.9)))
    print(g.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
