# -*- coding: utf-8 -*-
"""바닥 위치 X 의 **점 추정** — OLS 회귀. 확률모델이 아니다.

*** 이것은 확률모델이 아니다. ***
    이 스크립트는 log X 를 9개 특징에 OLS 회귀해 **숫자 하나**(조건부 중앙값)를 내고
    거기에 배수 {0.5, 0.75, 1.0, 1.25} 를 곱해 배치한다. 배수는 분위수가 아니며
    아무 확률적 의미가 없다.

    확률모델은 P(X >= u | F) 라는 **함수 전체**다. 배치의 기대값
        EV(u) = P(X>=u|F) * E[r | X>=u, F] - c
    에서 두 항 모두 분포에서 나오는데, 점 추정에서는 어느 쪽도 나오지 않는다.

    => 정확한 형태와 합격 기준은 PROB_MODEL.md.
    => 이 스크립트의 실패를 "확률모델 실패"로 읽지 말 것. 확률모델은 미검정이다.

결과 (5종 252이벤트, 2023-01-02~2026-05-23, 표본 외 126)
    Spearman(실제 X, 예측 X) = +0.252,  R2(log X) = +0.029
    짝지은 차이 27개 중 유의 1개 = 우연 기대치(1.35). 고정 offset 과 구별되지 않는다.

왜 이것인가
  완벽 예지로 바닥에 진입하면 1분 보유 +156bp / 15분 +258bp 다. 실제 전략(트리거 -2%)은
  1분 -9.9bp / 15분 +88bp 다. **격차 전부가 진입 위치에서 온다.**
  바닥에서 0.5% 만 위여도 1분 +106bp, 2% 위면 1분 -44bp 로 무너진다.

  그런데 X 는 사건마다 0.14% ~ 19.5% 로 흩어진다. 고정값으로는 맞출 수 없다.
  => 사전 정보로 X 를 **확률적으로** 근사할 수 있는지가 전략의 성패를 가른다.

앞선 시도와 무엇이 다른가
  hazard_model.py 는 S(u|x) 를 적합한 뒤 **EV 를 최대화하는 u** 를 골랐고 실패했다.
  그런데 EV 는 평평하므로 argmax 가 잡음을 따라간다. 여기서는 EV 를 거치지 않고
  **X 자체를 예측**하고, (1) 예측력을 직접 재고 (2) 예측 X 에 배치해 본다.
  (사후 판정: 둘 다 sigma 를 모형화하지 않은 같은 계열의 실패였다. TARGET_DESIGN.md §5)

변수 (전부 트리거 시점 관측 가능)
  오더북      log D(1%)/OIV,  log D(5%)/D(1%)      호가 두께와 볼록도
  거래량      vol_1, vol_2, vol_5                  최근 거래량 중 현재가 아래 1/2/5% 비중
                                                   -> '가격대별 OI' 의 대용치
  청산강도    |dOI|
  변동성      log sigma
  사건크기    log |bar_ret|, log |z|
  쏠림        LSR z, 펀딩비

  거래량 프로파일은 HAZARD_FINDINGS 에서 '교락된다'고 기각됐지만 그것은 **단변량
  3분위 층화** 결과다. 오더북·유량과 함께 다변량으로 넣으면 부호가 달라 분리될 수 있다.

실행:
    python analysis/predict_x.py
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
import analysis.bookdepth as BD                            # noqa: E402
from analysis.event_study_h2 import load, find_events, nw_tstat   # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
BAR_MS, MIN_MS = 300_000, 60_000
TTL, COST = 60, 7e-4
MAX_SNAP_LAG_MS = 2 * 60_000
GRID = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
BID = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
VOL_LOOKBACK = 3 * 1440        # 거래량 프로파일 창 = 3일(1분봉)
HOLDS = [1, 3, 15]


def load_1m(s: str) -> pd.DataFrame:
    cols = ["open_time", "high", "low", "close", "quote_volume"]
    return (pd.read_parquet(os.path.join(BULK, "klines_1m", "%s.parquet" % s))[cols]
            .sort_values("open_time").reset_index(drop=True))


def build(symbol: str, k: float, doi_thr: float, min_gap: int) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, _ = BD.load_clean(symbol, BID + ASK, verbose=False)
    if df5.empty or m1.empty or dep.empty:
        return pd.DataFrame()
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
    qv = m1["quote_volume"].to_numpy()
    n1 = len(ot)
    t5, c5 = df5["open_time"].to_numpy(), df5["close"].to_numpy()
    sig5, ret5, z5 = (df5[c].to_numpy() for c in ("sigma", "ret", "z"))
    doi5 = df5["doi"].to_numpy()
    oiv5 = df5["sum_open_interest_value"].to_numpy()
    lsr5 = df5["sum_toptrader_long_short_ratio"].to_numpy()
    dts = dep["ts_ms"].to_numpy()
    bidv, askv = dep[BID].to_numpy(), dep[ASK].to_numpy()

    out = []
    for r in ev.itertuples():
        i = r.i
        p0 = c5[i]
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sig5[i]) and sig5[i] > 0):
            continue
        if not (np.isfinite(oiv5[i]) and oiv5[i] > 0 and np.isfinite(doi5[i])):
            continue
        trig = int(t5[i])
        j0 = int(np.searchsorted(dts, trig, side="right")) - 1
        if j0 < 0 or trig - int(dts[j0]) > MAX_SNAP_LAG_MS:
            continue
        prof = (bidv[j0] if r.side == 1 else askv[j0]).astype("float64")
        if not np.all(np.isfinite(prof)) or np.any(prof <= 0) or np.any(np.diff(prof) < 0):
            continue

        a = int(np.searchsorted(ot, trig + BAR_MS, side="left"))
        b = int(np.searchsorted(ot, trig + BAR_MS + TTL * MIN_MS, side="left"))
        if a >= n1 or b <= a or b + max(HOLDS) >= n1:
            continue

        # ---- 거래량 프로파일: 최근 3일 중 현재가 아래(위) 1/2/5% 구간의 거래대금 비중
        v0 = max(a - VOL_LOOKBACK, 0)
        pc, vv = cl[v0:a], qv[v0:a]
        tot = float(np.nansum(vv))
        vol_f = {}
        for band in (0.01, 0.02, 0.05):
            if r.side == 1:
                m = (pc <= p0) & (pc >= p0 * (1 - band))
            else:
                m = (pc >= p0) & (pc <= p0 * (1 + band))
            vol_f[band] = float(np.nansum(vv[m]) / tot) if tot > 0 else np.nan

        # ---- 실제 바닥
        win = slice(a, b)
        if r.side == 1:
            kk = a + int(np.argmin(lo[win]))
            bot = lo[kk]
            X = 1.0 - bot / p0
        else:
            kk = a + int(np.argmax(hi[win]))
            bot = hi[kk]
            X = bot / p0 - 1.0
        if not (np.isfinite(X) and X > 1e-5):
            continue

        rec = {"symbol": symbol, "trig_ms": trig, "side": int(r.side), "p0": float(p0),
               "X": float(X), "a": a, "b": b,
               "log_sigma": float(np.log(sig5[i])),
               "log_bar": float(np.log(max(abs(ret5[i]), 1e-8))),
               "log_z": float(np.log(max(abs(z5[i]), 1e-8))),
               "doi_mag": float(abs(doi5[i])),
               "log_D1": float(np.log(prof[0] / oiv5[i])),
               "log_conv": float(np.log(prof[-1] / prof[0])),
               "vol_1": vol_f[0.01], "vol_2": vol_f[0.02], "vol_5": vol_f[0.05],
               "lsr": float(lsr5[i]) * r.side if np.isfinite(lsr5[i]) else np.nan}
        out.append(rec)
    return pd.DataFrame(out)


FEATS = ["log_sigma", "log_bar", "log_z", "doi_mag", "log_D1", "log_conv",
         "vol_1", "vol_2", "vol_5"]


def ols(X, y):
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def main() -> int:
    ap = argparse.ArgumentParser(description="how predictable is the bottom X")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 220)
    syms = a.symbols if a.symbols else C.MAJORS

    frames, m1c = [], {}
    for s in syms:
        try:
            d = build(s, a.k, a.doi, a.min_gap)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            m1c[s] = load_1m(s)
            U.log("%s: %d" % (s, len(d)))
    if not frames:
        U.log("no events")
        return 1
    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d = d.sort_values("dt").reset_index(drop=True)
    d = d.dropna(subset=FEATS + ["X"]).reset_index(drop=True)

    cut = len(d) // 2
    tr, te = d.iloc[:cut], d.iloc[cut:].reset_index(drop=True)
    print("\n=== 표본 ===")
    print("이벤트 %d | 심볼 %d | %s ~ %s" % (len(d), d.symbol.nunique(),
                                            d["dt"].min().date(), d["dt"].max().date()))
    print("훈련 %d / 평가 %d" % (len(tr), len(te)))
    print("X 분위: " + "  ".join("%.0f%%=%.2f%%" % (100 * q, 100 * d["X"].quantile(q))
                                 for q in (.1, .25, .5, .75, .9)))

    # ---------------------------------------------------------- 예측력
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in FEATS])
    ytr = np.log(tr["X"].to_numpy())
    w = ols(Xtr, ytr)
    Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in FEATS])
    pred = np.exp(Xte @ w)

    print("\n=== 1. log X 회귀 계수 (훈련) ===")
    for nm, wi in zip(["const"] + FEATS, w):
        print("    %-10s %+9.4f" % (nm, wi))

    rho = float(pd.Series(te["X"]).corr(pd.Series(pred), method="spearman"))
    r2 = 1 - np.sum((np.log(te["X"]) - np.log(pred)) ** 2) / \
        np.sum((np.log(te["X"]) - np.log(te["X"]).mean()) ** 2)
    print("\n=== 2. 표본 외 예측력 ===")
    print("  Spearman(실제 X, 예측 X) = %+.3f" % rho)
    print("  log X 의 R2 = %+.3f" % r2)
    print("  예측 X 분위: " + "  ".join("%.0f%%=%.2f%%" % (100 * q, 100 * np.quantile(pred, q))
                                        for q in (.1, .5, .9)))
    print("  비교: 상수(훈련 중앙 %.2f%%) 로 예측하면 Spearman=0, R2=%.3f"
          % (100 * tr["X"].median(),
             1 - np.sum((np.log(te["X"]) - np.log(tr["X"].median())) ** 2)
             / np.sum((np.log(te["X"]) - np.log(te["X"]).mean()) ** 2)))

    # ---------------------------------------------------------- 배치
    print("\n=== 3. 예측 X 에 배치하면 (평가구간 %d건) ===" % len(te))
    print("  %-16s %8s %8s %10s %10s %10s"
          % ("배치", "u중앙%", "체결률", "1분bp", "3분bp", "15분bp"))

    def run(u_arr, lab):
        res = {h: [] for h in HOLDS}
        nf = 0
        for pos in range(len(te)):
            r = te.iloc[pos]
            u = u_arr[pos]
            if not np.isfinite(u) or u <= 0:
                continue
            m1 = m1c[r["symbol"]]
            ot = m1["open_time"].to_numpy()
            lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
            n1 = len(ot)
            aa, bb = int(r["a"]), int(r["b"])
            limit = r["p0"] * (1 - u) if r["side"] == 1 else r["p0"] * (1 + u)
            seg = (lo[aa:bb] <= limit) if r["side"] == 1 else (hi[aa:bb] >= limit)
            idx = np.flatnonzero(seg)
            if idx.size == 0:
                continue
            nf += 1
            j = aa + int(idx[0])
            for h in HOLDS:
                e = min(j + h, n1 - 1)
                res[h].append((cl[e] / limit - 1.0) * r["side"] - COST)
        fr = nf / max(len(te), 1)
        vals = [1e4 * np.mean(res[h]) * fr if res[h] else np.nan for h in HOLDS]
        print("  %-16s %8.2f %7.1f%% %10.1f %10.1f %10.1f"
              % (lab, 100 * np.nanmedian(u_arr), 100 * fr, *vals))

    for mult in (0.5, 0.75, 1.0, 1.25):
        run(np.clip(pred * mult, 0.001, 0.15), "예측X x %.2f" % mult)
    for u0 in (0.005, 0.01, 0.02, 0.03):
        run(np.full(len(te), u0), "고정 %.1f%%" % (100 * u0))
    run(np.clip(te["X"].to_numpy(), 0.001, 0.5), "실제X (예지)")
    print("\n  '예측X' 가 '고정' 을 넘고 '실제X' 에 가까울수록 모델이 작동하는 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
