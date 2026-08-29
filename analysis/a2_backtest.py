# -*- coding: utf-8 -*-
"""A-2 — 재구성 지도로 연쇄청산의 **정지점**을 예측할 수 있는가. 6년 백테스트.

무엇이 A-1 과 다른가
  A-1 은 "청산이 어디서 일어나는가" 를 봤다. 그런데 '현재가에 가까울수록 청산이 많다'
  는 뻔한 사실이고(거리만으로 R2=0.14), 우리가 알고 싶은 것은 **연쇄가 어디서 멈추는가** 다.

정지점의 정의 (TARGET_DESIGN 6.1 의 고정점)
      u* = inf{ u : D(u) >= V(u) },   V(u) = 적분_0^u L_hat(p) dp
  누적 호가가 누적 강제공급을 처음으로 넘어서는 지점. 지도가 얇아지면 연쇄가 연료를
  잃고 거기서 멈춘다.

  현실 보정 하나: 화면의 호가를 전부 먹을 수 있는 게 아니고(잔존율 W), 강제분 외
  재량매도 M(u) 도 있다. 둘을 스칼라 theta 로 묶어 **훈련 표본에서 적합**한다:
      u*(theta) = inf{ u : theta * D(u) >= V(u) }

세 검정
  (1) 예측력  log X ~ log u*.  표본 외 R2 를 사전 관측 기준선(0.014)과 비교한다.
              지도가 사전 정보로서 값어치를 하려면 여기서 올라야 한다.
  (2) 규모별  큰 사건(진짜 캐스케이드)에서 더 잘 맞는가. 사용자 요구의 핵심.
  (3) 배치    u* 에 지정가를 걸었을 때 고정 offset 을 이기는가.
              짝지은 차이 + Bonferroni. 무차별성을 깨는지가 최종 질문이다.

*** 무차별성 경고: S(u) 와 E[r|fill] 이 상쇄해 '어디 걸어도 같다' 가 여섯 번 확인됐다.
    (1)(2)가 통과해도 (3)이 떨어질 수 있다. 셋은 다른 질문이다. ***

실행:
    python analysis/a2_backtest.py
    python analysis/a2_backtest.py --hold 15 --lookback 30 --half-life 7
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
import analysis.bookdepth as BD                                    # noqa: E402
from analysis.heatmap import load_fr, load_oi, build_map, MAP_GRID  # noqa: E402
from analysis.event_study_h2 import load, find_events              # noqa: E402
from analysis.vd_structure import load_1m, PRE                     # noqa: E402

BAR_MS, MIN_MS = 300_000, 60_000
TTL = 60
COST = 7e-4
MAX_SNAP_LAG_MS = 2 * 60_000
BID = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
DEPTH_U = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
U_MIN, U_MAX = 0.001, 0.25
NBOOT = 4000
RNG = np.random.default_rng(23)


def fit_depth(dprof: np.ndarray) -> tuple:
    """Cum(u) = B u^kappa 를 5밴드에 로그-로그 적합. liq_cluster.py 와 같은 형태.

    1% 미만 구간이 중요한데 밴드가 1% 부터 시작하므로 **외삽이 필요하다.**
    상수로 채우면(np.interp 의 left) u->0 에서 깊이가 남아 있는 것이 되어
    u* 가 인위적으로 작아진다. 멱함수 외삽이 물리적으로 맞다.
    """
    x, y = np.log(DEPTH_U), np.log(dprof)
    A = np.column_stack([np.ones(5), x])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(w[1]), float(w[0])          # (kappa, logB)


def solve_ustar(cum_map: np.ndarray, ug: np.ndarray,
                kappa: float, logB: float, theta: float) -> float:
    """theta*D(u) >= V(u) 가 처음 성립하는 u. 없으면 U_MAX."""
    D = np.exp(logB + kappa * np.log(ug))
    ok = (theta * D) >= cum_map
    if not np.any(ok):
        return U_MAX
    return float(ug[int(np.argmax(ok))])


def build(symbol, fr, k, doi_thr, min_gap, lb_bars, hl_bars, iso_share):
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, _ = BD.load_clean(symbol, BID + ASK, verbose=False)
    if df5.empty or m1.empty or dep.empty:
        return pd.DataFrame()
    oi = load_oi(symbol)
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
    n1 = len(ot)
    t5, c5 = df5["open_time"].to_numpy(), df5["close"].to_numpy()
    sig5, ret5, z5 = (df5[c].to_numpy() for c in ("sigma", "ret", "z"))
    doi5, oiv5 = df5["doi"].to_numpy(), df5["sum_open_interest_value"].to_numpy()
    dts = dep["ts_ms"].to_numpy()
    bidv, askv = dep[BID].to_numpy(), dep[ASK].to_numpy()
    oi_t = oi["open_time"].to_numpy()

    ug = np.arange(0.001, U_MAX + 1e-9, 0.001)
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
        # 지도: 트리거 바에서 본 것 (사전 정보만)
        gi = int(np.searchsorted(oi_t, trig, side="right")) - 1
        if gi < lb_bars + 10 or gi >= len(oi) - 1:
            continue
        pm, L, S = build_map(oi, gi, fr, lb_bars, hl_bars, iso_share)
        mp = L if r.side == 1 else S
        sgn = -1.0 if r.side == 1 else 1.0
        m = (MAP_GRID * sgn) > 0
        gu, gw = (MAP_GRID[m] * sgn), mp[m]
        o = np.argsort(gu)
        gu, gw = gu[o], gw[o]
        if gw.sum() <= 0:
            continue
        cum_map = np.interp(ug, gu, np.cumsum(gw))

        a = int(np.searchsorted(ot, trig + BAR_MS, side="left"))
        b = a + TTL
        if a >= n1 or b + 30 >= n1:
            continue
        w = slice(a, b)
        if r.side == 1:
            kk = a + int(np.argmin(lo[w])); X = 1.0 - lo[kk] / p0
        else:
            kk = a + int(np.argmax(hi[w])); X = hi[kk] / p0 - 1.0
        if not (np.isfinite(X) and X > 1e-5):
            continue
        kappa, logB = fit_depth(prof)
        if not (np.isfinite(kappa) and kappa > 0):
            continue
        out.append({"symbol": symbol, "trig_ms": trig, "side": int(r.side),
                    "p0": float(p0), "a": a, "b": b, "X": float(X),
                    "log_sigma": float(np.log(sig5[i])),
                    "log_bar": float(np.log(max(abs(ret5[i]), 1e-8))),
                    "log_z": float(np.log(max(abs(z5[i]), 1e-8))),
                    "doi_mag": float(abs(doi5[i])),
                    "log_D1": float(np.log(prof[0] / oiv5[i])),
                    "log_conv": float(np.log(prof[-1] / prof[0])),
                    "cum_map": cum_map, "kappa": kappa, "logB": logB,
                    "map_1pct": float(np.interp(0.01, gu, np.cumsum(gw))),
                    "D_1pct": float(prof[0])})
    return pd.DataFrame(out)


def oos(tr, te, feats):
    if len(tr) <= len(feats) + 1 or len(te) < 5:
        return np.nan, np.nan
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in feats])
    Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in feats])
    w = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ np.log(tr["X"].to_numpy()))
    p = Xte @ w
    y = np.log(te["X"].to_numpy())
    den = float(np.sum((y - y.mean()) ** 2))
    if den <= 0:
        return np.nan, np.nan
    return (1 - float(np.sum((y - p) ** 2)) / den,
            float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def pnl(te, m1c, u_arr, hold):
    out = np.zeros(len(te))
    filled = np.zeros(len(te), dtype=bool)
    for i in range(len(te)):
        u = u_arr[i]
        if not np.isfinite(u) or u <= 0:
            continue
        r = te.iloc[i]
        lo, hi, cl = m1c[r["symbol"]]
        n1 = len(cl)
        aa, bb = int(r["a"]), int(r["b"])
        if bb > n1:
            continue
        lim = r["p0"] * (1 - u) if r["side"] == 1 else r["p0"] * (1 + u)
        seg = (lo[aa:bb] <= lim) if r["side"] == 1 else (hi[aa:bb] >= lim)
        idx = np.flatnonzero(seg)
        if idx.size == 0:
            continue
        j = aa + int(idx[0])
        e = min(j + hold, n1 - 1)
        out[i] = 1e4 * ((cl[e] / lim - 1.0) * r["side"] - COST)
        filled[i] = True
    return out, filled


def boot(v, nb=NBOOT, alpha=0.05):
    n = len(v)
    idx = RNG.integers(0, n, size=(nb, n))
    b = v[idx].mean(axis=1)
    return float(np.mean(v)), float(np.percentile(b, 100 * alpha / 2)), \
        float(np.percentile(b, 100 * (1 - alpha / 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description="A-2: map-implied stopping point, 6y backtest")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--lookback", type=float, default=30.0)
    ap.add_argument("--half-life", type=float, default=7.0)
    ap.add_argument("--iso-share", type=float, default=0.055)
    a = ap.parse_args()
    U.init_stdout()

    fr = load_fr()
    syms = a.symbols if a.symbols else C.MAJORS
    lb, hl = int(a.lookback * 288), a.half_life * 288
    frames, m1c = [], {}
    for s in syms:
        try:
            d = build(s, fr, a.k, a.doi, a.min_gap, lb, hl, a.iso_share)
        except FileNotFoundError as e:
            U.log(str(e)); continue
        if d.empty:
            continue
        frames.append(d)
        m1 = load_1m(s)
        m1c[s] = (m1["low"].to_numpy(), m1["high"].to_numpy(), m1["close"].to_numpy())
        U.log("%s: %d" % (s, len(d)))
    if not frames:
        U.log("이벤트 없음"); return 1
    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d = d.sort_values("dt").reset_index(drop=True)
    cut = len(d) // 2
    tr, te = d.iloc[:cut].reset_index(drop=True), d.iloc[cut:].reset_index(drop=True)
    ug = np.arange(0.001, U_MAX + 1e-9, 0.001)

    print("\n" + "=" * 74)
    print("A-2 — 재구성 지도로 연쇄청산의 정지점을 예측할 수 있는가")
    print("=" * 74)
    print("표본 %d | %d종 | %s ~ %s | 훈련 %d / 평가 %d | 보유 %d분"
          % (len(d), d.symbol.nunique(), d["dt"].min().date(), d["dt"].max().date(),
             len(tr), len(te), a.hold))

    # theta 를 훈련 표본에서 적합 (u* 의 중앙이 X 의 중앙과 맞도록)
    def med_us(df, th):
        return np.median([solve_ustar(r["cum_map"], ug, r["kappa"], r["logB"], th)
                          for _, r in df.iterrows()])
    # med_us 는 theta 에 대해 **감소**한다 (theta 크면 조건이 일찍 성립 -> u* 작음).
    lo_t, hi_t = 1e-6, 1e6
    tgt = float(tr["X"].median())
    for _ in range(60):
        mid = np.sqrt(lo_t * hi_t)
        if med_us(tr, mid) > tgt:
            lo_t = mid          # u* 가 크다 -> theta 를 키워야 한다
        else:
            hi_t = mid
    theta = float(np.sqrt(lo_t * hi_t))
    print("theta 적합(훈련) = %.4g   (훈련 u* 중앙 %.3f%% vs X 중앙 %.3f%%)"
          % (theta, 100 * med_us(tr, theta), 100 * tgt))

    for df in (tr, te):
        df["u_star"] = [solve_ustar(r["cum_map"], ug, r["kappa"], r["logB"], theta)
                        for _, r in df.iterrows()]
        df["log_us"] = np.log(df["u_star"].clip(lower=U_MIN))
        df["log_MD"] = np.log((df["map_1pct"] / df["D_1pct"]).clip(lower=1e-9))

    print("\n--- 1. 예측력: log X ~ log u*  (표본 외) ---")
    print("  %-34s %9s %9s" % ("변수", "R2", "Spearman"))
    for lab, f in (("사전 관측만 (기준선)", PRE),
                   ("지도 정지점 log u* 단독", ["log_us"]),
                   ("지도/호가 비 log(map/D) 단독", ["log_MD"]),
                   ("사전 + log u*", PRE + ["log_us"]),
                   ("사전 + log u* + log(map/D)", PRE + ["log_us", "log_MD"])):
        r2, rho = oos(tr, te, f)
        print("  %-34s %+9.3f %+9.3f" % (lab, r2, rho))
    print("  기준선을 넘어야 지도가 **사전 정보로서** 값어치를 한 것이다.")

    print("\n--- 2. 규모별: 큰 사건(진짜 캐스케이드)에서 더 맞는가 ---")
    te2 = te.copy()
    te2["mag"] = pd.qcut(te2["doi_mag"], 3, labels=False)
    print("  %-10s %8s %9s %9s %7s" % ("|dOI| 3분위", "중앙", "R2(u*)", "Spearman", "n"))
    for qi in (0, 1, 2):
        sub = te2[te2.mag == qi]
        if len(sub) < 20:
            continue
        r2, rho = oos(tr, sub, ["log_us"])
        print("  %-10s %7.2f%% %+9.3f %+9.3f %7d"
              % (["하", "중", "상"][qi], 100 * sub.doi_mag.median(), r2, rho, len(sub)))

    print("\n--- 3. 배치: u* 에 걸면 고정 offset 을 이기는가 (평가 %d건) ---" % len(te))
    res = {}
    print("  %-16s %8s %8s %10s %22s" % ("규칙", "u중앙%", "체결률", "EV bp", "95% CI"))
    for mult in (0.75, 1.0, 1.5):
        u = np.clip(te["u_star"].to_numpy() * mult, U_MIN, U_MAX)
        v, f = pnl(te, m1c, u, a.hold)
        res[("MAP", mult)] = v
        m, l, h = boot(v)
        print("  %-16s %8.2f %7.1f%% %10.1f     [%+7.1f, %+7.1f]"
              % ("u* x%.2f" % mult, 100 * np.median(u), 100 * f.mean(), m, l, h))
    for u0 in (0.005, 0.01, 0.02, 0.03):
        v, f = pnl(te, m1c, np.full(len(te), u0), a.hold)
        res[("FIX", u0)] = v
        m, l, h = boot(v)
        print("  %-16s %8.2f %7.1f%% %10.1f     [%+7.1f, %+7.1f]"
              % ("고정 %.1f%%" % (100 * u0), 100 * u0, 100 * f.mean(), m, l, h))

    print("\n  -- 짝지은 차이 (u* x1.0 기준), Bonferroni --")
    base = res[("MAP", 1.0)]
    comps = [("FIX", u0) for u0 in (0.005, 0.01, 0.02, 0.03)]
    alpha = 0.05 / len(comps)
    for kk in comps:
        diff = base - res[kk]
        m, l, h = boot(diff, alpha=alpha)
        sig = " ***" if (l > 0 or h < 0) else ""
        print("     u* - 고정 %.1f%%  = %+7.1f bp  [%+7.1f, %+7.1f]%s"
              % (100 * kk[1], m, l, h, sig))

    print("\n  *** 무차별성 경고: (1)(2)가 통과해도 (3)은 떨어질 수 있다.")
    print("      S(u) 와 E[r|fill] 의 상쇄가 여섯 번 확인됐다. 셋은 다른 질문이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
