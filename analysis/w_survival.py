# -*- coding: utf-8 -*-
"""QW — 호가 잔존율 W 를 예보할 수 있는가.

배경 (PROB_MODEL.md 9절)
  확률모델이 떨어진 원인은 D 를 관측 상수로 놓은 것이다. 실제로 먹을 수 있는 깊이는

      D_eff = D_t * W,      log X = a + gamma(log V - log D_t - log W)

  이고 W(잔존율)는 사실상 미관측이다.
  u_act/u_pred 의 산포 [1.7, 13.8] 가 곧 W 의 분포다(LIQ_FINDINGS).
  (주: "1/W = 109~523배" 는 2025-10-10 극단일 수치다. 이 스크립트가 재보고,
   전체 청산 이벤트에서는 중앙 3.4배 / p90 17배로 정정됐다.)

이 스크립트가 답하는 것 — V 때와 같은 3단 구조
  (1) 상한   실현 W 를 회귀에 넣으면 log X 가 얼마나 더 설명되는가.
             안 오르면 W 는 애초에 중요하지 않다 -> QW 폐기.
  (2) 예보력 사전 관측치로 log W 를 표본 외에서 맞힐 수 있는가.
  (3) 결합   '예측 W' 를 넣으면 X 예보가 실제로 좋아지는가.

*** 측정 한계 — 반드시 함께 보고할 것 ***
  30초 스냅샷은 **소진(먹혀서 줄어듦)과 철수(취소돼서 사라짐)를 분리하지 못한다.**
  D_min/D_pre 는 둘이 섞인 값이다. 순수 철수분은 웹소켓 diff 로만 분리된다
  (TARGET_DESIGN 3.1). 여기서는 혼재된 값의 **예측 가능성**만 본다.
  다만 방향은 보수적이다 — 혼재값조차 예측 불가면 순수 철수분도 어렵다.

실행:
    python analysis/w_survival.py
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
import analysis.bookdepth as BD                             # noqa: E402
from analysis.event_study_h2 import load, find_events       # noqa: E402
from analysis.vd_structure import ols_cluster, PRE          # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
BAR_MS, MIN_MS = 300_000, 60_000
TTL = 60
MAX_SNAP_LAG_MS = 2 * 60_000
BID = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
BASE_TICKS = 120                  # 30초 x 120 = 60분 기준선
MIN_PATH = 20                     # 창 안 스냅샷 최소 개수


def load_1m(s: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(BULK, "klines_1m", "%s.parquet" % s))
    cols = ["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    return df[[c for c in cols if c in df.columns]].sort_values("open_time").reset_index(drop=True)


def build(symbol: str, k: float, doi_thr: float, min_gap: int) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, _ = BD.load_clean(symbol, BID + ASK, verbose=False)
    if df5.empty or m1.empty or dep.empty:
        return pd.DataFrame()
    if "taker_buy_quote_volume" not in m1.columns:
        return pd.DataFrame()
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi = m1["low"].to_numpy(), m1["high"].to_numpy()
    qv, tbq = m1["quote_volume"].to_numpy(), m1["taker_buy_quote_volume"].to_numpy()
    n1 = len(ot)
    t5, c5 = df5["open_time"].to_numpy(), df5["close"].to_numpy()
    sig5, ret5, z5 = (df5[c].to_numpy() for c in ("sigma", "ret", "z"))
    doi5, oiv5 = df5["doi"].to_numpy(), df5["sum_open_interest_value"].to_numpy()
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
        b = a + TTL
        if a >= n1 or b >= n1:
            continue
        w = slice(a, b)
        if r.side == 1:
            kk = a + int(np.argmin(lo[w])); X = 1.0 - lo[kk] / p0
        else:
            kk = a + int(np.argmax(hi[w])); X = hi[kk] / p0 - 1.0
        if not (np.isfinite(X) and X > 1e-5):
            continue

        # ---- 깊이 궤적 (진행방향 1% 밴드)
        col = bidv[:, 0] if r.side == 1 else askv[:, 0]
        s0 = int(np.searchsorted(dts, int(ot[a]), side="left"))
        s1 = int(np.searchsorted(dts, int(ot[a]) + TTL * MIN_MS, side="left"))
        path = col[s0:s1].astype("float64")
        path = path[np.isfinite(path) & (path > 0)]
        if len(path) < MIN_PATH:
            continue
        D_pre = float(prof[0])
        D_min = float(path.min())
        W_min = D_min / D_pre
        # 바닥 시각의 깊이
        t_bot = int(ot[kk])
        jb = int(np.searchsorted(dts, t_bot, side="right")) - 1
        D_bot = float(col[jb]) if (jb >= 0 and np.isfinite(col[jb]) and col[jb] > 0) else np.nan
        W_bot = D_bot / D_pre if np.isfinite(D_bot) else np.nan
        # 캐스케이드 전 기준선 대비 (트리거 시점 자체가 이미 얇을 수 있다)
        pb = max(s0 - BASE_TICKS, 0)
        base = col[pb:s0].astype("float64")
        base = base[np.isfinite(base) & (base > 0)]
        W_pre = D_pre / float(np.median(base)) if base.size >= 20 else np.nan

        def sellvol(i0, i1):
            s = slice(i0, i1)
            tot, buy = float(np.nansum(qv[s])), float(np.nansum(tbq[s]))
            v = (tot - buy) if r.side == 1 else buy
            return v if v > 0 else np.nan

        v60 = sellvol(a, min(a + 60, n1))
        if not np.isfinite(v60):
            continue
        out.append({
            "symbol": symbol, "trig_ms": trig, "X": float(X),
            "log_sigma": float(np.log(sig5[i])),
            "log_bar": float(np.log(max(abs(ret5[i]), 1e-8))),
            "log_z": float(np.log(max(abs(z5[i]), 1e-8))),
            "doi_mag": float(abs(doi5[i])),
            "log_D1": float(np.log(D_pre / oiv5[i])),
            "log_conv": float(np.log(prof[-1] / prof[0])),
            "VD60": float(np.log(v60 / D_pre)),
            "W_min": float(W_min), "W_bot": float(W_bot), "W_pre": float(W_pre),
            "log_Wmin": float(np.log(max(W_min, 1e-6))),
            "log_Wbot": float(np.log(max(W_bot, 1e-6))) if np.isfinite(W_bot) else np.nan,
            "n_path": len(path)})
    return pd.DataFrame(out)


def oos(tr, te, feats, target="X", logt=True):
    if len(tr) <= len(feats) + 1 or len(te) < 5:
        return np.nan, np.nan
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in feats])
    Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in feats])
    ytr = np.log(tr[target].to_numpy()) if logt else tr[target].to_numpy()
    yte = np.log(te[target].to_numpy()) if logt else te[target].to_numpy()
    w = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ ytr)
    p = Xte @ w
    den = float(np.sum((yte - yte.mean()) ** 2))
    if den <= 0:
        return np.nan, np.nan
    r2 = 1.0 - float(np.sum((yte - p) ** 2)) / den
    rho = float(pd.Series(yte).corr(pd.Series(p), method="spearman"))
    return r2, rho


def main() -> int:
    ap = argparse.ArgumentParser(description="QW: is depth survival W predictable")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    a = ap.parse_args()

    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    frames = []
    for s in syms:
        try:
            d = build(s, a.k, a.doi, a.min_gap)
        except FileNotFoundError as e:
            U.log(str(e)); continue
        if not d.empty:
            frames.append(d); U.log("%s: %d" % (s, len(d)))
    if not frames:
        U.log("이벤트 없음"); return 1
    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d["day"] = d["dt"].dt.strftime("%Y-%m-%d")
    d = d.sort_values("dt").reset_index(drop=True)
    d = d.dropna(subset=PRE + ["VD60", "log_Wmin", "X"]).reset_index(drop=True)
    cut = len(d) // 2
    tr, te = d.iloc[:cut], d.iloc[cut:].reset_index(drop=True)

    print("\n" + "=" * 74)
    print("QW — 호가 잔존율 W 를 예보할 수 있는가")
    print("=" * 74)
    print("표본 %d | %d종 | %s ~ %s | 훈련 %d / 평가 %d"
          % (len(d), d.symbol.nunique(), d["dt"].min().date(), d["dt"].max().date(),
             len(tr), len(te)))
    print("*** W = D_min/D_pre 는 소진과 철수가 섞인 값이다. 웹소텟 diff 로만 분리된다.")

    print("\n--- 0. W 의 분포 ---")
    print("  %-10s %9s %9s %9s %9s %9s" % ("", "p10", "p25", "중앙", "p75", "p90"))
    for lab, col in (("W_min", "W_min"), ("W_bot", "W_bot"), ("W_pre", "W_pre")):
        s = d[col].dropna()
        if s.empty:
            continue
        print("  %-10s %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (lab, *[s.quantile(q) for q in (.1, .25, .5, .75, .9)]))
    print("  1/W_min : 중앙 %.1f배  p90 %.1f배  최대 %.1f배"
          % (1 / d.W_min.median(), 1 / d.W_min.quantile(.1), 1 / d.W_min.min()))
    print("  W_min = 창 안 최저깊이/트리거깊이,  W_bot = 바닥시각 깊이/트리거깊이")
    print("  W_pre = 트리거깊이/직전 60분 중앙  (트리거 시점이 이미 얇은가)")

    print("\n--- 1. 상한: 실현 W 를 알면 log X 가 얼마나 더 설명되나 ---")
    rows = [("사전 관측만", PRE),
            ("log(V/D_t) 60분", ["VD60"]),
            ("log(V/D_t) + 사전", PRE + ["VD60"]),
            ("log(V/D_t) + log W_min", ["VD60", "log_Wmin"]),
            ("log W_min 단독", ["log_Wmin"]),
            ("전부", PRE + ["VD60", "log_Wmin"])]
    print("  %-30s %9s %9s" % ("변수", "R2", "Spearman"))
    for lab, f in rows:
        r2, rho = oos(tr, te, f)
        print("  %-30s %+9.3f %+9.3f" % (lab, r2, rho))

    X1 = np.column_stack([np.ones(len(d)), d["VD60"], d["log_Wmin"]])
    beta, se = ols_cluster(X1, np.log(d["X"].to_numpy()), d["day"].to_numpy())
    print("  전체표본 계수 (일자클러스터):  VD60 %+.3f (SE %.3f)   log W_min %+.3f (SE %.3f)"
          % (beta[1], se[1], beta[2], se[2]))
    print("  이론 예측: log X = ... - gamma log W  이므로 log W_min 계수는 **음수**여야 한다.")

    print("\n--- 2. 예보력: 사전 관측치로 log W_min 을 맞힐 수 있나 ---")
    r2w, rhow = oos(tr, te, PRE, target="W_min")
    print("  사전 관측만 -> log W_min :  R2 = %+.3f   Spearman = %+.3f" % (r2w, rhow))
    r2w2, rhow2 = oos(tr, te, PRE + ["VD60"], target="W_min")
    print("  사전 + V/D_t             :  R2 = %+.3f   Spearman = %+.3f" % (r2w2, rhow2))
    print("  (참고: log X 에 대한 사전 관측 R2 는 +0.014 였다)")

    print("\n--- 2b. 이상치 진단: 1/W_min 상위 10건 ---")
    d["invW"] = 1.0 / d["W_min"].clip(lower=1e-9)
    print("  %-10s %-12s %10s %9s %9s %7s" % ("심볼", "날짜", "1/W_min", "X%", "VD60", "n_path"))
    for r in d.nlargest(10, "invW").itertuples():
        print("  %-10s %-12s %10.1f %8.2f%% %+9.2f %7d"
              % (r.symbol, str(r.dt.date()), r.invW, 100 * r.X, r.VD60, r.n_path))
    print("  -> 1/W 가 수백~수만이면 데이터 결함일 수 있다(호가 한 스냅샷이 거의 0).")
    print("     실제 붕괴라면 그 사건의 X 도 커야 한다.")

    print("\n--- 3. 결합: '예측 W' 가 X 예보를 개선하나 (재구성) ---")
    print("  *** 이전 판은 무효였다: Wpred 를 PRE 의 선형결합으로 만들어 놓고")
    print("      PRE 를 이미 포함한 회귀에 넣었다. 공선성이라 R2 가 변할 수 없다.")
    print("      아래는 PRE 를 빼고 비교하므로 공선성이 없다.")
    # 이상치가 OLS 를 망치므로 W 적합은 절단 표본에서 한다(예측은 전체에 적용).
    lw_tr = np.log(tr["W_min"].to_numpy())
    lo_c, hi_c = np.percentile(lw_tr, [1, 99])
    keep = (lw_tr >= lo_c) & (lw_tr <= hi_c)
    Xtr_p = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in PRE])
    Xte_p = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in PRE])
    ww = np.linalg.pinv(Xtr_p[keep].T @ Xtr_p[keep]) @ (Xtr_p[keep].T @ lw_tr[keep])
    tr2, te2 = tr.copy(), te.copy()
    tr2["Wpred"], te2["Wpred"] = Xtr_p @ ww, Xte_p @ ww
    print("  (W 적합은 1~99%% 절단 표본, 예측은 전체에 적용. 절단 %d/%d)"
          % (int((~keep).sum()), len(tr)))
    print("  %-34s %9s %9s" % ("변수", "R2", "Spearman"))
    for lab, f in (("V/D_t 단독", ["VD60"]),
                   ("V/D_t + 예측 W   <- 공선성 없음", ["VD60", "Wpred"]),
                   ("V/D_t + 실현 W (상한)", ["VD60", "log_Wmin"]),
                   ("사전 + V/D_t (기준)", PRE + ["VD60"])):
        r2, rho = oos(tr2, te2, f)
        print("  %-34s %+9.3f %+9.3f" % (lab, r2, rho))

    # 구조식 제약판: log X = a + gamma (VD60 - Wpred). gamma 하나만 추정.
    print("\n  구조식 제약판 — log X = a + gamma (VD60 - log W_pred), 모수 2개")
    ztr = tr2["VD60"].to_numpy() - tr2["Wpred"].to_numpy()
    zte = te2["VD60"].to_numpy() - te2["Wpred"].to_numpy()
    A = np.column_stack([np.ones(len(ztr)), ztr])
    wg = np.linalg.pinv(A.T @ A) @ (A.T @ np.log(tr2["X"].to_numpy()))
    p = np.column_stack([np.ones(len(zte)), zte]) @ wg
    yt = np.log(te2["X"].to_numpy())
    r2c = 1.0 - float(np.sum((yt - p) ** 2)) / float(np.sum((yt - yt.mean()) ** 2))
    rhoc = float(pd.Series(yt).corr(pd.Series(p), method="spearman"))
    print("  gamma = %+.3f,  절편 %+.3f  ->  R2 = %+.3f  Spearman = %+.3f"
          % (wg[1], wg[0], r2c, rhoc))
    print("  (모수 2개짜리가 자유 회귀를 이기면 구조 제약이 값어치를 하는 것이다.)")

    print("\n판정 순서: (1) 상한이 안 오르면 W 는 중요하지 않다 -> QW 폐기.")
    print("          (2) 상한은 오르는데 예보가 0 이면 -> 웹소켓 diff 로 W 를 직접 봐야 한다.")
    print("          (3) 둘 다 되면 -> 확률모델에 log W 항을 넣고 재검정.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
