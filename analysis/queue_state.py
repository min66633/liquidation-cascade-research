# -*- coding: utf-8 -*-
"""큐-반응 + 상태의존 지수 + 예측→수익 전환. 웹소켓 1초 데이터.

세 가지를 한다 (사용자 지적 순서)

A. **큐-반응 (호가를 넣고 빼는 것)** — 네 부품 중 유일하게 안 쓴 것.
   depth_ws_flow 가 밴드별 **1초 순변화**를 이미 쌓고 있다. 이것이 Delta(v,t) 다.
   핵심 질문: 청산이 들어올 때 호가가 **소진되는 것 이상으로 빠지는가**
   (= 유령 유동성 / 약탈적 취소). 그렇다면 유효깊이 D_eff < D 이고,
   그 차이가 '왜 작은 물량이 크게 미는가' 를 설명한다.

B. **상태의존 지수** — ws_depth_test.py 가 b2 = 0.049 (t=3.2) 를 냈다.
   제곱근 법칙은 0.5 다. 그런데 그 표본은 Q/D 중앙이 6e-05 로 **법칙이 작동하는
   구간에 도달하지 못했다.** 지수가 상수가 아니라 **Q/D 수준에 따라 커진다면**
   0.049 는 저구간 값일 뿐이고 설계는 살아 있다. 구간별로 직접 잰다.
   (이것이 프로젝트가 오래 가설로 둔 임계형 max(0, V/D - c)^beta 의 검정이다.)

C. **예측 -> 수익 전환** — 청산 직전/근처에서 밀림 거리를 예측해 지정가로 바꾼다.
   예측이 실현을 **순서대로 맞히는가**(순위상관), 그리고 그 예측 깊이에 건 지정가가
   고정 깊이를 이기는가.

*** 표본 한계: 겹침 15시간, 사건 486개. 부호·순위는 검정되나 규모 외삽은 불가. ***

실행:
    python analysis/queue_state.py
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
from analysis.response_liq import ols_cluster, cmean            # noqa: E402
from analysis.ws_depth_test import load_depth, load_liq         # noqa: E402

HOR = 15               # 밀림 거리 지평(초)
FEE = 7.0              # 왕복 bp (메이커 진입 + 테이커 청산)


def load_flow():
    fs = sorted(glob.glob(os.path.join(C.DATA, "depth_ws_flow", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)


def build(band: str):
    dp, fl, lq = load_depth(), load_flow(), load_liq()
    t0 = max(dp["ts_ms"].min(), lq["exch_ms"].min())
    t1 = min(dp["ts_ms"].max(), lq["exch_ms"].max())
    lq = lq[(lq["exch_ms"] >= t0) & (lq["exch_ms"] <= t1)]
    rows = []
    for s, g in lq.groupby("symbol"):
        D = dp[dp.symbol == s]
        F = fl[fl.symbol == s]
        if len(D) < 200 or len(F) < 200:
            continue
        ts = D["ts_ms"].to_numpy()
        mid = D["mid"].to_numpy(dtype=np.float64)
        bid = D["bid_" + band].to_numpy(dtype=np.float64)
        ask = D["ask_" + band].to_numpy(dtype=np.float64)
        fts = F["ts_ms"].to_numpy()
        dbid = F["dbid_" + band].to_numpy(dtype=np.float64)
        dask = F["dask_" + band].to_numpy(dtype=np.float64)
        lm = np.log(np.maximum(mid, 1e-12))
        sig = pd.Series(np.concatenate([[np.nan], np.diff(lm)])).rolling(
            60, min_periods=20).std().to_numpy()
        g = g.copy()
        g["sec"] = g["exch_ms"] // 1000
        agg = g.groupby("sec").apply(
            lambda x: pd.Series({
                "Q": float(x["ntl"].sum()),
                "down": 1 if (x["ntl"] * (x["down"] == 1)).sum()
                >= (x["ntl"] * (x["down"] == -1)).sum() else -1}),
            include_groups=False).reset_index()
        for r in agg.itertuples():
            tq = int(r.sec) * 1000
            i = int(np.searchsorted(ts, tq)) - 1
            if i < 61 or i >= len(ts) - HOR - 2:
                continue
            if not (np.isfinite(sig[i]) and sig[i] > 0 and mid[i] > 0):
                continue
            dn = int(r.down)
            Dpre = bid[i] if dn == 1 else ask[i]
            if not (np.isfinite(Dpre) and Dpre > 0):
                continue
            j = int(np.searchsorted(ts, tq + HOR * 1000))
            if j >= len(ts):
                continue
            Dpost = (bid[j] if dn == 1 else ask[j])
            # 사건 구간의 호가 순변화(넣고 뺀 것의 합) — 큐-반응 항
            k0 = int(np.searchsorted(fts, tq))
            k1 = int(np.searchsorted(fts, tq + HOR * 1000))
            fseg = (dbid[k0:k1] if dn == 1 else dask[k0:k1])
            add = float(np.nansum(fseg[fseg > 0]))     # 넣은 것
            rem = float(-np.nansum(fseg[fseg < 0]))    # 뺀 것 (체결+취소)
            seg = mid[i:j + 1]
            x = ((mid[i] - seg.min()) / mid[i] if dn == 1
                 else (seg.max() - mid[i]) / mid[i]) * 1e4
            # 사전 상태 (룩어헤드 없음): 직전 60초 호가 불균형·유입
            imb = (bid[i] - ask[i]) / max(bid[i] + ask[i], 1e-9)
            pre = float(np.nansum(dbid[max(k0 - 60, 0):k0] if dn == 1
                                  else dask[max(k0 - 60, 0):k0]))
            rows.append({"symbol": s, "hour": tq // 3_600_000, "tq": tq,
                         "Q": r.Q, "D": Dpre, "Dpost": Dpost, "sig": sig[i],
                         "add": add, "rem": rem, "imb": imb, "pre": pre,
                         "down": dn, "X": x, "mid": mid[i],
                         "fwd": ((seg[-1] / seg[0] - 1.0) * (-dn) * 1e4)})
    d = pd.DataFrame(rows)
    return d[np.isfinite(d["X"]) & (d["Q"] > 0) & (d["D"] > 0)].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="queue-reactive + state-dependent exponent")
    ap.add_argument("--band", default="b0_5")
    a = ap.parse_args()
    U.init_stdout()
    print("=" * 78)
    print("큐-반응 + 상태의존 지수 + 예측→수익  (웹소켓 1초, 밴드 %s)" % a.band)
    print("=" * 78)
    d = build(a.band)
    if len(d) < 100:
        print("표본 부족 (%d)" % len(d))
        return 1
    print("**사용 데이터 기간: %s ~ %s | 사건 %d개 / %d종**"
          % (str(pd.to_datetime(d.tq.min(), unit="ms"))[:19],
             str(pd.to_datetime(d.tq.max(), unit="ms"))[:19],
             len(d), d.symbol.nunique()))
    qd = (d["Q"] / d["D"]).to_numpy()
    print("Q/D 중앙 %.3g | p90 %.3g | p99 %.3g | 최대 %.3g"
          % (np.median(qd), np.quantile(qd, .9), np.quantile(qd, .99), qd.max()))

    print("\n" + "=" * 78)
    print("A. 큐-반응 — 청산이 오면 호가는 소진되는 것 **이상으로** 빠지는가")
    print("=" * 78)
    print("  뺀 것(rem) / 청산물량(Q) 이 1보다 크게 크면 체결 소진을 넘는 **취소**다.")
    d["ratio"] = d["rem"] / d["Q"]
    d["net"] = (d["add"] - d["rem"]) / d["D"]
    print("  rem/Q : 중앙 %.1f | p25 %.1f | p75 %.1f" %
          (d.ratio.median(), d.ratio.quantile(.25), d.ratio.quantile(.75)))
    print("  깊이 순변화 (add-rem)/D : 중앙 %+.4f | p25 %+.4f | p75 %+.4f"
          % (d.net.median(), d.net.quantile(.25), d.net.quantile(.75)))
    print("  15초 뒤 깊이/직전 깊이 : 중앙 %.3f" % (d.Dpost / d.D).median())
    print("\n  [A1] 물량이 클수록 호가가 더 빠지는가  (net = a + b log(Q/D))")
    m = np.isfinite(d["net"]) & (qd > 0)
    X = np.column_stack([np.ones(int(m.sum())), np.log(qd[m])])
    b, se, _ = ols_cluster(X, d["net"].to_numpy()[m], d["hour"].to_numpy()[m])
    print("      b = %+.4f (t=%.1f)  음수면 큰 청산일수록 호가가 **더 빠진다**"
          % (b[1], b[1] / se[1] if se[1] > 0 else np.nan))
    print("\n  [A2] 유효깊이로 바꾸면 지수가 달라지는가")
    print("      D_eff = D + (add - rem)  — 사건 중 실제로 남은 깊이")
    deff = np.maximum(d["D"].to_numpy() + d["add"].to_numpy() - d["rem"].to_numpy(), 1.0)
    y = np.log(np.maximum(d["X"].to_numpy(), 1e-6))
    ok = np.isfinite(y) & (d["X"].to_numpy() > 0)
    for lab, den in (("D (표준 깊이)", d["D"].to_numpy()),
                     ("D_eff (큐-반응)", deff)):
        Xm = np.column_stack([np.ones(int(ok.sum())), np.log(d["sig"].to_numpy()[ok]),
                              np.log(d["Q"].to_numpy()[ok] / den[ok])])
        bb, ss, _ = ols_cluster(Xm, y[ok], d["hour"].to_numpy()[ok])
        print("      %-16s b2 = %+.4f (t=%.1f)"
              % (lab, bb[2], bb[2] / ss[2] if ss[2] > 0 else np.nan))

    print("\n" + "=" * 78)
    print("B. 상태의존 지수 — b2 가 Q/D 수준에 따라 커지는가")
    print("=" * 78)
    print("  제곱근 법칙 0.5 / 전표본 0.049. 저구간 값일 뿐인지 본다.")
    d["bin"] = pd.qcut(qd, 4, labels=False, duplicates="drop")
    print("  %5s %7s %12s %12s | %9s %7s"
          % ("사분위", "n", "Q/D 중앙", "X 중앙 bp", "b2", "t"))
    for q in sorted(pd.unique(d["bin"].dropna())):
        g = d[d["bin"] == q]
        yy = np.log(np.maximum(g["X"].to_numpy(), 1e-6))
        mm = np.isfinite(yy) & (g["X"].to_numpy() > 0)
        if mm.sum() < 40:
            continue
        Xm = np.column_stack([np.ones(int(mm.sum())),
                              np.log(g["sig"].to_numpy()[mm]),
                              np.log((g["Q"] / g["D"]).to_numpy()[mm])])
        bb, ss, _ = ols_cluster(Xm, yy[mm], g["hour"].to_numpy()[mm])
        print("  %5d %7d %12.3g %12.1f | %9.3f %7.1f"
              % (q, len(g), (g["Q"] / g["D"]).median(), g["X"].median(),
                 bb[2], bb[2] / ss[2] if ss[2] > 0 else np.nan))
    lq_ = np.log(qd)
    print("\n  [B1b] **심볼 고정효과** — 사분위 안에서 기울기가 죽는 이유 진단")
    print("       전표본 b2 가 양수인데 사분위 안에서 0/음수면, 기울기가 구간 **사이**")
    print("       에서만 나온다는 뜻이다. 심볼 구성 같은 횡단면 요인일 수 있다.")
    sy = pd.get_dummies(d["symbol"]).to_numpy(dtype=np.float64)[:, 1:]
    for lab, Xm in (
            ("고정효과 없음",
             np.column_stack([np.ones(int(ok.sum())), np.log(d["sig"].to_numpy()[ok]),
                              lq_[ok]])),
            ("**심볼 고정효과**",
             np.column_stack([np.ones(int(ok.sum())), np.log(d["sig"].to_numpy()[ok]),
                              lq_[ok], sy[ok]]))):
        bb, ss, _ = ols_cluster(Xm, y[ok], d["hour"].to_numpy()[ok])
        print("       %-16s b2 = %+.4f (t=%.1f)"
              % (lab, bb[2], bb[2] / ss[2] if ss[2] > 0 else np.nan))
    print("       FE 를 넣고 b2 가 죽으면 앞선 b2=0.049 는 **횡단면 인공물**이다.")

    print("\n  [B2] 상호작용항으로 직접 검정: log X ~ log sig + log(Q/D) + log(Q/D)^2")
    Xm = np.column_stack([np.ones(int(ok.sum())), np.log(d["sig"].to_numpy()[ok]),
                          lq_[ok], lq_[ok] ** 2])
    bb, ss, _ = ols_cluster(Xm, y[ok], d["hour"].to_numpy()[ok])
    print("      1차 %+.4f (t=%.1f) | **2차 %+.5f (t=%.1f)**"
          % (bb[2], bb[2] / ss[2], bb[3], bb[3] / ss[3] if ss[3] > 0 else np.nan))
    print("      2차가 유의한 양수면 **지수가 Q/D 와 함께 커진다** = 임계형 지지.")

    print("\n" + "=" * 78)
    print("C. 예측 → 수익 전환")
    print("=" * 78)
    print("  예측 Xhat = exp(a + b1 log sig + b2 log(Q/D)). 전반부 적합 → 후반부 검정.")
    cut = int(len(d) * 0.6)
    d = d.sort_values("tq").reset_index(drop=True)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    ytr = np.log(np.maximum(tr["X"].to_numpy(), 1e-6))
    Xtr = np.column_stack([np.ones(len(tr)), np.log(tr["sig"]),
                           np.log(tr["Q"] / tr["D"])])
    bt = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ ytr)
    Xte = np.column_stack([np.ones(len(te)), np.log(te["sig"]),
                           np.log(te["Q"] / te["D"])])
    xhat = np.exp(Xte @ bt)
    xr = te["X"].to_numpy()
    rho = float(pd.Series(xhat).corr(pd.Series(xr), method="spearman"))
    print("  훈련 %d / 검정 %d | **예측-실현 순위상관(Spearman) = %.3f**"
          % (len(tr), len(te), rho))
    print("  사분위별 실현 X 중앙(bp) — 단조 증가해야 예측이 값어치가 있다:")
    tb = pd.qcut(pd.Series(xhat), 4, labels=False, duplicates="drop")
    print("    " + " | ".join("Q%d %.1f" % (q, np.median(xr[tb == q]))
                              for q in sorted(pd.unique(tb.dropna()))))
    print("\n  지정가를 예측 깊이에 걸었을 때 (진입 후 %d초 청산, 왕복 %.0fbp)"
          % (HOR, FEE))
    print("  %-20s %8s %10s %8s" % ("방식", "체결률", "이벤트당bp", "t"))
    dn = te["down"].to_numpy()
    fw = te["fwd"].to_numpy()          # 사건 방향 반대(= 되돌림) 부호로 이미 정렬됨
    for lab, dep in (("예측 Xhat", xhat), ("예측 0.5*Xhat", 0.5 * xhat),
                     ("고정 0bp", np.zeros(len(te))),
                     ("고정 5bp", np.full(len(te), 5.0)),
                     ("고정 20bp", np.full(len(te), 20.0))):
        filled = te["X"].to_numpy() >= dep
        r = np.where(filled, fw + dep - FEE, 0.0)
        m, se, t, _ = cmean(r, te["hour"].to_numpy())
        print("  %-20s %7.1f%% %10.1f %8.1f" % (lab, 100 * filled.mean(), m, t))
    print("\n  체결가가 dep 만큼 유리하므로 수익에 +dep 를 더한다(체결 시).")
    print("  *** 15시간 표본이다. 부호·순위는 읽되 크기는 읽지 말 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
