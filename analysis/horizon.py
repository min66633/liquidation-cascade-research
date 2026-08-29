# -*- coding: utf-8 -*-
"""시간 지평을 데이터가 정하게 한다 — 60분은 내가 임의로 박은 값이었다.

무엇이 문제였나
  prob_entry.py / two_leg.py 에는 **60이 세 군데** 박혀 있었고 전부 내가 정했다.
      HMAX=60   모델 목표 X 를 재는 지평 (t0 후 몇 분 안의 밀림을 볼 것인가)
      W=60      지정가를 살려두는 시간
      EMAX=60   체결 후 보유 시간
  바닥은 몇 초 만에 오기도 하고 수십 분 걸리기도 한다(도달시각 p25=0분 p50=5 p90=46).
  회복 속도도 마찬가지다. 하나로 고정하면 그 분산이 전부 손실로 간다.

이 파일이 하는 것
  1) W x emax 격자를 통째로 훑는다 (2/3/5/10/15/30/60분). 60분이 최적인지 **본다**.
  2) **시간도 확률모형으로 추정한다.** 밀림 폭 X 와 같은 특징·같은 워크포워드로
         log T = b' f + eps,   tau_beta = exp(b'f + Q_beta(eps))
     T = 바닥 도달까지 걸린 분. 그리고 W, emax 를 tau_beta 에서 **사건마다** 정한다.
  3) 고정 격자 최고 vs 상태의존을 나란히 비교한다.

  진입 규칙은 지금까지 가장 나은 것으로 고정한다: 조건부 지정매수 alpha=0.90,
  손절 없음(two_leg.py 에서 손절은 어떤 폭이든 수익을 죽였다), 침투 2bp.

*** 목표 X 의 지평(HMAX)은 여기서 못 바꾼다 ***
  build() 가 창을 HMAX+EMAX=120분으로 잘라 저장하므로 X 는 60분 기준이다.
  대신 **W 와 emax 를 X 의 지평과 독립으로** 훑어, 60분 지평에서 만든 예측이
  짧은 운용 시간에도 쓸모가 있는지를 본다. HMAX 자체의 감도는 별도 과제로 남긴다.

실행:
    python analysis/horizon.py
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
from analysis.prob_entry import build, walk_forward, HMAX             # noqa: E402
from analysis.response_liq import cmean                               # noqa: E402

GRID = [2, 3, 5, 10, 15, 30, 60]


def sim(dd, ww, qal, W, emax, delta_bp, fee_m, fee_t):
    """조건부 지정매수 -> emax 분 뒤 종가. 손절 없음.

    W, emax 는 스칼라 또는 사건별 배열. 배열이면 상태의존이다.
    """
    n = len(dd)
    Wv = np.full(n, W) if np.isscalar(W) else np.asarray(W)
    Ev = np.full(n, emax) if np.isscalar(emax) else np.asarray(emax)
    dl = delta_bp * 1e-4
    out = []
    for i in range(n):
        if not np.isfinite(qal[i]):
            continue
        sd = int(dd["side"].iat[i])
        O, H, L, Cl = ww[i]
        p0 = float(O[0])
        w = int(np.clip(Wv[i], 1, HMAX))
        e = int(np.clip(Ev[i], 1, len(Cl) - 1))
        r = {"symbol": dd["symbol"].iat[i], "t": dd["t"].iat[i], "day": dd["day"].iat[i],
             "year": dd["year"].iat[i], "filled": False, "ret": 0.0,
             "wait": np.nan, "hold": np.nan}
        p_lim = p0 * (1.0 - sd * qal[i] * 1e-4)
        if sd == 1:
            hit = np.flatnonzero(L[:w + 1] <= p_lim * (1.0 - dl))
        else:
            hit = np.flatnonzero(H[:w + 1] >= p_lim * (1.0 + dl))
        if len(hit):
            fj = int(hit[0])
            p_in = (min(O[fj], p_lim) if sd == 1 else max(O[fj], p_lim)) if fj > 0 else p_lim
            ej = min(fj + e, len(Cl) - 1)
            r["ret"] = (Cl[ej] / p_in - 1.0) * sd * 1e4 - (fee_m + fee_t)
            r["filled"], r["wait"], r["hold"] = True, fj, ej - fj
        out.append(r)
    return pd.DataFrame(out)


def stat(x: pd.DataFrame) -> dict:
    if len(x) == 0:
        return {}
    r = x["ret"].to_numpy()
    m, _, t, _ = cmean(r, x["day"].to_numpy())
    yrs = (x["t"].max() - x["t"].min()) / (365.25 * 86_400_000)
    rt = r[r != 0.0]
    w, l = rt[rt > 0], rt[rt < 0]
    eq = np.cumsum(r)
    return {"n": len(x), "fill": float(x["filled"].mean()), "bp": m, "t": t,
            "med": float(np.median(rt)) if len(rt) else np.nan,
            "win": float((rt > 0).mean()) if len(rt) else np.nan,
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "maxdd": float((eq - np.maximum.accumulate(eq)).min()),
            "worst": float(rt.min()) if len(rt) else np.nan,
            "hold": float(x["hold"].median()) if x["filled"].any() else np.nan,
            "wait": float(x["wait"].median()) if x["filled"].any() else np.nan}


def head():
    print("  %-26s | %5s %6s | %8s %5s %8s | %7s %6s | %6s %8s %8s %5s"
          % ("설정", "n", "체결률", "시도당bp", "t", "중앙bp", "승률", "손익비",
             "샤프", "최대낙폭", "최악1건", "보유"))


def line(lab, s):
    if not s:
        return
    print("  %-26s | %5d %6.3f | %8.1f %5.1f %8.1f | %6.1f%% %6.2f | %6.2f %8.0f %8.0f %5.0f"
          % (lab, s["n"], s["fill"], s["bp"], s["t"], s["med"], 100 * s["win"],
             s["pl"], s["sharpe"], s["maxdd"], s["worst"], s["hold"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="let data pick the time horizons")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    fm, ft = a.fee_maker, a.fee_taker

    print("=" * 122)
    print("시간 지평을 데이터가 정한다 — 60분은 임의값이었다")
    print("=" * 122)
    d, win = build(syms, a.k, a.doi, a.gap)
    if d is None or len(d) < 300:
        print("이벤트 부족")
        return 1

    # 바닥 도달 시간 T (분). 목표 X 와 같은 창에서 잰다.
    T = np.empty(len(d))
    for i in range(len(d)):
        sd = int(d["side"].iat[i])
        L, H = win[i][2], win[i][1]
        seg = L[:HMAX + 1] if sd == 1 else H[:HMAX + 1]
        T[i] = float(np.argmin(seg) if sd == 1 else np.argmax(seg))
    d = d.copy()
    d["T"] = np.maximum(T, 0.5)          # 0분을 로그에 넣으려면 바닥이 필요하다

    alphas = [a.alpha, 0.5]
    betas = [0.30, 0.50, 0.70, 0.90]
    Qx, oos = walk_forward(d, alphas, col="X")
    Qt, oos_t = walk_forward(d, betas, col="T")
    ok = oos & oos_t
    dd = d[ok].reset_index(drop=True)
    ww = win[ok]
    qal = Qx[a.alpha][ok]
    print("**사용 데이터 기간: %s ~ %s / %d종 / 전체 %d건 / OOS %d건 / alpha=%.2f**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d), len(dd), a.alpha))

    print("\n" + "-" * 122)
    print("0. 시간 모형이 맞는가 — OOS 위반율이 beta 와 같아야 한다")
    print("-" * 122)
    To = dd["T"].to_numpy()
    print("  %-8s %14s %12s %10s" % ("beta", "예측 T 중앙(분)", "실제위반율", "편차"))
    for b in betas:
        q = Qt[b][ok]
        v = float((To < q).mean())
        print("  %-8.2f %14.1f %12.3f %10.3f" % (b, np.median(q), v, v - b))
    print("  실제 T 분포(분): " +
          " ".join("p%02d %.0f" % (p, np.percentile(To, p)) for p in (10, 25, 50, 75, 90)))

    print("\n" + "-" * 122)
    print("1. W(지정가 유효) x emax(보유) 격자 — 60분이 최적인가")
    print("-" * 122)
    print("  각 칸: 시도당bp / 샤프.  W 가 짧으면 체결률이 떨어지고, emax 가 짧으면 회복을 놓친다.\n")
    print("  %-6s | %s" % ("W\\emax", " | ".join("%-13d" % e for e in GRID)))
    best = None
    fills = {}
    for W in GRID:
        cells = []
        for e in GRID:
            s = stat(sim(dd, ww, qal, W, e, a.delta, fm, ft))
            fills[W] = s["fill"]
            cells.append("%6.1f %6.2f" % (s["bp"], s["sharpe"]))
            if best is None or s["bp"] > best[0]["bp"]:
                best = (s, W, e)
        print("  %-6d | %s" % (W, " | ".join(cells)))
    print("\n  W 별 체결률: " + " ".join("W%02d %.2f" % (W, fills[W]) for W in GRID))
    print("  최고 시도당bp: W=%d / emax=%d" % (best[1], best[2]))

    print("\n" + "-" * 122)
    print("2. 상태의존 — W 와 emax 를 사건마다 예측 T 에서 정한다")
    print("-" * 122)
    print("  '이 사건은 바닥까지 오래 걸릴 것' 이면 오래 기다리고, 빠를 것이면 짧게.\n")
    head()
    line("고정 W=%d emax=%d (격자최고)" % (best[1], best[2]), best[0])
    line("고정 W=60 emax=60 (기존)",
         stat(sim(dd, ww, qal, 60, 60, a.delta, fm, ft)))
    for b in betas:
        tau = Qt[b][ok]
        for mult, lab in ((1.0, "emax=tau"), (2.0, "emax=2tau"), (3.0, "emax=3tau")):
            s = stat(sim(dd, ww, qal, np.ceil(tau), np.ceil(mult * tau),
                         a.delta, fm, ft))
            line("상태의존 b=%.2f %s" % (b, lab), s)

    print("\n" + "-" * 122)
    print("3. 안정성 — OOS 전/후반 (격자최고 vs 기존 60/60)")
    print("-" * 122)
    head()
    for lab, W, e in (("격자최고 W%d/e%d" % (best[1], best[2]), best[1], best[2]),
                      ("기존 W60/e60", 60, 60)):
        x = sim(dd, ww, qal, W, e, a.delta, fm, ft)
        h = len(x) // 2
        line(lab + " 전체", stat(x))
        line("  전반부", stat(x.iloc[:h]))
        line("  후반부", stat(x.iloc[h:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
