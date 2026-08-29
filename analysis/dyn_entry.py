# -*- coding: utf-8 -*-
"""동적 지정가 — 매 분 다시 읽고 주문을 옮긴다. 설계 원문 그대로.

무엇이 틀렸었나 (사용자 지적, 2026-08-05)
  설계 원문: "가격이 한쪽으로 움직이면 그에 따라 오더북도 변한다.
              그 변화를 **매 순간 다시 읽어** ... **계속 재계산한다**."
  그런데 prob_entry / two_leg / horizon 은 전부 t0 에 q_alpha 를 한 번 계산해
  p_lim 을 박아두고 기다렸다. **정적 지정가**다. 그래서 residual_life.py 에서
  구조적 실패가 나왔다:
      정적 주문의 체결 조건은 곧 X >= q_alpha 다.
      -> 체결되는 순간은 언제나 '바닥이 내 주문보다 아래' 인 경우뿐.
      -> 게다가 m(u)=E[X-u|X>=u] 가 u 에 대해 **증가**한다(파레토 꼬리).
         깊이 걸수록 남은 밀림이 커진다. 어떤 alpha 를 골라도 안 바뀐다.
      실측: 체결 직후 MAE 중앙이 alpha=0.5 -174bp / 0.7 -212 / 0.9 -293.

동적 지정가가 그것을 어떻게 푸는가
  매 분 u 마다 **현재 저점 M_u 기준으로 '남은 밀림' R_u 를 다시 추정**한다.
      M_u = min(L[0..u])                     지금까지의 최저가
      R_u = (M_u - M_H) / p0 * 1e4  >= 0     앞으로 M_u 보다 얼마나 더 내려가나
      R_u = 0  이면 **지금 이 저점이 바닥**이다.
  주문가 = M_u * (1 - r_alpha(u))  로 매 분 옮긴다.
      r_alpha 가 0 에 가까우면 주문이 현재 저점까지 **올라온다** (= 지금이 바닥이라는 판단)
      아직 더 밀릴 것 같으면 주문이 아래에 머문다.
  즉 alpha 는 '얼마나 깊게' 가 아니라 '얼마나 확신할 때 붙일 것인가' 가 된다.

  R_u 에는 0 에 점질량이 있다(바닥을 이미 찍은 경우). 그래서 log(R_u + 1) 을 쓰고
  분위를 되돌릴 때 0 으로 바닥을 씌운다.

시각 u 에 알 수 있는 것만 특징으로 쓴다
  사건 고정 특징 (prob_entry 와 동일) + 경과 u + 이미 진행된 낙폭 D_u/q50
  + 최근 1/3분 수익 + 최근 신저점 갱신 여부 + 최근 실현변동성
  ** D_u/q50 과 '신저점 갱신' 이 소진 신호의 핵심이다 **
  주문은 분 u 종료 시점에 옮기므로 **분 u+1 부터** 체결될 수 있다 (봉내 룩어헤드 방지).

실행:
    python analysis/dyn_entry.py
    python analysis/dyn_entry.py --hold 5
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
from analysis.prob_entry import build, walk_forward, FEAT, HMAX       # noqa: E402
from analysis.response_liq import cmean                               # noqa: E402

WMAX = 30                      # 재호가를 계속하는 최대 분 (이후 포기)
GFEAT = FEAT + ["lu", "dprog", "r1", "r3m", "newlow", "rv5"]


def panel(dd, ww, scale, wmax=WMAX):
    """(사건 x 분) 패널. 각 행에서 '남은 밀림' R_u 를 목표로 잡는다.

    scale: 진행 낙폭 D_u 를 정규화할 값(bp). **X 모형의 q50 을 쓰면 안 된다** —
    그러면 워크포워드가 이중으로 걸려 표본이 반토막 난다. sigma(과거 288봉, 현재봉
    제외) 는 t0 에 이미 알려진 값이므로 누출 없이 전량을 쓸 수 있다.
    """
    rows = []
    base = dd[FEAT].to_numpy(dtype=np.float64)
    for i in range(len(dd)):
        if not np.isfinite(scale[i]) or scale[i] <= 0:
            continue
        sd = int(dd["side"].iat[i])
        O, H, L, Cl = ww[i]
        p0 = float(O[0])
        # 진행 방향으로 부호를 통일: sd=+1 이면 저가가 극값, sd=-1 이면 고가
        ext = (L[:HMAX + 1] if sd == 1 else H[:HMAX + 1])
        run = np.minimum.accumulate(ext) if sd == 1 else np.maximum.accumulate(ext)
        final = run[-1]
        clo = Cl[:HMAX + 1]
        lr = np.concatenate([[0.0], np.diff(np.log(np.maximum(clo, 1e-12)))])
        for u in range(0, min(wmax, HMAX) + 1):
            Mu = run[u]
            # 남은 밀림 (bp, >=0). 0 이면 지금 저점이 최종 바닥이다.
            R = abs(Mu - final) / p0 * 1e4
            Du = abs(p0 - Mu) / p0 * 1e4                     # 이미 진행된 낙폭
            r1 = lr[u] * sd * 1e4
            r3 = (lr[max(u - 2, 0):u + 1].sum()) * sd * 1e4
            nl = 1.0 if (u > 0 and run[u] != run[u - 1]) else 0.0   # 직전 분 신저점 갱신
            rv5 = float(np.std(lr[max(u - 4, 0):u + 1])) * 1e4
            rows.append({"ei": i, "u": u, "R": R, "Mu": Mu, "p0": p0, "sd": sd,
                         "t": dd["t"].iat[i], "day": dd["day"].iat[i],
                         "year": dd["year"].iat[i], "symbol": dd["symbol"].iat[i],
                         "lu": np.log1p(u), "dprog": Du / scale[i],
                         "r1": r1, "r3m": r3, "newlow": nl, "rv5": rv5,
                         **{c: base[i, k] for k, c in enumerate(FEAT)}})
    return pd.DataFrame(rows)


def wf_panel(P, alphas, min_ev=200, refit=50, purge_days=1, feats=None):
    """사건 단위 워크포워드. 한 사건의 모든 분 행은 같은 쪽(훈련/시험)에 간다."""
    n = len(P)
    y = np.log(P["R"].to_numpy() + 1.0)
    G = P[list(feats) if feats else GFEAT].to_numpy(dtype=np.float64)
    ev = P["ei"].to_numpy()
    day = P["day"].to_numpy()
    Q = {a: np.full(n, np.nan) for a in alphas}
    oos = np.zeros(n, dtype=bool)
    uev = np.unique(ev)
    beta = resq = None
    last = -10**9
    for k, e in enumerate(uev):
        if k < min_ev:
            continue
        sel = ev == e
        d0 = day[sel][0]
        if k - last >= refit or beta is None:
            tr = (ev < e) & (day < d0 - purge_days) & np.isfinite(y)
            tr &= np.isfinite(G).all(1)
            if len(np.unique(ev[tr])) < min_ev:
                continue
            X = np.column_stack([np.ones(int(tr.sum())), G[tr]])
            b = np.linalg.pinv(X.T @ X) @ (X.T @ y[tr])
            r = y[tr] - X @ b
            beta, resq = b, {a: float(np.quantile(r, a)) for a in alphas}
            last = k
        if beta is None:
            continue
        idx = np.flatnonzero(sel)
        ok = np.isfinite(G[idx]).all(1)
        idx = idx[ok]
        if not len(idx):
            continue
        m = np.column_stack([np.ones(len(idx)), G[idx]]) @ beta
        for a in alphas:
            Q[a][idx] = np.maximum(np.exp(m + resq[a]) - 1.0, 0.0)
        oos[idx] = True
    return Q, oos


def sim_dyn(P, ww, qcol, hold, wmax, delta_bp, fee_m, fee_t, static_q=None,
            min_prog=0.0):
    """매 분 주문을 옮긴다. static_q 를 주면 t0 값으로 고정(정적 대조군).

    min_prog: 이미 진행된 낙폭 D_u 가 sigma 의 이 배수를 넘기 전에는 주문을 걸지
    않는다. 동적 재호가는 **바닥을 잘 맞히지만 선별을 안 한다** — 정적 alpha=0.90
    의 우위가 사실은 '크게 오버슈트한 사건만 잡은' 선별 효과였다. 둘을 곱한다.
    """
    dl = delta_bp * 1e-4
    out = []
    for ei, g in P.groupby("ei", sort=True):
        g = g.sort_values("u")
        uu = g["u"].to_numpy()
        Mu = g["Mu"].to_numpy()
        qq = g[qcol].to_numpy()
        pg = g["dprog"].to_numpy()
        sd = int(g["sd"].iat[0])
        p0 = float(g["p0"].iat[0])
        O, H, L, Cl = ww[int(ei)]
        rec = {"symbol": g["symbol"].iat[0], "t": g["t"].iat[0], "day": g["day"].iat[0],
               "year": g["year"].iat[0], "filled": False, "ret": 0.0,
               "wait": np.nan, "q": np.nan, "mae": np.nan}
        for k in range(len(uu)):
            u = int(uu[k])
            if u >= wmax or u + 1 > HMAX:
                break
            q = static_q[int(ei)] if static_q is not None else qq[k]
            if not np.isfinite(q):
                continue
            if pg[k] < min_prog:          # 아직 충분히 안 밀렸다 -> 주문을 걸지 않는다
                continue
            # 정적은 p0 기준, 동적은 **현재 저점 Mu 기준**
            ref = p0 if static_q is not None else Mu[k]
            p_lim = ref * (1.0 - sd * q * 1e-4)
            v = u + 1                      # 분 u 종료 시 재호가 -> 분 u+1 부터 체결
            hit = (L[v] <= p_lim * (1.0 - dl)) if sd == 1 else (H[v] >= p_lim * (1.0 + dl))
            if hit:
                p_in = (min(O[v], p_lim) if sd == 1 else max(O[v], p_lim))
                ej = min(v + hold, len(Cl) - 1)
                seg = (L[v:ej + 1] if sd == 1 else H[v:ej + 1])
                rec["ret"] = (Cl[ej] / p_in - 1.0) * sd * 1e4 - (fee_m + fee_t)
                rec["mae"] = (((seg.min() / p_in - 1.0) if sd == 1
                               else (p_in / seg.max() - 1.0)) * 1e4)
                rec["filled"], rec["wait"], rec["q"] = True, v, q
                break
        out.append(rec)
    return pd.DataFrame(out)


def stat(x):
    if len(x) == 0:
        return {}
    r = x["ret"].to_numpy()
    m, _, t, _ = cmean(r, x["day"].to_numpy())
    yrs = (x["t"].max() - x["t"].min()) / (365.25 * 86_400_000)
    rt = r[r != 0.0]
    w, l = rt[rt > 0], rt[rt < 0]
    eq = np.cumsum(r)
    f = x[x["filled"]]
    return {"n": len(x), "fill": float(x["filled"].mean()), "bp": m, "t": t,
            "med": float(np.median(rt)) if len(rt) else np.nan,
            "win": float((rt > 0).mean()) if len(rt) else np.nan,
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "maxdd": float((eq - np.maximum.accumulate(eq)).min()),
            "worst": float(rt.min()) if len(rt) else np.nan,
            "mae": float(f["mae"].median()) if len(f) else np.nan,
            "wait": float(f["wait"].median()) if len(f) else np.nan}


def head():
    print("  %-24s | %5s %6s | %8s %5s %8s | %7s %6s | %6s %8s | %9s %5s"
          % ("설정", "n", "체결률", "시도당bp", "t", "중앙bp", "승률", "손익비",
             "샤프", "최악1건", "체결후MAE", "대기"))


def line(lab, s):
    if not s:
        return
    print("  %-24s | %5d %6.3f | %8.1f %5.1f %8.1f | %6.1f%% %6.2f | %6.2f %8.0f | %9.1f %5.0f"
          % (lab, s["n"], s["fill"], s["bp"], s["t"], s["med"], 100 * s["win"],
             s["pl"], s["sharpe"], s["worst"], s["mae"], s["wait"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="dynamic re-priced limit entry")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--wmax", type=int, default=WMAX)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    fm, ft = a.fee_maker, a.fee_taker

    print("=" * 122)
    print("동적 지정가 — 매 분 현재 저점 기준으로 '남은 밀림' 을 다시 추정하고 주문을 옮긴다")
    print("=" * 122)
    d, win = build(syms, a.k, a.doi, a.gap)
    if d is None or len(d) < 300:
        print("이벤트 부족")
        return 1
    # 정적 대조군용 X 모형 (사건 단위). 동적 모형과는 별도 워크포워드다.
    Qx, oosx = walk_forward(d, [0.5, 0.9], col="X")
    # 동적 패널은 **전량**에 대해 만든다. 정규화는 sigma(과거창) 로 — 누출 없음.
    scale = d["sig5"].to_numpy(dtype=np.float64) * 1e4      # bp 단위 sigma
    P = panel(d, win, scale, a.wmax)
    alphas = [0.10, 0.30, 0.50, 0.70, 0.90]
    Qr, oosr = wf_panel(P, alphas, min_ev=min(200, max(50, len(d) // 4)))
    for al in alphas:
        P["q%.2f" % al] = Qr[al]
    P = P[oosr].reset_index(drop=True)
    # **두 모형이 모두 OOS 인 사건만** 비교한다 (같은 표본이어야 대조가 성립)
    ev_ok = np.flatnonzero(oosx)
    P = P[P["ei"].isin(set(ev_ok.tolist()))].reset_index(drop=True)
    ww = win
    q50 = np.where(oosx, Qx[0.5], np.nan)
    q90 = np.where(oosx, Qx[0.9], np.nan)
    nev = P["ei"].nunique()
    print("**사용 데이터 기간: %s ~ %s / %d종 / 전체 %d건 / 비교 대상 OOS %d건**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d), nev))
    print("보유 %d분 / 재호가 최대 %d분 / 패널 OOS %d행" % (a.hold, a.wmax, len(P)))
    if nev < 100 or len(P) < 1000:
        print("표본 부족 — 종료")
        return 1

    print("\n" + "-" * 122)
    print("0. '남은 밀림' 모형이 맞는가 — OOS 위반율이 alpha 와 같아야 한다")
    print("-" * 122)
    R = P["R"].to_numpy()
    print("  실제 R 분포(bp): " +
          " ".join("p%02d %.0f" % (p, np.percentile(R, p)) for p in (10, 25, 50, 75, 90)))
    print("  R = 0 (지금이 바닥) 인 행 비율 %.3f\n" % float((R <= 0.5).mean()))
    print("  %-8s %14s %12s %10s" % ("alpha", "예측 r 중앙bp", "실제위반율", "편차"))
    for al in alphas:
        q = P["q%.2f" % al].to_numpy()
        v = float((R < q).mean())
        print("  %-8.2f %14.0f %12.3f %10.3f" % (al, np.median(q), v, v - al))

    print("\n" + "-" * 122)
    print("1. 동적 vs 정적 — 같은 사건, 같은 보유(%d분), 같은 침투(%dbp)" % (a.hold, a.delta))
    print("-" * 122)
    print("  ** 체결후MAE 가 0 에 가까울수록 실제로 바닥에서 잡은 것이다. **\n")
    head()
    line("정적 a=0.90 (기존)",
         stat(sim_dyn(P, ww, "q0.50", a.hold, a.wmax, a.delta, fm, ft, static_q=q90)))
    line("정적 a=0.50 (기존)",
         stat(sim_dyn(P, ww, "q0.50", a.hold, a.wmax, a.delta, fm, ft, static_q=q50)))
    for al in alphas:
        line("**동적 a=%.2f**" % al,
             stat(sim_dyn(P, ww, "q%.2f" % al, a.hold, a.wmax, a.delta, fm, ft)))

    print("\n" + "-" * 122)
    print("2. 보유 시간 감도 (동적)")
    print("-" * 122)
    head()
    for hd in (2, 3, 5, 10, 15, 30):
        for al in (0.30, 0.50):
            line("동적 a=%.2f 보유%d분" % (al, hd),
                 stat(sim_dyn(P, ww, "q%.2f" % al, hd, a.wmax, a.delta, fm, ft)))

    print("\n" + "-" * 122)
    print("3. 안정성 — OOS 전/후반 (동적 최선 vs 정적 a=0.90)")
    print("-" * 122)
    head()
    cands = [("동적 a=0.30 prog>=%.0f" % mp,
              sim_dyn(P, ww, "q0.30", a.hold, a.wmax, a.delta, fm, ft, min_prog=mp))
             for mp in (0.0, 4.0, 8.0)]
    cands.append(("정적 a=0.90", sim_dyn(P, ww, "q0.50", a.hold, a.wmax, a.delta,
                                       fm, ft, static_q=q90)))
    for lab, x in cands:
        h = len(x) // 2
        line(lab + " 전체", stat(x))
        line("  전반부", stat(x.iloc[:h]))
        line("  후반부", stat(x.iloc[h:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
