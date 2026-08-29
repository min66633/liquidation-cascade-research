# -*- coding: utf-8 -*-
"""사다리의 결합분포와 위험조정수익 — EV 평평은 1차 모멘트만의 진술이다.

MODEL.md 가 빠뜨린 두 가지를 채운다.

(1) §5.2  위험조정수익도 평평한가
    체결을 $B\\sim Bern(S)$, 조건부 수익을 평균 m 표준편차 s 로 두면
        Var(Br) = S s^2 + S(1-S) m^2
        Sharpe(u) = sqrt(S) m / sqrt(s^2 + (1-S) m^2)
    s ∝ m 이면 Sharpe ∝ sqrt(S) 로 얕은 쪽이 유리하지만, s 가 상수면 m<<s 구간에서
    부호가 뒤집힌다. **가정할 것이 아니라 s(u) 를 직접 잰다.**

(2) §5.3  사다리는 분산되지 않는다
    S(u) 는 레벨별 독립 확률이 아니다. X = 지평 내 최대 낙폭이라 하면 체결집합은
    항상 [0,X] 이고 S(u) = P(X >= u) — 즉 S 는 **X 의 생존함수 하나**이며 체결은
    완전 중첩(comonotone)이다. 레벨을 넓게 깔아도 분산되지 않고 X 한 변수에 대한
    노출만 커진다. 실현손익은 X 에 2차이고, 되돌림이 없으면 숏풋이 된다.
    레벨별 EV 의 합은 포트폴리오 손익이 아니므로 여기서 실제 손익함수를 만든다.

데이터: analysis/stopping_hazard.py 가 저장한 hazard_panel.parquet 재사용.
        (Binance USD-M 5종, k=8sigma dOI<=-2% 롱청산 295건 x 레벨 40개)

실행:
    python analysis/ladder_risk.py
    python analysis/ladder_risk.py --horizon 60 --cost-bps 7
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

PANEL = os.path.join(C.DATA, "analysis", "hazard_panel.parquet")


def load_panel(horizon: int, cost_bps: float) -> pd.DataFrame:
    if not os.path.exists(PANEL):
        raise FileNotFoundError("missing %s (run analysis/stopping_hazard.py)" % PANEL)
    d = pd.read_parquet(PANEL)
    col = "r%d" % horizon
    if col not in d.columns:
        raise KeyError("panel has no %s (have %s)"
                       % (col, [c for c in d.columns if c.startswith("r")]))
    d = d.copy()
    # 패널의 r 은 비용 전이다. 체결된 건에만 왕복비용을 뺀다.
    d["r"] = np.where(d["reached"], d[col] - cost_bps / 1e4, np.nan)
    d["u_abs"] = d["u"].abs()
    # u_min 은 지평 내 최저 상대가격이다. 곧바로 반등해 한 번도 기준가 아래로
    # 안 갔으면 양수가 되는데, abs() 를 쓰면 없는 낙폭을 날조한다.
    d["X"] = (-d["u_min"]).clip(lower=0.0)
    d["ev_id"] = d["symbol"].astype(str) + "#" + d["open_time"].astype(str)
    return d


def moments(d: pd.DataFrame) -> pd.DataFrame:
    """레벨별 S, m, s 와 그로부터 유도한 시도당 EV/분산/Sharpe."""
    rows = []
    for u, g in d.groupby("u_abs"):
        n_ev = g["ev_id"].nunique()
        f = g[g["reached"] & np.isfinite(g["r"])]
        S = len(f) / max(n_ev, 1)
        if len(f) < 5:
            continue
        m = float(f["r"].mean())
        s = float(f["r"].std(ddof=1))
        ev = S * m
        var = S * s ** 2 + S * (1.0 - S) * m ** 2
        rows.append({"u%": 100 * u, "n_ev": n_ev, "S": S,
                     "m_bp": 1e4 * m, "s_bp": 1e4 * s,
                     "s/m": s / m if m > 0 else np.nan,
                     "EV_bp": 1e4 * ev,
                     "sd_bp": 1e4 * np.sqrt(var),
                     "Sharpe": ev / np.sqrt(var) if var > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("u%").reset_index(drop=True)


def ladder_pnl(d: pd.DataFrame, levels: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """이벤트별 사다리 손익. weights 는 합 1 (배정자본 기준)."""
    w = pd.Series(weights, index=np.round(levels, 6))
    sub = d[d["u_abs"].round(6).isin(w.index)].copy()
    if sub.empty:                    # 격자 밖 레벨을 요청한 경우 (--single 1.3 등)
        return pd.DataFrame(columns=["X", "w_filled", "pnl_alloc",
                                     "ret_alloc_bp", "ret_used_bp"])
    sub["w"] = sub["u_abs"].round(6).map(w)
    sub["filled"] = sub["reached"] & np.isfinite(sub["r"])

    g = sub.groupby("ev_id")
    out = pd.DataFrame({
        "X": g["X"].first(),
        "w_filled": g.apply(lambda t: float(t.loc[t["filled"], "w"].sum()),
                            include_groups=False),
        "pnl_alloc": g.apply(lambda t: float((t.loc[t["filled"], "w"]
                                              * t.loc[t["filled"], "r"]).sum()),
                             include_groups=False),
    })
    # 배정자본 대비 / 실제 투입자본 대비
    out["ret_alloc_bp"] = 1e4 * out["pnl_alloc"]
    out["ret_used_bp"] = 1e4 * out["pnl_alloc"] / out["w_filled"].replace(0.0, np.nan)
    return out


def summarize_pnl(name: str, p: pd.DataFrame) -> dict:
    a = p["ret_alloc_bp"].to_numpy(dtype="float64")
    used = p["w_filled"].to_numpy(dtype="float64")
    return {"사다리": name, "n_ev": len(p),
            "자본투입률%": 100 * float(np.nanmean(used)),
            "평균_배정bp": float(np.nanmean(a)),
            "중앙_배정bp": float(np.nanmedian(a)),
            "표준편차bp": float(np.nanstd(a, ddof=1)),
            "Sharpe": float(np.nanmean(a) / np.nanstd(a, ddof=1)) if np.nanstd(a, ddof=1) > 0 else np.nan,
            # 체결 0 인 이벤트(무손익)를 분모에 포함한 '시도당' 기준이다.
            # 체결조건부 승률이 아니므로 사다리 간 비교에만 쓴다.
            "양수%": 100 * float(np.nanmean(a > 0)),
            "p05bp": float(np.nanpercentile(a, 5)),
            "최악bp": float(np.nanmin(a))}


def main() -> int:
    ap = argparse.ArgumentParser(description="ladder joint distribution and risk-adjusted EV")
    ap.add_argument("--horizon", type=int, default=15, choices=[15, 60])
    ap.add_argument("--cost-bps", type=float, default=7.0)
    ap.add_argument("--single", type=float, default=2.0, help="단일 레벨 비교 기준(%)")
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 210)
    d = load_panel(a.horizon, a.cost_bps)
    n_ev = d["ev_id"].nunique()
    print("표본: 이벤트 %d | 레벨 %d | 지평 %d분 | 비용 %.0fbp | 심볼 %d"
          % (n_ev, d["u_abs"].nunique(), a.horizon, a.cost_bps, d["symbol"].nunique()))

    # ------------------------------------------------------------ §5.2
    mo = moments(d)
    print("\n=== §5.2  EV 는 평평한데 Sharpe 도 평평한가 ===")
    sel = mo[mo["u%"].isin([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])]
    print(sel.round(3).to_string(index=False))
    if len(mo) > 2:
        lo, hi = mo.iloc[0], mo.iloc[-1]
        print("\n  EV     %.1f -> %.1f bp   (%.2f배)"
              % (lo["EV_bp"], hi["EV_bp"], hi["EV_bp"] / lo["EV_bp"] if lo["EV_bp"] else np.nan))
        print("  Sharpe %.3f -> %.3f      (%.2f배)"
              % (lo["Sharpe"], hi["Sharpe"], hi["Sharpe"] / lo["Sharpe"] if lo["Sharpe"] else np.nan))
        print("  s/m    %.2f -> %.2f   (상수면 s 고정, 1 근처 고정이면 s ∝ m)"
              % (lo["s/m"], hi["s/m"]))
        c = mo[["u%", "Sharpe"]].corr(method="spearman").iloc[0, 1]
        print("  Spearman(u, Sharpe) = %+.3f   (음수면 얕은 쪽이 유리)" % c)

    # ------------------------------------------------------------ §5.3
    print("\n=== §5.3  체결은 중첩된다 — S(u) = P(X >= u) 확인 ===")
    X = d.groupby("ev_id")["X"].first()
    print("  X(지평 내 최대 낙폭) 분위: " +
          "  ".join("%.0f%%=%.2f%%" % (100 * q, 100 * X.quantile(q))
                    for q in (0.25, 0.5, 0.75, 0.9, 0.99)))
    chk = []
    for u, g in d.groupby("u_abs"):
        chk.append({"u%": 100 * u,
                    "S_실측": g["reached"].mean(),
                    "P(X>=u)": float((X >= u).mean())})
    ck = pd.DataFrame(chk).sort_values("u%")
    err = float((ck["S_실측"] - ck["P(X>=u)"]).abs().max())
    print("  |S_실측 - P(X>=u)| 최대 = %.4f  -> %s"
          % (err, "동일 (comonotone 확인)" if err < 0.02 else "불일치 — 정의 재확인 필요"))

    # 사다리 비교
    lv = np.sort(d["u_abs"].unique())
    lv = lv[lv <= 0.06]                        # -6% 까지
    ladders = {}
    w_eq = np.ones(len(lv)) / len(lv)
    ladders["균등 40레벨"] = (lv, w_eq)
    near = lv[lv <= 0.02]
    ladders["얕은쪽만 (~2%)"] = (near, np.ones(len(near)) / len(near))
    far = lv[lv >= 0.02]
    ladders["깊은쪽만 (2%~)"] = (far, np.ones(len(far)) / len(far))
    one = np.array([round(a.single / 100.0, 6)])
    ladders["단일 %.1f%%" % a.single] = (one, np.array([1.0]))

    print("\n  사다리별 손익 (배정자본 기준. 자본투입률 = 실제로 체결된 비중)")
    rows, pnls = [], {}
    for name, (levels, w) in ladders.items():
        p = ladder_pnl(d, levels, w)
        pnls[name] = p
        rows.append(summarize_pnl(name, p))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # 숏풋 형태 확인 — X 구간별 손익
    print("\n=== §5.3b  손익은 X 의 함수다 (숏풋이면 X 가 클수록 급격히 악화) ===")
    p = pnls["균등 40레벨"].copy()
    p["Xb"] = pd.cut(100 * p["X"], [0, 0.5, 1, 2, 3, 5, 10, 100], right=False)
    t = p.groupby("Xb", observed=True).agg(
        n=("ret_alloc_bp", "size"),
        자본투입률=("w_filled", lambda s: 100 * s.mean()),
        평균bp=("ret_alloc_bp", "mean"),
        중앙bp=("ret_alloc_bp", "median"),
        최악bp=("ret_alloc_bp", "min"))
    print("  (X = 지평 내 최대 낙폭 %)")
    print(t.round(1).to_string())

    tot = float(p["ret_alloc_bp"].sum())
    n_worst = max(1, len(p) // 20)
    worst = p.nsmallest(n_worst, "ret_alloc_bp")["ret_alloc_bp"].sum()
    print("\n  총손익 %.0fbp 중 최악 5%%(%d건)의 기여 = %.0fbp (%.0f%%)"
          % (tot, n_worst, worst, 100 * worst / tot if tot else np.nan))
    # Xb 는 Categorical(Interval) 이라 parquet 로 못 나간다 — 문자열로 바꿔 저장.
    p["Xb"] = p["Xb"].astype(str)
    U.atomic_write_parquet(p.reset_index(),
                           os.path.join(C.DATA, "analysis", "ladder_pnl.parquet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
