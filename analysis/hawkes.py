# -*- coding: utf-8 -*-
"""2·4단계 — Hawkes 과정으로 연쇄청산의 자기여기를 적합한다.

왜 Hawkes 인가
  캐스케이드는 '청산이 청산을 부르는' 자기여기 점과정이다. 지도가 '어디에 연료가
  있나' 를 말한다면 Hawkes 는 **'불이 번질까'** 를 말한다.

      lambda(t) = mu + sum_{t_i < t} alpha * w_i * exp(-beta (t - t_i))

  분기비  n = alpha * E[w] / beta
      n < 1  몇 개 넘어지다 멈춤 (아임계)
      n -> 1 임계 — 거대 캐스케이드
      n > 1  폭주

  MODEL.md 에 R_0 ~ (1/D)(dL/du) 로 **이미 적혀 있고 한 번도 적합한 적이 없다.**

이 스크립트가 하는 것 (계획 4.1)
  (1) 마크드 Hawkes MLE — 지수 커널, 크기 마크 w_i
  (2) **잔차 진단** — 시간 재조정 tau_i = 적분 lambda 후 간격이 Exp(1) 인가 (KS)
      이게 통과 못하면 모형이 데이터에 안 맞는 것이다. 적합값을 믿으면 안 된다.
  (3) 분기비 n 과 그 심볼별 이질성
  (4) **적시성 곡선** — 캐스케이드 시작 후 t 분에 추정한 lambda 가 얼마나 안정적인가
      (Architect 반론: n 이 사후적으로만 관측되면 문제를 옮긴 것일 뿐이다)

*** 이음매 계측 (계획 4.7) 은 synth.py 에서 한다. 여기서는 모듈 자체만 본다. ***

한계
  - Bybit 47시간. 한 레짐(롱 편중 하락).
  - 심볼 풀링. 심볼당 170건이라 개별 적합 불가 -> 이질성은 사후 점검만.

실행:
    python analysis/hawkes.py
    python analysis/hawkes.py --min-usd 1000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

MIN_EVENTS = 50


def load_liq(min_usd: float) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "bybit_liq", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/bybit_liq 비어 있음")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["symbol"].isin(C.MAJORS)].copy()
    d["ntl"] = d["size"] * d["bankruptcy_px"]
    d = d[np.isfinite(d["ntl"]) & (d["ntl"] >= min_usd)]
    return d.sort_values("exch_ms").reset_index(drop=True)


def nll(par, t, w, T):
    """마크드 Hawkes 음의 로그우도 (지수 커널, 재귀식).

    lambda(t) = mu + sum alpha w_i exp(-beta (t-t_i))
    재귀: A_k = exp(-beta dt) (A_{k-1} + w_{k-1})  -> O(n)
    """
    lmu, lal, lbe = par
    mu, al, be = np.exp(lmu), np.exp(lal), np.exp(lbe)
    n = len(t)
    A = 0.0
    ll = 0.0
    for k in range(n):
        if k > 0:
            A = np.exp(-be * (t[k] - t[k - 1])) * (A + w[k - 1])
        lam = mu + al * A
        if lam <= 0:
            return 1e12
        ll += np.log(lam)
    # 적분항: mu*T + (alpha/beta) * sum w_i (1 - exp(-beta (T - t_i)))
    ll -= mu * T + (al / be) * float(np.sum(w * (1.0 - np.exp(-be * (T - t)))))
    return -ll


# --- 멱함수 커널 ---------------------------------------------------------
# phi(tau) = alpha / (1 + tau/tau0)^(1+eps) 를 지수 M개의 합으로 근사한다.
#   beta_j 를 로그 격자에 깔고 가중치를 beta_j^eps 로 주면 포락선이 멱함수가 된다.
#   (Hardiman-Bouchaud 이 쓴 표준 근사. 지수 재귀를 성분별로 유지해 O(nM).)
# 관측 간격이 0.1초~2.4만초(5자릿수)이므로 격자도 그만큼 깔아야 한다.
BETA_GRID = np.array([10.0 ** k for k in np.arange(1.0, -4.01, -0.5)])   # 10/s ~ 1e-4/s


def _pl_weights(eps: float) -> np.ndarray:
    """포락선이 tau^-(1+eps) 가 되도록 성분 가중치를 만든다 (합=1 로 정규화)."""
    a = BETA_GRID ** eps
    s = float(np.sum(a / BETA_GRID))          # 각 성분의 적분 = a_j/beta_j
    return a / s if s > 0 else a


def nll_pl(par, t, w, T):
    """멱함수 커널 음의 로그우도. par = (log mu, log alpha, log eps)."""
    lmu, lal, lep = par
    mu, al, eps = np.exp(lmu), np.exp(lal), np.exp(lep)
    if not (1e-4 < eps < 5.0):
        return 1e12
    aj = al * _pl_weights(eps)                # 성분별 진폭
    n = len(t)
    A = np.zeros(len(BETA_GRID))
    ll = 0.0
    for k in range(n):
        if k > 0:
            A = np.exp(-BETA_GRID * (t[k] - t[k - 1])) * (A + w[k - 1])
        lam = mu + float(np.dot(aj, A))
        if lam <= 0:
            return 1e12
        ll += np.log(lam)
    rem = T - t
    for j, b in enumerate(BETA_GRID):
        ll -= (aj[j] / b) * float(np.sum(w * (1.0 - np.exp(-b * rem))))
    ll -= mu * T
    return -ll


def fit_pl(t, w, T):
    x0 = np.array([np.log(max(len(t) / T, 1e-6)), np.log(0.3), np.log(0.3)])
    r = optimize.minimize(nll_pl, x0, args=(t, w, T), method="Nelder-Mead",
                          options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-5})
    mu, al, eps = np.exp(r.x)
    return mu, al, eps, float(r.fun), bool(r.success)


def compensator_pl(t, w, mu, al, eps):
    aj = al * _pl_weights(eps)
    n = len(t)
    out = np.empty(n)
    S = np.zeros(len(BETA_GRID))
    W = 0.0
    for k in range(n):
        if k > 0:
            S = np.exp(-BETA_GRID * (t[k] - t[k - 1])) * (S + w[k - 1])
            W += w[k - 1]
        out[k] = mu * t[k] + float(np.sum((aj / BETA_GRID) * (W - S)))
    return out


def fit_hawkes(t: np.ndarray, w: np.ndarray, T: float):
    x0 = np.array([np.log(max(len(t) / T, 1e-6)), np.log(0.5), np.log(1.0)])
    r = optimize.minimize(nll, x0, args=(t, w, T), method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    mu, al, be = np.exp(r.x)
    return mu, al, be, float(r.fun), bool(r.success)


def compensator(t, w, mu, al, be):
    """Lambda(t_k) = mu*t_k + (al/be) sum_{i<k} w_i (1 - exp(-be (t_k - t_i)))

    재귀로 O(n). 시간 재조정 잔차 tau_k = Lambda(t_k) - Lambda(t_{k-1}) 는
    모형이 맞으면 Exp(1) 이다 (Papangelou / random time change).
    """
    n = len(t)
    out = np.empty(n)
    S = 0.0           # sum w_i exp(-be(t-t_i))
    W = 0.0           # sum w_i
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            S = np.exp(-be * dt) * (S + w[k - 1])
            W += w[k - 1]
        out[k] = mu * t[k] + (al / be) * (W - S)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Hawkes fit on realized liquidations")
    ap.add_argument("--min-usd", type=float, default=500.0,
                    help="이 미만 청산은 먼지라 제외 (중앙 \\$166)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--kernel", choices=("exp", "pl"), default="pl",
                    help="pl = 멱함수(장기기억). exp 는 잔차 0/6 불합격했다")
    a = ap.parse_args()
    U.init_stdout()

    d = load_liq(a.min_usd)
    syms = a.symbols if a.symbols else sorted(d["symbol"].unique())
    t0, t1 = int(d.exch_ms.min()), int(d.exch_ms.max())
    print("=" * 74)
    print("Hawkes 적합 — 연쇄청산의 자기여기 (계획 2·4단계)")
    print("=" * 74)
    print("청산 %d건 (>=\\$%.0f) / %d종 | %s ~ %s (%.1f시간)"
          % (len(d), a.min_usd, d.symbol.nunique(),
             pd.to_datetime(t0, unit="ms", utc=True),
             pd.to_datetime(t1, unit="ms", utc=True), (t1 - t0) / 3.6e6))

    print("\n--- 1. 심볼별 적합 (시간 단위 = 초, 마크 = 명목가/중앙값) ---")
    print("  %-10s %7s %10s %10s %10s %9s %9s"
          % ("심볼", "n", "mu(/s)", "alpha", "beta(/s)", "분기비 n", "잔차KS p"))
    rows = []
    for s in syms:
        g = d[d.symbol == s]
        if len(g) < MIN_EVENTS:
            continue
        t = (g["exch_ms"].to_numpy() - t0) / 1000.0
        # 동시각 이벤트는 아주 작은 지터로 분리 (같은 ms 에 여러 체결이 온다)
        t = t + np.arange(len(t)) * 1e-6
        w = g["ntl"].to_numpy()
        w = w / np.median(w)
        T = float((t1 - t0) / 1000.0)
        wbar = float(np.mean(w))
        if a.kernel == "exp":
            mu, al, be, f, ok = fit_hawkes(t, w, T)
            n_br = al * wbar / be
            lam = compensator(t, w, mu, al, be)
            p3 = be
        else:
            mu, al, eps, f, ok = fit_pl(t, w, T)
            # 성분 가중치가 합=1 로 정규화돼 있으므로 총 적분 = alpha
            n_br = al * wbar
            lam = compensator_pl(t, w, mu, al, eps)
            p3 = eps
        tau = np.diff(lam)
        tau = tau[np.isfinite(tau) & (tau > 0)]
        ks = stats.kstest(tau, "expon") if len(tau) > 20 else None
        rows.append({"symbol": s, "n": len(g), "mu": mu, "alpha": al, "beta": p3,
                     "n_br": n_br, "ks_p": ks.pvalue if ks else np.nan,
                     "ks_D": ks.statistic if ks else np.nan, "ok": ok})
        print("  %-10s %7d %10.4g %10.4g %10.4g %9.3f %9.4f"
              % (s, len(g), mu, al, p3, n_br, ks.pvalue if ks else np.nan))
    r = pd.DataFrame(rows)
    if r.empty:
        print("적합 가능한 심볼 없음 — 축적 필요")
        return 1

    print("\n--- 2. 잔차 진단 (계획 합격기준: KS p > 0.05) ---")
    npass = int((r.ks_p > 0.05).sum())
    print("  통과 %d / %d 심볼" % (npass, len(r)))
    print("  KS p 중앙 %.4f | 최소 %.4f" % (r.ks_p.median(), r.ks_p.min()))
    print("  *** 잔차가 Exp(1) 이 아니면 지수 커널이 틀린 것이다.")
    print("      멱함수 커널(장기기억)이나 다변량(심볼 간 전염)이 필요할 수 있다.")

    print("\n--- 3. 분기비 n — '불이 번질까' ---")
    print("  n 중앙 %.3f | p25 %.3f | p75 %.3f | 최대 %.3f"
          % (r.n_br.median(), r.n_br.quantile(.25), r.n_br.quantile(.75), r.n_br.max()))
    print("  n>=1 인 심볼 %d개 (폭주 영역)" % int((r.n_br >= 1).sum()))
    print("  n>=0.8 인 심볼 %d개 (임계 근처)" % int((r.n_br >= 0.8).sum()))
    print("  해석: n 은 '청산 1건이 낳는 2차 청산 기대 건수' 다.")
    print("        n 이 1 에 가까울수록 작은 충격이 큰 캐스케이드로 번진다.")

    print("\n--- 4. 적시성: 캐스케이드 시작 후 t 초에 lambda 가 얼마나 튀는가 ---")
    print("  Architect 반론 — n 이 사후적으로만 보이면 문제를 옮긴 것일 뿐이다.")
    big = r.nlargest(min(5, len(r)), "n")
    print("  %-10s %8s %8s %8s %8s %8s %8s"
          % ("심볼", "t=5s", "t=15s", "t=30s", "t=60s", "t=120s", "t=300s"))
    for _, rr in big.iterrows():
        g = d[d.symbol == rr.symbol]
        t = (g["exch_ms"].to_numpy() - t0) / 1000.0 + np.arange(len(g)) * 1e-6
        w = g["ntl"].to_numpy(); w = w / np.median(w)
        # 가장 큰 청산 군집을 캐스케이드 시작으로 본다
        k0 = int(np.argmax(w))
        base = rr.mu
        vals = []
        for dt in (5, 15, 30, 60, 120, 300):
            m = (t >= t[k0]) & (t < t[k0] + dt)
            if not np.any(m):
                vals.append(np.nan); continue
            if a.kernel == "exp":
                ker = np.exp(-rr.beta * (t[k0] + dt - t[m]))
                lam = rr.mu + rr.alpha * float(np.sum(w[m] * ker))
            else:
                aj = rr.alpha * _pl_weights(rr.beta)
                dtau = (t[k0] + dt - t[m])[:, None]
                lam = rr.mu + float(np.sum(
                    w[m][:, None] * aj[None, :] * np.exp(-BETA_GRID[None, :] * dtau)))
            vals.append(lam / base if base > 0 else np.nan)
        print("  %-10s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f"
              % (rr.symbol, *vals))
    print("  (기저 강도 mu 대비 배수. 클수록 그 시점에 '번지는 중' 이라는 신호가 강하다)")
    print("  값이 t=5s 에 이미 크면 조기 판별 가능. t=300s 에야 커지면 너무 늦다.")

    print("\n--- 5. 심볼 이질성 ---")
    print("  분기비 n: 평균 %.3f  sd %.3f  변동계수 %.3f"
          % (r.n_br.mean(), r.n_br.std(ddof=1), r.n_br.std(ddof=1) / max(r.n_br.mean(), 1e-9)))
    print("  beta(감쇠율) 중앙 %.4g /s  -> 반감기 %.1f초"
          % (r.beta.median(), np.log(2) / max(r.beta.median(), 1e-12)))
    print("\n  *** 47시간 한 레짐이다. 축적하며 주간 재적합해야 한다(계획 R2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
