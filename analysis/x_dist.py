# -*- coding: utf-8 -*-
"""X 의 조건부 확률분포 — 이 연구의 주 산출물.

무엇이 다른가 (앞선 전부와)
  지금까지 배치는 전부 **가격**으로 정했다 — 고정 2%, 예측 X_hat x 1.25 등.
  확률모델이 있으면 반대로 간다:

      체결확률 p 를 먼저 정하고  ->  u_t = q_{1-p}( X | F_t ) 에 건다

  가격은 사건마다 달라지고 체결확률은 p 로 고정된다. 그리고 이것이 보정 검정과
  정확히 같은 물건이다 — 30% 로 걸었으면 실제로 30% 가 체결돼야 한다. 틀리면 모델이 틀렸다.

모델 (PROB_MODEL.md 5절)
  log X | F_t ~ N( m(F_t), s(F_t)^2 )
  핵심은 **m 과 s 를 둘 다 모형화**하는 것. s 를 상수로 두면 그건 다시 점추정이다.

  P( X >= u | F_t ) = 1 - Phi( (log u - m_t) / s_t )
  u_t(p) = exp( m_t + s_t * Phi^{-1}(1 - p) )

사다리 비교 (무엇이 값어치를 만드는지 분리)
  M0 무조건부        m=const,  s=const     -> 사건 구분 없음 = 고정 offset
  M1 평균만 조건부    m=w'x,    s=const     -> 위치만 움직임 (predict_x 와 같은 정보)
  M2 평균+분산 조건부 m=w'x,    s=exp(v'x)  -> **진짜 확률모델**
  M3 구조식 경유      log V 를 M2 로 적합 -> log X = a + gamma(log V - log D)
                                             -> L(p) 가 끼워질 자리를 가진 판

합격 기준 (사전 등록, PROB_MODEL.md 7절 / 계획 5절)
  (1) 보정  9개 p 전부 |실제-명목| <= 5%p 이고 Kupiec p > 0.05
  (2) 판별  조건부 IQR 상/하위 3분위의 **실제** X IQR 비 >= 1.5
  (3) 경제  짝지은 차이가 Bonferroni 보정 후 유의
  순서대로 통과해야 한다. 보정이 깨지면 (3)은 의미가 없다.

실행:
    python analysis/x_dist.py
    python analysis/x_dist.py --hold 15 --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from analysis.vd_structure import build, load_1m, PRE, ols_cluster   # noqa: E402

COST = 7e-4                    # maker 진입 2bp + taker 청산 5bp
PGRID = np.arange(0.1, 0.95, 0.1)      # 목표 체결확률
U_MIN, U_MAX = 1e-3, 0.30              # 배치 하한/상한 (0.1% ~ 30%)
NBOOT = 4000
RNG = np.random.default_rng(11)
# Harvey(1976) 2단계: log(chi2_1) 의 평균이 -1.2704 이므로 절편을 그만큼 되돌린다.
LOG_CHI2_1_MEAN = -1.2704


# ------------------------------------------------------------------ 위치-척도 적합
def fit_loc_scale(Xtr: np.ndarray, ytr: np.ndarray, hetero: bool):
    """log y ~ N(m, s^2). hetero=False 면 s 는 상수.

    2단계 추정(Harvey 1976): 평균을 OLS 로 잡고, 잔차 제곱의 로그를 다시 회귀해
    분산을 잡는다. 그 다음 훈련 PIT 가 균등이 되도록 s 를 한 번 재척도한다
    (2단계 추정량은 편의가 있어 그대로 두면 보정이 체계적으로 어긋난다).
    """
    n = len(ytr)
    wm = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ ytr)
    resid = ytr - Xtr @ wm
    if not hetero:
        s0 = float(np.sqrt(np.sum(resid ** 2) / max(n - Xtr.shape[1], 1)))
        return wm, None, s0
    z = np.log(np.maximum(resid ** 2, 1e-12))
    ws = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ z)
    ws = ws.copy()
    ws[0] -= LOG_CHI2_1_MEAN                       # 절편 편의 보정
    s_tr = np.exp(0.5 * (Xtr @ ws))
    s_tr = np.clip(s_tr, 1e-4, 50.0)
    k = float(np.sqrt(np.mean((resid / s_tr) ** 2)))   # 훈련 PIT 재척도
    return wm, ws, (k if np.isfinite(k) and k > 0 else 1.0)


def predict_loc_scale(Xte, wm, ws, scale, hetero: bool):
    m = Xte @ wm
    if not hetero:
        s = np.full(len(Xte), scale)
    else:
        s = scale * np.exp(0.5 * (Xte @ ws))
    return m, np.clip(s, 1e-4, 50.0)


# ------------------------------------------------------------------ 검정 도구
def kupiec(n: int, x: int, p: float) -> float:
    """무조건부 커버리지 LR 검정. x/n 이 p 와 다른가. chi2_1 p-value."""
    if n == 0 or p <= 0 or p >= 1:
        return np.nan
    ph = x / n
    if ph <= 0 or ph >= 1:
        # 경계에서는 대립가설 우도가 1 이므로 로그항이 사라진다
        ll1 = 0.0
    else:
        ll1 = x * np.log(ph) + (n - x) * np.log(1 - ph)
    ll0 = x * np.log(p) + (n - x) * np.log(1 - p)
    lr = -2.0 * (ll0 - ll1)
    return float(stats.chi2.sf(max(lr, 0.0), 1))


def boot_ci(v: np.ndarray, nboot: int = NBOOT):
    n = len(v)
    if n < 5:
        return float(np.mean(v)) if n else np.nan, np.nan, np.nan
    idx = RNG.integers(0, n, size=(nboot, n))
    b = v[idx].mean(axis=1)
    return float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------ 손익
def pnl_vector(te: pd.DataFrame, m1c: dict, u_arr: np.ndarray, hold: int):
    """이벤트별 손익(bp). 미체결 = 0. 길이 = len(te)."""
    out = np.zeros(len(te))
    filled = np.zeros(len(te), dtype=bool)
    sym = te["symbol"].to_numpy()
    aa = te["a"].to_numpy().astype(int)
    bb = te["b"].to_numpy().astype(int)
    p0 = te["p0"].to_numpy()
    sd = te["side"].to_numpy().astype(int)
    for i in range(len(te)):
        u = u_arr[i]
        if not np.isfinite(u) or u <= 0:
            continue
        lo, hi, cl = m1c[sym[i]]
        n1 = len(cl)
        if bb[i] > n1:
            continue
        limit = p0[i] * (1 - u) if sd[i] == 1 else p0[i] * (1 + u)
        seg = (lo[aa[i]:bb[i]] <= limit) if sd[i] == 1 else (hi[aa[i]:bb[i]] >= limit)
        idx = np.flatnonzero(seg)
        if idx.size == 0:
            continue
        j = aa[i] + int(idx[0])
        e = min(j + hold, n1 - 1)
        out[i] = 1e4 * ((cl[e] / limit - 1.0) * sd[i] - COST)
        filled[i] = True
    return out, filled


# ------------------------------------------------------------------ 모델 정의
MODELS = [
    ("M0 무조건부   (m=c, s=c)", False, False),
    ("M1 평균만     (m=w'x, s=c)", True, False),
    ("M2 평균+분산  (m=w'x, s=exp)", True, True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="conditional distribution of X + calibration")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=15, help="보유 분 (계획 8-3: 15 고정)")
    ap.add_argument("--rebuild", action="store_true", help="캐시 무시하고 다시 만든다")
    a = ap.parse_args()

    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    # 이벤트 표 구성이 심볼당 ~50초라 캐시한다. 4~5단계가 같은 표를 재사용한다.
    cdir = os.path.join(C.DATA, "analysis")
    os.makedirs(cdir, exist_ok=True)
    tag = "%s_k%g_doi%g_gap%d" % ("all" if not a.symbols else "sub%d" % len(syms),
                                  a.k, a.doi, a.min_gap)
    cpath = os.path.join(cdir, "events_%s.parquet" % tag)

    d = None
    if os.path.exists(cpath) and not a.rebuild:
        try:
            d = pd.read_parquet(cpath)
            if set(d["symbol"].unique()) - set(syms):
                d = None                      # 캐시가 요청 심볼과 안 맞으면 버린다
            else:
                U.log("캐시 사용: %s (%d행)" % (cpath, len(d)))
        except Exception as e:                # noqa: BLE001 — 손상 캐시는 재생성
            U.log("캐시 읽기 실패(%s) — 재생성" % e)
            d = None

    if d is None:
        frames = []
        for s in syms:
            try:
                dd = build(s, a.k, a.doi, a.min_gap)
            except FileNotFoundError as e:
                U.log(str(e))
                continue
            if dd.empty:
                continue
            frames.append(dd)
            U.log("%s: %d" % (s, len(dd)))
        if not frames:
            U.log("이벤트 없음")
            return 1
        d = pd.concat(frames, ignore_index=True)
        try:
            d.to_parquet(cpath, index=False)
            U.log("캐시 저장: %s" % cpath)
        except Exception as e:                # noqa: BLE001 — 캐시 실패는 치명적이지 않다
            U.log("캐시 저장 실패: %s" % e)

    m1c = {}
    for s in sorted(d["symbol"].unique()):
        m1 = load_1m(s)
        m1c[s] = (m1["low"].to_numpy(), m1["high"].to_numpy(), m1["close"].to_numpy())
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d = d.sort_values("dt").reset_index(drop=True)
    d = d.dropna(subset=PRE + ["X"]).reset_index(drop=True)

    cut = len(d) // 2
    tr = d.iloc[:cut].reset_index(drop=True)
    te = d.iloc[cut:].reset_index(drop=True)
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in PRE])
    Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in PRE])
    ytr = np.log(tr["X"].to_numpy())
    yte = np.log(te["X"].to_numpy())

    print("\n" + "=" * 76)
    print("X 의 조건부 확률분포 — 보정 / 판별 / 경제성")
    print("=" * 76)
    print("표본 %d | %d종 | %s ~ %s | 훈련 %d / 평가 %d | 보유 %d분 | 비용 %.0fbp"
          % (len(d), d.symbol.nunique(), d["dt"].min().date(), d["dt"].max().date(),
             len(tr), len(te), a.hold, 1e4 * COST))
    print("X 분위(평가): " + "  ".join("%d%%=%.2f%%" % (100 * q, 100 * te["X"].quantile(q))
                                       for q in (.1, .25, .5, .75, .9)))

    fits = {}
    for name, cond_m, cond_s in MODELS:
        wm, ws, sc = fit_loc_scale(Xtr if cond_m else Xtr[:, :1],
                                   ytr, cond_s)
        m, s = predict_loc_scale(Xte if cond_m else Xte[:, :1], wm, ws, sc, cond_s)
        fits[name] = (m, s)

    # ---------------------------------------------------------- 0. PIT
    print("\n--- 0. PIT 균등성 (분포 전체를 한 번에 본다) ---")
    print("  F_t(X_t) 가 Uniform(0,1) 이어야 한다. KS 로 검정.")
    print("  %-30s %10s %10s %10s" % ("모델", "KS D", "p", "PIT 평균"))
    for name, _, _ in MODELS:
        m, s = fits[name]
        pit = stats.norm.cdf((yte - m) / s)
        ks = stats.kstest(pit, "uniform")
        print("  %-30s %10.4f %10.4f %10.3f" % (name, ks.statistic, ks.pvalue, pit.mean()))

    # ---------------------------------------------------------- 1. 보정
    print("\n--- 1. 보정: 목표 체결확률 p 로 걸면 실제로 p 가 체결되는가 ---")
    print("  u_t(p) = exp( m_t + s_t * z_{1-p} ),  체결 = X >= u")
    print("  합격: 9개 p 전부 |실제-명목| <= 5%p 이고 Kupiec p > 0.05")
    for name, _, _ in MODELS:
        m, s = fits[name]
        print("\n  [%s]" % name)
        print("    %8s %10s %10s %9s %10s" % ("명목 p", "실제", "차이(%p)", "Kupiec p", "u중앙%"))
        ok = True
        nk = 0
        for p in PGRID:
            u = np.clip(np.exp(m + s * stats.norm.ppf(1 - p)), U_MIN, U_MAX)
            hit = int(np.sum(te["X"].to_numpy() >= u))
            ph = hit / len(te)
            kp = kupiec(len(te), hit, p)
            bad = (abs(ph - p) > 0.05) or (np.isfinite(kp) and kp <= 0.05)
            ok &= not bad
            nk += int(np.isfinite(kp) and kp <= 0.05)
            print("    %8.1f %10.3f %+10.1f %9.3f %10.2f%s"
                  % (p, ph, 100 * (ph - p), kp, 100 * np.median(u), "  X" if bad else ""))
        # 참고: 비율의 SE. n 이 작으면 +-5%p 기준 자체가 1 SE 미만이라 도달 불가다.
        se50 = 100 * np.sqrt(0.25 / len(te))
        print("    => 보정 %s   (Kupiec 위반 %d/9 | p=0.5 에서 비율 SE = %.1f%%p)"
              % ("통과" if ok else "불합격", nk, se50))

    # ---------------------------------------------------------- 2. 판별
    print("\n--- 2. 판별: '넓다고 예측한 사건'이 실제로 넓은가 ---")
    print("  합격: 상/하위 3분위의 실제 IQR 비 >= 1.5")
    z75, z25 = stats.norm.ppf(0.75), stats.norm.ppf(0.25)
    Xv = te["X"].to_numpy()
    lXv = np.log(Xv)

    def terciles(v):
        r = pd.Series(v).rank(method="first").to_numpy()
        return np.floor(3.0 * (r - 1) / len(r)).astype(int).clip(0, 2)

    print("\n  (2a) 원안 — 예측 IQR(수준) 3분위 -> 실제 X 의 IQR")
    print("       *** 교락 주의: 로그정규에서 IQR ∝ exp(m) 이라, s 가 상수여도")
    print("       중앙값이 큰 사건은 IQR 이 크다. 즉 '퍼짐'이 아니라 '수준'을 재고 있다.")
    for name, _, _ in MODELS:
        m, s = fits[name]
        iqr_pred = np.exp(m + s * z75) - np.exp(m + s * z25)
        if np.allclose(iqr_pred, iqr_pred[0]):
            print("  %-30s 예측 IQR 전부 동일 — 판별 불가 (설계상 당연)" % name)
            continue
        t3 = terciles(iqr_pred)
        lo_ = float(np.percentile(Xv[t3 == 0], 75) - np.percentile(Xv[t3 == 0], 25))
        hi_ = float(np.percentile(Xv[t3 == 2], 75) - np.percentile(Xv[t3 == 2], 25))
        ratio = hi_ / lo_ if lo_ > 0 else np.nan
        print("  %-30s 하위 %.3f%%  상위 %.3f%%  비 %.2f  %s"
              % (name, 100 * lo_, 100 * hi_, ratio, "통과" if ratio >= 1.5 else "불합격"))

    print("\n  (2b) 교정 — 예측 s(퍼짐) 3분위 -> 실제 **log X** 의 IQR (척도 무관)")
    print("       수준 효과를 제거한다. sigma 모형이 실제로 기여하는지는 이쪽이 답한다.")
    for name, _, cond_s in MODELS:
        m, s = fits[name]
        if np.allclose(s, s[0]):
            print("  %-30s s 가 상수 — 판별 대상 아님 (설계상 당연)" % name)
            continue
        t3 = terciles(s)
        lo_ = float(np.percentile(lXv[t3 == 0], 75) - np.percentile(lXv[t3 == 0], 25))
        hi_ = float(np.percentile(lXv[t3 == 2], 75) - np.percentile(lXv[t3 == 2], 25))
        ratio = hi_ / lo_ if lo_ > 0 else np.nan
        print("  %-30s 하위 %.3f  상위 %.3f  비 %.2f  %s"
              % (name, lo_, hi_, ratio, "통과" if ratio >= 1.5 else "불합격"))
        print("       (예측 s 3분위 중앙: %.3f / %.3f / %.3f)"
              % tuple(float(np.median(s[t3 == q])) for q in (0, 1, 2)))

    # ---------------------------------------------------------- 3. 경제성
    print("\n--- 3. 경제성: p 별 실제 손익 (평가 %d건, 보유 %d분) ---" % (len(te), a.hold))
    print("  '몇 %% 확률에서 진입할 것인가' 의 답은 EV 를 최대로 만드는 p 다.")
    pnl = {}
    for name, _, _ in MODELS:
        m, s = fits[name]
        print("\n  [%s]" % name)
        print("    %8s %9s %10s %10s %22s"
              % ("목표 p", "실제체결", "u중앙%", "EV bp", "95% CI"))
        for p in PGRID:
            u = np.clip(np.exp(m + s * stats.norm.ppf(1 - p)), U_MIN, U_MAX)
            v, f = pnl_vector(te, m1c, u, a.hold)
            pnl[(name, round(float(p), 1))] = v
            mn, lo_, hi_ = boot_ci(v)
            print("    %8.1f %8.1f%% %10.2f %10.1f     [%+7.1f, %+7.1f]"
                  % (p, 100 * f.mean(), 100 * np.median(u), mn, lo_, hi_))

    # 고정 offset 대조군
    print("\n  [고정 offset (대조군)]")
    print("    %8s %9s %10s %10s %22s" % ("u", "실제체결", "-", "EV bp", "95% CI"))
    for u0 in (0.005, 0.01, 0.02, 0.03):
        v, f = pnl_vector(te, m1c, np.full(len(te), u0), a.hold)
        pnl[("FIX", u0)] = v
        mn, lo_, hi_ = boot_ci(v)
        print("    %8.1f%% %8.1f%% %10s %10.1f     [%+7.1f, %+7.1f]"
              % (100 * u0, 100 * f.mean(), "-", mn, lo_, hi_))

    # ---------------------------------------------------------- 4. 짝지은 차이
    print("\n--- 4. M2(진짜 확률모델) 가 대조군을 이기는가 — 짝지은 차이 ---")
    # p* 는 반드시 **훈련 표본**에서 고른다. 평가에서 고르면 그 자체가 표본 내 선택이라
    # M2 에만 유리하게 부풀려진다(3종 스모크에서 실제로 발생).
    wm2, ws2, sc2 = fit_loc_scale(Xtr, ytr, True)
    m_tr, s_tr = predict_loc_scale(Xtr, wm2, ws2, sc2, True)
    tr_ev = {}
    for p in PGRID:
        u = np.clip(np.exp(m_tr + s_tr * stats.norm.ppf(1 - p)), U_MIN, U_MAX)
        v, _ = pnl_vector(tr, m1c, u, a.hold)
        tr_ev[round(float(p), 1)] = float(np.mean(v))
    best_p = max(tr_ev, key=tr_ev.get)
    base = pnl[(MODELS[2][0], best_p)]
    print("  p* 선택 = 훈련표본 (평가에서 고르면 M2 에만 유리하게 부풀려진다)")
    print("  훈련 EV: " + "  ".join("p=%.1f:%.0f" % (k, v) for k, v in sorted(tr_ev.items())))
    print("  => p* = %.1f  (훈련 %.1fbp, 평가 %.1fbp)"
          % (best_p, tr_ev[best_p], np.mean(base)))
    comps = ([(MODELS[0][0], best_p), (MODELS[1][0], best_p)]
             + [("FIX", u0) for u0 in (0.005, 0.01, 0.02, 0.03)])
    m_tests = len(comps)
    alpha = 0.05 / m_tests
    print("  비교 %d개 -> Bonferroni alpha = %.4f (CI %.1f%%)"
          % (m_tests, alpha, 100 * (1 - alpha)))
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for key in comps:
        diff = base - pnl[key]
        n = len(diff)
        idx = RNG.integers(0, n, size=(NBOOT, n))
        b = diff[idx].mean(axis=1)
        lo_, hi_ = np.percentile(b, lo_q), np.percentile(b, hi_q)
        sig = " ***" if (lo_ > 0 or hi_ < 0) else ""
        lab = key[0] if key[0] != "FIX" else "고정 %.1f%%" % (100 * key[1])
        print("    M2 - %-30s = %+7.1f bp  [%+7.1f, %+7.1f]%s"
              % (lab, np.mean(diff), lo_, hi_, sig))

    print("\n  판정 순서: 보정 -> 판별 -> 경제성. 보정이 깨지면 경제성은 의미가 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
