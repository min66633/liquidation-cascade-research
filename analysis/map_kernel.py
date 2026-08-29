# -*- coding: utf-8 -*-
"""D-1 — 청산맵 L(p) 의 커널 추정. **총 디레버리징** 기준.

설계상의 위치 (DESIGN_LOCK.md §1.1)
    X_hat = inf{ u : ∫[D(v,t)+Δ(v,t)]dv >= ∫L(p_t(1-v))dv * φ }
  L(p) 는 **분자**다. 이 스크립트는 그 L 을 만드는 커널을 추정한다.
  분모(오더북·유입취소)는 D-3 에서 붙인다. 여기서 X 를 예보하려 하지 말 것.

무엇을 추정하나
  "가격 p0 에 진입한 포지션은, 가격이 p 로 움직였을 때 얼마나 청산되는가"
      h(x),   x = log(p/p0)   = 진입 이후 수익률(로그)
  이것을 **이탈 위험률(exit hazard)** 이라 부른다. 강제청산만이 아니라
  **자발 손절·이익실현·교차마진 청산을 전부 포함한 총 디레버리징**이다
  (DESIGN_LOCK.md §2: 청산 프린트는 OI 감소의 약 1% 뿐이다).

식별 방법
  매 시점 미결제약정을 **진입가격별 히스토그램** H_k 로 들고 다닌다.
    OI 증가 -> 현재가 버킷에 추가
    OI 감소 -> **비례 배분**으로 제거 (1차 근사; 이 가정으로부터의 **이탈**을 잰다)
  각 시점에서 코호트별 현재 수익률 x_k = log(p_t) - log(p_k) 를 계산하고,
  x 를 구간으로 나눠 노출량 C_j(t) = Σ_{k∈j} H_k 를 만든 뒤

      ΔOI^-(t) = Σ_j β_j C_j(t) + ε

  를 회귀한다. **β_j 가 곧 h(x) 다** — 그 수익률 구간에 있는 포지션이 단위당
  얼마나 청산되는가. 비례배분 귀무가설이면 β 가 j 에 무관하게 상수여야 한다.
  **β 가 x 에 따라 기울면 그것이 지도의 정보다.**

주의
  OI 는 **계약수(sum_open_interest)** 를 쓴다. 명목가는 가격이 움직이기만 해도
  변하므로 흐름과 가격효과가 섞인다.

실행:
    python analysis/map_kernel.py
    python analysis/map_kernel.py --step 6 --symbols BTCUSDT ETHUSDT
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
from analysis.event_study_h2 import load                        # noqa: E402
from analysis.response_liq import ols_cluster                   # noqa: E402

# 수익률 구간 (로그). 음수 = 손실(롱 기준). 청산은 손실 쪽에 몰릴 것으로 예상.
# 0 을 경계로 두면 안 된다. 0 양옆 두 구간은 코호트가 미세한 가격변동으로 서로
# 넘나들어 **거의 완전공선**이고, 회귀가 크기 같고 부호 반대인 거대 계수를 준다
# (첫 판 실측: -0.01085 / +0.01185, 그것이 손익비 0.21배를 통째로 만들었다).
# 0 을 **품는 하나의 구간**으로 합치고, 손익 비교에서는 그 구간을 양쪽 모두 제외한다.
EDGES = np.array([-np.inf, -0.30, -0.20, -0.15, -0.10, -0.07, -0.05, -0.03,
                  -0.015, 0.015, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, np.inf])
LAB = ["<-30%", "-30~-20", "-20~-15", "-15~-10", "-10~-7", "-7~-5", "-5~-3",
       "-3~-1.5", "**-1.5~1.5**", "1.5~3", "3~5", "5~7", "7~10",
       "10~15", "15~20", ">20%"]
MAXC = 4000            # 코호트 상한 (초과 시 작은 것부터 병합)
MINFRAC = 1e-7         # 이 비율 미만 코호트는 버린다


def cohort_panel(sym: str, step: int, h: np.ndarray | None = None):
    """진입가격별 코호트를 **롱/숏 분리해서** 굴린다.

    *** 왜 분리가 필수인가 ***
      집계 OI 는 롱+숏이다. 가격이 내려가면 같은 코호트 안에서 롱은 손실, 숏은
      이익이다. x = log(p_t/p_0) 하나로 묶으면 둘을 평균내 **대칭을 인위적으로
      만들어낸다** — 첫 판이 정확히 그 오류였다(손익 비 1.04, t=0.6).
      롱은 x, 숏은 -x 를 쓴다.

    롱숏 비율은 metrics 의 sum_toptrader_long_short_ratio 를 쓴다(상위 트레이더
    포지션 크기 기준). 전체 트레이더가 아니라 **근사**임을 명시한다.
    """
    try:
        df = load(sym)
    except FileNotFoundError:
        return None
    d = df.iloc[::step].reset_index(drop=True)
    oi = d["sum_open_interest"].to_numpy(dtype=np.float64)
    px = d["close"].to_numpy(dtype=np.float64)
    ot = d["open_time"].to_numpy()
    rr = d["sum_toptrader_long_short_ratio"].to_numpy(dtype=np.float64)
    rr = pd.Series(rr).ffill().bfill().to_numpy()
    m = (np.isfinite(oi) & (oi > 0) & np.isfinite(px) & (px > 0)
         & np.isfinite(rr) & (rr > 0))
    oi, px, ot, rr = oi[m], px[m], ot[m], rr[m]
    n = len(oi)
    if n < 5000:
        return None
    lp = np.log(px)
    fl = rr / (1.0 + rr)                       # 롱 비중
    nb = len(EDGES) - 1

    # 롱/숏 두 벌의 코호트
    cp = [np.empty(MAXC), np.empty(MAXC)]      # 0=롱, 1=숏
    cq = [np.zeros(MAXC), np.zeros(MAXC)]
    nc = [1, 1]
    cp[0][0], cq[0][0] = lp[0], oi[0] * fl[0]
    cp[1][0], cq[1][0] = lp[0], oi[0] * (1.0 - fl[0])

    rows = []
    for t in range(1, n):
        dq = oi[t] - oi[t - 1]
        tot_prev = cq[0][:nc[0]].sum() + cq[1][:nc[1]].sum()
        if tot_prev > 0:
            expo = np.zeros(nb)
            for sd in (0, 1):
                if nc[sd] == 0:
                    continue
                # 롱은 x, 숏은 -x. 이것이 이번 판의 핵심 수정이다.
                x = (lp[t - 1] - cp[sd][:nc[sd]]) * (1.0 if sd == 0 else -1.0)
                idx = np.digitize(x, EDGES[1:-1])
                expo += np.bincount(idx, weights=cq[sd][:nc[sd]], minlength=nb)[:nb]
            rows.append((ot[t], tot_prev, max(-dq, 0.0), max(dq, 0.0), expo))

        if dq > 0:
            add = (dq * fl[t], dq * (1.0 - fl[t]))
            for sd in (0, 1):
                if add[sd] <= 0:
                    continue
                if nc[sd] >= MAXC:
                    j = int(np.argmin(cq[sd][:nc[sd]]))
                    k = int(np.argmin(np.where(np.arange(nc[sd]) == j, np.inf,
                                               cq[sd][:nc[sd]])))
                    s2 = cq[sd][j] + cq[sd][k]
                    cp[sd][j] = ((cp[sd][j] * cq[sd][j] + cp[sd][k] * cq[sd][k])
                                 / max(s2, 1e-12))
                    cq[sd][j] = s2
                    cp[sd][k], cq[sd][k] = cp[sd][nc[sd] - 1], cq[sd][nc[sd] - 1]
                    nc[sd] -= 1
                cp[sd][nc[sd]], cq[sd][nc[sd]] = lp[t], add[sd]
                nc[sd] += 1
        elif dq < 0 and tot_prev > 0:
            need = min(-dq, tot_prev)
            if h is None:
                # 반복 0회차: 비례 배분. **이것이 귀무가설이고 편향의 원인이다.**
                keep = max(1.0 - need / tot_prev, 0.0)
                for sd in (0, 1):
                    cq[sd][:nc[sd]] *= keep
            else:
                # 반복 1회차 이후: 추정된 h(x) 에 비례해서 제거한다.
                # 손실 포지션이 실제로 더 빨리 닫힌다면 장부가 그것을 반영하게 되고,
                # 그러면 다음 회차의 분모(노출량)가 올바르게 줄어든다.
                wts = []
                for sd in (0, 1):
                    if nc[sd] == 0:
                        wts.append(np.zeros(0))
                        continue
                    x = (lp[t - 1] - cp[sd][:nc[sd]]) * (1.0 if sd == 0 else -1.0)
                    hh = h[np.digitize(x, EDGES[1:-1])]
                    wts.append(np.maximum(hh, 0.0) * cq[sd][:nc[sd]])
                tw = wts[0].sum() + wts[1].sum()
                for sd in (0, 1):
                    if nc[sd] == 0:
                        continue
                    if tw > 0:
                        take = need * wts[sd] / tw
                        cq[sd][:nc[sd]] = np.maximum(
                            cq[sd][:nc[sd]] - np.minimum(take, cq[sd][:nc[sd]]), 0.0)
                    else:
                        cq[sd][:nc[sd]] *= max(1.0 - need / tot_prev, 0.0)
            for sd in (0, 1):
                if nc[sd] > 200:
                    thr = MINFRAC * max(tot_prev, 1e-9)
                    good = cq[sd][:nc[sd]] > thr
                    g = int(good.sum())
                    if 0 < g < nc[sd]:
                        cp[sd][:g] = cp[sd][:nc[sd]][good]
                        cq[sd][:g] = cq[sd][:nc[sd]][good]
                        nc[sd] = g
    if not rows:
        return None
    E = np.vstack([r[4] for r in rows])
    return pd.DataFrame({
        "t": [r[0] for r in rows], "symbol": sym,
        "oi": [r[1] for r in rows], "out": [r[2] for r in rows],
        "inn": [r[3] for r in rows],
        **{"e%d" % j: E[:, j] for j in range(nb)}})


def main() -> int:
    ap = argparse.ArgumentParser(description="D-1 map kernel: exit hazard h(x)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--step", type=int, default=6, help="5분봉 몇 개마다 (6=30분)")
    ap.add_argument("--iters", type=int, default=6, help="EM 반복 최대 횟수")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 80)
    print("D-1 청산맵 커널 — 진입가 대비 수익률별 **총 디레버리징** 위험률 h(x)")
    print("=" * 80)
    print("설계상 위치: L(p) 는 **분자**다. 분모(오더북)는 D-3. 여기서 X 를 예보하지 않는다.")
    print("OI 는 계약수 기준(명목가는 가격효과가 섞인다). 시간격자 %d분\n" % (5 * a.step))

    nb = len(EDGES) - 1
    ecols = ["e%d" % j for j in range(nb)]

    def fit(h_in):
        fr = []
        for s in syms:
            p = cohort_panel(s, a.step, h_in)
            if p is not None:
                fr.append(p)
        if not fr:
            return None, None, None, None, None
        dd = pd.concat(fr, ignore_index=True).sort_values("t").reset_index(drop=True)
        oi_ = dd["oi"].to_numpy()
        y_ = dd["out"].to_numpy() / oi_
        X_ = dd[ecols].to_numpy() / oi_[:, None]
        day_ = (dd["t"].to_numpy() // 86_400_000)
        ok_ = np.isfinite(y_) & np.isfinite(X_).all(axis=1) & (oi_ > 0)
        b_, se_, V_ = ols_cluster(X_[ok_], y_[ok_], day_[ok_])
        return b_, se_, V_, X_[ok_].mean(axis=0), dd

    print("--- EM 반복 — 비례배분 편향 제거 ---")
    print("  0회차는 비례배분(귀무가설). 이후 추정된 h 로 장부를 다시 굴린다.")
    print("  손익비가 1 에서 멀어지면 편향이 답을 지우고 있었다는 뜻이다.\n")
    print("  %5s | %10s %10s %8s | %10s" % ("회차", "손실 h", "이익 h", "손익비", "변화량"))
    h_cur = None
    prev = None
    for it in range(a.iters):
        b, se, V, w, d = fit(h_cur)
        if b is None:
            print("표본 없음")
            return 1
        lo_ = [j for j in range(nb) if EDGES[j + 1] <= 0]
        hi_ = [j for j in range(nb) if EDGES[j] >= 0]
        hl_ = float(np.average(b[lo_], weights=np.maximum(w[lo_], 1e-12)))
        hh_ = float(np.average(b[hi_], weights=np.maximum(w[hi_], 1e-12)))
        chg = np.nan if prev is None else float(np.abs(b - prev).sum()
                                                / max(np.abs(prev).sum(), 1e-12))
        print("  %5d | %10.5f %10.5f %8.2f | %10s"
              % (it, hl_, hh_, hl_ / hh_ if hh_ else np.nan,
                 "-" if prev is None else "%.4f" % chg))
        prev = b.copy()
        h_cur = np.maximum(b, 1e-6)          # 다음 회차 장부용 (음수는 바닥)
        if np.isfinite(chg) and chg < 0.01:
            print("  -> 수렴 (변화 1%% 미만)")
            break
    ok = np.ones(1, dtype=bool)              # 아래 출력에서 재사용하지 않음
    print("\n**사용 데이터 기간: %s ~ %s / %d종 / 구간 %d개**"
          % (str(pd.Timestamp(d.t.min(), unit="ms"))[:10],
             str(pd.Timestamp(d.t.max(), unit="ms"))[:10], d.symbol.nunique(), len(d)))

    print("\n--- 이탈 위험률 h(x) — 절편 없음, 일클러스터 CR1 ---")
    print("  귀무가설(비례배분): 모든 구간에서 h 가 **같다**.")
    print("  h 가 x 에 따라 기울면 그것이 지도의 정보다.\n")
    print("  %-10s %10s %10s %8s %12s"
          % ("수익률구간", "h(x)", "표준오차", "t", "평균노출비중"))
    for j in range(nb):
        print("  %-10s %10.5f %10.5f %8.1f %11.1f%%"
              % (LAB[j], b[j], se[j], b[j] / se[j] if se[j] > 0 else np.nan,
                 100 * w[j] / max(w.sum(), 1e-12)))
    hbar = float(np.average(b, weights=np.maximum(w, 0)))
    print("\n  노출가중 평균 h = %.5f (= 구간당 평균 이탈률)" % hbar)

    print("\n--- 손실 쪽 vs 이익 쪽 (설계의 핵심 비대칭) ---")
    # 0 을 품는 구간은 양쪽 모두에서 제외한다 (공선성 오염원)
    lo = [j for j in range(nb) if EDGES[j + 1] <= 0]
    hi = [j for j in range(nb) if EDGES[j] >= 0]
    print("  (0 을 품는 구간 '%s' 는 양쪽에서 제외)" % LAB[len(lo)])
    wl, wh = w[lo].sum(), w[hi].sum()
    hl = float(np.average(b[lo], weights=np.maximum(w[lo], 1e-12)))
    hh = float(np.average(b[hi], weights=np.maximum(w[hi], 1e-12)))
    print("  손실 구간 h = %.5f (노출 %.1f%%) | 이익 구간 h = %.5f (노출 %.1f%%)"
          % (hl, 100 * wl / w.sum(), hh, 100 * wh / w.sum()))
    print("  비 = **%.2f배**" % (hl / hh if hh else np.nan))
    c = np.zeros(nb)
    c[lo] = np.maximum(w[lo], 0) / max(wl, 1e-12)
    c[hi] = -np.maximum(w[hi], 0) / max(wh, 1e-12)
    v = float(c @ V @ c)
    est = float(c @ b)
    print("  차이 %.5f (t=%.1f)  — 유의한 양수면 **손실 쪽에서 더 많이 청산된다**"
          % (est, est / np.sqrt(v) if v > 0 else np.nan))

    print("\n--- 깊은 손실 구간 (강제청산이 몰릴 곳) ---")
    for j in range(4):
        print("  %-10s h=%.5f (t=%.1f) — 전체 평균의 **%.2f배**"
              % (LAB[j], b[j], b[j] / se[j] if se[j] > 0 else np.nan,
                 b[j] / hbar if hbar else np.nan))
    print("\n  *** 이 h(x) 가 지도의 커널이다. L(p) = Σ_cohort q * [h 적분] 으로 만든다.")
    print("      다음 단계 D-2: 격리분과 나머지의 계수를 분리한다.")
    out = os.path.join(C.DATA, "analysis", "map_kernel.parquet")
    pd.DataFrame({"bin": LAB, "edge_lo": EDGES[:-1], "edge_hi": EDGES[1:],
                  "h": b, "se": se, "expo_w": w}).to_parquet(out, index=False)
    print("      저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
