# -*- coding: utf-8 -*-
"""D-1b — f_R 추정: **강제청산이 어디서 일어나는가**. 레버리지가 정한다.

설계상의 위치 (DESIGN_LOCK.md §2)
    L(p,t) = Σ_τ ΔOI+(τ) · f_R(p/p(τ)) · S(t-τ)
  f_R = **강제청산 위치**. 행동이 아니라 레버리지가 정한다. ← 이 스크립트
  S   = 생존. 자발 청산이 관여한다.                        ← D-1a 완료
  ΔOI+= 진입 시점·가격.

왜 집계 OI 에서 뽑으면 안 되나
  집계 OI 감소의 **약 99%가 자발 청산**이다. 거기서 추정한 h(x) 는 S 이지
  f_R 이 아니다. 강제청산은 선택이 아니므로 행동 함수로 잡히지 않는다.
  f_R 은 **포지션의 청산가를 직접 관측**해야 나온다 — HL 격리 포지션이 그것을 준다.

교차검증 (이 스크립트의 핵심)
  D-1a 의 h(x) 는 x < -30% 에서 4.04배(t=8.6)로 튀었다. 그것이 '행동' 이 아니라
  **강제청산이 섞여 들어온 흔적** 이라면, **완전히 다른 데이터**인 HL 청산가 분포가
  같은 자리에 질량을 가져야 한다. 맞으면 강한 독립 증거다.

한계
  HL 은 BTC perp OI 의 약 15% 이고 실현 청산가의 70% 가 지도에 없다(Q1).
  그러나 **f_R 은 커버리지가 아니라 '레버리지 분포의 모양'** 이므로, 관측된 계좌가
  전체를 대표하는지가 관건이다. 대표성은 별도 문제로 표시한다.

실행:
    python analysis/fr_kernel.py
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
from analysis.liq_distance import load_positions                # noqa: E402
from analysis.map_kernel import EDGES, LAB                       # noqa: E402


def wdens(v, w, edges):
    """가중 밀도 — 각 구간에 들어가는 position_value 비중."""
    idx = np.digitize(v, edges[1:-1])
    tot = np.bincount(idx, weights=w, minlength=len(edges) - 1)[:len(edges) - 1]
    return tot / max(tot.sum(), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description="D-1b f_R from leverage")
    ap.add_argument("--min-lev", type=float, default=1.0)
    a = ap.parse_args()
    U.init_stdout()

    print("=" * 80)
    print("D-1b — f_R: 강제청산 위치 커널. 레버리지가 정한다")
    print("=" * 80)
    d = load_positions()
    d = d[d["dir_ok"]]
    iso = d[d["lev_type"] == "isolated"].copy()
    crs = d[d["lev_type"] == "cross"].copy()
    print("**사용 데이터 기간: %s ~ %s**"
          % (str(d.ts_dt.min())[:10], str(d.ts_dt.max())[:10]))
    print("격리 포지션 %d행 (명목 $%.4g) | 교차 %d행 (명목 $%.4g)"
          % (len(iso), iso.position_value.sum(), len(crs), crs.position_value.sum()))
    print("*** f_R 은 **격리**에서만 추정 가능하다 — 교차는 계좌 전체 담보에 의존해")
    print("    외부에서 청산가를 계산할 수 없다 (DESIGN_LOCK §2.4).\n")

    # *** 축 정의 (2026-08-04 정정) ***
    # h(x) 의 x 는 **포지션 자신의 수익률**이다(롱은 +가격상승, 숏은 +가격하락).
    # 청산은 **항상 손실 쪽**에서 일어난다 — 롱은 가격이 내려서, 숏은 올라서.
    # 따라서 f_R 은 전부 **음수 쪽**이어야 한다.
    # 첫 판에서 숏을 '가격 방향' 으로 +dist 에 놓아 36%가 양수 쪽에 찍혔다. 오류였다.
    x = -iso["dist"].to_numpy()
    w = iso["position_value"].to_numpy()

    print("--- f_R — 진입가 대비 청산거리 분포 (명목 가중) ---")
    print("  x = 포지션 **자신의 수익률**. 청산은 항상 손실 쪽이므로 전부 음수다.\n")
    fr = wdens(x, w, EDGES)
    print("  %-14s %12s %14s" % ("구간", "f_R 비중", "누적(손실쪽부터)"))
    cum = 0.0
    for j in range(len(EDGES) - 1):
        cum += fr[j]
        print("  %-14s %11.2f%% %13.2f%%" % (LAB[j], 100 * fr[j], 100 * cum))

    print("\n--- 레버리지 환산 (근사: dist ≈ 1/leverage) ---")
    lev = 1.0 / np.maximum(iso["dist"].to_numpy(), 1e-6)
    o = np.argsort(lev)
    cw = np.cumsum(w[o]) / max(w.sum(), 1e-12)
    print("  명목가중 레버리지 분위:")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print("    p%02d = %.1f배" % (100 * q, float(np.interp(q, cw, lev[o]))))

    print("\n" + "=" * 80)
    print("교차검증 — h(x) 의 스파이크가 f_R 과 같은 자리인가")
    print("=" * 80)
    kp = os.path.join(C.DATA, "analysis", "map_kernel.parquet")
    if not os.path.exists(kp):
        print("  map_kernel.parquet 없음 — D-1a 를 먼저 돌릴 것")
        return 0
    mk = pd.read_parquet(kp)
    h = mk["h"].to_numpy()
    hw = mk["expo_w"].to_numpy()
    hbar = float(np.average(h, weights=np.maximum(hw, 0)))
    print("  h(x) 는 집계 OI(=자발 99%%)에서, f_R 은 HL 청산가에서. **완전 독립 데이터.**\n")
    print("  %-14s %10s %10s %12s" % ("구간", "h/평균", "f_R 비중", "판정"))
    for j in range(len(EDGES) - 1):
        ratio = h[j] / hbar if hbar else np.nan
        mark = ""
        if ratio > 1.5 and fr[j] > 0.05:
            mark = "**둘 다 높음**"
        elif ratio > 1.5:
            mark = "h만 높음"
        elif fr[j] > 0.05:
            mark = "f_R만 높음"
        print("  %-14s %10.2f %9.2f%% %12s" % (LAB[j], ratio, 100 * fr[j], mark))

    lo = [j for j in range(len(EDGES) - 1) if EDGES[j + 1] <= 0]
    if len(lo) > 2:
        r = np.corrcoef(h[lo] / hbar, fr[lo])[0, 1]
        print("\n  손실 구간에서 h/평균 과 f_R 의 상관 = **%.3f**" % r)
        print("  양의 상관이면 h 의 스파이크가 **강제청산 위치와 겹친다**는 뜻이다.")
        print("  -> 그러면 x<-30%% 의 4.04배는 행동이 아니라 f_R 이 섞인 것이고,")
        print("     f_R 을 분리해내야 S 가 깨끗해진다.")

    out = os.path.join(C.DATA, "analysis", "fr_kernel.parquet")
    pd.DataFrame({"bin": LAB, "edge_lo": EDGES[:-1], "edge_hi": EDGES[1:],
                  "f_R": fr}).to_parquet(out, index=False)
    print("\n  저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
