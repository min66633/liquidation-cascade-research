# -*- coding: utf-8 -*-
"""완전예지 상한 — 캐스케이드 창 안에서 '꺾이는 점을 다 안다면' 얼마인가.

왜 필요한가
  "중간중간 꺾이는 걸 예측하면 단순보유보다 낫다" 는 **자명하게 참**이다.
  진짜 질문은 **얼마나 남아 있느냐** 다. 완전예지 상한이 단순보유와 비슷하면
  회전 전략을 정교화할 이유가 없고, 훨씬 크면 그 격차가 추격할 가치의 크기다.
  R-6 은 기계적 규칙 100조합만 훑었을 뿐 **상한을 재지 않았다.**

무엇을 재나 (1분봉, 이벤트 창 240분)
  (a) 단순보유 15분 / 240분                  — R-6 의 최선
  (b) 완전예지 **1회** 매매 (최저점 매수 -> 이후 최고점 매도)
  (c) 완전예지 **무제한** 매매 (거래비용 포함 최적 DP)
  (d) (c) 의 최적 거래 횟수

  (c) 는 표준 DP:
      hold[t] = max(hold[t-1],  free[t-1] - p[t])
      free[t] = max(free[t-1],  hold[t-1] + p[t] - c)
  비용 c 를 넣으면 잔파동을 잡는 것이 손해가 되는 지점에서 자동으로 멈춘다.

*** 이것은 상한이지 전략이 아니다. 어떤 예측기도 여기 도달할 수 없다. ***

실행:
    python analysis/oracle.py
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
from analysis.response_liq import cmean          # noqa: E402
from analysis.turnover import build              # noqa: E402


def oracle_multi(p: np.ndarray, cost: float):
    """무제한 매매 최대 수익(비용 포함). 반환 (수익 비율, 거래 수)."""
    hold, free = -p[0], 0.0
    nh, nf = 0, 0
    for t in range(1, len(p)):
        cand = free - p[t]
        if cand > hold:
            hold, nh = cand, nf
        cand = hold + p[t] - cost
        if cand > free:
            free, nf = cand, nh + 1
    return free, nf


def main() -> int:
    ap = argparse.ArgumentParser(description="perfect-foresight upper bound")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--cost", type=float, default=4.0, help="왕복 비용 bp")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 76)
    print("완전예지 상한 — 꺾이는 점을 전부 안다면 얼마인가")
    print("=" * 76)
    ws = build(syms, a.window)
    if not ws:
        print("창 없음")
        return 1
    days = np.array([w["day"] for w in ws])
    print("이벤트 창 %d개 / 창 %d분 / 왕복비용 %.1fbp\n" % (len(ws), a.window, a.cost))

    hold15, hold240, one, multi, ntr = [], [], [], [], []
    for w in ws:
        A = w["arr"]
        s0, s1, sd = w["s0"], w["s1"], w["side"]
        c = A["close"][s0:s1].astype(np.float64)
        p0 = c[0]
        # side=-1 이면 가격을 뒤집어 롱 문제로 만든다
        p = (c / p0) if sd == 1 else (2.0 - c / p0)
        hold15.append((p[min(15, len(p) - 1)] - p[0]) * 1e4 - a.cost)
        hold240.append((p[-1] - p[0]) * 1e4 - a.cost)
        # (b) 완전예지 1회: 최저점 이후 최고점
        run_min = np.minimum.accumulate(p)
        one.append(float(np.max(p - run_min)) * 1e4 - a.cost)
        # (c) 완전예지 무제한
        g, n = oracle_multi(p, a.cost / 1e4)
        multi.append(g * 1e4)
        ntr.append(n)

    rows = [("단순보유 15분", hold15), ("단순보유 240분", hold240),
            ("완전예지 1회", one), ("완전예지 무제한", multi)]
    print("  %-18s %10s %10s %10s %10s"
          % ("전략", "이벤트당", "t", "중앙", "p90"))
    base = None
    for lab, v in rows:
        x = np.array(v, dtype=np.float64)
        m, se, t, _ = cmean(x, days)
        if base is None:
            base = m
        print("  %-18s %10.1f %10.1f %10.1f %10.1f"
              % (lab, m, t, np.median(x), np.quantile(x, .9)))
    nt = np.array(ntr)
    print("\n  완전예지 무제한의 최적 거래 수: 중앙 %.0f | 평균 %.1f | p90 %.0f | 최대 %d"
          % (np.median(nt), nt.mean(), np.quantile(nt, .9), nt.max()))

    mh = float(np.mean(hold240))
    mm = float(np.mean(multi))
    mo = float(np.mean(one))
    print("\n  상한 / 단순보유(240분) = **%.1f배**" % (mm / mh if mh else np.nan))
    print("  상한 / 단순보유(15분)  = %.1f배" % (mm / np.mean(hold15)))
    print("  1회 완전예지 / 단순보유(240분) = %.1f배" % (mo / mh if mh else np.nan))

    print("\n  [비용 민감도] 잔파동은 비용에 먼저 죽는다")
    print("  %10s %14s %12s" % ("왕복비용bp", "완전예지 무제한", "최적 거래수"))
    for cst in (0.0, 2.0, 4.0, 10.0, 20.0):
        g_, n_ = [], []
        for w in ws:
            A = w["arr"]
            s0, s1, sd = w["s0"], w["s1"], w["side"]
            c = A["close"][s0:s1].astype(np.float64)
            p = (c / c[0]) if sd == 1 else (2.0 - c / c[0])
            g, n = oracle_multi(p, cst / 1e4)
            g_.append(g * 1e4)
            n_.append(n)
        print("  %10.1f %14.1f %12.1f"
              % (cst, float(np.mean(g_)), float(np.mean(n_))))

    print("\n  *** 상한은 도달 불가능하다. R-6 의 최선 기계적 규칙(37.5bp)과")
    print("      단순보유(50.4bp)가 이 상한의 몇 %% 인지가 '남은 여지' 다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
