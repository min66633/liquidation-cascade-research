# -*- coding: utf-8 -*-
"""과최적화 검정 — "좋은 결과만 골라 본 것 아닌가" 에 대한 직접 답.

사용자 (2026-08-06)
  "그렇게 단순한 조건으로 되는건 애초에 우리가 데이터를 보고 좋은 결과만
   봐서 그런거 아닌가요?"

이 지적은 타당하다. §6.19 의 설정(D=15분, X=300bp, 상한 15분)은 **같은 데이터에서
격자를 뒤져 고른 칸**이다. 이 스크립트는 그 우려를 세 가지로 검정한다.

  1. **표면이 고원인가 뾰족한 칸인가** — 인접 칸이 같이 좋으면 우연이 아니다
  2. **시점 분할 선택** — 전반부에서만 골라 후반부에서 **단 한 번** 평가.
     그리고 확장창 워크포워드로 연도별 OOS 를 낸다. 이것이 핵심 검정이다.
  3. **다중검정 보정** — 뒤진 칸 수 하에서 필요한 t 임계값

고칠 수 없는 편향 (반드시 같이 읽을 것)
  - 심볼 21종은 **2026년 시점의 메이저**다. 상장폐지·쇠퇴 종목이 없다.
    생존자 편향이고 이 데이터로는 교정 불가능하다.
  - 전 구간이 하나의 시장 체제(2020~2026 암호화폐)다. 체제 밖 일반화는 미검증.

실행:
    python analysis/overfit.py
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
from analysis.response_liq import cmean                                # noqa: E402
from analysis.simple_bottom import prep, events, frame, NOBS           # noqa: E402

W = 116
DS = (5, 15, 30, 60)
XS = (100, 150, 200, 300, 500, 800)
HS = (5, 10, 15, 30, 60, 120)
COST = 10.0
MS = 86_400_000


def conc(t_ms, hold):
    e = np.concatenate([t_ms, t_ms + hold * 60_000])
    dl = np.concatenate([np.ones(len(t_ms)), -np.ones(len(t_ms))])
    o = np.lexsort((dl, e))
    cur = np.cumsum(dl[o])
    return max(int(cur.max()), 1), max(float(np.percentile(cur, 99)), 1.0)


def cell(d, CL, h, mask=None):
    """한 칸의 성적. mask 로 기간을 자른다."""
    if mask is None:
        mask = np.ones(len(d), dtype=bool)
    if mask.sum() < 50:
        return None
    r = CL[mask, h] - COST
    day = d["day"].to_numpy()[mask]
    t_ms = d["t"].to_numpy()[mask]
    m, se, t, _ = cmean(r, day)
    yrs = (t_ms.max() - t_ms.min()) / (365.25 * MS)
    if yrs <= 0:
        return None
    M, p99 = conc(t_ms, h)
    yr_bp = m * len(r) / yrs
    return {"n": int(mask.sum()), "bp": m, "t": t, "yrs": yrs,
            "per_yr": len(r) / yrs, "yr_bp": yr_bp,
            "M": M, "p99": p99, "cap": yr_bp / M, "cap99": yr_bp / p99,
            "win": float((r > 0).mean())}


def build(P):
    """(D,X) 별로 사건과 경로를 한 번만 만든다."""
    G = {}
    for D in DS:
        for X in XS:
            ev = events(P, D, X, 5, -1)
            if len(ev) < 200:
                continue
            d, LO, CL, HI = frame(P, ev, -1)
            G[(D, X)] = (d, CL)
            U.log("  (D=%d, X=%d) n=%d" % (D, X, len(d)))
    return G


def sec(n, t):
    print("\n" + "-" * W)
    print("%d. %s" % (n, t))
    print("-" * W)


def main() -> int:
    ap = argparse.ArgumentParser(description="overfitting check")
    ap.add_argument("--symbols", nargs="*", default=None)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    P = prep(syms)
    lo = min(int(v[0][0]) for v in P.values())
    hi = max(int(v[0][-1]) for v in P.values())

    print("=" * W)
    print("과최적화 검정 — '좋은 결과만 골라 본 것 아닌가'")
    print("=" * W)
    print("데이터: 21종 1분봉 %s ~ %s"
          % (pd.to_datetime(lo, unit="ms").strftime("%Y-%m-%d"),
             pd.to_datetime(hi, unit="ms").strftime("%Y-%m-%d")))
    print("격자: D %s x X %s x 상한 %s = **%d 칸**"
          % (DS, XS, HS, len(DS) * len(XS) * len(HS)))
    print("선택 기준: 자본정규화 (연간총bp / 최대동시). 경제적 목적함수와 일치시킨다.")
    U.log("격자 구축")
    G = build(P)
    print("\n실제 유효 칸: %d 개 (D,X 조합) x %d 상한 = %d"
          % (len(G), len(HS), len(G) * len(HS)))

    sec(1, "표면 — 고원인가 뾰족한 한 칸인가 (상한 15분 고정, 건당bp / t)")
    print("  %-8s | %s" % ("D분", " ".join("%-16s" % ("%dbp" % x) for x in XS)))
    for D in DS:
        out = []
        for X in XS:
            if (D, X) not in G:
                out.append("%-16s" % "-")
                continue
            d, CL = G[(D, X)]
            s = cell(d, CL, 15)
            out.append("%-16s" % ("-" if s is None
                                  else "%6.1f (%4.1f)" % (s["bp"], s["t"])))
        print("  D=%-6d | %s" % (D, " ".join(out)))
    print("\n  ** 인접 칸이 같이 양수면 고원이다. 한 칸만 튀면 우연이다. **")

    print("\n  전 격자 요약 (상한 전부 포함):")
    allc = []
    for (D, X), (d, CL) in G.items():
        for h in HS:
            s = cell(d, CL, h)
            if s:
                allc.append((D, X, h, s))
    ts = np.array([s["t"] for _, _, _, s in allc])
    print("  칸 %d개 | t>0 인 칸 %d (%.0f%%) | t>2 인 칸 %d (%.0f%%) | t 중앙 %.1f | t 최대 %.1f"
          % (len(allc), (ts > 0).sum(), 100 * (ts > 0).mean(),
             (ts > 2).sum(), 100 * (ts > 2).mean(), np.median(ts), ts.max()))

    sec(2, "★★ 시점 분할 — 전반부에서만 고르고 후반부에서 단 한 번 평가")
    mid = lo + (hi - lo) // 2
    print("  훈련: %s ~ %s | 검정: %s ~ %s\n"
          % (pd.to_datetime(lo, unit="ms").strftime("%Y-%m"),
             pd.to_datetime(mid, unit="ms").strftime("%Y-%m"),
             pd.to_datetime(mid, unit="ms").strftime("%Y-%m"),
             pd.to_datetime(hi, unit="ms").strftime("%Y-%m")))
    best, brec = None, None
    for (D, X), (d, CL) in G.items():
        tt = d["t"].to_numpy()
        for h in HS:
            s = cell(d, CL, h, tt < mid)
            if s and (best is None or s["cap"] > best["cap"]):
                best, brec = s, (D, X, h)
    D, X, h = brec
    d, CL = G[(D, X)]
    tt = d["t"].to_numpy()
    tr = cell(d, CL, h, tt < mid)
    te = cell(d, CL, h, tt >= mid)
    print("  훈련에서 고른 칸: **D=%d분, X=%dbp, 상한 %d분**" % (D, X, h))
    print("  %-10s | %6s %7s | %8s %5s | %6s | %9s %8s %9s"
          % ("구간", "n", "연간", "건당bp", "t", "승률", "연간총bp", "최대동시", "자본정규"))
    for lab, s in (("훈련", tr), ("**검정**", te)):
        print("  %-10s | %6d %7.0f | %8.1f %5.1f | %5.1f%% | %9.0f %8d %9.0f"
              % (lab, s["n"], s["per_yr"], s["bp"], s["t"], 100 * s["win"],
                 s["yr_bp"], s["M"], s["cap"]))
    # 검정 구간에서 이 칸의 순위
    rank = []
    for (D2, X2), (d2, CL2) in G.items():
        t2 = d2["t"].to_numpy()
        for h2 in HS:
            s2 = cell(d2, CL2, h2, t2 >= mid)
            if s2:
                rank.append(((D2, X2, h2), s2["cap"]))
    rank.sort(key=lambda z: -z[1])
    pos = [i for i, (k, _) in enumerate(rank) if k == brec]
    print("\n  검정 구간에서 이 칸의 자본정규화 순위: **%d위 / %d칸**"
          % (pos[0] + 1 if pos else -1, len(rank)))
    print("  검정 구간 1위 칸: D=%d X=%d 상한%d (자본정규 %.0f)"
          % (*rank[0][0], rank[0][1]))
    print("\n  ** 검정 순위가 상위권이면 표면이 실재한다. 하위권이면 훈련 적합이다. **")

    print("\n  역방향 (후반부에서 고르고 전반부에서 평가):")
    best2, brec2 = None, None
    for (D2, X2), (d2, CL2) in G.items():
        t2 = d2["t"].to_numpy()
        for h2 in HS:
            s2 = cell(d2, CL2, h2, t2 >= mid)
            if s2 and (best2 is None or s2["cap"] > best2["cap"]):
                best2, brec2 = s2, (D2, X2, h2)
    D2, X2, h2 = brec2
    d2, CL2 = G[(D2, X2)]
    t2 = d2["t"].to_numpy()
    s_oos = cell(d2, CL2, h2, t2 < mid)
    print("  후반부에서 고른 칸: D=%d분, X=%dbp, 상한 %d분 -> 전반부 성적: "
          "건당 %.1fbp (t=%.1f), 자본정규 %.0f"
          % (D2, X2, h2, s_oos["bp"], s_oos["t"], s_oos["cap"]))

    sec(3, "★★ 확장창 워크포워드 — 매년 과거만 보고 고르고 다음 해에 평가")
    print("  훈련 최소 2년. 매년 전 격자를 다시 뒤져 자본정규화 최고 칸을 고른다.")
    print("  그 칸을 **그 해에만** 적용한다. 이것이 실제 운용에 가장 가깝다.\n")
    print("  %-6s | %-22s | %6s %8s %5s | %6s | %9s"
          % ("검정연도", "그 해에 고른 칸", "n", "건당bp", "t", "승률", "자본정규"))
    oos_r, oos_d, oos_cap = [], [], []
    for Y in (2023, 2024, 2025, 2026):
        y0 = int(pd.Timestamp("%d-01-01" % Y).value // 10**6)
        y1 = int(pd.Timestamp("%d-01-01" % (Y + 1)).value // 10**6)
        bb, bk = None, None
        for (D3, X3), (d3, CL3) in G.items():
            t3 = d3["t"].to_numpy()
            s3 = cell(d3, CL3, HS[0], t3 < y0)      # placeholder, 아래서 갱신
            for h3 in HS:
                s3 = cell(d3, CL3, h3, t3 < y0)
                if s3 and s3["yrs"] >= 2 and (bb is None or s3["cap"] > bb["cap"]):
                    bb, bk = s3, (D3, X3, h3)
        if bk is None:
            continue
        D3, X3, h3 = bk
        d3, CL3 = G[(D3, X3)]
        t3 = d3["t"].to_numpy()
        m3 = (t3 >= y0) & (t3 < y1)
        s3 = cell(d3, CL3, h3, m3)
        if s3 is None:
            print("  %-6d | %-22s | (표본부족)" % (Y, "D=%d X=%d 상한%d" % bk))
            continue
        r3 = CL3[m3, h3] - COST
        oos_r.append(r3)
        oos_d.append(d3["day"].to_numpy()[m3])
        oos_cap.append(s3["cap"])
        print("  %-6d | %-22s | %6d %8.1f %5.1f | %5.1f%% | %9.0f"
              % (Y, "D=%d X=%d 상한%d" % bk, s3["n"], s3["bp"], s3["t"],
                 100 * s3["win"], s3["cap"]))
    if oos_r:
        rr = np.concatenate(oos_r)
        dd = np.concatenate(oos_d)
        m, se, t, _ = cmean(rr, dd)
        print("\n  ** 워크포워드 OOS 전체: n=%d, 건당 %.1fbp, t=%.1f, 승률 %.1f%% **"
              % (len(rr), m, t, 100 * (rr > 0).mean()))
        print("     양수인 해: %d/%d | 자본정규 중앙 %.0f"
              % (sum(1 for x in oos_cap if x > 0), len(oos_cap), np.median(oos_cap)))

    sec(4, "다중검정 보정 — 이 정도 뒤졌으면 t 가 얼마여야 하나")
    N = len(allc)
    print("  이 스크립트가 명시적으로 평가한 칸: %d" % N)
    print("  이번 세션 전체(fast_trigger/asymmetry/robust/simple_bottom)에서 같은 데이터로")
    print("  돌린 설정은 어림 **300개 이상**이다. 보수적으로 N=300 으로 잡는다.\n")
    from math import log, sqrt
    for NN in (N, 300, 1000):
        thr = sqrt(2 * log(NN))
        print("  N=%-5d -> 독립 가정 시 귀무 하 최대 t 기대값 ~ %.2f (5%% 임계 ~%.2f)"
              % (NN, thr, sqrt(2 * log(NN / 0.05))))
    print("\n  ** 칸들은 독립이 아니므로(격자가 겹친다) 실제 임계값은 이보다 낮다.")
    print("     그래도 t 가 3~4 를 못 넘으면 탐색의 부산물로 봐야 한다. **")

    sec(5, "고칠 수 없는 편향 — 데이터로 답할 수 없는 것")
    print("  1. **생존자 편향.** 심볼 21종은 2026년 시점의 메이저다. 이 기간에")
    print("     상장폐지되거나 쇠퇴한 코인이 표본에 없다. 급락 후 반등하지 못하고")
    print("     그대로 사라진 종목들이 빠져 있다는 뜻이다. **이 데이터로는 교정 불가.**")
    print("     교정하려면 당시 상장 전체 목록으로 다시 받아야 한다.")
    print("  2. **단일 체제.** 2020~2026 암호화폐 한 체제뿐이다. 6년이지만")
    print("     독립 사건 수는 그보다 훨씬 적다 (같은 날 여러 심볼이 같이 급락한다).")
    print("  3. **비용·용량 가정.** 왕복 10bp 는 수수료만이다. 캐스케이드 직후")
    print("     얇은 알트에 시장가로 들어가는 충격은 안 들어 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
