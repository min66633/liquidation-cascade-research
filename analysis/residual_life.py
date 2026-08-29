# -*- coding: utf-8 -*-
"""정적 지정가로 바닥을 잡을 수 있는가 — 평균잔여수명으로 판정한다.

문제 제기 (사용자 지적, 2026-08-05)
  "손절이 해로운 건 즉, 바닥을 맞추지 못했기 때문에 슈팅을 더 맞는 거잖아요.
   로직에 문제가 있는 거 아닌가요"

  맞다. 그리고 구조적이다.
      분위 지정가는 p_lim = p0*(1 - q_alpha) 에 건다.
      **체결 조건이 곧 X >= q_alpha 다.**
      즉 체결되는 순간은 언제나 '바닥이 내 주문보다 아래' 인 경우다.
      정의상 내려가는 도중에 잡히지, 바닥에서 잡힐 수 없다.
  손절이 어떤 폭이든 수익을 죽였던 것(two_leg.py: 60.1 -> 최대 18.7)은
  전략이 그 '체결 후 추가 밀림' 을 견뎌서 돈을 벌기 때문이다.

판정 방법 — 평균잔여수명 (mean residual life)
      m(u) = E[ X - u | X >= u ]
  u 를 깊게 할수록 m(u) 가
      **감소**하면  깊이 걸수록 바닥에 가까워진다 -> 정적 지정가로 바닥 포착 가능
      **증가**하면  깊이 걸수록 남은 밀림이 오히려 커진다 -> **정적으로는 불가능**
  후자는 두꺼운 꼬리의 정의다. 지수분포면 m(u) 는 상수(무기억).

  X 는 심볼·사건마다 규모가 다르므로 예측 q50 으로 정규화해서 본다: x = X / q50.

회전율 문제 (같은 사용자 지적)
  "괜히 길게 가져가면서 위험에 노출시키는 것보다 짧게 짧게 가져가며 회전율 극대화는?"
  보유 시간별로 (시도당bp / 분당bp / 최악1건 / 평균 동시보유) 를 같이 낸다.
  **자본이 제약인지부터 봐야 한다.** 신호가 드물면 회전율은 애초에 제약이 아니다.

실행:
    python analysis/residual_life.py
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
from analysis.horizon import sim, stat                                # noqa: E402
from analysis.response_liq import cmean                               # noqa: E402

HOLDS = [1, 2, 3, 5, 10, 15, 30, 60]


def main() -> int:
    ap = argparse.ArgumentParser(description="can a static limit ever catch the bottom")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    fm, ft = a.fee_maker, a.fee_taker

    print("=" * 116)
    print("정적 지정가로 바닥을 잡을 수 있는가 — 평균잔여수명 판정 + 회전율")
    print("=" * 116)
    d, win = build(syms, a.k, a.doi, a.gap)
    if d is None or len(d) < 300:
        print("이벤트 부족")
        return 1
    alphas = [0.5, 0.7, 0.9]
    Q, oos = walk_forward(d, alphas, col="X")
    dd = d[oos].reset_index(drop=True)
    ww = win[oos]
    Qo = {al: Q[al][oos] for al in alphas}
    X = dd["X"].to_numpy()
    q50 = Qo[0.5]
    yrs = (dd["t"].max() - dd["t"].min()) / (365.25 * 86_400_000)
    print("**사용 데이터 기간: %s ~ %s / %d종 / OOS %d건 / %.2f년**"
          % (str(pd.Timestamp(int(dd.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(dd.t.max()), unit="ms"))[:10],
             dd.symbol.nunique(), len(dd), yrs))

    print("\n" + "-" * 116)
    print("1. ★ 평균잔여수명 m(u) = E[X-u | X>=u] — 깊이 걸수록 바닥에 가까워지는가")
    print("-" * 116)
    print("  x = X / q50 (예측 밀림으로 정규화). u 를 깊게 하며 남은 밀림을 잰다.")
    print("  m(u) 가 **감소**하면 정적 지정가로 바닥 포착 가능, **증가**하면 불가능.\n")
    x = X / np.maximum(q50, 1e-9)
    print("  %-10s %8s %10s | %12s %12s %12s"
          % ("u (q50배)", "체결n", "체결률", "m(u) 배수", "m(u)/u", "중앙 잔여"))
    for u in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        m = x >= u
        if m.sum() < 20:
            print("  %-10.2f %8d %10s | %12s %12s %12s"
                  % (u, m.sum(), "-", "표본부족", "-", "-"))
            continue
        res = x[m] - u
        print("  %-10.2f %8d %10.3f | %12.3f %12.3f %12.3f"
              % (u, m.sum(), m.mean(), res.mean(), res.mean() / u, np.median(res)))
    print("\n  참고: 지수분포(무기억)면 m(u) 는 u 에 무관한 상수다.")
    print("        m(u) 가 u 에 비례해 커지면 **파레토(두꺼운 꼬리)** — 바닥이 계속 도망간다.")

    print("\n" + "-" * 116)
    print("2. 체결 직후 실제로 얼마나 더 밀리는가 — 바닥에서 샀다면 0 이어야 한다")
    print("-" * 116)
    print("  %-10s %8s | %10s %10s %10s %10s"
          % ("alpha", "체결n", "MAE중앙", "MAE p25", "MAE p10", "MAE 평균"))
    for al in alphas:
        qa = Qo[al]
        maes = []
        for i in range(len(dd)):
            if not np.isfinite(qa[i]):
                continue
            sd = int(dd["side"].iat[i])
            O, H, L, Cl = ww[i]
            p0 = float(O[0])
            p_lim = p0 * (1.0 - sd * qa[i] * 1e-4)
            dl = a.delta * 1e-4
            hit = (np.flatnonzero(L[:HMAX + 1] <= p_lim * (1.0 - dl)) if sd == 1
                   else np.flatnonzero(H[:HMAX + 1] >= p_lim * (1.0 + dl)))
            if not len(hit):
                continue
            fj = int(hit[0])
            p_in = (min(O[fj], p_lim) if sd == 1 else max(O[fj], p_lim)) if fj > 0 else p_lim
            seg = (L[fj:] if sd == 1 else H[fj:])
            mae = ((seg.min() / p_in - 1.0) if sd == 1
                   else (p_in / seg.max() - 1.0)) * 1e4
            maes.append(mae)
        maes = np.array(maes)
        print("  %-10.2f %8d | %10.1f %10.1f %10.1f %10.1f"
              % (al, len(maes), np.median(maes), np.percentile(maes, 25),
                 np.percentile(maes, 10), maes.mean()))
    print("  ** 바닥을 맞혔다면 0 에 가까워야 한다. 크게 음수면 '내려가는 도중에 잡힌 것'. **")

    print("\n" + "-" * 116)
    print("3. 회전율 — 짧게 가져가면 자본효율이 오르는가, 애초에 자본이 제약인가")
    print("-" * 116)
    print("  W=60 고정, alpha=0.90, 손절 없음. 보유 시간만 바꾼다.\n")
    print("  %-8s | %9s %8s | %10s %9s | %9s %10s | %12s"
          % ("보유(분)", "시도당bp", "샤프", "연간 총bp", "분당bp", "최악1건",
             "최대낙폭", "평균 동시보유"))
    mins_per_yr = 365.25 * 24 * 60
    for e in HOLDS:
        s = stat(sim(dd, ww, Qo[0.9], 60, e, a.delta, fm, ft))
        nfill = s["fill"] * s["n"]
        expo = nfill * e                       # 총 노출 분
        tot_yr = s["bp"] * s["n"] / yrs
        print("  %-8d | %9.1f %8.2f | %10.0f %9.2f | %9.0f %10.0f | %12.4f"
              % (e, s["bp"], s["sharpe"], tot_yr,
                 (s["bp"] * s["n"]) / max(expo, 1e-9), s["worst"], s["maxdd"],
                 expo / (yrs * mins_per_yr)))
    print("\n  ** 평균 동시보유가 1 보다 훨씬 작으면 자본은 제약이 아니다 —")
    print("     신호가 드물어 대부분의 시간을 놀고 있다는 뜻이고, 회전율을 올릴 여지가 없다.")
    print("     그때 짧은 보유의 이득은 자본효율이 아니라 **사고 노출 시간 축소**뿐이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
