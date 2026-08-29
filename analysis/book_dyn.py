# -*- coding: utf-8 -*-
"""오더북 특징을 넣으면 달라지는가 — 정적·동적 둘 다, 넣고/빼고 대조.

이 파일이 답하려는 것
  지금까지 예측 특징이 전부 5분봉 파생(OI·포지션비·거래량)이었다. 설계의
  ②오더북 깊이와 ③유입·취소가 빠져 있었다. book_feat.py 로 뽑은
      ldep    압력받는 쪽 +-1% 깊이 / 과거 1일 평균   -> ②
      limb    불균형
      lslope  +-5% / +-1% 깊이 프로파일
      ddep5   5분간 깊이 변화                        -> ③ 회복력·취소압
      ddep30  30분간 깊이 변화
  를 넣으면
    (가) 밀림 폭 X 의 예측이 좋아지는가 (정적 진입)
    (나) '남은 밀림' R_u 의 예측이 좋아지는가 (동적 진입)  <- 여기가 핵심
    (다) 결국 시도당 수익이 좋아지는가
  를 **같은 표본에서** 대조한다.

  (나)가 이 연구의 핵심 주장이다: 깊이가 다시 차오르는 것은 가격이 반등하기
  **전에** 보인다. 그렇다면 R_u 예측에서 오더북이 가격보다 앞서야 한다.

  ** 비교는 오더북 특징이 **있는 행만** 으로 한다. 없는 행을 한쪽에만 주면
     표본이 달라져 대조가 성립하지 않는다. **

실행:
    python analysis/book_dyn.py
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
from analysis.prob_entry import build, walk_forward, FEAT               # noqa: E402
from analysis.dyn_entry import (panel, wf_panel, sim_dyn, stat, head, line,
                                GFEAT, WMAX)                            # noqa: E402
from analysis.book_feat import BFEAT, CACHE                             # noqa: E402


def attach(P: pd.DataFrame, bf: pd.DataFrame) -> pd.DataFrame:
    """(symbol, t, u) 로 오더북 특징을 붙인다."""
    b = bf.drop_duplicates(["symbol", "t", "u"], keep="last")
    m = P.merge(b[["symbol", "t", "u"] + BFEAT], on=["symbol", "t", "u"], how="left")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="does the orderbook add anything")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--wmax", type=int, default=WMAX)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    fm, ft = a.fee_maker, a.fee_taker

    print("=" * 124)
    print("오더북을 넣으면 달라지는가 — 정적·동적, 같은 표본에서 넣고/빼고 대조")
    print("=" * 124)
    if not os.path.exists(a.cache):
        print("book_feat 캐시 없음: %s  (먼저 python analysis/book_feat.py)" % a.cache)
        return 1
    bf = pd.read_parquet(a.cache)
    d, win = build(syms, a.k, a.doi, a.gap)
    if d is None or len(d) < 300:
        print("이벤트 부족")
        return 1

    # ---------- (가) 정적: 밀림 폭 X ----------
    b0 = bf[bf.u == 0].drop_duplicates(["symbol", "t"], keep="last")
    d = d.merge(b0[["symbol", "t"] + BFEAT], on=["symbol", "t"], how="left")
    has = np.isfinite(d[BFEAT].to_numpy(dtype=np.float64)).all(axis=1)
    print("**사용 데이터 기간: %s ~ %s / %d종 / 사건 %d건 / 오더북 있는 사건 %d (%.1f%%)**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d), int(has.sum()), 100 * has.mean()))
    # **오더북이 있는 사건만** 남겨야 두 모형의 표본이 같다
    keep = np.flatnonzero(has)
    d = d.iloc[keep].reset_index(drop=True)
    win = win[keep]
    print("대조 표본 %d건으로 제한" % len(d))

    alphas = [0.5, 0.9]
    print("\n" + "-" * 124)
    print("1. 정적 — 밀림 폭 X 예측에 오더북이 기여하는가")
    print("-" * 124)
    res = {}
    for lab, fl in (("가격·OI 만", FEAT), ("**+ 오더북**", FEAT + BFEAT)):
        Q, oos = walk_forward(d, alphas, col="X", feats=fl)
        res[lab] = (Q, oos)
        Xo = d["X"].to_numpy()[oos]
        cal = " ".join("a%.2f 위반%.3f(편차%+.3f)"
                       % (al, (Xo < Q[al][oos]).mean(), (Xo < Q[al][oos]).mean() - al)
                       for al in alphas)
        # 설명력 (OOS): 예측 중앙 대비 로그 절대오차
        m50 = Q[0.5][oos]
        err = np.abs(np.log(np.maximum(Xo, 1e-9)) - np.log(np.maximum(m50, 1e-9)))
        print("  %-14s OOS %4d | 중앙절대로그오차 **%.4f** | %s"
              % (lab, int(oos.sum()), float(np.median(err)), cal))

    print("\n" + "-" * 124)
    print("2. ★ 동적 — '남은 밀림' R_u 예측에 오더북이 기여하는가 (핵심)")
    print("-" * 124)
    scale = d["sig5"].to_numpy(dtype=np.float64) * 1e4
    P = panel(d, win, scale, a.wmax)
    P = attach(P, bf)
    okb = np.isfinite(P[BFEAT].to_numpy(dtype=np.float64)).all(axis=1)
    P = P[okb].reset_index(drop=True)
    print("  패널 %d행 (오더북 있는 행만) / 사건 %d개\n" % (len(P), P.ei.nunique()))
    alr = [0.10, 0.30, 0.50, 0.70, 0.90]
    dyn = {}
    for lab, fl in (("가격만", GFEAT), ("**+ 오더북**", GFEAT + BFEAT)):
        Q, oos = wf_panel(P, alr, feats=fl, min_ev=min(200, max(50, P.ei.nunique() // 4)))
        R = P["R"].to_numpy()
        cal = " ".join("a%.2f 편차%+.3f" % (al, (R[oos] < Q[al][oos]).mean() - al)
                       for al in (0.10, 0.50, 0.90))
        err = np.abs(np.log1p(R[oos]) - np.log1p(Q[0.5][oos]))
        print("  %-14s OOS %6d행 | 중앙절대로그오차 **%.4f** | %s"
              % (lab, int(oos.sum()), float(np.median(err)), cal))
        dyn[lab] = (Q, oos)

    print("\n" + "-" * 124)
    print("2b. ★ 하나씩 넣어본다 — 악화가 '신호 부재' 인가 '모수 증가' 인가")
    print("-" * 124)
    print("  5개를 한꺼번에 넣으면 계수가 17->22 개가 된다. 훈련 표본이 수백 건이라")
    print("  신호가 있어도 분산 때문에 나빠질 수 있다. 하나씩 넣어 갈라낸다.\n")
    R = P["R"].to_numpy()
    base_err = None
    print("  %-22s %8s %14s %10s" % ("특징", "계수 수", "중앙절대오차", "개선"))
    for lab, fl in ([("가격만 (기준)", GFEAT)]
                    + [("+ " + c, GFEAT + [c]) for c in BFEAT]
                    + [("+ 전부", GFEAT + BFEAT)]):
        Q, oos = wf_panel(P, [0.5], feats=fl,
                          min_ev=min(200, max(50, P.ei.nunique() // 4)))
        if oos.sum() < 1000:
            print("  %-22s %8d  표본부족" % (lab, len(fl)))
            continue
        err = float(np.median(np.abs(np.log1p(R[oos]) - np.log1p(Q[0.5][oos]))))
        if base_err is None:
            base_err = err
        print("  %-22s %8d %14.4f %+10.4f"
              % (lab, len(fl) + 1, err, base_err - err))
    print("  ** 개선이 양수인 특징이 하나도 없으면 신호 자체가 없다는 뜻이다. **")

    print("\n" + "-" * 124)
    print("3. 결국 돈이 되는가 — 같은 표본, 같은 보유(%d분)" % a.hold)
    print("-" * 124)
    head()
    # 정적 대조군
    for lab in ("가격·OI 만", "**+ 오더북**"):
        Q, oos = res[lab]
        q90 = np.where(oos, Q[0.9], np.nan)
        # 정적 시뮬은 사건 인덱스 기준이므로 패널의 ei 와 맞춰 쓴다
        Pl = P.copy()
        for al in alr:
            Pl["q%.2f" % al] = dyn["가격만"][0][al]
        line("정적 a=0.90 %s" % lab,
             stat(sim_dyn(Pl, win, "q0.50", a.hold, a.wmax, a.delta, fm, ft,
                          static_q=q90)))
    for lab in ("가격만", "**+ 오더북**"):
        Q, oos = dyn[lab]
        Pl = P.copy()
        for al in alr:
            Pl["q%.2f" % al] = np.where(oos, Q[al], np.nan)
        for al in (0.10, 0.30, 0.50):
            line("동적 a=%.2f %s" % (al, lab),
                 stat(sim_dyn(Pl, win, "q%.2f" % al, a.hold, a.wmax, a.delta, fm, ft)))
    print("\n  ** 2번(예측 정확도)이 좋아졌는데 3번(수익)이 안 좋아지면,")
    print("     '남은 밀림을 잘 맞히는 것' 이 수익으로 안 이어진다는 뜻이다 —")
    print("     앞선 결론(수익은 타이밍이 아니라 **이탈 크기**에서 나온다)과 일관된다. **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
