# -*- coding: utf-8 -*-
"""30초 해상도가 급락 순간의 깊이 붕괴를 놓치는가 — capacity2 의 보수성 검정.

문제
  `capacity2.py` 는 30초 격자의 `book_depth` 를 쓴다. 진입 시각 직전의 스냅샷을
  가져오므로 **최대 30초 묵은 깊이**를 본다 (실측 지연 중앙 29초).
  캐스케이드는 초 단위로 진행되고 호가는 그 사이에 빠진다. 30초 전 깊이가
  실제보다 **두껍다면 capacity2 의 슬리피지는 과소평가**이고 용량은 과대평가다.

방법
  웹소켓 1초 패널(`ws_panel`)에서 같은 것을 두 번 잰다.
    d_true(t) : 시각 t 의 실제 매수호가 깊이
    d_30(t)   : t 이하의 가장 최근 **30초 격자** 시점의 깊이  (= book_depth 가 주는 값)
  비율 d_30/d_true 를 **동시 가격 움직임 크기별**로 층화한다.
  1 보다 크면 30초 격자가 깊이를 부풀린다는 뜻이다.

주의
  웹소켓은 밴드 정의가 다르다(±0.5/1/2% 명목가). book_depth 는 ±1~5%.
  여기서 재는 것은 **절대 깊이가 아니라 해상도로 인한 비율 왜곡**이므로
  밴드가 정확히 같을 필요는 없다. b1(±1%) 로 맞춰 쓴다.

실행:
    python analysis/ws_depth_lag.py
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
from analysis.ws_panel import load, gaps                               # noqa: E402

W = 112
WIN = 60


def main() -> int:
    ap = argparse.ArgumentParser(description="does 30s resolution miss depth collapse")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--grid", type=int, default=30)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * W)
    print("30초 해상도가 급락 순간의 깊이 붕괴를 놓치는가")
    print("=" * W)
    print("d_30/d_true — 1보다 크면 %d초 격자가 실제보다 두꺼운 깊이를 보여준다." % a.grid)
    print("(그러면 capacity2 의 슬리피지는 과소평가, 용량은 과대평가다)\n")

    R, MOVE, SYM = [], [], []
    for s in syms:
        try:
            d = load(s)
        except FileNotFoundError:
            continue
        if "bid_b1" not in d:
            continue
        sec_idx = d.index.to_numpy()
        bid = d["bid_b1"].to_numpy(dtype=np.float64)
        mid = pd.Series(d["mid"].to_numpy(dtype=np.float64)).ffill(limit=5).to_numpy()
        # 30초 격자에서 마지막으로 관측된 깊이 (격자 시점에 결측이면 그 이전 격자)
        on_grid = (sec_idx % a.grid) == 0
        gi = np.where(on_grid & np.isfinite(bid), np.arange(len(bid)), -1)
        gi = np.maximum.accumulate(gi)
        okg = gi >= 0
        d30 = np.full(len(bid), np.nan)
        d30[okg] = bid[gi[okg]]
        # 60초 가격 변화
        m = pd.Series(mid)
        r = (m / m.shift(WIN) - 1.0).to_numpy() * 1e4
        # 연속 구간 안에서만
        valid = np.zeros(len(bid), dtype=bool)
        for x, y in gaps(d):
            if y - x > WIN + a.grid:
                valid[x + WIN + a.grid:y + 1] = True
        ok = valid & np.isfinite(bid) & (bid > 0) & np.isfinite(d30) & (d30 > 0) \
            & np.isfinite(r)
        R.append(d30[ok] / bid[ok])
        MOVE.append(r[ok])
        SYM.append(np.full(int(ok.sum()), s))
    if not R:
        print("데이터 없음")
        return 1
    R = np.concatenate(R)
    MOVE = np.concatenate(MOVE)
    SYM = np.concatenate(SYM)
    print("관측 %d 심볼-초 (심볼 %d종)\n" % (len(R), len(set(SYM))))

    print("  %-16s | %7s | %s" % ("60초 가격변화", "n",
                                  " ".join("%8s" % ("p%g" % q)
                                           for q in (25, 50, 75, 90, 99))))
    bins = [(-1e9, -200), (-200, -100), (-100, -50), (-50, -20),
            (-20, 20), (20, 1e9)]
    for lo, hi in bins:
        m = (MOVE >= lo) & (MOVE < hi)
        if m.sum() < 100:
            print("  %-16s | %7d | (표본부족)" % ("%d~%dbp" % (lo, hi), int(m.sum())))
            continue
        v = R[m]
        print("  %-16s | %7d | %s"
              % ("%d~%dbp" % (max(lo, -9999), min(hi, 9999)), int(m.sum()),
                 " ".join("%8.3f" % np.percentile(v, q)
                          for q in (25, 50, 75, 90, 99))))
    print("\n  ** 하락이 클수록 중앙값이 1 을 크게 넘으면 30초 격자가 붕괴를 놓치는 것이다. **")

    print("\n  급락 구간(60초 -100bp 이하)만 다시:")
    m = MOVE <= -100
    if m.sum() >= 100:
        v = R[m]
        print("    중앙 %.3f | 평균 %.3f | 1 초과 비율 %.1f%% | p90 %.3f"
              % (np.median(v), np.mean(v), 100 * (v > 1).mean(),
                 np.percentile(v, 90)))
        print("    -> capacity2 의 깊이를 **%.0f%%** 로 줄여 읽어야 한다 (중앙 기준)"
              % (100 / np.median(v)))
    else:
        print("    표본부족 (n=%d)" % int(m.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
