# -*- coding: utf-8 -*-
"""R-5 — 지정매수를 얼마나 아래에 걸 것인가. 핵심은 **역선택**이다.

왜 이 검정인가
  R-2 5절: 시장가로 open[i+1] 에 들어가면 평균 +51bp 인데 **MAE 중앙이 -60bp** 다.
  가격이 실제로 더 내려간다는 뜻이므로, 아래에 건 지정가는 체결되고 더 좋은 가격을
  받는다. 그것이 사용자의 원래 설계다.

  **그런데 공짜가 아니다.** 깊게 걸수록:
    (+) 체결가가 좋아진다
    (-) 체결률이 떨어진다 (안 내려온 사건은 놓친다)
    (-) **역선택**: 깊은 지정가가 체결되는 사건은 정확히 **캐스케이드가 계속된**
        사건이다. 좋은 가격 대신 **나쁜 사건만 골라 잡을** 수 있다.
  세 번째가 이 검정의 전부다. 앞의 둘만 보면 깊을수록 좋다는 오답이 나온다.

측정
  기준가 P0 = open[i+1] (판정 시점 이후 첫 체결 가능 가격).
  깊이 d bp 아래에 지정가. 체결 창 W 바 안에 저가가 닿으면 체결로 본다.
  체결 후 H 바 보유하고 청산.

  이벤트당 기대값 = 체결률 x 조건부 평균수익.
  **체결 안 된 사건은 수익 0** 이지 손실이 아니다 — 그래서 체결률 하락 자체는
  치명적이지 않다. 조건부 평균이 무너지는지가 관건이다.

체결 가정의 낙관성 (반드시 감안)
  low[t] <= limit 이면 체결로 처리한다. 이는 **큐 맨 앞** 을 가정한 상한이다.
  급락 중 그 가격을 스쳐 지나가면 실제로는 미체결일 수 있다. 따라서
  '저가가 limit 를 x bp 초과해 뚫었을 때만 체결' 하는 보수판도 함께 낸다.

실행:
    python analysis/entry_price.py
    python analysis/entry_price.py --fill-through 10   # 보수적 체결 가정
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
from analysis.event_study_h2 import load, find_events           # noqa: E402
from analysis.response_liq import ols_cluster, cmean            # noqa: E402
from analysis.scale_check import K, DOI_THR, MIN_GAP, VOL_WIN   # noqa: E402

DEPTHS = [0, 25, 50, 75, 100, 150, 200, 300, 500]   # 기준가 아래 bp
WAIT = 3           # 체결 대기 바(5분봉 3개 = 15분)
HOLD = 3           # 체결 후 보유 바(15분) — R-2 5c: 15분에 결판난다
COST = 2.0         # 메이커 왕복 bp


def build(symbols, wait: int, hold: int, through: float) -> pd.DataFrame:
    """이벤트 x 깊이 격자. 체결 여부·체결가·수익·체결 후 MAE."""
    rows = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, K, DOI_THR, MIN_GAP)
        if len(ev) == 0:
            continue
        op = df["open"].to_numpy(dtype=np.float64)
        cl = df["close"].to_numpy(dtype=np.float64)
        hi = df["high"].to_numpy(dtype=np.float64)
        lo = df["low"].to_numpy(dtype=np.float64)
        ret = df["ret"].to_numpy(dtype=np.float64)
        ctg = df["contig"].to_numpy(dtype=bool)
        ot = df["open_time"].to_numpy()
        n = len(df)
        sig = (pd.Series(ret).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .std().to_numpy()) * np.sqrt(float(VOL_WIN))

        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i, sd = int(r.i), int(r.side)
            j = i + 1
            if j + wait + hold >= n or not (np.isfinite(op[j]) and op[j] > 0):
                continue
            if not (np.isfinite(sig[i]) and sig[i] > 0):
                continue
            if not ctg[j:j + wait + hold].all():
                continue                       # 바 결손 구간은 제외
            p0 = op[j]
            # side=+1: 롱청산 -> 하락 -> **매수**. 유리한 방향은 아래.
            # side=-1: 숏청산 -> 상승 -> **매도**. 유리한 방향은 위.
            for d in DEPTHS:
                lim = p0 * (1.0 - sd * d / 1e4)
                seg_lo, seg_hi = lo[j:j + wait], hi[j:j + wait]
                if sd == 1:
                    touched = seg_lo <= lim * (1.0 - through / 1e4)
                else:
                    touched = seg_hi >= lim * (1.0 + through / 1e4)
                if not touched.any():
                    rows.append({"symbol": s, "day": int(ot[i] // 86_400_000),
                                 "side": sd, "d": d, "fill": 0,
                                 "ret": np.nan, "mae": np.nan, "i": i})
                    continue
                fb = j + int(np.argmax(touched))          # 체결 바
                t = fb + hold
                if t >= n:
                    continue
                rr = (cl[t] / lim - 1.0) * sd * 1e4 - COST
                if sd == 1:
                    mae = (lo[fb:t + 1].min() / lim - 1.0) * 1e4
                else:
                    mae = -(hi[fb:t + 1].max() / lim - 1.0) * 1e4
                rows.append({"symbol": s, "day": int(ot[i] // 86_400_000),
                             "side": sd, "d": d, "fill": 1,
                             "ret": rr, "mae": mae, "i": i})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="R-5 limit entry depth")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--wait", type=int, default=WAIT)
    ap.add_argument("--hold", type=int, default=HOLD)
    ap.add_argument("--fill-through", type=float, default=0.0,
                    help="저가가 지정가를 이만큼(bp) 뚫어야 체결로 인정")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 78)
    print("R-5 — 지정매수 깊이. 좋은 가격 vs 체결률 vs **역선택**")
    print("=" * 78)
    d = build(syms, a.wait, a.hold, a.fill_through)
    if len(d) == 0:
        print("표본 없음")
        return 1
    nev = d.groupby(["symbol", "i"]).ngroups
    print("이벤트 %d건 | 대기 %d바(%d분) | 보유 %d바(%d분) | 비용 %.1fbp"
          % (nev, a.wait, 5 * a.wait, a.hold, 5 * a.hold, COST))
    print("체결 가정: 저가가 지정가를 %.0fbp 뚫으면 체결 (0 = 닿기만 하면 체결)"
          % a.fill_through)
    print("*** 미체결은 수익 0 이지 손실이 아니다. 이벤트당 기대값 = 체결률 x 조건부평균\n")

    print("  %5s | %7s | %10s %7s | %10s %10s | %10s %8s"
          % ("깊이bp", "체결률", "조건부평균", "t", "조건부중앙", "승률%",
             "이벤트당EV", "MAE중앙"))
    best = None
    for dep in DEPTHS:
        g = d[d["d"] == dep]
        fr = float(g["fill"].mean())
        f = g[g["fill"] == 1]
        if len(f) < 30:
            continue
        m, se, t, _ = cmean(f["ret"].to_numpy(), f["day"].to_numpy())
        ev_ = fr * m
        print("  %5d | %6.1f%% | %10.1f %7.1f | %10.1f %9.1f%% | %10.1f %8.1f"
              % (dep, 100 * fr, m, t, f["ret"].median(),
                 100 * (f["ret"] > 0).mean(), ev_, f["mae"].median()))
        if best is None or ev_ > best[1]:
            best = (dep, ev_, m, fr, t)
    if best:
        print("\n  최대 EV: 깊이 %dbp | 이벤트당 %.1fbp (체결률 %.1f%%, 조건부 %.1fbp, t=%.1f)"
              % (best[0], best[1], 100 * best[3], best[2], best[4]))

    print("\n" + "-" * 78)
    print("역선택 직접 검정 — 깊게 체결된 건이 '더 나쁜 사건' 인가")
    print("-" * 78)
    print("  같은 사건을 두 깊이에서 비교한다. 깊은 쪽이 체결됐을 때,")
    print("  **얕은 쪽(0bp, 사실상 시장가) 수익**이 전체 평균보다 나쁜가?")
    base = d[d["d"] == 0].set_index(["symbol", "i"])["ret"]
    b_all, _, _, _ = cmean(base.dropna().to_numpy(),
                           d[d["d"] == 0].dropna(subset=["ret"])["day"].to_numpy())
    print("\n  기준(0bp) 전체 평균 %.1f bp\n" % b_all)
    print("  %7s | %9s | %14s | %12s"
          % ("깊이bp", "체결 n", "그 건들의 0bp수익", "차이"))
    for dep in DEPTHS[1:]:
        g = d[(d["d"] == dep) & (d["fill"] == 1)]
        if len(g) < 30:
            continue
        k = base.reindex(pd.MultiIndex.from_frame(g[["symbol", "i"]])).to_numpy()
        m = np.isfinite(k)
        if m.sum() < 30:
            continue
        bb, _, _, _ = cmean(k[m], g["day"].to_numpy()[m])
        print("  %7d | %9d | %14.1f | %12.1f" % (dep, int(m.sum()), bb, bb - b_all))
    print("\n  차이가 크게 음수면 **역선택이 실재**한다 — 깊은 지정가는 나쁜 사건만 잡는다.")
    print("  그럼에도 조건부 평균이 양수면, 좋은 진입가가 역선택을 이긴 것이다.")

    print("\n" + "-" * 78)
    print("대조군")
    print("-" * 78)
    g0 = d[(d["d"] == 0) & (d["fill"] == 1)]
    m0, _, t0, _ = cmean(g0["ret"].to_numpy(), g0["day"].to_numpy())
    print("  시장가 상당(0bp, 체결률 %.0f%%)  : 이벤트당 %.1f bp (t=%.1f)"
          % (100 * d[d.d == 0]["fill"].mean(), m0 * d[d.d == 0]["fill"].mean(), t0))
    print("  페이퍼 기준선(고정 2%%/15분)     : 이벤트당 28.2 bp (t(NW)=3.8)")
    print("\n  *** 체결 가정이 낙관적이다. --fill-through 10 으로 재실행해 볼 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
