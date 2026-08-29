# -*- coding: utf-8 -*-
"""A-1 — 재구성 지도가 실현 청산 위치를 맞히는가.

무엇을 대조하는가
  지도   heatmap.py 로 공개 데이터(5분 OI + 가격)에서 재구성한 L_hat(p, t)
  실현   Bybit allLiquidation 전건 (가격 + 크기 + 방향)

Q1 에서 범한 교락을 피한다
  Q1 은 실현 위치 u 의 지도 두께를 **반대편 -u** 와 비교해 82.2% 를 얻었다.
  그런데 그 구간이 롱 편중(건수 74%)이라 '어느 쪽' 과 '어느 가격대' 를 구분하지 못했다.
  여기서는 **같은 방향의 다른 거리** 를 대조군으로 쓴다.

세 검정
  (1) 밴드 패널 — (심볼 x 시간 x 거리밴드) 단위로
      실현 청산액 ~ 지도 두께.  밴드 고정효과로 '거리' 를 통제한다.
      가격이 도달하지 못한 밴드는 제외해야 한다(청산이 물리적으로 불가능).
  (2) 같은 방향 순위 — 실현 청산이 난 거리가, 같은 방향 지도 질량의 몇 분위인가.
      지도가 무정보면 균등분포. 두꺼운 쪽에 몰리면 상위로 쏠린다.
  (3) 대조군 비교 — 지도 vs '균등' vs 'OI 만' (지도 없이 총 OI 로 배분)

한계
  - 44시간. 롱 편중 구간. 레짐 일반화 불가.
  - 거래소 교차: 지도는 Binance OI/가격, 실현은 Bybit. 메이저는 가격이 붙어 다닌다.
  - f_R 이 HL 표본이라 바이낸스에 대해 편향 가능(TARGET_DESIGN 3.2).

실행:
    python analysis/a1_map_validate.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from analysis.heatmap import load_fr, load_oi, build_map, MAP_GRID   # noqa: E402

HOUR_MS = 3_600_000
# 거리 밴드 (양수 = 현재가로부터의 거리)
BANDS = [(0.000, 0.005), (0.005, 0.010), (0.010, 0.020),
         (0.020, 0.030), (0.030, 0.050), (0.050, 0.100)]


def load_bybit() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "bybit_liq", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["symbol"].isin(C.MAJORS)].copy()
    d["ntl"] = d["size"] * d["bankruptcy_px"]
    d = d[np.isfinite(d["ntl"]) & (d["ntl"] > 0) & np.isfinite(d["bankruptcy_px"])]
    return d.sort_values("exch_ms").reset_index(drop=True)


def ols(X, y):
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)


def main() -> int:
    ap = argparse.ArgumentParser(description="A-1: does the reconstructed map predict liquidations")
    ap.add_argument("--lookback", type=float, default=30.0)
    ap.add_argument("--half-life", type=float, default=7.0)
    ap.add_argument("--iso-share", type=float, default=0.055)
    a = ap.parse_args()
    U.init_stdout()

    fr = load_fr()
    byb = load_bybit()
    byb["hour"] = byb["exch_ms"] // HOUR_MS
    syms = sorted(byb["symbol"].unique())
    lb = int(a.lookback * 288)
    hl = a.half_life * 288

    print("=" * 76)
    print("A-1 — 재구성 지도가 실현 청산 위치를 맞히는가")
    print("=" * 76)
    t0 = pd.to_datetime(byb.exch_ms.min(), unit="ms", utc=True)
    t1 = pd.to_datetime(byb.exch_ms.max(), unit="ms", utc=True)
    print("실현청산 %d건 / %d종 | %s ~ %s (%.1f시간)"
          % (len(byb), len(syms), t0, t1, (t1 - t0).total_seconds() / 3600))

    rows, ranks, drops = [], [], {"oi": 0, "align": 0, "reach": 0}
    for sym in syms:
        try:
            d = load_oi(sym)
        except FileNotFoundError:
            drops["oi"] += 1
            continue
        ot = d["open_time"].to_numpy()
        gb = byb[byb.symbol == sym]
        for hr, g in gb.groupby("hour"):
            t_start = int(hr * HOUR_MS)
            i = int(np.searchsorted(ot, t_start, side="right")) - 1
            # OI 데이터가 그 시각까지 없으면(벌크는 T-1) 건너뛴다
            if i < lb + 10 or i >= len(d) - 1 or t_start - int(ot[i]) > 2 * HOUR_MS:
                drops["align"] += 1
                continue
            p0, L, S = build_map(d, i, fr, lb, hl, a.iso_share)
            if L.sum() + S.sum() <= 0:
                continue
            # 그 시간 실제 도달 범위 (도달 못 한 밴드는 청산이 불가능)
            px = g["bankruptcy_px"].to_numpy()
            lo_reach = float(px.min()) / p0 - 1.0
            hi_reach = float(px.max()) / p0 - 1.0

            for side, mp, sgn in (("long", L, -1.0), ("short", S, +1.0)):
                gs = g[g.pos_side == side]
                reach = abs(lo_reach) if side == "long" else abs(hi_reach)
                if reach <= 0:
                    continue
                u_real = (gs["bankruptcy_px"].to_numpy() / p0 - 1.0) * sgn
                for lo, hi in BANDS:
                    if lo >= reach:            # 도달 못 한 밴드 제외
                        drops["reach"] += 1
                        continue
                    m = (MAP_GRID * sgn >= lo) & (MAP_GRID * sgn < hi)
                    Lb = float(mp[m].sum())
                    Rb = float(gs["ntl"].to_numpy()[(u_real >= lo) & (u_real < hi)].sum())
                    rows.append({"symbol": sym, "hour": hr, "side": side,
                                 "band": "%.1f-%.1f%%" % (100 * lo, 100 * hi),
                                 "bi": BANDS.index((lo, hi)), "L": Lb, "R": Rb})
                # 같은 방향 순위: 실현 거리가 지도 질량의 몇 분위인가
                mm = (MAP_GRID * sgn > 0)
                gu = MAP_GRID[mm] * sgn
                gw = mp[mm]
                o = np.argsort(gu)
                gu, gw = gu[o], gw[o]
                cw = np.cumsum(gw)
                if cw[-1] <= 0:
                    continue
                for ur, nt in zip(u_real, gs["ntl"].to_numpy()):
                    if ur <= 0 or ur > reach:
                        continue
                    q = float(np.interp(ur, gu, cw) / cw[-1])
                    # 도달범위로 절단된 분위 (도달 못 한 곳은 애초에 불가능하므로)
                    qmax = float(np.interp(reach, gu, cw) / cw[-1])
                    ranks.append({"symbol": sym, "side": side, "ntl": nt,
                                  "q": q, "q_trunc": q / qmax if qmax > 0 else np.nan})

    p = pd.DataFrame(rows)
    r = pd.DataFrame(ranks)
    print("탈락: OI없음 %d종 / 정합실패 %d / 미도달밴드 %d"
          % (drops["oi"], drops["align"], drops["reach"]))
    print("밴드 관측 %d | 순위 관측 %d" % (len(p), len(r)))
    if len(p) < 30:
        print("표본 부족 — 축적 필요")
        return 1

    print("\n--- 1. 밴드 패널: 지도 두께가 실현 청산액을 예측하는가 ---")
    print("  밴드 고정효과로 '거리' 통제. 도달 못 한 밴드는 제외했다.")
    print("  %-12s %7s %13s %13s %9s" % ("밴드", "n", "지도 중앙$", "실현 중앙$", "실현>0"))
    for bi, g in p.groupby("bi"):
        print("  %-12s %7d %13.4g %13.4g %8.1f%%"
              % (g["band"].iloc[0], len(g), g.L.median(), g.R.median(),
                 100 * float((g.R > 0).mean())))

    q = p[(p.L > 0)].copy()
    q["lL"] = np.log(q.L)
    q["hit"] = (q.R > 0).astype(float)
    # 밴드 더미 + log 지도두께
    dum = pd.get_dummies(q["bi"], prefix="b", drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(q)), q["lL"].to_numpy(), dum.to_numpy()])
    for lab, y in (("청산 발생 여부", q["hit"].to_numpy()),
                   ("log(1+청산액)", np.log1p(q.R.to_numpy()))):
        w = ols(X, y)
        yh = X @ w
        den = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - yh) ** 2)) / den if den > 0 else np.nan
        # 클러스터(심볼) 로버스트 SE
        res = y - yh
        XtXi = np.linalg.pinv(X.T @ X)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for s_ in q["symbol"].unique():
            m_ = (q["symbol"] == s_).to_numpy()
            sc = X[m_].T @ res[m_]
            meat += np.outer(sc, sc)
        G = q["symbol"].nunique()
        V = (G / max(G - 1, 1)) * (XtXi @ meat @ XtXi)
        se = float(np.sqrt(max(V[1, 1], 0)))
        print("  %-16s  log(지도두께) 계수 %+.4f  SE %.4f  t %+.2f  R2 %.3f"
              % (lab, w[1], se, w[1] / se if se > 0 else np.nan, r2))
    print("  (계수가 양수·유의하면 지도가 두꺼운 밴드에서 청산이 더 난다는 뜻)")

    print("\n--- 2. 같은 방향 순위: 실현 청산이 지도 질량의 몇 분위에서 났나 ---")
    print("  지도가 무정보면 균등분포(평균 0.5). 두꺼운 쪽에 몰리면 다르게 나온다.")
    if len(r) >= 50:
        for col, lab in (("q", "전체 지도 기준"), ("q_trunc", "도달범위 절단 기준")):
            v = r[col].dropna().to_numpy()
            if v.size < 50:
                continue
            wv = r.loc[r[col].notna(), "ntl"].to_numpy()
            m = float(np.mean(v))
            mw = float(np.sum(v * wv) / np.sum(wv))
            se = float(np.std(v, ddof=1) / np.sqrt(len(v)))
            print("  %-18s n=%5d  평균 %.3f (SE %.3f, z=%+.2f)  명목가가중 %.3f"
                  % (lab, len(v), m, se, (m - 0.5) / se if se > 0 else np.nan, mw))
        from scipy import stats as st
        v = r["q_trunc"].dropna().to_numpy()
        if v.size >= 50:
            ks = st.kstest(v, "uniform")
            print("  균등성 KS: D=%.4f  p=%.4f  (p<0.05 면 균등 아님 = 지도에 정보 있음)"
                  % (ks.statistic, ks.pvalue))
    else:
        print("  표본 부족")

    print("\n--- 3. 대조군: 지도 vs 균등 vs 거리만 ---")
    print("  '거리만' = 지도 없이 '가까울수록 많이 청산된다' 만 쓰는 규칙")
    q2 = q.copy()
    q2["inv_d"] = 1.0 / (0.5 * (np.array([BANDS[b][0] for b in q2.bi])
                                + np.array([BANDS[b][1] for b in q2.bi])))
    y = np.log1p(q2.R.to_numpy())
    for lab, cols in (("균등(절편만)", []),
                      ("거리만", ["inv_d"]),
                      ("지도만", ["lL"]),
                      ("지도+거리", ["lL", "inv_d"])):
        Xc = np.column_stack([np.ones(len(q2))] + [np.log(q2[c].to_numpy()) for c in cols])
        w = ols(Xc, y)
        yh = Xc @ w
        den = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - yh) ** 2)) / den if den > 0 else np.nan
        print("  %-14s R2 = %+.4f" % (lab, r2))
    print("  '지도만' 이 '거리만' 을 넘어야 지도가 거리 이상의 정보를 준 것이다.")

    print("\n  *** 44시간, 롱 편중 구간, 거래소 교차(Binance 지도 x Bybit 실현).")
    print("  *** f_R 이 HL 표본이라 편향 가능. 1차 검정은 '방향 확인' 용도다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
