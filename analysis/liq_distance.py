# -*- coding: utf-8 -*-
"""대안 A 의 유일한 실질 입력 — 청산거리 분포 f_R 을 잰다.

A 의 구성
    L_hat(p, t) = sum_{tau<t}  dOI(tau) x f_R( p / p_entry(tau) ) x S(t-tau)

  dOI 와 p_entry 는 관측된다(5분 OI + 1분봉, 2020-09~).
  자유로운 것은 f_R(청산거리 분포)과 S(생존함수) 둘뿐이다.

왜 레버리지를 명시적으로 안 다루는가
  레버리지는 중간변수다. 최종 필요물은 '진입가 대비 몇 %에서 청산되는가' 이고,
  그것이 R = p_liq / p_entry 다. 레버리지별로 나눠 계산해 더하는 것과
  R 의 경험분포를 통째로 쓰는 것은 같다 — 후자가 자유모수가 적다.

  그리고 교과서 공식 p_liq = p_entry(1-1/L) 은 |오차|<10% 가 19.3% 뿐이라
  (lev_dist.py) 쓸 수 없다. 경험분포는 그 문제를 우회한다.

이 스크립트가 답하는 것
  (1) f_R 의 모양 — 롱/숏 따로. 단봉인가 다봉인가. 꼬리는 어디까지인가.
  (2) 안정성 — 코인별 / 시간별. 단면 이질성은 '측정해서 쓰면 되는' 것이고,
      시간 불안정이라야 진짜 문제다.
  (3) 명목가 가중 vs 건수 — 큰 포지션과 작은 포지션의 R 이 다른가.
  (4) 격리 vs 교차 — 교차는 p_liq 가 계좌 자기자본에 의존해 움직이므로
      f_R 이 안정적일 이유가 없다. 그것을 확인한다.

*** 한계: HL 표본이다. HL 은 교차마진 94.5% 로 정교한 고래 중심이고,
    바이낸스 리테일과 사용자층이 다르다. 여기서 잰 f_R 은 **초기값**이며,
    정본은 Bybit 실현청산 역산으로 대체해야 한다(축적 중). ***

실행:
    python analysis/liq_distance.py
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

R_MIN, R_MAX = 0.02, 0.98     # |R-1| 이 이 밖이면 데이터 결함으로 본다
GRID = np.arange(0.005, 0.905, 0.005)   # 청산거리 격자 0.5% ~ 90%


def load_positions() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "hl_positions", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/hl_positions 비어 있음")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for c in ("liquidation_px", "entry_px", "position_value"):
        d = d[np.isfinite(d[c]) & (d[c] > 0)]
    d["dir"] = np.where(d["szi"] > 0, "long", "short")
    # R = 청산가 / 진입가.  롱이면 R<1(아래에서 청산), 숏이면 R>1.
    d["R"] = d["liquidation_px"] / d["entry_px"]
    # 청산거리: 진입가 대비 몇 % 움직이면 청산되는가 (부호 없는 크기)
    d["dist"] = np.abs(d["R"] - 1.0)
    d = d[(d["dist"] > R_MIN * 0.5) & (d["dist"] < R_MAX)]
    # 방향 정합성: 롱은 R<1, 숏은 R>1 이어야 한다
    good = ((d["dir"] == "long") & (d["R"] < 1.0)) | ((d["dir"] == "short") & (d["R"] > 1.0))
    d["dir_ok"] = good
    d["ts_dt"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    return d


def wq(v, w, q):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    cw = np.cumsum(w)
    return float(np.interp(q * cw[-1], cw, v)) if cw[-1] > 0 else np.nan


def describe(g: pd.DataFrame, label: str) -> None:
    v, w = g["dist"].to_numpy(), g["position_value"].to_numpy()
    if len(g) < 30:
        print("  %-22s n 부족 (%d)" % (label, len(g)))
        return
    qs = [wq(v, w, q) for q in (.1, .25, .5, .75, .9)]
    print("  %-22s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %11.4g %7d"
          % (label, *[100 * x for x in qs], float(w.sum()), len(g)))


def main() -> int:
    ap = argparse.ArgumentParser(description="empirical liquidation-distance distribution")
    a = ap.parse_args()
    U.init_stdout()

    d = load_positions()
    print("=" * 78)
    print("대안 A 의 입력 — 청산거리 분포 f_R  (R = p_liq / p_entry)")
    print("=" * 78)
    print("포지션 %d행 | 코인 %d | 스윕 %d | %s ~ %s"
          % (len(d), d.coin.nunique(), d.sweep_id.nunique(),
             d.ts_dt.min(), d.ts_dt.max()))
    print("방향 정합(롱=아래청산 / 숏=위청산): %.2f%%" % (100 * d.dir_ok.mean()))
    d = d[d.dir_ok]

    hdr = ("  %-22s %7s %7s %7s %7s %7s %11s %7s"
           % ("구분", "p10", "p25", "중앙", "p75", "p90", "명목가$", "n"))

    print("\n--- 1. f_R 의 모양 (명목가 가중, 청산거리 |R-1|) ---")
    print(hdr)
    for lt in ("isolated", "cross"):
        for dr in ("long", "short"):
            describe(d[(d.lev_type == lt) & (d.dir == dr)], "%s / %s" % (lt, dr))
    describe(d[d.lev_type == "isolated"], "격리 전체")
    describe(d[d.lev_type == "cross"], "교차 전체")

    print("\n--- 2. 격자 밀도 (격리, 명목가 비중) — 이게 곧 f_R 이다 ---")
    iso = d[d.lev_type == "isolated"]
    for dr in ("long", "short"):
        g = iso[iso.dir == dr]
        if len(g) < 100:
            continue
        v, w = g["dist"].to_numpy(), g["position_value"].to_numpy()
        tot = w.sum()
        print("  [%s]  명목가 $%.4g" % (dr, tot))
        cells = []
        for lo, hi in ((0, .01), (.01, .02), (.02, .03), (.03, .05), (.05, .08),
                       (.08, .12), (.12, .20), (.20, .35), (.35, 1.0)):
            m = (v >= lo) & (v < hi)
            cells.append("%3.0f~%3.0f%%:%5.1f%%" % (100 * lo, 100 * hi,
                                                    100 * w[m].sum() / tot))
        print("    " + "  ".join(cells[:5]))
        print("    " + "  ".join(cells[5:]))

    print("\n--- 3. 안정성 (a) 코인별 — 명목가 상위 8종, 격리 ---")
    print("  단면 이질성은 '측정해서 쓰면 되는' 것이다. 시간 불안정이라야 문제다.")
    print(hdr)
    top = (iso.groupby("coin")["position_value"].sum()
           .sort_values(ascending=False).head(8).index)
    for c in top:
        describe(iso[iso.coin == c], c)

    print("\n--- 3. 안정성 (b) 시간별 — 6시간 단위, 격리 ---")
    print(hdr)
    iso = iso.copy()
    iso["blk"] = iso["ts_dt"].dt.floor("6h")
    for b, g in iso.groupby("blk"):
        describe(g, str(b)[:16])

    print("\n--- 4. 명목가 가중 vs 건수 (큰 포지션이 다르게 행동하나) ---")
    for lt in ("isolated", "cross"):
        g = d[d.lev_type == lt]
        if len(g) < 100:
            continue
        v, w = g["dist"].to_numpy(), g["position_value"].to_numpy()
        print("  %-10s 명목가가중 중앙 %6.2f%%   단순 중앙 %6.2f%%   비 %.2f"
              % (lt, 100 * wq(v, w, .5), 100 * np.median(v),
                 wq(v, w, .5) / max(np.median(v), 1e-9)))
    print("  비가 1 보다 작으면 큰 포지션이 더 가까이서 청산된다(고레버리지).")

    print("\n--- 5. 판정 ---")
    med = []
    for b, g in iso.groupby("blk"):
        if len(g) >= 30:
            med.append(wq(g["dist"].to_numpy(), g["position_value"].to_numpy(), .5))
    med = np.array(med)
    if med.size >= 2:
        print("  시간 간 중앙 청산거리: 평균 %.2f%%  sd %.2f%%  변동계수 %.3f"
              % (100 * med.mean(), 100 * med.std(ddof=1), med.std(ddof=1) / med.mean()))
    cmed = []
    for c in top:
        g = iso[iso.coin == c]
        if len(g) >= 30:
            cmed.append(wq(g["dist"].to_numpy(), g["position_value"].to_numpy(), .5))
    cmed = np.array(cmed)
    if cmed.size >= 2:
        print("  코인 간 중앙 청산거리: 평균 %.2f%%  sd %.2f%%  변동계수 %.3f  범위 [%.1f%%, %.1f%%]"
              % (100 * cmed.mean(), 100 * cmed.std(ddof=1), cmed.std(ddof=1) / cmed.mean(),
                 100 * cmed.min(), 100 * cmed.max()))
    print("\n  A 가 성립하려면 **시간** 변동계수가 작아야 한다(f_R 을 고정해 쓸 수 있어야).")
    print("  코인 변동계수는 커도 된다 — 코인별로 측정해 쓰면 된다.")
    print("  *** 표본 2일. 시간 안정성은 '레짐 간' 이 아니라 '이틀 안' 만 본 것이다.")
    print("  *** HL 표본이라 바이낸스에 대해 편향될 수 있다. 정본은 Bybit 역산.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
