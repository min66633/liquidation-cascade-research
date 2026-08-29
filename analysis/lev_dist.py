# -*- coding: utf-8 -*-
"""대안 A 의 전제 검정 — 격리 포지션의 레버리지 분포가 안정적인가.

대안 A 가 하려는 것
  청산가는 진입가와 레버리지로 결정된다:  p_liq ~ p_entry (1 - 1/L)   (롱)
  따라서 과거 공개 데이터만으로 히트맵을 **재구성**할 수 있다:

      L_hat(p) = sum_tau  dOI(tau) x P( L : p_entry(tau)(1-1/L) = p ) x decay(tau)

  재료: dOI 5분(2020-09~), 1분봉 가격(6년) — 둘 다 있다.
  자유모수: **레버리지 분포**와 decay.

  이게 되면 2020년부터 백테스트가 된다. HL 실측 경로(6개월 대기, 70% 공백)와
  비교가 안 되는 검정력이다.

이 스크립트가 검정하는 것 — A 의 전제
  레버리지 분포가 (1) 심볼 간 (2) 시간에 걸쳐 (3) 가격대별로 안정적인가.
  불안정하면 자유모수가 사실상 자유롭게 움직여 A 가 과적합이 된다.

  HL 이 지도로는 부족해도(phi~0.05, 70% 공백) **모수 추정에는 쓸 수 있다** —
  레버리지 분포는 소수 표본으로도 형태가 잡힌다. 그것이 HL 데이터의 새 용도다.

*** 격리만 본다. 교차는 청산가가 계좌 자기자본에 의존해 움직이므로
    p_liq = p_entry(1-1/L) 이 성립하지 않는다. ***

실행:
    python analysis/lev_dist.py
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

BAND = 0.05


def load_positions() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "hl_positions", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/hl_positions 비어 있음")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for c in ("liquidation_px", "entry_px", "position_value", "lev_value"):
        d = d[np.isfinite(d[c])]
    d = d[(d["liquidation_px"] > 0) & (d["entry_px"] > 0) & (d["position_value"] > 0)]
    d["dir"] = np.where(d["szi"] > 0, "long", "short")
    return d


def load_mids() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "hl_mids", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d[np.isfinite(d["mid_px"]) & (d["mid_px"] > 0)]


def wq(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """명목가 가중 분위수. 계좌 수가 아니라 '돈'의 분포를 봐야 한다."""
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    return float(np.interp(q * cw[-1], cw, v))


def main() -> int:
    ap = argparse.ArgumentParser(description="is the leverage distribution stable")
    ap.add_argument("--band", type=float, default=BAND)
    a = ap.parse_args()
    U.init_stdout()

    pos, mids = load_positions(), load_mids()
    mid_map = (mids.drop_duplicates(subset=["sweep_id", "coin"], keep="last")
               .set_index(["sweep_id", "coin"])["mid_px"].to_dict())
    pos["mid"] = [mid_map.get((s, c), np.nan) for s, c in
                  zip(pos["sweep_id"], pos["coin"])]
    pos = pos[np.isfinite(pos["mid"]) & (pos["mid"] > 0)].copy()
    pos["u"] = pos["liquidation_px"] / pos["mid"] - 1.0
    pos["ts_dt"] = pd.to_datetime(pos["ts"], unit="ms", utc=True)

    iso = pos[pos["lev_type"] == "isolated"].copy()
    print("=" * 74)
    print("대안 A 전제 검정 — 격리 포지션의 레버리지 분포가 안정적인가")
    print("=" * 74)
    print("전체 포지션 %d행 | 격리 %d행 (%.1f%%) | 스윕 %d | 코인 %d"
          % (len(pos), len(iso), 100 * len(iso) / len(pos),
             pos.sweep_id.nunique(), pos.coin.nunique()))
    print("구간 %s ~ %s" % (pos.ts_dt.min(), pos.ts_dt.max()))
    print("명목가 비중: 격리 %.1f%% / 교차 %.1f%%"
          % (100 * iso.position_value.sum() / pos.position_value.sum(),
             100 * pos[pos.lev_type == "cross"].position_value.sum() / pos.position_value.sum()))

    # ---------------------------------------------------------- 0. 항등식 검증
    print("\n--- 0. p_liq = p_entry (1 - 1/L) 이 실제로 성립하는가 ---")
    print("  이 항등식이 A 의 뼈대다. 성립 안 하면 A 자체가 불가능하다.")
    s = iso.copy()
    sign = np.where(s["dir"] == "long", 1.0, -1.0)
    implied = 1.0 / np.abs(1.0 - s["liquidation_px"] / s["entry_px"])
    s["lev_implied"] = implied
    ok = np.isfinite(s["lev_implied"]) & (s["lev_implied"] > 0) & (s["lev_implied"] < 200)
    s = s[ok]
    rel = (s["lev_implied"] - s["lev_value"]) / s["lev_value"].replace(0, np.nan)
    print("  격리 %d행 중 유효 %d" % (len(iso), len(s)))
    print("  내재 레버리지 vs 신고 lev_value 상대오차: 중앙 %+.3f  p10 %+.3f  p90 %+.3f"
          % (rel.median(), rel.quantile(.1), rel.quantile(.9)))
    print("  |상대오차| < 0.1 인 비율: %.1f%%" % (100 * float((rel.abs() < 0.1).mean())))
    print("  (유지증거금·펀딩 때문에 정확히 일치하진 않는다. 10%% 이내면 근사로 쓸 만하다.)")

    # ---------------------------------------------------------- 1. 분포 형태
    print("\n--- 1. 레버리지 분포 (명목가 가중, 격리) ---")
    v, w = s["lev_value"].to_numpy(), s["position_value"].to_numpy()
    print("  분위: " + "  ".join("%d%%=%.1fx" % (100 * q, wq(v, w, q))
                                 for q in (.1, .25, .5, .75, .9, .99)))
    print("  명목가 가중 평균 %.2fx | 단순 평균 %.2fx (계좌수 기준)"
          % (float(np.sum(v * w) / np.sum(w)), float(v.mean())))
    print("\n  레버리지 구간별 명목가 비중:")
    bins = [(0, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1000)]
    tot = float(w.sum())
    for lo, hi in bins:
        m = (v >= lo) & (v < hi)
        print("    %3.0f~%4.0fx : %6.2f%%  (건수 %d)"
              % (lo, hi, 100 * float(w[m].sum()) / tot, int(m.sum())))

    # ---------------------------------------------------------- 2. 안정성
    print("\n--- 2. 안정성 (A 의 자유모수가 실제로 자유로운가) ---")
    print("  (a) 코인별 — 명목가 상위 8종")
    top = (iso.groupby("coin")["position_value"].sum().sort_values(ascending=False)
           .head(8).index)
    print("  %-8s %8s %8s %8s %9s %7s" % ("코인", "중앙x", "p25", "p75", "가중평균", "n"))
    for c in top:
        g = s[s.coin == c]
        if len(g) < 20:
            continue
        gv, gw = g["lev_value"].to_numpy(), g["position_value"].to_numpy()
        print("  %-8s %7.1fx %7.1fx %7.1fx %8.2fx %7d"
              % (c, wq(gv, gw, .5), wq(gv, gw, .25), wq(gv, gw, .75),
                 float(np.sum(gv * gw) / np.sum(gw)), len(g)))

    print("\n  (b) 시간별 — 6시간 단위")
    s["blk"] = s["ts_dt"].dt.floor("6h")
    print("  %-20s %8s %8s %9s %7s" % ("시각(UTC)", "중앙x", "p75", "가중평균", "n"))
    for b, g in s.groupby("blk"):
        gv, gw = g["lev_value"].to_numpy(), g["position_value"].to_numpy()
        print("  %-20s %7.1fx %7.1fx %8.2fx %7d"
              % (str(b)[:16], wq(gv, gw, .5), wq(gv, gw, .75),
                 float(np.sum(gv * gw) / np.sum(gw)), len(g)))

    print("\n  (c) 현재가 거리별 — 근접 물량이 고레버리지인가")
    print("  %-12s %8s %8s %9s %12s %7s"
          % ("|u| 구간", "중앙x", "p75", "가중평균", "명목가$", "n"))
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)):
        g = s[(s["u"].abs() >= lo) & (s["u"].abs() < hi)]
        if len(g) < 20:
            continue
        gv, gw = g["lev_value"].to_numpy(), g["position_value"].to_numpy()
        print("  %4.0f~%4.0f%%     %7.1fx %7.1fx %8.2fx %12.4g %7d"
              % (100 * lo, 100 * hi, wq(gv, gw, .5), wq(gv, gw, .75),
                 float(np.sum(gv * gw) / np.sum(gw)), float(gw.sum()), len(g)))

    print("\n--- 3. 판정 ---")
    med_by_coin = [wq(s[s.coin == c]["lev_value"].to_numpy(),
                      s[s.coin == c]["position_value"].to_numpy(), .5)
                   for c in top if len(s[s.coin == c]) >= 20]
    med_by_blk = [wq(g["lev_value"].to_numpy(), g["position_value"].to_numpy(), .5)
                  for _, g in s.groupby("blk") if len(g) >= 20]
    for lab, arr in (("코인 간", med_by_coin), ("시간 간", med_by_blk)):
        arr = np.array([x for x in arr if np.isfinite(x)])
        if arr.size >= 2:
            print("  %s 중앙 레버리지 산포: 평균 %.2fx  sd %.2fx  변동계수 %.3f  범위 [%.1f, %.1f]"
                  % (lab, arr.mean(), arr.std(ddof=1), arr.std(ddof=1) / arr.mean(),
                     arr.min(), arr.max()))
    print("  변동계수가 작으면(<0.3) 단일 분포로 다뤄도 되고 A 의 자유모수가 실제로 고정된다.")
    print("  크면 코인별/레짐별로 따로 둬야 하고, 그만큼 과적합 위험이 오른다.")
    print("\n  *** 표본이 2일이다. 시간 안정성은 '레짐 간'이 아니라 '이틀 안'만 본 것이다. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
