# -*- coding: utf-8 -*-
"""대안 A 의 빌더 — 공개 데이터만으로 청산 히트맵 L_hat(p, t) 을 재구성한다.

원리 (코호트 누적)
  시각 tau 에 가격 p(tau) 에서 열린 물량은, 청산거리 분포 f_R 에 따라 청산가로 흩어진다.
  그것을 과거 전체에 대해 더하면 현재 시점의 지도가 된다.

      L_hat(p, t) = sum_{tau<t}  dOI+(tau) x p(tau) x f_R( p / p(tau) ) x S(t-tau)

  dOI+ : OI 증가분(코인 수) = 그 시각에 새로 열린 포지션
  p(tau): 그 시각 가격 = 진입가 대용
  f_R  : 청산거리 경험분포 (liq_distance.py 에서 측정. 롱/숏 따로)
  S    : 자발적 종료 생존함수

  *** 교과서 공식 p_liq = p_entry(1-1/L) 을 쓰지 않는다. |오차|<10% 가 19.3% 뿐이다
      (lev_dist.py). f_R 경험분포를 통째로 쓰면 레버리지를 명시적으로 다룰 필요가 없다. ***

왜 이 경로인가
  HL 실측 L(p) 는 phi~0.05 라 실현 청산 가격대의 70% 가 공백이다(q1_representativeness).
  반면 이 재구성은 **바이낸스 자체 OI/가격**을 쓰므로 시장 전체를 본다.
  그리고 2020-09 까지 소급되어 백테스트가 가능하다.

정규화
  모든 열린 포지션은 언젠가 어딘가에서 청산되거나 종료된다. 따라서

      sum_p L_hat(p, t) = OI(t) x (격리 비중)

  이 제약이 S 의 수준을 고정한다. 모양(반감기)만 자유모수로 남는다.

한계 (반드시 함께 보고)
  - f_R 을 HL 에서 가져왔다. HL 은 교차 94.5% 로 사용자층이 바이낸스와 다르고,
    코인별 형태가 뒤집힐 수 있다(HL 은 BTC/ETH 가 고레버리지, 알트가 저레버리지).
    **정본은 Bybit 실현청산 역산이며 축적 중이다.** f_R 만 갈아끼우면 된다.
  - 롱/숏 분리는 LSR 로 근사한다. dOI 자체는 방향을 모른다
    (perp 는 모든 롱에 대응하는 숏이 있다).
  - 격리 비중은 HL 관측치(명목가 5.5%, 근접 19.7%)를 쓴다. 바이낸스 값은 모른다.

실행:
    python analysis/heatmap.py --symbol BTCUSDT
    python analysis/heatmap.py --symbol ETHUSDT --lookback 30 --half-life 7
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

BULK = os.path.join(C.DATA, "binance_bulk")
BAR_MS = 300_000
# 청산거리 격자 (진입가 대비). f_R 을 이 위에 올린다.
DIST_GRID = np.concatenate([np.arange(0.005, 0.05, 0.005),
                            np.arange(0.05, 0.20, 0.01),
                            np.arange(0.20, 0.65, 0.05)])
# 지도 출력 격자 (현재가 대비)
MAP_GRID = np.arange(-0.30, 0.3001, 0.0025)


def load_fr() -> dict:
    """HL 포지션에서 청산거리 경험분포 f_R 을 만든다. 격리만, 롱/숏 따로."""
    fs = sorted(glob.glob(os.path.join(C.DATA, "hl_positions", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/hl_positions 비어 있음 (f_R 측정 불가)")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for c in ("liquidation_px", "entry_px", "position_value"):
        d = d[np.isfinite(d[c]) & (d[c] > 0)]
    d = d[d["lev_type"] == "isolated"]
    d["dir"] = np.where(d["szi"] > 0, "long", "short")
    d["R"] = d["liquidation_px"] / d["entry_px"]
    d["dist"] = np.abs(d["R"] - 1.0)
    ok = ((d["dir"] == "long") & (d["R"] < 1)) | ((d["dir"] == "short") & (d["R"] > 1))
    d = d[ok & (d["dist"] > 0.002) & (d["dist"] < 0.65)]

    out = {}
    edges = np.concatenate([[0.0], DIST_GRID])
    for dr in ("long", "short"):
        g = d[d["dir"] == dr]
        h, _ = np.histogram(g["dist"], bins=edges, weights=g["position_value"])
        s = h.sum()
        out[dr] = (h / s) if s > 0 else np.full(len(h), np.nan)
        out[dr + "_n"] = len(g)
    out["mid"] = 0.5 * (edges[:-1] + edges[1:])
    return out


def load_oi(symbol: str) -> pd.DataFrame:
    kp = os.path.join(BULK, "klines_5m", "%s.parquet" % symbol)
    mp = os.path.join(BULK, "metrics", "%s.parquet" % symbol)
    if not (os.path.exists(kp) and os.path.exists(mp)):
        raise FileNotFoundError("bulk 없음: %s" % symbol)
    k = pd.read_parquet(kp)[["open_time", "close"]]
    m = pd.read_parquet(mp)[["open_time", "sum_open_interest",
                             "sum_open_interest_value",
                             "sum_toptrader_long_short_ratio"]]
    m = m[m["open_time"] % BAR_MS == 0]
    d = k.merge(m, on="open_time", how="inner").sort_values("open_time")
    d = d[np.isfinite(d["sum_open_interest"]) & (d["sum_open_interest"] > 0)]
    d = d[np.isfinite(d["close"]) & (d["close"] > 0)].reset_index(drop=True)
    return d


def build_map(d: pd.DataFrame, i: int, fr: dict, lookback_bars: int,
              half_life_bars: float, iso_share: float) -> tuple:
    """시각 i 에서 본 지도. (현재가, 롱맵, 숏맵) — MAP_GRID 위의 명목가."""
    lo = max(i - lookback_bars, 1)
    oi = d["sum_open_interest"].to_numpy()
    px = d["close"].to_numpy()
    lsr = d["sum_toptrader_long_short_ratio"].to_numpy()
    p_now = float(px[i])

    doi = np.diff(oi[lo - 1:i + 1])                 # 코인 단위 증가분
    p_tau = px[lo:i + 1]
    lsr_tau = lsr[lo:i + 1]
    age = np.arange(len(doi))[::-1].astype("float64")   # 최근이 0

    pos = doi > 0
    if not np.any(pos):
        return p_now, np.zeros(len(MAP_GRID)), np.zeros(len(MAP_GRID))
    ntl = doi[pos] * p_tau[pos]                     # 진입 시점 명목가
    surv = np.exp(-age[pos] / max(half_life_bars, 1e-6) * np.log(2.0))
    w = ntl * surv
    pe = p_tau[pos]

    # 롱/숏 분리: LSR = 롱/숏 계좌비. 없으면 50:50.
    r = lsr_tau[pos]
    r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
    f_long = r / (1.0 + r)

    L = np.zeros(len(MAP_GRID))
    Sm = np.zeros(len(MAP_GRID))
    for dr, frac, sign in (("long", f_long, -1.0), ("short", 1.0 - f_long, +1.0)):
        dens = fr[dr]
        if not np.all(np.isfinite(dens)):
            continue
        wd = w * frac
        for j, dist in enumerate(fr["mid"]):
            if dens[j] <= 0:
                continue
            # 청산가 = 진입가 x (1 + sign*dist).  현재가 대비 상대위치로 변환.
            p_liq = pe * (1.0 + sign * dist)
            u = p_liq / p_now - 1.0
            m = (u >= MAP_GRID[0]) & (u <= MAP_GRID[-1])
            if not np.any(m):
                continue
            idx = np.searchsorted(MAP_GRID, u[m])
            idx = np.clip(idx, 0, len(MAP_GRID) - 1)
            np.add.at(L if dr == "long" else Sm, idx, wd[m] * dens[j])

    tot = L.sum() + Sm.sum()
    if tot > 0:
        target = float(oi[i] * p_now * iso_share)   # 정규화: 지도 총합 = 격리 OI 명목가
        L *= target / tot
        Sm *= target / tot
    return p_now, L, Sm


def main() -> int:
    ap = argparse.ArgumentParser(description="reconstruct liquidation heatmap from public data")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--lookback", type=float, default=30.0, help="코호트 창 (일)")
    ap.add_argument("--half-life", type=float, default=7.0, help="생존 반감기 (일)")
    ap.add_argument("--iso-share", type=float, default=0.055,
                    help="격리 비중 (HL 명목가 실측 5.5%)")
    ap.add_argument("--n-snap", type=int, default=5, help="출력할 스냅샷 수")
    a = ap.parse_args()
    U.init_stdout()

    fr = load_fr()
    d = load_oi(a.symbol)
    lb = int(a.lookback * 288)
    hl = a.half_life * 288

    print("=" * 76)
    print("청산 히트맵 재구성 — %s" % a.symbol)
    print("=" * 76)
    print("f_R (HL 격리, 명목가 가중): 롱 n=%d / 숏 n=%d"
          % (fr["long_n"], fr["short_n"]))
    print("  롱 중앙거리 %.2f%% | 숏 중앙거리 %.2f%%"
          % (100 * fr["mid"][np.searchsorted(np.cumsum(fr["long"]), 0.5)],
             100 * fr["mid"][np.searchsorted(np.cumsum(fr["short"]), 0.5)]))
    print("바 %d개 | %s ~ %s | 코호트창 %.0f일 | 반감기 %.0f일 | 격리비중 %.1f%%"
          % (len(d), pd.to_datetime(d.open_time.iloc[0], unit="ms", utc=True).date(),
             pd.to_datetime(d.open_time.iloc[-1], unit="ms", utc=True).date(),
             a.lookback, a.half_life, 100 * a.iso_share))

    idxs = np.linspace(lb + 10, len(d) - 1, a.n_snap).astype(int)
    print("\n--- 스냅샷별 지도 요약 ---")
    print("  %-12s %10s %11s %11s %11s %11s %11s"
          % ("시각", "가격", "OI명목$", "롱 -1~0%$", "롱 -5~0%$", "숏 0~1%$", "숏 0~5%$"))
    for i in idxs:
        p, L, S = build_map(d, i, fr, lb, hl, a.iso_share)
        ts = pd.to_datetime(d.open_time.iloc[i], unit="ms", utc=True)
        oiv = float(d["sum_open_interest"].iloc[i] * p)
        m1 = (MAP_GRID >= -0.01) & (MAP_GRID < 0)
        m5 = (MAP_GRID >= -0.05) & (MAP_GRID < 0)
        s1 = (MAP_GRID > 0) & (MAP_GRID <= 0.01)
        s5 = (MAP_GRID > 0) & (MAP_GRID <= 0.05)
        print("  %-12s %10.2f %11.4g %11.4g %11.4g %11.4g %11.4g"
              % (str(ts)[:10], p, oiv, L[m1].sum(), L[m5].sum(),
                 S[s1].sum(), S[s5].sum()))

    # 마지막 스냅샷의 상세 프로파일
    i = int(idxs[-1])
    p, L, S = build_map(d, i, fr, lb, hl, a.iso_share)
    ts = pd.to_datetime(d.open_time.iloc[i], unit="ms", utc=True)
    print("\n--- 상세: %s  현재가 %.2f ---" % (str(ts)[:16], p))
    print("  %-14s %13s %13s" % ("현재가 대비", "롱청산(아래)$", "숏청산(위)$"))
    for lo, hi in ((-0.005, 0.0), (-0.01, -0.005), (-0.02, -0.01), (-0.03, -0.02),
                   (-0.05, -0.03), (-0.10, -0.05), (-0.20, -0.10)):
        m = (MAP_GRID >= lo) & (MAP_GRID < hi)
        print("  %5.1f ~ %5.1f%%  %13.4g %13s" % (100 * lo, 100 * hi, L[m].sum(), "-"))
    for lo, hi in ((0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.03),
                   (0.03, 0.05), (0.05, 0.10), (0.10, 0.20)):
        m = (MAP_GRID > lo) & (MAP_GRID <= hi)
        print("  %5.1f ~ %5.1f%%  %13s %13.4g" % (100 * lo, 100 * hi, "-", S[m].sum()))

    print("\n  지도 총합 $%.4g  (격리 OI 명목가 목표 $%.4g)"
          % (L.sum() + S.sum(),
             float(d["sum_open_interest"].iloc[i] * p * a.iso_share)))
    print("\n  *** f_R 이 HL 표본이라 바이낸스에 대해 편향 가능. 정본은 Bybit 역산.")
    print("  *** 다음: 실현 청산(Bybit)이 이 지도가 두꺼운 곳에서 났는지 검정.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
