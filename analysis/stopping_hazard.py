# -*- coding: utf-8 -*-
"""캐스케이드 정지 지점의 확률모델 — 가격축 위의 생존/해저드.

모델
  캐스케이드는 시간이 아니라 가격축을 따라 진행한다. 두꺼운 청산 군집을 통과하면
  강제매도가 나와 더 밀리고, 단위 가격당 강제매도 공급 L(p)가 그 가격대의 흡수력보다
  얇아지는 지점에서 멈춘다. 정확한 정지점은 못 맞히지만 분포는 추정할 수 있다.

    S(u) = P(캐스케이드가 상대가격 u까지 내려감)          <- 생존함수 = 지정가 체결확률
    h(u) = P(u에서 정지 | u까지 도달)                      <- 해저드
    S(u) = prod_{v>u} (1 - h(v))

  검정 가능한 예측: **h(u)는 L(u)의 감소함수다.** 연료가 두꺼우면 계속 가고 얇으면 멈춘다.

  지정가 배치의 기대값:
    EV(u) = S(u) x E[r | u에서 체결]
  고정 offset은 이 식을 공변량 없이 푼 특수해다. L(u)로 조건부화하면 이벤트마다
  다른 배치가 나온다.

과거 L(p) 복원
  청산가는 진입가와 실효 레버리지의 함수다:
      Lhat(p) = sum_x w(x) * V(p / (1 - 1/x))
  V(q) = 최근 거래량 프로파일(1분봉으로 계산), w(x) = 실효 레버리지 분포.

  중요: w(x)는 '명목' 레버리지가 아니라 '실효' 레버리지여야 한다. Hyperliquid 실측
  결과 포지션의 97.8%가 교차마진이고, 명목 중앙값 10x인데 청산거리로 역산한 실효
  레버리지 중앙값은 2.6x다. 명목값을 쓰는 Coinglass류 히트맵은 연료를 현재가에
  체계적으로 너무 가깝게 배치한다.

실행:
    python analysis/stopping_hazard.py
    python analysis/stopping_hazard.py --k 6 --window-min 120
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
from analysis.event_study_h2 import load, find_events   # noqa: E402
from analysis.maker_1m import load_1m                   # noqa: E402

MIN_MS = 60_000

# Hyperliquid 실측 실효 레버리지 분포 — **근접 연료에 한정해서** 뽑아야 한다.
#
# 전체 롱 포지션으로 뽑으면 중앙값 2.6배가 나오는데, 이는 청산가가 진입가보다 38%
# 아래라는 뜻이고, 그러면 "현재가 2% 아래의 연료"는 "현재가보다 59% 위에서 잡힌 진입"
# 에서 와야 한다. 그런 구간은 거래량 프로파일 범위 밖이라 복원값이 전부 0이 된다
# (첫 실행에서 실제로 이 오류가 났다).
#
# 실제로 문제되는 것은 현재가 -30% 이내의 연료이고(명목가의 52%), 그 부분집합의
# 실효 레버리지 분위(10/25/50/75/90)는 2.9 / 3.7 / 5.4 / 8.6 / 28.1 이다.
# 즉 근접 연료는 고레버리지 꼬리에서 나온다 — 최근 현재가 근처에서 잡힌 포지션들이다.
#
# 주의: 단일 스냅샷 롱 포지션 52건 기반이라 표본이 얇다. L(p)가 쌓이면 갱신할 것.
EFF_LEV = np.array([2.9, 3.7, 5.4, 8.6, 28.1])
EFF_W = np.array([0.15, 0.20, 0.25, 0.20, 0.20])


def price_grid(step: float, lo: float) -> np.ndarray:
    """상대가격 격자. u=-0.0025, -0.005, ... , lo 까지 (음수)."""
    n = int(round(abs(lo) / step))
    return -step * np.arange(1, n + 1)


def volume_profile(m1: pd.DataFrame, end_idx: int, lookback_min: int,
                   bins: np.ndarray) -> np.ndarray:
    """직전 lookback_min분의 거래량을 가격 구간에 배분. 포지션이 잡힌 위치의 대용."""
    a = max(0, end_idx - lookback_min)
    if end_idx - a < 60:
        return np.zeros(len(bins) - 1)
    px = m1["close"].to_numpy()[a:end_idx]
    vol = m1["volume"].to_numpy()[a:end_idx] if "volume" in m1.columns else np.ones(end_idx - a)
    h, _ = np.histogram(px, bins=bins, weights=vol)
    return h


def fuel_profile(vp: np.ndarray, centers: np.ndarray, p0: float,
                 grid_u: np.ndarray) -> np.ndarray:
    """거래량 프로파일 -> 청산 연료 프로파일 Lhat(u).

    진입가 q, 실효 레버리지 x인 롱은 q(1-1/x)에서 청산된다. 따라서 청산가 p에 쌓이는
    연료는 진입가 p/(1-1/x)의 거래량에서 온다. 레버리지 분포로 가중 합산한다.
    """
    out = np.zeros(len(grid_u))
    target = p0 * (1.0 + grid_u)                   # 청산가 후보들
    for x, w in zip(EFF_LEV, EFF_W):
        entry = target / (1.0 - 1.0 / x)           # 그 청산가를 만드는 진입가
        out += w * np.interp(entry, centers, vp, left=0.0, right=0.0)
    return out


def build(symbols: list[str], k: float, doi_thr: float, min_gap: int,
          window_min: int, lookback_min: int, step: float, lo: float) -> pd.DataFrame:
    """이벤트 x 격자 레벨 단위의 생존 패널을 만든다."""
    grid_u = price_grid(step, lo)
    rows = []
    for s in symbols:
        try:
            df5 = load(s)
            m1 = load_1m(s)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        ev = find_events(df5, k, doi_thr, min_gap)
        ev = ev[ev.is_liq & (ev.side == 1)]        # 롱청산(급락)만 — 방향 혼합을 피한다
        U.log("%s: %d long-liquidation events" % (s, len(ev)))

        ot = m1["open_time"].to_numpy()
        lo1 = m1["low"].to_numpy()
        cl1 = m1["close"].to_numpy()
        t5 = df5["open_time"].to_numpy()
        c5 = df5["close"].to_numpy()
        n1 = len(ot)

        for r in ev.itertuples():
            p0 = c5[r.i]
            if not (np.isfinite(p0) and p0 > 0):
                continue
            start = int(t5[r.i]) + 5 * MIN_MS
            a = int(np.searchsorted(ot, start, side="left"))
            b = int(np.searchsorted(ot, start + window_min * MIN_MS, side="left"))
            if a >= n1 or b <= a + 5:
                continue

            # 실제 도달 최저점
            run_min = np.minimum.accumulate(lo1[a:b])
            u_min = run_min[-1] / p0 - 1.0

            # 복원 연료 프로파일 (이벤트 시작 이전 데이터만 사용 -> 룩어헤드 없음)
            bins = np.linspace(p0 * (1.0 + lo) * 0.9, p0 * 1.10, 400)
            centers = 0.5 * (bins[1:] + bins[:-1])
            vp = volume_profile(m1, a, lookback_min, bins)
            if vp.sum() <= 0:
                continue
            fuel = fuel_profile(vp, centers, p0, grid_u)
            fuel = fuel / max(fuel.sum(), 1e-12)        # 이벤트 내 상대 분포로 정규화

            # 각 레벨의 도달 여부 + 도달 시점 이후 수익
            for j, u in enumerate(grid_u):
                reached = bool(u_min <= u)
                rec = {"symbol": s, "i": int(r.i), "open_time": int(t5[r.i]),
                       "u": float(u), "j": j, "reached": reached,
                       "fuel": float(fuel[j]),
                       "fuel_below": float(fuel[j:].sum()),
                       "fuel_above": float(fuel[:j].sum()),
                       "u_min": float(u_min)}
                if reached:
                    lim = p0 * (1.0 + u)
                    hit = a + int(np.argmax(lo1[a:b] <= lim))
                    for h in (15, 60):
                        e = hit + h
                        rec["r%d" % h] = (cl1[e] / lim - 1.0) if e < n1 else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def survival_table(panel: pd.DataFrame) -> pd.DataFrame:
    """레벨별 생존확률 S(u)와 조건부 수익."""
    g = panel.groupby("u")
    out = pd.DataFrame({
        "n_events": g["reached"].size(),
        "S_u": g["reached"].mean(),
    })
    r = panel[panel.reached].groupby("u")
    out["n_reached"] = r["reached"].size()
    for h in (15, 60):
        col = "r%d" % h
        if col in panel.columns:
            out["mean_r%d_bp" % h] = 1e4 * r[col].mean()
            out["EV%d_bp" % h] = out["S_u"] * out["mean_r%d_bp" % h]
    return out.reset_index()


def hazard_test(panel: pd.DataFrame) -> pd.DataFrame:
    """핵심 검정 — 레벨 u에 도달했을 때 '더 내려갈' 확률이 fuel(u)와 함께 커지는가.

    조건부 표본: 레벨 j에 도달한 이벤트만. 결과: 레벨 j+1에도 도달했는가(=계속 진행).
    """
    p = panel.sort_values(["symbol", "i", "j"]).copy()
    p["cont"] = p.groupby(["symbol", "i"])["reached"].shift(-1)
    cond = p[p.reached & p["cont"].notna()].copy()
    if cond.empty:
        return pd.DataFrame()
    cond["cont"] = cond["cont"].astype(bool)
    # 연료를 이벤트 내 분위로 층화 — 절대 스케일 차이를 제거한다.
    cond["fuel_q"] = cond.groupby(["symbol", "i"])["fuel"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["thin", "mid", "thick"])
        if s.notna().sum() >= 3 else pd.Series(["mid"] * len(s), index=s.index))
    t = cond.groupby("fuel_q", observed=True).agg(
        n=("cont", "size"), P_continue=("cont", "mean"),
        mean_fuel=("fuel", "mean")).reset_index()
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description="price-space stopping hazard for cascades")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--window-min", type=int, default=60)
    ap.add_argument("--lookback-min", type=int, default=43200, help="volume-profile lookback (min)")
    ap.add_argument("--step", type=float, default=0.0025)
    ap.add_argument("--lo", type=float, default=-0.10)
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS
    U.log("stopping hazard: k=%.1f window=%dmin grid=%.2f%%..%.0f%% lookback=%dd"
          % (a.k, a.window_min, 100 * a.step, 100 * a.lo, a.lookback_min // 1440))

    panel = build(symbols, a.k, a.doi, a.min_gap, a.window_min,
                  a.lookback_min, a.step, a.lo)
    if panel.empty:
        U.log("empty panel")
        return 1
    U.atomic_write_parquet(panel, os.path.join(C.DATA, "analysis", "hazard_panel.parquet"))

    pd.set_option("display.width", 200)
    st = survival_table(panel)
    print("\n=== S(u) 생존함수 = 지정가 체결확률, 그리고 EV(u) ===")
    show = st[st["u"] >= -0.06].copy()
    show["u%"] = 100 * show["u"]
    cols = ["u%", "n_events", "n_reached", "S_u", "mean_r15_bp", "EV15_bp", "mean_r60_bp", "EV60_bp"]
    print(show[[c for c in cols if c in show.columns]].to_string(index=False, float_format=lambda v: "%.2f" % v))

    print("\n=== 핵심 검정: 도달 후 '계속 진행할' 확률이 연료 두께와 함께 커지는가 ===")
    ht = hazard_test(panel)
    print(ht.to_string(index=False, float_format=lambda v: "%.4f" % v) if not ht.empty else "(no data)")
    print("\n예측: thin -> P(continue) 낮음(정지), thick -> 높음(계속). 단조가 아니면 복원 연료에 정보가 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
