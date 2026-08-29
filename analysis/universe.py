# -*- coding: utf-8 -*-
"""심볼·연도별 분해 — 이 전략은 어디서 돈을 버는가.

왜 필요한가
  지금까지 모든 결과가 5종(BTC/ETH/SOL/XRP/DOGE) 표본이다. 21종으로 넓히면
  (a) 검정력이 오르고 (b) 2024년 EV 가 -49.0bp 로 유일한 음수 연도인 것이
  심볼 특유인지 시스템 문제인지 판별된다.

**커버리지 검사가 먼저다** (2026-08-01 사고)
  5분봉/metrics 는 전 기간인데 1분봉이 95일치뿐인 심볼이 16개 있었다. 이벤트는
  5분봉으로 탐지하고 체결은 1분봉으로 판정하므로, 1분봉이 없는 날의 이벤트는
  전부 '체결 안 됨'으로 세어졌다. 13종이 체결 0건이 나왔고 나는 그것을 잠깐
  '확장하니 EV 가 60% 줄었다'고 읽었다. 데이터 결함이었다.
  => 이 스크립트는 분해 전에 **이벤트 기간이 1분봉 커버 안에 있는지** 확인하고,
     미달 심볼은 결과에서 빼되 조용히 빼지 않고 명시한다.

입력
  data/analysis/maker1m_fixed_liq.parquet   (analysis/maker_1m.py --group liq)
  data/analysis/maker1m_fixed_ctrl.parquet  (--group ctrl)

실행:
    python analysis/maker_1m.py --mode fixed --group liq  --symbols <21종>
    python analysis/maker_1m.py --mode fixed --group ctrl --symbols <21종>
    python analysis/universe.py
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

BULK = os.path.join(C.DATA, "binance_bulk")
CORE5 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
MIN_COVER = 0.90          # 이벤트의 이 비율 이상이 1분봉 범위 안에 있어야 신뢰


def coverage(symbols: list[str]) -> pd.DataFrame:
    """심볼별 1분봉 커버리지. 행수/기간이 아니라 **밀도**를 본다."""
    rows = []
    for s in symbols:
        p = os.path.join(BULK, "klines_1m", "%s.parquet" % s)
        if not os.path.exists(p):
            rows.append({"symbol": s, "n_1m": 0, "days_span": 0, "density": 0.0})
            continue
        d = pd.read_parquet(p, columns=["open_time"])
        if d.empty:
            rows.append({"symbol": s, "n_1m": 0, "days_span": 0, "density": 0.0})
            continue
        t0, t1 = int(d["open_time"].min()), int(d["open_time"].max())
        span_days = max((t1 - t0) / 86_400_000.0, 1e-9)
        rows.append({"symbol": s, "n_1m": len(d), "days_span": span_days,
                     "start": pd.to_datetime(t0, unit="ms", utc=True).date(),
                     "end": pd.to_datetime(t1, unit="ms", utc=True).date(),
                     # 하루 1440봉이 정상. 실제/기대 비율.
                     "density": len(d) / (span_days * 1440.0)})
    return pd.DataFrame(rows)


def agg(h: pd.DataFrame, hor: int) -> dict:
    col, mcol = "r%d" % hor, "mae%d" % hor
    f = h[h["filled"] & np.isfinite(h[col])]
    n = len(h)
    fr = len(f) / max(n, 1)
    m = 1e4 * f[col].mean() if len(f) else np.nan
    return {"n_ev": n, "n_fill": len(f), "fill%": 100 * fr,
            "cond_bp": m, "per_ev_bp": m * fr if np.isfinite(m) else np.nan,
            "win%": 100 * (f[col] > 0).mean() if len(f) else np.nan,
            "mae_p05": 1e4 * np.nanpercentile(f[mcol], 5) if len(f) else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser(description="per-symbol / per-year decomposition")
    ap.add_argument("--offset", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=15, choices=[15, 60, 240])
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 220)

    liq_p = os.path.join(C.DATA, "analysis", "maker1m_fixed_liq.parquet")
    if not os.path.exists(liq_p):
        U.log("missing %s — run analysis/maker_1m.py --group liq first" % liq_p)
        return 1
    d = pd.read_parquet(liq_p)
    d = d[np.isclose(d["mult"], a.offset)].copy()
    if d.empty:
        U.log("offset %.3f 이 표본에 없다" % a.offset)
        return 1
    d["dt"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    d["year"] = d["dt"].dt.year
    syms = sorted(d["symbol"].unique())

    # ------------------------------------------------------ 커버리지 게이트
    cov = coverage(syms)
    print("=== 0. 1분봉 커버리지 검사 (분해보다 먼저) ===")
    print("  density = 실제 1분봉 수 / (기간일수 x 1440). 1.0 이 정상.")
    bad = cov[cov["density"] < MIN_COVER]
    print(cov.sort_values("density").round(3).to_string(index=False))
    if len(bad):
        print("\n  ** 커버리지 미달 %d종 — 결과에서 제외한다 **" % len(bad))
        print("     %s" % ", ".join(bad["symbol"]))
        print("     (이벤트는 5분봉으로 탐지되지만 체결 판정용 1분봉이 없어")
        print("      전부 '체결 안 됨'으로 세어진다. 조용히 두면 EV 가 날조된다.)")
        d = d[~d["symbol"].isin(bad["symbol"])]
        syms = sorted(d["symbol"].unique())
    else:
        print("\n  전 종목 통과.")
    print("\n  분해 대상 %d종 / 이벤트 %d건 | offset %.1f%% | 지평 %d분"
          % (len(syms), len(d), 100 * a.offset, a.horizon))
    if not len(d):
        return 0

    # ------------------------------------------------------ 심볼별
    print("\n=== 1. 심볼별 (이벤트당 EV 내림차순) ===")
    rows = []
    for s, h in d.groupby("symbol"):
        r = agg(h, a.horizon)
        r["symbol"] = s
        r["구"] = "핵심5" if s in CORE5 else "확장"
        rows.append(r)
    t = pd.DataFrame(rows)
    t = t[["symbol", "구", "n_ev", "fill%", "n_fill", "cond_bp",
           "per_ev_bp", "win%", "mae_p05"]].sort_values("per_ev_bp", ascending=False)
    print(t.round(1).to_string(index=False))

    tot = t["n_ev"].sum()
    print("\n  그룹별 (이벤트 가중)")
    for g, h in t.groupby("구"):
        w = np.nansum(h["per_ev_bp"] * h["n_ev"]) / h["n_ev"].sum()
        print("    %-5s 심볼 %2d | 이벤트 %4d | 체결률 %.1f%% | 이벤트당 %+.1fbp"
              % (g, len(h), h["n_ev"].sum(),
                 100 * h["n_fill"].sum() / h["n_ev"].sum(), w))
    w_all = np.nansum(t["per_ev_bp"] * t["n_ev"]) / tot
    print("    전체  심볼 %2d | 이벤트 %4d | 이벤트당 %+.1fbp" % (len(t), tot, w_all))

    # ------------------------------------------------------ 누적 기여
    print("\n=== 2. 상위 몇 종이 전부인가 (이벤트 가중 누적) ===")
    t2 = t.dropna(subset=["per_ev_bp"]).copy()
    t2["contrib"] = t2["per_ev_bp"] * t2["n_ev"]
    t2 = t2.sort_values("contrib", ascending=False)
    t2["cum%"] = 100 * t2["contrib"].cumsum() / t2["contrib"].sum()
    print(t2[["symbol", "구", "n_ev", "per_ev_bp", "cum%"]].round(1).to_string(index=False))

    # ------------------------------------------------------ 연도별
    print("\n=== 3. 연도별 — 2024년 음수는 심볼 특유인가 ===")
    rows = []
    for yr, h in d.groupby("year"):
        r = agg(h, a.horizon)
        r["year"] = int(yr)
        r["n_sym"] = h["symbol"].nunique()
        rows.append(r)
    ty = pd.DataFrame(rows)[["year", "n_sym", "n_ev", "fill%", "n_fill",
                             "cond_bp", "per_ev_bp", "win%"]]
    print(ty.round(1).to_string(index=False))

    print("\n  연도 x 그룹 (이벤트당 bp)")
    piv = {}
    for (yr, g), h in d.assign(
            구=np.where(d["symbol"].isin(CORE5), "핵심5", "확장")).groupby(["year", "구"]):
        piv[(int(yr), g)] = agg(h, a.horizon)["per_ev_bp"]
    yrs = sorted({k[0] for k in piv})
    print("    %6s %10s %10s" % ("연도", "핵심5", "확장"))
    for yr in yrs:
        print("    %6d %10s %10s"
              % (yr,
                 ("%+.1f" % piv[(yr, "핵심5")]) if (yr, "핵심5") in piv and
                 np.isfinite(piv[(yr, "핵심5")]) else "-",
                 ("%+.1f" % piv[(yr, "확장")]) if (yr, "확장") in piv and
                 np.isfinite(piv[(yr, "확장")]) else "-"))

    # ------------------------------------------------------ 플라시보
    ctrl_p = os.path.join(C.DATA, "analysis", "maker1m_fixed_ctrl.parquet")
    if os.path.exists(ctrl_p):
        c = pd.read_parquet(ctrl_p)
        c = c[np.isclose(c["mult"], a.offset)]
        c = c[c["symbol"].isin(syms)]
        if len(c):
            rl, rc = agg(d, a.horizon), agg(c, a.horizon)
            print("\n=== 4. 플라시보 (같은 심볼 집합) ===")
            print("  %-6s %6s %8s %10s %12s" % ("", "n", "체결률", "조건부bp", "이벤트당bp"))
            for lab, r in (("LIQ", rl), ("CTRL", rc)):
                print("  %-6s %6d %7.1f%% %10.1f %12.1f"
                      % (lab, r["n_ev"], r["fill%"], r["cond_bp"], r["per_ev_bp"]))
            print("  격차 %+.1fbp — 0 근처면 청산 채널이 아니라 단순 반전을 잰 것이다."
                  % (rl["per_ev_bp"] - rc["per_ev_bp"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
