# -*- coding: utf-8 -*-
"""거래소 청산 피드의 스로틀 편향 측정.

배경
  Binance forceOrder 는 '심볼당 초당 1건' 스냅샷이라 청산이 과소집계된다고 알려져 있고,
  실측으로도 전 종목 90초 구독에 0건이었다(2026-07-31). Bybit 는 2025-02-25부터
  allLiquidation 전건을 준다. 이제 Tardis 무료 샘플로 **같은 날·같은 심볼**을 나란히
  놓을 수 있으므로, 문헌 인용이 아니라 직접 측정한다.

무엇을 재는가
  1) 건수비 / 명목가비  (Binance ÷ Bybit)  — 절대 수준의 과소집계 정도
  2) **강도 의존성** — 청산이 격렬한 구간일수록 누락이 커지는가.
     이게 핵심이다. 상수 배율이면 스케일 보정으로 끝나지만, 강도에 따라 변하면
     캐스케이드(가장 중요한 구간)에서 체계적으로 왜곡된다.
  3) 초당 건수 상한 검증 — Binance 가 실제로 초당 1건에 걸려 있는가

주의
  두 거래소의 시장 규모가 다르므로 비율의 절대값 자체는 '스로틀'만이 아니라
  '시장점유 차이'도 반영한다. 따라서 (2)의 강도 의존성이 스로틀의 직접 증거다.

실행:
    python analysis/feed_bias.py
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

MULTI = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")
FULL_FEED_FROM_MS = 1740441600000        # 2025-02-25, Bybit allLiquidation 전환


def load(majors_only: bool = True) -> pd.DataFrame:
    d = pd.read_parquet(MULTI)
    if majors_only:
        d = d[d["symbol"].isin(C.MAJORS)]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="measure exchange feed throttling bias")
    ap.add_argument("--bucket-sec", type=int, default=60)
    ap.add_argument("--all-symbols", action="store_true")
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 200)
    d = load(not a.all_symbols)
    # Bybit 전건 구간만 비교 대상 (그 이전은 Bybit도 스로틀이라 대조가 성립 안 함)
    d = d[d["ts_ms"] >= FULL_FEED_FROM_MS]
    if d.empty:
        U.log("no post-2025-02-25 data")
        return 1

    d = d.copy()
    d["day"] = pd.to_datetime(d["ts_ms"], unit="ms").dt.date

    print("=== 1) 거래소별 총량 (2025-02-25 이후, 메이저 21종) ===")
    g = d.groupby("exchange").agg(n=("notional", "size"),
                                  musd=("notional", lambda s: s.sum() / 1e6),
                                  days=("day", "nunique"), syms=("symbol", "nunique"))
    print(g.round(1).to_string())

    base = "bybit"
    tot = g["musd"]
    if base in tot.index:
        print()
        for ex in tot.index:
            if ex != base:
                print("  %-16s / bybit  건수 %.2fx   명목가 %.2fx"
                      % (ex, g.loc[ex, "n"] / g.loc[base, "n"],
                         tot[ex] / tot[base]))

    print()
    print("=== 2) 강도 의존성 — 격렬할수록 더 누락되는가 (핵심) ===")
    step = a.bucket_sec * 1000
    d["bucket"] = (d["ts_ms"] // step) * step
    piv = (d.groupby(["symbol", "bucket", "exchange"])["notional"].sum()
             .unstack(fill_value=0.0))
    for ex in ("bybit", "binance-futures", "okex-swap"):
        if ex not in piv.columns:
            piv[ex] = 0.0
    piv = piv[piv["bybit"] > 0].copy()
    if len(piv) < 100:
        print("  표본 부족 (%d)" % len(piv))
        return 0

    # Bybit(전건) 강도를 기준으로 분위를 나누고, 그 안에서 Binance 비율을 본다
    piv["q"] = pd.qcut(piv["bybit"].rank(method="first"), 6, labels=False)
    t = piv.groupby("q").apply(
        lambda g: pd.Series({
            "n_buckets": len(g),
            "bybit_med_usd": g["bybit"].median(),
            "binance/bybit": (g["binance-futures"].sum() / g["bybit"].sum()),
            "okx/bybit": (g["okex-swap"].sum() / g["bybit"].sum()),
        }), include_groups=False)
    print(t.round(3).to_string())
    print("  예측: 스로틀이 있으면 강도가 셀수록 binance/bybit 비율이 **낮아진다**")

    print()
    print("=== 3) Binance 초당 건수 상한 검증 ===")
    b = d[d["exchange"] == "binance-futures"].copy()
    if not b.empty:
        b["sec"] = b["ts_ms"] // 1000
        per = b.groupby(["symbol", "sec"]).size()
        print("  심볼-초 단위 건수 분포:", dict(per.value_counts().head(6).sort_index()))
        print("  초당 2건 이상 비율: %.2f%%" % (100 * (per > 1).mean()))
        print("  (문서상 심볼당 초당 1건 스냅샷 -> 1건 비중이 압도적이어야 함)")
    by = d[d["exchange"] == "bybit"].copy()
    if not by.empty:
        by["sec"] = by["ts_ms"] // 1000
        per2 = by.groupby(["symbol", "sec"]).size()
        print("  [대조] Bybit 초당 2건 이상 비율: %.2f%%  최대 %d건/초"
              % (100 * (per2 > 1).mean(), per2.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
