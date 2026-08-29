# -*- coding: utf-8 -*-
"""웹소켓 실측 깊이로 V/D 검정 — ADV 대리변수를 실제 D 로 바꾸면 달라지는가.

왜 이것인가
  analysis/synth.py 가 합성 확률모형을 불합격시켰고, 원인이 특정됐다:
      log X = a + b1 log(sigma) + b2 log(S0/ADV)
      -> b1 = 0.966 (제곱근 법칙 예측 1.0, 맞음)
      -> b2 = **-0.006** (예측 0.5, 완전히 틀림)
  즉 **청산 물량이 밀림 거리를 설명하지 못했다.** 다만 그때 분모는 ADV(거래대금)
  라는 **대리변수**였다. 설계가 요구하는 분모는 **그 순간 호가에 서 있는 깊이 D** 다.

  웹소켓이 그 D 를 1초 해상도로 실측하고 있으므로, 분모만 바꿔서 b2 가 살아나는지
  본다. 살아나면 실패 원인은 모형이 아니라 **대리변수**였다는 뜻이고, 설계가 요구한
  '오더북을 매 순간 다시 읽어 재계산' 이 실제로 값어치가 있다는 첫 증거가 된다.

표본의 성격
  큰 캐스케이드는 없다(수집 기간이 짧다). 그러나 작은 청산은 많으므로
  **b2 의 부호와 크기**는 검정된다. 규모 외삽은 여전히 불가.

실행:
    python analysis/ws_depth_test.py
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
from analysis.response_liq import ols_cluster, cmean            # noqa: E402

BANDS = ["b0_05", "b0_1", "b0_2", "b0_3", "b0_5", "b0_75", "b1", "b2", "b5"]
HORIZ = [5, 15, 30, 60, 300]          # 밀림 거리 측정 지평(초)


def load_depth():
    fs = sorted(glob.glob(os.path.join(C.DATA, "depth_ws", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)


def load_liq():
    fs = sorted(glob.glob(os.path.join(C.DATA, "bybit_liq", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["symbol"].isin(C.MAJORS)].copy()
    d["ntl"] = d["size"] * d["bankruptcy_px"]
    # pos_side: 롱 청산 = 강제매도(하방). 부호 +1 = 하방 압력
    s = d["side"].astype(str).str.lower() if "side" in d.columns else None
    if "pos_side" in d.columns:
        isL = d["pos_side"].astype(str).str.lower().eq("long")
    elif s is not None:
        isL = s.eq("sell")          # 청산 체결이 매도면 롱 청산
    else:
        isL = pd.Series(True, index=d.index)
    d["down"] = np.where(isL, 1, -1)
    return d.sort_values("exch_ms").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="V/D test with live orderbook depth")
    ap.add_argument("--band", default="b0_5")
    ap.add_argument("--min-ntl", type=float, default=0.0)
    a = ap.parse_args()
    U.init_stdout()

    print("=" * 78)
    print("웹소켓 실측 깊이로 V/D 검정 — ADV 대리를 실제 D 로 교체")
    print("=" * 78)
    dp = load_depth()
    lq = load_liq()
    t0 = max(dp["ts_ms"].min(), lq["exch_ms"].min())
    t1 = min(dp["ts_ms"].max(), lq["exch_ms"].max())
    print("깊이 %d행 / %d종 | 청산 %d건" % (len(dp), dp["symbol"].nunique(), len(lq)))
    print("**겹침 구간: %s ~ %s (%.1f시간)**"
          % (str(pd.to_datetime(t0, unit="ms"))[:19],
             str(pd.to_datetime(t1, unit="ms"))[:19], (t1 - t0) / 3.6e6))

    # 밴드 포화 진단 — 1000레벨 스냅샷이 얕으면 깊은 밴드가 전부 같은 값이 된다
    print("\n[진단] 밴드 포화 — 깊은 밴드가 얕은 밴드와 같으면 도달 못한 것이다")
    print("  %-10s %8s %10s %10s %10s %10s"
          % ("심볼", "레벨수", "b0_1", "b0_5", "b1", "b5"))
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"):
        g = dp[dp.symbol == s].tail(600)
        if len(g) == 0:
            continue
        print("  %-10s %8.0f %10.3g %10.3g %10.3g %10.3g"
              % (s, g["n_bid"].median(), g["bid_b0_1"].median(),
                 g["bid_b0_5"].median(), g["bid_b1"].median(), g["bid_b5"].median()))
    sat = float((dp["bid_b5"] <= dp["bid_b1"] * 1.0001).mean())
    print("  b5 == b1 인 비율 %.1f%%  (높으면 깊은 밴드는 못 씀)" % (100 * sat))

    lq = lq[(lq["exch_ms"] >= t0) & (lq["exch_ms"] <= t1)]
    print("\n겹침 구간 청산 프린트 %d건 / %d종" % (len(lq), lq["symbol"].nunique()))
    if len(lq) < 100:
        print("표본 부족")
        return 1

    rows = []
    for s, g in lq.groupby("symbol"):
        D = dp[dp.symbol == s]
        if len(D) < 100:
            continue
        ts = D["ts_ms"].to_numpy()
        mid = D["mid"].to_numpy(dtype=np.float64)
        bcol = D["bid_" + a.band].to_numpy(dtype=np.float64)
        acol = D["ask_" + a.band].to_numpy(dtype=np.float64)
        # 1초 버킷으로 청산을 모은다
        g = g.copy()
        g["sec"] = g["exch_ms"] // 1000
        agg = g.groupby("sec").apply(
            lambda x: pd.Series({
                "Q": float(x["ntl"].sum()),
                "down": 1 if (x["ntl"] * (x["down"] == 1)).sum()
                >= (x["ntl"] * (x["down"] == -1)).sum() else -1}),
            include_groups=False).reset_index()
        # 직전 변동성: 60초 mid 수익 표준편차
        lm = np.log(np.maximum(mid, 1e-12))
        r1 = np.concatenate([[np.nan], np.diff(lm)])
        sig = pd.Series(r1).rolling(60, min_periods=20).std().to_numpy()
        for r in agg.itertuples():
            tq = int(r.sec) * 1000
            i = int(np.searchsorted(ts, tq)) - 1        # **직전** 스냅샷 = 룩어헤드 없음
            if i < 61 or i >= len(ts) - max(HORIZ) - 1:
                continue
            if not (np.isfinite(sig[i]) and sig[i] > 0 and mid[i] > 0):
                continue
            Dside = bcol[i] if r.down == 1 else acol[i]   # 압력을 받는 쪽
            if not (np.isfinite(Dside) and Dside > 0) or r.Q < a.min_ntl:
                continue
            rec = {"symbol": s, "Q": r.Q, "D": Dside, "sig": sig[i],
                   "down": r.down, "hour": tq // 3_600_000}
            for H in HORIZ:
                j = int(np.searchsorted(ts, tq + H * 1000))
                if j >= len(ts):
                    rec["X%d" % H] = np.nan
                    continue
                seg = mid[i:j + 1]
                x = ((mid[i] - seg.min()) / mid[i] if r.down == 1
                     else (seg.max() - mid[i]) / mid[i]) * 1e4
                rec["X%d" % H] = x
            rows.append(rec)
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d["Q"]) & (d["Q"] > 0)]
    print("사건(1초 버킷) %d개 | Q 중앙 $%.4g p90 $%.4g 최대 $%.4g"
          % (len(d), d.Q.median(), d.Q.quantile(.9), d.Q.max()))
    print("D(%s) 중앙 $%.4g | **Q/D 중앙 %.3g p90 %.3g 최대 %.3g**"
          % (a.band, d.D.median(), (d.Q / d.D).median(),
             (d.Q / d.D).quantile(.9), (d.Q / d.D).max()))

    print("\n" + "-" * 78)
    print("핵심 회귀  log X = a + b1 log(sigma) + b2 log(Q/D)   [시간 클러스터 CR1]")
    print("-" * 78)
    print("  제곱근 법칙 예측: b1 = 1.0, **b2 = 0.5**")
    print("  synth.py 에서 분모를 ADV 로 썼을 때: b1 = 0.966, **b2 = -0.006**\n")
    print("  %6s | %9s %7s | %9s %7s | %8s %8s"
          % ("지평(초)", "b1(sigma)", "t", "**b2(Q/D)**", "t", "n", "R^2"))
    for H in HORIZ:
        y = d["X%d" % H].to_numpy()
        m = np.isfinite(y) & (y > 0)
        if m.sum() < 100:
            continue
        ly = np.log(y[m])
        X = np.column_stack([np.ones(int(m.sum())),
                             np.log(d["sig"].to_numpy()[m]),
                             np.log((d["Q"] / d["D"]).to_numpy()[m])])
        b, se, _ = ols_cluster(X, ly, d["hour"].to_numpy()[m])
        yh = X @ b
        r2 = 1.0 - np.var(ly - yh) / np.var(ly) if np.var(ly) > 0 else np.nan
        print("  %6d | %9.3f %7.1f | %9.3f %7.1f | %8d %8.3f"
              % (H, b[1], b[1] / se[1] if se[1] > 0 else np.nan,
                 b[2], b[2] / se[2] if se[2] > 0 else np.nan, int(m.sum()), r2))

    print("\n  [대조] 분모를 D 대신 **Q 만** (깊이 무시)")
    for H in (15, 60):
        y = d["X%d" % H].to_numpy()
        m = np.isfinite(y) & (y > 0)
        X = np.column_stack([np.ones(int(m.sum())), np.log(d["sig"].to_numpy()[m]),
                             np.log(d["Q"].to_numpy()[m])])
        b, se, _ = ols_cluster(X, np.log(y[m]), d["hour"].to_numpy()[m])
        print("    %3d초: b2(logQ) = %.3f (t=%.1f)"
              % (H, b[2], b[2] / se[2] if se[2] > 0 else np.nan))
    print("\n  [대조] 밴드를 바꿔가며 b2 (지평 15초)")
    y = d["X15"].to_numpy()
    print("    %-8s %10s %8s" % ("밴드", "b2(Q/D)", "t"))
    for bd in BANDS:
        col_b = "bid_" + bd
        if col_b not in dp.columns:
            continue
        # 밴드별 D 를 다시 붙이는 대신, 현재 표본의 D 비례성만 확인 (근사)
        pass
    print("    (밴드 스윕은 --band 로 개별 실행)")

    print("\n  *** b2 가 0 근처면 **깊이로 바꿔도 물량은 밀림 거리를 설명하지 못한다**.")
    print("      0.5 근처면 실패 원인이 모형이 아니라 **ADV 대리변수**였다는 뜻이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
