# -*- coding: utf-8 -*-
"""H3-maker 검정 — 캐스케이드 아래 지정가를 미리 깔면 수익이 나는가.

배경
  H2 검정 결과 시장가 진입은 (a) 30bp 이상 비용에서 소멸, (b) 손익의 63%가 상위 10건,
  (c) 최대 수익일이 체결 불가능한 날이었다. 비용이 구속 조건이므로 taker가 아니라
  maker로 가야 한다 — 되돌림 수익은 유동성 공급의 대가이기 때문이다.

검정 대상 가설
  캐스케이드 트리거 직후, 현재가보다 D 아래에 지정가 매수를 깔아둔다.
  청산 물량이 얇은 구간까지 슈팅이 나면 체결되고, 거기서 반등을 먹는다.

핵심 반론 = 역선택 (이 스크립트가 측정하려는 것)
  지정가는 '가격이 거기까지 왔을 때'만 체결된다. 그리고 거기까지 왔다는 사실 자체가
  캐스케이드가 계속된다는 신호다. 즉 체결 표본은 무조건부 표본보다 나쁘다.
  따라서 의미 있는 통계는 '이벤트 평균 수익'이 아니라 **체결 조건부 수익**이다.

한계 (중요)
  5분봉 저가로 체결을 판정한다. 실제로는 (a) 바 내부 경로를 모르고 (b) 닿았다고 체결되는
  것도 아니다(큐 우선순위). 따라서 체결률은 과대, 체결 조건부 수익은 낙관 편향이다.
  1분봉/aggTrades로 내려가야 정직한 판정이 되며, 이 스크립트는 1차 형태 확인용이다.

실행:
    python analysis/event_study_h3_maker.py
    python analysis/event_study_h3_maker.py --k 8 --mode rel
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
from analysis.event_study_h2 import load, find_events, nw_tstat   # noqa: E402

HORIZONS = [3, 12, 48]
# 고정 깊이(%) — 트리거 바 종가 대비
DEPTHS_ABS = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
# 상대 깊이 — 트리거 바가 움직인 크기의 배수 ("다음 군집까지" 대용)
DEPTHS_REL = [0.25, 0.50, 1.00, 1.50, 2.00]


def simulate(df: pd.DataFrame, ev: pd.DataFrame, depth: float, mode: str,
             max_wait: int, cost_bps: float, fill_buffer_bps: float = 0.0) -> list[dict]:
    """각 이벤트에 지정가를 깔고 체결 여부와 체결 조건부 수익을 기록.

    fill_buffer_bps: 지정가를 이만큼 '뚫고 지나가야' 체결로 인정한다. 0이면 닿기만 해도
    체결(낙관). 바닥을 정확히 찍고 반등한 경우 큐 뒤에 있어 못 받았을 수 있으므로,
    관통 요구는 체결 판정을 보수화한다.
    """
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    contig = df["contig"].to_numpy()
    ret = df["ret"].to_numpy()
    n = len(df)
    out = []

    for r in ev.itertuples():
        i = r.i
        ref = close[i]
        if not np.isfinite(ref) or ref <= 0:
            continue
        d = depth if mode == "abs" else depth * abs(ret[i])
        if not np.isfinite(d) or d <= 0:
            continue
        # side=+1: 롱청산(가격 급락) -> 아래에 매수 지정가
        # side=-1: 숏청산(가격 급등) -> 위에 매도 지정가
        limit = ref * (1.0 - d) if r.side == 1 else ref * (1.0 + d)

        buf = fill_buffer_bps / 1e4
        trigger = limit * (1.0 - buf) if r.side == 1 else limit * (1.0 + buf)
        fill_j = None
        for step in range(1, max_wait + 1):
            j = i + step
            if j >= n or not contig[j]:
                break
            if (r.side == 1 and low[j] <= trigger) or (r.side == -1 and high[j] >= trigger):
                fill_j = j
                break

        rec = {"i": i, "side": r.side, "is_liq": r.is_liq, "depth": depth,
               "filled": fill_j is not None, "wait": (fill_j - i) if fill_j else np.nan,
               "open_time": int(df.open_time.iat[i]), "symbol": df.symbol.iat[i]}
        if fill_j is not None:
            for h in HORIZONS:
                t = fill_j + h - 1
                if t >= n:
                    rec["r%d" % h] = np.nan
                    continue
                rec["r%d" % h] = (close[t] / limit - 1.0) * r.side - cost_bps / 1e4
                if r.side == 1:
                    rec["mae%d" % h] = low[fill_j:t + 1].min() / limit - 1.0
                else:
                    rec["mae%d" % h] = 1.0 - high[fill_j:t + 1].max() / limit
        out.append(rec)
    return out


def run(symbols: list[str], k: float, doi_thr: float, min_gap: int, mode: str,
        max_wait: int, cost_bps: float, fill_buffer_bps: float = 0.0) -> pd.DataFrame:
    depths = DEPTHS_ABS if mode == "abs" else DEPTHS_REL
    recs = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        ev = find_events(df, k, doi_thr, min_gap)
        ev = ev[ev.is_liq]                       # 청산 동반 이벤트만
        U.log("%s: %d liq events" % (s, len(ev)))
        for d in depths:
            recs.extend(simulate(df, ev, d, mode, max_wait, cost_bps, fill_buffer_bps))
    return pd.DataFrame(recs)


def summarize(res: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col, mcol = "r%d" % horizon, "mae%d" % horizon
    rows = []
    for d, g in res.groupby("depth"):
        n_all = len(g)
        f = g[g.filled]
        x = f[col].to_numpy(dtype="float64") if col in f.columns else np.array([])
        x = x[np.isfinite(x)]
        row = {"depth": d, "n_events": n_all, "fill%": 100.0 * len(f) / max(n_all, 1),
               "n_filled": len(x)}
        if len(x):
            row.update({
                "mean_bp": 1e4 * x.mean(), "median_bp": 1e4 * np.median(x),
                "t_NW": nw_tstat(x, horizon), "win%": 100.0 * (x > 0).mean(),
                "p05_bp": 1e4 * np.percentile(x, 5),
                "mae_p05_bp": 1e4 * np.nanpercentile(f[mcol].to_numpy(dtype="float64"), 5),
                "avg_wait": f["wait"].mean(),
                # 체결되지 않으면 거래가 없다 -> 이벤트당 기대값은 체결률로 희석된다.
                "per_event_bp": 1e4 * x.mean() * len(f) / max(n_all, 1),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="H3-maker: resting limit orders below a cascade")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--mode", choices=["abs", "rel"], default="abs")
    ap.add_argument("--max-wait", type=int, default=12, help="bars the order stays live")
    ap.add_argument("--cost-bps", type=float, default=7.0, help="maker-in/taker-out round trip")
    ap.add_argument("--fill-buffer-bps", type=float, default=0.0,
                    help="require price to trade THROUGH the limit by this much before counting a fill")
    ap.add_argument("--horizons", nargs="*", type=int, default=[3, 12])
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS
    U.log("H3-maker: k=%.1f, dOI<=%.3f, mode=%s, live=%d bars, cost=%.1fbp"
          % (a.k, a.doi, a.mode, a.max_wait, a.cost_bps))

    res = run(symbols, a.k, a.doi, a.min_gap, a.mode, a.max_wait, a.cost_bps, a.fill_buffer_bps)
    if res.empty:
        U.log("no events")
        return 1
    U.atomic_write_parquet(res, os.path.join(C.DATA, "analysis", "h3_maker_%s.parquet" % a.mode))

    pd.set_option("display.width", 220)
    for h in a.horizons:
        print("\n=== horizon %d bars (%d min), depth mode=%s ===" % (h, h * 5, a.mode))
        print(summarize(res, h).to_string(index=False, float_format=lambda v: "%.1f" % v))
    print("\ndepth = %s" % ("fraction below trigger close" if a.mode == "abs"
                            else "multiple of the trigger bar's own move"))
    print("mean_bp = return CONDITIONAL ON FILL (adverse selection shows up here)")
    print("per_event_bp = mean_bp x fill%, i.e. expected value per event including no-fills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
