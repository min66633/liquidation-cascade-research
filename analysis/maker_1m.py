# -*- coding: utf-8 -*-
"""H3-maker 정밀판 — 1분봉 체결 판정 + 적응형 배치거리.

용어 (이전 스크립트의 'depth'를 offset으로 개명 — 물량이 아니라 '가격 거리'다)
  offset = 트리거 바 종가에서 아래(롱청산 시)로 얼마나 떨어진 곳에 지정가를 거는가.
           offset=0.02, 트리거 종가 63,000 -> 61,740에 매수 지정가.
           청산 물량의 많고 적음과 무관한, 순수한 가격 거리다.

배치거리 결정 방식 (--mode)
  fixed : 항상 같은 % (기준선)
  bar   : 배수 x |트리거 바가 실제 움직인 폭|   -> 큰 캐스케이드일수록 멀리 건다
  sigma : 배수 x 직전 1일 5분봉 표준편차        -> 변동성 레짐에 연동
  doi   : 배수 x |OI 급감률|                     -> 청산 규모에 연동

왜 적응형인가
  고정 2%는 -1% 이벤트에는 너무 멀어 체결이 안 되고, -8% 캐스케이드에는 너무 가까워
  한복판에서 체결되어 남은 하락을 그대로 맞는다. 이벤트 크기에 비례시키면 '다음 얇은
  구간'까지의 거리를 조악하게나마 추종한다. 진짜 목표(L(p)가 얇아지는 지점에 배치)의
  대용품이며, L(p)는 이 기준선을 이겨야 존재 이유가 생긴다.

정밀도
  이벤트 탐지는 5분봉(metrics가 5분 해상도라 불가피).
  체결 판정과 선도수익은 1분봉으로 내린다 -> 5분봉 저가 판정의 낙관 편향을 줄인다.
  여전히 틱 단위는 아니므로 완전하진 않다(바 내부 경로·큐 우선순위 미반영).

실행:
    python analysis/maker_1m.py --mode fixed
    python analysis/maker_1m.py --mode bar --mults 0.5 1.0 1.5 2.0
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

BULK = os.path.join(C.DATA, "binance_bulk")
MIN_MS = 60_000
HOR_MIN = [15, 60, 240]                  # 보유 지평(분)
DEFAULT_MULTS = {
    "fixed": [0.005, 0.010, 0.015, 0.020, 0.030, 0.050],
    "bar": [0.25, 0.50, 1.00, 1.50, 2.00, 3.00],
    "sigma": [2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    "doi": [0.25, 0.50, 1.00, 2.00, 4.00, 8.00],
}


def load_1m(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s "
                                "(run: python downloaders/binance_bulk.py --interval 1m --no-metrics)" % symbol)
    df = pd.read_parquet(p)[["open_time", "open", "high", "low", "close"]]
    return df.sort_values("open_time").reset_index(drop=True)


def offset_for(mode: str, mult: float, bar_ret: float, sigma: float, doi: float) -> float:
    if mode == "fixed":
        return mult
    if mode == "bar":
        return mult * abs(bar_ret)
    if mode == "sigma":
        return mult * abs(sigma)
    if mode == "doi":
        return mult * abs(doi)
    raise ValueError(mode)


def simulate(ev: pd.DataFrame, df5: pd.DataFrame, m1: pd.DataFrame, mode: str,
             mult: float, live_min: int, cost_bps: float, fill_buf_bps: float,
             symbol: str) -> list[dict]:
    """1분봉으로 체결/수익을 판정. 주문은 트리거 바 종료 시점부터 live_min분간 유효."""
    ot = m1["open_time"].to_numpy()
    lo = m1["low"].to_numpy()
    hi = m1["high"].to_numpy()
    cl = m1["close"].to_numpy()
    n1 = len(ot)

    close5 = df5["close"].to_numpy()
    ret5 = df5["ret"].to_numpy()
    sig5 = df5["sigma"].to_numpy()
    doi5 = df5["doi"].to_numpy()
    t5 = df5["open_time"].to_numpy()

    out = []
    for r in ev.itertuples():
        i = r.i
        ref = close5[i]
        if not (np.isfinite(ref) and ref > 0):
            continue
        off = offset_for(mode, mult, ret5[i], sig5[i], doi5[i])
        if not np.isfinite(off) or off <= 0:
            continue

        limit = ref * (1.0 - off) if r.side == 1 else ref * (1.0 + off)
        buf = fill_buf_bps / 1e4
        trig = limit * (1.0 - buf) if r.side == 1 else limit * (1.0 + buf)

        # 주문 유효 구간: 트리거 5분바가 닫힌 직후부터
        start_ms = int(t5[i]) + 5 * MIN_MS
        a = int(np.searchsorted(ot, start_ms, side="left"))
        b = int(np.searchsorted(ot, start_ms + live_min * MIN_MS, side="left"))
        rec = {"i": i, "side": r.side, "mult": mult, "offset": off, "symbol": symbol,
               "open_time": int(t5[i]), "filled": False, "wait_min": np.nan}
        if a >= n1 or b <= a:
            out.append(rec)
            continue

        seg = lo[a:b] <= trig if r.side == 1 else hi[a:b] >= trig
        idx = np.flatnonzero(seg)
        if len(idx) == 0:
            out.append(rec)
            continue

        j = a + int(idx[0])
        rec["filled"] = True
        rec["wait_min"] = int((ot[j] - start_ms) // MIN_MS)
        for h in HOR_MIN:
            e = j + h
            if e >= n1:
                rec["r%d" % h] = np.nan
                continue
            rec["r%d" % h] = (cl[e] / limit - 1.0) * r.side - cost_bps / 1e4
            if r.side == 1:
                rec["mae%d" % h] = lo[j:e + 1].min() / limit - 1.0
            else:
                rec["mae%d" % h] = 1.0 - hi[j:e + 1].max() / limit
        out.append(rec)
    return out


def run(symbols: list[str], k: float, doi_thr: float, min_gap: int, mode: str,
        mults: list[float], live_min: int, cost_bps: float, fill_buf_bps: float,
        group: str = "liq") -> pd.DataFrame:
    recs = []
    for s in symbols:
        try:
            df5 = load(s)
            m1 = load_1m(s)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        ev = find_events(df5, k, doi_thr, min_gap)
        # 플라시보: CTRL = 같은 크기 급변인데 OI가 유지된(=강제청산 아닌) 이벤트
        ev = ev[ev.is_liq] if group == "liq" else ev[~ev.is_liq]
        U.log("%s: %d %s events | 1m bars %d (%s~%s)"
              % (s, len(ev), group, len(m1),
                 pd.to_datetime(m1.open_time.min(), unit="ms").date(),
                 pd.to_datetime(m1.open_time.max(), unit="ms").date()))
        for mult in mults:
            recs.extend(simulate(ev, df5, m1, mode, mult, live_min,
                                 cost_bps, fill_buf_bps, s))
    return pd.DataFrame(recs)


def summarize(res: pd.DataFrame, hor: int) -> pd.DataFrame:
    col, mcol = "r%d" % hor, "mae%d" % hor
    rows = []
    for mult, g in res.groupby("mult"):
        f = g[g.filled]
        x = f[col].to_numpy(dtype="float64") if col in f.columns else np.array([])
        x = x[np.isfinite(x)]
        row = {"mult": mult, "avg_offset%": 100.0 * g["offset"].mean(),
               "n_events": len(g), "fill%": 100.0 * len(f) / max(len(g), 1),
               "n_filled": len(x)}
        if len(x):
            row.update({
                "mean_bp": 1e4 * x.mean(), "median_bp": 1e4 * np.median(x),
                "t_NW": nw_tstat(x, max(hor // 5, 1)), "win%": 100.0 * (x > 0).mean(),
                "mae_p05_bp": 1e4 * np.nanpercentile(f[mcol].to_numpy(dtype="float64"), 5),
                "wait_min": f["wait_min"].mean(),
                "per_event_bp": 1e4 * x.mean() * len(f) / max(len(g), 1),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mult").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="maker backtest with 1m fill resolution")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--mode", choices=["fixed", "bar", "sigma", "doi"], default="fixed")
    ap.add_argument("--mults", nargs="*", type=float, default=None)
    ap.add_argument("--live-min", type=int, default=60, help="minutes the order stays live")
    ap.add_argument("--cost-bps", type=float, default=7.0)
    ap.add_argument("--fill-buffer-bps", type=float, default=0.0)
    ap.add_argument("--group", choices=["liq", "ctrl"], default="liq",
                    help="liq = move WITH deleveraging; ctrl = same-size move WITHOUT (placebo)")
    ap.add_argument("--horizons", nargs="*", type=int, default=[15, 60])
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS
    mults = a.mults if a.mults else DEFAULT_MULTS[a.mode]
    U.log("maker_1m: k=%.1f dOI<=%.3f mode=%s live=%dmin cost=%.1fbp buf=%.0fbp"
          % (a.k, a.doi, a.mode, a.live_min, a.cost_bps, a.fill_buffer_bps))

    res = run(symbols, a.k, a.doi, a.min_gap, a.mode, mults,
              a.live_min, a.cost_bps, a.fill_buffer_bps, a.group)
    if res.empty:
        U.log("no events")
        return 1
    U.atomic_write_parquet(res, os.path.join(C.DATA, "analysis", "maker1m_%s_%s.parquet" % (a.mode, a.group)))

    pd.set_option("display.width", 220)
    for h in a.horizons:
        print("\n=== hold %dmin | mode=%s | fill on 1m bars ===" % (h, a.mode))
        print(summarize(res, h).to_string(index=False, float_format=lambda v: "%.1f" % v))
    print("\nmult meaning: %s" % {
        "fixed": "offset = mult (constant fraction below trigger close)",
        "bar": "offset = mult x |trigger bar move|",
        "sigma": "offset = mult x trailing 5m sigma",
        "doi": "offset = mult x |OI drop|"}[a.mode])
    print("mean_bp = return CONDITIONAL ON FILL;  per_event_bp = mean_bp x fill%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
