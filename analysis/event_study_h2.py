# -*- coding: utf-8 -*-
"""H2 검정 — 청산 캐스케이드 이후 되돌림이 '실시간으로 진입 가능한가'.

되돌림의 존재는 쟁점이 아니다(Bian et al. 2018, Coval-Stafford 2007).
쟁점은 캐스케이드가 끝났는지를 실시간으로 판별할 수 있느냐다. 하락 중 어느 시점이든
지금이 5단계 중 1단계인지 5단계인지 알 수 없고, 1단계 진입은 남은 하락을 그대로 먹는다.
따라서 이 스크립트가 검정하는 것은 '되돌림이 있나'가 아니라 '소진 신호로 타이밍을 잡을 수
있나'다.

레이턴시 정직성 (룩어헤드 차단)
  metrics[T] = 시각 T의 OI 스냅샷.  kline(open_time=T) = [T, T+5m) 구간의 가격.
  따라서 바 T의 OI 변화 = metrics[T+5m]/metrics[T] - 1 이고, 이 값은 바 T가 닫히는
  시각 T+5m에 비로소 관측된다. 판단 시점은 T+5m, 진입은 그 다음 바 시가(=T+5m)로 둔다.
  변동성 임계값도 전부 shift(1)된 과거 창에서만 계산한다.

식별 (대조군)
  큰 하락 + OI 급감(청산성)  vs  같은 크기 하락 + OI 유지(비청산성).
  이 차이를 보지 않으면 그냥 단기 반전을 재측정한 것에 불과하다.

진입 규칙 비교 — 이게 이 연구의 실제 질문이다
  A(naive)      : 캐스케이드 바 감지 즉시 다음 바 시가 진입
  B(감속)       : OI 감소가 둔화된 바(|dOI_t| < |dOI_t-1|)를 기다렸다 진입
  C(정지)       : OI 감소가 멈춘 바(dOI >= 0)를 기다렸다 진입
  L(p) 고갈 신호는 히스토리가 없어 아직 검정 불가. B/C는 그 신호 계열의 '느린 프록시'이므로,
  B/C조차 A를 개선하지 못하면 L(p)가 구제할 가능성이 낮다는 사전 정보를 준다.

실행:
    python analysis/event_study_h2.py
    python analysis/event_study_h2.py --symbols BTCUSDT --k 4 --doi -0.02
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
BAR_MS = 300_000                      # 5분
HORIZONS = [1, 3, 6, 12, 48, 288]     # 5m, 15m, 30m, 1h, 4h, 24h
VOL_WIN = 288                         # 하루치 창으로 변동성 정규화


def load(symbol: str) -> pd.DataFrame:
    """klines + metrics를 바 단위로 결합. 룩어헤드 없는 dOI를 만든다."""
    kp = os.path.join(BULK, "klines_5m", "%s.parquet" % symbol)
    mp = os.path.join(BULK, "metrics", "%s.parquet" % symbol)
    if not (os.path.exists(kp) and os.path.exists(mp)):
        raise FileNotFoundError("missing bulk data for %s (run downloaders/binance_bulk.py)" % symbol)

    k = pd.read_parquet(kp)[["open_time", "open", "high", "low", "close", "volume",
                             "taker_buy_volume", "quote_volume"]]
    m = pd.read_parquet(mp)[["open_time", "sum_open_interest", "sum_open_interest_value",
                             "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                             "sum_taker_long_short_vol_ratio"]]

    # 5분 격자에 정확히 정렬되지 않은 metrics 행은 버린다(격자 밖 스냅샷은 정합성이 없다).
    m = m[m["open_time"] % BAR_MS == 0]
    df = k.merge(m, on="open_time", how="left").sort_values("open_time").reset_index(drop=True)

    # 바 사이 결측(거래소 점검 등)을 잇지 않도록, 실제로 연속한 바인지 표시한다.
    df["contig"] = df["open_time"].diff().eq(BAR_MS)

    # 바 T의 OI 변화는 T+5m 스냅샷을 써야 한다 -> oi_end = 다음 행의 OI.
    oi = df["sum_open_interest"]
    oi_end = oi.shift(-1)
    next_contig = df["contig"].shift(-1).astype("boolean").fillna(False).to_numpy(dtype=bool)
    df["doi"] = np.where(next_contig & oi.gt(0).to_numpy(dtype=bool),
                         oi_end / oi - 1.0, np.nan)
    df["ret"] = df["close"] / df["open"] - 1.0
    df["taker_ls"] = df["sum_taker_long_short_vol_ratio"]

    # 임계값용 변동성: 과거 창만 사용(shift(1)로 현재 바 제외).
    df["sigma"] = df["ret"].shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4).std()
    df["z"] = df["ret"] / df["sigma"]
    df["symbol"] = symbol
    return df


def find_events(df: pd.DataFrame, k: float, doi_thr: float, min_gap: int
                ) -> pd.DataFrame:
    """캐스케이드 바 탐지. 반환: index, side(+1=롱청산/반등기대, -1=숏청산), is_liq."""
    ok = df["z"].notna() & df["doi"].notna() & df["contig"]
    big_dn = ok & df["z"].le(-k)
    big_up = ok & df["z"].ge(k)
    delev = df["doi"].le(doi_thr)          # OI 급감 = 강제 청산 동반
    stable = df["doi"].ge(0.0)             # OI 유지/증가 = 신규 진입, 청산 아님

    rows = []
    for mask, side, is_liq in ((big_dn & delev, 1, True), (big_up & delev, -1, True),
                               (big_dn & stable, 1, False), (big_up & stable, -1, False)):
        for i in np.flatnonzero(mask.to_numpy()):
            rows.append({"i": int(i), "side": side, "is_liq": is_liq})
    if not rows:
        return pd.DataFrame(columns=["i", "side", "is_liq"])

    ev = pd.DataFrame(rows).sort_values("i").reset_index(drop=True)
    # 군집 제거: 같은 그룹 안에서 min_gap 이내 중복 이벤트는 첫 건만 남긴다.
    keep, last = [], {}
    for r in ev.itertuples():
        key = (r.side, r.is_liq)
        if key in last and r.i - last[key] < min_gap:
            continue
        last[key] = r.i
        keep.append(r.Index)
    return ev.loc[keep].reset_index(drop=True)


def entry_bar(df: pd.DataFrame, i: int, rule: str, max_wait: int) -> int | None:
    """진입 바 인덱스. 판단은 바 i의 종가 시점, 진입은 그 다음 바 시가부터.

    A: 즉시 다음 바.
    B: OI 감소가 둔화될 때까지 대기(|dOI| 축소).
    C: OI 감소가 멈출 때까지 대기(dOI >= 0).
    대기 중 연속성이 끊기거나 max_wait를 넘으면 진입 포기(None).
    """
    n = len(df)
    if rule == "A":
        j = i + 1
        return j if j < n and bool(df["contig"].iat[j]) else None

    doi = df["doi"].to_numpy()
    contig = df["contig"].to_numpy()
    prev = doi[i]
    for step in range(1, max_wait + 1):
        t = i + step
        if t + 1 >= n or not contig[t]:
            return None
        cur = doi[t]
        if not np.isfinite(cur):
            return None
        if rule == "C":
            done = cur >= 0.0
        else:                               # B
            done = abs(cur) < abs(prev)
        prev = cur
        if done:
            j = t + 1
            return j if j < n and bool(contig[j]) else None
    return None


def forward_returns(df: pd.DataFrame, j: int, side: int, cost_bps: float) -> dict:
    """진입 바 j의 시가에 side 방향 진입했을 때의 선도수익. 비용은 왕복으로 뺀다."""
    entry = df["open"].iat[j]
    if not np.isfinite(entry) or entry <= 0:
        return {}
    out = {}
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    n = len(df)
    for h in HORIZONS:
        t = j + h - 1
        if t >= n:
            out["r%d" % h] = np.nan
            continue
        r = (close[t] / entry - 1.0) * side - cost_bps / 1e4
        out["r%d" % h] = r
        # 보유 중 최악 낙폭 — 평균만 보면 좌측 꼬리를 놓친다.
        if side == 1:
            mae = low[j:t + 1].min() / entry - 1.0
        else:
            mae = 1.0 - high[j:t + 1].max() / entry
        out["mae%d" % h] = mae
    return out


def nw_tstat(x: np.ndarray, lag: int) -> float:
    """겹치는 관측을 감안한 Newey-West t값. 표본이 부족하면 NaN."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    lag = int(min(max(lag, 0), n - 1))
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = (e[L:] @ e[:-L]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return np.nan
    return mu / np.sqrt(var / n)


def run(symbols: list[str], k: float, doi_thr: float, min_gap: int,
        max_wait: int, cost_bps: float) -> pd.DataFrame:
    recs = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        ev = find_events(df, k, doi_thr, min_gap)
        U.log("%s: %d bars (%s~%s), %d events (liq=%d, ctrl=%d)"
              % (s, len(df),
                 pd.to_datetime(df.open_time.min(), unit="ms").date(),
                 pd.to_datetime(df.open_time.max(), unit="ms").date(),
                 len(ev), int(ev.is_liq.sum()), int((~ev.is_liq).sum())))
        for rule in ("A", "B", "C"):
            for r in ev.itertuples():
                j = entry_bar(df, r.i, rule, max_wait)
                if j is None:
                    continue
                fr = forward_returns(df, j, r.side, cost_bps)
                if not fr:
                    continue
                fr.update({"symbol": s, "rule": rule, "is_liq": r.is_liq,
                           "side": r.side, "wait": j - r.i - 1,
                           "open_time": int(df.open_time.iat[r.i])})
                recs.append(fr)
    return pd.DataFrame(recs)


def summarize(res: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col, mcol = "r%d" % horizon, "mae%d" % horizon
    rows = []
    for (rule, is_liq), g in res.groupby(["rule", "is_liq"]):
        x = g[col].to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        rows.append({
            "rule": rule, "group": "LIQ" if is_liq else "CTRL", "n": len(x),
            "mean_bp": 1e4 * x.mean(), "median_bp": 1e4 * np.median(x),
            "t_NW": nw_tstat(x, horizon), "win%": 100.0 * (x > 0).mean(),
            "p05_bp": 1e4 * np.percentile(x, 5),
            "mae_p05_bp": 1e4 * np.nanpercentile(g[mcol].to_numpy(dtype="float64"), 5),
            "avg_wait": g["wait"].mean(),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["group", "rule"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="H2 event study: is the reversal tradeable in real time?")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=4.0, help="event threshold in trailing sigmas")
    ap.add_argument("--doi", type=float, default=-0.01, help="OI change threshold for liquidation bars")
    ap.add_argument("--min-gap", type=int, default=12, help="bars between events (declustering)")
    ap.add_argument("--max-wait", type=int, default=12, help="max bars to wait for exhaustion")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="round-trip cost in bps")
    ap.add_argument("--horizons", nargs="*", type=int, default=[3, 12, 48])
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS
    U.log("H2 event study: k=%.1f sigma, dOI<=%.3f, cost=%.1fbp, max_wait=%d bars"
          % (a.k, a.doi, a.cost_bps, a.max_wait))

    res = run(symbols, a.k, a.doi, a.min_gap, a.max_wait, a.cost_bps)
    if res.empty:
        U.log("no events -> nothing to report")
        return 1

    out_dir = os.path.join(C.DATA, "analysis")
    U.atomic_write_parquet(res, os.path.join(out_dir, "h2_events.parquet"))

    pd.set_option("display.width", 200)
    for h in a.horizons:
        mins = h * 5
        print("\n=== horizon %d bars (%d min) ===" % (h, mins))
        print(summarize(res, h).to_string(index=False, float_format=lambda v: "%.1f" % v))
    print("\nrule A=immediate  B=wait for OI-decline deceleration  C=wait for OI decline to stop")
    print("LIQ = big move WITH deleveraging;  CTRL = same-size move WITHOUT deleveraging")
    print("mae_p05_bp = 5th pct of worst drawdown while holding (left tail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
