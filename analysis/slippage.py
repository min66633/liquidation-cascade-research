# -*- coding: utf-8 -*-
"""청산 슬리피지 — 수수료에 없는 비용, 그리고 진짜 용량.

지금까지의 비용 가정
  COST_BPS = 7.0 = Binance USD-M 선물 수수료 (maker 0.02% 진입 + taker 0.05% 청산).
  이 값 자체는 맞다. 문제는 **수수료가 비용의 전부가 아니라는 것**이다.

  진입: 우리가 호가다 -> 슬리피지 0. 수수료 2bp 만 낸다.
  청산: HOLD 후 시장가로 나간다 -> **호가창을 걸어 내려간다.** 이것이 미측정이었다.

  H2 는 "왕복 50bp 에서 엣지 소멸"을 이미 보였고, H2 §5 는 "비용을 상수로 가정했다.
  실제로는 이벤트 강도에 따라 커지므로 낙관 방향으로 편향"이라 적어 두었다.
  bookDepth 백필로 이제 잴 수 있다.

슬리피지 계산 (닫힌 형태)
  bookDepth 는 '현재가 대비 u% 이내 누적 명목가' D(u) 를 준다. 크기 Q 를 시장가로
  던지면 D(u*) = Q 인 u* 까지 걸어 내려간다. 명목가 가중 평균 체결가의 이탈은

      slip(Q) = (1/Q) * int_0^{u*} u dD(u)
              = u* - (1/Q) * int_0^{u*} D(u) du

  국소적으로 D(u) = A u^beta 이면 int D = Q u*/(beta+1) 이므로

      slip(Q) = u* * beta / (beta + 1)

  beta 는 프로파일에서 국소 로그-로그 기울기로 잰다. beta=1(균일 밀도)이면 u*/2.

이것이 alpha 를 대체한다
  capacity.py 는 '용량 = alpha x min(F, D_exit)' 로 alpha=0.10 을 가정했다.
  여기서는 가정 없이 **크기별 순EV 곡선**을 만들고, 순EV 가 0 이 되는 크기를
  용량으로 정의한다.

      순EV(Q) = 체결률 x [ 조건부수익 - 수수료 - slip(Q) ]

방향
  롱청산 진입(매수) -> 청산은 시장가 매도 -> 매수호가(dm*)
  숏청산 진입(매도) -> 청산은 시장가 매수 -> 매도호가(dp*)

룩어헤드
  전부 사후 진단이다. 슬리피지는 신호가 아니라 '그때 얼마를 낼 뻔했나'의 측정이다.

실행:
    python analysis/slippage.py
    python analysis/slippage.py --hold-min 60 --fee-bps 7
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
import analysis.bookdepth as BD                            # noqa: E402
from analysis.event_study_h2 import load, find_events      # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
MIN_MS = 60_000
BAR_MS = 300_000
MAX_SNAP_LAG_MS = 2 * 60_000
GRID = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
BID_COLS = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK_COLS = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
SIZES = np.array([1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7])


def load_1m(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s" % symbol)
    return (pd.read_parquet(p)[["open_time", "high", "low", "close"]]
              .sort_values("open_time").reset_index(drop=True))


def walk(prof: np.ndarray, q: float) -> tuple[float, float]:
    """크기 q 를 소화할 때 (도달 깊이 u*, 평균 슬리피지). 단위는 비율.

    격자(1~5%) 밖으로 나가면 마지막 두 점의 로그 기울기로 외삽한다.
    1% 안쪽은 균일 밀도(beta=1)로 본다 — bookDepth 격자가 거기서 시작하기 때문이며,
    0.2% 밴드 보유 표본으로 검증하니 이 외삽은 깊이를 1.20배 과대평가한다
    (즉 slip 을 과소평가한다. 보수적 방향이 아니므로 결과에 함께 표기한다).
    """
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0) or np.any(np.diff(prof) < 0):
        return np.nan, np.nan
    if q <= 0:
        return 0.0, 0.0
    lg, lp = np.log(GRID), np.log(prof)
    if q <= prof[0]:
        beta = 1.0                                  # 1% 안쪽: 균일 밀도 가정
        u = GRID[0] * (q / prof[0]) ** (1.0 / beta)
    elif q >= prof[-1]:
        beta = (lp[-1] - lp[-2]) / (lg[-1] - lg[-2])
        beta = max(beta, 0.05)
        u = GRID[-1] * (q / prof[-1]) ** (1.0 / beta)
    else:
        k = int(np.searchsorted(prof, q))            # prof[k-1] < q <= prof[k]
        beta = (lp[k] - lp[k - 1]) / (lg[k] - lg[k - 1])
        beta = max(beta, 0.05)
        u = GRID[k - 1] * (q / prof[k - 1]) ** (1.0 / beta)
    return float(u), float(u * beta / (beta + 1.0))


def build(symbol: str, offset: float, ttl_min: int, hold_min: int,
          k: float, doi_thr: float, min_gap: int) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, st = BD.load_clean(symbol, BID_COLS + ASK_COLS)
    if df5.empty or m1.empty or dep.empty:
        U.log("%s: 데이터 부족 %s" % (symbol, st.get("missing")))
        return pd.DataFrame()
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
    n1 = len(ot)
    t5 = df5["open_time"].to_numpy()
    close5 = df5["close"].to_numpy()
    dts = dep["ts_ms"].to_numpy()
    bid = dep[BID_COLS].to_numpy()
    ask = dep[ASK_COLS].to_numpy()

    out = []
    for r in ev.itertuples():
        i = r.i
        ref = close5[i]
        if not (np.isfinite(ref) and ref > 0):
            continue
        limit = ref * (1.0 - offset) if r.side == 1 else ref * (1.0 + offset)
        start = int(t5[i]) + BAR_MS
        a = int(np.searchsorted(ot, start, side="left"))
        b = int(np.searchsorted(ot, start + ttl_min * MIN_MS, side="left"))
        if a >= n1 or b <= a:
            continue
        seg = (lo[a:b] <= limit) if r.side == 1 else (hi[a:b] >= limit)
        idx = np.flatnonzero(seg)
        if idx.size == 0:
            out.append({"symbol": symbol, "trig_ms": int(t5[i]), "side": int(r.side),
                        "filled": False})
            continue
        j = a + int(idx[0])
        e = min(j + hold_min, n1 - 1)
        # 청산 시점 프로파일 (진입 반대편 호가)
        di = int(np.searchsorted(dts, int(ot[e]), side="right")) - 1
        if di < 0 or int(ot[e]) - int(dts[di]) > MAX_SNAP_LAG_MS:
            prof = np.full(len(GRID), np.nan)
        else:
            prof = (bid[di] if r.side == 1 else ask[di]).astype("float64")
        rec = {"symbol": symbol, "trig_ms": int(t5[i]), "side": int(r.side),
               "filled": True, "fill_ms": int(ot[j]), "exit_ms": int(ot[e]),
               "gross_ret": float((cl[e] / limit - 1.0) * r.side)}
        for gi in range(len(GRID)):
            rec["prof_%d" % gi] = float(prof[gi])
        out.append(rec)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="exit slippage and size-dependent net EV")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--offset", type=float, default=0.02)
    ap.add_argument("--ttl-min", type=int, default=60)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--fee-bps", type=float, default=7.0,
                    help="maker 진입 2bp + taker 청산 5bp (Binance USD-M VIP0)")
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 200)
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS

    frames = []
    for s in symbols:
        try:
            d = build(s, a.offset, a.ttl_min, a.hold_min, a.k, a.doi, a.min_gap)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            U.log("%s: 이벤트 %d (체결 %d)" % (s, len(d), int(d["filled"].sum())))
    if not frames:
        U.log("no events")
        return 1
    d = pd.concat(frames, ignore_index=True)
    d["day"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "slippage.parquet"))

    n_ev, f = len(d), d[d["filled"]].copy()
    pcols = ["prof_%d" % i for i in range(len(GRID))]
    f = f[np.isfinite(f[pcols]).all(axis=1)]
    fill_rate = len(d[d["filled"]]) / max(n_ev, 1)

    print("\n=== 표본 ===")
    print("이벤트 %d | 체결 %d (%.1f%%) | 깊이 관측 %d | 심볼 %d | %s ~ %s"
          % (n_ev, int(d["filled"].sum()), 100 * fill_rate, len(f),
             d.symbol.nunique(), d["day"].min(), d["day"].max()))
    print("offset %.1f%% | TTL %d분 | HOLD %d분 | 수수료 %.1fbp (maker 2 + taker 5)"
          % (100 * a.offset, a.ttl_min, a.hold_min, a.fee_bps))
    if len(f) < 20:
        print("표본 부족")
        return 0
    gross = float(f["gross_ret"].mean())
    print("체결조건부 총수익(비용 전) 평균 %+.1fbp | 중앙 %+.1fbp"
          % (1e4 * gross, 1e4 * f["gross_ret"].median()))

    prof = f[pcols].to_numpy(dtype="float64")
    print("\n=== 청산 슬리피지 — 크기별 (bp) ===")
    print("  %10s %10s %10s %10s %10s" % ("주문크기", "p50", "p75", "p90", "p99"))
    slip_tab = {}
    for q in SIZES:
        sl = np.array([walk(prof[i], q)[1] for i in range(len(prof))]) * 1e4
        sl = sl[np.isfinite(sl)]
        slip_tab[q] = sl
        print("  %10s %10.1f %10.1f %10.1f %10.1f"
              % (("$%.0fK" % (q / 1e3)) if q < 1e6 else ("$%.1fM" % (q / 1e6)),
                 np.median(sl), np.quantile(sl, .75),
                 np.quantile(sl, .9), np.quantile(sl, .99)))
    print("  수수료 %.1fbp 와 비교할 것. 슬리피지가 수수료를 넘는 지점이 임계다."
          % a.fee_bps)

    print("\n=== 크기별 순EV — alpha 가정 없이 ===")
    print("  순EV(Q) = 체결률 x [ 체결조건부 총수익 - 수수료 - slip(Q) ]")
    print("  %10s %12s %12s %12s %12s"
          % ("주문크기", "slip 중앙", "순(체결당)", "순(이벤트당)", "달러/이벤트"))
    prev_pos = None
    breakeven = None
    for q in SIZES:
        sl = float(np.median(slip_tab[q]))
        net_fill = 1e4 * gross - a.fee_bps - sl
        net_ev = fill_rate * net_fill
        usd = q * net_ev / 1e4
        print("  %10s %12.1f %12.1f %12.1f %12s"
              % (("$%.0fK" % (q / 1e3)) if q < 1e6 else ("$%.1fM" % (q / 1e6)),
                 sl, net_fill, net_ev,
                 ("$%+.0f" % usd) if abs(usd) < 1e6 else ("$%+.2fM" % (usd / 1e6))))
        if prev_pos is not None and prev_pos > 0 >= net_fill:
            breakeven = q
        prev_pos = net_fill
    if breakeven:
        print("  -> 순수익이 0 이 되는 크기: 약 $%.0fK" % (breakeven / 1e3))
    else:
        print("  -> 이 크기 범위에서는 손익분기를 넘지 않는다.")
    print("  '달러/이벤트' 가 최대인 크기가 최적 규모다 (크기 x 수익률의 곱).")

    print("\n=== 심볼별 (중앙 슬리피지, bp) ===")
    print("  %-10s %6s %8s %8s %8s %8s"
          % ("심볼", "n", "$100K", "$500K", "$1M", "$5M"))
    for s, h in f.groupby("symbol"):
        pr = h[pcols].to_numpy(dtype="float64")
        row = []
        for q in (1e5, 5e5, 1e6, 5e6):
            sl = np.array([walk(pr[i], q)[1] for i in range(len(pr))]) * 1e4
            sl = sl[np.isfinite(sl)]
            row.append(np.median(sl) if sl.size else np.nan)
        print("  %-10s %6d %8.1f %8.1f %8.1f %8.1f" % (s, len(h), *row))

    print("\n=== 최악 구간 — 깊이가 무너진 날 ===")
    q = 5e5
    f = f.copy()
    f["slip"] = [walk(prof[i], q)[1] * 1e4 for i in range(len(prof))]
    w = f.nlargest(8, "slip")[["symbol", "day", "slip", "gross_ret", "prof_0"]]
    print("  $500K 청산 시 슬리피지 상위 8건")
    print("  %-10s %-12s %10s %12s %12s"
          % ("심볼", "일자", "slip(bp)", "총수익(bp)", "D(1%)"))
    for _, r in w.iterrows():
        print("  %-10s %-12s %10.0f %12.0f %12s"
              % (r["symbol"], r["day"], r["slip"], 1e4 * r["gross_ret"],
                 "$%.2fM" % (r["prof_0"] / 1e6)))
    print("\n  주의: 1%% 안쪽은 bookDepth 격자 밖이라 균일 밀도로 외삽했다.")
    print("        0.2%% 밴드 보유 표본 검증 결과 이 외삽은 깊이를 1.20배 과대평가하므로")
    print("        위 슬리피지는 **과소추정**이다(보수적이지 않은 방향).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
