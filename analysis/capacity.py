# -*- coding: utf-8 -*-
"""전략 용량(capacity) — "체결됐다"는 판정이 말해주지 않는 것.

문제
  백테스트는 '저가가 지정가에 닿으면 체결'로 판정한다. 이는 **가격**에 대한 진술이지
  **물량**에 대한 진술이 아니다. 2025-10-10 SOL 의 -1% 누적 호가는 \\$0.07M 이었다
  (평소 \\$37.3M). 그 순간 "체결됐다"고 해서 의미 있는 규모를 넣을 수 있었던 것은 아니다.
  EV 와 무관하게 용량이 전략의 상한을 정한다.

두 개의 물리적 상한
  (1) 진입 — 유량.  지정매수는 **테이커 매도**가 와야만 체결된다. 따라서 체결 가능
      물량의 절대 상한은 우리 지정가 이하에서 실제로 발생한 테이커 매도 대금이다.
          F = sum( quote_volume - taker_buy_quote_volume )  over bars with low <= limit
      우리가 큐에 혼자 있지 않으므로 실제로 받는 것은 이보다 작다.
  (2) 청산 — 깊이.  HOLD 후 시장가로 나간다. 그때 대기 중인 반대편 호가 명목가가
      슬리피지 없이 나갈 수 있는 상한이다.  D_exit = bookDepth 의 -1% 누적 명목가.

  용량 = alpha x min(F, D_exit).  alpha 는 '우리가 그 흐름/호가의 몇 %까지 차지해도
  시장을 바꾸지 않는가'로, 기본 0.10 을 쓴다. alpha 는 가정이며 결과는 alpha 에 선형이다.

큐 맥락
  B = 트리거 시점 -2% 이내 누적 대기 매수(bookDepth dm2_0). 가격이 우리 지정가까지
  내려오려면 이만큼이 먼저 소진돼야 한다. F < B 면 우리 앞의 큐도 다 못 치운 것이므로
  '관통'이 아니라 '닿기만' 한 것에 가깝다. 진단용으로 같이 기록한다.
  (주의: 호가는 계속 리프레시되므로 B 는 실제 큐의 대용치일 뿐이다.)

룩어헤드
  B 는 트리거 바 **이전** 스냅샷, D_exit 는 청산 시각 이전 스냅샷을 쓴다. F 는 체결
  이후 구간의 실현 유량이므로 사후량이다 — 용량은 사후 진단이지 신호가 아니다.

데이터
  1분봉 2020-09~2026-07 (5종), bookDepth 185일(2023-01~2026-07).
  깊이가 없는 날의 이벤트는 유량 상한만 계산하고 깊이 항은 NaN 으로 둔다.

실행:
    python analysis/capacity.py
    python analysis/capacity.py --alpha 0.05 --offset 0.02
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
DEPTH_DIR = os.path.join(BULK, "book_depth")
MIN_MS = 60_000
BAR_MS = 300_000
# 깊이는 분 단위로 무너진다. 피드가 30초 간격이므로 2분보다 오래된 스냅샷은 쓰지 않는다.
MAX_SNAP_LAG_MS = 2 * 60_000


def load_1m_full(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s" % symbol)
    cols = ["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    d = pd.read_parquet(p)[cols].sort_values("open_time").reset_index(drop=True)
    # 테이커 매도 대금 = 전체 대금 - 테이커 매수 대금. 음수는 데이터 결함이므로 0 으로.
    d["sell_qv"] = (d["quote_volume"] - d["taker_buy_quote_volume"]).clip(lower=0.0)
    return d


DEPTH_COLS = ("ts_ms", "dm1_0", "dm2_0", "dp1_0", "dp2_0")


def load_depth(symbol: str) -> pd.DataFrame:
    """dm* = 매수호가(현재가 아래 누적), dp* = 매도호가(위 누적).

    방향이 중요하다 — 롱청산(우리가 매수)이면 청산은 시장가 **매도**라 매수호가(dm)를,
    숏청산(우리가 매도)이면 청산은 시장가 **매수**라 매도호가(dp)를 소모한다.

    2025년 고정 필드는 analysis/bookdepth 가 걸러낸다. 안 거르면 D_exit 가 상수라
    용량이 날조된다(리뷰 실측: 6년 최대 수익 사건의 D_exit 가 4.6시간 고정 구간 안).
    """
    d, st = BD.load_clean(symbol, list(DEPTH_COLS[1:]))
    if d.empty and st["missing"]:
        U.log("%s: bookDepth 사용 불가 %s -> 깊이 항 생략" % (symbol, st["missing"]))
    return d


def _snap(ts_arr: np.ndarray, vals: dict, when_ms: int) -> dict:
    """when_ms 이전의 마지막 스냅샷. 너무 오래됐으면 NaN (그 날은 커버 안 됨)."""
    out = {k: np.nan for k in vals}
    if ts_arr.size == 0:
        return out
    i = int(np.searchsorted(ts_arr, when_ms, side="right")) - 1
    if i < 0 or when_ms - int(ts_arr[i]) > MAX_SNAP_LAG_MS:
        return out
    for k, v in vals.items():
        out[k] = float(v[i])
    return out


def build(symbol: str, offset: float, ttl_min: int, hold_min: int,
          k: float, doi_thr: float, min_gap: int) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m_full(symbol)
    dep = load_depth(symbol)
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty or m1.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo = m1["low"].to_numpy()
    hi = m1["high"].to_numpy()
    cl = m1["close"].to_numpy()
    sq = m1["sell_qv"].to_numpy()
    qv = m1["quote_volume"].to_numpy()
    n1 = len(ot)

    t5 = df5["open_time"].to_numpy()
    close5 = df5["close"].to_numpy()

    dts = dep["ts_ms"].to_numpy() if not dep.empty else np.array([], dtype="int64")
    dvals = ({c: dep[c].to_numpy() for c in DEPTH_COLS[1:]} if not dep.empty
             else {c: np.array([]) for c in DEPTH_COLS[1:]})

    out = []
    for r in ev.itertuples():
        i = r.i
        ref = close5[i]
        if not (np.isfinite(ref) and ref > 0):
            continue
        trig_ms = int(t5[i])
        limit = ref * (1.0 - offset) if r.side == 1 else ref * (1.0 + offset)

        start = trig_ms + BAR_MS
        a = int(np.searchsorted(ot, start, side="left"))
        b = int(np.searchsorted(ot, start + ttl_min * MIN_MS, side="left"))
        rec = {"symbol": symbol, "trig_ms": trig_ms, "side": int(r.side),
               "ref": float(ref), "limit": float(limit), "filled": False}
        # 우리가 소모하는 쪽: 롱청산(매수 진입)이면 청산은 시장가 매도 -> 매수호가(dm)
        #                     숏청산(매도 진입)이면 청산은 시장가 매수 -> 매도호가(dp)
        c1, c2 = ("dm1_0", "dm2_0") if r.side == 1 else ("dp1_0", "dp2_0")
        # 트리거 시점의 큐 맥락 (사전 관측 가능)
        rec["B_queue"] = _snap(dts, dvals, trig_ms)[c2]
        if a >= n1 or b <= a:
            out.append(rec)
            continue

        seg = (lo[a:b] <= limit) if r.side == 1 else (hi[a:b] >= limit)
        idx = np.flatnonzero(seg)
        if idx.size == 0:
            out.append(rec)
            continue

        j = a + int(idx[0])
        e = min(j + hold_min, n1 - 1)
        rec["filled"] = True
        rec["fill_ms"] = int(ot[j])
        rec["wait_min"] = int((ot[j] - start) // MIN_MS)

        # (1) 유량 상한 — 지정가 이하(롱청산 기준)에서 발생한 테이커 매도 대금
        win = slice(j, e + 1)
        at_or_below = (lo[win] <= limit) if r.side == 1 else (hi[win] >= limit)
        # 롱청산이면 우리는 매수자이므로 '테이커 매도'가 우리를 체결시킨다.
        # 숏청산(우리가 매도자)이면 '테이커 매수'가 우리를 체결시킨다.
        flow = sq[win] if r.side == 1 else (qv[win] - sq[win])
        rec["F_fill"] = float(flow[0])
        rec["F_win"] = float(np.nansum(np.where(at_or_below, flow, 0.0)))
        rec["n_bar_below"] = int(np.nansum(at_or_below))

        # (2) 청산 시점 깊이 상한 (진입 방향의 반대쪽 호가를 시장가로 소모한다)
        rec["D_exit"] = _snap(dts, dvals, int(ot[e]))[c1]
        rec["exit_ms"] = int(ot[e])
        # 수익 (비용 전) — 용량 가중 손익을 보려면 필요
        rec["ret"] = float((cl[e] / limit - 1.0) * r.side)
        out.append(rec)
    return pd.DataFrame(out)


def q(x: np.ndarray, p: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if x.size else np.nan


def money(v: float) -> str:
    if not np.isfinite(v):
        return "     n/a"
    if v >= 1e6:
        return "$%6.1fM" % (v / 1e6)
    if v >= 1e3:
        return "$%6.0fK" % (v / 1e3)
    return "$%7.0f" % v


def main() -> int:
    ap = argparse.ArgumentParser(description="strategy capacity from flow and depth limits")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--offset", type=float, default=0.02)
    ap.add_argument("--ttl-min", type=int, default=60)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="우리가 차지해도 되는 흐름/호가 비중 (가정. 결과는 여기 선형)")
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
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "capacity.parquet"))

    f = d[d["filled"]].copy()
    print("\n=== 표본 ===")
    print("이벤트 %d (체결 %d, %.1f%%) | 심볼 %d | 일수 %d | %s ~ %s"
          % (len(d), len(f), 100 * len(f) / max(len(d), 1), d.symbol.nunique(),
             d["day"].nunique(), d["day"].min(), d["day"].max()))
    print("offset %.1f%% | TTL %d분 | HOLD %d분 | alpha %.2f"
          % (100 * a.offset, a.ttl_min, a.hold_min, a.alpha))
    # D_exit 는 체결된 건에만 기록되므로 체결 0 이면 컬럼 자체가 없다 -> 먼저 빠진다.
    if f.empty or "D_exit" not in f.columns:
        print("체결 이벤트 없음 — 용량 계산 불가")
        return 0
    cov = int(np.isfinite(f["D_exit"]).sum())
    print("bookDepth 커버: 체결 %d 중 %d건 (%.0f%%)"
          % (len(f), cov, 100 * cov / max(len(f), 1)))

    print("\n=== (1) 진입 유량 상한 — 지정가 이하 테이커 매도 대금 ===")
    for lab, col in (("체결 바 1분", "F_fill"), ("보유구간 합", "F_win")):
        x = f[col].to_numpy(dtype="float64")
        print("  %-10s  p10 %s  중앙 %s  p90 %s"
              % (lab, money(q(x, .1)), money(q(x, .5)), money(q(x, .9))))
    print("  지정가 이하에 머문 1분봉 수 중앙 %.0f개 (HOLD %d분 중)"
          % (np.nanmedian(f["n_bar_below"]), a.hold_min))

    print("\n=== (2) 청산 깊이 상한 — 청산 시각 -1% 누적 호가 ===")
    x = f["D_exit"].to_numpy(dtype="float64")
    print("  p10 %s  중앙 %s  p90 %s" % (money(q(x, .1)), money(q(x, .5)), money(q(x, .9))))

    print("\n=== 용량 = alpha x min(F_win, D_exit) ===")
    f["cap"] = a.alpha * np.minimum(f["F_win"], f["D_exit"])
    f["bind"] = np.where(f["F_win"] <= f["D_exit"], "유량", "깊이")
    x = f["cap"].to_numpy(dtype="float64")
    print("  이벤트당  p10 %s  중앙 %s  p90 %s  최소 %s"
          % (money(q(x, .1)), money(q(x, .5)), money(q(x, .9)),
             money(np.nanmin(x)) if np.isfinite(x).any() else float("nan")))
    ok = f[np.isfinite(f["cap"])]
    if len(ok):
        print("  구속 조건:  유량 %.0f%%   깊이 %.0f%%"
              % (100 * (ok["bind"] == "유량").mean(), 100 * (ok["bind"] == "깊이").mean()))

    print("\n  심볼별 이벤트당 용량 중앙값")
    t = (f.groupby("symbol")
           .agg(n=("cap", lambda s: int(np.isfinite(s).sum())),
                cap_med=("cap", "median"),
                F_med=("F_win", "median"),
                D_med=("D_exit", "median")))
    for s, r in t.iterrows():
        print("    %-9s n=%3d   용량 %s   (유량 %s / 깊이 %s)"
              % (s, r["n"], money(r["cap_med"]), money(r["F_med"]), money(r["D_med"])))

    print("\n=== 캐스케이드일에는 어떻게 되나 ===")
    big = f[f["day"] == "2025-10-10"]
    if big.empty:
        print("  2025-10-10 에 체결 이벤트 없음")
    else:
        for _, r in big.iterrows():
            print("  %-9s 유량 %s  깊이 %s  ->  용량 %s   (평상시 중앙 %s)"
                  % (r["symbol"], money(r["F_win"]), money(r["D_exit"]),
                     money(r["cap"]), money(float(t.loc[r["symbol"], "cap_med"]))))

    print("\n=== 용량 제약 하의 전략 규모 ===")
    ev_per_year = len(f) / max((pd.to_datetime(d["day"]).max()
                                - pd.to_datetime(d["day"]).min()).days / 365.25, 1e-9)
    med = q(f["cap"].to_numpy(dtype="float64"), .5)
    p10 = q(f["cap"].to_numpy(dtype="float64"), .1)
    print("  체결 이벤트 연 %.0f건 (5종 기준)" % ev_per_year)
    print("  이벤트당 중앙 용량 %s -> 이 규모로 돌리면 연 %s 회전"
          % (money(med), money(med * ev_per_year)))
    print("  하위 10%% 이벤트에서는 %s 밖에 못 넣는다 — 그 사건들이 큰 수익 구간이면"
          % money(p10))
    print("  백테스트 수익률은 실현 불가능하다. 아래에서 확인한다.")

    print("\n=== 용량 가중 수익 vs 균등 가중 수익 ===")
    g = f[np.isfinite(f["cap"]) & np.isfinite(f["ret"])].copy()
    if len(g) > 10:
        def diff_bp(h: pd.DataFrame) -> float:
            if h.empty or h["cap"].sum() <= 0:
                return np.nan
            return 1e4 * (float((h["cap"] * h["ret"]).sum() / h["cap"].sum())
                          - float(h["ret"].mean()))

        eq = float(g["ret"].mean())
        cw = float((g["cap"] * g["ret"]).sum() / g["cap"].sum())
        print("  균등 가중 평균수익      %+.1f bp  (n=%d, 비용 전)" % (1e4 * eq, len(g)))
        print("  용량 가중 평균수익      %+.1f bp  <- 실제로 넣을 수 있는 만큼만 반영"
              % (1e4 * cw))
        print("  차이 %+.1f bp — 음수면 '큰 수익 사건일수록 용량이 작다'는 뜻이다."
              % (1e4 * (cw - eq)))

        # 이 차이는 소수 사건이 좌우한다. 강건성을 반드시 같이 찍는다(리뷰 지적).
        loo = np.array([diff_bp(g.drop(index=i)) for i in g.index], dtype="float64")
        loo = loo[np.isfinite(loo)]
        if loo.size:
            worst = g.index[int(np.argmax([diff_bp(g.drop(index=i)) for i in g.index]))]
            print("  leave-one-out 범위 %+.1f ~ %+.1f bp   (가장 큰 영향: %s %s)"
                  % (loo.min(), loo.max(), g.loc[worst, "symbol"], g.loc[worst, "day"]))
        # 일자 클러스터 부트스트랩 — 같은 날 사건은 독립이 아니다
        days = g["day"].to_numpy()
        uniq = pd.unique(days)
        rs = np.random.default_rng(12345)
        boot = []
        for _ in range(2000):
            pick = rs.choice(uniq, size=uniq.size, replace=True)
            h = pd.concat([g[g["day"] == dd] for dd in pick], ignore_index=True)
            v = diff_bp(h)
            if np.isfinite(v):
                boot.append(v)
        if len(boot) > 100:
            lo_, hi_ = np.percentile(boot, [2.5, 97.5])
            print("  일자 클러스터 부트스트랩 95%% CI [%+.1f, %+.1f] bp  (일수 %d, 2000회)"
                  % (lo_, hi_, uniq.size))
            if lo_ < 0 < hi_:
                print("  -> CI 가 0 을 포함한다. 방향은 시사적이지만 확정된 것이 아니다.")
        r_hi = g.loc[g["ret"] >= g["ret"].quantile(0.9), "cap"].median()
        r_lo = g.loc[g["ret"] <= g["ret"].quantile(0.5), "cap"].median()
        print("  상위 10%% 수익 사건의 중앙 용량 %s  vs  하위 절반 %s"
              % (money(r_hi), money(r_lo)))
    else:
        print("  표본 부족 (n=%d)" % len(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
