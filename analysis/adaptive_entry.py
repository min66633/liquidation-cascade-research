# -*- coding: utf-8 -*-
"""적응형 진입 — 고정 offset 을 호가·유량 기반 조건부 배치로 대체할 수 있나.

왜 다시 하나 (Q2a 와 무엇이 다른가)
  Q2a(analysis/clearing.py)는 "수급 청산 회계가 실제 낙폭 X 를 **정확히 맞히는가**"를
  물었고 R2 = -0.17 로 실패했다. 그런데 **배치 규칙은 X 를 맞힐 필요가 없다.**
  고정 offset 보다 EV 가 나으면 된다. 잣대가 다른데 하나로 묶어 기각했다.

  노이즈가 많아도 방향만 맞으면 EV 는 개선될 수 있다. 여기서는 예측 정확도가 아니라
  **이벤트당 EV / MAE / 체결률**로 직접 비교한다.

지금 되는 것과 안 되는 것
  대기주문 D(u)  가격대별  -> bookDepth 1,304일. **있다**
  유량 M                  -> 1분봉 테이커. **있다**
  공급 L(p)      가격대별  -> Hyperliquid 1일치. **없다** (Q1 대기)
  따라서 여기서는 **수요측(호가) + 유량 스케일**만으로 배치를 정한다.
  공급 지도가 들어오면 같은 틀에 L(p) 를 더해 재실행한다.

배치 규칙
  fixed      u = 상수 (기준선)
  clear      u = D(u) = k x 예상유량  을 푸는 u   (회계 기반)
  depthmult  u = D(u) = k x D(1%)     을 푸는 u   (호가 상대 두께)
  sigma      u = k x 직전 변동성                   (참고용, H3 에서 이미 실패)

  예상유량 = (훈련구간 중앙 순흐름/OI명목가) x 이번 OI명목가.
  전부 **트리거 시점에 관측 가능**한 것만 쓴다.

정직성
  - 깊이 프로파일은 트리거 **이전** 스냅샷 (룩어헤드 차단)
  - 유량 계수는 훈련구간에서만 적합, 평가구간에 적용
  - 전 규칙을 **같은 이벤트 집합**에서 비교 (한 규칙만 표본이 다르면 비교 불가)

실행:
    python analysis/adaptive_entry.py
    python analysis/adaptive_entry.py --hold-min 60
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
from analysis.event_study_h2 import load, find_events, nw_tstat   # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
MIN_MS = 60_000
BAR_MS = 300_000
MAX_SNAP_LAG_MS = 2 * 60_000
GRID = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
BID = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
U_MIN, U_MAX = 0.003, 0.10        # 배치 가능 범위 (너무 얕으면 역선택, 너무 깊으면 미체결)


def load_1m(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s" % symbol)
    cols = ["open_time", "high", "low", "close", "quote_volume",
            "taker_buy_quote_volume"]
    d = pd.read_parquet(p)[cols].sort_values("open_time").reset_index(drop=True)
    d["buy_qv"] = d["taker_buy_quote_volume"].clip(lower=0.0)
    d["sell_qv"] = (d["quote_volume"] - d["taker_buy_quote_volume"]).clip(lower=0.0)
    return d


def depth_at_u(prof: np.ndarray, u: float) -> float:
    """누적 깊이. 1% 안쪽은 균일 밀도, 5% 밖은 로그 외삽."""
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0) or np.any(np.diff(prof) < 0):
        return np.nan
    if u <= GRID[0]:
        return float(prof[0] * u / GRID[0])
    if u <= GRID[-1]:
        return float(np.exp(np.interp(np.log(u), np.log(GRID), np.log(prof))))
    sl = (np.log(prof[-1]) - np.log(prof[-2])) / (np.log(GRID[-1]) - np.log(GRID[-2]))
    return float(prof[-1] * np.exp(sl * (np.log(u) - np.log(GRID[-1]))))


def u_for_depth(prof: np.ndarray, target: float) -> float:
    """누적 깊이가 target 이 되는 u. 배치 가능 범위로 자른다."""
    if not np.isfinite(target) or target <= 0:
        return np.nan
    if depth_at_u(prof, U_MAX) < target:
        return U_MAX
    if depth_at_u(prof, U_MIN) >= target:
        return U_MIN
    lo, hi = U_MIN, U_MAX
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if depth_at_u(prof, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build(symbol: str, ttl_min: int, hold_min: int, k: float, doi_thr: float,
          min_gap: int) -> pd.DataFrame:
    """이벤트별로 배치에 필요한 사전 관측치 + 1분봉 경로를 모은다."""
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, st = BD.load_clean(symbol, BID + ASK, verbose=False)
    if df5.empty or m1.empty or dep.empty:
        return pd.DataFrame()
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
    bq, sq = m1["buy_qv"].to_numpy(), m1["sell_qv"].to_numpy()
    n1 = len(ot)
    t5 = df5["open_time"].to_numpy()
    close5 = df5["close"].to_numpy()
    sig5 = df5["sigma"].to_numpy()
    ret5 = df5["ret"].to_numpy()
    oiv5 = df5["sum_open_interest_value"].to_numpy()
    dts = dep["ts_ms"].to_numpy()
    bid, ask = dep[BID].to_numpy(), dep[ASK].to_numpy()

    out = []
    for r in ev.itertuples():
        i = r.i
        p0 = close5[i]
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sig5[i]) and sig5[i] > 0):
            continue
        trig = int(t5[i])
        j0 = int(np.searchsorted(dts, trig, side="right")) - 1
        if j0 < 0 or trig - int(dts[j0]) > MAX_SNAP_LAG_MS:
            continue
        prof = (bid[j0] if r.side == 1 else ask[j0]).astype("float64")
        if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
            continue

        a = int(np.searchsorted(ot, trig + BAR_MS, side="left"))
        b = int(np.searchsorted(ot, trig + BAR_MS + ttl_min * MIN_MS, side="left"))
        if a >= n1 or b <= a or b + hold_min >= n1:
            continue

        rec = {"symbol": symbol, "trig_ms": trig, "side": int(r.side), "p0": float(p0),
               "sigma": float(sig5[i]), "bar_ret": float(abs(ret5[i])),
               "oiv": float(oiv5[i]) if np.isfinite(oiv5[i]) else np.nan,
               "a": a, "b": b}
        for gi in range(len(GRID)):
            rec["prof_%d" % gi] = float(prof[gi])
        # 사후 실현 순흐름 — 예상유량 계수 적합에만 쓰고 배치에는 절대 안 쓴다
        f = (sq[a:b] - bq[a:b]) if r.side == 1 else (bq[a:b] - sq[a:b])
        rec["NS_real"] = float(np.nansum(f))
        out.append(rec)
    if not out:
        return pd.DataFrame()
    d = pd.DataFrame(out)
    d["_m1"] = symbol            # 경로는 아래 simulate 에서 다시 읽는다
    return d


def simulate(d: pd.DataFrame, u_col: str, hold_min: int, cost_bps: float,
             m1cache: dict) -> pd.DataFrame:
    """배치거리 u_col 로 체결/수익 판정.

    itertuples 는 공백·%가 든 컬럼명을 _1, _2 로 바꿔 버리므로 getattr 로 못 꺼낸다.
    u 는 별도 배열로 뽑아 위치로 맞춘다.
    """
    res = []
    for sym, h in d.groupby("symbol"):
        m1 = m1cache[sym]
        ot = m1["open_time"].to_numpy()
        lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
        n1 = len(ot)
        uvals = h[u_col].to_numpy(dtype="float64")
        for pos, r in enumerate(h.itertuples()):
            u = uvals[pos]
            if not np.isfinite(u) or u <= 0:
                res.append({"idx": r.Index, "filled": False, "u": np.nan})
                continue
            limit = r.p0 * (1 - u) if r.side == 1 else r.p0 * (1 + u)
            seg = (lo[r.a:r.b] <= limit) if r.side == 1 else (hi[r.a:r.b] >= limit)
            idx = np.flatnonzero(seg)
            if idx.size == 0:
                res.append({"idx": r.Index, "filled": False, "u": u})
                continue
            j = r.a + int(idx[0])
            e = min(j + hold_min, n1 - 1)
            ret = (cl[e] / limit - 1.0) * r.side - cost_bps / 1e4
            mae = ((lo[j:e + 1].min() / limit - 1.0) if r.side == 1
                   else (1.0 - hi[j:e + 1].max() / limit))
            res.append({"idx": r.Index, "filled": True, "u": u,
                        "ret": float(ret), "mae": float(mae)})
    return pd.DataFrame(res).set_index("idx")


def summarize(d: pd.DataFrame, s: pd.DataFrame, hold_min: int) -> dict:
    f = s[s["filled"] & np.isfinite(s["ret"])]
    n = len(d)
    fr = len(f) / max(n, 1)
    if not len(f):
        return {"n_ev": n, "fill%": 0.0, "u_med%": np.nan, "cond_bp": np.nan,
                "per_ev_bp": np.nan, "win%": np.nan, "mae_p05": np.nan, "t_NW": np.nan}
    r = f["ret"].to_numpy()
    return {"n_ev": n, "fill%": 100 * fr, "u_med%": 100 * np.nanmedian(s["u"]),
            "cond_bp": 1e4 * r.mean(), "per_ev_bp": 1e4 * r.mean() * fr,
            "win%": 100 * (r > 0).mean(),
            "mae_p05": 1e4 * np.nanpercentile(f["mae"], 5),
            "t_NW": nw_tstat(r, max(hold_min // 5, 1))}


def main() -> int:
    ap = argparse.ArgumentParser(description="adaptive entry vs fixed offset")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--ttl-min", type=int, default=60)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--cost-bps", type=float, default=7.0)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 220)
    symbols = a.symbols if a.symbols else C.MAJORS

    frames, m1cache = [], {}
    for s in symbols:
        try:
            d = build(s, a.ttl_min, a.hold_min, a.k, a.doi, a.min_gap)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            m1cache[s] = load_1m(s)
            U.log("%s: %d 이벤트" % (s, len(d)))
    if not frames:
        U.log("no events")
        return 1
    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d = d.sort_values("dt").reset_index(drop=True)

    # 훈련/평가 분할 — 유량 계수는 훈련구간에서만
    cut = len(d) // 2
    tr = d.iloc[:cut]
    ok = tr[np.isfinite(tr["NS_real"]) & np.isfinite(tr["oiv"]) & (tr["oiv"] > 0)]
    c_flow = float((ok["NS_real"] / ok["oiv"]).median()) if len(ok) >= 20 else np.nan
    print("\n=== 표본 ===")
    print("이벤트 %d | 심볼 %d | %s ~ %s | TTL %d분 HOLD %d분 비용 %.0fbp"
          % (len(d), d.symbol.nunique(), d["dt"].min().date(), d["dt"].max().date(),
             a.ttl_min, a.hold_min, a.cost_bps))
    print("예상유량 계수(훈련 %d건에서 적합): 순흐름/OI명목가 중앙 %.4f" % (len(ok), c_flow))
    print("평가구간: %s ~ %s (%d건)"
          % (d["dt"].iloc[cut].date(), d["dt"].iloc[-1].date(), len(d) - cut))

    pcols = ["prof_%d" % i for i in range(len(GRID))]
    prof = d[pcols].to_numpy(dtype="float64")
    est = c_flow * d["oiv"].to_numpy(dtype="float64")

    # ------------------------------------------------------------ 배치 규칙
    rules = {}
    for u0 in (0.005, 0.01, 0.02, 0.03, 0.05):
        rules["fixed %.1f%%" % (100 * u0)] = np.full(len(d), u0)
    for kk in (0.25, 0.5, 1.0, 2.0, 4.0):
        rules["clear k=%.2f" % kk] = np.array(
            [u_for_depth(prof[i], est[i] * kk) for i in range(len(d))])
    for kk in (0.5, 1.0, 2.0, 4.0):
        rules["depthmult k=%.1f" % kk] = np.array(
            [u_for_depth(prof[i], prof[i, 0] * kk) for i in range(len(d))])
    for kk in (4.0, 8.0, 16.0):
        rules["sigma x%.0f" % kk] = np.clip(
            kk * d["sigma"].to_numpy(dtype="float64"), U_MIN, U_MAX)

    for name, u in rules.items():
        d[name] = u

    # 전 규칙이 유효 u 를 낸 이벤트만 비교 (공정 비교)
    valid = np.ones(len(d), dtype=bool)
    for name in rules:
        valid &= np.isfinite(d[name].to_numpy())
    dv = d[valid].reset_index(drop=True)
    te = dv[dv["dt"] >= d["dt"].iloc[cut]].reset_index(drop=True)
    print("공통 유효 이벤트 %d / %d  (평가구간 %d)" % (len(dv), len(d), len(te)))

    for lab, sub in (("전 구간 (유량계수 in-sample 주의)", dv), ("평가구간만", te)):
        print("\n=== %s ===" % lab)
        rows = []
        for name in rules:
            s = simulate(sub, name, a.hold_min, a.cost_bps, m1cache)
            r = summarize(sub, s, a.hold_min)
            r["규칙"] = name
            rows.append(r)
        t = pd.DataFrame(rows)[["규칙", "n_ev", "u_med%", "fill%", "cond_bp",
                                "per_ev_bp", "win%", "mae_p05", "t_NW"]]
        print(t.sort_values("per_ev_bp", ascending=False).round(1).to_string(index=False))

    print("\n판정: 적응형이 'fixed 2.0%' 의 per_ev_bp 를 **의미 있게** 넘어야 한다.")
    print("      MAE 가 함께 개선되면 더 좋다. 둘 다 아니면 고정이 답이다.")
    print("주의: 공급 지도 L(p) 는 아직 없다. 여기 결과는 **수요측만** 쓴 것이고,")
    print("      HL 청산맵이 쌓이면 같은 틀에 L(p) 를 더해 다시 돌린다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
