# -*- coding: utf-8 -*-
"""조건부 확률모델 — S(u|x) 를 실제로 적합한다.

지금까지 없던 것
  MODEL.md 는 S(u), h(u), EV(u) 를 정의만 해두고, 측정한 것은 **무조건부** S(u) 뿐이다.
  "사건마다 다른 도달확률"을 내는 추정기는 한 번도 적합된 적이 없다.
  손으로 만든 배치 규칙(analysis/adaptive_entry.py)은 표본 내/외에서 순위가 뒤집혀
  값어치를 못 보였다. 그래서 규칙이 아니라 **모델**을 적합한다.

모델
  이산 시간(=가격 격자) 해저드. 레벨 u 마다 '여기까지 왔는데 여기서 멈출 확률'을
  로지스틱으로 놓는다.

      h(u | x) = sigmoid( b0 + b1*log(u) + b_x . x )
      S(u | x) = prod_{v <= u} ( 1 - h(v | x) )

  x (트리거 시점 관측 가능한 것만):
      log( D(u) / OIV )     그 레벨까지의 누적 대기 매수 / OI 명목가   <- 수요/공급 비
      log( sigma )          직전 1일 변동성
      dOI                   트리거 봉 OI 변화율 (청산 강도)
      log |bar_ret|         트리거 봉 낙폭
      log( D(1%) / OIV )    호가 전반 두께

  **L(p) 는 아직 없다.** HL 청산맵이 쌓이면 x 에 log(L(u)/D(u)) 를 더해 재적합한다.

  수익 쪽은 무조건부 E[r|fill(u)] 를 레벨별로 쓴다(훈련구간에서만 추정).
  그러면 배치는

      EV(u | x) = S(u|x) * E[r|fill(u)] - c ,   u* = argmax_u EV(u|x)

적합/평가
  훈련: 앞 절반(날짜순).  평가: 뒤 절반. 계수는 훈련에서만 적합한다.
  비교 대상: 고정 offset 전 격자, 그리고 무조건부 S(u) 로 고른 u* (조건부 정보 없음).
  **조건부 모델이 무조건부를 못 이기면 x 에 정보가 없다는 뜻이다.**

실행:
    python analysis/hazard_model.py
    python analysis/hazard_model.py --hold-min 60
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
# 배치 후보 격자 — 0.25%p 간격, 0.25%~6%
LEVELS = np.round(np.arange(0.0025, 0.0601, 0.0025), 6)


def load_1m(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s" % symbol)
    return (pd.read_parquet(p)[["open_time", "high", "low", "close"]]
              .sort_values("open_time").reset_index(drop=True))


def depth_at_u(prof: np.ndarray, u: float) -> float:
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0) or np.any(np.diff(prof) < 0):
        return np.nan
    if u <= GRID[0]:
        return float(prof[0] * u / GRID[0])
    if u <= GRID[-1]:
        return float(np.exp(np.interp(np.log(u), np.log(GRID), np.log(prof))))
    sl = (np.log(prof[-1]) - np.log(prof[-2])) / (np.log(GRID[-1]) - np.log(GRID[-2]))
    return float(prof[-1] * np.exp(sl * (np.log(u) - np.log(GRID[-1]))))


def build_events(symbol: str, ttl_min: int, hold_min: int, k: float,
                 doi_thr: float, min_gap: int) -> pd.DataFrame:
    """이벤트 단위: 사전 상태 x + 실제 경로(레벨별 도달/수익)."""
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, _ = BD.load_clean(symbol, BID + ASK, verbose=False)
    if df5.empty or m1.empty or dep.empty:
        return pd.DataFrame()
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
    n1 = len(ot)
    t5, close5 = df5["open_time"].to_numpy(), df5["close"].to_numpy()
    sig5, ret5 = df5["sigma"].to_numpy(), df5["ret"].to_numpy()
    doi5 = df5["doi"].to_numpy()
    oiv5 = df5["sum_open_interest_value"].to_numpy()
    dts = dep["ts_ms"].to_numpy()
    bid, ask = dep[BID].to_numpy(), dep[ASK].to_numpy()

    out = []
    for r in ev.itertuples():
        i = r.i
        p0 = close5[i]
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sig5[i]) and sig5[i] > 0):
            continue
        if not (np.isfinite(oiv5[i]) and oiv5[i] > 0 and np.isfinite(doi5[i])):
            continue
        trig = int(t5[i])
        j0 = int(np.searchsorted(dts, trig, side="right")) - 1
        if j0 < 0 or trig - int(dts[j0]) > MAX_SNAP_LAG_MS:
            continue
        prof = (bid[j0] if r.side == 1 else ask[j0]).astype("float64")
        if not np.all(np.isfinite(prof)) or np.any(prof <= 0) or np.any(np.diff(prof) < 0):
            continue

        a = int(np.searchsorted(ot, trig + BAR_MS, side="left"))
        b = int(np.searchsorted(ot, trig + BAR_MS + ttl_min * MIN_MS, side="left"))
        if a >= n1 or b <= a or b + hold_min >= n1:
            continue

        rec = {"symbol": symbol, "trig_ms": trig, "side": int(r.side), "p0": float(p0),
               "sigma": float(sig5[i]), "bar_ret": float(abs(ret5[i])),
               "doi": float(doi5[i]), "oiv": float(oiv5[i]),
               "d1": float(prof[0])}
        # 레벨별 도달 여부와 체결 시 수익
        for li, u in enumerate(LEVELS):
            limit = p0 * (1 - u) if r.side == 1 else p0 * (1 + u)
            seg = (lo[a:b] <= limit) if r.side == 1 else (hi[a:b] >= limit)
            idx = np.flatnonzero(seg)
            if idx.size == 0:
                rec["reach_%d" % li] = 0
                rec["ret_%d" % li] = np.nan
                rec["mae_%d" % li] = np.nan
            else:
                j = a + int(idx[0])
                e = min(j + hold_min, n1 - 1)
                rec["reach_%d" % li] = 1
                rec["ret_%d" % li] = float((cl[e] / limit - 1.0) * r.side)
                rec["mae_%d" % li] = float((lo[j:e + 1].min() / limit - 1.0)
                                           if r.side == 1
                                           else (1.0 - hi[j:e + 1].max() / limit))
            rec["D_%d" % li] = depth_at_u(prof, float(u))
        out.append(rec)
    return pd.DataFrame(out)


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 200) -> np.ndarray:
    """뉴턴-랩슨 로지스틱 회귀 (L2 정칙화). scikit 의존을 피한다."""
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(iters):
        z = X @ w
        pr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (pr - y) + l2 * w
        s = pr * (1 - pr)
        H = (X * s[:, None]).T @ X + l2 * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def make_panel(d: pd.DataFrame, use_x: bool) -> tuple[np.ndarray, np.ndarray, list]:
    """이벤트 x 레벨 패널. 해저드 = '여기까지 왔는데 다음 레벨로 못 감'."""
    rows, ys = [], []
    names = ["const", "log_u"]
    if use_x:
        names += ["log_DoverOIV", "log_sigma", "doi", "log_barret", "log_D1overOIV"]
    for r in d.itertuples():
        for li in range(len(LEVELS) - 1):
            reached = getattr(r, "reach_%d" % li)
            if not reached:
                break                      # 못 왔으면 그 아래는 위험집합이 아니다
            nxt = getattr(r, "reach_%d" % (li + 1))
            D = getattr(r, "D_%d" % li)
            if not np.isfinite(D) or D <= 0:
                continue
            x = [1.0, np.log(LEVELS[li])]
            if use_x:
                x += [np.log(max(D / r.oiv, 1e-12)),
                      np.log(max(r.sigma, 1e-12)),
                      float(r.doi),
                      np.log(max(r.bar_ret, 1e-12)),
                      np.log(max(r.d1 / r.oiv, 1e-12))]
            rows.append(x)
            ys.append(1 - nxt)             # 1 = 여기서 멈춤
    if not rows:
        return np.zeros((0, len(names))), np.zeros(0), names
    return np.asarray(rows, dtype="float64"), np.asarray(ys, dtype="float64"), names


def survival(d: pd.DataFrame, w: np.ndarray, use_x: bool) -> np.ndarray:
    """이벤트별 S(u|x) 를 전 레벨에 대해."""
    n, L = len(d), len(LEVELS)
    S = np.ones((n, L))
    for ei, r in enumerate(d.itertuples()):
        s = 1.0
        for li in range(L):
            if li > 0:
                D = getattr(r, "D_%d" % (li - 1))
                if not np.isfinite(D) or D <= 0:
                    S[ei, li:] = np.nan
                    break
                x = [1.0, np.log(LEVELS[li - 1])]
                if use_x:
                    x += [np.log(max(D / r.oiv, 1e-12)),
                          np.log(max(r.sigma, 1e-12)),
                          float(r.doi),
                          np.log(max(r.bar_ret, 1e-12)),
                          np.log(max(r.d1 / r.oiv, 1e-12))]
                h = 1.0 / (1.0 + np.exp(-np.clip(np.dot(x, w), -30, 30)))
                s *= (1.0 - h)
            S[ei, li] = s
    return S


def evaluate(d: pd.DataFrame, u_idx: np.ndarray, cost: float) -> dict:
    """선택한 레벨로 체결/수익을 집계."""
    n = len(d)
    rets, maes, filled = [], [], 0
    for ei, r in enumerate(d.itertuples()):
        li = int(u_idx[ei])
        if getattr(r, "reach_%d" % li):
            v = getattr(r, "ret_%d" % li)
            if np.isfinite(v):
                filled += 1
                rets.append(v - cost)
                maes.append(getattr(r, "mae_%d" % li))
    if not rets:
        return {"fill%": 0.0, "cond_bp": np.nan, "per_ev_bp": np.nan,
                "win%": np.nan, "mae_p05": np.nan, "t_NW": np.nan, "u_med%": np.nan}
    rr = np.asarray(rets)
    fr = filled / max(n, 1)
    return {"fill%": 100 * fr, "cond_bp": 1e4 * rr.mean(),
            "per_ev_bp": 1e4 * rr.mean() * fr, "win%": 100 * (rr > 0).mean(),
            "mae_p05": 1e4 * np.nanpercentile(maes, 5), "t_NW": nw_tstat(rr, 3),
            "u_med%": 100 * float(np.median(LEVELS[u_idx.astype(int)]))}


def main() -> int:
    ap = argparse.ArgumentParser(description="fitted conditional hazard model")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--ttl-min", type=int, default=60)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--cost-bps", type=float, default=7.0)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 220)
    symbols = a.symbols if a.symbols else C.MAJORS
    cost = a.cost_bps / 1e4

    frames = []
    for s in symbols:
        try:
            d = build_events(s, a.ttl_min, a.hold_min, a.k, a.doi, a.min_gap)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            U.log("%s: %d 이벤트" % (s, len(d)))
    if not frames:
        U.log("no events")
        return 1
    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d = d.sort_values("dt").reset_index(drop=True)
    U.atomic_write_parquet(d.drop(columns=["dt"]),
                           os.path.join(C.DATA, "analysis", "hazard_model.parquet"))

    cut = len(d) // 2
    tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
    print("\n=== 표본 ===")
    print("이벤트 %d | 심볼 %d | %s ~ %s | 레벨 %d개 (0.25%%~6%%)"
          % (len(d), d.symbol.nunique(), d["dt"].min().date(), d["dt"].max().date(),
             len(LEVELS)))
    print("훈련 %d (%s~%s) / 평가 %d (%s~%s)"
          % (len(tr), tr["dt"].min().date(), tr["dt"].max().date(),
             len(te), te["dt"].min().date(), te["dt"].max().date()))

    # 훈련에서 레벨별 무조건부 E[r|fill] 추정
    er = np.full(len(LEVELS), np.nan)
    for li in range(len(LEVELS)):
        v = tr["ret_%d" % li].to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if v.size >= 10:
            er[li] = v.mean()
    ok_lv = np.isfinite(er)
    print("훈련에서 E[r|fill] 추정된 레벨 %d/%d" % (ok_lv.sum(), len(LEVELS)))

    # ------------------------------------------------------------ 적합
    out = []
    for lab, use_x in (("무조건부 (log u 만)", False), ("조건부 (호가+상태)", True)):
        X, y, names = make_panel(tr, use_x)
        if len(X) < 200:
            print("%s: 패널 부족 %d" % (lab, len(X)))
            continue
        w = logistic_fit(X, y, a.l2)
        print("\n--- %s ---  패널 %d행, 정지율 %.3f" % (lab, len(X), y.mean()))
        for nm, wi in zip(names, w):
            print("    %-16s %+8.4f" % (nm, wi))
        S = survival(te, w, use_x)
        EVm = S * er[None, :] - cost
        EVm[:, ~ok_lv] = -np.inf
        u_idx = np.nanargmax(np.where(np.isfinite(EVm), EVm, -np.inf), axis=1)
        r = evaluate(te, u_idx, cost)
        r["규칙"] = lab
        out.append(r)

    # 고정 offset 비교
    for u0 in (0.005, 0.01, 0.02, 0.03, 0.05):
        li = int(np.argmin(np.abs(LEVELS - u0)))
        r = evaluate(te, np.full(len(te), li), cost)
        r["규칙"] = "fixed %.1f%%" % (100 * u0)
        out.append(r)

    t = pd.DataFrame(out)[["규칙", "u_med%", "fill%", "cond_bp", "per_ev_bp",
                           "win%", "mae_p05", "t_NW"]]
    print("\n=== 평가구간 %d건 — 조건부 모델 vs 무조건부 vs 고정 ===" % len(te))
    print(t.sort_values("per_ev_bp", ascending=False).round(1).to_string(index=False))
    print("\n판정:")
    print("  조건부 > 무조건부  이면 x(호가·변동성·dOI)에 배치 정보가 있다는 뜻.")
    print("  조건부 ~ 무조건부  이면 x 는 무의미하고, 남은 희망은 L(p) 뿐이다.")
    print("  둘 다 고정을 못 이기면 무차별성이 조건부로도 안 깨진 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
