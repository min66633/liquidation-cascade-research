# -*- coding: utf-8 -*-
"""구조식 log X = a + gamma * log(V/D) 의 정식화와 진단.

무엇을 주장하는가
  캐스케이드가 바닥을 치는 거리 X 는 '쏟아진 물량 V' 를 '받아줄 대기물량 D' 로 나눈
  값이 정한다. D(u) = A u^beta 를 흡수하며 밀린다면 적분에서

      X ~ (V/A)^{1/(1+beta)}   =>   log X = a + gamma log(V/D),  gamma = 1/(1+beta)

  가 나온다. 즉 gamma 는 (0,1] 에 있어야 하고, beta=0(가격축 균일 호가)이면 gamma=1 이다.

표기 주의
  gamma 는 MODEL.md 의 충격지수 b(~0.04, 기각됨)와 '다른 양'이다. PROB_MODEL.md 0절.

이 스크립트가 검정하는 것
  1. gamma 추정 — 일자 클러스터 표준오차. iid 를 쓰면 t 가 2배 부풀려진 전력이 있다.
  2. 창 길이 순환 제거 — V 를 '바닥까지' 누적하면 바닥이 깊을수록 창이 길어 V 가 커진다.
     전부 고정창(5/15/60분)으로 재고, 오염판을 나란히 찍어 순환분이 보이게 한다.
  3. **역인과 진단** — V 는 강제 + 재량 매도의 합이고 '가격이 밀려서 매도가 나왔다' 가
     섞여 있다. L(p) 는 강제분만 예보하므로, 강제분이 실제로 관계를 만드는지 봐야 한다.
     강제분 비중 s = (|dOI| * OIV) / V 로 층화한다.
       메커니즘이 진짜면  -> s 가 높은 층에서 R2 가 높다
       전부 역인과면      -> 관계 없거나 반대
  4. gamma 안정성 — 심볼별 / 연도별. 5종은 전부 유동성 최상위라 알트에서 다를 수 있다.

실행:
    python analysis/vd_structure.py                       # config.MAJORS 전체
    python analysis/vd_structure.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT
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
import analysis.bookdepth as BD                             # noqa: E402
from analysis.event_study_h2 import load, find_events       # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
BAR_MS, MIN_MS = 300_000, 60_000
TTL = 60                              # 관측 창 (분)
MAX_SNAP_LAG_MS = 2 * 60_000          # 트리거 시점 호가 스냅샷 허용 지연
BID = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
WINDOWS = (("05", 5), ("15", 15), ("60", 60))
MIN_EVENTS_PER_FIT = 25               # 이보다 적으면 층별 적합을 신뢰하지 않는다


# ------------------------------------------------------------------ 회귀 도구
def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS + 클러스터 로버스트(CR1) 표준오차.

    일자 클러스터를 쓰는 이유: 같은 날의 이벤트는 같은 충격을 공유해 잔차가 상관된다.
    iid 표준오차는 이 프로젝트에서 t 를 2배(17.24 -> 8.04) 부풀린 전력이 있다.
    """
    n, k = X.shape
    if n <= k:
        return np.full(k, np.nan), np.full(k, np.nan)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    uniq = np.unique(groups)
    meat = np.zeros((k, k))
    for g in uniq:
        m = groups == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    G = len(uniq)
    if G <= 1:
        return beta, np.full(k, np.nan)
    scale = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
    V = scale * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return beta, se


def r2_oos(tr: pd.DataFrame, te: pd.DataFrame, feats) -> tuple:
    """시간순 분할 표본 외 R2 와 Spearman. 상수항 포함."""
    if len(tr) <= len(feats) + 1 or len(te) < 5:
        return np.nan, np.nan
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in feats])
    Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in feats])
    w = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ np.log(tr["X"].to_numpy()))
    p = Xte @ w
    yt = np.log(te["X"].to_numpy())
    den = float(np.sum((yt - yt.mean()) ** 2))
    if den <= 0:
        return np.nan, np.nan
    r2 = 1.0 - float(np.sum((yt - p) ** 2)) / den
    rho = float(pd.Series(yt).corr(pd.Series(p), method="spearman"))
    return r2, rho


def fit_gamma(d: pd.DataFrame, col: str) -> tuple:
    """log X ~ a + gamma * col. (gamma, se, t, R2_in, n, a)"""
    if len(d) < MIN_EVENTS_PER_FIT:
        return (np.nan,) * 4 + (len(d), np.nan)
    X = np.column_stack([np.ones(len(d)), d[col].to_numpy()])
    y = np.log(d["X"].to_numpy())
    beta, se = ols_cluster(X, y, d["day"].to_numpy())
    yhat = X @ beta
    den = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / den if den > 0 else np.nan
    t = beta[1] / se[1] if np.isfinite(se[1]) and se[1] > 0 else np.nan
    return float(beta[1]), float(se[1]), float(t), float(r2), len(d), float(beta[0])


def heterogeneity(gs: np.ndarray, ses: np.ndarray) -> dict:
    """심볼별 gamma 의 이질성이 진짜인가 표본오차인가.

    단순 sd 는 표본오차를 포함해 과대추정한다. Cochran Q / I2 / tau(DerSimonian-Laird)
    로 분리한다. Q 가 유의하지 않으면 '단일 gamma 로 다뤄도 된다' 는 뜻이고,
    그것이 계획 pre-mortem 3(5종 -> 21종 외삽 금지)의 판정 기준이다.
    """
    from scipy import stats                       # 이 함수에서만 필요
    w = 1.0 / ses ** 2
    k = len(gs)
    g_fe = float(np.sum(w * gs) / np.sum(w))
    se_fe = float(np.sqrt(1.0 / np.sum(w)))
    Q = float(np.sum(w * (gs - g_fe) ** 2))
    df = k - 1
    Cq = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    return {"g_fe": g_fe, "se_fe": se_fe, "Q": Q, "df": df,
            "p": float(stats.chi2.sf(Q, df)) if df > 0 else np.nan,
            "I2": max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0,
            "tau": float(np.sqrt(max(0.0, (Q - df) / Cq))) if Cq > 0 else np.nan,
            "z_vs1": (g_fe - 1.0) / se_fe if se_fe > 0 else np.nan,
            "p_vs1": (float(2 * stats.norm.sf(abs((g_fe - 1.0) / se_fe)))
                      if se_fe > 0 else np.nan)}


# ------------------------------------------------------------------ 표본 구성
def build(symbol: str, k: float, doi_thr: float, min_gap: int) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m(symbol)
    dep, _ = BD.load_clean(symbol, BID + ASK, verbose=False)
    if df5.empty or m1.empty or dep.empty:
        return pd.DataFrame()
    if "taker_buy_quote_volume" not in m1.columns:
        U.log("%s: taker_buy_quote_volume 없음 — 건너뜀" % symbol)
        return pd.DataFrame()
    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi = m1["low"].to_numpy(), m1["high"].to_numpy()
    qv = m1["quote_volume"].to_numpy()
    tbq = m1["taker_buy_quote_volume"].to_numpy()
    n1 = len(ot)

    t5, c5 = df5["open_time"].to_numpy(), df5["close"].to_numpy()
    sig5, ret5, z5 = (df5[c].to_numpy() for c in ("sigma", "ret", "z"))
    doi5 = df5["doi"].to_numpy()
    oiv5 = df5["sum_open_interest_value"].to_numpy()
    dts = dep["ts_ms"].to_numpy()
    bidv, askv = dep[BID].to_numpy(), dep[ASK].to_numpy()

    out = []
    for r in ev.itertuples():
        i = r.i
        p0 = c5[i]
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sig5[i]) and sig5[i] > 0):
            continue
        if not (np.isfinite(oiv5[i]) and oiv5[i] > 0 and np.isfinite(doi5[i])):
            continue

        trig = int(t5[i])
        j0 = int(np.searchsorted(dts, trig, side="right")) - 1
        if j0 < 0 or trig - int(dts[j0]) > MAX_SNAP_LAG_MS:
            continue
        prof = (bidv[j0] if r.side == 1 else askv[j0]).astype("float64")
        # 누적 프로파일이므로 단조여야 한다. 아니면 그 스냅샷은 신뢰할 수 없다.
        if not np.all(np.isfinite(prof)) or np.any(prof <= 0) or np.any(np.diff(prof) < 0):
            continue

        a = int(np.searchsorted(ot, trig + BAR_MS, side="left"))   # 진입 가능 시작
        b = a + TTL
        if a >= n1 or b >= n1:
            continue

        # ---- 실제 바닥
        w = slice(a, b)
        if r.side == 1:
            kk = a + int(np.argmin(lo[w]))
            X = 1.0 - lo[kk] / p0
        else:
            kk = a + int(np.argmax(hi[w]))
            X = hi[kk] / p0 - 1.0
        if not (np.isfinite(X) and X > 1e-5):
            continue

        def sellvol(i0: int, i1: int) -> float:
            """진행방향 테이커 명목가. 롱청산이면 테이커 매도, 숏청산이면 매수."""
            s = slice(i0, i1)
            tot = float(np.nansum(qv[s]))
            buy = float(np.nansum(tbq[s]))
            v = (tot - buy) if r.side == 1 else buy
            return v if v > 0 else np.nan

        D1 = float(prof[0])                       # 1% 이내 대기물량 (명목가)
        rec = {"symbol": symbol, "trig_ms": trig, "side": int(r.side),
               "X": float(X), "dur": int(kk - a + 1),
               # 손익 계산용 (x_dist.py 가 재사용한다)
               "a": int(a), "b": int(b), "p0": float(p0), "bot_i": int(kk),
               "log_D1abs": float(np.log(D1)),
               "log_sigma": float(np.log(sig5[i])),
               "log_bar": float(np.log(max(abs(ret5[i]), 1e-8))),
               "log_z": float(np.log(max(abs(z5[i]), 1e-8))),
               "doi_mag": float(abs(doi5[i])),
               "log_D1": float(np.log(D1 / oiv5[i])),
               "log_conv": float(np.log(prof[-1] / prof[0])),
               "log_dur": float(np.log(kk - a + 1)),
               "oiv": float(oiv5[i])}

        for lab, hh in WINDOWS:
            v = sellvol(a, min(a + hh, n1))
            rec["VD_" + lab] = float(np.log(v / D1)) if np.isfinite(v) else np.nan
            rec["V_" + lab] = v
        v_bot = sellvol(a, kk + 1)                # 창 길이 순환 있는 판 (대조용)
        rec["VD_bot"] = float(np.log(v_bot / D1)) if np.isfinite(v_bot) else np.nan

        # 강제분 비중 = (OI 감소 명목가) / (테이커 물량). 역인과 진단용.
        forced_notional = abs(doi5[i]) * oiv5[i]
        rec["fshare"] = (forced_notional / rec["V_60"]
                         if np.isfinite(rec["V_60"]) and rec["V_60"] > 0 else np.nan)
        out.append(rec)
    return pd.DataFrame(out)


def load_1m(s: str) -> pd.DataFrame:
    path = os.path.join(BULK, "klines_1m", "%s.parquet" % s)
    if not os.path.exists(path):
        raise FileNotFoundError("1m klines 없음: %s" % path)
    df = pd.read_parquet(path)
    cols = ["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    keep = [c for c in cols if c in df.columns]
    return df[keep].sort_values("open_time").reset_index(drop=True)


PRE = ["log_sigma", "log_bar", "log_z", "doi_mag", "log_D1", "log_conv"]


def tercile(v: np.ndarray) -> np.ndarray:
    """3분위 라벨 0/1/2. 동률이 많으면 경계가 뭉칠 수 있어 순위 기반으로 자른다."""
    r = pd.Series(v).rank(method="first").to_numpy()
    return np.floor(3.0 * (r - 1) / len(v)).astype(int).clip(0, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="structural equation log X = a + gamma log(V/D)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 200)
    syms = a.symbols if a.symbols else C.MAJORS

    frames = []
    for s in syms:
        try:
            d = build(s, a.k, a.doi, a.min_gap)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            U.log("%s: %d" % (s, len(d)))
    if not frames:
        U.log("이벤트 없음")
        return 1

    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d["day"] = d["dt"].dt.strftime("%Y-%m-%d")
    d["year"] = d["dt"].dt.year
    d = d.sort_values("dt").reset_index(drop=True)
    need = PRE + ["VD_05", "VD_15", "VD_60", "VD_bot", "X", "fshare"]
    n_raw = len(d)
    d = d.dropna(subset=need).reset_index(drop=True)

    cut = len(d) // 2
    tr, te = d.iloc[:cut], d.iloc[cut:].reset_index(drop=True)

    print("\n" + "=" * 78)
    print("구조식  log X = a + gamma * log(V/D)")
    print("=" * 78)
    print("표본 %d (결측 제외 전 %d) | %d종 | %s ~ %s | 일수 %d"
          % (len(d), n_raw, d.symbol.nunique(), d["dt"].min().date(),
             d["dt"].max().date(), d["day"].nunique()))
    print("훈련 %d / 평가 %d (시간순 50/50)" % (len(tr), len(te)))
    print("X 분위: " + "  ".join("%d%%=%.2f%%" % (100 * q, 100 * d["X"].quantile(q))
                                 for q in (.1, .25, .5, .75, .9)))

    # ---------------------------------------------------------- 1. 표본 외 설명력
    print("\n--- 1. 표본 외 설명력 (log X) ---")
    print("  %-40s %9s %9s" % ("변수", "R2", "Spearman"))
    rows = [("사전 관측만 (기준선)", PRE),
            ("log(V/D) 바닥까지  <- 창길이 순환 있음", ["VD_bot"]),
            ("log(V/D) 60분 고정", ["VD_60"]),
            ("log(V/D) 15분 고정", ["VD_15"]),
            ("log(V/D)  5분 고정", ["VD_05"]),
            ("log(V/D) 60분 + 사전", PRE + ["VD_60"])]
    for lab, f in rows:
        r2, rho = r2_oos(tr, te, f)
        print("  %-40s %+9.3f %+9.3f" % (lab, r2, rho))

    # ---------------------------------------------------------- 2. gamma 추정
    print("\n--- 2. gamma 추정 (전체표본, 일자 클러스터 SE) ---")
    print("  이론: gamma = 1/(1+beta) in (0,1].  beta=0(가격축 균일 호가)이면 gamma=1")
    print("  %-14s %9s %8s %8s %9s %8s %6s"
          % ("창", "gamma", "SE", "t", "beta", "R2(내)", "n"))
    for lab, _ in WINDOWS:
        col = "VD_" + lab
        g, se, t, r2, n, a0 = fit_gamma(d, col)
        beta = (1.0 / g - 1.0) if (np.isfinite(g) and g > 0) else np.nan
        print("  %-14s %+9.3f %8.3f %8.2f %+9.2f %8.3f %6d"
              % (lab + "분 고정", g, se, t, beta, r2, n))
    g, se, t, r2, n, a0 = fit_gamma(d, "VD_bot")
    print("  %-14s %+9.3f %8.3f %8.2f %+9s %8.3f %6d"
          % ("바닥까지(오염)", g, se, t, "-", r2, n))

    # ---------------------------------------------------------- 3. 역인과 진단
    print("\n--- 3. 역인과 진단: 강제분 비중 s = |dOI|*OIV / V(60분) ---")
    print("  메커니즘이 진짜면 s 가 높은 층에서 R2 가 높아야 한다.")
    print("  전부 역인과(가격이 밀려서 매도가 나옴)면 관계가 없거나 반대다.")
    d["fs_t"] = tercile(d["fshare"].to_numpy())
    print("  %-10s %10s %9s %8s %8s %8s %6s"
          % ("s 3분위", "s 중앙", "gamma", "SE", "t", "R2(내)", "n"))
    for q in (0, 1, 2):
        sub = d[d["fs_t"] == q]
        g, se, t, r2, n, a0 = fit_gamma(sub, "VD_60")
        print("  %-10s %9.1f%% %+9.3f %8.3f %8.2f %8.3f %6d"
              % (["하 (재량↑)", "중", "상 (강제↑)"][q],
                 100 * sub["fshare"].median(), g, se, t, r2, n))

    print("\n  보조: |dOI| 3분위별 (계획 원안)")
    d["doi_t"] = tercile(d["doi_mag"].to_numpy())
    print("  %-10s %10s %9s %8s %8s %8s %6s"
          % ("|dOI|", "중앙", "gamma", "SE", "t", "R2(내)", "n"))
    for q in (0, 1, 2):
        sub = d[d["doi_t"] == q]
        g, se, t, r2, n, a0 = fit_gamma(sub, "VD_60")
        print("  %-10s %9.2f%% %+9.3f %8.3f %8.2f %8.3f %6d"
              % (["하", "중", "상"][q], 100 * sub["doi_mag"].median(), g, se, t, r2, n))

    # ---------------------------------------------------------- 4. 안정성
    print("\n--- 4. gamma 안정성: 심볼별 (n>=%d 만) ---" % MIN_EVENTS_PER_FIT)
    print("  %-10s %9s %8s %8s %8s %6s" % ("심볼", "gamma", "SE", "t", "R2(내)", "n"))
    gs = []
    for s in sorted(d.symbol.unique()):
        sub = d[d.symbol == s]
        g, se, t, r2, n, a0 = fit_gamma(sub, "VD_60")
        if np.isfinite(g):
            gs.append(g)
            print("  %-10s %+9.3f %8.3f %8.2f %8.3f %6d" % (s, g, se, t, r2, n))
        else:
            print("  %-10s %9s %8s %8s %8s %6d" % (s, "n 부족", "-", "-", "-", n))
    if len(gs) >= 2:
        gs = np.array(gs)
        print("  심볼 간: 평균 %+.3f  sd %.3f  범위 [%+.3f, %+.3f]  (%d종)"
              % (gs.mean(), gs.std(ddof=1), gs.min(), gs.max(), len(gs)))
        print("  => sd 가 크면 5종 결과를 21종에 외삽하면 안 된다 (계획 pre-mortem 3).")

    print("\n--- 5. gamma 안정성: 연도별 ---")
    print("  %-10s %9s %8s %8s %8s %6s" % ("연도", "gamma", "SE", "t", "R2(내)", "n"))
    for yr in sorted(d.year.unique()):
        sub = d[d.year == yr]
        g, se, t, r2, n, a0 = fit_gamma(sub, "VD_60")
        if np.isfinite(g):
            print("  %-10d %+9.3f %8.3f %8.2f %8.3f %6d" % (yr, g, se, t, r2, n))
        else:
            print("  %-10d %9s %8s %8s %8s %6d" % (yr, "n 부족", "-", "-", "-", n))

    # ---------------------------------------------------------- 6. 순환분
    print("\n--- 6. 창 길이 순환분 분리 ---")
    X2 = np.column_stack([np.ones(len(d)), d["VD_bot"].to_numpy(), d["log_dur"].to_numpy()])
    y = np.log(d["X"].to_numpy())
    beta2, se2 = ols_cluster(X2, y, d["day"].to_numpy())
    gb, _, _, r2b, _, ab = fit_gamma(d, "VD_bot")
    Xd = np.column_stack([np.ones(len(d)), d["log_dur"].to_numpy()])
    bd, _ = ols_cluster(Xd, y, d["day"].to_numpy())
    yd = Xd @ bd
    r2d = 1.0 - float(np.sum((y - yd) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    print("  단독      log X = %+.3f %+.3f*VD_bot                R2(내)=%+.3f"
          % (ab, gb, r2b))
    print("  dur 통제  log X = %+.3f %+.3f*VD_bot %+.3f*log(dur)"
          % (beta2[0], beta2[1], beta2[2]))
    print("            (SE %.3f, %.3f)" % (se2[1], se2[2]))
    print("  log(dur) 단독 R2(내) = %+.3f" % r2d)
    print("  => 고정창 gamma 와 'dur 통제' gamma 가 비슷하면 순환 처리가 일관된 것이다.")

    print("\n상관: " + "  ".join(
        "corr(logX,%s)=%+.3f" % (c, float(pd.Series(np.log(d["X"])).corr(d[c])))
        for c in ("VD_bot", "VD_60", "VD_15", "VD_05")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
