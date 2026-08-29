# -*- coding: utf-8 -*-
"""R-1 — 청산 흐름의 응답함수. 롱청산(강제매도)과 숏청산(강제매수)을 나눠서 잰다.

무엇을 가르는가
  analysis/response.py 로 **일반 테이커 주문흐름**은 확실히 가격을 민다는 것이 나왔다
  (4.377bp, t=855, 96% 영구). 그러나 그것은 청산 흐름이 아니다. 프로젝트 골격인
  b ~ 0.04 / "청산은 가격을 못 민다" 판정은 청산 물량 대비 깊이에 대한 것이었다.
  남은 분기:
    (가) 청산도 일반 주문흐름처럼 민다      -> b ~ 0.04 는 명세 오류. 채널 A 부활
    (나) 청산은 예견 가능해 다르게 흡수된다 -> 두 결과 양립. 채널 A 판정 유지

왜 롱/숏을 반드시 나누는가
  롱청산 = 강제매도 = 하방 압력.  숏청산 = 강제매수 = 상방 압력.
  둘을 부호 하나로 뭉치면 '청산 방향을 따라 밀리는가' 와 '그냥 하락이 빠른가' 가
  섞인다. 크립토는 롱 편중이라 뭉친 계수는 롱청산이 지배한다. 방향별로 따로 재고
  대칭성(beta_S = -beta_L)을 명시적으로 검정해야 판정이 성립한다.

핵심 교란 — 역인과
  청산은 가격이 밀려서 일어난다. 롱청산은 하락 뒤에 발생한다. 따라서
  E[r_{t+l} * eps_liq] 는 인과 임팩트가 정확히 0이어도 모멘텀만으로 양수가 된다.
  T0 이 그 크기를 직접 보여주고, T2/T3/T4 가 각각 다른 방식으로 처리한다.

표본의 구조적 한계 (결과를 읽기 전에 반드시)
  Tardis 무료 구간은 매월 1일만 준다. 95일/2,159일 = 4.4% 이고, 일중 변동폭 상위 5%
  날 중 보유 비율도 5.6% 로 기저율과 같다. 즉 **캐스케이드 날에 대한 표집 이득이 0**
  이다. 이 스크립트는 '평시 규모 영역' 의 판정이며 캐스케이드 규모로 외삽할 수 없다.
  (그래도 유효한 이유: 판정 대상인 b ~ 0.04 자체가 평시 표본 산출물이다.)

실행:
    python analysis/response_liq.py
    python analysis/response_liq.py --exchange bybit --full-feed     # T5 재확인용
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

BULK = os.path.join(C.DATA, "binance_bulk", "klines_1m")
LIQ = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")
OUT = os.path.join(C.DATA, "analysis", "r1_liq_bars.parquet")

LAGS = [1, 3, 5, 10, 30, 60, 120, 240]
PRE = [5, 15, 60]                      # 사전 수익 구간(분)
VOL_WIN = 1440                         # 변동성/ADV 롤링 창(분)


# --------------------------------------------------------------- 추정 도구
def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS + 클러스터 로버스트(CR1). 전체 공분산 V 까지 돌려준다.

    일자 클러스터를 쓰는 이유: 같은 날의 관측은 같은 충격을 공유해 잔차가 상관된다.
    더구나 여기서는 겹치는 전방 창(l=240 이면 240분 중복)을 쓰므로 iid 표준오차는
    구조적으로 과대한 t 를 만든다. (analysis/response.py 의 장기 t 가 그 오류였다.)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = X.shape
    if n <= k:
        return np.full(k, np.nan), np.full(k, np.nan), np.full((k, k), np.nan)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    if G <= 1:
        return beta, np.full(k, np.nan), np.full((k, k), np.nan)
    meat = np.zeros((k, k))
    Xr = X * resid[:, None]
    for g in range(G):
        s = Xr[inv == g].sum(axis=0)
        meat += np.outer(s, s)
    scale = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
    V = scale * (XtX_inv @ meat @ XtX_inv)
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0)), V


def cmean(x: np.ndarray, groups: np.ndarray):
    """평균 + 일클러스터 CR1 표준오차. (X = 1 인 OLS 와 동일)"""
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if m.sum() < 30:
        return np.nan, np.nan, np.nan, int(m.sum())
    b, se, _ = ols_cluster(np.ones((int(m.sum()), 1)), x[m], np.asarray(groups)[m])
    t = b[0] / se[0] if se[0] > 0 else np.nan
    return float(b[0]), float(se[0]), float(t), int(m.sum())


def wald(beta, V, c):
    """선형결합 c'beta 의 추정치/표준오차/t."""
    c = np.asarray(c, dtype=np.float64)
    v = float(c @ V @ c)
    est = float(c @ beta)
    se = np.sqrt(v) if v > 0 else np.nan
    return est, se, (est / se if se and se > 0 else np.nan)


# --------------------------------------------------------------- 데이터
def liq_bars(exchange: str, full_feed_only: bool) -> pd.DataFrame:
    """청산 프린트를 (심볼, 분) 으로 집계.

    부호 규약 — 여기서 틀리면 전부 틀린다:
      pos_side == 'long'  : 롱 포지션이 청산됨 -> **강제매도** -> 하방 압력 (qL)
      pos_side == 'short' : 숏 포지션이 청산됨 -> **강제매수** -> 상방 압력 (qS)
    """
    d = pd.read_parquet(LIQ)
    d = d[d["exchange"] == exchange]
    if full_feed_only:
        d = d[d["full_feed"]]
    if len(d) == 0:
        raise SystemExit("해당 거래소/구간에 청산 프린트가 없다: %s" % exchange)
    isL = (d["pos_side"].to_numpy() == "long")
    ntl = d["notional"].to_numpy(dtype=np.float64)
    out = pd.DataFrame({
        "symbol": d["symbol"].to_numpy(),
        "bar": (d["ts_ms"].to_numpy() // 60000).astype(np.int64),
        "qL": np.where(isL, ntl, 0.0),
        "qS": np.where(~isL, ntl, 0.0),
        "nL": isL.astype(np.int32),
        "nS": (~isL).astype(np.int32),
    })
    return out.groupby(["symbol", "bar"], as_index=False).sum()


def past_ret(cl: np.ndarray, w: int) -> np.ndarray:
    """t-1-w -> t-1 수익(bp). 바 t 자신은 **포함하지 않는다**(동시성 오염 방지)."""
    out = np.full(len(cl), np.nan)
    if len(cl) > w + 1:
        out[w + 1:] = (cl[w:-1] / cl[:-w - 1] - 1.0) * 1e4
    return out


def build_panel(lb: pd.DataFrame) -> pd.DataFrame:
    """청산 데이터가 있는 (심볼, 일) 안의 모든 분봉 패널.

    '있는 날' 을 심볼-일 단위로 정의하는 이유: 어떤 심볼이 그날 수집 대상이
    아니었던 것과 '그날 청산이 0건이었던 것' 을 구분할 수 없다. 청산 프린트가
    1건 이상 있는 심볼-일만 쓰면 거짓 0 이 섞이지 않는다. 선택은 **일 단위**라
    그 안의 청산바 vs 비청산바 비교는 오염되지 않는다.
    """
    lb = lb.copy()
    lb["day"] = (lb["bar"] // 1440).astype(np.int64)
    # 심볼별 '수집된 날' 집합. 심볼당 3백만 봉이라 행 단위 파이썬 루프는 못 쓴다.
    sday = {s: g["day"].unique() for s, g in lb.groupby("symbol")}
    syms = sorted(lb["symbol"].unique())

    frames = []
    for s in syms:
        p = os.path.join(BULK, "%s.parquet" % s)
        if not os.path.exists(p):
            continue
        k = pd.read_parquet(p)
        need = {"open_time", "close", "quote_volume", "taker_buy_quote_volume"}
        if not need.issubset(k.columns):
            continue
        k = k.sort_values("open_time").reset_index(drop=True)
        bar = (k["open_time"].to_numpy() // 60000).astype(np.int64)
        cl = k["close"].to_numpy(dtype=np.float64)
        qv = k["quote_volume"].to_numpy(dtype=np.float64)
        tb = k["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
        n = len(cl)
        if n < VOL_WIN * 2:
            continue

        day = bar // 1440
        want = np.isin(day, sday[s])
        if want.sum() == 0:
            continue

        cols = {"symbol": np.full(int(want.sum()), s, dtype=object),
                "bar": bar[want], "day": day[want],
                "close": cl[want], "qv": qv[want]}

        # 일반 테이커 순유량 (명목가). 부호는 매수 우세면 +1.
        net = 2.0 * tb - qv
        cols["gnet"] = net[want]

        # 변동성 / ADV — 제곱근 법칙과 통제변수용. 미래 정보를 쓰지 않도록 과거만.
        r1 = np.concatenate([[np.nan], cl[1:] / cl[:-1] - 1.0])
        sr = pd.Series(r1)
        sig1 = sr.rolling(VOL_WIN, min_periods=200).std().to_numpy()
        adv = (pd.Series(qv).rolling(VOL_WIN, min_periods=200).mean().to_numpy()
               * float(VOL_WIN))
        cols["sig_d"] = (sig1 * np.sqrt(float(VOL_WIN)))[want]     # 일 변동성
        cols["adv"] = adv[want]

        # 수익률 계열은 float32. 21종 x 95일 x 1440분 x 19열이면 float64 로 400MB 를
        # 넘는다. bp 단위 값에 유효숫자 7자리면 충분하고, 추정은 float64 로 승격한다.
        for w in PRE:
            cols["pre%d" % w] = past_ret(cl, w)[want].astype(np.float32)

        for L in LAGS:
            r_imp = np.full(n, np.nan)      # 임팩트 포함: t-1 -> t+L
            r_rev = np.full(n, np.nan)      # 되돌림만  : t   -> t+L
            if n > L + 2:
                r_imp[1:n - L] = (cl[1 + L:] / cl[:n - L - 1] - 1.0) * 1e4
                r_rev[:n - L] = (cl[L:] / cl[:n - L] - 1.0) * 1e4
            cols["imp%d" % L] = r_imp[want].astype(np.float32)
            cols["rev%d" % L] = r_rev[want].astype(np.float32)

        frames.append(pd.DataFrame(cols))

    if not frames:
        raise SystemExit("1분봉과 겹치는 심볼이 없다")
    pan = pd.concat(frames, ignore_index=True)
    pan = pan.merge(lb[["symbol", "bar", "qL", "qS", "nL", "nS"]],
                    on=["symbol", "bar"], how="left")
    for c in ("qL", "qS", "nL", "nS"):
        pan[c] = pan[c].fillna(0.0)
    pan = pan[(pan["qv"] > 0) & np.isfinite(pan["close"]) & (pan["close"] > 0)]
    return pan.reset_index(drop=True)


# --------------------------------------------------------------- 검정
def t0_confound(pan: pd.DataFrame) -> None:
    """T0. 역인과 교란의 크기를 눈으로 보여준다."""
    print("\n" + "=" * 78)
    print("T0. 역인과 교란 — 청산 **직전** 수익 (바 t 는 제외)")
    print("=" * 78)
    print("  청산은 가격이 밀려서 일어난다. 이 표가 크면 클수록 '청산 이후 수익' 을")
    print("  그대로 임팩트로 읽으면 안 된다는 뜻이다.\n")
    pureL = (pan["qL"] > 0) & (pan["qS"] == 0)
    pureS = (pan["qS"] > 0) & (pan["qL"] == 0)
    both = (pan["qL"] > 0) & (pan["qS"] > 0)
    none = (pan["qL"] == 0) & (pan["qS"] == 0)
    print("  %-16s %9s %11s %11s %11s" % ("바 종류", "n", "pre5 bp", "pre15 bp", "pre60 bp"))
    for lab, m in (("순수 롱청산", pureL), ("순수 숏청산", pureS),
                   ("양방향 동시", both), ("청산 없음", none)):
        if m.sum() < 30:
            continue
        vals = []
        for w in PRE:
            b, se, t, _ = cmean(pan.loc[m, "pre%d" % w].to_numpy(), pan.loc[m, "day"].to_numpy())
            vals.append("%7.2f(%.0f)" % (b, t) if np.isfinite(t) else "      -")
        print("  %-16s %9d %11s %11s %11s" % (lab, int(m.sum()), *vals))
    print("  괄호는 일클러스터 t. 롱청산 앞은 하락, 숏청산 앞은 상승이 예상된다.")


def t1_response(pan: pd.DataFrame) -> dict:
    """T1. 방향별 응답함수 + 일반흐름 대조 + 대칭성."""
    print("\n" + "=" * 78)
    print("T1. 응답함수 R(l) — 방향별. 모든 t 는 일클러스터 CR1")
    print("=" * 78)
    pureL = ((pan["qL"] > 0) & (pan["qS"] == 0)).to_numpy()
    pureS = ((pan["qS"] > 0) & (pan["qL"] == 0)).to_numpy()
    gsign = np.sign(pan["gnet"].to_numpy())
    day = pan["day"].to_numpy()
    res = {}

    print("\n  [1a] 청산 방향으로 부호를 맞춘 수익 (양수 = 청산 방향으로 밀렸다)")
    print("  %5s | %9s %7s | %9s %7s | %9s %7s"
          % ("l(분)", "롱청산bp", "t", "숏청산bp", "t", "차(숏-롱)", "t"))
    for L in LAGS:
        r = pan["imp%d" % L].to_numpy()
        # 롱청산 = 강제매도 -> 청산 방향으로 밀렸다면 r < 0 이므로 -1 을 곱한다
        aL, _, tL, nL = cmean(-r[pureL], day[pureL])
        aS, _, tS, nS = cmean(r[pureS], day[pureS])
        # 차이 검정: 두 집단을 합쳐 더미 회귀 (일클러스터로 상관 처리)
        m = (pureL | pureS) & np.isfinite(r)
        y = np.where(pureL[m], -r[m], r[m])
        X = np.column_stack([np.ones(m.sum()), pureS[m].astype(float)])
        b, se, _ = ols_cluster(X, y, day[m])
        td = b[1] / se[1] if se[1] > 0 else np.nan
        print("  %5d | %9.3f %7.1f | %9.3f %7.1f | %9.3f %7.1f"
              % (L, aL, tL, aS, tS, b[1], td))
        res[L] = {"L": aL, "S": aS, "nL": nL, "nS": nS, "diff": b[1], "t_diff": td}

    print("\n  [1b] 같은 바에서 **일반 테이커 흐름** 부호로 잰 것 (표본 건전성 대조)")
    print("       iid 열은 response.py 가 쓴 (틀린) 표준오차. 배율이 정정 크기다.")
    print("  %5s | %9s %8s %8s %7s | %12s"
          % ("l(분)", "일반흐름bp", "t(클러스터)", "t(iid)", "배율", "n"))
    for L in LAGS:
        r = pan["imp%d" % L].to_numpy()
        m = np.isfinite(r) & (gsign != 0)
        x = (r[m] * gsign[m]).astype(np.float64)
        a, _, t, n = cmean(x, day[m])
        t_iid = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
        print("  %5d | %9.3f %8.1f %8.1f %7.1f | %12d"
              % (L, a, t, t_iid, t_iid / t if t else np.nan, n))
        res.setdefault("gen", {})[L] = a
    print("  이 값이 전표본(4.38bp)과 같은 크기면 부분표본이 정상이라는 뜻이다.")
    return res


def t2_permanence(pan: pd.DataFrame, res: dict) -> None:
    """T2. 영구/일시 분해 — 교란에 비대칭으로 강한 검정."""
    print("\n" + "=" * 78)
    print("T2. 영구/일시 분해 — 되돌림이 있는가")
    print("=" * 78)
    print("  모멘텀 교란은 **지속**을, 기계적 압력은 **되돌림**을 예측한다.")
    print("  되돌림 발견 = 강한 증거. 되돌림 부재 = '임팩트 없음' 과 '영구 임팩트' 양립(약한 증거).\n")
    pureL = ((pan["qL"] > 0) & (pan["qS"] == 0)).to_numpy()
    pureS = ((pan["qS"] > 0) & (pan["qL"] == 0)).to_numpy()
    gsign = np.sign(pan["gnet"].to_numpy())
    day = pan["day"].to_numpy()
    print("  [2a] 바 t 종가 **이후**의 수익, 청산 방향 기준 (음수 = 되돌림)")
    print("  %5s | %9s %7s | %9s %7s | %9s %7s"
          % ("l(분)", "롱청산bp", "t", "숏청산bp", "t", "일반흐름bp", "t"))
    for L in LAGS:
        r = pan["rev%d" % L].to_numpy()
        aL, _, tL, _ = cmean(-r[pureL], day[pureL])
        aS, _, tS, _ = cmean(r[pureS], day[pureS])
        m = np.isfinite(r) & (gsign != 0)
        aG, _, tG, _ = cmean(r[m] * gsign[m], day[m])
        print("  %5d | %9.3f %7.1f | %9.3f %7.1f | %9.3f %7.1f"
              % (L, aL, tL, aS, tS, aG, tG))

    print("\n  [2b] 영구 비율 = R(240)/R(1)")
    for k, lab in (("L", "롱청산"), ("S", "숏청산")):
        r1, r240 = res[1][k], res[240][k]
        pr = r240 / r1 if abs(r1) > 1e-9 else np.nan
        print("    %-8s R(1)=%8.3f  R(240)=%8.3f  영구비율 %7.1f%%" % (lab, r1, r240, 100 * pr))
    g1, g240 = res["gen"][1], res["gen"][240]
    print("    %-8s R(1)=%8.3f  R(240)=%8.3f  영구비율 %7.1f%%"
          % ("일반흐름", g1, g240, 100 * g240 / g1 if abs(g1) > 1e-9 else np.nan))
    print("  청산의 영구비율이 일반흐름보다 뚜렷이 낮으면 그 차이가 기계적 압력이고,")
    print("  그것이 곧 지정매수 전략의 수익원이다.")


def _design(pan: pd.DataFrame):
    """T3/T4 공통 설계행렬 재료. 물량은 전부 바 거래대금으로 정규화."""
    qv = pan["qv"].to_numpy()
    xL = pan["qL"].to_numpy() / qv
    xS = pan["qS"].to_numpy() / qv
    xG = pan["gnet"].to_numpy() / qv
    ctrl = np.column_stack([pan["pre5"].to_numpy(), pan["pre15"].to_numpy(),
                            pan["pre60"].to_numpy(), pan["sig_d"].to_numpy()])
    return xL, xS, xG, ctrl


def t3_incremental(pan: pd.DataFrame) -> None:
    """T3. 일반흐름·과거수익·변동성 통제 후 청산의 증분 계수.

    imp 판과 rev 판을 둘 다 낸다. 이 구분이 T3 의 전부다:
      imp (t-1 -> t+l) 은 바 t **자신의** 움직임을 포함한다. 그런데 바로 그 움직임이
        같은 분 안에서 청산을 촉발했다. 즉 동시성 역인과가 계수에 그대로 들어간다.
      rev (t -> t+l) 은 종가 이후만 본다. 촉발 경로가 끊기므로 인과 해석에 훨씬
        가깝고, **애초에 거래 가능한 것도 이쪽뿐이다** (지나간 바는 못 산다).
    """
    print("\n" + "=" * 78)
    print("T3. 증분 회귀 — r ~ xS + xL + xG + 과거수익 + sigma  (일클러스터 CR1)")
    print("=" * 78)
    print("  xS = 숏청산액/거래대금 (강제매수), xL = 롱청산액/거래대금 (강제매도).")
    print("  기대: beta_S > 0, beta_L < 0. 대칭이면 beta_S + beta_L = 0.")
    xL, xS, xG, ctrl = _design(pan)
    day = pan["day"].to_numpy()
    for kind, note in (("imp", "바 t 포함 — **동시성 오염 있음**"),
                       ("rev", "종가 이후만 — 오염 없음. 거래 가능한 것은 이쪽")):
        print("\n  [3%s] %s (%s)" % ("a" if kind == "imp" else "b", kind, note))
        print("  %5s | %10s %7s | %10s %7s | %10s %7s | %9s %6s"
              % ("l(분)", "beta_S", "t", "beta_L", "t", "beta_gen", "t", "S+L", "t"))
        for L in LAGS:
            y = pan["%s%d" % (kind, L)].to_numpy()
            X = np.column_stack([np.ones(len(y)), xS, xL, xG, ctrl])
            m = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if m.sum() < 500:
                continue
            b, se, V = ols_cluster(X[m], y[m], day[m])
            c = np.zeros(X.shape[1]); c[1] = 1.0; c[2] = 1.0
            est, _, td = wald(b, V, c)
            print("  %5d | %10.2f %7.1f | %10.2f %7.1f | %10.2f %7.1f | %9.2f %6.1f"
                  % (L, b[1], b[1] / se[1], b[2], b[2] / se[2],
                     b[3], b[3] / se[3], est, td))
    print("\n  마지막 열 |t| 가 크면 **비대칭**이다(한쪽이 더 세게 민다).")
    print("  [3b] 에서 beta_L > 0 이면 '롱청산 뒤에 되돌린다' 는 뜻이고,")
    print("  그것이 지정매수 전략이 실제로 먹을 수 있는 유일한 부분이다.")


def t4_anticipated(pan: pd.DataFrame) -> None:
    """T4. 예견분 vs 서프라이즈분 — (가)/(나) 를 직접 가른다.

    청산량을 과거 수익·변동성으로 예측해 적합분/잔차로 쪼갠다. 생성회귀 편의를
    피하려고 **홀수일에서 적합해 짝수일에 적용**하고 그 반대도 해서 합친다.
    """
    print("\n" + "=" * 78)
    print("T4. 예견분 vs 서프라이즈분 (표본외 2겹)")
    print("=" * 78)
    print("  (가) 청산도 일반흐름처럼 민다 -> 적합분·잔차분 둘 다 임팩트 있음")
    print("  (나) 예견돼 흡수된다         -> **적합분 임팩트 ~ 0**, 잔차분만 있음\n")
    xL, xS, xG, ctrl = _design(pan)
    day = pan["day"].to_numpy()
    P = np.column_stack([np.ones(len(xL)), ctrl,
                         np.abs(ctrl[:, 1]), np.abs(ctrl[:, 2])])   # |pre15|, |pre60|
    okP = np.isfinite(P).all(axis=1)
    fold = (day % 2).astype(int)

    hat = {"S": np.full(len(xS), np.nan), "L": np.full(len(xL), np.nan)}
    for key, x in (("S", xS), ("L", xL)):
        for f in (0, 1):
            tr = okP & (fold != f) & np.isfinite(x)
            te = okP & (fold == f)
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            bb = np.linalg.pinv(P[tr].T @ P[tr]) @ (P[tr].T @ x[tr])
            hat[key][te] = P[te] @ bb
    tilS, tilL = xS - hat["S"], xL - hat["L"]

    for key, x, h in (("S", xS, hat["S"]), ("L", xL, hat["L"])):
        m = np.isfinite(x) & np.isfinite(h)
        if m.sum() > 100:
            v = np.var(x[m])
            print("  x%s 예측 R^2(표본외) = %.4f" % (key, 1 - np.var(x[m] - h[m]) / v if v > 0 else np.nan))

    print("\n  %5s | %10s %6s %10s %6s | %10s %6s %10s %6s"
          % ("l(분)", "S 예견", "t", "S 서프라이즈", "t", "L 예견", "t", "L 서프라이즈", "t"))
    for L in LAGS:
        y = pan["imp%d" % L].to_numpy()
        X = np.column_stack([np.ones(len(y)), hat["S"], tilS, hat["L"], tilL, xG, ctrl])
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < 500:
            continue
        b, se, _ = ols_cluster(X[m], y[m], day[m])
        print("  %5d | %10.2f %6.1f %10.2f %6.1f | %10.2f %6.1f %10.2f %6.1f"
              % (L, b[1], b[1] / se[1], b[2], b[2] / se[2],
                 b[3], b[3] / se[3], b[4], b[4] / se[4]))


def t5_sqrt_law(pan: pd.DataFrame) -> None:
    """T5. 제곱근 법칙 계수 Y — 캐스케이드 모델에 그대로 들어가는 수."""
    print("\n" + "=" * 78)
    print("T5. 제곱근 법칙  dP = Y * sigma_d * sqrt(Q/ADV)   (원점 통과, 일클러스터)")
    print("=" * 78)
    day = pan["day"].to_numpy()
    sig = pan["sig_d"].to_numpy()
    adv = pan["adv"].to_numpy()
    qL, qS = pan["qL"].to_numpy(), pan["qS"].to_numpy()
    gnet = pan["gnet"].to_numpy()
    L5 = pan["imp5"].to_numpy()
    base_ok = np.isfinite(sig) & (sig > 0) & np.isfinite(adv) & (adv > 0) & np.isfinite(L5)

    R5 = pan["rev5"].to_numpy()
    print("  imp5 = 바 t 포함(동시성 오염). rev5 = 종가 이후만(오염 없음).")
    print("  %-14s %9s %8s | %9s %8s | %12s"
          % ("흐름", "Y(imp5)", "t", "Y(rev5)", "t", "n"))
    rows = []
    specs = [("롱청산(매도)", qL, -1.0), ("숏청산(매수)", qS, +1.0),
             ("청산 합산", qS + qL, None), ("일반 테이커", np.abs(gnet), None)]
    for lab, Q, sgn in specs:
        if sgn is None:
            s = np.sign(qS - qL) if lab == "청산 합산" else np.sign(gnet)
        else:
            s = np.full(len(Q), sgn)
        m = base_ok & (Q > 0) & (s != 0) & np.isfinite(R5)
        if m.sum() < 200:
            continue
        x = (sig[m] * np.sqrt(Q[m] / adv[m]) * 1e4)[:, None]
        bi, sei, _ = ols_cluster(x, L5[m] * s[m], day[m])
        br, ser, _ = ols_cluster(x, R5[m] * s[m], day[m])
        rows.append((lab, float(bi[0]), float(br[0])))
        print("  %-14s %9.4f %8.1f | %9.4f %8.1f | %12d"
              % (lab, bi[0], bi[0] / sei[0] if sei[0] > 0 else np.nan,
                 br[0], br[0] / ser[0] if ser[0] > 0 else np.nan, int(m.sum())))
    d = {r[0]: r[1] for r in rows}
    dr = {r[0]: r[2] for r in rows}
    if "청산 합산" in d and abs(d.get("일반 테이커", 0)) > 1e-9:
        print("\n  Y_liq / Y_gen = %.2f (imp5)   %.2f (rev5)"
              % (d["청산 합산"] / d["일반 테이커"],
                 dr["청산 합산"] / dr["일반 테이커"] if abs(dr.get("일반 테이커", 0)) > 1e-9
                 else np.nan))
        print("  1 에 가까우면 (가). 뚜렷이 작으면 (나). imp5 판은 상한이다.")

    # 외삽 범위 공시 — 이걸 빼면 [5b] 를 결과로 오독한다
    ep = os.path.join(C.DATA, "analysis", "events_all_k8_doi-0.02_gap12.parquet")
    if os.path.exists(ep) and "청산 합산" in d:
        mm = base_ok & ((qL + qS) > 0)
        ratio = ((qL + qS) / adv)[mm]
        ev = pd.read_parquet(ep)
        oi = (ev["doi_mag"] * ev["oiv"]).to_numpy()
        ad = float(np.nanmedian(adv[base_ok]))
        tgt = float(np.median(oi)) / ad
        print("\n  [5b] 캐스케이드 규모로의 외삽 — **가능한지부터 본다**")
        print("    표본 Q/ADV: 중앙 %.3g  p99 %.3g  최대 %.3g"
              % (np.median(ratio), np.quantile(ratio, .99), ratio.max()))
        print("    이벤트 중앙 Q/ADV = %.3g  ->  중앙 대비 **%.0f배**, 최대 대비 %.1f배"
              % (tgt, tgt / np.median(ratio), tgt / ratio.max()))
        sd = float(np.nanmedian(sig[base_ok]))
        dP = d["청산 합산"] * sd * np.sqrt(tgt) * 1e4
        print("    형식적으로 대입하면 dP = %.0f bp 지만, 표본 밖 외삽이라 **무효**다."
              % dP)
        print("    제곱근 법칙의 함수형이 이 구간에서 성립하는지 자체가 미검정이고,")
        print("    Y(imp5) 는 동시성 오염으로 위쪽으로 편향돼 있다.")


def main() -> int:
    ap = argparse.ArgumentParser(description="R-1 liquidation response function")
    ap.add_argument("--exchange", default="binance-futures")
    ap.add_argument("--full-feed", action="store_true",
                    help="전건 피드 구간만 (크기 기반 검정용)")
    a = ap.parse_args()
    U.init_stdout()

    print("=" * 78)
    print("R-1 — 청산 흐름의 응답함수 (롱청산=강제매도 / 숏청산=강제매수 분리)")
    print("=" * 78)
    lb = liq_bars(a.exchange, a.full_feed)
    pan = build_panel(lb)

    d0 = pd.to_datetime(pan["bar"].min() * 60000, unit="ms")
    d1 = pd.to_datetime(pan["bar"].max() * 60000, unit="ms")
    ndays = pan["day"].nunique()
    hasliq = ((pan["qL"] > 0) | (pan["qS"] > 0))
    print("\n거래소 %s%s" % (a.exchange, " (full_feed만)" if a.full_feed else ""))
    print("**사용 데이터 기간: %s ~ %s / 수집일 %d일 / 심볼 %d종**"
          % (str(d0)[:10], str(d1)[:10], ndays, pan["symbol"].nunique()))
    print("총 분봉 %d개 중 청산 발생 %d개 (%.1f%%)"
          % (len(pan), int(hasliq.sum()), 100 * hasliq.mean()))
    print("  순수 롱청산 %d | 순수 숏청산 %d | 양방향 %d"
          % (int(((pan.qL > 0) & (pan.qS == 0)).sum()),
             int(((pan.qS > 0) & (pan.qL == 0)).sum()),
             int(((pan.qL > 0) & (pan.qS > 0)).sum())))
    print("\n*** 표본 한계: Tardis 무료 구간은 매월 1일만 준다. 달력 표본이지 사건")
    print("    표본이 아니며, 일중 변동폭 상위 5%% 날 커버리지가 기저율과 같다.")
    print("    -> 이 결과는 **평시 규모 영역**의 판정이다. 캐스케이드 규모로 외삽 불가.")

    t0_confound(pan)
    res = t1_response(pan)
    t2_permanence(pan, res)
    t3_incremental(pan)
    t4_anticipated(pan)
    t5_sqrt_law(pan)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pan[hasliq].to_parquet(OUT, index=False)
    print("\n청산 발생 분봉 저장: %s (%d행)" % (OUT, int(hasliq.sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
