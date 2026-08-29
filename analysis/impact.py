# -*- coding: utf-8 -*-
"""가격충격 함수 — 강제청산 물량 V가 터지면 가격이 얼마나 더 밀리는가.

질문
  "큰 청산 구간에 도달해 물량이 강제청산되면 가격이 얼마나 더 슈팅될지 추정 가능한가"
  -> 이것은 시계열 문제가 아니라 (청산량, 가격변위) 쌍의 문제다. 즉 가격충격 함수다.

    |dP| = a * V^b     (로그로 보면 log|dP| = log a + b log V)
      b ~ 0.5  -> 표준 제곱근 법칙. 강제청산이라고 특별할 게 없다는 뜻.
      b > 0.5  -> 초선형. 물량이 커질수록 단위당 충격이 커진다 = 캐스케이드 증폭.

핵심 검정 (이게 진짜 질문이다)
  같은 $1M이라도 **강제청산 물량**이 **평범한 거래량**보다 가격을 더 미는가?
  회귀에 둘 다 넣고 청산 물량의 계수가 유의하게 크면, 청산은 단순 거래량이 아니다.
  아니라면 청산 지도를 보는 것의 이점이 사라진다.

데이터
  청산: Tardis 무료 샘플(매월 1일), Bybit, 마이크로초 단위 가격+수량. 21.2만건 / $2.15B / 81일.
  가격: Binance 1분봉(전체 이력).
  주의 — 2025-02-25 이전 Bybit 피드는 심볼당 초당 1건 스로틀이라 청산이 과소집계된다.
  기본은 전건 피드 구간(2025-02-25~)만 쓰고, 이전 구간은 참고로만 따로 본다.

실행:
    python analysis/impact.py
    python analysis/impact.py --bucket-min 1 --horizon 5 --all-period
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
TARDIS = os.path.join(C.DATA, "tardis_liq")
BULK = os.path.join(C.DATA, "binance_bulk")
MIN_MS = 60_000


def load_1m_with_volume(symbol: str) -> pd.DataFrame:
    """1분봉 + 거래량. maker_1m.load_1m은 OHLC만 읽으므로 여기서 따로 읽는다
    (거래량이 없으면 ADV 정규화가 전부 NaN이 되어 표본이 통째로 사라진다)."""
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s "
                                "(run: python downloaders/binance_bulk.py --interval 1m --no-metrics)"
                                % symbol)
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    d = pd.read_parquet(p)
    keep = [c for c in cols if c in d.columns]
    return d[keep].sort_values("open_time").reset_index(drop=True)


MULTI = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")
_MULTI_CACHE: dict = {}


def load_liq(symbol: str, full_only: bool, exchange: str = "bybit") -> pd.DataFrame:
    """청산 원본. 기본은 Bybit 전건 피드.

    거래소를 합치지 않는 이유 (2026-07-31 실측, analysis/feed_bias.py)
      Binance forceOrder 는 심볼당 초당 1건에 걸려 있고(초당 2건 이상 0.38% vs
      Bybit 32.3%), 누락 정도가 청산 강도에 따라 변한다 — Bybit 대비 명목가 비율이
      조용할 때 88.7배에서 격렬할 때 0.48배로 185배 움직인다. OKX 도 같은 패턴이다.
      따라서 단순 합산하면 '가장 중요한 구간'에서 체계적으로 왜곡된다.
      물량은 Bybit 전건만 쓰고, Binance/OKX 는 가격 위치·편향 측정용으로만 둔다.
    """
    if not _MULTI_CACHE:
        if os.path.exists(MULTI):
            _MULTI_CACHE["d"] = pd.read_parquet(MULTI)
        else:
            _MULTI_CACHE["d"] = None
    d = _MULTI_CACHE["d"]

    if d is not None:
        sel = d[(d["symbol"] == symbol) & (d["exchange"] == exchange)]
        if not sel.empty:
            return sel[sel["full_feed"]] if full_only else sel

    # 폴백: 구 단일거래소 다운로드분
    p = os.path.join(TARDIS, "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError(
            "missing liquidation data for %s (run downloaders/tardis_multi.py)" % symbol)
    old = pd.read_parquet(p)
    return old[old["full_feed"]] if full_only else old


def build(symbol: str, bucket_min: int, horizon: int, full_only: bool) -> pd.DataFrame:
    liq = load_liq(symbol, full_only)
    if liq.empty:
        return pd.DataFrame()
    m1 = load_1m_with_volume(symbol)
    if m1.empty:
        return pd.DataFrame()

    step = bucket_min * MIN_MS
    liq = liq.copy()
    liq["bucket"] = (liq["ts_ms"] // step) * step

    # 방향별 청산 명목가를 버킷 단위로 집계
    g = (liq.groupby(["bucket", "pos_side"])["notional"].sum()
            .unstack(fill_value=0.0).rename(columns={"long": "liq_long", "short": "liq_short"}))
    for c in ("liq_long", "liq_short"):
        if c not in g.columns:
            g[c] = 0.0
    g = g.reset_index()

    # 1분봉을 같은 버킷으로 집계
    k = m1.copy()
    k["bucket"] = (k["open_time"] // step) * step
    agg = k.groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), n_bars=("close", "size")).reset_index()
    # 거래량은 1분봉에 있으면 사용
    if "volume" in m1.columns:
        agg = agg.merge(k.groupby("bucket")["volume"].sum().rename("volume").reset_index(),
                        on="bucket", how="left")
    else:
        agg["volume"] = np.nan

    df = agg.merge(g, on="bucket", how="left").fillna({"liq_long": 0.0, "liq_short": 0.0})
    df = df.sort_values("bucket").reset_index(drop=True)
    # 버킷이 연속인지 (샘플 데이터는 하루씩 떨어져 있어 경계 처리가 필요)
    df["contig"] = df["bucket"].diff().eq(step)

    # 사전 변동성/거래량 정규화 (직전 하루 창, 현재 버킷 제외 -> 룩어헤드 없음)
    win = max(int(1440 / bucket_min), 30)
    ret = df["close"] / df["close"].shift(1) - 1.0
    df["sigma"] = ret.shift(1).rolling(win, min_periods=win // 4).std()
    df["dollar_vol"] = df["volume"] * df["close"]
    df["adv"] = df["dollar_vol"].shift(1).rolling(win, min_periods=win // 4).mean()

    # 선도 변위: 청산 방향으로 얼마나 더 밀렸나 (최대 변위 = 슈팅 폭)
    fwd_low = df["low"].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    fwd_high = df["high"].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    df["push_long"] = (df["close"] / fwd_low - 1.0)    # 롱청산 -> 아래로 밀린 폭(양수)
    df["push_short"] = (fwd_high / df["close"] - 1.0)  # 숏청산 -> 위로 밀린 폭(양수)
    df["symbol"] = symbol
    return df


def panel(symbols: list[str], bucket_min: int, horizon: int, full_only: bool) -> pd.DataFrame:
    rows = []
    for s in symbols:
        try:
            d = build(s, bucket_min, horizon, full_only)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if d.empty:
            continue
        # 방향별로 긴 형식으로 편다 — 롱청산/숏청산을 한 표본으로 합친다
        for side, lcol, pcol in (("long", "liq_long", "push_long"),
                                 ("short", "liq_short", "push_short")):
            t = d[["bucket", "symbol", "sigma", "adv", "dollar_vol", "contig"]].copy()
            t["side"] = side
            t["liq_usd"] = d[lcol]
            t["push"] = d[pcol]
            rows.append(t)
        U.log("%s: %d buckets, %d with liquidations" % (s, len(d), int((d.liq_long + d.liq_short > 0).sum())))
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    p = p[p["contig"] & np.isfinite(p["push"]) & np.isfinite(p["sigma"]) & (p["sigma"] > 0)]
    p = p[np.isfinite(p["adv"]) & (p["adv"] > 0)]
    return p.reset_index(drop=True)


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """log y = a + b log x 최소제곱. 반환 (a, b, R2, n)."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = np.log(x[ok]), np.log(y[ok])
    if len(x) < 30:
        return (np.nan, np.nan, np.nan, len(x))
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (float(coef[0]), float(coef[1]), 1 - ss_res / max(ss_tot, 1e-12), len(x))


def main() -> int:
    ap = argparse.ArgumentParser(description="price impact of forced liquidations")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--bucket-min", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=5, help="buckets ahead to measure the push")
    ap.add_argument("--all-period", action="store_true",
                    help="include pre-2025-02-25 (throttled, under-counted) data")
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.MAJORS
    full_only = not a.all_period
    U.log("impact: bucket=%dmin horizon=%d full_feed_only=%s" % (a.bucket_min, a.horizon, full_only))

    p = panel(symbols, a.bucket_min, a.horizon, full_only)
    if p.empty:
        U.log("empty panel")
        return 1
    # 청산이 없는 버킷까지 저장하면 3100만 행(1.1GB)이 된다. 분석에 쓰는 행만 남긴다.
    U.atomic_write_parquet(p[p["liq_usd"] > 0],
                           os.path.join(C.DATA, "analysis", "impact_panel.parquet"))

    pd.set_option("display.width", 200)
    liq = p[p["liq_usd"] > 0].copy()
    print("\n=== 표본 ===")
    print("전체 버킷 %d | 청산 발생 버킷 %d | 청산 총액 $%.0fM"
          % (len(p), len(liq), liq["liq_usd"].sum() / 1e6))

    # 정규화: 변위는 사전 변동성 단위, 물량은 사전 ADV 단위
    liq["push_sig"] = liq["push"] / liq["sigma"]
    liq["v_adv"] = liq["liq_usd"] / liq["adv"]

    print("\n=== 충격함수 |dP|/sigma = a * (V/ADV)^b ===")
    a0, b, r2, n = fit_power(liq["v_adv"].to_numpy(), liq["push_sig"].to_numpy())
    print("  전체:  b = %.3f  (R2 %.3f, n=%d)" % (b, r2, n))
    print("  참고: b=0.5 이면 표준 제곱근 법칙, b>0.5 이면 초선형(캐스케이드 증폭)")
    for s, g in liq.groupby("symbol"):
        a1, b1, r21, n1 = fit_power(g["v_adv"].to_numpy(), g["push_sig"].to_numpy())
        print("  %-9s b = %.3f  (R2 %.3f, n=%d)" % (s, b1, r21, n1))

    print("\n=== 물량 구간별 변위 분포 (확률적 추정치) ===")
    liq["q"] = pd.qcut(liq["v_adv"].rank(method="first"), 8, labels=False)
    t = liq.groupby("q").agg(
        n=("push_sig", "size"),
        V_adv_med=("v_adv", "median"),
        liq_usd_med=("liq_usd", "median"),
        push_p25=("push_sig", lambda s: s.quantile(0.25)),
        push_med=("push_sig", "median"),
        push_p75=("push_sig", lambda s: s.quantile(0.75)),
        push_p95=("push_sig", lambda s: s.quantile(0.95)),
    )
    print(t.round(3).to_string())
    print("  push_* 단위 = 사전 변동성(sigma) 배수. 예: 2.0 이면 평소 1버킷 변동의 2배 밀림")

    print("\n=== 핵심 검정: 강제청산 $1은 평범한 거래량 $1보다 더 미는가 ===")
    d = liq[(liq["dollar_vol"] > 0)].copy()
    d["other_vol"] = (d["dollar_vol"] - d["liq_usd"]).clip(lower=1.0)
    X = np.column_stack([np.ones(len(d)),
                         np.log(d["liq_usd"].clip(lower=1.0)),
                         np.log(d["other_vol"])])
    y = np.log(d["push_sig"].clip(lower=1e-6))
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(y) > 100:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        s2 = float(resid @ resid) / max(len(y) - X.shape[1], 1)
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        print("  log(청산물량)  계수 %.4f  (SE %.4f, t=%.2f)" % (coef[1], se[1], coef[1] / se[1]))
        print("  log(기타거래량) 계수 %.4f  (SE %.4f, t=%.2f)" % (coef[2], se[2], coef[2] / se[2]))
        print("  n=%d" % len(y))
        print("  청산 계수가 유의하게 크면 -> 강제청산은 단순 거래량이 아니다(지도를 볼 이유가 있다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
