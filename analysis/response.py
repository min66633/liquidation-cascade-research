# -*- coding: utf-8 -*-
"""응답함수 R(l) — '청산은 가격을 밀지 않는다' 가 검정력 문제였는지 판정한다.

왜 이것인가 (프로젝트 골격이 걸려 있다)
  현재 결론: 충격지수 b ~ 0.04, R^2 ~ 0, t = 1.02  ->  "청산은 가격을 못 민다"
             -> 채널 A(충격-반동) 폐기, 채널 B(유동성 프리미엄)만 생존.
  MODEL.md / MECHANISM.md / EXPLAINER.md 가 전부 이 판정 위에 서 있다.

  그런데 **R^2 는 임팩트를 검정하는 통계량이 아니다.**
  주식시장에서도 개별 체결 단위 회귀는 R^2 ~ 0 인데 임팩트는 정밀하게 측정된다.
  올바른 추정량은 **조건부 평균**인 응답함수다:

      R(l) = E[ (p_{t+l} - p_t) * eps_t ]        eps_t = 그 시각 주문흐름의 부호

  개별 분산이 아무리 커도 N 건을 평균내면 표준오차가 sigma/sqrt(N) 로 좁아진다.
  R^2 = 0 과 R(l) != 0 은 완벽히 양립한다.

제곱근 법칙 대조
      dP = Y * sigma * sqrt(Q/V)          모수 1개, 표본 내 식별 가능
  vs  max(0, V/D - c)^beta                모수 3개, 표본 내 식별 불가
  Y 를 적합하고, 관측된 t 값이 그 법칙 하에서 예측되는 값과 맞는지 본다.
  맞으면 "임팩트 없음" 이 아니라 "검정력 없음" 이었다는 뜻이다.

*** 검정력 계산을 반드시 함께 낸다. 귀무 결과를 부재의 증거로 읽지 않기 위해. ***

실행:
    python analysis/response.py
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
LAGS = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]      # 1분봉 기준 지연(분)


def load_1m(s: str) -> pd.DataFrame:
    cols = ["open_time", "close", "quote_volume", "taker_buy_quote_volume"]
    df = pd.read_parquet(os.path.join(BULK, "klines_1m", "%s.parquet" % s))
    keep = [c for c in cols if c in df.columns]
    return df[keep].sort_values("open_time").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="response function R(l) and square-root law")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--max-bars", type=int, default=600_000)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 76)
    print("응답함수 R(l) — 'R^2 ~ 0' 이 임팩트 부재인가 검정력 부족인가")
    print("=" * 76)

    agg = {L: [] for L in LAGS}
    ysum, ycnt = [], []
    tot_bars = 0
    for s in syms:
        try:
            d = load_1m(s)
        except FileNotFoundError:
            continue
        if "taker_buy_quote_volume" not in d.columns or len(d) < 10_000:
            continue
        d = d.tail(a.max_bars)
        cl = d["close"].to_numpy()
        qv = d["quote_volume"].to_numpy()
        tb = d["taker_buy_quote_volume"].to_numpy()
        n = len(cl)
        net = 2.0 * tb - qv                       # 테이커 매수 - 매도 (명목가)
        eps = np.sign(net)
        ok = (qv > 0) & np.isfinite(eps) & (eps != 0) & np.isfinite(cl) & (cl > 0)
        tot_bars += int(ok.sum())
        for L in LAGS:
            if n <= L + 2:
                continue
            # 임팩트는 **그 바 안에서** 일어나 close[t] 에 이미 반영돼 있다.
            # 따라서 임팩트를 재려면 시작점이 close[t-1] 이어야 한다.
            # close[t] 에서 시작하면 임팩트가 아니라 그 뒤의 되돌림만 잰다.
            r_imp = np.full(n, np.nan)          # 임팩트 포함 (t-1 -> t+L)
            r_rev = np.full(n, np.nan)          # 되돌림만  (t   -> t+L)
            r_imp[1:n - L] = (cl[1 + L:] / cl[:n - L - 1] - 1.0) * 1e4
            r_rev[:n - L] = (cl[L:] / cl[:n - L] - 1.0) * 1e4
            m = ok & np.isfinite(r_imp) & np.isfinite(r_rev)
            if m.sum() > 100:
                agg[L].append(np.column_stack([r_imp[m] * eps[m], r_rev[m] * eps[m]]))
        # 제곱근 법칙용: 바별 |순유량|/거래량 과 |수익|
        sig = pd.Series(cl).pct_change().rolling(1440, min_periods=200).std().to_numpy()
        m2 = ok & np.isfinite(sig) & (sig > 0)
        if m2.sum() > 1000:
            ysum.append(np.abs(np.diff(np.append(cl[0], cl))[m2] / cl[m2]))
            ycnt.append((sig[m2], np.abs(net[m2]) / qv[m2]))

    print("총 1분봉 %d개 / %d종\n" % (tot_bars, len(syms)))
    print("--- 1. 응답함수 R(l) = E[(p_{t+l}-p_t) * eps_t]  (bp) ---")
    print("  %6s %11s %8s %11s %8s %12s %10s"
          % ("지연(분)", "임팩트bp", "t", "되돌림bp", "t", "n", "R^2(참고)"))
    for L in LAGS:
        if not agg[L]:
            continue
        v = np.concatenate(agg[L], axis=0)
        out = []
        for j in (0, 1):
            x = v[:, j]
            mm = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(len(x)))
            out += [mm, mm / se if se > 0 else np.nan]
        x = v[:, 0]
        r2 = out[0] ** 2 / float(x.var(ddof=1)) if x.var(ddof=1) > 0 else np.nan
        print("  %6d %11.4f %8.1f %11.4f %8.1f %12d %10.2e"
              % (L, out[0], out[1], out[2], out[3], len(v), r2))
    print("  R(l) 이 유의하게 양수면 '주문흐름이 가격을 민다' 는 뜻이다.")
    print("  R^2 열이 사실상 0 인데 t 가 크면 -> **R^2 는 틀린 통계량**이라는 증거다.")

    print("\n--- 2. 검정력: 제곱근 법칙이 참일 때 예상되는 t ---")
    print("  dP = Y * sigma * sqrt(Q/V).  Y=0.5 로 두고 우리 이벤트 규모를 넣는다.")
    ep = os.path.join(C.DATA, "analysis", "events_all_k8_doi-0.02_gap12.parquet")
    if os.path.exists(ep):
        ev = pd.read_parquet(ep)
        oi_drop = (ev["doi_mag"] * ev["oiv"]).to_numpy()
        print("  이벤트 %d건 | OI감소액 중앙 $%.4g" % (len(ev), np.median(oi_drop)))
        for lab, Q in (("중앙", np.median(oi_drop)), ("p90", np.quantile(oi_drop, .9))):
            for ADV in (1e9, 5e9):
                for sig_d in (0.03, 0.05):
                    V5 = ADV / 288.0                       # 5분 거래량
                    sig5 = sig_d / np.sqrt(288.0)
                    dP = 0.5 * sig5 * np.sqrt(Q / V5) * 1e4
                    noise = sig5 * 1e4
                    t_exp = dP / (noise / np.sqrt(len(ev)))
                    print("    %-4s Q=$%.3g  ADV=$%.0e  일sigma=%.0f%%"
                          "  ->  dP=%.2fbp  기대 t=%.2f"
                          % (lab, Q, ADV, 100 * sig_d, dP, t_exp))
        print("  기대 t 가 1 근처면, 관측된 t=1.02 는 **법칙이 참일 때 나올 값**이다.")
        print("  즉 귀무 결과가 부재의 증거가 아니라 검정력 부재의 증거다.")
    else:
        print("  이벤트 캐시 없음 — x_dist.py 를 먼저 돌릴 것")

    print("\n  *** 한계: eps 를 1분봉 테이커 순유량 부호로 근사했다.")
    print("      진짜 체결 단위 부호(aggTrade)가 아니므로 R(l) 은 하한이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
