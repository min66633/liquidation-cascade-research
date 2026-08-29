# -*- coding: utf-8 -*-
"""R-4 — R-2 의 '11배 초과' 가 정말 자기여기 증폭인가.

무엇을 가르는가
  R-2 는 캐스케이드 뒤 되돌림이 제곱근 법칙 예측의 **11배**라는 것을 냈고,
  그 초과가 1/(1-n) = 5.03 (n=0.801) 증폭과 정합한다고 했다. 그런데 그 n 은
  **타임스탬프 Hawkes 적합값이고 잔차 진단이 불합격**이었다. 즉 정합한다는 진술의
  기준점 자체가 흔들린다.

  여기서는 n 을 시각이 아니라 **OI 파괴량에서 직접** 잰다.
  분기과정에서 총 사건 크기는 S = S0 / (1-n) 이므로
      A = S / S0  ->  n = 1 - 1/A
  S0 = 방아쇠 바에서 파괴된 OI, S = 캐스케이드가 멈출 때까지 누적 파괴 OI.
  타임스탬프도 커널 가정도 쓰지 않는다. **Hawkes 와 독립인 추정량이다.**

핵심 검정 (2절)
  증폭 가설이 맞다면 되돌림은
      y ~ Y_ref * sigma * sqrt(S0/ADV) * A
  이어야 한다. 그래서 y / (sigma*sqrt(S0/ADV)) 를 A 에 회귀했을 때
  **기울기가 Y_ref=1.33 근처, 절편이 0 근처**면 초과분이 전부 증폭으로 설명된다.
  A 가 아무것도 예측하지 못하면 증폭 이야기는 기각된다.

*** 룩어헤드 경고 ***
  A 는 방아쇠 바 **이후** 바들을 써서 계산한다. 진입 시점에는 모르는 값이다.
  따라서 이 스크립트는 **기전 설명**이지 매매 신호가 아니다. 실시간으로 쓰려면
  호가에서 n 을 재야 하고(N-1) 그건 웹소켓 축적을 기다려야 한다.

실행:
    python analysis/amplification.py
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
from analysis.event_study_h2 import load, find_events, BAR_MS   # noqa: E402
from analysis.response_liq import ols_cluster, cmean            # noqa: E402
from analysis.scale_check import K, DOI_THR, MIN_GAP, Y_REF, VOL_WIN   # noqa: E402

LAGS = [1, 3, 6, 12, 24, 48]          # 5분바 -> 5,15,30,60,120,240분
MAXB = 24                              # 캐스케이드 추적 최대 바(2시간)
LIQ = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")


def build(symbols, cont_thr: float) -> pd.DataFrame:
    """이벤트별 증폭 배율 A = S/S0 와 실현 되돌림."""
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, K, DOI_THR, MIN_GAP)
        if len(ev) == 0:
            continue
        op = df["open"].to_numpy(dtype=np.float64)
        cl = df["close"].to_numpy(dtype=np.float64)
        qv = df["quote_volume"].to_numpy(dtype=np.float64)
        ret = df["ret"].to_numpy(dtype=np.float64)
        oiv = df["sum_open_interest_value"].to_numpy(dtype=np.float64)
        doi = df["doi"].to_numpy(dtype=np.float64)
        ctg = df["contig"].to_numpy(dtype=bool)
        ot = df["open_time"].to_numpy()
        n = len(df)
        sig = (pd.Series(ret).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .std().to_numpy()) * np.sqrt(float(VOL_WIN))
        adv = (pd.Series(qv).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .mean().to_numpy()) * float(VOL_WIN)

        for r in ev.itertuples():
            if not r.is_liq:
                continue                      # 증폭은 청산 이벤트에서만 정의된다
            i = int(r.i)
            j = i + 1
            if j >= n or not (np.isfinite(op[j]) and op[j] > 0):
                continue
            if not (np.isfinite(sig[i]) and sig[i] > 0
                    and np.isfinite(adv[i]) and adv[i] > 0):
                continue
            if not (np.isfinite(doi[i]) and doi[i] < 0 and oiv[i] > 0):
                continue

            s0 = -doi[i] * oiv[i]
            tot, gen = s0, 1
            # 캐스케이드 지속: OI 가 계속 줄어드는 동안. cont_thr 로 '멈춤' 을 정의한다.
            for t in range(i + 1, min(i + 1 + MAXB, n)):
                if not (ctg[t] and np.isfinite(doi[t]) and oiv[t] > 0):
                    break
                if doi[t] >= cont_thr:
                    break
                tot += -doi[t] * oiv[t]
                gen += 1

            rec = {"symbol": s, "side": int(r.side), "ts": int(ot[i]),
                   "day": int(ot[i] // 86_400_000),
                   "s0": s0, "stot": tot, "A": tot / s0, "gen": gen,
                   "sig_d": sig[i], "adv": adv[i]}
            for L in LAGS:
                t = j + L
                rec["rev%d" % L] = ((cl[t] / op[j] - 1.0) * r.side * 1e4
                                    if t < n else np.nan)
            out.append(rec)
    return pd.DataFrame(out)


def sec1(d: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("1. 증폭 배율 A = S/S0 — OI 파괴량에서 직접. Hawkes 와 독립")
    print("=" * 78)
    print("  분기과정: S = S0/(1-n)  ->  n = 1 - 1/A")
    print("  S0 = 방아쇠 바의 파괴 OI, S = 멈출 때까지 누적.\n")
    A = d["A"].to_numpy()
    print("  이벤트 %d건" % len(d))
    print("  A  : 평균 %.3f | 중앙 %.3f | p75 %.3f | p90 %.3f | 최대 %.2f"
          % (A.mean(), np.median(A), np.quantile(A, .75), np.quantile(A, .9), A.max()))
    print("  세대(바): 중앙 %.0f | p90 %.0f | 최대 %d | 1바에서 끝난 비율 %.1f%%"
          % (np.median(d["gen"]), np.quantile(d["gen"], .9), d["gen"].max(),
             100 * (d["gen"] == 1).mean()))
    m, se, t, _ = cmean(A, d["day"].to_numpy())
    print("\n  평균 A = %.3f (일클러스터 SE %.3f)  ->  **n = 1 - 1/A = %.3f**"
          % (m, se, 1 - 1 / m if m > 0 else np.nan))
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print("  A 의 95%% CI [%.3f, %.3f]  ->  n 의 CI [%.3f, %.3f]"
          % (lo, hi, 1 - 1 / lo if lo > 0 else np.nan, 1 - 1 / hi if hi > 0 else np.nan))
    print("\n  대조: 타임스탬프 Hawkes 멱함수 n = 0.801 -> A = 5.03 (잔차 불합격)")
    print("        타임스탬프 Hawkes 지수     n = 0.669 -> A = 3.02")
    print("  *** A 가 이보다 훨씬 작으면 '증폭이 초과 11배를 설명한다' 는 성립하지 않는다.")


def sec2(d: pd.DataFrame, lag: int) -> None:
    print("\n" + "=" * 78)
    print("2. 핵심 검정 — 되돌림이 A 에 비례하는가")
    print("=" * 78)
    print("  증폭 가설:  y = Y_ref * sigma * sqrt(S0/ADV) * A")
    print("  -> y / (sigma*sqrt(S0/ADV)) 를 A 에 회귀하면 기울기 ~ Y_ref = %.2f," % Y_REF)
    print("     절편 ~ 0 이어야 한다. A 가 무력하면 증폭 이야기는 기각된다.\n")
    base = (d["sig_d"].to_numpy() * np.sqrt(d["s0"].to_numpy() / d["adv"].to_numpy())
            * 1e4)
    A = d["A"].to_numpy()
    day = d["day"].to_numpy()
    print("  %6s | %10s %7s | %10s %7s | %9s" % ("지연(분)", "기울기", "t", "절편", "t", "n"))
    for L in LAGS:
        y = d["rev%d" % L].to_numpy() / np.where(base > 0, base, np.nan)
        m = np.isfinite(y) & np.isfinite(A)
        if m.sum() < 100:
            continue
        X = np.column_stack([np.ones(int(m.sum())), A[m]])
        b, se, _ = ols_cluster(X, y[m], day[m])
        print("  %6d | %10.3f %7.1f | %10.3f %7.1f | %9d"
              % (5 * L, b[1], b[1] / se[1] if se[1] > 0 else np.nan,
                 b[0], b[0] / se[0] if se[0] > 0 else np.nan, int(m.sum())))
    lag_c = "rev%d" % lag
    print("\n  [2b] A 오분위별 되돌림 (원 단위 bp) — 단조 증가해야 한다")
    dd = d.copy()
    dd["bin"] = pd.qcut(dd["A"], 5, labels=False, duplicates="drop")
    print("    %4s %7s %10s %12s %12s"
          % ("분위", "n", "A 중앙", "되돌림 bp", "t"))
    for q in sorted(dd["bin"].dropna().unique()):
        g = dd[dd["bin"] == q]
        a, _, t, _ = cmean(g[lag_c].to_numpy(), g["day"].to_numpy())
        print("    %4d %7d %10.3f %12.2f %12.1f"
              % (q, len(g), g["A"].median(), a, t))

    print("\n  [2c] A 를 총물량으로 흡수하면? y ~ Y * sigma*sqrt(S_tot/ADV)")
    print("       (증폭이 '물량이 늘어난 것' 이라면 이 Y 가 1.33 근처여야 한다)")
    x2 = (d["sig_d"].to_numpy()
          * np.sqrt(d["stot"].to_numpy() / d["adv"].to_numpy()) * 1e4)
    y2 = d[lag_c].to_numpy()
    m = np.isfinite(y2) & np.isfinite(x2) & (x2 > 0)
    b, se, _ = ols_cluster(x2[m][:, None], y2[m], day[m])
    print("       Y(총물량 기준) = %.2f +- %.2f (t=%.1f)  vs Y_ref = %.2f"
          % (b[0], se[0], b[0] / se[0] if se[0] > 0 else np.nan, Y_REF))
    b2, se2, _ = ols_cluster(base[m][:, None], y2[m], day[m])
    print("       Y(방아쇠만 기준) = %.2f +- %.2f (t=%.1f)"
          % (b2[0], se2[0], b2[0] / se2[0] if se2[0] > 0 else np.nan))


def sec3() -> None:
    """theta 구간추정량(Ferro-Segers) — n 의 세 번째 독립 추정."""
    print("\n" + "=" * 78)
    print("3. 극단지표 theta 구간추정량 — n 의 세 번째 경로")
    print("=" * 78)
    print("  theta ~ 1-n 이므로 theta 를 직접 재면 n 이 나온다. Ferro-Segers(2003)")
    print("  구간추정량은 커널 가정도 임계 초과 개수 가정도 쓰지 않는다.\n")
    if not os.path.exists(LIQ):
        print("  청산 프린트 없음 — 건너뜀")
        return
    d = pd.read_parquet(LIQ)
    d = d[(d["exchange"] == "bybit") & d["full_feed"]]
    if len(d) < 1000:
        print("  전건 표본 부족 — 건너뜀")
        return
    print("  %-10s %8s %8s %9s %9s" % ("심볼", "초과 N", "임계 p", "theta", "n=1-theta"))
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        g = d[d["symbol"] == s]
        if len(g) < 2000:
            continue
        for p in (0.95, 0.99):
            u = float(g["notional"].quantile(p))
            t = np.sort(g.loc[g["notional"] > u, "ts_ms"].to_numpy()) / 1000.0
            if len(t) < 50:
                continue
            S = np.diff(t)
            S = S[S > 0]
            N = len(S) + 1
            if len(S) < 20:
                continue
            if S.max() <= 2:
                th = 2.0 * S.sum() ** 2 / ((N - 1) * np.sum(S ** 2))
            else:
                th = (2.0 * np.sum(S - 1.0) ** 2
                      / ((N - 1) * np.sum((S - 1.0) * (S - 2.0))))
            th = float(min(1.0, max(1e-6, th)))
            print("  %-10s %8d %8.3f %9.3f %9.3f" % (s, N, p, th, 1 - th))
    print("\n  n = 1-theta 가 0.80 근처면 Hawkes 값과 정합. 훨씬 작으면 불일치다.")


def main() -> int:
    ap = argparse.ArgumentParser(description="R-4 cascade amplification")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--cont", type=float, default=0.0,
                    help="캐스케이드 지속 판정: doi 가 이 값 이상이면 종료")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("R-4 — '11배 초과' 가 자기여기 증폭인가")
    print("*** A 는 방아쇠 이후 바를 쓴다 = 룩어헤드. **기전 설명**이지 매매 신호가 아니다.")

    d = build(syms, a.cont)
    if len(d) == 0:
        print("이벤트 없음")
        return 1
    t0 = pd.to_datetime(d["ts"].min(), unit="ms")
    t1 = pd.to_datetime(d["ts"].max(), unit="ms")
    print("\n**사용 데이터 기간: %s ~ %s / %d종 / 5분봉 / 청산 이벤트 %d건**"
          % (str(t0)[:10], str(t1)[:10], d["symbol"].nunique(), len(d)))

    sec1(d)
    sec2(d, a.lag)
    sec3()

    print("\n" + "=" * 78)
    print("민감도: --cont 로 '캐스케이드 종료' 정의를 바꿔 재실행할 것")
    print("  --cont 0.0    OI 가 늘기 시작하면 종료(기본)")
    print("  --cont -0.001 아주 약한 감소도 종료로 간주 -> A 가 작아진다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
