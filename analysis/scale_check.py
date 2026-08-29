# -*- coding: utf-8 -*-
"""R-2 — R-1 이 잰 sqrt(Q) 되돌림 법칙이 캐스케이드 규모로 확장되는가.

무엇을 하는가
  R-1(analysis/response_liq.py)은 **평시 규모**에서 청산 뒤 되돌림이 sqrt(Q) 에
  비례해 커진다는 것을 두 표본에서 유의하게 냈다(Y(rev5) = -0.92 / -1.33).
  그런데 그 표본에는 캐스케이드 규모가 **구조적으로 없다**(Tardis 무료 = 매월 1일).

  여기서는 청산 프린트 대신 **OI 급감**을 청산 물량 대용으로 써서, 실제
  캐스케이드 규모 이벤트에서 같은 법칙이 성립하는지 본다. 유료 데이터 불필요.

사전등록 예측 (실행 전에 정해 둔다 — 사후 맞추기 방지)
  **거래소 단위를 끝까지 일치시킨다.** R-1 의 Y_bybit = 1.33 (rev5) 은 bybit 전건
  프린트 물량 Q_bybit 에 대해, Binance ADV 를 분모로 적합됐다. 그러므로 예측도
      Q_event = (OI 감소액) x (bybit 청산 / OI감소 비율)
  로 같은 단위로 넣는다. 이렇게 하면 시장점유 가정(0.245 / 0.47)이 양변에서
  상쇄돼 **가정 없는 자기정합 예측**이 된다.
  -> 사전등록: 기울기 Y = 1.33 (95% CI 를 보수적으로 잡아 0.7 ~ 2.0).

  (첫 판에서 Y_full = 0.38~0.66 으로 환산했던 것은 **이중계산**이었다. 분자만
   시장 전체로 올리고 분모의 OI 감소액은 Binance 단위로 두었다.)

세 가지 함정을 명시적으로 막는다
  (1) 룩어헤드 — 바닥(bot_i) 기준으로 재면 안 된다. 이전에 잡힌 오류다.
      기준가는 **판정 시점 이후 첫 체결 가능 가격** = open[i+1].
  (2) 위약 부재 — 큰 움직임은 청산이 없어도 되돌린다. find_events 가 같은 |z|
      임계에서 OI가 **유지/증가**한 바(is_liq=False)를 주므로 그것과 대조한다.
  (3) Q 단위 불일치 — OI 감소액에는 자발적 청산도 섞인다. 1절에서 Tardis 겹침
      구간으로 청산/OI감소 비율을 실측해 보정한다.

실행:
    python analysis/scale_check.py
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

LIQ = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")
EVCACHE = os.path.join(C.DATA, "analysis", "events_all_k8_doi-0.02_gap12.parquet")

K = 8.0                      # 이벤트 임계(과거 sigma 배수) — 캐시 파일명과 동일
DOI_THR = -0.02
MIN_GAP = 12
LAGS = [1, 3, 6, 12, 24, 48]          # 5m 바 단위 -> 5,15,30,60,120,240분
VOL_WIN = 288                          # 하루

Y_REF = 1.33                           # R-1 bybit 전건 rev5 적합값 (같은 단위)
Y_LO, Y_HI = 0.70, 2.00                # 사전등록 예측 구간 (보수적)
N_BRANCH = 0.801                       # Hawkes 멱함수 커널 분기비 (잔차 불합격 주의)


# ------------------------------------------------------------------ 1. Q 다리
def q_bridge() -> float:
    """청산 프린트 명목가 / OI 감소액 비율. Tardis 겹침 구간에서 실측."""
    print("=" * 78)
    print("1. Q 다리 — OI 감소액 중 실제 청산은 얼마인가")
    print("=" * 78)
    print("  OI 감소에는 자발적 청산도 섞인다. R-1 의 Y 는 **청산 프린트** 물량에")
    print("  대해 적합됐으므로, OI 감소액을 그대로 쓰면 Q 를 과대평가한다.\n")
    if not os.path.exists(LIQ):
        print("  청산 프린트 없음 — 보정 없이 진행(상한만 산출)")
        return np.nan

    raw = pd.read_parquet(LIQ)
    out = {}
    for tag, sel in (("binance-futures (스로틀)", raw["exchange"] == "binance-futures"),
                     ("bybit 전건", (raw["exchange"] == "bybit") & raw["full_feed"])):
        out[tag] = _bridge_one(raw[sel], tag)
    print("\n  *** 두 피드를 비교하는 이유: Binance forceOrder 는 초당 1건 스냅샷이라")
    print("      **버스트 중에 더 심하게** 잘린다. 큰 OI 감소에서 비율이 떨어지는 것이")
    print("      실체인지 스로틀 인공물인지는 전건 피드로만 가릴 수 있다.")
    b = out.get("bybit 전건")
    if b is not None and np.isfinite(b[1]):
        print("\n  채택: bybit 전건 상위5%% 비율 **%.4f** — 시장점유 환산을 하지 않는다." % b[1])
        print("  이유: R-1 의 Y_bybit=%.2f 도 **bybit 관측 물량**에 대해 적합됐다."
              % Y_REF)
        print("  같은 단위로 예측하면 시장점유 가정(0.245 / 0.47)이 양변에서 상쇄돼")
        print("  **가정 없는 자기정합 예측**이 된다. 앞 판의 환산은 이중계산이었다.")
        return float(b[1])
    return float(out["binance-futures (스로틀)"][1])


def _bridge_one(d: pd.DataFrame, tag: str):
    """한 피드에 대해 청산명목가/OI감소액 비율을 크기 구간별로."""
    if len(d) == 0:
        return (np.nan, np.nan)
    d = d.copy()
    d["bar"] = (d["ts_ms"] // BAR_MS) * BAR_MS
    g = d.groupby(["symbol", "bar"], as_index=False)["notional"].sum()

    rows = []
    for s in sorted(g["symbol"].unique()):
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        sub = g[g["symbol"] == s]
        m = df.merge(sub[["bar", "notional"]], left_on="open_time", right_on="bar",
                     how="inner")
        if len(m) == 0:
            continue
        # OI 가 실제로 줄어든 바만. doi 는 룩어헤드 없는 정의(load 참조).
        m = m[(m["doi"] < 0) & (m["sum_open_interest_value"] > 0) & m["doi"].notna()]
        if len(m) == 0:
            continue
        rows.append(pd.DataFrame({
            "symbol": s,
            "q_liq": m["notional"].to_numpy(),
            "q_oi": (-m["doi"] * m["sum_open_interest_value"]).to_numpy(),
            "doi_mag": (-m["doi"]).to_numpy()}))
    if not rows:
        print("  [%s] 겹치는 바가 없다" % tag)
        return (np.nan, np.nan)

    b = pd.concat(rows, ignore_index=True)
    b = b[(b["q_oi"] > 0) & (b["q_liq"] > 0)]
    if len(b) < 100:
        return (np.nan, np.nan)
    b["ratio"] = b["q_liq"] / b["q_oi"]
    print("\n  [%s] 겹치는 5분바 %d개 / %d종" % (tag, len(b), b["symbol"].nunique()))
    print("  %-12s %8s %10s %10s %10s" % ("OI감소 크기", "n", "비율중앙", "p25", "p75"))
    qs = b["doi_mag"].quantile([0, .5, .8, .95, .99, 1.0]).to_numpy()
    for j, lab in enumerate(["하위50%", "50-80%", "80-95%", "95-99%", "상위1%"]):
        m = (b["doi_mag"] >= qs[j]) & (b["doi_mag"] <= qs[j + 1])
        if m.sum() < 20:
            continue
        r = b.loc[m, "ratio"]
        print("  %-12s %8d %10.4f %10.4f %10.4f"
              % (lab, int(m.sum()), r.median(), r.quantile(.25), r.quantile(.75)))
    share = float(b["ratio"].median())
    big = b[b["doi_mag"] >= b["doi_mag"].quantile(.95)]
    share_big = float(big["ratio"].median()) if len(big) >= 20 else np.nan
    print("  전체 중앙 %.4f | 상위 5%% 중앙 %.4f  (하위50%%/상위1%% 비 %.2f)"
          % (share, share_big, qs[0] * 0 + b.loc[b.doi_mag <= qs[1], "ratio"].median()
             / max(b.loc[b.doi_mag >= qs[4], "ratio"].median(), 1e-12)))
    return (share, share_big)


# ------------------------------------------------ 2. 이벤트 + 위약 + 실현 되돌림
def build_events(symbols) -> pd.DataFrame:
    """캐스케이드 이벤트와 위약(OI 유지) 이벤트를 실현 되돌림과 함께 만든다."""
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
        hi = df["high"].to_numpy(dtype=np.float64)
        lo = df["low"].to_numpy(dtype=np.float64)
        qv = df["quote_volume"].to_numpy(dtype=np.float64)
        ret = df["ret"].to_numpy(dtype=np.float64)
        oiv = df["sum_open_interest_value"].to_numpy(dtype=np.float64)
        doi = df["doi"].to_numpy(dtype=np.float64)
        ot = df["open_time"].to_numpy()
        n = len(df)

        # 과거만 쓰는 변동성/거래량 (현재 바 제외 -> shift(1))
        sig = (pd.Series(ret).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .std().to_numpy()) * np.sqrt(float(VOL_WIN))     # 일 변동성
        adv = (pd.Series(qv).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .mean().to_numpy()) * float(VOL_WIN)             # 일 거래대금

        for r in ev.itertuples():
            i = int(r.i)
            j = i + 1                       # 체결 가능한 첫 바 (판정은 i 종가 시점)
            if j >= n or not np.isfinite(op[j]) or op[j] <= 0:
                continue
            if not (np.isfinite(sig[i]) and sig[i] > 0
                    and np.isfinite(adv[i]) and adv[i] > 0):
                continue
            rec = {"symbol": s, "i": i, "side": int(r.side), "is_liq": bool(r.is_liq),
                   "ts": int(ot[i]), "day": int(ot[i] // 86_400_000),
                   "sig_d": sig[i], "adv": adv[i],
                   "q_oi": (-doi[i] * oiv[i]) if np.isfinite(doi[i]) else np.nan,
                   "z": ret[i] / (sig[i] / np.sqrt(float(VOL_WIN)))}
            for L in LAGS:
                t = j + L
                rec["rev%d" % L] = ((cl[t] / op[j] - 1.0) * r.side * 1e4
                                    if t < n else np.nan)
                # 경로: 진입 후 최대 역행(MAE)/순행(MFE). 끝점만 보면 못 보는 것.
                if t < n:
                    seg_lo, seg_hi = lo[j:t + 1], hi[j:t + 1]
                    if r.side == 1:      # 롱 진입: 역행 = 더 내려감
                        rec["mae%d" % L] = (seg_lo.min() / op[j] - 1.0) * 1e4
                        rec["mfe%d" % L] = (seg_hi.max() / op[j] - 1.0) * 1e4
                    else:                # 숏 진입: 부호 반전
                        rec["mae%d" % L] = -(seg_hi.max() / op[j] - 1.0) * 1e4
                        rec["mfe%d" % L] = -(seg_lo.min() / op[j] - 1.0) * 1e4
                else:
                    rec["mae%d" % L] = np.nan
                    rec["mfe%d" % L] = np.nan
            out.append(rec)
    return pd.DataFrame(out)


def path_analysis(ev: pd.DataFrame) -> None:
    """5. 경로 — 평균 뒤에 무엇이 있는가.

    끝점 대 끝점 수익(close[t+L]/open[j])은 다음을 전혀 말해주지 않는다:
      - 진입 후 얼마나 더 내려갔는가 (MAE)
      - 회복이 유지되는가, 되돌렸다가 다시 밀리는가
      - 평균이 소수의 대박에서 나오는가
    """
    d = ev[ev["is_liq"]]
    print("\n" + "=" * 78)
    print("5. 경로 — '평균 되돌림' 뒤의 분포와 진행")
    print("=" * 78)
    print("  진입: open[i+1]. side 방향으로 부호 정렬(양수 = 이익 방향).\n")
    print("  [5a] 끝점 수익 **분포** (평균 하나로 말하면 안 되는 이유)")
    print("  %6s | %8s %8s | %8s %8s %8s %8s %8s | %7s"
          % ("지연(분)", "평균", "중앙", "p5", "p25", "p75", "p95", "표준편차", "승률%"))
    for L in LAGS:
        x = d["rev%d" % L].dropna().to_numpy()
        if len(x) < 50:
            continue
        q = np.percentile(x, [5, 25, 75, 95])
        print("  %6d | %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f %8.1f | %7.1f"
              % (5 * L, x.mean(), np.median(x), q[0], q[1], q[2], q[3],
                 x.std(ddof=1), 100 * (x > 0).mean()))

    print("\n  [5b] 최대 역행 MAE — 진입 후 **얼마나 더 내려가는가**")
    print("  %6s | %10s %10s %10s %10s | %12s"
          % ("지연(분)", "MAE 중앙", "MAE p25", "MAE p5", "MFE 중앙", "MAE<-100bp %"))
    for L in LAGS:
        a = d["mae%d" % L].dropna().to_numpy()
        f = d["mfe%d" % L].dropna().to_numpy()
        if len(a) < 50:
            continue
        print("  %6d | %10.1f %10.1f %10.1f %10.1f | %12.1f"
              % (5 * L, np.median(a), np.percentile(a, 25), np.percentile(a, 5),
                 np.median(f), 100 * (a < -100).mean()))
    print("  MAE 가 크면 '평균은 +여도 중간에 크게 물린다' 는 뜻이다.")
    print("  지정매수라면 이 낙폭이 곧 진입 기회지만, 시장가 진입이면 순손실 구간이다.")

    print("\n  [5c] 되돌렸다가 **다시 밀리는가** — 사용자 지적 직접 검정")
    base = d["rev3"].to_numpy()          # 15분 시점
    ok = np.isfinite(base)
    up = ok & (base > 0)
    print("    15분에 이익이던 %d건 중, 이후에도 이익인 비율:" % int(up.sum()))
    for L in (6, 12, 24, 48):
        y = d["rev%d" % L].to_numpy()
        m = up & np.isfinite(y)
        if m.sum() < 30:
            continue
        print("      %4d분 후: %5.1f%% 유지 | 평균 %7.1f bp | 15분 대비 변화 %7.1f bp"
              % (5 * L, 100 * (y[m] > 0).mean(), y[m].mean(),
                 y[m].mean() - base[m].mean()))
    dn = ok & (base <= 0)
    print("    15분에 손실이던 %d건의 이후 평균:" % int(dn.sum()))
    for L in (6, 12, 24, 48):
        y = d["rev%d" % L].to_numpy()
        m = dn & np.isfinite(y)
        if m.sum() < 30:
            continue
        print("      %4d분 후: 평균 %7.1f bp | 회복 비율 %5.1f%%"
              % (5 * L, y[m].mean(), 100 * (y[m] > 0).mean()))

    print("\n  [5d] 평균이 소수 대박에서 오는가 — 상위 절사")
    for L in (1, 3, 12):
        x = d["rev%d" % L].dropna().to_numpy()
        if len(x) < 50:
            continue
        hi5 = np.percentile(x, 95)
        print("    %4d분: 전체평균 %7.1f | 상위5%% 제외 %7.1f | 상하위5%% 제외 %7.1f"
              % (5 * L, x.mean(), x[x <= hi5].mean(),
                 x[(x <= hi5) & (x >= np.percentile(x, 5))].mean()))


def realized(ev: pd.DataFrame) -> None:
    """2. 실현 되돌림 — 청산 이벤트 vs 위약(OI 유지)."""
    print("\n" + "=" * 78)
    print("2. 실현 되돌림 — 청산 이벤트 vs **위약**(같은 크기 움직임, OI 유지)")
    print("=" * 78)
    print("  양수 = 이벤트 방향의 반대로 되돌아왔다(= 지정매수가 먹는 방향).")
    print("  기준가는 open[i+1] — 판정 시점 이후 첫 체결 가능 가격. 바닥 기준 아님.\n")
    liq = ev[ev["is_liq"]]
    pla = ev[~ev["is_liq"]]
    print("  청산 이벤트 %d건 | 위약 %d건" % (len(liq), len(pla)))
    print("\n  %6s | %10s %7s | %10s %7s | %10s %7s"
          % ("지연(분)", "청산 bp", "t", "위약 bp", "t", "차이 bp", "t"))
    for L in LAGS:
        c = "rev%d" % L
        a1, _, t1, _ = cmean(liq[c].to_numpy(), liq["day"].to_numpy())
        a2, _, t2, _ = cmean(pla[c].to_numpy(), pla["day"].to_numpy())
        m = ev[c].notna().to_numpy()
        X = np.column_stack([np.ones(int(m.sum())),
                             ev.loc[m, "is_liq"].to_numpy().astype(float)])
        b, se, _ = ols_cluster(X, ev.loc[m, c].to_numpy(), ev.loc[m, "day"].to_numpy())
        td = b[1] / se[1] if se[1] > 0 else np.nan
        print("  %6d | %10.2f %7.1f | %10.2f %7.1f | %10.2f %7.1f"
              % (5 * L, a1, t1, a2, t2, b[1], td))
    print("  차이 열이 유의해야 '청산 때문' 이라고 말할 수 있다.")
    print("  유의하지 않으면 되돌림은 큰 움직임 일반의 성질이지 청산 특유가 아니다.")


# ------------------------------------------------------ 3. 예측 대 실현
def predict_vs_real(ev: pd.DataFrame, share: float, lag: int = 1) -> None:
    print("\n" + "=" * 78)
    print("3. 사전등록 예측 대 실현 — sqrt(Q) 법칙이 캐스케이드 규모로 가는가")
    print("=" * 78)
    liq = ev[ev["is_liq"] & ev["q_oi"].notna() & (ev["q_oi"] > 0)].copy()
    sh = share if np.isfinite(share) else 1.0
    liq["q"] = liq["q_oi"] * sh
    liq["x"] = liq["sig_d"] * np.sqrt(liq["q"] / liq["adv"]) * 1e4
    y = liq["rev%d" % lag].to_numpy()
    ok = np.isfinite(y) & np.isfinite(liq["x"].to_numpy())
    liq, y = liq[ok], y[ok]
    print("  이벤트 %d건 | 청산비중 보정 %.3f | 되돌림 지연 %d분"
          % (len(liq), sh, 5 * lag))
    print("  Q/ADV: 중앙 %.3g  p90 %.3g  최대 %.3g"
          % (np.median(liq["q"] / liq["adv"]), np.quantile(liq["q"] / liq["adv"], .9),
             (liq["q"] / liq["adv"]).max()))

    print("\n  [3a] 사전등록 예측 vs 실현 (전체 평균)")
    a, _, t, _ = cmean(y, liq["day"].to_numpy())
    mx = float(np.mean(liq["x"]))
    print("    실현 되돌림              %8.2f bp  (t=%.1f)" % (a, t))
    print("    예측 Y=%.2f (R-1 bybit)  %8.2f bp   ->  실현/예측 = **%.1f배**"
          % (Y_REF, Y_REF * mx, a / (Y_REF * mx) if mx else np.nan))

    print("\n  [3b] 회귀 실현 = a + b * 예측(Y=1 기준).  b 가 사전등록 구간 안인가")
    X = np.column_stack([np.ones(len(liq)), liq["x"].to_numpy()])
    bb, se, _ = ols_cluster(X, y, liq["day"].to_numpy())
    print("    절편 %8.2f (t=%.1f) | 기울기 b = %.3f +- %.3f (t=%.1f)"
          % (bb[0], bb[0] / se[0] if se[0] > 0 else np.nan,
             bb[1], se[1], bb[1] / se[1] if se[1] > 0 else np.nan))
    lo, hi = bb[1] - 1.96 * se[1], bb[1] + 1.96 * se[1]
    print("    95%% CI [%.3f, %.3f]  vs 사전등록 [%.2f, %.2f]  ->  %s"
          % (lo, hi, Y_LO, Y_HI,
             "겹침 = 법칙 확장 지지" if (hi >= Y_LO and lo <= Y_HI) else "불일치"))

    print("\n  [3c] 규모 오분위 — 법칙이 어디서 깨지는가")
    liq = liq.copy()
    liq["bin"] = pd.qcut(liq["q"] / liq["adv"], 5, labels=False, duplicates="drop")
    print("    %4s %7s %12s %11s %11s %9s"
          % ("분위", "n", "Q/ADV 중앙", "실현 bp", "예측(0.5) bp", "실현/예측"))
    for q in sorted(liq["bin"].dropna().unique()):
        g = liq[liq["bin"] == q]
        yy = g["rev%d" % lag].to_numpy()
        a, _, t, _ = cmean(yy, g["day"].to_numpy())
        pr = float(np.mean(0.5 * g["x"]))
        print("    %4d %7d %12.3g %6.1f(%4.1f) %11.1f %9.2f"
              % (q, len(g), np.median(g["q"] / g["adv"]), a, t, pr,
                 a / pr if pr else np.nan))
    print("    실현/예측이 큰 분위에서 1보다 뚜렷이 작으면 **포화**다")
    print("    (물량이 커져도 되돌림이 그만큼 커지지 않는다 = 전략 전제 약화).")

    print("\n  [3d] 기울기의 정체 — sqrt(Q) 인가 sigma 인가")
    print("    x = sigma_d * sqrt(Q/ADV) 는 두 요인의 곱이다. 이벤트가 8시그마로")
    print("    선별돼 sigma 가 함께 높으므로, 기울기가 어느 쪽에서 오는지 갈라야 한다.")
    print("    되돌림을 sigma 로 나누면 제곱근 법칙은 y = Y*sqrt(Q/ADV) 가 된다.")
    ynorm = y / (liq["sig_d"].to_numpy() * 1e4)
    rq = np.sqrt((liq["q"] / liq["adv"]).to_numpy())
    dd = liq["day"].to_numpy()
    for lab, X in (("절편 없음 (법칙 그대로)", rq[:, None]),
                   ("절편 포함", np.column_stack([np.ones(len(rq)), rq]))):
        bb, se, _ = ols_cluster(X, ynorm, dd)
        if X.shape[1] == 1:
            print("    %-22s Y = %8.3f +- %.3f (t=%.1f)"
                  % (lab, bb[0], se[0], bb[0] / se[0] if se[0] > 0 else np.nan))
        else:
            print("    %-22s 절편 %7.4f (t=%.1f) | Y = %8.3f +- %.3f (t=%.1f)"
                  % (lab, bb[0], bb[0] / se[0] if se[0] > 0 else np.nan,
                     bb[1], se[1], bb[1] / se[1] if se[1] > 0 else np.nan))
    b2, s2, _ = ols_cluster(np.column_stack([np.ones(len(rq)),
                                             liq["sig_d"].to_numpy()]), ynorm, dd)
    print("    대조: sigma_d 만으로 회귀 -> 계수 %.3f (t=%.1f)"
          % (b2[1], b2[1] / s2[1] if s2[1] > 0 else np.nan))
    print("    절편이 유의하고 Y 가 죽으면 되돌림은 **Q 와 무관한 고정 크기**다.")

    print("\n  [3e] 초과분이 자기여기(캐스케이드 증폭)로 설명되는가")
    print("    제곱근 법칙은 **단일 메타주문** 법칙이라 자기여기가 없다. 캐스케이드는")
    print("    첫 청산이 다음 청산을 부르므로 사건 수가 1/(1-n) 배로 늘어난다.")
    bb, se, _ = ols_cluster(rq[:, None], ynorm, dd)
    lo, hi = bb[0] - 1.96 * se[0], bb[0] + 1.96 * se[0]
    amp_lin = 1.0 / (1.0 - N_BRANCH)          # 임팩트가 선형 가산될 때
    amp_sqrt = np.sqrt(amp_lin)               # 물량만 증폭되고 sqrt 가 걸릴 때
    print("    실측 Y = %.2f, 95%% CI [%.2f, %.2f]   (n=%d)" % (bb[0], lo, hi, len(rq)))
    for lab, pred in (("증폭 없음", Y_REF),
                      ("물량 증폭 sqrt(1/(1-n)) = %.2f" % amp_sqrt, Y_REF * amp_sqrt),
                      ("임팩트 선형가산 1/(1-n) = %.2f" % amp_lin, Y_REF * amp_lin)):
        mark = "**CI 안**" if lo <= pred <= hi else "CI 밖"
        print("    %-34s -> Y_pred = %6.2f   %s" % (lab, pred, mark))
    print("    n=%.3f 은 멱함수 Hawkes 값인데 **잔차 진단 불합격**이라 참고치다."
          % N_BRANCH)


# ------------------------------------------------------ 4. GPD 디클러스터링
def gpd_decluster() -> None:
    """theta ~ 1-n 로 유효 표본을 줄여 GPD 신뢰구간을 다시 낸다."""
    print("\n" + "=" * 78)
    print("4. GPD 디클러스터링 — theta ~ 1-n = 0.199 로 CI 재계산")
    print("=" * 78)
    if not os.path.exists(EVCACHE):
        print("  이벤트 캐시 없음 — 건너뜀")
        return
    d = pd.read_parquet(EVCACHE)
    x = d["X"].to_numpy(dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    theta = 0.199
    print("  X (캐스케이드 깊이) %d건. 극단지표 theta=%.3f -> 유효 독립 표본 %.0f건\n"
          % (len(x), theta, len(x) * theta))
    print("  %10s %7s %9s %9s %11s %11s"
          % ("임계", "초과n", "xi", "sigma", "SE(순진)", "SE(디클러스터)"))
    for p in (0.80, 0.90, 0.95, 0.975):
        u = float(np.quantile(x, p))
        ex = x[x > u] - u
        k = len(ex)
        if k < 25:
            continue
        # 적률법(POT). MLE 대신 쓰는 이유: 표본이 작고 xi 가 0 근처라 안정적이다.
        mu, s2 = float(ex.mean()), float(ex.var(ddof=1))
        if s2 <= 0 or mu <= 0:
            continue
        xi = 0.5 * (1.0 - mu * mu / s2)
        sg = 0.5 * mu * (mu * mu / s2 + 1.0)
        # xi 의 점근 분산 (Hosking-Wallis): (1-xi)^2(1-2xi)... 근사로 (1+xi)^2/k
        se_naive = np.sqrt((1.0 + xi) ** 2 / k)
        se_dec = se_naive / np.sqrt(theta)
        print("  %10.4f %7d %9.4f %9.4f %11.4f %11.4f"
              % (u, k, xi, sg, se_naive, se_dec))
    print("\n  SE 가 sqrt(1/theta) = %.2f 배 넓어진다." % (1 / np.sqrt(theta)))
    print("  xi 점추정은 그대로다 — 바뀌는 것은 **불확실성 폭**뿐이다.")


def main() -> int:
    ap = argparse.ArgumentParser(description="R-2 scale extension of the sqrt(Q) law")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--lag", type=int, default=1, help="되돌림 지연(5분바 단위)")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("R-2 — sqrt(Q) 되돌림 법칙의 규모 확장 검정")
    print("사전등록 예측: Y_full = %.2f ~ %.2f (R-1 에서 시장점유 환산)\n" % (Y_LO, Y_HI))

    share = q_bridge()
    ev = build_events(syms)
    if len(ev) == 0:
        print("이벤트가 없다")
        return 1
    t0 = pd.to_datetime(ev["ts"].min(), unit="ms")
    t1 = pd.to_datetime(ev["ts"].max(), unit="ms")
    print("\n**사용 데이터 기간: %s ~ %s / %d종 / 5분봉**"
          % (str(t0)[:10], str(t1)[:10], ev["symbol"].nunique()))
    print("이벤트 %d건 (청산 %d / 위약 %d)"
          % (len(ev), int(ev["is_liq"].sum()), int((~ev["is_liq"]).sum())))

    realized(ev)
    predict_vs_real(ev, share, a.lag)
    path_analysis(ev)
    gpd_decluster()
    return 0


if __name__ == "__main__":
    sys.exit(main())
