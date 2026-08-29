# -*- coding: utf-8 -*-
"""전략 백테스트 — 부품 검정이 아니라 **하나의 일관된 메커니즘**을 끝까지 돌린다.

왜 이것인가
  지금까지의 검정은 전부 **부품**이었다(V/D 탄성, 증폭 배율, 캘리브레이션, 응답함수).
  부품이 각각 죽어도 조립된 전략은 돌 수 있고 반대도 마찬가지다. 그리고 나는
  50.4bp 를 '이벤트당 평균 ± t' 로만 보고했지 **자산곡선·최대낙폭·연도별·민감도**를
  한 번도 내지 않았다. 그것은 전략 평가가 아니다.

메커니즘 (고정. 여기서 탐색하지 않는다)
  방아쇠 : 5분봉 |z| >= K  **그리고**  같은 5분에 OI 가 DOI 이하로 감소
           z = 5분수익 / 과거 288봉 표준편차 (현재봉 제외)
           OI 변화는 다음 스냅샷을 써야 하므로 **바 i 의 확인은 시각 i+1 에 도착**한다
  방향   : 움직임의 **반대** (하락+OI감소 = 롱청산 -> 매수)
  진입   : 바 i+1 의 **시가**. 확인이 도착하는 바로 그 시각이다 -> 룩어헤드 없음
  청산   : HOLD 분 뒤 종가
  비용   : 왕복 COST bp
  군집제거: 같은 심볼·같은 방향 MIN_GAP 봉 안의 중복은 첫 건만

  K=8, DOI=-0.02, MIN_GAP=12 는 **이 세션 이전에 정해진 값**이다(캐시 파일명에 박혀
  있다). 여기서 고르지 않는다. 대신 민감도 표로 칼날 위가 아님을 보인다.

산출
  1. 자산곡선(누적 bp) + 최대낙폭 + 샤프
  2. 연도별 / 심볼별 분해
  3. 파라미터 민감도 (K x DOI x HOLD) — **과최적화 점검**
  4. 전반부/후반부 분할 — 감쇠하는가
  5. 동시보유 최대 개수 — 필요 자본

실행:
    python analysis/backtest.py
    python analysis/backtest.py --cost 10 --hold 15
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
from analysis.event_study_h2 import load, find_events           # noqa: E402
from analysis.response_liq import cmean                         # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
VOL_WIN = 288                 # 5분봉 하루


def run(symbols, k, doi_thr, gap, hold, cost) -> pd.DataFrame:
    """메커니즘 1회 실행. 이벤트별 (시각, 심볼, 방향, 순수익bp, MAE)."""
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, k, doi_thr, gap)
        if len(ev) == 0:
            continue
        p1 = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p1):
            continue
        m = pd.read_parquet(p1, columns=["open_time", "open", "high", "low", "close"])
        m = m.sort_values("open_time").reset_index(drop=True)
        ot1 = m["open_time"].to_numpy()
        O, H, L, Cl = (m[c].to_numpy(dtype=np.float64)
                       for c in ("open", "high", "low", "close"))
        n1 = len(ot1)
        ot5 = df["open_time"].to_numpy()
        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5):
                continue
            t_ent = int(ot5[i + 1])                 # 확인 도착 = 진입 가능 시각
            j = int(np.searchsorted(ot1, t_ent))
            if j >= n1 or ot1[j] != t_ent or j + hold >= n1:
                continue
            # 1분봉 연속성 확인 — 결손 구간이면 버린다
            if ot1[j + hold] - ot1[j] != hold * 60_000:
                continue
            p_in = O[j]
            p_out = Cl[j + hold]
            if not (np.isfinite(p_in) and p_in > 0 and np.isfinite(p_out)):
                continue
            ret = (p_out / p_in - 1.0) * sd * 1e4 - cost
            mae = ((L[j:j + hold + 1].min() / p_in - 1.0) if sd == 1
                   else -(H[j:j + hold + 1].max() / p_in - 1.0)) * 1e4
            # sigma 는 방아쇠 봉의 과거 288봉 표준편차(5분 단위, 비율)
            sg = float(df["sigma"].to_numpy()[i])
            out.append({"t": t_ent, "symbol": s, "side": sd, "ret": ret, "mae": mae,
                        "sig5": sg, "z": float(df["z"].to_numpy()[i]),
                        "day": t_ent // 86_400_000,
                        "year": pd.Timestamp(t_ent, unit="ms").year})
    return pd.DataFrame(out).sort_values("t").reset_index(drop=True)


def stats(d: pd.DataFrame, hold: int) -> dict:
    if len(d) == 0:
        return {}
    r = d["ret"].to_numpy()
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    m, se, t, _ = cmean(r, d["day"].to_numpy())
    yrs = (d["t"].iloc[-1] - d["t"].iloc[0]) / (365.25 * 86_400_000)
    # 이벤트 단위 샤프를 연율화: 연 이벤트수로 스케일
    per_yr = len(d) / yrs if yrs > 0 else np.nan
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(per_yr)) if r.std(ddof=1) > 0 else np.nan
    return {"n": len(d), "yrs": yrs, "per_yr": per_yr, "mean": m, "t": t,
            "median": float(np.median(r)), "sd": float(r.std(ddof=1)),
            "win": float((r > 0).mean()), "total": float(eq[-1]),
            "maxdd": float(dd.min()), "sharpe": sharpe,
            "mae_med": float(np.median(d["mae"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description="single coherent mechanism backtest")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=15, help="보유(분)")
    ap.add_argument("--cost", type=float, default=10.0,
                    help="왕복 bp. 시장가 진입/청산이면 테이커 5bp x 2 = 10")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 78)
    print("전략 백테스트 — 단일 메커니즘")
    print("=" * 78)
    print("방아쇠: 5분 |z|>=%.0f 그리고 OI 변화<=%.0f%% | 군집제거 %d봉"
          % (a.k, 100 * a.doi, a.gap))
    print("진입: 확인 도착 시각의 시가(시장가) | 청산: %d분 뒤 | 비용: 왕복 %.0fbp"
          % (a.hold, a.cost))
    print("*** 확인은 바 i+1 시각에 도착하고 진입도 그 시각이다 -> 룩어헤드 없음\n")

    d = run(syms, a.k, a.doi, a.gap, a.hold, a.cost)
    if len(d) < 50:
        print("이벤트 부족 (%d)" % len(d))
        return 1
    S = stats(d, a.hold)
    print("**사용 데이터 기간: %s ~ %s / %d종 / %.1f년**"
          % (str(pd.Timestamp(d.t.iloc[0], unit="ms"))[:10],
             str(pd.Timestamp(d.t.iloc[-1], unit="ms"))[:10], d.symbol.nunique(),
             S["yrs"]))
    print("\n--- 1. 전체 성적 ---")
    print("  이벤트 %d건 (연 %.0f건) | 평균 **%.1f bp** (t=%.1f) | 중앙 %.1f"
          % (S["n"], S["per_yr"], S["mean"], S["t"], S["median"]))
    print("  표준편차 %.0f bp | 승률 %.1f%% | MAE 중앙 %.0f bp"
          % (S["sd"], 100 * S["win"], S["mae_med"]))
    print("  누적 **%.0f bp** (= %.1f 배수, 이벤트당 전액투입 단리 가정)"
          % (S["total"], S["total"] / 1e4))
    print("  **최대낙폭 %.0f bp** (누적 대비 %.1f%%) | 연율 샤프 **%.2f**"
          % (S["maxdd"], 100 * abs(S["maxdd"]) / max(S["total"], 1), S["sharpe"]))

    print("\n--- 2. 연도별 — 감쇠하는가 ---")
    print("  %6s %7s %10s %8s %9s %9s" % ("연도", "n", "평균bp", "승률%", "합계bp", "누적bp"))
    cum = 0.0
    for y, g in d.groupby("year"):
        cum += g["ret"].sum()
        print("  %6d %7d %10.1f %8.1f %9.0f %9.0f"
              % (y, len(g), g["ret"].mean(), 100 * (g["ret"] > 0).mean(),
                 g["ret"].sum(), cum))

    print("\n--- 3. 전반부 / 후반부 ---")
    h = len(d) // 2
    for lab, g in (("전반부", d.iloc[:h]), ("후반부", d.iloc[h:])):
        m, se, t, _ = cmean(g["ret"].to_numpy(), g["day"].to_numpy())
        print("  %-6s %s ~ %s | n=%d | 평균 %.1f bp (t=%.1f) | 승률 %.1f%%"
              % (lab, str(pd.Timestamp(g.t.iloc[0], unit="ms"))[:10],
                 str(pd.Timestamp(g.t.iloc[-1], unit="ms"))[:10], len(g), m, t,
                 100 * (g["ret"] > 0).mean()))

    print("\n--- 4. 심볼별 (상위/하위 5) ---")
    bs = d.groupby("symbol")["ret"].agg(["size", "mean", "sum"]).sort_values("mean")
    print("  %-10s %6s %10s %9s" % ("심볼", "n", "평균bp", "합계bp"))
    for s_, row in pd.concat([bs.head(5), bs.tail(5)]).iterrows():
        print("  %-10s %6d %10.1f %9.0f" % (s_, row["size"], row["mean"], row["sum"]))
    print("  양수 심볼 %d / %d" % (int((bs["mean"] > 0).sum()), len(bs)))

    print("\n--- 4b. **변동성 정규화** — bp 차이가 sigma 때문인가 ---")
    print("  z 는 각 심볼 자신의 sigma 로 나눈 값이라 8시그마의 '퍼센트 크기' 가 다르다.")
    print("  b1(log sigma)=1.017 이면 되돌림은 sigma 에 정비례한다 -> 나누면 같아야 한다.")
    d = d.copy()
    d["sig_bp"] = d["sig5"] * 1e4                       # 5분 sigma (bp)
    d["ret_n"] = d["ret"] / d["sig_bp"]                 # sigma 단위 수익
    print("  %-10s %6s %10s %10s %12s %8s"
          % ("심볼", "n", "sigma bp", "평균bp", "**평균/sigma**", "t"))
    rows = []
    for s_, g in d.groupby("symbol"):
        if len(g) < 15:
            continue
        # cmean 은 n>=30 하한이 있어 소형 심볼이 NaN 이 된다. 여기선 심볼별 요약이라
        # 평균 자체가 관심 대상이므로 단순 평균 + iid t 를 쓴다(SE 는 참고용).
        x = g["ret_n"].to_numpy()
        x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        se_ = x.std(ddof=1) / np.sqrt(len(x))
        rows.append((s_, len(x), g["sig_bp"].median(), g["ret"].mean(),
                     float(x.mean()), float(x.mean() / se_) if se_ > 0 else np.nan))
    rows.sort(key=lambda x: x[4])
    for r_ in rows[:5] + rows[-5:]:
        print("  %-10s %6d %10.1f %10.1f %12.3f %8.1f" % r_)
    mn, se, tt, _ = cmean(d["ret_n"].to_numpy(), d["day"].to_numpy())
    v = np.array([r_[4] for r_ in rows])
    print("  전체 평균/sigma = **%.3f** (t=%.1f) | 심볼간 표준편차 %.3f | 양수 %d/%d"
          % (mn, tt, v.std(ddof=1), int((v > 0).sum()), len(v)))
    print("  심볼간 산포가 작으면 **bp 차이는 전부 sigma 때문**이고, 변동성 타게팅으로")
    print("  균등화된다 -> BTC 도 쓸 수 있고 용량 문제가 완화된다.")

    print("\n  [4c] 변동성 타게팅 시뮬레이션 — 각 이벤트를 1/sigma 로 가중")
    w = 1.0 / d["sig_bp"].to_numpy()
    w = w / np.median(w)                                # 중앙 노출 = 1배
    wr = d["ret"].to_numpy() * w
    m2, se2, t2, _ = cmean(wr, d["day"].to_numpy())
    print("      가중 평균 %.1f bp (t=%.1f) | 표준편차 %.0f | 최대가중 %.1f배"
          % (m2, t2, wr.std(ddof=1), w.max()))
    eqw = np.cumsum(wr)
    ddw = eqw - np.maximum.accumulate(eqw)
    print("      누적 %.0f bp | 최대낙폭 %.0f bp (%.1f%%) | 샤프(일클러스터) %.2f"
          % (eqw[-1], ddw.min(), 100 * abs(ddw.min()) / max(eqw[-1], 1),
             t2 / np.sqrt(S["yrs"])))
    bw = d.assign(wr=wr).groupby("symbol")["wr"].agg(["size", "mean", "sum"])
    bw = bw.sort_values("sum", ascending=False)
    top5 = bw.head(5)["sum"].sum()
    print("      상위5 심볼 이익 비중 %.1f%% (비가중 시 42%%)"
          % (100 * top5 / max(eqw[-1], 1)))

    print("\n--- 5. 파라미터 민감도 — 칼날 위인가 ---")
    print("  기준값(K=8, DOI=-2%%, HOLD=%d)은 이 세션 **이전**에 정해진 값이다." % a.hold)
    print("  %5s %7s | %8s %8s %8s %8s" % ("K", "DOI%", "n", "평균bp", "t", "샤프"))
    for kk in (4.0, 6.0, 8.0, 10.0):
        for dd in (-0.01, -0.02, -0.03):
            g = run(syms, kk, dd, a.gap, a.hold, a.cost)
            if len(g) < 50:
                continue
            s2 = stats(g, a.hold)
            mark = "  <- 기준" if (kk == a.k and abs(dd - a.doi) < 1e-9) else ""
            print("  %5.0f %7.0f | %8d %8.1f %8.1f %8.2f%s"
                  % (kk, 100 * dd, s2["n"], s2["mean"], s2["t"], s2["sharpe"], mark))

    print("\n  보유시간 민감도 (K=%.0f, DOI=%.0f%%)" % (a.k, 100 * a.doi))
    print("  %7s | %8s %8s %8s %8s" % ("보유분", "평균bp", "t", "샤프", "MAE중앙"))
    for hh in (5, 15, 30, 60, 120):
        g = run(syms, a.k, a.doi, a.gap, hh, a.cost)
        if len(g) < 50:
            continue
        s2 = stats(g, hh)
        print("  %7d | %8.1f %8.1f %8.2f %8.0f"
              % (hh, s2["mean"], s2["t"], s2["sharpe"], s2["mae_med"]))

    print("\n--- 6. 동시보유 (필요 자본) ---")
    ends = d["t"].to_numpy() + a.hold * 60_000
    starts = d["t"].to_numpy()
    conc = np.array([int(((starts <= s0) & (ends > s0)).sum()) for s0 in starts])
    print("  동시보유 중앙 %d | p90 %d | 최대 **%d**"
          % (int(np.median(conc)), int(np.quantile(conc, .9)), int(conc.max())))
    print("  -> 최대치 기준으로 자본을 나누면 이벤트당 노출은 1/%d 이 된다." % conc.max())
    print("\n  *** 비용 %.0fbp 는 시장가 왕복(테이커 5bp x2) 가정이다." % a.cost)
    print("      메이커 진입이 가능하면 7bp, 그만큼 좋아진다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
