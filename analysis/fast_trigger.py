# -*- coding: utf-8 -*-
"""1분봉 방아쇠 + 중복제거 창 스윕 — 확인 지연을 10분에서 1분으로 줄인다.

문제 제기 (사용자, 2026-08-05)
  1. "지금 방아쇠가 왜 또 5분봉이에요? 1분봉이나 ws로 해야죠."
     -> 5분봉을 쓴 이유는 **OI 스냅샷이 5분 간격**이기 때문이다. 가격은 1분봉으로
        갈 수 있고, dOI 는 직전에 이미 알려진 5분 값을 쓰면 된다.
        지금 규칙은 5분봉 i 가 닫히고 다음 5분 스냅샷까지 기다려 확인하므로
        **확인이 최대 10분 늦는다**. pnl_source.py 에서 최저점이 진입 순간(0분)인
        경우가 44.2% 였던 것이 이 지연 때문일 수 있다.
  2. "같은 심볼·방향 60분 내 중복 가능해야죠. 회전율 높이게."
     -> gap=12(60분) 은 임의값이다. 한 캐스케이드 안에서 여러 번 진입하는 것이
        원래 설계 의도였다. 스윕한다.
  3. 매도는 우위가 없다(t=-0.3, n=245) -> **매수만** 간다.

방아쇠 (하이브리드)
  z1(i) = 1분봉 i 의 수익 / 과거 1440분 표준편차(현재봉 제외)
  dOI   = **t_{i+1} 시점에 이미 알려진** 가장 최근 5분 OI 변화
          (5분 경계 T 에서의 dOI 는 OI(T)/OI(T-5m)-1 이고 시각 T 에 알려진다.
           그래서 t 에서 쓸 수 있는 것은 floor_5m(t) 의 값이다 — 룩어헤드 없음)
  조건  : z1 <= -K  그리고  dOI <= thr        (하락 + 청산 동반 -> 매수)
  진입  : 바 i+1 의 시가 (확인이 도착하는 바로 그 시각)

청산 (유동적. 고정 보유가 아니다)
  익절 p_in*(1+tp*sigma5) 지정가 / 손절 p_in*(1-sl*sigma5) 시장가(+슬리피지)
  / tmax 분 시간정지.  셋 중 먼저.
  ** 봉내 동시 도달은 손절 우선 (보수적). **

  sigma5 는 5분봉 sigma 를 그대로 쓴다 — 앞선 검정(asymmetry.py)과 척도를 맞춰
  익절 8 / 손절 2 배수를 그대로 비교하기 위해서다.

실행:
    python analysis/fast_trigger.py
    python analysis/fast_trigger.py --ks 4 6 8 --gaps 1 5 15
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
from analysis.event_study_h2 import load                               # noqa: E402
from analysis.response_liq import cmean                                # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
VOL1 = 1440                    # 1분봉 하루
MINS_YR = 365.25 * 24 * 60


def prep(symbols):
    """1분봉 + (1분 격자에 정렬된) 알려진 dOI / sigma5."""
    out = {}
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        p1 = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p1):
            continue
        m = pd.read_parquet(p1, columns=["open_time", "open", "high", "low", "close"])
        m = m.sort_values("open_time").reset_index(drop=True)
        ot1 = m["open_time"].to_numpy()
        O = m["open"].to_numpy(dtype=np.float64)
        H = m["high"].to_numpy(dtype=np.float64)
        L = m["low"].to_numpy(dtype=np.float64)
        Cl = m["close"].to_numpy(dtype=np.float64)
        r1 = Cl / np.maximum(O, 1e-12) - 1.0
        sd1 = pd.Series(r1).rolling(VOL1, min_periods=VOL1 // 4).std().shift(1).to_numpy()
        z1 = np.where(sd1 > 0, r1 / sd1, np.nan)

        # 5분 격자의 dOI/sigma 를 1분 격자에 **알려진 시점 기준**으로 붙인다.
        ot5 = df["open_time"].to_numpy()
        doi5 = df["doi"].to_numpy()
        sg5 = df["sigma"].to_numpy()
        # 5분봉 i 의 dOI 는 다음 스냅샷이 필요하므로 ot5[i]+5분 에 알려진다.
        known_t = ot5 + 300_000
        ki = np.searchsorted(known_t, ot1, side="right") - 1
        ok = ki >= 0
        doi_a = np.full(len(ot1), np.nan)
        sig_a = np.full(len(ot1), np.nan)
        doi_a[ok] = doi5[ki[ok]]
        sig_a[ok] = sg5[ki[ok]]
        out[s] = (ot1, O, H, L, Cl, z1, doi_a, sig_a)
    return out


def events(ot1, z1, doi, sig, k, thr, gap):
    """매수 방아쇠(하락+청산)만. gap 분 안의 중복은 첫 건만."""
    m = np.isfinite(z1) & np.isfinite(doi) & np.isfinite(sig) & (sig > 0)
    m &= (z1 <= -k) & (doi <= thr)
    # 1분 격자 연속성: 직전 봉이 이어져 있어야 z1 이 의미가 있다
    m[1:] &= (np.diff(ot1) == 60_000)
    m[0] = False
    idx = np.flatnonzero(m)
    if not len(idx) or gap <= 1:
        return idx
    keep, last = [], -10**9
    for i in idx:
        if ot1[i] - last < gap * 60_000:
            continue
        last = ot1[i]
        keep.append(i)
    return np.array(keep, dtype=int)


def run(prepped, k, thr, gap, tmax, tp, sl, cost, slip) -> pd.DataFrame:
    rows = []
    for s, (ot1, O, H, L, Cl, z1, doi, sig) in prepped.items():
        n1 = len(ot1)
        for i in events(ot1, z1, doi, sig, k, thr, gap):
            j = i + 1                                   # 확인 도착 = 진입 가능 시각
            if j + tmax >= n1 or ot1[j + tmax] - ot1[j] != tmax * 60_000:
                continue
            p_in, sg = O[j], sig[i]
            if not (np.isfinite(p_in) and p_in > 0 and np.isfinite(sg) and sg > 0):
                continue
            if not (np.isfinite(O[j:j + tmax + 1]).all()
                    and np.isfinite(H[j:j + tmax + 1]).all()
                    and np.isfinite(L[j:j + tmax + 1]).all()
                    and np.isfinite(Cl[j:j + tmax + 1]).all()):
                continue
            tpx, slx = p_in * (1.0 + tp * sg), p_in * (1.0 - sl * sg)
            p_out, why, hold = Cl[j + tmax], "to", tmax
            for u in range(j + 1, j + tmax + 1):        # 진입 봉은 건너뛴다
                if L[u] <= slx:                          # 봉내 동시 -> 손절 우선
                    p_out, why, hold = slx * (1.0 - slip * 1e-4), "sl", u - j
                    break
                if H[u] >= tpx:
                    p_out, why, hold = tpx, "tp", u - j
                    break
            rows.append({"symbol": s, "t": int(ot1[j]),
                         "t_exit": int(ot1[j]) + hold * 60_000,
                         "ret": (p_out / p_in - 1.0) * 1e4 - cost,
                         "why": why, "hold": hold, "z": z1[i],
                         "day": int(ot1[j]) // 86_400_000})
    return pd.DataFrame(rows)


def concurrency(d: pd.DataFrame) -> dict:
    """실제 동시보유 포지션 수. gap 을 풀면 한 캐스케이드에 진입이 겹겹이 쌓인다.

    각 진입을 1단위로 잡고 계산한 수익은, 최대 동시보유가 M 이면 사실상
    **M 배 레버리지**를 쓴 것이다. 같은 자본에서 비교하려면 M 으로 나눠야 한다.
    """
    if len(d) == 0:
        return {}
    ev = np.concatenate([d["t"].to_numpy(), d["t_exit"].to_numpy()])
    delta = np.concatenate([np.ones(len(d)), -np.ones(len(d))])
    o = np.lexsort((delta, ev))            # 같은 시각이면 청산(-1) 을 먼저
    cur = np.cumsum(delta[o])
    out = {"max_all": int(cur.max()),
           "p99_all": float(np.percentile(cur, 99)),
           "mean_all": float(cur.mean())}
    mx = 0
    for s, g in d.groupby("symbol"):
        e = np.concatenate([g["t"].to_numpy(), g["t_exit"].to_numpy()])
        dl = np.concatenate([np.ones(len(g)), -np.ones(len(g))])
        oo = np.lexsort((dl, e))
        mx = max(mx, int(np.cumsum(dl[oo]).max()))
    out["max_sym"] = mx
    return out


def summ(d):
    if len(d) < 30:
        return None
    r = d["ret"].to_numpy()
    m, _, t, _ = cmean(r, d["day"].to_numpy())
    yrs = (d["t"].max() - d["t"].min()) / (365.25 * 86_400_000)
    per_yr = len(d) / yrs if yrs > 0 else np.nan
    w, l = r[r > 0], r[r < 0]
    eq = np.cumsum(r)
    hold = float(d["hold"].mean())
    return {"n": len(d), "per_yr": per_yr, "bp": m, "t": t,
            "win": float((r > 0).mean()),
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "yr_bp": m * per_yr, "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "maxdd": float((eq - np.maximum.accumulate(eq)).min()),
            "hold": float(d["hold"].median()),
            "conc": len(d) * hold / (yrs * MINS_YR) if yrs > 0 else np.nan}


def head():
    print("  %-22s | %6s %7s | %8s %5s | %7s %6s | %10s %6s %8s | %5s %8s"
          % ("설정", "n", "연간", "시도당bp", "t", "승률", "손익비",
             "연간총bp", "샤프", "최대낙폭", "보유", "동시보유"))


def line(lab, s):
    if s is None:
        print("  %-22s | (표본부족)" % lab)
        return
    print("  %-22s | %6d %7.0f | %8.1f %5.1f | %6.1f%% %6.2f | %10.0f %6.2f %8.0f | %5.0f %8.4f"
          % (lab, s["n"], s["per_yr"], s["bp"], s["t"], 100 * s["win"], s["pl"],
             s["yr_bp"], s["sharpe"], s["maxdd"], s["hold"], s["conc"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="1-minute trigger with gap sweep")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--doi", type=float, default=-0.005)
    ap.add_argument("--tmax", type=int, default=15)
    ap.add_argument("--tp", type=float, default=8.0)
    ap.add_argument("--sl", type=float, default=2.0)
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--slip", type=float, default=5.0)
    ap.add_argument("--ks", type=float, nargs="+", default=[3, 4, 5, 6, 8, 10])
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 3, 5, 15, 30, 60])
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 122)
    print("1분봉 방아쇠 (매수만) — 확인 지연 10분 -> 1분. 중복제거 창 스윕")
    print("=" * 122)
    print("청산: 익절 %.0fs / 손절 %.0fs / %d분 상한 중 **먼저**. 왕복 %.0fbp + 손절 슬리피지 %.0fbp"
          % (a.tp, a.sl, a.tmax, a.cost, a.slip))
    P = prep(syms)
    print("심볼 %d종 로드" % len(P))

    print("\n" + "-" * 122)
    print("1. K x 중복제거 창 (dOI <= %.3f)" % a.doi)
    print("-" * 122)
    head()
    best = None
    for k in a.ks:
        for g in a.gaps:
            d = run(P, k, a.doi, g, a.tmax, a.tp, a.sl, a.cost, a.slip)
            s = summ(d)
            if s is None:
                continue
            line("K=%.0f gap=%dm" % (k, g), s)
            if best is None or s["yr_bp"] > best[0]["yr_bp"]:
                best = (s, k, g)
        print()
    if best:
        s, k, g = best
        print("  ** 연간총bp 최대: K=%.0f gap=%d분 -> 연 %.0f건 x %.1fbp = **%.0f bp/년** (샤프 %.2f)"
              % (k, g, s["per_yr"], s["bp"], s["yr_bp"], s["sharpe"]))
        print("\n  전/후반 분할:")
        head()
        d = run(P, k, a.doi, g, a.tmax, a.tp, a.sl, a.cost,
                a.slip).sort_values("t").reset_index(drop=True)
        h = len(d) // 2
        line("  전반부", summ(d.iloc[:h]))
        line("  후반부", summ(d.iloc[h:]))
        wy = d["why"].value_counts(normalize=True)
        print("  청산 사유: 익절 %.2f / 손절 %.2f / 시간정지 %.2f | 보유 중앙 %.0f분 평균 %.1f분"
              % (wy.get("tp", 0), wy.get("sl", 0), wy.get("to", 0),
                 d["hold"].median(), d["hold"].mean()))

        # 손절이 과반 발동하면 그 폭은 너무 타이트하다 — 청산 규칙을 다시 훑는다
        print("\n" + "-" * 122)
        print("2. 청산 규칙 재탐색 (K=%.0f gap=%d분 고정) — 손절 발동률이 높으면 폭이 좁은 것이다"
              % (k, g))
        print("-" * 122)
        head()
        for tmax in (5, 15, 30):
            for tp_, sl_ in ((8.0, 2.0), (8.0, 4.0), (8.0, 8.0),
                             (4.0, 2.0), (16.0, 4.0), (99.0, 99.0)):
                x = run(P, k, a.doi, g, tmax, tp_, sl_, a.cost, a.slip)
                s = summ(x)
                if s is None:
                    continue
                wy = x["why"].value_counts(normalize=True)
                tag = ("시간청산만" if tp_ > 50 else "익%.0f손%.0f" % (tp_, sl_))
                line("%-8s 상한%d분 [손%.2f]" % (tag, tmax, wy.get("sl", 0)), s)
            print()
    print("\n" + "-" * 122)
    print("3. ★ 자본 정규화 — 동시보유 중첩을 세고 나서 다시 비교한다")
    print("-" * 122)
    print("  gap 을 풀면 한 캐스케이드에 진입이 겹친다. 각 진입을 1단위로 잡은 수익은")
    print("  최대 동시보유가 M 이면 사실상 **M배 레버리지**다. 같은 자본이려면 M 으로 나눈다.")
    print("  '진입 후 g분간 같은 심볼 재진입 금지' = gap=g 와 같은 규칙이다.\n")
    print("  %-14s | %6s %7s | %8s %8s | %9s %9s %9s | %11s %11s"
          % ("설정", "n", "연간", "시도당bp", "연간총bp", "최대동시",
             "심볼최대", "p99", "자본정규화", "낙폭정규화"))
    for k in a.ks:
        for g in a.gaps:
            d = run(P, k, a.doi, g, a.tmax, a.tp, a.sl, a.cost, a.slip)
            s = summ(d)
            if s is None:
                continue
            c = concurrency(d)
            M = max(c["max_all"], 1)
            print("  K=%-2.0f gap=%-2dm   | %6d %7.0f | %8.1f %8.0f | %9d %9d %9.1f | %11.0f %11.0f"
                  % (k, g, s["n"], s["per_yr"], s["bp"], s["yr_bp"],
                     c["max_all"], c["max_sym"], c["p99_all"],
                     s["yr_bp"] / M, s["maxdd"] / M))
        print()
    print("  ** 자본정규화 = 연간총bp / 최대동시보유. 최악의 순간에 필요한 자본 기준이다.")
    print("     p99 가 최대보다 훨씬 작으면 최대는 드문 사건이므로 p99 기준이 더 현실적이다. **")
    print("\n  ** 5분봉 방아쇠 기준(asymmetry.py 조합3): 연 154건 40.8bp 연간총 6305 샤프 1.82 **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
