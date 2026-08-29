# -*- coding: utf-8 -*-
"""방향 비대칭 · 레짐 조건부 · 익절/손절 — 지금까지 안 본 세 가지.

문제 제기 (사용자, 2026-08-05)
  1. "하락장일 때는 반등이 나올 수 있는데, 상승장일 때 반대로 숏 잡았다가
      반등 안 나올 수도 있다."
     -> 지금까지 side=+1(롱청산 후 **매수**) 과 side=-1(숏청산 후 **매도**) 을
        **합쳐서 평균만** 냈다. 방향별로 나눠 본 적이 없고, 시장 레짐 조건부로도
        본 적이 없다. 숏 쪽이 죽어 있으면 표본의 절반이 잡음이거나 마이너스다.
  2. "익절 규칙은요?"
     -> 없다. 시장가 진입 후 hold 분 뒤 **종가 청산**이 전부다.
  3. "손절도 가격에 따라 하는 게 아니라 시간에 따라 그냥 청산이잖아요"
     -> 맞다. two_leg.py 에서 가격 손절을 붙여봤지만 그건 **정적 지정가 진입**
        기준이었다. 지금 최선 설정(시장가 진입, K=12/dOI=-0.005)에서는 미검정.

무엇을 재는가
  1) 방향별  side=+1(매수) vs -1(매도)
  2) 레짐별  심볼 자신의 과거 30일 수익 / BTC 과거 30일 수익 (둘 다 과거창, 누출 없음)
  3) 교차    방향 x 레짐  <- 사용자가 지적한 정확한 칸이 여기 있다
             ('상승장에서 매도' 칸이 죽어 있는가)
  4) 익절/손절  진입가 대비 sigma 배수. 시간정지와 함께.

*** 봉내 순서 ***
  1분봉 하나에서 익절선·손절선에 둘 다 닿으면 순서를 알 수 없다.
  **손절 우선**(보수적)으로 잡는다. 반대로 하면 가짜 승률이 나온다.
  손절은 시장가이므로 슬리피지를 불리하게 더한다.

실행:
    python analysis/asymmetry.py
    python analysis/asymmetry.py --k 12 --doi -0.005
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
from analysis.event_study_h2 import load, find_events                 # noqa: E402
from analysis.response_liq import cmean                               # noqa: E402
from analysis.trigger_sweep import prep, MINS_YR                      # noqa: E402

REG_BARS = 8640                # 5분봉 30일


def regimes(prepped):
    """심볼별 과거 30일 수익(레짐)과, BTC 레짐을 시간축으로 붙일 준비."""
    reg = {}
    for s, (df, *_rest) in prepped.items():
        r = pd.Series(df["ret"].to_numpy())
        # shift(1) 로 현재 바 제외 — 과거창만 쓴다
        reg[s] = (df["open_time"].to_numpy(),
                  r.rolling(REG_BARS, min_periods=REG_BARS // 3).sum()
                   .shift(1).to_numpy())
    return reg


def run(prepped, reg, k, doi_thr, gap, hold, cost, tp=None, sl=None,
        slip=5.0) -> pd.DataFrame:
    """시장가 진입. tp/sl 이 None 이면 시간 청산만 (기존 규칙)."""
    btc = reg.get("BTCUSDT")
    rows = []
    for s, (df, ot1, O, H, L, Cl) in prepped.items():
        ev = find_events(df, k, doi_thr, gap)
        ev = ev[ev.is_liq]
        if not len(ev):
            continue
        ot5 = df["open_time"].to_numpy()
        sig5 = df["sigma"].to_numpy()
        ot_r, rv = reg[s]
        n1 = len(ot1)
        for r in ev.itertuples():
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5):
                continue
            t0 = int(ot5[i + 1])
            j = int(np.searchsorted(ot1, t0))
            if j >= n1 or ot1[j] != t0 or j + hold >= n1:
                continue
            if ot1[j + hold] - ot1[j] != hold * 60_000:
                continue
            p_in = O[j]
            sg = float(sig5[i])
            if not (np.isfinite(p_in) and p_in > 0 and np.isfinite(sg) and sg > 0):
                continue
            why, fee_out = "to", 0.0
            if tp is None or sl is None:
                p_out = Cl[j + hold]
            else:
                tpx = p_in * (1.0 + sd * tp * sg)
                slx = p_in * (1.0 - sd * sl * sg)
                p_out, why = Cl[j + hold], "to"
                for u in range(j + 1, j + hold + 1):     # 진입 봉은 건너뛴다
                    hs = (L[u] <= slx) if sd == 1 else (H[u] >= slx)
                    ht = (H[u] >= tpx) if sd == 1 else (L[u] <= tpx)
                    if hs:                                # 봉내 동시 -> 손절 우선
                        p_out = slx * (1.0 - sd * slip * 1e-4)   # 시장가 슬리피지
                        why = "sl"
                        break
                    if ht:
                        p_out, why = tpx, "tp"
                        break
            if not np.isfinite(p_out):
                continue
            # 레짐: 심볼 자신 / BTC
            ki = int(np.searchsorted(ot_r, t0, side="right")) - 1
            rs = float(rv[ki]) if 0 <= ki < len(rv) else np.nan
            rb = np.nan
            if btc is not None:
                kb = int(np.searchsorted(btc[0], t0, side="right")) - 1
                if 0 <= kb < len(btc[1]):
                    rb = float(btc[1][kb])
            rows.append({"symbol": s, "t": t0, "side": sd, "sig5": sg,
                         "ret": (p_out / p_in - 1.0) * sd * 1e4 - cost,
                         "why": why, "reg_sym": rs, "reg_btc": rb,
                         "day": t0 // 86_400_000})
    return pd.DataFrame(rows)


def summ(d, hold):
    if len(d) < 25:
        return None
    r = d["ret"].to_numpy()
    m, _, t, _ = cmean(r, d["day"].to_numpy())
    yrs = (d["t"].max() - d["t"].min()) / (365.25 * 86_400_000)
    per_yr = len(d) / yrs if yrs > 0 else np.nan
    w, l = r[r > 0], r[r < 0]
    eq = np.cumsum(r)
    return {"n": len(d), "per_yr": per_yr, "bp": m, "t": t,
            "win": float((r > 0).mean()),
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "yr_bp": m * per_yr, "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "maxdd": float((eq - np.maximum.accumulate(eq)).min())}


def line(lab, s):
    if s is None:
        print("  %-30s  (표본부족)" % lab)
        return
    print("  %-30s | %5d %7.0f | %8.1f %5.1f | %6.1f%% %6.2f | %9.0f %6.2f %8.0f"
          % (lab, s["n"], s["per_yr"], s["bp"], s["t"], 100 * s["win"], s["pl"],
             s["yr_bp"], s["sharpe"], s["maxdd"]))


def head():
    print("  %-30s | %5s %7s | %8s %5s | %7s %6s | %9s %6s %8s"
          % ("구분", "n", "연간", "시도당bp", "t", "승률", "손익비",
             "연간총bp", "샤프", "최대낙폭"))


def main() -> int:
    ap = argparse.ArgumentParser(description="side asymmetry, regime, TP/SL")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=12.0)
    ap.add_argument("--doi", type=float, default=-0.005)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--slip", type=float, default=5.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 112)
    print("방향 비대칭 · 레짐 조건부 · 익절/손절  (K=%.0f dOI=%.3f 보유%d분 왕복%.0fbp)"
          % (a.k, a.doi, a.hold, a.cost))
    print("=" * 112)
    P = prep(syms)
    R = regimes(P)
    d = run(P, R, a.k, a.doi, a.gap, a.hold, a.cost)
    if len(d) < 100:
        print("표본 부족")
        return 1
    print("**사용 데이터 기간: %s ~ %s / %d종 / %d건**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d)))

    print("\n" + "-" * 112)
    print("1. 방향 — 매수(롱청산 후) vs 매도(숏청산 후)")
    print("-" * 112)
    head()
    line("전체", summ(d, a.hold))
    line("  매수 (하락 후 반등 기대)", summ(d[d.side == 1], a.hold))
    line("  매도 (상승 후 되돌림 기대)", summ(d[d.side == -1], a.hold))
    print("  ** 매도가 죽어 있으면 표본의 그만큼이 잡음이거나 마이너스다. **")

    print("\n" + "-" * 112)
    print("2. 레짐 — 과거 30일 추세 (과거창만, 누출 없음)")
    print("-" * 112)
    for col, nm in (("reg_btc", "BTC 30일"), ("reg_sym", "심볼 자신 30일")):
        v = d[col].to_numpy()
        ok = np.isfinite(v)
        if ok.sum() < 100:
            continue
        thr = 0.0
        print("  [%s]  임계 0 (상승/하락)" % nm)
        head()
        line("  상승장", summ(d[ok & (v > thr)], a.hold))
        line("  하락장", summ(d[ok & (v <= thr)], a.hold))

    print("\n" + "-" * 112)
    print("3. ★ 교차 — 방향 x 레짐 (사용자가 지적한 칸이 여기다)")
    print("-" * 112)
    v = d["reg_btc"].to_numpy()
    ok = np.isfinite(v)
    head()
    for sd, sn in ((1, "매수"), (-1, "매도")):
        for up, un in ((True, "상승장"), (False, "하락장")):
            m = ok & (d.side == sd) & ((v > 0) if up else (v <= 0))
            line("%s x %s (BTC)" % (sn, un), summ(d[m], a.hold))
    print("  ** '매도 x 상승장' 이 마이너스면 그 칸을 빼는 것만으로 개선된다. **")

    print("\n" + "-" * 112)
    print("4. 익절/손절 — 지금은 둘 다 없고 시간 청산뿐이다 (sigma 배수, 손절우선+슬리피지)")
    print("-" * 112)
    head()
    line("시간청산만 (현행)", summ(d, a.hold))
    for tp in (2.0, 4.0, 8.0):
        for sl in (2.0, 4.0, 8.0):
            x = run(P, R, a.k, a.doi, a.gap, a.hold, a.cost, tp=tp, sl=sl, slip=a.slip)
            s = summ(x, a.hold)
            if s is None:
                continue
            why = x["why"].value_counts(normalize=True)
            line("익절%.0fs 손절%.0fs [익%.2f/손%.2f/시%.2f]"
                 % (tp, sl, why.get("tp", 0), why.get("sl", 0), why.get("to", 0)), s)
    print("  ** 어떤 조합도 시간청산을 못 넘으면, 익절·손절이 이 전략에 해롭다는 뜻이다. **")

    print("\n" + "-" * 112)
    print("5. ★ 조합 — 방향 필터 + 레짐 필터 + 익절/손절 을 겹쳐 쌓는다")
    print("-" * 112)
    print("  세 개가 각각 개선한다고 합쳐서도 개선된다는 보장은 없다. 겹쳐서 확인한다.\n")
    x8 = run(P, R, a.k, a.doi, a.gap, a.hold, a.cost, tp=8.0, sl=2.0, slip=a.slip)
    vb = d["reg_btc"].to_numpy()
    vb8 = x8["reg_btc"].to_numpy()
    head()
    line("(0) 현행 전부·시간청산", summ(d, a.hold))
    line("(1) +매수만", summ(d[d.side == 1], a.hold))
    line("(2) +매도x하락장 제외",
         summ(d[~((d.side == -1) & np.isfinite(vb) & (vb <= 0))], a.hold))
    line("(3) 매수만 + 익절8s손절2s", summ(x8[x8.side == 1], a.hold))
    line("(4) 매도x하락장제외 + 익절8s손절2s",
         summ(x8[~((x8.side == -1) & np.isfinite(vb8) & (vb8 <= 0))], a.hold))
    # 안정성: 위 후보들을 전/후반으로
    print("\n  전/후반 분할:")
    head()
    cands = [("(0) 현행", d),
             ("(1) 매수만", d[d.side == 1]),
             ("(2) 매도x하락장 제외",
              d[~((d.side == -1) & np.isfinite(vb) & (vb <= 0))]),
             ("(3) 매수만+익8손2", x8[x8.side == 1])]
    for lab, g in cands:
        g = g.sort_values("t").reset_index(drop=True)
        h = len(g) // 2
        line(lab + " 전반", summ(g.iloc[:h], a.hold))
        line(lab + " 후반", summ(g.iloc[h:], a.hold))
    print("  ** 전/후반 둘 다 서는 조합만 쓸 수 있다. **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
