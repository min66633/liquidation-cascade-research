# -*- coding: utf-8 -*-
"""방아쇠를 낮추면 회전율이 오르는가, 기대값이 죽는가 — HFT 논리가 성립하는 영역 찾기.

문제 제기 (사용자, 2026-08-05)
  "수익이 막 거창하지 않아도 승률이랑 수익비 고려해서 기대값만 높고,
   회전율만 높으면 해볼만 하다. HFT 의 건당 수익이 막 대단하지는 않잖아요."

  맞다. 그런데 **지금 방아쇠는 그 논리가 성립하는 영역이 아니다.**
      K=8, dOI<=-2%, gap=12 -> 연 179건, 평균 동시보유 **0.0003**
      = 21종을 다 합쳐도 이틀에 한 번. 자본이 99.97% 놀고 있다.
  HFT 경제학(작은 우위 x 큰 횟수)을 쓰려면 횟수를 20~200배로 늘려야 하고,
  그러려면 K 를 내려야 한다. 그때 우위가 살아남는지는 **한 번도 안 봤다**.
  K=8/dOI=-0.02/gap=12 는 이 세션 이전에 정해진 값이고 그대로 물려 썼다.

무엇을 재는가
  K x dOI 격자에서
      n, 연간 건수, 시도당bp, t(일클러스터), 승률, 손익비
      **연간 총bp = 시도당bp x 연간 건수**   <- 회전율까지 반영한 실제 크기
      샤프 = t/sqrt(년수)
      평균 동시보유 = n x 보유분 / 전체 분   <- 자본이 제약인지
  진입은 **시장가**로 고정한다. 여기서 묻는 것은 '방아쇠가 살아있는가' 뿐이므로
  모형이 끼면 해석이 섞인다.

  ** 낮은 K 에서 시도당bp 가 줄어도 연간 총bp 가 커지면 그쪽이 낫다. **
  ** 단 비용에 민감해진다 — 왕복 10bp 가 우위보다 커지는 지점을 같이 본다. **

실행:
    python analysis/trigger_sweep.py
    python analysis/trigger_sweep.py --hold 5
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

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
MINS_YR = 365.25 * 24 * 60


def prep(symbols):
    """심볼당 5분봉/1분봉을 한 번만 읽는다. K 를 훑을 때마다 다시 읽으면 못 끝난다."""
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
        out[s] = (df, m["open_time"].to_numpy(),
                  m["open"].to_numpy(dtype=np.float64),
                  m["high"].to_numpy(dtype=np.float64),
                  m["low"].to_numpy(dtype=np.float64),
                  m["close"].to_numpy(dtype=np.float64))
    return out


def run(prepped, k, doi_thr, gap, hold, cost) -> pd.DataFrame:
    rows = []
    for s, (df, ot1, O, H, L, Cl) in prepped.items():
        ev = find_events(df, k, doi_thr, gap)
        ev = ev[ev.is_liq]
        if not len(ev):
            continue
        ot5 = df["open_time"].to_numpy()
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
            p_in, p_out = O[j], Cl[j + hold]
            if not (np.isfinite(p_in) and p_in > 0 and np.isfinite(p_out)):
                continue
            mae = ((L[j:j + hold + 1].min() / p_in - 1.0) if sd == 1
                   else -(H[j:j + hold + 1].max() / p_in - 1.0)) * 1e4
            rows.append({"symbol": s, "t": t0, "side": sd,
                         "ret": (p_out / p_in - 1.0) * sd * 1e4 - cost,
                         "gross": (p_out / p_in - 1.0) * sd * 1e4,
                         "mae": mae, "day": t0 // 86_400_000})
    return pd.DataFrame(rows)


def summ(d, hold):
    if len(d) < 30:
        return None
    r = d["ret"].to_numpy()
    g = d["gross"].to_numpy()
    m, _, t, _ = cmean(r, d["day"].to_numpy())
    mg, _, tg, _ = cmean(g, d["day"].to_numpy())
    yrs = (d["t"].max() - d["t"].min()) / (365.25 * 86_400_000)
    per_yr = len(d) / yrs if yrs > 0 else np.nan
    rt = r[r != 0]
    w, l = rt[rt > 0], rt[rt < 0]
    return {"n": len(d), "per_yr": per_yr, "bp": m, "t": t, "gross": mg, "tg": tg,
            "win": float((rt > 0).mean()) if len(rt) else np.nan,
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "yr_bp": m * per_yr, "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "conc": len(d) * hold / (yrs * MINS_YR) if yrs > 0 else np.nan,
            "mae": float(np.median(d["mae"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description="does the edge survive a lower trigger")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--ks", type=float, nargs="+",
                    default=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument("--dois", type=float, nargs="+", default=[-0.005, -0.01, -0.02])
    ap.add_argument("--stab", type=float, nargs="+", default=[12.0, -0.005, 15.0, -0.005],
                    help="안정성 점검할 (K, dOI) 쌍들, 평평하게 나열")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 122)
    print("방아쇠를 낮추면 — 회전율 vs 기대값. 진입 시장가 고정, 보유 %d분, 왕복 %.0fbp"
          % (a.hold, a.cost))
    print("=" * 122)
    P = prep(syms)
    print("심볼 %d종 로드 완료" % len(P))
    print("\n  %-5s %-7s | %6s %8s | %8s %5s | %8s %5s | %7s %6s | %10s %6s | %8s %9s"
          % ("K", "dOI", "n", "연간건수", "시도당bp", "t", "총이익bp", "t",
             "승률", "손익비", "**연간총bp**", "샤프", "동시보유", "MAE중앙"))
    best = None
    for doi in a.dois:
        for k in a.ks:
            d = run(P, k, doi, a.gap, a.hold, a.cost)
            s = summ(d, a.hold)
            if s is None:
                print("  %-5.1f %-7.3f | %6s (표본부족)" % (k, doi, len(d)))
                continue
            print("  %-5.1f %-7.3f | %6d %8.0f | %8.1f %5.1f | %8.1f %5.1f | %6.1f%% %6.2f | %10.0f %6.2f | %8.4f %9.0f"
                  % (k, doi, s["n"], s["per_yr"], s["bp"], s["t"], s["gross"], s["tg"],
                     100 * s["win"], s["pl"], s["yr_bp"], s["sharpe"], s["conc"], s["mae"]))
            if best is None or s["yr_bp"] > best[0]["yr_bp"]:
                best = (s, k, doi)
    if best:
        s, k, doi = best
        print("\n  ** 연간 총bp 최대: K=%.1f dOI=%.3f -> 연 %.0f건 x %.1fbp = **%.0f bp/년** (샤프 %.2f)"
              % (k, doi, s["per_yr"], s["bp"], s["yr_bp"], s["sharpe"]))

    # ---- 격자에서 고른 칸은 그 자체가 선택이다. 전/후반이 둘 다 서야 한다 ----
    print("\n" + "-" * 122)
    print("안정성 — 전/후반 분할 (격자 선택 편향 점검)")
    print("-" * 122)
    cells = [(8.0, -0.02)]                       # 현재 설정 (기준)
    if best:
        cells.append((best[1], best[2]))
    pairs = [(a.stab[i], a.stab[i + 1]) for i in range(0, len(a.stab) - 1, 2)]
    for kk, dd_ in pairs:
        if (kk, dd_) not in cells:
            cells.append((kk, dd_))
    print("  %-14s %-8s | %6s %8s | %8s %5s | %7s %6s | %10s %6s"
          % ("설정", "구간", "n", "연간건수", "시도당bp", "t", "승률", "손익비",
             "연간총bp", "샤프"))
    for kk, dd_ in cells:
        d = run(P, kk, dd_, a.gap, a.hold, a.cost).sort_values("t").reset_index(drop=True)
        h = len(d) // 2
        for lab, g in (("전체", d), ("전반부", d.iloc[:h]), ("후반부", d.iloc[h:])):
            s = summ(g, a.hold)
            if s is None:
                continue
            print("  %-14s %-8s | %6d %8.0f | %8.1f %5.1f | %6.1f%% %6.2f | %10.0f %6.2f"
                  % ("K=%.0f dOI=%.3f" % (kk, dd_) if lab == "전체" else "", lab,
                     s["n"], s["per_yr"], s["bp"], s["t"], 100 * s["win"], s["pl"],
                     s["yr_bp"], s["sharpe"]))
    print("  ** 한쪽만 서면 그 칸은 기간에 맞춰진 것이다. **")
    print("\n  총이익bp = 비용 차감 **전**. 시도당bp 와의 차이가 곧 비용 %0.fbp 다." % a.cost)
    print("  낮은 K 에서 총이익이 비용에 못 미치면 그 영역은 원리상 못 쓴다.")
    print("  동시보유가 1 을 넘기 시작해야 비로소 자본이 제약이 된다 (= HFT 영역).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
