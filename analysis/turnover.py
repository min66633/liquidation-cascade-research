# -*- coding: utf-8 -*-
"""R-6 — 이벤트당 1회 진입 가정을 버린다. 고회전 + 위험조정 + 모수 스윕.

무엇이 잘못됐었나 (사용자 지적 3개)
  (1) '대기 15분' 은 지정가를 3봉 걸어두고 안 닿으면 포기한다는 뜻인데, 그 15분을
      근거 없이 박았다.
  (2) 보유 15분도 R-2 의 '15분에 결판난다' 한 줄을 그대로 앵커링한 것이다.
  (3) **가장 큰 오류**: 이벤트당 1회 진입을 가정했다. 캐스케이드는 원웨이가 아니다.
      호가가 들어오거나 심리로 약반등했다가 다시 빠진다. 그러면 반등에 털고
      **다음 바닥을 다시 잡는** 고회전이 맞다. 그게 원래 설계('매 순간 다시 읽어
      재계산')에 훨씬 가깝다.

그래서 바꾼 것
  - 5분봉 -> **1분봉**. 반등-재하락 주기가 5분보다 짧을 수 있다.
  - 단발 진입 -> **추격 지정가 반복**. 매 분 직전 종가 대비 d bp 아래에 다시 건다.
  - 고정 모수 -> **스윕**. d(진입깊이) x g(이익목표) x H(시간손절) 격자.
  - EV 만 -> **위험조정**. 표준편차·MAE·EV/위험 을 같이 낸다.

체결 규약 (1분봉 OHLC 한계 안에서 보수적으로)
  진입: low <= limit 이면 limit 에 체결.
  청산: high >= target 이면 target 에 체결. 단 **같은 봉에서 진입과 청산을 모두
        인정하지 않는다** (봉 안 순서를 모르므로). 시간손절은 종가 시장가.
  비용: 메이커 2bp/편, 시간손절 시 청산은 테이커 5bp. (앞선 R-5 의 왕복 2bp 는
        너무 낙관적이었다 — 메이커 왕복은 4bp 다.)

실행:
    python analysis/turnover.py
    python analysis/turnover.py --window 120
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
from analysis.scale_check import K, DOI_THR, MIN_GAP            # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
FEE_MAKER = 2.0        # bp / 편
FEE_TAKER = 5.0        # bp / 편

DEPTHS = [10, 25, 50, 100, 200]        # 직전 종가 대비 진입 깊이(bp)
TARGETS = [10, 25, 50, 100, 200]       # 진입가 대비 이익목표(bp)
STOPS = [5, 15, 30, 60]                # 시간손절(분)


def sim(op, hi, lo, cl, s0, s1, side, d, g, H):
    """추격 지정가 반복. 반환: (거래별 수익 bp 리스트, 보유분 합, 최대역행 리스트)

    side=+1 : 롱청산 -> 하락 -> 매수 후 반등에 청산
    side=-1 : 숏청산 -> 상승 -> 매도 후 되돌림에 청산
    """
    rets, mins, maes = [], 0, []
    pos = False
    fill = 0.0
    held = 0
    mae = 0.0
    for t in range(s0 + 1, s1):
        if not pos:
            ref = cl[t - 1]
            lim = ref * (1.0 - side * d / 1e4)
            hit = (lo[t] <= lim) if side == 1 else (hi[t] >= lim)
            if hit:
                pos, fill, held, mae = True, lim, 0, 0.0
            continue
        # 보유 중 — 같은 봉 진입/청산은 인정하지 않으므로 여기부터 시작
        held += 1
        adv = ((lo[t] / fill - 1.0) if side == 1 else (fill / hi[t] - 1.0)) * 1e4
        mae = min(mae, adv)
        tgt = fill * (1.0 + side * g / 1e4)
        won = (hi[t] >= tgt) if side == 1 else (lo[t] <= tgt)
        if won:
            rets.append(g - 2 * FEE_MAKER)          # 양편 메이커
            mins += held
            maes.append(mae)
            pos = False
        elif held >= H:
            r = (cl[t] / fill - 1.0) * side * 1e4
            rets.append(r - FEE_MAKER - FEE_TAKER)  # 진입 메이커 + 청산 테이커
            mins += held
            maes.append(mae)
            pos = False
    if pos:                                          # 창 끝에서 강제 청산
        r = (cl[s1 - 1] / fill - 1.0) * side * 1e4
        rets.append(r - FEE_MAKER - FEE_TAKER)
        mins += held
        maes.append(mae)
    return rets, mins, maes


def load1m(s):
    p = os.path.join(BULK1, "%s.parquet" % s)
    if not os.path.exists(p):
        return None
    k = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close"])
    return k.sort_values("open_time").reset_index(drop=True)


def build(symbols, window):
    """이벤트별 1분봉 창을 잘라 둔다."""
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, K, DOI_THR, MIN_GAP)
        if len(ev) == 0:
            continue
        k = load1m(s)
        if k is None:
            continue
        ot1 = k["open_time"].to_numpy()
        arr = {c: k[c].to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
        ot5 = df["open_time"].to_numpy()
        n1 = len(ot1)
        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i = int(r.i)
            if i + 1 >= len(ot5):
                continue
            t0 = int(ot5[i + 1])                      # 진입 가능 시각
            s0 = int(np.searchsorted(ot1, t0))
            s1 = s0 + window
            if s0 <= 0 or s1 >= n1:
                continue
            if ot1[s1 - 1] - ot1[s0] != (window - 1) * 60_000:
                continue                              # 봉 결손 창은 버린다
            out.append({"symbol": s, "side": int(r.side), "s0": s0, "s1": s1,
                        "day": int(t0 // 86_400_000), "arr": arr})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="R-6 turnover / risk-adjusted")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=240, help="이벤트 창(분)")
    ap.add_argument("--fee-maker", type=float, default=FEE_MAKER)
    ap.add_argument("--fee-taker", type=float, default=FEE_TAKER)
    a = ap.parse_args()
    globals()["FEE_MAKER"] = a.fee_maker
    globals()["FEE_TAKER"] = a.fee_taker
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 82)
    print("R-6 — 고회전 반복 진입. 이벤트당 1회 가정을 버린다")
    print("=" * 82)
    print("비용: 메이커 %.0fbp/편, 시간손절 청산은 테이커 %.0fbp" % (FEE_MAKER, FEE_TAKER))
    print("     (R-5 의 왕복 2bp 는 낙관적이었다. 메이커 왕복은 4bp 다.)")
    ws = build(syms, a.window)
    if not ws:
        print("창 없음")
        return 1
    days = np.array([w["day"] for w in ws])
    print("\n**이벤트 창 %d개 / %d종 / 창 길이 %d분 (1분봉)**"
          % (len(ws), len({w["symbol"] for w in ws}), a.window))

    print("\n" + "-" * 82)
    print("모수 격자 — 앵커링하지 않고 전부 훑는다")
    print("-" * 82)
    print("  %5s %5s %5s | %7s %9s %9s | %9s %8s %9s"
          % ("깊이", "목표", "손절", "거래/창", "이벤트당", "t", "거래당", "표준편차",
             "EV/위험"))
    best = []
    for d in DEPTHS:
        for g in TARGETS:
            for H in STOPS:
                per_ev, all_tr, all_mae = [], [], []
                for w in ws:
                    A = w["arr"]
                    rr, _, mm = sim(A["open"], A["high"], A["low"], A["close"],
                                    w["s0"], w["s1"], w["side"], d, g, H)
                    per_ev.append(float(np.sum(rr)) if rr else 0.0)
                    all_tr.extend(rr)
                    all_mae.extend(mm)
                if len(all_tr) < 100:
                    continue
                pe = np.array(per_ev)
                m, se, t, _ = cmean(pe, days)
                tr = np.array(all_tr)
                sd = float(pe.std(ddof=1))
                risk = abs(float(np.median(all_mae))) if all_mae else np.nan
                evr = m / risk if risk and risk > 0 else np.nan
                best.append((m, t, d, g, H, len(tr) / len(ws), tr.mean(), sd, evr))
    best.sort(key=lambda x: -x[0])
    for row in best[:12]:
        m, t, d, g, H, npc, trm, sd, evr = row
        print("  %5d %5d %5d | %7.2f %9.1f %9.1f | %9.1f %8.0f %9.3f"
              % (d, g, H, npc, m, t, trm, sd, evr))
    print("  ... 상위 12개 (이벤트당 기준). 전체 %d조합" % len(best))

    print("\n" + "-" * 82)
    print("위험조정 기준 상위 (EV / |MAE 중앙|)")
    print("-" * 82)
    print("  %5s %5s %5s | %7s %9s %9s | %9s %9s"
          % ("깊이", "목표", "손절", "거래/창", "이벤트당", "t", "표준편차", "EV/위험"))
    for row in sorted([b for b in best if np.isfinite(b[8])], key=lambda x: -x[8])[:8]:
        m, t, d, g, H, npc, trm, sd, evr = row
        print("  %5d %5d %5d | %7.2f %9.1f %9.1f | %9.0f %9.3f"
              % (d, g, H, npc, m, t, sd, evr))

    print("\n" + "-" * 82)
    print("대조 — 단발 진입(R-5 구조)을 같은 1분봉·같은 비용으로")
    print("-" * 82)
    print("  %-22s | %7s | %9s %6s | %8s %9s"
          % ("설정", "체결률", "이벤트당", "t", "표준편차", "EV/위험"))
    for d in (0, 25, 50, 100):
        for H in (15, 60, 240):
            rs, fills, maes = [], 0, []
            for w in ws:
                A = w["arr"]
                s0, s1, sd_ = w["s0"], w["s1"], w["side"]
                lim = A["close"][s0 - 1] * (1.0 - sd_ * d / 1e4)
                seg = slice(s0, min(s0 + 15, s1))
                hit = ((A["low"][seg] <= lim) if sd_ == 1 else (A["high"][seg] >= lim))
                if not hit.any():
                    rs.append(0.0)
                    continue
                fb = s0 + int(np.argmax(hit))
                te = min(fb + H, s1 - 1)
                rs.append((A["close"][te] / lim - 1.0) * sd_ * 1e4
                          - FEE_MAKER - FEE_TAKER)
                # 단발도 같은 정의로 최대역행을 잰다 (보유 구간 전체)
                if sd_ == 1:
                    maes.append((A["low"][fb:te + 1].min() / lim - 1.0) * 1e4)
                else:
                    maes.append(-(A["high"][fb:te + 1].max() / lim - 1.0) * 1e4)
                fills += 1
            arr_ = np.array(rs)
            m, se, t, _ = cmean(arr_, days)
            risk = abs(float(np.median(maes))) if maes else np.nan
            print("  깊이%4dbp 보유%4d분      | %6.1f%% | %9.1f %6.1f | %8.0f %9.3f"
                  % (d, H, 100 * fills / len(ws), m, t, arr_.std(ddof=1),
                     m / risk if risk and risk > 0 else np.nan))
    print("\n  *** 위험은 양쪽 다 '거래 1건의 MAE 중앙' 으로 통일했다.")
    print("  이벤트당 EV 로는 단발이, EV/위험 으로는 고회전이 이길 수 있다.")
    print("  둘이 갈리면 **같은 위험예산으로 크기를 맞춰** 비교해야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
