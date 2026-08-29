# -*- coding: utf-8 -*-
"""캐스케이드를 **타는** 전략 — 되돌림을 노리는 게 아니라 슈팅을 먹는다.

왜 이 전환인가
  intra_event.py: 사건 순간에 **역방향(매수)** 진입하면 -229.6bp (t=-6.2).
  이유는 명백하다 — 그 시점 이후로 **X 중앙 425bp 가 더 밀린다** (X/r0 = 5.30).
  즉 바닥을 안 쳤는데 진입해서 손해다.

  **그러면 같은 정보로 방향만 뒤집으면 된다.** 425bp 는 되돌림 51bp 의 8배다.
  되돌림을 노리려면 바닥을 맞혀야 하지만, 슈팅을 타려면 **시작만 맞히면 된다.**
  후자가 훨씬 쉬운 문제다.

세 가지를 같이 잰다
  (1) 방향: 순방향(타기) vs 역방향(되돌림 노리기)
  (2) 방아쇠: OI확인(5분 지연·룩어헤드) vs 가격만(실시간·룩어헤드 없음)
  (3) 청산: 고정시간 vs 추적손절(캐스케이드가 끝나는 지점을 모르므로)

*** 룩어헤드 표시 ***
  OI확인 방아쇠는 5분봉이 끝나야 알 수 있는데 진입은 그 안 1분봉이다.
  **거래 불가**다. 실시간 판정은 '가격만' 쪽을 봐야 한다.
  단 oi_fast(5초 OI)가 쌓이면 OI확인도 실시간이 된다 — 그래서 둘 다 잰다.

실행:
    python analysis/momentum.py
    python analysis/momentum.py --zk 4
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
VOL_WIN = 1440
FEE = 10.0             # 왕복 bp (시장가 진입·청산 = 테이커 5bp x 2)


def build(symbols, zk, window, use_oi, delay=0):
    """사건 시작 1분봉을 찾아 그 종가부터의 경로를 담는다."""
    out = []
    for s in symbols:
        p1 = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p1):
            continue
        k1 = pd.read_parquet(p1, columns=["open_time", "high", "low", "close"])
        k1 = k1.sort_values("open_time").reset_index(drop=True)
        ot1 = k1["open_time"].to_numpy()
        H, L, Cl = (k1[c].to_numpy(dtype=np.float64) for c in ("high", "low", "close"))
        n1 = len(Cl)
        r1 = np.concatenate([[np.nan], Cl[1:] / Cl[:-1] - 1.0])
        sg = pd.Series(r1).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 3
                                            ).std().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            z1 = r1 / sg
        cont = np.concatenate([[False], np.diff(ot1) == 60_000])

        if use_oi is True:
            # OI 확인 방아쇠 — 5분봉 사건 안에서 첫 |z1|>=zk 봉을 찾는다 (룩어헤드)
            try:
                df = load(s)
            except FileNotFoundError:
                continue
            ev = find_events(df, K, DOI_THR, MIN_GAP)
            ot5 = df["open_time"].to_numpy()
            cand = []
            for r in ev.itertuples():
                if not r.is_liq:
                    continue
                b0 = int(np.searchsorted(ot1, int(ot5[int(r.i)])))
                if b0 <= 0 or b0 >= n1 or ot1[b0] != int(ot5[int(r.i)]):
                    continue
                for t in range(b0, min(b0 + 5, n1)):
                    if np.isfinite(z1[t]) and (-z1[t] * r.side) >= zk and cont[t]:
                        cand.append((t, int(r.side)))
                        break
        elif use_oi == "rt":
            # *** 완전 실시간 OI 방아쇠 ***
            #   바 i 의 dOI 는 **바 i+1 시각**에야 알 수 있다(다음 스냅샷이 필요).
            #   그러므로 시각 T 에 쓸 수 있는 dOI 는 ot5[i+1] <= T 인 것 중 최신이다.
            #   방향도 5분봉 전체가 아니라 **그 1분봉 자신의 부호**로 정한다.
            try:
                df = load(s)
            except FileNotFoundError:
                continue
            ot5 = df["open_time"].to_numpy()
            doi = df["doi"].to_numpy(dtype=np.float64)
            known_t = ot5[1:]                     # doi[i] 가 알려지는 시각
            known_v = doi[:-1]
            cand = []
            last = {}
            for t in np.where(np.isfinite(z1) & (np.abs(z1) >= zk) & cont)[0]:
                sd = 1 if z1[t] < 0 else -1
                if t - last.get(sd, -10**9) < 60:
                    continue
                ki = int(np.searchsorted(known_t, ot1[t], side="right")) - 1
                if ki < 0 or not np.isfinite(known_v[ki]) or known_v[ki] > DOI_THR:
                    continue
                last[sd] = t
                cand.append((int(t), sd))
        else:
            # 가격만 — 완전 실시간. 방향은 그 봉의 부호
            cand = []
            last = {}
            for t in np.where(np.isfinite(z1) & (np.abs(z1) >= zk) & cont)[0]:
                sd = 1 if z1[t] < 0 else -1
                if t - last.get(sd, -10**9) < 60:
                    continue
                last[sd] = t
                cand.append((int(t), sd))

        for j, sd in cand:
            j = j + delay          # **체결 지연**: 신호 후 delay 분 뒤에 진입
            if j + window >= n1 or j <= VOL_WIN:
                continue
            if not cont[j + 1:j + window].all():
                continue
            p0 = Cl[j]
            if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sg[j]) and sg[j] > 0):
                continue
            out.append({"symbol": s, "side": sd, "t": int(ot1[j]),
                        "day": int(ot1[j] // 86_400_000),
                        "r0": abs(r1[j]) * 1e4, "sig": sg[j] * np.sqrt(float(VOL_WIN)),
                        "p0": p0,
                        "lo": L[j + 1:j + 1 + window].astype(np.float32),
                        "hi": H[j + 1:j + 1 + window].astype(np.float32),
                        "cl": Cl[j + 1:j + 1 + window].astype(np.float32)})
    return pd.DataFrame(out)


def pnl_fixed(d, direction, hold):
    """고정시간 청산. direction=+1 이면 사건 방향으로 탄다(슈팅), -1 이면 역방향."""
    rs = []
    for i in range(len(d)):
        sd = int(d["side"].iloc[i]) * (-1 if direction > 0 else 1)
        # side=+1 은 '하락 사건'. 슈팅을 타려면 **매도**(-1), 되돌림이면 매수(+1)
        cl = d["cl"].iloc[i]
        h = min(hold, len(cl) - 1)
        rs.append((float(cl[h]) / d["p0"].iloc[i] - 1.0) * sd * 1e4 - FEE)
    return np.array(rs)


def pnl_trail(d, direction, trail_bp, maxhold):
    """추적손절 — 캐스케이드가 어디서 끝나는지 모르므로 고점 대비 trail 만큼 되돌면 청산."""
    rs = []
    for i in range(len(d)):
        sd = int(d["side"].iloc[i]) * (-1 if direction > 0 else 1)
        cl = np.asarray(d["cl"].iloc[i], dtype=np.float64)
        hi = np.asarray(d["hi"].iloc[i], dtype=np.float64)
        lo = np.asarray(d["lo"].iloc[i], dtype=np.float64)
        p0 = float(d["p0"].iloc[i])
        n = min(maxhold, len(cl))
        best = 0.0
        ret = 0.0
        for t in range(n):
            # 유리한 쪽 극값
            fav = ((hi[t] / p0 - 1.0) if sd == 1 else (1.0 - lo[t] / p0)) * 1e4
            best = max(best, fav)
            cur = (cl[t] / p0 - 1.0) * sd * 1e4
            if best - cur >= trail_bp:
                ret = cur
                break
        else:
            ret = (cl[n - 1] / p0 - 1.0) * sd * 1e4
        rs.append(ret - FEE)
    return np.array(rs)


def main() -> int:
    ap = argparse.ArgumentParser(description="ride the cascade vs fade it")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--zk", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=240)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 80)
    print("캐스케이드를 **타는가** vs **되돌림을 노리는가**")
    print("=" * 80)
    print("근거: 탐지 시점 이후 X 중앙 **425bp** 가 더 밀린다(X/r0=5.30).")
    print("      되돌림은 51bp. **슈팅이 8배 크다.**")
    print("      되돌림은 바닥을 맞혀야 하지만 슈팅은 **시작만** 맞히면 된다.\n")

    print("*** 룩어헤드 감사 ***")
    print("  OI확인판은 '5분봉 전체가 8시그마' 인 구간 **안에서** 3시그마 봉을 고른다.")
    print("  즉 **3시그마로 시작해서 결국 8시그마까지 간 경우만** 남는다 —")
    print("  흐지부지된 건 전부 빠진다. **생존자 편향이다.**")
    print("  방향도 5분봉 전체 부호에서 가져왔다(실시간엔 모른다).")
    print("  아래 세 판을 나란히 놓고, 편향을 뺐을 때 얼마가 남는지 본다.\n")

    for use_oi, tag, dly in (
            (True, "① OI확인 (**룩어헤드+생존자편향 — 거래불가**)", 0),
            ("rt", "② **실시간 OI** (직전 완료 5분봉 dOI 만 사용, 방향은 1분봉 자신)", 0),
            ("rt", "③ **실시간 OI + 1분 체결지연**", 1),
            (False, "④ 가격만 (대조)", 0)):
        d = build(syms, a.zk, a.window, use_oi, dly)
        if len(d) < 100:
            print("[%s] 표본 부족 (%d)\n" % (tag, len(d)))
            continue
        print("=" * 80)
        print("[%s]" % tag)
        print("=" * 80)
        print("  %s ~ %s / %d종 / 사건 %d건 | r0 중앙 %.0fbp"
              % (str(pd.Timestamp(d.t.min(), unit="ms"))[:10],
                 str(pd.Timestamp(d.t.max(), unit="ms"))[:10],
                 d.symbol.nunique(), len(d), d.r0.median()))
        day = d["day"].to_numpy()
        print("\n  %-22s %10s %8s %9s %8s"
              % ("방식", "이벤트당bp", "t", "승률%", "표준편차"))
        rows = []
        for direction, dl in ((+1, "**슈팅(순방향)**"), (-1, "되돌림(역방향)")):
            for hold in (5, 15, 60, 240):
                r = pnl_fixed(d, direction, hold)
                m, se, t, _ = cmean(r, day)
                rows.append((m, "%s 고정%3d분" % (dl, hold), t,
                             100 * (r > 0).mean(), r.std(ddof=1)))
            for tr in (50, 150, 400):
                r = pnl_trail(d, direction, tr, a.window)
                m, se, t, _ = cmean(r, day)
                rows.append((m, "%s 추적%3dbp" % (dl, tr), t,
                             100 * (r > 0).mean(), r.std(ddof=1)))
        for m, lab, t, w, sd_ in sorted(rows, key=lambda x: -x[0]):
            print("  %-22s %10.1f %8.1f %9.1f %8.0f" % (lab, m, t, w, sd_))
        print()
    print("  *** 슈팅이 되돌림을 이기면 전략의 방향을 바꿔야 한다.")
    print("      단 '가격만' 방아쇠에서도 이겨야 실제로 거래 가능하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
