# -*- coding: utf-8 -*-
"""반등 크기가 사건 크기에 비례하는가 — "노이즈를 잡은 것 아니냐" 에 대한 직접 검정.

사용자 지적 (2026-08-05)
  "캐스케이드 후 반등인데 그거밖에 반등이 없다고요? 막 엄청 작은 반등같은
   노이즈에서 잡아서 그런거 아니에요? 노이즈 제거부터 해서 확실한 바닥부터 잡아야."
  "캐스케이드는 좀 적당한 타임프레임에서 예측하는 동시에 작은 타임프레임까지
   고려해서 진입하라는 얘기였는데"

내가 뭘 잘못했나
  ws_micro.py 의 방아쇠는 sigma60 을 **직전 1시간**의 60초 수익 표준편차로 잡았다.
  조용한 한 시간 뒤에는 20bp 움직임도 z=-5 가 된다. 그래서 '사건' 1,991건의
  바닥 깊이 중앙이 **-9bp** 였다. 그건 캐스케이드가 아니라 틱 노이즈다.
  봉 연구의 방아쇠는 **직전 1일** sigma 의 10배였고 알트에서 보통 80~150bp 다.

이 스크립트가 답하는 것
  1. 사건을 **크기별로 층화**해서 반등 크기가 크기에 비례하는지 본다.
     비례하면 "작은 반등" 은 내가 작은 사건을 잡았기 때문이다 (사용자 가설 지지).
     비례하지 않으면 큰 캐스케이드도 반등이 5~7bp 라는 뜻이다.
  2. **노이즈 제거**: 바닥을 단일 틱이 아니라 s초 이동중앙값으로 정의한다.
     1초 mid 의 최저 틱은 웍일 수 있고 거기에 체결될 수 없다.
  3. **안정적 정규화**: sigma 를 직전 24시간으로 바꿔 진짜 캐스케이드만 남긴다.
     그 조건에서 이 창(심볼당 67.7시간)에 사건이 몇 건인지 센다.

실행:
    python analysis/ws_size.py
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
from analysis.ws_panel import load, gaps                               # noqa: E402

WIN = 60
HOR = 600
W = 118


def prep(syms):
    out = {}
    for s in syms:
        try:
            d = load(s)
        except FileNotFoundError:
            continue
        mid = pd.Series(d["mid"].to_numpy(dtype=np.float64)).ffill(limit=5)
        out[s] = (d, mid.to_numpy())
    return out


def scan(P, vol, k, minbp, smooth):
    """사건과 경로. smooth 초 이동중앙값으로 노이즈 제거한 가격을 쓴다."""
    rows, R, Rs = [], [], []
    for s, (d, mid) in P.items():
        m = pd.Series(mid)
        # 노이즈 제거: 중앙값은 단일 틱 웍에 끌려가지 않는다. 과거만 쓴다(인과적).
        sm = m.rolling(smooth, min_periods=1).median().to_numpy() if smooth > 1 else mid
        r = (m / m.shift(WIN) - 1.0)
        sd = r.rolling(vol, min_periods=vol // 4).std().shift(1)
        z = (r / sd).to_numpy()
        rbp = r.to_numpy() * 1e4
        valid = np.zeros(len(mid), dtype=bool)
        for a, b in gaps(d):
            lo = a + WIN + vol // 4
            if b - lo > HOR:
                valid[lo:b - HOR + 1] = True
        hit = np.flatnonzero(valid & np.isfinite(z) & (z <= -k)
                             & np.isfinite(rbp) & (rbp <= -minbp))
        last = -10**9
        for i in hit:
            if i - last < WIN:
                continue
            seg, segs = mid[i:i + HOR + 1], sm[i:i + HOR + 1]
            if not (np.isfinite(seg).all() and np.isfinite(segs).all()):
                continue
            last = i
            p0 = seg[0]
            rel = (seg / p0 - 1.0) * 1e4
            rels = (segs / p0 - 1.0) * 1e4
            tb = int(np.argmin(rels))
            rows.append({"symbol": s, "sec": int(i), "z": float(z[i]),
                         "drop": float(rbp[i]),          # 방아쇠의 60초 하락 (bp)
                         "t_bot": tb, "bot": float(rels[tb]),
                         "reb": float(rels[min(tb + 300, HOR)] - rels[tb])})
            R.append(rel)
            Rs.append(rels)
    return (pd.DataFrame(rows),
            np.array(R) if R else np.zeros((0, HOR + 1)),
            np.array(Rs) if Rs else np.zeros((0, HOR + 1)))


def boot(x, reps=4000, seed=5):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 6:
        return np.nan, np.nan, np.nan
    b = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sec_(n, t):
    print("\n" + "-" * W)
    print("%d. %s" % (n, t))
    print("-" * W)


def main() -> int:
    ap = argparse.ArgumentParser(description="does the rebound scale with event size")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--cost", type=float, default=10.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    P = prep(syms)

    print("=" * W)
    print("반등은 사건 크기에 비례하는가 — '노이즈를 잡은 것 아니냐' 직접 검정")
    print("=" * W)
    print("데이터: 웹소켓 1초 패널 21종, 2026-08-02 ~ 2026-08-05 (심볼당 67.7시간)")
    print("바닥 정의: %d초 이동중앙값의 최저점 (단일 틱 웍 제거, 과거만 사용)" % a.smooth)

    sec_(1, "★ 사건 크기별 층화 — 크기가 커지면 반등도 커지는가")
    d, R, Rs = scan(P, vol=3600, k=3.0, minbp=0.0, smooth=a.smooth)
    print("  느슨한 방아쇠(z60<=-3, 1시간 sigma)로 %d건을 모은 뒤 **60초 하락 크기**로 나눈다.\n"
          % len(d))
    bins = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 1e9)]
    print("  %-14s | %5s | %8s %8s | %9s %9s | %10s"
          % ("60초 하락", "n", "바닥bp", "바닥초", "반등bp", "반등/하락", "60초보유bp"))
    for lo, hi in bins:
        m = (-d["drop"] >= lo) & (-d["drop"] < hi)
        g = d[m]
        if len(g) < 5:
            print("  %-14s | %5d | (표본부족)" % ("%d~%dbp" % (lo, min(hi, 9999)), len(g)))
            continue
        ii = g.index.to_numpy()
        r60 = (R[ii, 60] - R[ii, 0]) - a.cost
        print("  %-14s | %5d | %8.0f %8.0f | %9.0f %9.2f | %10.1f"
              % ("%d~%dbp" % (lo, min(hi, 9999)), len(g),
                 g["bot"].median(), g["t_bot"].median(),
                 g["reb"].median(), g["reb"].median() / max(-g["bot"].median(), 1e-9),
                 r60.mean()))
    print("\n  반등 = 바닥에서 +300초 뒤까지의 회복(bp). 반등/하락 = 되돌림 비율.")
    print("  ** 되돌림 비율이 크기와 무관하게 일정하면 '작아서 작다' 가 맞다. **")

    sec_(2, "★ 바닥 이후 최적 보유 — 바닥을 안다고 가정했을 때의 상한 (완전예지)")
    print("  실제로는 바닥을 모른다. 이건 **먹을 게 있기는 한가** 를 재는 상한이다.\n")
    print("  %-14s | %5s | %s" % ("60초 하락", "n",
                                  " ".join("%8s" % ("+%ds" % h)
                                           for h in (15, 30, 60, 120, 300))))
    for lo, hi in bins:
        m = (-d["drop"] >= lo) & (-d["drop"] < hi)
        g = d[m]
        if len(g) < 5:
            continue
        ii = g.index.to_numpy()
        tb = g["t_bot"].to_numpy()
        out = []
        for h in (15, 30, 60, 120, 300):
            e = np.minimum(tb + h, HOR)
            v = Rs[ii, e] - Rs[ii, tb] - a.cost
            out.append("%8.0f" % np.mean(v))
        print("  %-14s | %5d | %s" % ("%d~%dbp" % (lo, min(hi, 9999)), len(g),
                                      " ".join(out)))
    print("\n  ** 이 값이 작으면 바닥을 완벽히 맞혀도 비용을 못 넘는다는 뜻이다. **")

    sec_(3, "★ 안정적 정규화(직전 24시간 sigma) — 진짜 캐스케이드는 이 창에 몇 건인가")
    print("  봉 연구의 방아쇠와 척도를 맞춘다. sigma 를 24시간으로 바꾸면 조용한")
    print("  구간의 작은 움직임이 z 를 부풀리지 못한다.\n")
    print("  %-8s | %s" % ("K", " ".join("%-20s" % ("하락>=%dbp" % b)
                                         for b in (0, 50, 100, 200))))
    for k in (4, 6, 8, 10):
        out = []
        for b in (0, 50, 100, 200):
            dd, RR, RRs = scan(P, vol=86400, k=float(k), minbp=float(b),
                               smooth=a.smooth)
            if len(dd) == 0:
                out.append("%-20s" % "n=0")
                continue
            out.append("%-20s" % ("n=%-3d 바닥%5.0fbp" % (len(dd), dd["bot"].median())))
        print("  K=%-6d | %s" % (k, " ".join(out)))
    print("\n  ** 이 표가 웹소켓 창으로 캐스케이드를 검정할 수 있는지 결정한다. **")

    sec_(4, "노이즈 제거의 효과 — 웍 바닥 vs 이동중앙값 바닥")
    for sm in (1, 3, 5, 10, 30):
        dd, RR, RRs = scan(P, vol=3600, k=5.0, minbp=50.0, smooth=sm)
        if not len(dd):
            continue
        print("  평활 %2d초 | n=%3d | 바닥 중앙 %6.0fbp | 바닥까지 중앙 %4.0f초"
              % (sm, len(dd), dd["bot"].median(), dd["t_bot"].median()))
    print("\n  ** 평활 창이 길수록 바닥이 얕아진다. 그 차이가 **체결 불가능한 웍** 이다. **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
