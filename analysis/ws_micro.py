# -*- coding: utf-8 -*-
"""1초 미시구조 — 바닥이 어디에, 언제 오는가. 그리고 지연이 얼마를 먹는가.

사용자 지시 (2026-08-05)
  "미시구조로 갈겁니다. 무조건 바닥예측해서 짧게 먹고 나오는걸로. 15분도 너무 길어요"
  "손절은 시간손절"
  "봉으로 계산하는 게 맞아요?"

그래서 여기서는
  - 봉을 쓰지 않는다. 방아쇠·진입·청산 전부 **초** 단위.
  - 손절을 쓰지 않는다. **시간 청산만** (D-10 §5.17.2: 가격 손절은 전 폭에서 음수).
  - 보유는 초 단위로 스윕한다 (15초 ~ 300초).
  - 진입 지연 L 을 명시적 축으로 둔다. L=60초가 사실상 지금의 1분봉 판이다.

방아쇠 (`ws_panel.census` 와 동일)
  z60(t) = [mid(t)/mid(t-60s) - 1] / sigma60,  매 초 평가, 60초 중복제거.
  sigma60 = 직전 3600초의 60초 수익 표준편차(현재 제외).

무엇을 재는가
  1. 바닥의 시각과 깊이 — 방아쇠 이후 최저 mid 까지 몇 초, 몇 bp
  2. 진입 지연 L 의 비용 — L 을 0..60초로 밀면서 같은 규칙의 손익
  3. 보유 시간 — 15..300초. 짧게 먹고 나오는 게 되는가
  4. ②③ — 방아쇠 시점의 **매수호가 깊이**와 **밴드 유량**이 남은 밀림을 맞히는가
     (설계의 핵심. 30초 해상도에서는 증분 0 이었다 — DESIGN_LOCK §5.16)

주의 — 표본이 작다
  67.7 심볼-시간이다. K=8 에 41건, K=10 에 25건. **t 통계량을 믿지 말 것.**
  여기서 나오는 것은 손익 추정이 아니라 **경로의 모양**이다.
  손익 유의성은 6년 봉 데이터(analysis/latency.py) 쪽에서 봐야 한다.

실행:
    python analysis/ws_micro.py
    python analysis/ws_micro.py --k 5 --cost 10
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
from analysis.response_liq import ols_cluster                          # noqa: E402

WIN, VOL = 60, 3600
HOR = 600                       # 사건 뒤 관찰 창 (초)
W = 118


def events(d, k, win=WIN, vol=VOL, dedup=WIN, minbp=0.0):
    """봉 없는 방아쇠. (사건 인덱스 배열, z 배열) 을 낸다.

    minbp : **절대 크기 하한**(bp). z 만 쓰면 '직전 1시간 대비 극단'을 고를 뿐
        경제적으로 큰 사건을 고르지 못한다. 왕복 비용이 10bp 인데 바닥 깊이
        중앙이 9bp 면 애초에 넘을 수 없다 (K=5 실측). 캐스케이드는 크기로도
        정의해야 한다.
    """
    mid = d["mid"].to_numpy(dtype=np.float64)
    m = pd.Series(mid)
    r = m / m.shift(win) - 1.0
    sd = r.rolling(vol, min_periods=vol // 4).std().shift(1)
    z = (r / sd).to_numpy()
    rbp = r.to_numpy() * 1e4
    valid = np.zeros(len(mid), dtype=bool)
    for a, b in gaps(d):
        lo = a + win + vol // 4
        if b - lo > HOR:
            valid[lo:b - HOR + 1] = True      # 관찰 창이 통째로 들어와야 한다
    hit = np.flatnonzero(valid & np.isfinite(z) & (z <= -k)
                         & np.isfinite(rbp) & (rbp <= -minbp))
    keep, last = [], -10**9
    for i in hit:
        if i - last < dedup:
            continue
        last = i
        keep.append(i)
    return np.array(keep, dtype=int), z


def paths(syms, k, minbp=0.0, cache={}):
    """사건별 경로와 방아쇠 시점 상태를 모은다."""
    rows, P = [], []
    for s in syms:
        if s in cache:
            d = cache[s]
        else:
            try:
                d = load(s)
            except FileNotFoundError:
                continue
            cache[s] = d
        mid = d["mid"].to_numpy(dtype=np.float64)
        mid = pd.Series(mid).ffill(limit=5).to_numpy()   # 짧은 결측만 메운다
        idx, z = events(d, k, minbp=minbp)
        if not len(idx):
            continue
        bd = {c: d[c].to_numpy(dtype=np.float64) if c in d else None
              for c in ("bid_b0_5", "bid_b1", "bid_b2", "ask_b1",
                        "dbid_b0_5", "dbid_b1", "dask_b1", "oi_usd")}
        for i in idx:
            seg = mid[i:i + HOR + 1]
            if not np.isfinite(seg).all():
                continue
            p0 = seg[0]
            rel = (seg / p0 - 1.0) * 1e4                  # bp, 방아쇠 시점 기준
            tb = int(np.argmin(rel))                      # 바닥까지 초
            # 방아쇠 시점 상태 (전부 과거 정보)
            def q(c, back=300):
                v = bd[c]
                if v is None or not np.isfinite(v[i]):
                    return np.nan
                base = np.nanmedian(v[max(0, i - back):i])
                return v[i] / base if base and np.isfinite(base) and base > 0 else np.nan
            def f(c, back=60):
                v = bd[c]
                if v is None:
                    return np.nan
                w = v[max(0, i - back):i]
                w = w[np.isfinite(w)]
                return float(w.sum()) if len(w) else np.nan
            oi = bd["oi_usd"]
            doi = np.nan
            if oi is not None and i >= 300 and np.isfinite(oi[i]) and np.isfinite(oi[i - 300]) and oi[i - 300] > 0:
                doi = oi[i] / oi[i - 300] - 1.0
            rows.append({"symbol": s, "sec": int(i), "z": float(z[i]),
                         "t_bot": tb, "bot_bp": float(rel[tb]),
                         "ldep": q("bid_b1"), "ldep05": q("bid_b0_5"),
                         "limb": (q("bid_b1") / q("ask_b1")
                                  if np.isfinite(q("ask_b1")) and q("ask_b1") else np.nan),
                         "fbid": f("dbid_b1"), "fask": f("dask_b1"),
                         "doi5m": doi})
            P.append(rel)
    return pd.DataFrame(rows), (np.array(P) if P else np.zeros((0, HOR + 1)))


def boot(x, reps=4000, seed=3):
    """사건 블록 부트스트랩. 일 클러스터는 3~4일뿐이라 쓸 수 없다."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    b = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sec(n, t):
    print("\n" + "-" * W)
    print("%d. %s" % (n, t))
    print("-" * W)


def main() -> int:
    ap = argparse.ArgumentParser(description="1-second microstructure of cascades")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--minbp", type=float, default=0.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * W)
    print("1초 미시구조 — 바닥의 시각·깊이, 진입 지연의 비용, 짧은 보유")
    print("=" * W)
    print("방아쇠: z60 <= -%.0f, 매 초 평가, 60초 중복제거 | 관찰 창 %d초 | 왕복 %.0fbp"
          % (a.k, HOR, a.cost))
    print("청산: **시간 청산만** (가격 손절 없음 — DESIGN_LOCK §5.17.2)")

    sec(0, "★ 방아쇠 격자 — z 만으로는 '작지만 통계적으로 극단인' 사건을 고른다")
    print("  왕복 %.0fbp 를 넘으려면 바닥 깊이가 그보다 커야 한다. 절대 크기(60초 하락)")
    print("  하한을 같이 걸어 본다. 칸: n | 바닥깊이 중앙bp | L=0·보유60초 손익bp\n" % ())
    MB = (0, 25, 50, 100, 200)
    print("  %-6s | %s" % ("K", " ".join("%-22s" % ("60초하락>=%dbp" % b) for b in MB)))
    for kk in (3, 4, 5, 6, 8):
        out = []
        for b in MB:
            dd, PP = paths(syms, float(kk), minbp=float(b))
            if len(dd) < 5:
                out.append("%-22s" % ("n=%d" % len(dd)))
                continue
            r = (PP[:, 60] - PP[:, 0]) - a.cost
            out.append("%-22s" % ("%4d | %5.0f | %6.1f"
                                  % (len(dd), np.median(dd["bot_bp"]), r.mean())))
        print("  K=%-4d | %s" % (kk, " ".join(out)))
    print("\n  ** 바닥 깊이 중앙이 왕복 %.0fbp 를 못 넘는 칸은 원리상 불가능하다. **"
          % a.cost)

    d, P = paths(syms, a.k, minbp=a.minbp)
    if not len(d):
        print("\n사건 0건")
        return 1
    print("\n" + "=" * W)
    print("이하 상세: K=%.0f, 60초 하락 >= %.0fbp | 사건 %d건 / 심볼 %d종"
          % (a.k, a.minbp, len(d), d["symbol"].nunique()))
    print("=" * W)

    sec(1, "바닥은 언제, 얼마나 깊은가 (방아쇠 시점 = 0초, 0bp)")
    print("  %-14s | %s" % ("", " ".join("%7s" % ("p%g" % q) for q in
                                         (5, 25, 50, 75, 95))))
    for lab, v in (("바닥까지(초)", d["t_bot"]), ("바닥 깊이(bp)", d["bot_bp"])):
        print("  %-14s | %s" % (lab, " ".join("%7.0f" % np.percentile(v, q)
                                              for q in (5, 25, 50, 75, 95))))
    q = d["t_bot"].to_numpy()
    for cut in (5, 10, 30, 60, 120, 300):
        print("  바닥이 %3d초 안에 온 사건: %5.1f%%" % (cut, 100 * (q <= cut).mean()))
    print("\n  평균 경로 (bp, 방아쇠 대비):")
    print("  %s" % " ".join("%4ds" % t for t in (5, 10, 15, 30, 60, 120, 300, 600)))
    print("  %s" % " ".join("%5.0f" % np.nanmean(P[:, t])
                            for t in (5, 10, 15, 30, 60, 120, 300, 600)))
    print("  중앙 경로:")
    print("  %s" % " ".join("%5.0f" % np.nanmedian(P[:, t])
                            for t in (5, 10, 15, 30, 60, 120, 300, 600)))

    sec(2, "★ 진입 지연의 비용 — L=60초가 지금의 1분봉 판이다")
    print("  진입 = 방아쇠 + L 초의 mid, 청산 = 진입 + H 초의 mid, 왕복 %.0fbp 차감\n"
          % a.cost)
    Hs = (15, 30, 60, 120, 300)
    print("  %-8s | %s" % ("지연 L", " ".join("%14s" % ("보유 %ds" % h) for h in Hs)))
    for L in (0, 1, 2, 5, 10, 20, 30, 45, 60):
        out = []
        for h in Hs:
            if L + h > HOR:
                out.append("%14s" % "-")
                continue
            r = (P[:, L + h] - P[:, L]) - a.cost
            m, lo, hi = boot(r)
            out.append("%14s" % ("%6.0f [%3.0f,%3.0f]" % (m, lo, hi)))
        print("  L=%-6d | %s" % (L, " ".join(out)))
    print("\n  ** 대괄호는 사건 부트스트랩 95%% 구간. n=%d 이므로 넓다. **" % len(d))
    print("  ** L=0 과 L=60 의 차이가 봉 격자가 버리는 금액이다. **")

    sec(3, "보유 시간 — 짧게 먹고 나오는 게 되는가 (L=0 고정)")
    print("  %-10s | %9s %9s %9s | %7s" % ("보유", "평균bp", "CI하한", "CI상한", "승률"))
    for h in (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600):
        r = (P[:, h] - P[:, 0]) - a.cost
        m, lo, hi = boot(r)
        print("  %-10s | %9.0f %9.0f %9.0f | %6.1f%%"
              % ("%d초" % h, m, lo, hi, 100 * np.mean(r > 0)))

    sec(4, "②③ — 방아쇠 시점의 깊이·유량이 남은 밀림을 맞히는가")
    print("  목표: 방아쇠 이후 **남은 하락** (= 바닥 깊이, bp). 음수가 클수록 더 밀린다.")
    print("  설명변수는 전부 방아쇠 시점까지의 정보다.\n")
    y = d["bot_bp"].to_numpy()
    feats = [("z", "방아쇠 강도"), ("ldep", "매수깊이 b1 / 과거300초 중앙"),
             ("ldep05", "매수깊이 b0.5 / 중앙"), ("limb", "매수/매도 깊이비"),
             ("fbid", "매수밴드 유량 60초 합"), ("fask", "매도밴드 유량 60초 합"),
             ("doi5m", "OI 300초 변화")]
    print("  %-28s | %6s %9s %7s %7s" % ("변수", "n", "계수", "t", "단순R2"))
    base = np.column_stack([np.ones(len(d)), d["z"].to_numpy()])
    for c, nm in feats:
        x = d[c].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 15:
            print("  %-28s | %6d %9s" % (nm, int(m.sum()), "(표본부족)"))
            continue
        xs = (x[m] - np.nanmean(x[m])) / (np.nanstd(x[m]) + 1e-12)
        X = np.column_stack([np.ones(m.sum()), xs])
        b, se, _ = ols_cluster(X, y[m], d["symbol"].to_numpy()[m])
        r2 = 1 - np.var(y[m] - X @ b) / np.var(y[m])
        print("  %-28s | %6d %9.1f %7.1f %7.3f"
              % (nm, int(m.sum()), b[1], b[1] / se[1] if se[1] > 0 else np.nan, r2))
    print("\n  ** 심볼 클러스터 SE. 계수는 표준화(1 표준편차당 bp) **")

    print("\n  ** z 를 통제한 뒤의 증분 — 깊이가 z 와 중복이면 여기서 0 이 된다 **")
    print("  %-28s | %6s %9s %7s %9s" % ("변수 (z 통제)", "n", "계수", "t", "증분R2"))
    for c, nm in feats[1:]:
        x = d[c].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(d["z"].to_numpy())
        if m.sum() < 15:
            continue
        xs = (x[m] - np.nanmean(x[m])) / (np.nanstd(x[m]) + 1e-12)
        X0 = base[m]
        X1 = np.column_stack([X0, xs])
        b0, _, _ = ols_cluster(X0, y[m], d["symbol"].to_numpy()[m])
        b1, se1, _ = ols_cluster(X1, y[m], d["symbol"].to_numpy()[m])
        r0 = 1 - np.var(y[m] - X0 @ b0) / np.var(y[m])
        r1 = 1 - np.var(y[m] - X1 @ b1) / np.var(y[m])
        print("  %-28s | %6d %9.1f %7.1f %9.3f"
              % (nm, int(m.sum()), b1[2], b1[2] / se1[2] if se1[2] > 0 else np.nan,
                 r1 - r0))

    sec(5, "심볼별 — 한두 종이 만든 것인가")
    print("  %-10s | %5s %9s %9s %9s" % ("심볼", "n", "바닥초중앙", "바닥bp중앙", "60초보유bp"))
    for s, g in d.groupby("symbol"):
        ii = g.index.to_numpy()
        r60 = (P[ii, 60] - P[ii, 0]) - a.cost
        print("  %-10s | %5d %9.0f %9.0f %9.0f"
              % (s, len(g), g["t_bot"].median(), g["bot_bp"].median(), r60.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
