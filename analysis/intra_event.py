# -*- coding: utf-8 -*-
"""사건이 **일어나는 순간** 어디까지 밀릴지 — 초기 충격으로 총 밀림을 추정한다.

무엇이 빠져 있었나
  지금까지 X 를 `open[i+1]` (5분 방아쇠봉이 **끝난 뒤**) 부터 쟀다. 그러면
  캐스케이드가 시작되고 5~10분 지난 시점이고, 그때는 이미 상당부분이 끝나 있다.
  그리고 그 시점에는 **방향 정보가 없다**(sigma_model.py: 청산방향 대 반대방향 = 0.96).

  사건이 **일어나는 순간**은 다르다. 그 순간에는
    (1) 방향이 확정돼 있다 — 지금 밀리고 있는 쪽이다.
    (2) **초기 충격 크기 r0 을 관측할 수 있다.** 이것이 안 써본 설명변수다.
  분기과정에서 총 크기는 초기 크기에 비례한다(S = S0/(1-n)). 가격도 같은 구조라면
  r0 이 총 밀림 거리를 예측해야 한다. **그것이 이 스크립트의 검정이다.**

설계
  1분봉으로 내려간다. 5분 방아쇠봉 **안에서** 처음으로 |z_1m| >= ZK 인 봉을 찾고,
  그 **종가**를 진입 기준으로 삼는다(그 봉은 완결됐으므로 관측 가능 = 룩어헤드 없음).
    r0 = 그 1분봉의 수익(방향 확정)
    X  = 그 종가 이후 **같은 방향으로** 추가로 밀리는 최대 거리(bp)
  회귀: log X = a + b1 log sigma + b2 log|r0| (+ b3 log(Q/D))
  b2 > 0 이고 유의하면 **초기 충격으로 총 밀림을 추정할 수 있다.**

  추가로 '탐지 시점에 이미 얼마나 끝났는가' 를 낸다. 이것이 크면 늦은 것이다.

실행:
    python analysis/intra_event.py
    python analysis/intra_event.py --zk 3
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
from analysis.response_liq import ols_cluster, cmean            # noqa: E402
from analysis.scale_check import K, DOI_THR, MIN_GAP, VOL_WIN   # noqa: E402
from analysis.synth import kupiec, ks_unif, pinball, LEVELS     # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
FEE_M, FEE_T = 2.0, 5.0


def build(symbols, window, zk):
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, K, DOI_THR, MIN_GAP)
        if len(ev) == 0:
            continue
        p1 = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p1):
            continue
        k = pd.read_parquet(p1, columns=["open_time", "open", "high", "low", "close"])
        k = k.sort_values("open_time").reset_index(drop=True)
        ot1 = k["open_time"].to_numpy()
        O, H, L, Cl = (k[c].to_numpy(dtype=np.float64)
                       for c in ("open", "high", "low", "close"))
        n1 = len(ot1)
        # 1분 수익과 그 과거 변동성 (현재 봉 제외)
        r1 = np.concatenate([[np.nan], Cl[1:] / Cl[:-1] - 1.0])
        s1 = pd.Series(r1).shift(1).rolling(1440, min_periods=360).std().to_numpy()
        ot5 = df["open_time"].to_numpy()
        oiv = df["sum_open_interest_value"].to_numpy(dtype=np.float64)
        doi = df["doi"].to_numpy(dtype=np.float64)
        qv = df["quote_volume"].to_numpy(dtype=np.float64)
        adv = (pd.Series(qv).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .mean().to_numpy()) * float(VOL_WIN)
        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5) or not np.isfinite(doi[i]) or doi[i] >= 0:
                continue
            if not (np.isfinite(adv[i]) and adv[i] > 0 and oiv[i] > 0):
                continue
            b0 = int(np.searchsorted(ot1, int(ot5[i])))      # 방아쇠봉 첫 1분봉
            if b0 <= 1440 or b0 + window + 6 >= n1:
                continue
            if ot1[b0] != int(ot5[i]):
                continue
            # 방아쇠 5분봉 안에서 처음으로 |z| >= zk 이고 사건 방향인 1분봉
            j = -1
            for t in range(b0, b0 + 5):
                if not (np.isfinite(s1[t]) and s1[t] > 0):
                    continue
                z = r1[t] / s1[t]
                if np.isfinite(z) and (-z * sd) >= zk:       # sd=+1 이면 하락
                    j = t
                    break
            if j < 0:
                continue
            p_ent = Cl[j]
            if not (np.isfinite(p_ent) and p_ent > 0):
                continue
            r0 = abs(r1[j]) * 1e4                            # 초기 충격 크기(bp)
            seg_lo, seg_hi = L[j + 1:j + 1 + window], H[j + 1:j + 1 + window]
            if len(seg_lo) < window:
                continue
            x = ((p_ent - seg_lo.min()) / p_ent if sd == 1
                 else (seg_hi.max() - p_ent) / p_ent) * 1e4
            # 탐지 전에 이미 끝난 몫: 방아쇠봉 시가 -> 진입가 이동
            pre = abs(Cl[j] / O[b0] - 1.0) * 1e4
            out.append({"symbol": s, "side": sd, "t0": int(ot1[j]),
                        "day": int(ot1[j] // 86_400_000),
                        "r0": max(r0, 1e-6), "sig": s1[j] * np.sqrt(1440.0),
                        "X": max(x, 1e-6), "pre": pre, "p_ent": p_ent,
                        "S0": -doi[i] * oiv[i], "adv": adv[i],
                        "lo": seg_lo.copy(), "hi": seg_hi.copy(),
                        "cl": Cl[j + 1:j + 1 + window].copy()})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="intra-event: predict total push from r0")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--zk", type=float, default=3.0)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--hold", type=int, default=15)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 80)
    print("사건이 일어나는 **순간** — 초기 충격 r0 으로 총 밀림 X 를 추정한다")
    print("=" * 80)
    d = build(syms, a.window, a.zk).sort_values("t0").reset_index(drop=True)
    if len(d) < 150:
        print("표본 부족 (%d)" % len(d))
        return 1
    print("**사용 데이터 기간: %s ~ %s / %d종 / 사건 %d건 / 1분봉 / z>=%.1f**"
          % (str(pd.to_datetime(d.t0.min(), unit="ms"))[:10],
             str(pd.to_datetime(d.t0.max(), unit="ms"))[:10],
             d.symbol.nunique(), len(d), a.zk))
    print("초기 충격 r0 중앙 %.0f bp | 추가 밀림 X 중앙 %.0f bp | **X/r0 중앙 %.2f**"
          % (d.r0.median(), d.X.median(), (d.X / d.r0).median()))
    print("탐지 전 이미 끝난 이동 중앙 %.0f bp (진입 전 놓친 몫)" % d.pre.median())

    ly = np.log(d["X"].to_numpy())
    ls, lr = np.log(d["sig"].to_numpy()), np.log(d["r0"].to_numpy())
    lq = np.log(d["S0"].to_numpy() / d["adv"].to_numpy())
    cl = d["day"].to_numpy()
    ok = np.isfinite(ly) & np.isfinite(ls) & np.isfinite(lr) & np.isfinite(lq)
    print("\n" + "-" * 80)
    print("회귀  log X = a + b1 log(sigma) + b2 **log(r0)** [+ b3 log(S0/ADV)]")
    print("-" * 80)
    print("  %-30s %8s %6s | %8s %6s | %8s %6s | %6s"
          % ("설정", "b1(sig)", "t", "**b2(r0)**", "t", "b3(Q/ADV)", "t", "R^2"))
    for lab, cols in (("sigma 만", [ls]), ("sigma + **r0**", [ls, lr]),
                      ("sigma + r0 + Q/ADV", [ls, lr, lq])):
        Xm = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
        b, se, _ = ols_cluster(Xm, ly[ok], cl[ok])
        r2 = 1.0 - np.var(ly[ok] - Xm @ b) / np.var(ly[ok])
        cells = []
        for kk in range(1, 4):
            cells.append("%8.3f %6.1f" % (b[kk], b[kk] / se[kk])
                         if kk < len(b) else "%8s %6s" % ("-", "-"))
        print("  %-30s %s | %s | %s | %6.3f" % (lab, *cells, r2))
    print("  b2 가 유의한 양수면 **초기 충격으로 총 밀림을 추정할 수 있다.**")

    # 조건부 분포 + 캘리브레이션 + P&L
    cut = int(len(d) * a.train)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    print("\n훈련 %d | 검정 %d (%s~)"
          % (len(tr), len(te), str(pd.to_datetime(te.t0.iloc[0], unit="ms"))[:10]))
    Xtr = np.column_stack([np.ones(len(tr)), np.log(tr["sig"]), np.log(tr["r0"])])
    bt = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ np.log(tr["X"].to_numpy()))
    print("훈련 적합: a=%.3f b1(sig)=%.3f **b2(r0)=%.3f**" % (bt[0], bt[1], bt[2]))
    mtr = Xtr @ bt
    Xte = np.column_stack([np.ones(len(te)), np.log(te["sig"]), np.log(te["r0"])])
    mte = Xte @ bt
    Zr = np.log(tr["X"].to_numpy()) - mtr
    Z0 = np.log(tr["X"].to_numpy())
    xte = te["X"].to_numpy()

    from math import erf
    def p1(s):
        return np.nan if not np.isfinite(s) else float(1.0 - erf(np.sqrt(max(s, 0) / 2)))
    print("\n  %-6s | %-24s | %-24s" % ("수준p", "Mr0 (sigma+r0)", "M0 (무조건부)"))
    for p in LEVELS:
        cells = []
        for m_, Z_ in ((mte, Zr), (np.zeros(len(te)), Z0)):
            q = np.exp(m_ + float(np.quantile(Z_, p)))
            v = (xte < q).astype(int)
            lr_ = kupiec(len(v), int(v.sum()), p)
            cells.append("위반%5.3f LR%6.1f p%.3f" % (v.mean(), lr_, p1(lr_)))
        print("  %-6.2f | %-24s | %-24s" % (p, *cells))
    for lab, m_, Z_ in (("Mr0", mte, Zr), ("M0", np.zeros(len(te)), Z0)):
        u = np.array([float((Z_ <= v).mean())
                      for v in (np.log(xte) - m_)])
        ks, pv = ks_unif(np.clip(u, 1e-6, 1 - 1e-6))
        print("  %-4s PIT KS D=%.4f p=%.4f  %s"
              % (lab, ks, pv, "합격" if pv > 0.05 else "**불합격**"))
    t1 = t0_ = 0.0
    for p in LEVELS:
        t1 += pinball(xte, np.exp(mte + float(np.quantile(Zr, p))), p)
        t0_ += pinball(xte, np.exp(float(np.quantile(Z0, p))) * np.ones(len(te)), p)
    print("  핀볼 합계  Mr0 %.1f | M0 %.1f  -> **개선 %.1f%%**"
          % (t1, t0_, 100 * (t0_ - t1) / t0_))

    print("\n" + "-" * 80)
    print("지정가 배치 → P&L (보유 %d분, 왕복 %.0fbp)" % (a.hold, FEE_M + FEE_T))
    print("-" * 80)
    LO, HI, CL = te["lo"].tolist(), te["hi"].tolist(), te["cl"].tolist()
    PE, SD = te["p_ent"].to_numpy(), te["side"].to_numpy()

    def pnl(dep):
        rs, f = [], 0
        for i in range(len(te)):
            sd_, p_ = int(SD[i]), float(PE[i])
            lim = p_ * (1.0 - sd_ * dep[i] / 1e4)
            hit = (LO[i] <= lim) if sd_ == 1 else (HI[i] >= lim)
            if not hit.any():
                rs.append(0.0)
                continue
            fb = int(np.argmax(hit))
            ex = min(fb + a.hold, len(CL[i]) - 1)
            rs.append((CL[i][ex] / lim - 1.0) * sd_ * 1e4 - FEE_M - FEE_T)
            f += 1
        return np.array(rs), f

    rows = []
    for p in (0.10, 0.25, 0.50, 0.75):
        dep = np.exp(mte + float(np.quantile(Zr, p)))
        r, f = pnl(dep)
        m, se, t, _ = cmean(r, te["day"].to_numpy())
        rows.append((m, "모형 q%.2f (중앙%4.0fbp)" % (p, np.median(dep)),
                     f / len(te), t))
    for dep0 in (0.0, 50.0, 100.0, 200.0):
        r, f = pnl(np.full(len(te), dep0))
        m, se, t, _ = cmean(r, te["day"].to_numpy())
        rows.append((m, "고정 %.0fbp" % dep0, f / len(te), t))
    print("  %-26s %8s %10s %7s" % ("방식", "체결률", "이벤트당", "t"))
    for m, lab, fr, t in sorted(rows, key=lambda x: -x[0]):
        print("  %-26s %7.1f%% %10.1f %7.1f" % (lab, 100 * fr, m, t))
    print("\n  *** 검정 %d건. 순위를 보되 크기는 조심." % len(te))
    return 0


if __name__ == "__main__":
    sys.exit(main())
