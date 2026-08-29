# -*- coding: utf-8 -*-
"""방향(확정) + 변동성(확률)만으로 바닥 분포를 추정한다. Q/D 없이.

논리
  V/D 는 밀림 거리를 설명하지 못했다(vd_cascade.py: b2 = 0.05~0.11, t<1,
  캐스케이드 규모에서도). 그러나 두 가지는 남아 있다:
    (1) **방향은 기계적으로 확정된다** — 롱청산 = 강제매도 = 하방.
    (2) **크기는 변동성이 설명한다** — b1 = 0.87, t = 4.8~6.0, R^2 0.12~0.16.
  방향 + 크기 분포 = **바닥의 확률 추정**. 설계가 요구한 산출물이 그것이다.

왜 이걸 따로 해야 하나
  synth.py 는 m_t = log(sigma) + 0.5*log(S0/ADV) 로 **두 항을 묶어** 넣었고
  M0(무조건부)에 졌다. 원인은 S0/ADV 항의 진짜 지수가 0인데 0.5 를 강제해
  **순수 잡음을 주입**한 것이었다. 잡음항을 빼고 sigma 만 쓰는 모형은
  **한 번도 검정되지 않았다.** 그것을 여기서 한다.

모형 3개 (동일한 검정 통과 기준)
  M0  : log X = Z0                       무조건부
  M1  : log X = log sig + 0.5 log(S0/ADV) + Z     (synth.py, 실패한 판)
  Msig: log X = a + b*log sig + Z         **sigma 만, b 는 훈련구간 적합**
  판정: Kupiec / PIT KS / 삼분위 커버리지 / 핀볼 손실 + 지정가 P&L

실행:
    python analysis/sigma_model.py
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
from analysis.response_liq import cmean                         # noqa: E402
from analysis.synth import build, kupiec, ks_unif, pinball, LEVELS   # noqa: E402

FEE_M, FEE_T = 2.0, 5.0


def main() -> int:
    ap = argparse.ArgumentParser(description="direction + volatility only model")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--hold", type=int, default=15)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 80)
    print("방향(확정) + 변동성(확률) 만으로 바닥 분포 — Q/D 없이")
    print("=" * 80)
    d, _ = build(syms, a.window)
    d = d.sort_values("t0").reset_index(drop=True)
    print("**사용 데이터 기간: %s ~ %s / %d종 / 이벤트 %d건 / 창 %d분**"
          % (str(pd.to_datetime(d.t0.min(), unit="ms"))[:10],
             str(pd.to_datetime(d.t0.max(), unit="ms"))[:10],
             d.symbol.nunique(), len(d), a.window))

    # --- 방향 확인: 청산 방향으로 실제로 더 크게 움직이는가
    adv_, fav = [], []
    for r in d.itertuples():
        p0 = r.p0
        if r.side == 1:
            adv_.append((p0 - r.lo.min()) / p0 * 1e4)
            fav.append((r.hi.max() - p0) / p0 * 1e4)
        else:
            adv_.append((r.hi.max() - p0) / p0 * 1e4)
            fav.append((p0 - r.lo.min()) / p0 * 1e4)
    adv_, fav = np.array(adv_), np.array(fav)
    print("\n[방향 확인] 창 안 최대 이동, 청산 방향 vs 반대 방향")
    print("  청산 방향 중앙 %.0f bp | 반대 방향 중앙 %.0f bp | 비 %.2f"
          % (np.median(adv_), np.median(fav), np.median(adv_) / np.median(fav)))
    print("  청산 방향이 더 큰 사건 비율 %.1f%%" % (100 * (adv_ > fav).mean()))
    print("  -> 1.0 근처면 '방향을 안다' 는 것도 성립하지 않는다.")

    d["lx"] = np.log(np.maximum(d["X"], 1e-6))
    d["ls"] = np.log(d["sig"])
    d["lq"] = np.log(d["S0"] / d["adv"])
    cut = int(len(d) * a.train)
    tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
    print("\n훈련 %d (~%s) | 검정 %d (%s~)"
          % (len(tr), str(pd.to_datetime(tr.t0.iloc[-1], unit="ms"))[:10],
             len(te), str(pd.to_datetime(te.t0.iloc[0], unit="ms"))[:10]))

    # Msig 계수 훈련 적합
    Xs = np.column_stack([np.ones(len(tr)), tr["ls"].to_numpy()])
    bs = np.linalg.pinv(Xs.T @ Xs) @ (Xs.T @ tr["lx"].to_numpy())
    print("Msig 훈련 적합: a=%.3f, **b(log sigma)=%.3f**" % (bs[0], bs[1]))

    MODELS = {
        "M0": (np.zeros(len(tr)), np.zeros(len(te))),
        "M1": (tr["ls"] + 0.5 * tr["lq"], te["ls"] + 0.5 * te["lq"]),
        "Msig": (bs[0] + bs[1] * tr["ls"], bs[0] + bs[1] * te["ls"]),
    }
    Z = {k: (tr["lx"].to_numpy() - np.asarray(v[0], dtype=float))
         for k, v in MODELS.items()}
    MT = {k: np.asarray(v[1], dtype=float) for k, v in MODELS.items()}
    xte = te["X"].to_numpy()

    print("\n" + "-" * 80)
    print("검정 1. Kupiec 무조건부 커버리지 (검정 %d건)" % len(te))
    print("-" * 80)
    print("  %-6s | " % "수준p" + " | ".join("%-18s" % m for m in MODELS))
    from math import erf
    def p1(s):
        return np.nan if not np.isfinite(s) else float(1.0 - erf(np.sqrt(max(s, 0) / 2)))
    for p in LEVELS:
        cells = []
        for k in MODELS:
            q = np.exp(MT[k] + float(np.quantile(Z[k], p)))
            v = (xte < q).astype(int)
            lr = kupiec(len(v), int(v.sum()), p)
            cells.append("%5.3f LR%5.1f p%.3f" % (v.mean(), lr, p1(lr)))
        print("  %-6.2f | " % p + " | ".join(cells))
    print("  기대 위반율 = p. p값 0.05 미만이면 불합격.")

    print("\n" + "-" * 80)
    print("검정 2. PIT 균등성 / 검정 3. 핀볼 손실")
    print("-" * 80)
    for k in MODELS:
        u = np.array([float((Z[k] <= v).mean())
                      for v in (te["lx"].to_numpy() - MT[k])])
        ks, pv = ks_unif(np.clip(u, 1e-6, 1 - 1e-6))
        print("  %-5s PIT KS D=%.4f p=%.4f  %s"
              % (k, ks, pv, "합격" if pv > 0.05 else "**불합격**"))
    print("\n  %-6s | %10s %10s %10s | %s" % ("수준p", "M0", "M1", "Msig", "Msig 개선%"))
    tot = {k: 0.0 for k in MODELS}
    for p in LEVELS:
        row = {}
        for k in MODELS:
            q = np.exp(MT[k] + float(np.quantile(Z[k], p)))
            row[k] = pinball(xte, q, p)
            tot[k] += row[k]
        print("  %-6.2f | %10.1f %10.1f %10.1f | %9.1f"
              % (p, row["M0"], row["M1"], row["Msig"],
                 100 * (row["M0"] - row["Msig"]) / row["M0"]))
    print("  %-6s | %10.1f %10.1f %10.1f | %9.1f"
          % ("합계", tot["M0"], tot["M1"], tot["Msig"],
             100 * (tot["M0"] - tot["Msig"]) / tot["M0"]))
    print("  Msig 개선%가 양수면 **변동성 조건부가 무조건부를 이긴 것**이다.")

    print("\n" + "-" * 80)
    print("검정 4. 삼분위 커버리지 — 조건부가 실제로 작동하는가")
    print("-" * 80)
    ter = pd.qcut(te["ls"], 3, labels=False, duplicates="drop").to_numpy()
    for p in (0.25, 0.50, 0.75):
        cells = []
        for k in MODELS:
            q = np.exp(MT[k] + float(np.quantile(Z[k], p)))
            r = [float((xte[ter == g] < q[ter == g]).mean()) for g in (0, 1, 2)]
            cells.append("%s 폭%.3f" % (" ".join("%.2f" % v for v in r),
                                       max(r) - min(r)))
        print("  p=%.2f | " % p + " | ".join("%-6s %s" % (k, c)
                                             for k, c in zip(MODELS, cells)))
    print("  폭이 작을수록 좋다.")

    print("\n" + "-" * 80)
    print("검정 5. 지정가 배치 → P&L (보유 %d분, 왕복 %.0fbp)" % (a.hold, FEE_M + FEE_T))
    print("-" * 80)
    LOs, HIs, CLs = te["lo"].tolist(), te["hi"].tolist(), te["cl"].tolist()
    P0, SD = te["p0"].to_numpy(), te["side"].to_numpy()

    def pnl(dep):
        rs, f = [], 0
        for i in range(len(te)):
            sd_, p0_ = int(SD[i]), float(P0[i])
            lim = p0_ * (1.0 - sd_ * dep[i] / 1e4)
            hit = (LOs[i] <= lim) if sd_ == 1 else (HIs[i] >= lim)
            if not hit.any():
                rs.append(0.0)
                continue
            fb = int(np.argmax(hit))
            ex = min(fb + a.hold, len(CLs[i]) - 1)
            rs.append((CLs[i][ex] / lim - 1.0) * sd_ * 1e4 - FEE_M - FEE_T)
            f += 1
        return np.array(rs), f

    print("  %-26s %8s %10s %7s %10s"
          % ("방식", "체결률", "이벤트당", "t", "조건부평균"))
    rows = []
    for k in MODELS:
        for p in (0.10, 0.25, 0.50):
            dep = np.exp(MT[k] + float(np.quantile(Z[k], p)))
            r, f = pnl(dep)
            m, se, t, _ = cmean(r, te["day"].to_numpy())
            rows.append((m, "%s q%.2f (중앙%3.0fbp)" % (k, p, np.median(dep)),
                         f / len(te), t, r[r != 0].mean() if f else np.nan))
    for dep0 in (0.0, 25.0, 50.0, 100.0):
        r, f = pnl(np.full(len(te), dep0))
        m, se, t, _ = cmean(r, te["day"].to_numpy())
        rows.append((m, "고정 %.0fbp" % dep0, f / len(te), t,
                     r[r != 0].mean() if f else np.nan))
    for m, lab, fr, t, cm in sorted(rows, key=lambda x: -x[0]):
        print("  %-26s %7.1f%% %10.1f %7.1f %10.1f" % (lab, 100 * fr, m, t, cm))
    print("\n  *** 검정구간 %d건이라 t 가 전반적으로 낮다. 순위를 보되 크기는 조심."
          % len(te))
    return 0


if __name__ == "__main__":
    sys.exit(main())
