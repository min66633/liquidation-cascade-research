# -*- coding: utf-8 -*-
"""실시간 방아쇠 — 진입 시점에 **관측 가능한 것만으로** 사건을 정의한다.

왜 다시 하나
  intra_event.py 는 사건 순간 진입에서 모형 지정가 +90.4bp (t=3.2) 를 냈다.
  지금까지 최고 성적이다. 그런데 **사건 선별**에 5분봉 |z|>=8 과 5분 OI 급감을
  썼고, 둘 다 그 5분봉이 **끝나야** 안다. 진입은 그 안의 1분봉이므로 룩어헤드다.

  주의: '캐스케이드인 줄 안다' 자체가 룩어헤드인 것은 아니다. 설계에서 그 지식의
  정당한 출처는 **청산맵**이다 — OI 가 쌓인 가격대에 도달하면 격리 청산이 시작되고
  그것은 실시간으로 보인다. 문제는 내가 지도 대신 **실현 OI 급감(5분 지연)** 을
  대용으로 썼다는 점이다.

  지도가 아직 없으므로, 여기서는 **1분봉 가격 정보만으로** 방아쇠를 정의한다.
  이것이 지도 없이 가능한 하한이다. 지도가 생기면 이 위에 얹으면 된다.

방아쇠 (전부 그 1분봉 **종가 시점에 관측 가능**)
  z = r_1m / sigma_1m,   sigma 는 과거 1440봉(현재 봉 제외) 표준편차
  |z| >= ZK 이면 방아쇠. 방향은 그 봉의 부호. OI·5분봉 확인 **없음**.
  같은 심볼·같은 방향으로 GAP 분 안의 중복은 첫 건만 남긴다.

모형 (intra_event.py 와 동일 형태, 훈련구간에서만 적합)
  log X = a + b1 log(sigma) + b2 log(r0)
  지정가를 q_p 분위수에 걸고 보유 후 청산. 고정 깊이와 비교.

실행:
    python analysis/realtime_trigger.py
    python analysis/realtime_trigger.py --zk 6 --gap 120
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
from analysis.response_liq import ols_cluster, cmean            # noqa: E402
from analysis.synth import kupiec, ks_unif, pinball, LEVELS     # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
FEE_M, FEE_T = 2.0, 5.0
VOL_WIN = 1440


def build(symbols, zk, gap, window):
    out = []
    for s in symbols:
        p = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p):
            continue
        k = pd.read_parquet(p, columns=["open_time", "high", "low", "close"])
        k = k.sort_values("open_time").reset_index(drop=True)
        ot = k["open_time"].to_numpy()
        H = k["high"].to_numpy(dtype=np.float64)
        L = k["low"].to_numpy(dtype=np.float64)
        Cl = k["close"].to_numpy(dtype=np.float64)
        n = len(Cl)
        if n < VOL_WIN * 2 + window:
            continue
        r = np.concatenate([[np.nan], Cl[1:] / Cl[:-1] - 1.0])
        # **현재 봉 제외** — shift(1) 로 룩어헤드 차단
        sg = pd.Series(r).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 3
                                           ).std().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = r / sg
        # 봉 연속성: 1분 격자에서 벗어난 구간은 버린다
        cont = np.concatenate([[False], np.diff(ot) == 60_000])
        for sd in (1, -1):                    # +1 = 하락(매수), -1 = 상승(매도)
            hit = np.where(np.isfinite(z) & ((-z * sd) >= zk) & cont)[0]
            last = -10**9
            for j in hit:
                if j - last < gap:
                    continue
                if j + window >= n or j <= VOL_WIN:
                    continue
                if not cont[j + 1:j + window].all():
                    continue
                p_ent = Cl[j]
                if not (np.isfinite(p_ent) and p_ent > 0 and sg[j] > 0):
                    continue
                last = j
                lo, hi, cl_ = (L[j + 1:j + 1 + window], H[j + 1:j + 1 + window],
                               Cl[j + 1:j + 1 + window])
                x = ((p_ent - lo.min()) / p_ent if sd == 1
                     else (hi.max() - p_ent) / p_ent) * 1e4
                out.append({"symbol": s, "side": sd, "t0": int(ot[j]),
                            "day": int(ot[j] // 86_400_000),
                            "r0": max(abs(r[j]) * 1e4, 1e-6),
                            "sig": sg[j] * np.sqrt(float(VOL_WIN)),
                            "X": max(x, 1e-6),
                            "lo": lo.astype(np.float32),
                            "hi": hi.astype(np.float32),
                            "cl": cl_.astype(np.float32), "p": p_ent})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="fully observable real-time trigger")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--zk", type=float, default=5.0)
    ap.add_argument("--gap", type=int, default=60, help="같은 방향 재방아쇠 최소 간격(분)")
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--hold", type=int, default=15)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 80)
    print("실시간 방아쇠 — 1분봉 |z| >= %.1f 만으로. OI·5분봉 확인 **없음**" % a.zk)
    print("=" * 80)
    d = build(syms, a.zk, a.gap, a.window).sort_values("t0").reset_index(drop=True)
    if len(d) < 300:
        print("표본 부족 (%d)" % len(d))
        return 1
    print("**사용 데이터 기간: %s ~ %s / %d종 / 방아쇠 %d건 / 간격 %d분**"
          % (str(pd.to_datetime(d.t0.min(), unit="ms"))[:10],
             str(pd.to_datetime(d.t0.max(), unit="ms"))[:10],
             d.symbol.nunique(), len(d), a.gap))
    print("  하락 방아쇠 %d | 상승 방아쇠 %d" % ((d.side == 1).sum(), (d.side == -1).sum()))
    print("초기 충격 r0 중앙 %.0f bp | 추가 밀림 X 중앙 %.0f bp | X/r0 중앙 %.2f"
          % (d.r0.median(), d.X.median(), (d.X / d.r0).median()))
    print("  대조 — OI 확인 있던 판(intra_event): r0 84bp, X 425bp, 비 5.30")

    ly = np.log(d["X"].to_numpy())
    ls, lr = np.log(d["sig"].to_numpy()), np.log(d["r0"].to_numpy())
    cl = d["day"].to_numpy()
    ok = np.isfinite(ly) & np.isfinite(ls) & np.isfinite(lr)
    print("\n" + "-" * 80)
    print("회귀  log X = a + b1 log(sigma) + b2 log(r0)   [일클러스터 CR1]")
    print("-" * 80)
    print("  %-20s %9s %6s | %9s %6s | %6s" % ("설정", "b1(sig)", "t", "b2(r0)", "t", "R^2"))
    for lab, cols in (("sigma 만", [ls]), ("sigma + r0", [ls, lr])):
        Xm = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
        b, se, _ = ols_cluster(Xm, ly[ok], cl[ok])
        r2 = 1.0 - np.var(ly[ok] - Xm @ b) / np.var(ly[ok])
        cells = [("%9.3f %6.1f" % (b[i], b[i] / se[i])) if i < len(b)
                 else "%9s %6s" % ("-", "-") for i in (1, 2)]
        print("  %-20s %s | %s | %6.3f" % (lab, *cells, r2))

    cut = int(len(d) * a.train)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    Xtr = np.column_stack([np.ones(len(tr)), np.log(tr["sig"]), np.log(tr["r0"])])
    bt = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ np.log(tr["X"].to_numpy()))
    mtr = Xtr @ bt
    Xte = np.column_stack([np.ones(len(te)), np.log(te["sig"]), np.log(te["r0"])])
    mte = Xte @ bt
    Zr = np.log(tr["X"].to_numpy()) - mtr
    Z0 = np.log(tr["X"].to_numpy())
    xte = te["X"].to_numpy()
    print("\n훈련 %d (~%s) | 검정 %d (%s~)"
          % (len(tr), str(pd.to_datetime(tr.t0.iloc[-1], unit="ms"))[:10],
             len(te), str(pd.to_datetime(te.t0.iloc[0], unit="ms"))[:10]))
    print("훈련 적합: a=%.3f b1(sig)=%.3f b2(r0)=%.3f" % (bt[0], bt[1], bt[2]))

    from math import erf
    def p1(v):
        return np.nan if not np.isfinite(v) else float(1.0 - erf(np.sqrt(max(v, 0) / 2)))
    print("\n  %-6s | %-24s | %-24s" % ("수준p", "Mr0", "M0 무조건부"))
    for p in LEVELS:
        cells = []
        for m_, Z_ in ((mte, Zr), (np.zeros(len(te)), Z0)):
            q = np.exp(m_ + float(np.quantile(Z_, p)))
            v = (xte < q).astype(int)
            lr_ = kupiec(len(v), int(v.sum()), p)
            cells.append("위반%5.3f LR%7.1f p%.3f" % (v.mean(), lr_, p1(lr_)))
        print("  %-6.2f | %-24s | %-24s" % (p, *cells))
    for lab, m_, Z_ in (("Mr0", mte, Zr), ("M0", np.zeros(len(te)), Z0)):
        u = np.array([float((Z_ <= v).mean()) for v in (np.log(xte) - m_)])
        ks, pv = ks_unif(np.clip(u, 1e-6, 1 - 1e-6))
        print("  %-4s PIT KS D=%.4f p=%.4f  %s"
              % (lab, ks, pv, "합격" if pv > 0.05 else "**불합격**"))
    t1 = t0_ = 0.0
    for p in LEVELS:
        t1 += pinball(xte, np.exp(mte + float(np.quantile(Zr, p))), p)
        t0_ += pinball(xte, np.full(len(te), np.exp(float(np.quantile(Z0, p)))), p)
    print("  핀볼 합계  Mr0 %.1f | M0 %.1f  -> **개선 %.1f%%**"
          % (t1, t0_, 100 * (t0_ - t1) / t0_))

    print("\n" + "-" * 80)
    print("지정가 배치 → P&L (보유 %d분, 왕복 %.0fbp). **전부 실시간 관측 가능**"
          % (a.hold, FEE_M + FEE_T))
    print("-" * 80)
    LO, HI, CL = te["lo"].tolist(), te["hi"].tolist(), te["cl"].tolist()
    PE, SD = te["p"].to_numpy(), te["side"].to_numpy()

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
            rs.append((float(CL[i][ex]) / lim - 1.0) * sd_ * 1e4 - FEE_M - FEE_T)
            f += 1
        return np.array(rs), f

    rows = []
    for p in (0.25, 0.50, 0.75, 0.90):
        dep = np.exp(mte + float(np.quantile(Zr, p)))
        r, f = pnl(dep)
        m, se, t, _ = cmean(r, te["day"].to_numpy())
        rows.append((m, "모형 q%.2f (중앙%4.0fbp)" % (p, np.median(dep)), f / len(te), t))
    for dep0 in (0.0, 50.0, 100.0, 200.0, 400.0):
        r, f = pnl(np.full(len(te), dep0))
        m, se, t, _ = cmean(r, te["day"].to_numpy())
        rows.append((m, "고정 %.0fbp" % dep0, f / len(te), t))
    print("  %-26s %8s %10s %7s" % ("방식", "체결률", "이벤트당", "t"))
    for m, lab, fr, t in sorted(rows, key=lambda x: -x[0]):
        print("  %-26s %7.1f%% %10.1f %7.1f" % (lab, 100 * fr, m, t))
    print("\n  *** 이 표에는 룩어헤드가 없다. 방아쇠·모형·지정가 전부 관측 가능하다.")
    print("      검정 %d건. 크기보다 부호와 순위를 볼 것." % len(te))
    return 0


if __name__ == "__main__":
    sys.exit(main())
