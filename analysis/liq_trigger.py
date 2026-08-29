# -*- coding: utf-8 -*-
"""실시간 청산 프린트로 방아쇠를 확인한다 — OI 5분 지연을 없앤 판.

무엇을 가르는가
  realtime_trigger.py: 가격 급변만으로 방아쇠 -> **확률분포는 거의 완벽하게 교정**
    (Kupiec 위반율 0.251/0.499/0.751, 핀볼 +10.7%) 되지만 **수익은 0 이하**.
  intra_event.py: 가격 급변 + **OI 급감 확인** -> 모형 지정가 +90.4bp (t=3.2).
    그런데 OI 확인은 5분봉이 끝나야 온다(룩어헤드).

  차이의 정체: X/r0 가 1.88 (가격만) vs 5.30 (OI 확인). **정보는 가격 급변이 아니라
  '격리마진이 실제로 청산되고 있다' 는 사실에 있다.**

  **청산 프린트는 그 사실을 밀리초 단위로 준다.** 그 분(minute) 안에 도착한
  프린트는 그 분의 종가 시점에 전부 관측 가능하다. 즉 **룩어헤드가 없다.**
  OI 5분 지연을 없앤 상태에서 우위가 남는지 본다.

교란 통제 (반드시)
  Tardis 무료 구간은 매월 1일뿐이다. 그 날들만 쓰면 날 구성이 6년 전체와 다르다.
  그래서 **같은 날 안에서** 가격만 방아쇠와 청산확인 방아쇠를 둘 다 만들어 비교한다.

실행:
    python analysis/liq_trigger.py
    python analysis/liq_trigger.py --zk 4 --exchange bybit
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
LIQ = os.path.join(C.DATA, "tardis_multi", "liquidations.parquet")
FEE_M, FEE_T = 2.0, 5.0
VOL_WIN = 1440


def liq_by_min(exchange: str):
    """(심볼, 분) -> 방향별 청산 명목가. 롱청산 = 강제매도 = 하방(+1)."""
    d = pd.read_parquet(LIQ)
    d = d[d["exchange"] == exchange]
    isL = d["pos_side"].to_numpy() == "long"
    ntl = d["notional"].to_numpy(dtype=np.float64)
    g = pd.DataFrame({"symbol": d["symbol"].to_numpy(),
                      "m": (d["ts_ms"].to_numpy() // 60000).astype(np.int64),
                      "qdn": np.where(isL, ntl, 0.0),
                      "qup": np.where(~isL, ntl, 0.0)})
    return g.groupby(["symbol", "m"], as_index=False).sum()


def build(symbols, zk, gap, window, exchange, minq):
    lb = liq_by_min(exchange)
    days = set(np.unique(lb["m"].to_numpy() // 1440))
    out = []
    for s in symbols:
        p = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p):
            continue
        sub = lb[lb["symbol"] == s]
        lmap = dict(zip(sub["m"].to_numpy(),
                        zip(sub["qdn"].to_numpy(), sub["qup"].to_numpy())))
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
        sg = pd.Series(r).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 3
                                           ).std().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = r / sg
        cont = np.concatenate([[False], np.diff(ot) == 60_000])
        mins = (ot // 60000).astype(np.int64)
        onday = np.isin(mins // 1440, list(days))
        for sd in (1, -1):
            hit = np.where(np.isfinite(z) & ((-z * sd) >= zk) & cont & onday)[0]
            last = -10**9
            for j in hit:
                if j - last < gap or j + window >= n or j <= VOL_WIN:
                    continue
                if not cont[j + 1:j + window].all():
                    continue
                p_ent = Cl[j]
                if not (np.isfinite(p_ent) and p_ent > 0 and sg[j] > 0):
                    continue
                last = j
                qdn, qup = lmap.get(int(mins[j]), (0.0, 0.0))
                Q = qdn if sd == 1 else qup       # 그 방향의 청산 명목가
                lo, hi = L[j + 1:j + 1 + window], H[j + 1:j + 1 + window]
                x = ((p_ent - lo.min()) / p_ent if sd == 1
                     else (hi.max() - p_ent) / p_ent) * 1e4
                out.append({"symbol": s, "side": sd, "t0": int(ot[j]),
                            "day": int(ot[j] // 86_400_000),
                            "r0": max(abs(r[j]) * 1e4, 1e-6),
                            "sig": sg[j] * np.sqrt(float(VOL_WIN)),
                            "Q": Q, "conf": int(Q > minq),
                            "X": max(x, 1e-6), "p": p_ent,
                            "lo": lo.astype(np.float32), "hi": hi.astype(np.float32),
                            "cl": Cl[j + 1:j + 1 + window].astype(np.float32)})
    return pd.DataFrame(out)


def fit_eval(d, tag, train, hold, use_q):
    """모형 적합 + 캘리브레이션 + P&L. 반환: P&L 행 목록."""
    d = d.sort_values("t0").reset_index(drop=True)
    cut = int(len(d) * train)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    cols_tr = [np.ones(len(tr)), np.log(tr["sig"]), np.log(tr["r0"])]
    cols_te = [np.ones(len(te)), np.log(te["sig"]), np.log(te["r0"])]
    if use_q:
        cols_tr.append(np.log(np.maximum(tr["Q"], 1.0)))
        cols_te.append(np.log(np.maximum(te["Q"], 1.0)))
    Xtr, Xte = np.column_stack(cols_tr), np.column_stack(cols_te)
    bt = np.linalg.pinv(Xtr.T @ Xtr) @ (Xtr.T @ np.log(tr["X"].to_numpy()))
    mte = Xte @ bt
    Z = np.log(tr["X"].to_numpy()) - Xtr @ bt
    Z0 = np.log(tr["X"].to_numpy())
    xte = te["X"].to_numpy()
    print("\n  [%s] 훈련 %d / 검정 %d | b1(sig)=%.3f b2(r0)=%.3f%s"
          % (tag, len(tr), len(te), bt[1], bt[2],
             " b3(logQ)=%.4f" % bt[3] if use_q else ""))
    t1 = t0_ = 0.0
    for p in LEVELS:
        t1 += pinball(xte, np.exp(mte + float(np.quantile(Z, p))), p)
        t0_ += pinball(xte, np.full(len(te), np.exp(float(np.quantile(Z0, p)))), p)
    ks, pv = ks_unif(np.clip(np.array([float((Z <= v).mean())
                                       for v in (np.log(xte) - mte)]),
                             1e-6, 1 - 1e-6))
    cov = []
    for p in (0.25, 0.50, 0.75):
        q = np.exp(mte + float(np.quantile(Z, p)))
        cov.append("%.3f" % float((xte < q).mean()))
    print("       커버리지(0.25/0.50/0.75) %s | PIT D=%.4f p=%.3f | 핀볼 개선 %.1f%%"
          % ("/".join(cov), ks, pv, 100 * (t0_ - t1) / t0_))

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
            ex = min(fb + hold, len(CL[i]) - 1)
            rs.append((float(CL[i][ex]) / lim - 1.0) * sd_ * 1e4 - FEE_M - FEE_T)
            f += 1
        return np.array(rs), f

    rows = []
    for p in (0.25, 0.50, 0.75, 0.90):
        dep = np.exp(mte + float(np.quantile(Z, p)))
        rr, f = pnl(dep)
        m, se, t, _ = cmean(rr, te["day"].to_numpy())
        rows.append(("%s 모형 q%.2f (중앙%4.0fbp)" % (tag, p, np.median(dep)),
                     f / len(te), m, t))
    for d0 in (0.0, 100.0, 300.0):
        rr, f = pnl(np.full(len(te), d0))
        m, se, t, _ = cmean(rr, te["day"].to_numpy())
        rows.append(("%s 고정 %.0fbp" % (tag, d0), f / len(te), m, t))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="liquidation-print confirmed trigger")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--zk", type=float, default=5.0)
    ap.add_argument("--gap", type=int, default=60)
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--exchange", default="binance-futures")
    ap.add_argument("--min-q", type=float, default=0.0,
                    help="이 명목가 초과 청산이 있어야 '확인' 으로 본다")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 82)
    print("실시간 청산 프린트 확인 방아쇠 — OI 5분 지연 제거. 룩어헤드 없음")
    print("=" * 82)
    d = build(syms, a.zk, a.gap, a.window, a.exchange, a.min_q)
    if len(d) < 200:
        print("표본 부족 (%d)" % len(d))
        return 1
    nday = d["day"].nunique()
    print("**사용 데이터 기간: %s ~ %s / %d종 / %d일 / 방아쇠 %d건 (|z|>=%.1f)**"
          % (str(pd.to_datetime(d.t0.min(), unit="ms"))[:10],
             str(pd.to_datetime(d.t0.max(), unit="ms"))[:10],
             d.symbol.nunique(), nday, len(d), a.zk))
    cf = d[d["conf"] == 1]
    nc = d[d["conf"] == 0]
    print("  청산 확인됨 %d건 (%.1f%%) | 확인 안 됨 %d건" % (len(cf), 100 * len(cf) / len(d), len(nc)))
    print("  *** 같은 날 안에서 비교한다 — 날 구성 교란 없음\n")
    print("  %-16s %8s %10s %10s %10s" % ("집단", "n", "r0 중앙", "X 중앙", "**X/r0**"))
    for lab, g in (("전체", d), ("**청산 확인**", cf), ("확인 안 됨", nc)):
        if len(g) < 30:
            continue
        print("  %-16s %8d %10.0f %10.0f %10.2f"
              % (lab, len(g), g.r0.median(), g.X.median(), (g.X / g.r0).median()))
    print("  대조 — OI 확인(5분 지연) 판: X/r0 = 5.30 | 가격만(6년): 1.88")
    if len(cf) > 40 and len(nc) > 40:
        y = np.log(d["X"].to_numpy() / d["r0"].to_numpy())
        Xm = np.column_stack([np.ones(len(d)), d["conf"].to_numpy().astype(float),
                              np.log(d["sig"].to_numpy())])
        b, se, _ = ols_cluster(Xm, y, d["day"].to_numpy())
        print("  확인 더미 계수 %+.3f (t=%.1f)  — 유의한 양수면 청산 확인이 정보다"
              % (b[1], b[1] / se[1] if se[1] > 0 else np.nan))

    print("\n" + "-" * 82)
    print("모형 + 지정가 P&L (보유 %d분, 왕복 %.0fbp)" % (a.hold, FEE_M + FEE_T))
    print("-" * 82)
    rows = []
    if len(cf) >= 300:
        rows += fit_eval(cf, "확인", a.train, a.hold, use_q=True)
    if len(nc) >= 300:
        rows += fit_eval(nc, "미확인", a.train, a.hold, use_q=False)
    print("\n  %-34s %8s %10s %7s" % ("방식", "체결률", "이벤트당", "t"))
    for lab, fr, m, t in sorted(rows, key=lambda x: -x[2]):
        print("  %-34s %7.1f%% %10.1f %7.1f" % (lab, 100 * fr, m, t))
    print("\n  *** 전부 실시간 관측 가능하다. '확인' 이 '미확인' 을 이기면")
    print("      청산 프린트가 OI 5분 지연을 대체할 수 있다는 뜻이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
