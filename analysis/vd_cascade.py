# -*- coding: utf-8 -*-
"""캐스케이드 규모에서 V/D 탄성 — 과거 호가 깊이(book_depth)로. 웹소켓을 기다리지 않는다.

왜 지금 되는가
  queue_state.py / ws_depth_test.py 는 전부 같은 이유로 실패했다:
      Q/D 중앙 5.7e-05, p99 0.013 — **제곱근 법칙이 작동하는 구간에 닿지 않는다.**
  웹소켓은 15시간치라 큰 사건이 없다. 그런데 data.binance.vision 의 bookDepth 가
  **2023-01 ~ 2026-08, 21종, 일별 약 2,879스냅샷**으로 이미 받아져 있다(7.8GB).
  캐스케이드 이벤트(|z|>=8 + OI급감) 기간을 그대로 덮는다.

  -> 같은 회귀를 **실제 캐스케이드 규모**에서 돌린다.

컬럼
  dm1_0 ... dm5_0 : 중심가 **아래** 1~5% 누적 명목가 (매수호가 = 하방 압력을 받는 쪽)
  dp1_0 ... dp5_0 : 중심가 **위**  1~5% (매도호가)

검정
  log X = a + b1 log(sigma) + b2 log(Q/D)   [+ 심볼 고정효과]
  제곱근 법칙 예측 b2 = 0.5.
  ADV 분모(synth.py): -0.006 | 웹소켓 1초(FE 후): +0.033 (t=1.4, 무의미)
  **여기서 살아나면 원인은 모형이 아니라 '규모 구간' 이었다는 뜻이다.**

데이터 품질
  bookDepth 는 특정 날 notional 이 **고정(freeze)** 되는 알려진 결함이 있다
  (analysis/bookdepth.py 참조). 같은 값이 10회 이상 연속하면 그 구간을 버린다.

실행:
    python analysis/vd_cascade.py
    python analysis/vd_cascade.py --pct 2      # 2% 밴드로
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
from analysis.response_liq import ols_cluster                   # noqa: E402
from analysis.synth import build as build_events                # noqa: E402

BD = os.path.join(C.DATA, "binance_bulk", "book_depth")
_CACHE: dict = {}


def frozen(v: np.ndarray, min_run: int = 10) -> np.ndarray:
    """같은 값이 min_run 회 이상 연속하면 True (bookDepth 고정 결함)."""
    n = len(v)
    bad = np.zeros(n, dtype=bool)
    if n < min_run:
        return bad
    same = np.concatenate([[False], v[1:] == v[:-1]])
    start, run = 0, 0
    for i in range(n):
        if same[i]:
            run += 1
        else:
            if run + 1 >= min_run:
                bad[start:i] = True
            start, run = i, 0
    if run + 1 >= min_run:
        bad[start:n] = True
    return bad


def day_depth(sym: str, day: str, cols):
    """그날 파일. 고정 구간 제거 후 (ts, {col: arr})."""
    key = (sym, day)
    if key in _CACHE:
        return _CACHE[key]
    p = os.path.join(BD, sym, "%s.parquet" % day)
    out = None
    if os.path.exists(p):
        try:
            d = pd.read_parquet(p)
        except Exception:
            d = None
        if d is not None and "ts_ms" in d.columns and all(c in d.columns for c in cols):
            d = d.sort_values("ts_ms").reset_index(drop=True)
            bad = np.zeros(len(d), dtype=bool)
            for c in cols:
                bad |= frozen(d[c].to_numpy())
            d = d[~bad]
            d = d[(d[list(cols)] > 0).all(axis=1)]
            if len(d) > 20:
                out = (d["ts_ms"].to_numpy(),
                       {c: d[c].to_numpy(dtype=np.float64) for c in cols})
    if len(_CACHE) > 400:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="V/D elasticity at cascade scale")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--pct", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--window", type=int, default=240)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    cm, cp = "dm%d_0" % a.pct, "dp%d_0" % a.pct

    print("=" * 78)
    print("캐스케이드 규모 V/D 탄성 — 과거 호가 깊이 %d%% 밴드" % a.pct)
    print("=" * 78)
    ev, _ = build_events(syms, a.window)
    ev = ev.sort_values("t0").reset_index(drop=True)
    print("이벤트 %d건 (전체) | %s ~ %s"
          % (len(ev), str(pd.to_datetime(ev.t0.min(), unit="ms"))[:10],
             str(pd.to_datetime(ev.t0.max(), unit="ms"))[:10]))

    ev["day"] = pd.to_datetime(ev["t0"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    Ds, miss = [], 0
    for r in ev.itertuples():
        got = day_depth(r.symbol, r.day, (cm, cp))
        if got is None:
            Ds.append(np.nan)
            miss += 1
            continue
        ts, arr = got
        i = int(np.searchsorted(ts, r.t0)) - 1      # **직전** 스냅샷 = 룩어헤드 없음
        if i < 0 or i >= len(ts):
            Ds.append(np.nan)
            continue
        Ds.append(arr[cm][i] if r.side == 1 else arr[cp][i])
    ev["D"] = Ds
    d = ev[np.isfinite(ev["D"]) & (ev["D"] > 0)].copy()
    print("깊이 매칭 %d건 (파일 없음/불량 %d) | %s ~ %s"
          % (len(d), miss, str(pd.to_datetime(d.t0.min(), unit="ms"))[:10],
             str(pd.to_datetime(d.t0.max(), unit="ms"))[:10]))
    if len(d) < 100:
        print("표본 부족")
        return 1

    d["qd"] = d["S0"] / d["D"]
    qd = d["qd"].to_numpy()
    print("\n**Q/D 분포 — 이것이 관건이다**")
    print("  중앙 %.4g | p75 %.4g | p90 %.4g | p99 %.4g | 최대 %.4g"
          % (np.median(qd), *[np.quantile(qd, q) for q in (.75, .9, .99)], qd.max()))
    print("  대조 — 웹소켓 1초 표본: 중앙 5.7e-05, p99 0.013, 최대 0.139")
    print("  Q/D > 0.01 인 비율 %.1f%% | > 0.1 인 비율 %.1f%%"
          % (100 * (qd > 0.01).mean(), 100 * (qd > 0.1).mean()))
    print("  D 중앙 $%.4g | Q(S0) 중앙 $%.4g" % (d.D.median(), d.S0.median()))
    print("X (진입가 대비 최대 역행) 중앙 %.0f bp | p90 %.0f bp"
          % (d.X.median(), d.X.quantile(.9)))

    y = np.log(np.maximum(d["X"].to_numpy(), 1e-6))
    ls = np.log(d["sig"].to_numpy())
    lq = np.log(qd)
    ok = np.isfinite(y) & np.isfinite(ls) & np.isfinite(lq)
    cl = (d["t0"].to_numpy() // 86_400_000)          # 일 클러스터
    sy = pd.get_dummies(d["symbol"]).to_numpy(dtype=np.float64)[:, 1:]

    print("\n" + "-" * 78)
    print("핵심 회귀  log X = a + b1 log(sigma) + b2 log(Q/D)   [일클러스터 CR1]")
    print("-" * 78)
    print("  제곱근 법칙 예측: b1 = 1.0, **b2 = 0.5**")
    print("  %-22s %10s %7s | %10s %7s | %7s"
          % ("설정", "b1(sigma)", "t", "**b2(Q/D)**", "t", "R^2"))
    for lab, Xm in (
            ("고정효과 없음", np.column_stack([np.ones(int(ok.sum())), ls[ok], lq[ok]])),
            ("심볼 고정효과",
             np.column_stack([np.ones(int(ok.sum())), ls[ok], lq[ok], sy[ok]]))):
        b, se, _ = ols_cluster(Xm, y[ok], cl[ok])
        r2 = 1.0 - np.var(y[ok] - Xm @ b) / np.var(y[ok])
        print("  %-22s %10.3f %7.1f | %10.3f %7.1f | %7.3f"
              % (lab, b[1], b[1] / se[1], b[2], b[2] / se[2] if se[2] > 0 else np.nan, r2))
    print("\n  대조 — 같은 회귀를 분모만 바꿔서")
    for lab, den in (("ADV (synth.py)", d["adv"].to_numpy()),
                     ("없음 (Q만)", np.ones(len(d)))):
        lz = np.log(d["S0"].to_numpy() / den)
        m2 = ok & np.isfinite(lz)
        Xm = np.column_stack([np.ones(int(m2.sum())), ls[m2], lz[m2], sy[m2]])
        b, se, _ = ols_cluster(Xm, y[m2], cl[m2])
        print("    %-20s b2 = %+.4f (t=%.1f)  [심볼 FE 포함]"
              % (lab, b[2], b[2] / se[2] if se[2] > 0 else np.nan))

    print("\n" + "-" * 78)
    print("상태의존 — b2 가 Q/D 구간에 따라 커지는가 (임계형 검정)")
    print("-" * 78)
    d = d.assign(bin=pd.qcut(qd, 5, labels=False, duplicates="drop"))
    print("  %5s %7s %12s %11s | %9s %7s"
          % ("오분위", "n", "Q/D 중앙", "X 중앙bp", "b2", "t"))
    for q in sorted(pd.unique(d["bin"].dropna())):
        g = d[d["bin"] == q]
        yy = np.log(np.maximum(g["X"].to_numpy(), 1e-6))
        mm = np.isfinite(yy)
        if mm.sum() < 40:
            continue
        Xm = np.column_stack([np.ones(int(mm.sum())), np.log(g["sig"].to_numpy()[mm]),
                              np.log(g["qd"].to_numpy()[mm])])
        b, se, _ = ols_cluster(Xm, yy[mm], (g["t0"].to_numpy() // 86_400_000)[mm])
        print("  %5d %7d %12.4g %11.0f | %9.3f %7.1f"
              % (q, len(g), g["qd"].median(), g["X"].median(),
                 b[2], b[2] / se[2] if se[2] > 0 else np.nan))
    print("\n  2차항: log X ~ log sig + log(Q/D) + log(Q/D)^2 + 심볼FE")
    Xm = np.column_stack([np.ones(int(ok.sum())), ls[ok], lq[ok], lq[ok] ** 2, sy[ok]])
    b, se, _ = ols_cluster(Xm, y[ok], cl[ok])
    print("    1차 %+.4f (t=%.1f) | **2차 %+.4f (t=%.1f)**"
          % (b[2], b[2] / se[2], b[3], b[3] / se[3] if se[3] > 0 else np.nan))
    print("    2차가 유의한 양수 = 큰 Q/D 에서 탄성이 커진다 = 임계형 지지")
    return 0


if __name__ == "__main__":
    sys.exit(main())
