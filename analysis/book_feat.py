# -*- coding: utf-8 -*-
"""오더북 특징을 (사건 x 분) 격자에 붙인다 — ②깊이와 ③유입·취소의 과거 3년반 판.

왜 이게 필요했나 (2026-08-05, 내 오류)
  prob_entry / dyn_entry 의 예측 특징은 전부 5분봉 파생(OI·포지션비·거래량)이었다.
  설계의 ②오더북 깊이와 ③유입·취소가 통째로 빠져 있었다.
  이유가 어이없다 — 나는 book_depth 를 `book_depth/<SYMBOL>.parquet` (구 단일 파일)
  로만 읽어서 커버리지가 26.6% 인 줄 알았다. 실제 데이터는
  `book_depth/<SYMBOL>/<날짜>.parquet` 에 **27,140개 일별 파일, 21종 전부,
  2023-01-01~2026-08-03, 30초 간격**으로 다 있다. 커버리지는 사실상 100% 다.

무엇을 뽑는가 (전부 시각 tau 에 알려진 값)
  ldep    log( 압력받는 쪽 ±1% 깊이 / 과거 1일 중앙 )    -> **② 깊이**
  limb    (압력쪽 - 반대쪽)/(합)                          -> 불균형
  lslope  log( ±5% 깊이 / ±1% 깊이 )                     -> 깊이 프로파일
  ddep5   log( 지금 깊이 / 5분 전 깊이 )                 -> **③ 회복력·취소압**
  ddep30  log( 지금 깊이 / 30분 전 깊이 )
  ** ddep5 가 핵심이다. A(delta_frame.py) 에서 회복력이 t=-3.8~-9.3 로 나온 그 양의
     30초 해상도 판이다. 깊이가 다시 차오르는 것은 가격이 반등하기 **전에** 보인다. **

  압력받는 쪽: sd=+1(하락) 이면 아래쪽 dm, sd=-1(상승) 이면 위쪽 dp.

*** 반드시 bookdepth.load_clean 을 쓴다 ***
  2025년 구간에 벤더 결함이 있다(특정 percentage 의 notional 이 몇 시간씩 고정).
  원본을 그대로 읽으면 분모가 날조된다. load_clean 이 그 구간을 지운다.

실행:
    python analysis/book_feat.py                 # 캐시 생성
    python analysis/book_feat.py --symbols BTCUSDT ETHUSDT
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
from analysis.bookdepth import load_clean                             # noqa: E402
from analysis.event_study_h2 import load, find_events                 # noqa: E402

HMAX = 60                      # prob_entry 와 같아야 한다
WMAX = 30                      # dyn_entry 와 같아야 한다
CACHE = os.path.join(C.DATA, "analysis", "book_feat.parquet")
BFEAT = ["ldep", "limb", "lslope", "ddep5", "ddep30"]
DAY_MS = 86_400_000


def _series(sym: str):
    """정제된 깊이 시계열. (ts, dm1, dp1, dm5, dp5) — 없는 컬럼은 NaN."""
    d, st = load_clean(sym, ["dm1_0", "dp1_0"],
                       optional=["dm5_0", "dp5_0"], verbose=False)
    if d.empty:
        return None
    ts = d["ts_ms"].to_numpy()
    o = np.argsort(ts, kind="mergesort")
    d = d.iloc[o]
    ts = ts[o]
    g = lambda c: (d[c].to_numpy(dtype=np.float64) if c in d.columns
                   else np.full(len(d), np.nan))
    return ts, g("dm1_0"), g("dp1_0"), g("dm5_0"), g("dp5_0")


def build_features(symbols, k, doi_thr, gap) -> pd.DataFrame:
    """사건 x 분 격자에 오더북 특징. prob_entry.build 와 같은 사건 정의를 쓴다."""
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, k, doi_thr, gap)
        ev = ev[ev.is_liq]
        if not len(ev):
            continue
        ser = _series(s)
        if ser is None:
            U.log("%s bookDepth 없음 — 건너뜀" % s)
            continue
        ts, dm1, dp1, dm5, dp5 = ser
        ot5 = df["open_time"].to_numpy()
        # 과거 1일 중앙 깊이(정규화용). 30초 간격이므로 2880행 = 1일.
        # rolling median 은 느리므로 평균을 쓴다 (정규화 목적).
        m_dm = pd.Series(dm1).rolling(2880, min_periods=400).mean().shift(1).to_numpy()
        m_dp = pd.Series(dp1).rolling(2880, min_periods=400).mean().shift(1).to_numpy()
        for r in ev.itertuples():
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5):
                continue
            t0 = int(ot5[i + 1])
            for u in range(0, WMAX + 1):
                tau = t0 + u * 60_000
                j = int(np.searchsorted(ts, tau, side="right")) - 1
                if j < 0 or tau - int(ts[j]) > 180_000:     # 3분 넘게 낡았으면 버린다
                    continue
                # 압력받는 쪽 / 반대쪽
                near = dm1[j] if sd == 1 else dp1[j]
                far = dp1[j] if sd == 1 else dm1[j]
                deep = dm5[j] if sd == 1 else dp5[j]
                base = m_dm[j] if sd == 1 else m_dp[j]
                if not (np.isfinite(near) and near > 0):
                    continue
                # 5분 전 / 30분 전 (같은 축)
                def at(mins):
                    jj = int(np.searchsorted(ts, tau - mins * 60_000, side="right")) - 1
                    if jj < 0 or (tau - mins * 60_000) - int(ts[jj]) > 180_000:
                        return np.nan
                    v = dm1[jj] if sd == 1 else dp1[jj]
                    return v if (np.isfinite(v) and v > 0) else np.nan
                p5, p30 = at(5), at(30)
                out.append({
                    "symbol": s, "t": t0, "u": u,
                    "ldep": (np.log(near / base) if (np.isfinite(base) and base > 0)
                             else np.nan),
                    "limb": ((near - far) / (near + far)
                             if (np.isfinite(far) and near + far > 0) else np.nan),
                    "lslope": (np.log(deep / near) if (np.isfinite(deep) and deep > 0)
                               else np.nan),
                    "ddep5": np.log(near / p5) if np.isfinite(p5) else np.nan,
                    "ddep30": np.log(near / p30) if np.isfinite(p30) else np.nan,
                })
        U.log("%-9s 사건 %d / 격자행 %d" % (s, len(ev), len(out)))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="orderbook features on the event-minute grid")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--out", default=CACHE)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 96)
    print("오더북 특징 추출 — book_depth 일별 파일 (30초, ±1/5%), load_clean 필터 적용")
    print("=" * 96)
    d = build_features(syms, a.k, a.doi, a.gap)
    if d.empty:
        print("추출 실패")
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    U.atomic_write_parquet(d, a.out)
    print("\n저장 %s | %d행 / 사건 %d개 / 심볼 %d종"
          % (a.out, len(d), d.groupby(["symbol", "t"]).ngroups, d.symbol.nunique()))
    print("\n결측률:")
    for c in BFEAT:
        print("  %-8s %.3f" % (c, float(d[c].isna().mean())))
    print("\n분포 (u=0 행만):")
    z = d[d.u == 0]
    for c in BFEAT:
        v = z[c].dropna()
        if len(v):
            print("  %-8s n=%5d  p10 %+8.3f  p50 %+8.3f  p90 %+8.3f"
                  % (c, len(v), v.quantile(.1), v.quantile(.5), v.quantile(.9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
