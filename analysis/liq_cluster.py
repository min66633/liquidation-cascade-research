# -*- coding: utf-8 -*-
"""실현 청산량 x 그 시점 오더북 -> 실제로 가격이 얼마나 밀렸나.

왜 이것인가
  지금까지 강제매도량의 대용치로 |dOI| (5분 OI 순변화)를 썼다. 그런데 그것은

      |dOI| = 격리청산 + 교차청산 + 자발적종료 - 신규진입

  네 성분의 순합이라, 회귀변수 측정오차가 계수를 0 쪽으로 감쇠시킨다.
  실제로 corr(log X, |dOI|*OIV/D) = +0.224 로 전체 테이커 물량판(+0.573)보다 약했는데,
  '효과가 없다' 와 '감쇠됐다' 를 구별할 수 없었다.

  Bybit allLiquidation 은 **전건 피드**라 강제분만 직접 측정한다(Binance forceOrder 는
  심볼당 초당 1건 스로틀이라 못 쓴다). 이걸 V 로 쓰면 감쇠 문제가 원천적으로 사라진다.

두 가지를 잰다
  (1) 자유모수 없는 직접 예측 — 오더북을 V 만큼 먹으면 어디까지 가는가(u_pred)
      vs 실제로 간 곳(u_act). 회귀가 아니라 물리적 예측이므로 과적합 여지가 없다.
  (2) 오더북 모양에서 gamma 를 직접 추정 — 누적깊이 Cum(u) = B u^kappa 를 적합하면
      gamma = 1/kappa. 수익 회귀로 얻은 gamma=0.875 (VD_FINDINGS)와 **독립** 교차검증.

데이터
  청산   Bybit allLiquidation (전건, WS 실시간)
  호가   depth_poll 30초 스냅샷. 밴드 0.05/0.1/0.2/0.5/1/2%
         *** REST 도달 한계: BTC 0.19% / ETH 0.56% — 그 너머는 NaN.
             bid_reach_pct 로 걸러야 한다. 안 걸르면 0 을 '깊이 없음'으로 오독한다.
  가격   같은 depth_poll 의 mid (같은 소스라 정합이 맞고, 1분봉 결측과 무관)

한계 (결과와 함께 반드시 보고)
  - 거래소 교차: Bybit 청산 x Binance 호가/가격. 메이저는 가격이 붙어 다니지만 동일하지 않다.
  - 표본이 40시간 한 구간이고 롱청산 편중($17.1M vs $9.0M). 레짐 일반화 불가.
  - 1차 검정은 **부호와 크기 확인** 용도다.

실행:
    python analysis/liq_cluster.py
    python analysis/liq_cluster.py --min-usd 50000 --horizon 5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

# depth_poll 밴드 (퍼센트, 컬럼접미사)
BANDS = [(0.05, "0_05"), (0.10, "0_1"), (0.20, "0_2"),
         (0.50, "0_5"), (1.00, "1_0"), (2.00, "2_0")]
MAX_SNAP_LAG_MS = 90_000          # 30초 폴링이므로 90초까지 허용
MIN_BANDS = 3                     # kappa 적합에 최소 3점
REACH_SAFETY = 0.95               # 도달범위의 95% 까지만 신뢰


def load_liq() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "bybit_liq", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/bybit_liq 가 비어 있다 (collectors/bybit_liq_ws.py)")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["symbol"].isin(C.MAJORS)].copy()
    d["ntl"] = d["size"] * d["bankruptcy_px"]
    d = d[np.isfinite(d["ntl"]) & (d["ntl"] > 0)]
    return d.sort_values("exch_ms").reset_index(drop=True)


def load_depth() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "depth", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/depth 가 비어 있다 (collectors/depth_poll.py)")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["symbol"].isin(C.MAJORS)]
    d = d.drop_duplicates(subset=["symbol", "ts_ms"]).sort_values("ts_ms")
    return d.reset_index(drop=True)


def cum_profile(row: pd.Series, side: str) -> tuple:
    """도달범위 안의 밴드만 (u, 누적명목가) 로 뽑는다.

    side='bid' 는 롱청산(강제 매도)이 먹는 쪽. 'ask' 는 숏청산.
    REST 가 도달하지 못한 밴드는 NaN 이고, 0 으로 채우면 '깊이 없음'으로 오독된다.
    """
    reach = row.get("%s_reach_pct" % side, np.nan)
    if not np.isfinite(reach) or reach <= 0:
        return np.array([]), np.array([])
    us, cs = [], []
    for pct, suf in BANDS:
        if pct > reach * REACH_SAFETY:
            break
        v = row.get("%s_%s" % (side, suf), np.nan)
        if not np.isfinite(v) or v <= 0:
            break
        us.append(pct / 100.0)
        cs.append(float(v))
    u, c = np.array(us), np.array(cs)
    if len(u) >= 2 and np.any(np.diff(c) <= 0):      # 누적은 단조여야 한다
        return np.array([]), np.array([])
    return u, c


def fit_kappa(u: np.ndarray, c: np.ndarray) -> tuple:
    """Cum(u) = B u^kappa 를 로그-로그 OLS. (kappa, logB, R2)"""
    if len(u) < MIN_BANDS:
        return np.nan, np.nan, np.nan
    x, y = np.log(u), np.log(c)
    A = np.column_stack([np.ones(len(x)), x])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ w
    den = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / den if den > 0 else np.nan
    return float(w[1]), float(w[0]), r2


def main() -> int:
    ap = argparse.ArgumentParser(description="realized liquidation x orderbook -> price push")
    ap.add_argument("--min-usd", type=float, default=1e5, help="클러스터 최소 명목가")
    ap.add_argument("--bar-sec", type=int, default=60, help="클러스터 묶는 단위(초)")
    ap.add_argument("--horizon", type=int, default=10, help="실제 밀림 측정 지평(분)")
    a = ap.parse_args()

    U.init_stdout()
    liq, dep = load_liq(), load_depth()
    t0 = pd.to_datetime(liq["exch_ms"].min(), unit="ms", utc=True)
    t1 = pd.to_datetime(liq["exch_ms"].max(), unit="ms", utc=True)
    print("=" * 74)
    print("실현 청산 x 오더북 -> 실제 가격 슈팅  (1차 검정)")
    print("=" * 74)
    print("청산 %d건 / %d종 | %s ~ %s (%.1f시간)"
          % (len(liq), liq.symbol.nunique(), t0, t1,
             (t1 - t0).total_seconds() / 3600))
    print("호가 스냅샷 %d개 / %d종" % (len(dep), dep.symbol.nunique()))

    # ---- 클러스터: (심볼, bar) 단위로 우세 방향만
    bar = a.bar_sec * 1000
    liq["bar"] = liq["exch_ms"] // bar
    g = (liq.groupby(["symbol", "bar", "pos_side"])["ntl"].sum().reset_index())
    g = g.sort_values("ntl", ascending=False).drop_duplicates(["symbol", "bar"])
    g = g[g["ntl"] >= a.min_usd].copy()
    g["ts_ms"] = g["bar"] * bar + bar          # 클러스터 종료 시각
    print("\n클러스터(%ds, 우세방향, $%.0f 이상): %d개"
          % (a.bar_sec, a.min_usd, len(g)))
    if g.empty:
        print("임계를 낮추십시오.")
        return 1

    rows, drop = [], {"snap": 0, "prof": 0, "fwd": 0}
    for sym, sub in g.groupby("symbol"):
        ds = dep[dep.symbol == sym].reset_index(drop=True)
        if ds.empty:
            drop["snap"] += len(sub)
            continue
        dts = ds["ts_ms"].to_numpy()
        mids = ds["mid"].to_numpy()
        for r in sub.itertuples():
            j = int(np.searchsorted(dts, r.ts_ms, side="right")) - 1
            if j < 0 or r.ts_ms - int(dts[j]) > MAX_SNAP_LAG_MS:
                drop["snap"] += 1
                continue
            p0 = float(mids[j])
            if not (np.isfinite(p0) and p0 > 0):
                drop["snap"] += 1
                continue
            side = "bid" if r.pos_side == "long" else "ask"   # 롱청산 = 강제매도
            u, c = cum_profile(ds.iloc[j], side)
            kappa, logB, r2 = fit_kappa(u, c)
            if not np.isfinite(kappa) or kappa <= 0:
                drop["prof"] += 1
                continue

            # 실제 밀림: 이후 horizon 분 동안 mid 의 최대 역행
            k1 = int(np.searchsorted(dts, r.ts_ms + a.horizon * 60_000, side="left"))
            seg = mids[j:k1]
            if len(seg) < 2 or not np.any(np.isfinite(seg)):
                drop["fwd"] += 1
                continue
            u_act = (1.0 - float(np.nanmin(seg)) / p0 if side == "bid"
                     else float(np.nanmax(seg)) / p0 - 1.0)

            # 예측: 오더북을 V 만큼 먹으면 어디까지
            u_pred = float(np.exp((np.log(r.ntl) - logB) / kappa))
            rows.append({"symbol": sym, "ts_ms": int(r.ts_ms), "side": side,
                         "V": float(r.ntl), "p0": p0, "n_bands": len(u),
                         "kappa": kappa, "prof_r2": r2,
                         "D_deep": float(c[-1]), "u_deep": float(u[-1]),
                         "VoverD": float(r.ntl) / float(c[-1]),
                         "u_pred": u_pred, "u_act": max(u_act, 0.0)})

    d = pd.DataFrame(rows)
    print("탈락: 스냅샷 %d / 프로파일 %d / 전방 %d  -> 유효 %d"
          % (drop["snap"], drop["prof"], drop["fwd"], len(d)))
    if d.empty:
        return 1

    print("\n--- 1. 오더북 모양에서 gamma 직접 추정 (수익회귀와 독립) ---")
    print("  Cum(u) = B u^kappa 를 밴드에 적합.  gamma = 1/kappa")
    print("  kappa=1 이면 가격축 균일(gamma=1). VD_FINDINGS 수익회귀판은 gamma=0.875")
    kk = d["kappa"]
    print("  kappa 중앙 %.3f  [p25 %.3f, p75 %.3f]  -> gamma 중앙 %.3f"
          % (kk.median(), kk.quantile(.25), kk.quantile(.75), 1.0 / kk.median()))
    print("  프로파일 적합 R2 중앙 %.4f | 사용 밴드수 중앙 %.0f"
          % (d["prof_r2"].median(), d["n_bands"].median()))
    print("  심볼별 kappa 중앙:")
    for s, v in d.groupby("symbol")["kappa"].median().sort_values().items():
        print("    %-10s %.3f  (gamma %.3f, n=%d)"
              % (s, v, 1.0 / v, int((d.symbol == s).sum())))

    print("\n--- 2. 청산 규모가 호가 대비 얼마나 큰가 ---")
    v = d["VoverD"]
    print("  V / D(도달 최심) :  중앙 %.4f  p90 %.4f  최대 %.4f"
          % (v.median(), v.quantile(.9), v.max()))
    print("  -> 1 보다 훨씬 작으면 '청산이 호가를 먹지 못한다' 는 뜻이다.")
    print("  도달 최심 u 중앙 %.2f%% | 그 깊이 중앙 $%.4g"
          % (100 * d["u_deep"].median(), d["D_deep"].median()))

    print("\n--- 3. 예측 vs 실제 (자유모수 없음) ---")
    print("  u_pred = 오더북을 V 만큼 먹었을 때 도달 가격.  u_act = 실제 %d분 최대 역행"
          % a.horizon)
    print("  %-12s %10s %10s %10s" % ("", "중앙", "p25", "p75"))
    for lab, col in (("u_pred %", "u_pred"), ("u_act  %", "u_act")):
        s = d[col]
        print("  %-12s %9.4f%% %9.4f%% %9.4f%%"
              % (lab, 100 * s.median(), 100 * s.quantile(.25), 100 * s.quantile(.75)))
    ratio = (d["u_act"] / d["u_pred"].clip(lower=1e-12))
    print("  u_act / u_pred : 중앙 %.1f배  [p25 %.1f, p75 %.1f]"
          % (ratio.median(), ratio.quantile(.25), ratio.quantile(.75)))
    print("  1 근처면 '먹은 만큼 밀린다'. >>1 이면 증폭(2차청산/호가철수),")
    print("  <<1 이면 호가가 다시 차서 흡수한다는 뜻이다.")

    lp, la = np.log(d["u_pred"].clip(lower=1e-9)), np.log(d["u_act"].clip(lower=1e-9))
    ok = (np.isfinite(lp) & np.isfinite(la) & (d["u_act"] > 0)).to_numpy()
    if int(ok.sum()) >= 10:
        print("  corr(log u_pred, log u_act) = %+.3f  (n=%d)"
              % (float(np.corrcoef(lp[ok], la[ok])[0, 1]), int(ok.sum())))
        lv = np.log(d["VoverD"].clip(lower=1e-12))
        print("  corr(log V/D,    log u_act) = %+.3f"
              % (float(np.corrcoef(lv[ok], la[ok])[0, 1])))
        print("  *** 비교: |dOI| 판은 +0.224, 전체 테이커 물량판은 +0.573 (VD_FINDINGS)")

    print("\n--- 4. 가장 큰 클러스터 상위 15건 ---")
    print("  %-9s %-15s %-4s %11s %8s %9s %9s %7s"
          % ("심볼", "시각(UTC)", "방향", "V$", "V/D", "u_pred%", "u_act%", "배수"))
    for r in d.nlargest(15, "V").itertuples():
        ts = pd.to_datetime(r.ts_ms, unit="ms", utc=True).strftime("%m-%d %H:%M:%S")
        print("  %-9s %-15s %-4s %11.4g %8.4f %8.4f%% %8.4f%% %7.1f"
              % (r.symbol, ts, r.side, r.V, r.VoverD,
                 100 * r.u_pred, 100 * r.u_act,
                 r.u_act / max(r.u_pred, 1e-12)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
