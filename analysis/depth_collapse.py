# -*- coding: utf-8 -*-
"""D(u) 는 상수가 아니다 — 스트레스에서 호가 깊이가 무너지는 과정 측정.

왜 필요한가
  MODEL.md v2 의 충격항은 임계형 kappa*max(0, V/D - c)^beta 다. 그런데 c 는
  "> 0.31 (하한만)" 으로만 알려져 있고, 그 0.31 은 impact_depth.py 표본에서 나왔다.
  그 표본을 열어보니 **2026년 매월 1일 6일치**이고, 그 6일에 청산성 이벤트
  (8sigma + dOI <= -2%) 가 **0건**이다. 즉 "임계 미달"이 아니라 **사건 부재**다.
  회귀변수 max(0, V/D - c) 가 표본 전체에서 항등적으로 0 이면 kappa, beta, c 는
  우도에 들어가지 못한다 — 정밀도 문제가 아니라 지지집합 문제다.

  동시에 D 를 상수처럼 다루면 임계 돌파는 영원히 표본 밖으로 남는다. 위기에는
  V 가 커지는 것보다 D 가 무너지는 쪽이 빠를 수 있고, 그렇다면 분자가 아니라
  분모가 캐스케이드를 만든다.

이 스크립트가 답하는 것
  Q-A  청산성 이벤트에서 D 는 자기 사전 기준선 대비 몇 배 얇아지는가? (|z| 의 함수)
  Q-B  OI 급감액을 V 대용치로 쓰면 V/D 지지집합은 어디까지 늘어나는가?
  Q-C  늘어난 지지집합에서 충격항이 살아나는가? (Q4 의 무료 데이터판)

V 대용치에 대한 정직한 진술
  V_doi = max(0, OI명목가[T] - OI명목가[T+5m]) 는 강제청산뿐 아니라 **자발적 청산도
  포함**하므로 실제 청산액을 과대추정한다. 따라서 절대 수준은 Tardis 실측(0.31)과
  직접 비교할 수 없다. 해석 가능한 것은 **같은 대용치로 잰 레짐 간 비율**이다.
  게다가 과대추정 폭은 평상시가 더 클 것이므로(캐스케이드에서는 강제분 비중이 높다)
  측정된 스트레스/평상시 비율은 **보수적**이다 — 진짜 비율은 이보다 크다.

기준선을 두 개 쓰는 이유 (2026-08-01 수정)
  처음에는 60분 롤링 중앙값 하나만 기준선으로 썼는데, 그러면 **지속된 붕괴를 놓친다.**
  2025-10-10 BTC 의 -1% 깊이는 $2.1M 로 하루 종일 눌려 있었다(평소 ~$110M, 52배).
  60분 창은 그 눌린 수준을 따라 내려가므로 collapse 가 3배로만 보인다. 그래서
    D_pre  = 직전 60분 중앙값        -> **이벤트 순간의 급격한 증발**
    D_ref  = 직전 5개 보유일의 일중앙값 중앙값 -> **레짐 대비 그날 자체가 얼마나 얇았나**
  둘은 다른 것을 재며, 캐스케이드에서 위험한 쪽은 후자다.

룩어헤드
  D_at   : 바 T 이전 마지막 스냅샷        -> 사전 관측 가능. 회귀에 쓰는 분모.
  D_pre  : [T-60m, T) 의 중앙값            -> 사전 관측 가능. 단기 기준선.
  D_ref  : 직전 보유일들의 일중앙값        -> 사전 관측 가능. 레짐 기준선.
  D_fwd  : [T, T+15m) 의 최소              -> **사후**. 붕괴 폭 진단에만 쓰고 회귀 금지.

실행:
    python analysis/depth_collapse.py
    python analysis/depth_collapse.py --symbols BTCUSDT --depth-col dm1_0
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
import analysis.bookdepth as BD                # noqa: E402
from analysis.event_study_h2 import load       # noqa: E402
from analysis.impact import fit_power          # noqa: E402
from analysis.impact2d import ols              # noqa: E402


def cluster_ols(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """일자 클러스터 샌드위치 표준오차.

    ols() 는 iid 가정이라 t 를 2배 부풀린다(실측: naive 17.24 vs 클러스터 8.69).
    표본이 5분봉인데 종속변수가 15분 전방창이라 이웃끼리 겹치고, 5종이 같이 움직인다.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in pd.Series(np.arange(len(g))).groupby(g).groups.items():
        Xi = X[np.asarray(idx)]
        ui = resid[np.asarray(idx)]
        s = Xi.T @ ui
        meat += np.outer(s, s)
    n, kk = X.shape
    ng = int(pd.unique(g).size)
    scale = (ng / max(ng - 1, 1)) * ((n - 1) / max(n - kk, 1))
    V = XtX_inv @ meat @ XtX_inv * scale
    return beta, np.sqrt(np.clip(np.diag(V), 0.0, None)), n

DEPTH_DIR = os.path.join(C.DATA, "binance_bulk", "book_depth")
BAR_MS = 300_000
BASE_BARS = 12                # 기준선 창 = 60분
FWD_BARS = 3                  # 붕괴 관측 창 = 15분


def load_depth_5m(symbol: str, bid_col: str, ask_col: str) -> pd.DataFrame:
    """bookDepth(30초 간격)를 5분 격자로 접는다. 매수/매도 양쪽을 함께 접는다.

    last = 바 안 마지막 스냅샷, low = 바 안 최소. 데이터가 없는 5분 칸은 버린다
    (resample 이 만드는 빈 칸을 0 으로 두면 붕괴를 날조하게 된다).

    **2025년 고정 필드는 analysis/bookdepth.load_clean 이 먼저 걸러낸다.** 안 거르면
    분모가 상수라 V/D 꼬리가 통째로 날조된다(코드리뷰 2026-08-01).

    방향: 롱청산(z<0, 우리가 매수)이면 매수호가(dm)를, 숏청산이면 매도호가(dp)를 쓴다.
    """
    d, st = BD.load_clean(symbol, [bid_col, ask_col])
    if d.empty:
        raise FileNotFoundError(
            "no usable bookDepth for %s (missing=%s) — run "
            "downloaders/binance_depth.py --days-file ..." % (symbol, st["missing"]))
    d = d.copy()
    d["bar"] = (d["ts_ms"] // BAR_MS) * BAR_MS
    g = d.groupby("bar")
    out = pd.DataFrame({
        "open_time": g[bid_col].last().index.to_numpy(),
        "bid_last": g[bid_col].last().to_numpy(),
        "bid_low": g[bid_col].min().to_numpy(),
        "ask_last": g[ask_col].last().to_numpy(),
        "ask_low": g[ask_col].min().to_numpy(),
        "n_snap": g.size().to_numpy()})
    return out.sort_values("open_time").reset_index(drop=True)


def build(symbol: str, col: str, k: float, doi_thr: float) -> pd.DataFrame:
    """5분 바 단위 패널. 깊이가 있는 바만 남긴다."""
    df5 = load(symbol)
    ask_col = col.replace("dm", "dp")
    dep = load_depth_5m(symbol, col, ask_col)
    if df5.empty or dep.empty:
        return pd.DataFrame()

    m = df5.merge(dep, on="open_time", how="inner")
    if m.empty:
        return pd.DataFrame()
    m = m.sort_values("open_time").reset_index(drop=True)

    # 깊이 시계열은 날짜가 띄엄띄엄하다. 기준선/전방창은 **연속한 바**에서만 계산한다.
    contig_prev = m["open_time"].diff().eq(BAR_MS)
    grp = (~contig_prev).cumsum()          # 연속 구간 id
    day_key = pd.to_datetime(m["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    # 파생량은 **양쪽 호가에 대해 각각** 만든 뒤 마지막에 방향으로 고른다.
    # (먼저 고르고 lag 를 걸면 d_at 이 '직전 바의 방향'에서 와서 섞인다)
    for tag, last_c, low_c in (("bid", "bid_last", "bid_low"),
                               ("ask", "ask_last", "ask_low")):
        s = m[last_c]
        # D_at: 직전 바의 마지막 스냅샷 (사전 관측 가능)
        m[tag + "_at"] = s.groupby(grp).shift(1)
        # D_pre: 직전 BASE_BARS 개 바의 중앙값 (현재 바 제외)
        m[tag + "_pre"] = (s.groupby(grp)
                            .apply(lambda t: t.shift(1).rolling(
                                BASE_BARS, min_periods=BASE_BARS // 2).median())
                            .reset_index(level=0, drop=True))
        # D_fwd: 현재 바 포함 FWD_BARS 개 바의 최소 (사후 — 진단 전용)
        m[tag + "_fwd"] = (m[low_c].groupby(grp)
                            .apply(lambda t: t[::-1].rolling(FWD_BARS, min_periods=1)
                                              .min()[::-1])
                            .reset_index(level=0, drop=True))
        # 레짐 기준선: 직전 보유일 5일의 '일중앙 깊이'의 중앙값.
        # 보유일이 띄엄띄엄해서 창이 수개월에 걸칠 수 있다 — '레짐'은 느슨한 말이다.
        dm = m.groupby(day_key)[last_c].median()
        m[tag + "_ref"] = day_key.map(dm.shift(1).rolling(5, min_periods=2).median())

    # 방향 선택 — 하락(z<0)이면 매수호가가, 상승이면 매도호가가 소모된다.
    is_up = (m["z"] > 0).to_numpy()
    for f in ("last", "low", "at", "pre", "fwd", "ref"):
        m["d_" + f] = np.where(is_up, m["ask_" + f], m["bid_" + f])

    # V 대용치: 바 구간의 OI 명목가 감소분. df5 의 doi 와 같은 시점 규약을 쓴다.
    oiv = m["sum_open_interest_value"]
    oiv_end = oiv.groupby(grp).shift(-1)
    m["v_doi"] = (oiv - oiv_end).clip(lower=0.0)
    # 5분에 OI 가 거의 전부 증발하는 바는 물리적 사건이 아니라 metrics 결함이다.
    # (실측: 정확히 1.0 인 바가 존재 = 다음 스냅샷이 0/결측)
    m["rel_drop"] = m["v_doi"] / oiv.where(oiv > 0)
    m.loc[~(m["rel_drop"] < 0.5), "v_doi"] = np.nan

    # 변위: 바 T 종료 후 5분간의 불리 방향 최대 변위 / 사전 sigma
    #       (impact_depth.py 와 같은 정의. 방향은 z 부호로 결정)
    ref = m["close"]
    lo_fwd = (m["low"].groupby(grp)
                      .apply(lambda s: s.shift(-1)[::-1].rolling(FWD_BARS, min_periods=1)
                                        .min()[::-1])
                      .reset_index(level=0, drop=True))
    hi_fwd = (m["high"].groupby(grp)
                       .apply(lambda s: s.shift(-1)[::-1].rolling(FWD_BARS, min_periods=1)
                                         .max()[::-1])
                       .reset_index(level=0, drop=True))
    push_down = ref / lo_fwd - 1.0
    push_up = hi_fwd / ref - 1.0
    m["push"] = np.where(is_up, push_up, push_down)
    m["push_sig"] = m["push"] / m["sigma"]

    m["is_liq"] = (m["z"].abs() >= k) & (m["doi"] <= doi_thr)
    m["is_ev"] = m["z"].abs() >= k
    m["symbol"] = symbol
    m["day"] = pd.to_datetime(m["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    keep = ["symbol", "day", "open_time", "z", "doi", "sigma", "close",
            "d_at", "d_pre", "d_ref", "d_fwd", "d_last", "n_snap",
            "v_doi", "rel_drop", "push", "push_sig", "is_liq", "is_ev"]
    return m[keep]


def describe(x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n=0"
    return ("n=%5d  중앙 %8.4f  p90 %8.4f  p99 %8.4f  max %9.4f"
            % (x.size, np.median(x), np.quantile(x, 0.9),
               np.quantile(x, 0.99), x.max()))


def main() -> int:
    ap = argparse.ArgumentParser(description="order-book depth collapse under stress")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--depth-col", default="dm1_0", help="dm1_0 = -1% 누적 명목가")
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-snap", type=int, default=4,
                    help="바당 최소 깊이 스냅샷 수 (30초 간격이면 10이 정상)")
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 200)
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS

    frames = []
    for s in symbols:
        try:
            d = build(s, a.depth_col, a.k, a.doi)
        except (FileNotFoundError, KeyError) as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            U.log("%s: %d 바 / %d일" % (s, len(d), d["day"].nunique()))
    if not frames:
        U.log("no data")
        return 1

    d = pd.concat(frames, ignore_index=True)
    # 깊이 스냅샷이 너무 적은 바는 대표성이 없다
    d = d[d["n_snap"] >= a.min_snap]
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "depth_collapse.parquet"))

    print("\n=== 표본 ===")
    print("바 %d | 심볼 %d | 일수 %d | %s ~ %s"
          % (len(d), d.symbol.nunique(), d["day"].nunique(), d["day"].min(), d["day"].max()))
    print("8sigma 이벤트 %d | 그중 청산성 %d"
          % (int(d["is_ev"].sum()), int(d["is_liq"].sum())))

    # ---------------------------------------------------------------- Q-A
    print("\n=== Q-A. 깊이는 스트레스에서 무너지는가 ===")
    print("  단기 = D_pre/D_fwd (직전 60분 대비)   레짐 = D_ref/D_fwd (직전 5보유일 대비)")
    print("  60분 창은 지속된 붕괴를 따라 내려가므로 단기 배율은 하한이다.")
    ok = d[np.isfinite(d["d_fwd"]) & (d["d_fwd"] > 0)].copy()
    ok["c_short"] = ok["d_pre"] / ok["d_fwd"]
    ok["c_regime"] = ok["d_ref"] / ok["d_fwd"]
    bins = [0, 2, 4, 6, 8, 12, 20, np.inf]
    ok["zb"] = pd.cut(ok["z"].abs(), bins, right=False)
    t = ok.groupby("zb", observed=True).agg(
        n=("c_short", "size"),
        단기_med=("c_short", "median"),
        단기_p90=("c_short", lambda s: s.quantile(0.9)),
        레짐_med=("c_regime", "median"),
        레짐_p90=("c_regime", lambda s: s.quantile(0.9)),
        d_fwd_med_M=("d_fwd", lambda s: s.median() / 1e6))
    print(t.round(3).to_string())

    liq, calm = ok[ok["is_liq"]], ok[~ok["is_ev"]]
    if len(liq) and len(calm):
        for lab, c in (("단기", "c_short"), ("레짐", "c_regime")):
            a1, a2 = liq[c].median(), calm[c].median()
            if np.isfinite(a1) and np.isfinite(a2) and a2 > 0:
                print("  %s: 청산성 %.2f배 vs 평상시 %.2f배 -> 배율 %.2f"
                      % (lab, a1, a2, a1 / a2))

    # 캐스케이드 하루를 통째로 — 표본 안에 있는 가장 큰 사건
    print("\n  --- 2025-10-10 (사상 최대 캐스케이드, 표본 내) ---")
    big = d[d["day"] == "2025-10-10"]
    if big.empty:
        print("    표본에 없음")
    else:
        for s, g in big.groupby("symbol"):
            ref = float(np.nanmedian(g["d_ref"]))
            lo = float(np.nanmin(g["d_last"]))
            med = float(np.nanmedian(g["d_last"]))
            print("    %-9s 평소 $%7.1fM -> 그날 중앙 $%6.2fM (%.0f배) / 최저 $%6.2fM (%.0f배)"
                  % (s, ref / 1e6, med / 1e6, ref / max(med, 1e-9),
                     lo / 1e6, ref / max(lo, 1e-9)))

    # ---------------------------------------------------------------- Q-B
    print("\n=== Q-B. V/D 지지집합 (분모 = D_at, 사전 관측 가능) ===")
    v = d[np.isfinite(d["d_at"]) & (d["d_at"] > 0) & np.isfinite(d["v_doi"])].copy()
    v["ratio"] = v["v_doi"] / v["d_at"]
    print("  평상시 바      : %s" % describe(v.loc[~v["is_ev"], "ratio"].to_numpy()))
    print("  8sigma 이벤트  : %s" % describe(v.loc[v["is_ev"], "ratio"].to_numpy()))
    print("  청산성 이벤트  : %s" % describe(v.loc[v["is_liq"], "ratio"].to_numpy()))
    # 평상시는 OI 증가 바가 절반이라 중앙값이 0 이다. 배율은 상위분위로 본다.
    for q in (0.9, 0.99):
        a1 = float(v.loc[v["is_liq"], "ratio"].quantile(q)) if v["is_liq"].any() else np.nan
        a2 = float(v.loc[~v["is_ev"], "ratio"].quantile(q))
        if np.isfinite(a1) and np.isfinite(a2) and a2 > 0:
            print("  청산성/평상시 %.0f%%분위 배율 = %.1f배" % (100 * q, a1 / a2))

    TARDIS_MAX = 0.31           # impact_depth.py 6일 표본의 상한
    print("\n  Tardis 6일 표본 상한(%.2f) 초과 비율:" % TARDIS_MAX)
    for lab, s in (("평상시", v[~v["is_ev"]]), ("8sigma", v[v["is_ev"]]),
                   ("청산성", v[v["is_liq"]])):
        x = s["ratio"].to_numpy()
        x = x[np.isfinite(x)]
        if x.size:
            print("    %-7s %5d / %6d 바 (%.1f%%),  최대 %.1f = 상한의 %.0f배"
                  % (lab, int((x > TARDIS_MAX).sum()), x.size,
                     100.0 * float((x > TARDIS_MAX).mean()), x.max(),
                     x.max() / TARDIS_MAX))
    print("  주의: V_doi 는 자발적 청산 포함이라 절대수준은 Tardis 실측(중앙 0.0016,")
    print("        max 0.31)과 직접 비교 불가. 해석 가능한 것은 레짐 간 배율이며,")
    print("        과대추정 폭이 평상시에 더 크므로 위 배율은 보수적(하한)이다.")

    # ---------------------------------------------------------------- Q-C
    print("\n=== Q-C. 늘어난 지지집합에서 충격항이 살아나는가 ===")
    reg = v[np.isfinite(v["push_sig"]) & (v["push_sig"] > 0) & (v["ratio"] > 0)]
    if len(reg) > 200:
        _, b, r2, n = fit_power(reg["ratio"].to_numpy(), reg["push_sig"].to_numpy())
        print("  전체 바     : 지수 b=%.3f  R2=%.4f  n=%d" % (b, r2, n))
        sub = reg[reg["is_liq"]]
        if len(sub) > 30:
            _, b2, r22, n2 = fit_power(sub["ratio"].to_numpy(), sub["push_sig"].to_numpy())
            print("  청산성만    : 지수 b=%.3f  R2=%.4f  n=%d" % (b2, r22, n2))
        print("  b=0.5 이면 제곱근 법칙. R2 가 0 근처면 이 구간에서도 충격항 미검출.")

        print("\n  비율 8분위별 실제 변위 (임계가 있다면 상위 분위에서 꺾여야 한다)")
        reg = reg.copy()
        reg["q"] = pd.qcut(reg["ratio"].rank(method="first"), 8, labels=False)
        tt = reg.groupby("q").agg(n=("ratio", "size"),
                                  ratio_med=("ratio", "median"),
                                  push_med=("push_sig", "median"),
                                  push_p90=("push_sig", lambda s: s.quantile(0.9)),
                                  n_liq=("is_liq", "sum"))
        print(tt.round(4).to_string())

        # 상위 8분위(중앙 0.36) 안에 766 까지가 희석돼 있다. 꼬리를 고정 구간으로 다시 본다.
        print("\n  고정 구간별 — 임계 c 가 있다면 여기서 꺾인다")
        edges = [0.0, 0.31, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
        reg["rb"] = pd.cut(reg["ratio"], edges, right=False)
        t2 = reg.groupby("rb", observed=True).agg(
            n=("ratio", "size"),
            push_med=("push_sig", "median"),
            push_p90=("push_sig", lambda s: s.quantile(0.9)),
            n_liq=("is_liq", "sum"))
        # 최저 구간 대비 배율 — 임계 돌파라면 급등해야 한다
        base = float(t2["push_med"].iloc[0]) if len(t2) else np.nan
        if np.isfinite(base) and base > 0:
            t2["배율"] = t2["push_med"] / base
        print("  (0.31 = Tardis 6일 표본의 상한)")
        print(t2.round(3).to_string())

        big = reg[reg["ratio"] >= 10.0]
        if len(big) > 30:
            _, b3, r23, n3 = fit_power(big["ratio"].to_numpy(), big["push_sig"].to_numpy())
            print("\n  V/D >= 10 만: 지수 b=%.3f  R2=%.4f  n=%d  (중앙 변위 %.2f sigma)"
                  % (b3, r23, n3, float(big["push_sig"].median())))
        else:
            print("\n  V/D >= 10 표본 %d건 — 회귀 불가" % len(big))

        # ---- 통제 1: 분모 오염. 이미 깊이가 무너진 뒤의 '여진 바'를 분리한다.
        #      d_at 이 레짐 기준선의 30% 미만이면 그 바는 캐스케이드 유발이 아니라 잔해다.
        print("\n  [통제1] 분모 오염 — d_at < 0.3*d_ref 인 바(이미 붕괴된 뒤)를 분리")
        reg["thin"] = reg["d_at"] < 0.3 * reg["d_ref"]
        t3 = reg.groupby(["rb", "thin"], observed=True).agg(
            n=("ratio", "size"), push_med=("push_sig", "median"),
            n_liq=("is_liq", "sum"))
        print(t3.round(3).to_string())

        # ---- 통제 2: 변동성 군집. 같은 |z| 안에서 V/D 가 변위를 예측하는가.
        print("\n  [통제2] |z| 고정 — 같은 크기 사건 안에서 V/D 가 변위를 더 설명하는가")
        clean = reg[~reg["thin"]].copy()
        clean["zb"] = pd.cut(clean["z"].abs(), [0, 2, 4, 6, 8, np.inf], right=False)
        clean["rb2"] = pd.cut(clean["ratio"], [0.0, 0.31, 1.0, 3.0, np.inf], right=False)
        piv = clean.pivot_table(index="zb", columns="rb2", values="push_sig",
                                aggfunc="median", observed=True)
        cnt = clean.pivot_table(index="zb", columns="rb2", values="push_sig",
                                aggfunc="size", observed=True)
        print("  변위 중앙(sigma):")
        print(piv.round(3).to_string())
        print("  표본수:")
        print(cnt.fillna(0).astype(int).to_string())
        print("  같은 행(=같은 |z|) 안에서 좌->우로 증가해야 V/D 고유의 효과다.")

        # ---- 통제 2b: 중앙값 표는 극단 셀이 n=16~65 라 부호를 확정할 수 없다.
        #      log-log 회귀로 |z| 를 통제한 V/D 계수의 부호와 유의성을 본다.
        print("\n  [통제2b] log 변위 ~ log|z| + log(V/D), 소표본 셀 대신 회귀로 확인")
        print("  t_iid 는 5분봉 중첩(15분 전방창)과 심볼 공변동을 무시해 부풀려진다.")
        print("  %-12s %9s %8s %9s %9s %7s %6s"
              % ("표본", "b(V/D)", "t_iid", "t_클러스터", "b(|z|)", "n", "일수"))
        for lab, sub in (("전체", clean),
                         ("|z|>=4", clean[clean["z"].abs() >= 4]),
                         ("|z|>=6", clean[clean["z"].abs() >= 6]),
                         ("청산성만", clean[clean["is_liq"]])):
            s = sub[np.isfinite(sub["push_sig"]) & (sub["push_sig"] > 0)
                    & (sub["ratio"] > 0) & np.isfinite(sub["z"]) & (sub["z"].abs() > 0)]
            if len(s) < 60:
                print("  %-12s %9s (n=%d)" % (lab, "표본부족", len(s)))
                continue
            X = np.column_stack([np.ones(len(s)),
                                 np.log(s["ratio"].to_numpy()),
                                 np.log(s["z"].abs().to_numpy())])
            yy = np.log(s["push_sig"].to_numpy())
            co, se, n = ols(X, yy)
            cb, cse, _ = cluster_ols(X, yy, s["day"].to_numpy())
            print("  %-12s %9.4f %8.2f %9.2f %9.4f %7d %6d"
                  % (lab, co[1], co[1] / se[1] if se[1] > 0 else np.nan,
                     cb[1] / cse[1] if cse[1] > 0 else np.nan, co[2], n,
                     int(pd.unique(s["day"]).size)))
        print("  b(V/D) > 0 이면 충격(더 밀림), < 0 이면 되돌림(덜 밀림).")
    else:
        print("  표본 부족 (n=%d)" % len(reg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
