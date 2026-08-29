# -*- coding: utf-8 -*-
"""Q2a — 수급 청산 회계로 정지점을 설명할 수 있는가.

가설 (v1 의 올바른 형태)
  캐스케이드는 '아래에서 받아주는 물량이 위에서 쏟아지는 물량을 다 흡수하는 곳'에서
  멈춘다. 호가창을 걸어 내려가는 청산 조건으로 쓰면

      int_0^u* [ D(v) + M(v) ] dv  =  int_0^u* L(v) dv + (외생 매도)

  좌변은 대기 지정매수 + 도착 시장매수, 우변은 강제매도 + 외생매도다.

이 스크립트는 두 단계로 검정한다. 순서가 중요하다.

  (A) 사후 회계 정합성 — 실현값만 쓴다. 예측 문제를 배제하고 **기제 자체**를 본다.
      실제 바닥 X 까지 도착한 순 테이커 매도 NS 를, 트리거 시점에 보이던 호가 깊이
      프로파일 D(u) 로 나눈다. 회계가 성립하면 D(X) ~ NS 여야 한다.
      흡수율 = D(X) / NS.  1 이면 보이던 호가가 전부 설명, 0.01 이면 1% 만 설명.
      **여기서 실패하면 (B)는 볼 필요가 없다.**

  (B) 사전 예측 — 트리거 시점에 관측 가능한 것만으로 X 를 예측하고,
      상수(무조건부 중앙값) 및 변동성 스케일링과 비교한다. 표본을 앞/뒤로 갈라
      앞에서 적합하고 뒤에서 평가한다.

방향
  롱청산(z<0): 가격 하락, 우리가 보는 것은 **매수호가**(dm*), 미는 힘은 테이커 **매도**
  숏청산(z>0): 가격 상승, **매도호가**(dp*), 테이커 **매수**

룩어헤드
  D(u) 프로파일은 트리거 바 **이전** 스냅샷. NS 와 X 는 사후량이며 (A) 는 의도적으로
  사후 회계다. (B) 는 사후량을 일절 쓰지 않는다.

실행:
    python analysis/clearing.py
    python analysis/clearing.py --horizon-min 60 --k 6
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
import analysis.bookdepth as BD                            # noqa: E402
from analysis.event_study_h2 import load, find_events      # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
MIN_MS = 60_000
BAR_MS = 300_000
MAX_SNAP_LAG_MS = 2 * 60_000
# bookDepth 밴드 (누적 명목가). 격자가 이것뿐이라 그 사이는 보간한다.
#
# +-0.2% 는 1,304일 중 197일에만 있어 필수로 쓸 수 없다(analysis/bookdepth.py 참조).
# 따라서 격자는 1~5% 이고, **1% 안쪽은 외삽**이다. 실제 낙폭 X 의 중앙값이 1% 근처라
# 이 외삽이 결과를 좌우한다 — 그래서 +-0.2% 가 있는 197일로 외삽의 정확도를 별도로
# 검증한다(아래 '보간 검증' 블록).
GRID = [0.01, 0.02, 0.03, 0.04, 0.05]
BID_COLS = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK_COLS = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]
OPT_COLS = ["dm0_2", "dp0_2"]        # 있으면 보간 검증에만 쓴다


def load_1m_flow(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s" % symbol)
    cols = ["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    d = pd.read_parquet(p)[cols].sort_values("open_time").reset_index(drop=True)
    d["buy_qv"] = d["taker_buy_quote_volume"].clip(lower=0.0)
    d["sell_qv"] = (d["quote_volume"] - d["taker_buy_quote_volume"]).clip(lower=0.0)
    return d


def depth_at_u(prof: np.ndarray, u: float) -> float:
    """깊이 프로파일(격자 GRID 의 누적 명목가)에서 임의 u 의 누적 깊이.

    격자 안은 로그-선형 보간, 격자 밖(u > 5%)은 마지막 두 점의 로그 기울기로 외삽.
    누적량이므로 단조 증가여야 하며, 위반하면 NaN 을 돌려 그 사건을 버린다.
    """
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
        return np.nan
    if np.any(np.diff(prof) < 0):
        return np.nan
    g = np.asarray(GRID)
    if u <= g[0]:
        return float(prof[0] * u / g[0])          # 최근접 밴드 안은 선형
    if u <= g[-1]:
        return float(np.exp(np.interp(np.log(u), np.log(g), np.log(prof))))
    sl = (np.log(prof[-1]) - np.log(prof[-2])) / (np.log(g[-1]) - np.log(g[-2]))
    return float(prof[-1] * np.exp(sl * (np.log(u) - np.log(g[-1]))))


def u_for_depth(prof: np.ndarray, target: float, u_max: float = 0.50,
                cap: bool = True) -> float:
    """누적 깊이가 target 이 되는 u. depth_at_u 의 역함수(수치 이분법).

    **u_max 까지 가도 흡수가 안 되면 NaN 이 아니라 u_max 를 돌려준다(cap=True).**
    NaN 을 돌려주면 그 사건이 통계에서 조용히 빠지는데, 하필 그것이 회계가 가장 크게
    틀리는 사건들이다 — 실측: 253건 중 84건(33%)이 그렇게 빠졌고, 그 84건의 실제
    낙폭 중앙값은 0.36% 인데 회계는 50% 초과를 예측했다. 빼면 결과가 좋아 보인다.
    """
    if not np.isfinite(target) or target <= 0:
        return np.nan
    lo, hi = 1e-5, u_max
    if depth_at_u(prof, hi) < target:
        return u_max if cap else np.nan            # 흡수 불가 -> 상한으로 기록
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if depth_at_u(prof, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build(symbol: str, k: float, doi_thr: float, min_gap: int,
          horizon_min: int) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m_flow(symbol)
    dep, st = BD.load_clean(symbol, BID_COLS + ASK_COLS, optional=OPT_COLS)
    if df5.empty or m1.empty or dep.empty:
        U.log("%s: 데이터 부족 %s" % (symbol, st.get("missing")))
        return pd.DataFrame()

    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo = m1["low"].to_numpy()
    hi = m1["high"].to_numpy()
    bq = m1["buy_qv"].to_numpy()
    sq = m1["sell_qv"].to_numpy()
    n1 = len(ot)

    t5 = df5["open_time"].to_numpy()
    close5 = df5["close"].to_numpy()
    sig5 = df5["sigma"].to_numpy()
    ret5 = df5["ret"].to_numpy()
    doi5 = df5["doi"].to_numpy()
    oiv5 = df5["sum_open_interest_value"].to_numpy()

    dts = dep["ts_ms"].to_numpy()
    bid = dep[BID_COLS].to_numpy()
    ask = dep[ASK_COLS].to_numpy()
    # +-0.2% 는 일부 날에만 있다 — 보간 검증 전용
    o02 = (dep[OPT_COLS].to_numpy() if all(c in dep.columns for c in OPT_COLS)
           else np.full((len(dep), 2), np.nan))

    out = []
    for r in ev.itertuples():
        i = r.i
        p0 = close5[i]
        if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sig5[i]) and sig5[i] > 0):
            continue
        trig = int(t5[i])

        # 트리거 '이전' 깊이 스냅샷 (룩어헤드 차단)
        j = int(np.searchsorted(dts, trig, side="right")) - 1
        if j < 0 or trig - int(dts[j]) > MAX_SNAP_LAG_MS:
            continue
        prof = (bid[j] if r.side == 1 else ask[j]).astype("float64")

        # 관측 구간: 트리거 봉이 닫힌 직후부터 horizon 분
        a = int(np.searchsorted(ot, trig + BAR_MS, side="left"))
        b = min(a + horizon_min, n1)
        if a >= n1 or b <= a:
            continue

        # 실제 낙폭(또는 상승폭) X
        X = ((1.0 - lo[a:b].min() / p0) if r.side == 1
             else (hi[a:b].max() / p0 - 1.0))
        if not np.isfinite(X) or X <= 0:
            continue

        # 순 테이커 흐름 — 우리를 미는 방향
        flow = (sq[a:b] - bq[a:b]) if r.side == 1 else (bq[a:b] - sq[a:b])
        NS = float(np.nansum(flow))
        NS_gross = float(np.nansum(sq[a:b] if r.side == 1 else bq[a:b]))

        rec = {
            "symbol": symbol, "trig_ms": trig, "side": int(r.side),
            "p0": float(p0), "sigma": float(sig5[i]),
            "bar_ret": float(abs(ret5[i])), "doi": float(doi5[i]),
            "oiv": float(oiv5[i]) if np.isfinite(oiv5[i]) else np.nan,
            "X": float(X),
            "NS": NS, "NS_gross": NS_gross,
            "D_at_X": depth_at_u(prof, float(X)),      # 실제 바닥까지의 누적 대기 호가
            "D_1pct": float(prof[1]) if np.isfinite(prof[1]) else np.nan,
            "u_hat_net": u_for_depth(prof, max(NS, 0.0)),      # 순흐름을 흡수하는 u
            "u_hat_gross": u_for_depth(prof, max(NS_gross, 0.0)),
            # 50% 상한에 걸렸나 = 보이던 호가로는 실현 흐름을 전혀 못 받는 사건
            "capped": bool(np.isfinite(NS) and NS > 0
                           and depth_at_u(prof, 0.50) < NS),
            # 1% 안쪽 외삽의 정확도 검증용 (0.2% 밴드가 있는 날만)
            "d02_actual": float(o02[j, 0 if r.side == 1 else 1]),
            "d02_pred": depth_at_u(prof, 0.002),
        }
        # 깊이 프로파일을 저장한다 — (B) 에서 '예상 흐름'으로 u* 를 다시 풀어야 하는데,
        # 저장하지 않으면 실현 u_hat 을 스케일링하는 근사로 대신하게 되고 그러면
        # 실현값이 새어 들어간다(초판의 결함).
        for gi, gv in enumerate(GRID):
            rec["prof_%d" % gi] = float(prof[gi])
        out.append(rec)
    return pd.DataFrame(out)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 5:
        return np.nan
    y, yhat = y[ok], yhat[ok]
    ss = np.sum((y - yhat) ** 2)
    tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss / tot) if tot > 0 else np.nan


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[ok] - yhat[ok]))) if ok.sum() else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description="Q2a: supply-demand clearing vs realized stop")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--horizon-min", type=int, default=60)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 200)
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS

    frames = []
    for s in symbols:
        try:
            d = build(s, a.k, a.doi, a.min_gap, a.horizon_min)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
            U.log("%s: 이벤트 %d" % (s, len(d)))
    if not frames:
        U.log("no events")
        return 1

    d = pd.concat(frames, ignore_index=True)
    d["day"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "clearing.parquet"))

    print("\n=== 표본 ===")
    print("이벤트 %d | 심볼 %d | 일수 %d | %s ~ %s | 지평 %d분"
          % (len(d), d.symbol.nunique(), d["day"].nunique(),
             d["day"].min(), d["day"].max(), a.horizon_min))
    print("실제 낙폭 X 분위: " + "  ".join(
        "%.0f%%=%.2f%%" % (100 * q, 100 * d["X"].quantile(q))
        for q in (0.1, 0.25, 0.5, 0.75, 0.9)))

    # ------------------------------------------------- 보간 검증 (1% 안쪽)
    v = d[np.isfinite(d["d02_actual"]) & (d["d02_actual"] > 0)
          & np.isfinite(d["d02_pred"]) & (d["d02_pred"] > 0)]
    print("\n=== 보간 검증 — 1%% 안쪽 외삽이 얼마나 맞나 (0.2%% 밴드 보유 %d건) ==="
          % len(v))
    if len(v) >= 10:
        rr = (v["d02_pred"] / v["d02_actual"]).to_numpy(dtype="float64")
        print("  예측/실제 중앙 %.2f  p10 %.2f  p90 %.2f" %
              (np.median(rr), np.quantile(rr, .1), np.quantile(rr, .9)))
        print("  1 보다 크면 1%% 안쪽 깊이를 과대평가 -> u_hat 이 실제보다 얕게 나온다.")
    else:
        print("  표본 부족 — 1%% 안쪽 결과는 검증되지 않은 외삽이다.")

    # ---------------------------------------------------------------- (A)
    print("\n=== (A) 사후 회계 정합성 — 보이던 호가가 실제 흐름을 얼마나 설명하나 ===")
    g = d[np.isfinite(d["D_at_X"]) & np.isfinite(d["NS"]) & (d["NS"] > 0)].copy()
    if len(g) < 20:
        print("  표본 부족 (n=%d)" % len(g))
        return 0
    g["absorb_net"] = g["D_at_X"] / g["NS"]
    g["absorb_gross"] = g["D_at_X"] / g["NS_gross"].where(g["NS_gross"] > 0)
    print("  흡수율 = (트리거 시점 X 까지 누적 대기 호가) / (실제 도착한 흐름)")
    print("  1.0 이면 보이던 호가가 전부 설명. n=%d" % len(g))
    for lab, col in (("순 흐름 대비", "absorb_net"), ("총 흐름 대비", "absorb_gross")):
        x = g[col].to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        if x.size:
            print("    %-10s 중앙 %6.3f   p10 %6.3f   p90 %6.3f   n=%d"
                  % (lab, np.median(x), np.quantile(x, .1), np.quantile(x, .9), x.size))

    print("\n  회계가 예측한 정지점 u_hat vs 실제 X")
    for lab, col in (("순 흐름", "u_hat_net"), ("총 흐름", "u_hat_gross")):
        h = g[np.isfinite(g[col])]
        if len(h) < 10:
            print("    %-8s 표본 부족 (n=%d)" % (lab, len(h)))
            continue
        # 변수명을 (B) 와 겹치지 않게 둔다 — 겹쳐서 y 가 새어 들어간 적이 있다
        ax, ah = h["X"].to_numpy(), h[col].to_numpy()
        rho = float(pd.Series(ax).corr(pd.Series(ah), method="spearman"))
        print("    %-8s 중앙 u_hat %.2f%% vs 실제 X %.2f%%  |  비율 중앙 %.2f  |  "
              "Spearman %+.3f  |  n=%d"
              % (lab, 100 * np.median(ah), 100 * np.median(ax),
                 float(np.median(ah / ax)), rho, len(h)))

    print("\n  해석: 비율 < 1 이면 보이던 호가로는 실제만큼 못 내려간다는 뜻 —")
    print("        즉 호가가 증발하거나 예상보다 많은 흐름이 도착했다.")

    # 흡수 불가(상한 걸림) 사건 — 회계가 가장 크게 틀리는 군이다. 반드시 따로 본다.
    print("\n  회계가 50%% 초과 낙폭을 예측한 사건 (보이던 호가로 실현 흐름을 못 받음)")
    cp = g[g["capped"]] if "capped" in g.columns else g.iloc[:0]
    print("    n=%d / %d (%.0f%%)" % (len(cp), len(g), 100 * len(cp) / max(len(g), 1)))
    if len(cp):
        print("    그 사건들의 **실제** 낙폭: 중앙 %.3f%%  p90 %.3f%%"
              % (100 * cp["X"].median(), 100 * cp["X"].quantile(0.9)))
        print("    나머지의 실제 낙폭:       중앙 %.3f%%"
              % (100 * g.loc[~g["capped"], "X"].median()))
        print("    -> 이 군을 빼면 통계가 좋아 보인다. 빼지 않는다(u_hat 은 50%%로 기록).")

    # 사후 Spearman 은 '큰 사건은 크게 떨어진다'만으로도 나온다. 깊이의 기여를 보려면
    # (1) 흐름 단독, (2) 스칼라 정규화, (3) 심볼 내부와 모두 비교해야 한다.
    h = g[np.isfinite(g["u_hat_net"]) & (g["NS"] > 0) & (g["prof_0"] > 0)
          & np.isfinite(g["oiv"]) & (g["oiv"] > 0)].copy()
    if len(h) >= 20:
        h["ns_oi"] = h["NS"] / h["oiv"]
        h["ns_d1"] = h["NS"] / h["prof_0"]
        cand = [("회계 u_hat (프로파일 5밴드)", "u_hat_net"),
                ("순흐름 NS (정규화 없음)", "NS"),
                ("NS / D(1%) (스칼라 1개)", "ns_d1"),
                ("NS / OI (깊이 미사용)", "ns_oi")]
        print("\n  깊이의 기여 — 풀링 vs 심볼 내부 (n=%d)" % len(h))
        print("    %-26s %10s %10s" % ("예측변수", "풀링", "심볼내부평균"))
        for lab, col in cand:
            pooled = float(h["X"].corr(h[col], method="spearman"))
            per = [hh["X"].corr(hh[col], method="spearman")
                   for _, hh in h.groupby("symbol") if len(hh) >= 15]
            wi = float(np.mean(per)) if per else np.nan
            print("    %-26s %+10.3f %+10.3f" % (lab, pooled, wi))
        print("    풀링만 보면 깊이가 기여하는 듯 보이지만 그것은 심볼 크기 효과다.")
        print("    심볼 내부에서 u_hat 이 'NS/D(1%)' 나 'NS/OI' 를 못 이기면,")
        print("    호가 프로파일의 **형태**는 아무것도 더하지 않는 것이다.")

    # ---------------------------------------------------------------- (B)
    print("\n=== (B) 사전 예측 — 트리거 시점 정보만으로 X 를 맞히나 ===")
    d2 = d.sort_values("trig_ms").reset_index(drop=True)
    cut = len(d2) // 2
    tr, te = d2.iloc[:cut], d2.iloc[cut:]
    print("  훈련 %d건 (%s~%s)  /  평가 %d건 (%s~%s)"
          % (len(tr), tr["day"].iloc[0], tr["day"].iloc[-1],
             len(te), te["day"].iloc[0], te["day"].iloc[-1]))

    y = te["X"].to_numpy(dtype="float64")
    pcols = ["prof_%d" % i for i in range(len(GRID))]
    profs = te[pcols].to_numpy(dtype="float64")

    preds = {}
    preds["상수 (훈련 중앙값)"] = np.full(len(te), float(tr["X"].median()))
    c_sig = float((tr["X"] / tr["sigma"]).median())
    preds["변동성 x %.1f" % c_sig] = c_sig * te["sigma"].to_numpy(dtype="float64")
    c_bar = float((tr["X"] / tr["bar_ret"]).median())
    preds["트리거봉 x %.2f" % c_bar] = c_bar * te["bar_ret"].to_numpy(dtype="float64")

    # 예상 흐름: 훈련구간의 (순흐름 / OI명목가) 중앙값 x 이번 OI명목가.
    # 트리거 시점에 관측 가능한 것만 쓴다.
    ok_tr = tr[np.isfinite(tr["NS"]) & np.isfinite(tr["oiv"]) & (tr["oiv"] > 0)]
    est_ns = np.full(len(te), np.nan)
    if len(ok_tr) >= 10:
        c_flow = float((ok_tr["NS"] / ok_tr["oiv"]).median())
        est_ns = c_flow * te["oiv"].to_numpy(dtype="float64")
        print("  예상 흐름 계수: 순흐름/OI명목가 중앙 %.4f" % c_flow)

        # (a) 흐름만 — 깊이 없이. 훈련에서 X ~ a * NS^b 적합.
        hh = ok_tr[(ok_tr["NS"] > 0) & (ok_tr["X"] > 0)]
        if len(hh) >= 10:
            bb, aa = np.polyfit(np.log(hh["NS"]), np.log(hh["X"]), 1)
            with np.errstate(invalid="ignore"):
                preds["흐름만 (깊이 미사용)"] = np.exp(aa) * np.power(
                    np.clip(est_ns, 1e-9, None), bb)
            print("  흐름만 벤치마크: X = %.3g x NS^%.3f" % (np.exp(aa), bb))

        # (b) 회계 — 저장한 프로파일에 '예상' 흐름을 넣어 u* 를 직접 역산한다.
        #     실현 NS 를 일절 쓰지 않으므로 누출이 없다.
        uh = np.array([u_for_depth(profs[i], est_ns[i]) if np.isfinite(est_ns[i])
                       else np.nan for i in range(len(te))])
        preds["회계 (깊이 + 예상흐름)"] = uh
        # 보간 편의 보정판: 1% 안쪽 깊이를 1.20배 과대평가하므로 흐름을 그만큼 키운다
        uh_adj = np.array([u_for_depth(profs[i], est_ns[i] * 1.20)
                           if np.isfinite(est_ns[i]) else np.nan
                           for i in range(len(te))])
        preds["회계 (보간편의 보정)"] = uh_adj

    # **모든 예측기를 동일 부분표본에서 평가한다.**
    # 초판은 회계만 n=77, 나머지는 n=127 에서 재고 R2 를 나란히 찍었다 — 비교 불가였다.
    common = np.isfinite(y)
    for yh in preds.values():
        common &= np.isfinite(yh)
    n_common = int(common.sum())
    print("  공통 평가 표본 n=%d (전체 평가구간 %d 중)" % (n_common, len(te)))
    if n_common < 20:
        print("  공통 표본 부족 — 비교 불가")
        return 0

    yc = y[common]
    print("\n  %-24s %10s %10s %10s" % ("예측기", "MAE(%)", "R2", "Spearman"))
    for lab, yh in preds.items():
        v = yh[common]
        rho = float(pd.Series(yc).corr(pd.Series(v), method="spearman"))
        print("  %-24s %10.3f %10.3f %10.3f"
              % (lab, 100 * mae(yc, v), r2(yc, v), rho))
    print("  n=%d 동일. R2 는 평가구간 평균 대비 — 음수면 상수보다 못하다." % n_common)
    print("  판정: 회계가 '흐름만'을 못 이기면 깊이는 기여하지 않는 것이고,")
    print("        상수도 못 이기면 Q2a 는 실패다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
