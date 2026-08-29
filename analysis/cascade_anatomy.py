# -*- coding: utf-8 -*-
"""캐스케이드 해부 — "청산이 얇아지는 곳에서 멈춘다"를 과거 데이터로 직접 검정.

검정할 시나리오
  두꺼운 청산 군집 통과 -> 강제청산 -> 오버슈팅 -> **청산이 얇아지는 구간에서 정지**

왜 과거 지도가 필요 없는가
  이 시나리오를 검정하려면 (a) 가격 경로, (b) 경로를 따라 실제로 얼마나 청산됐나,
  (c) 정지 시점이 (b)의 고갈과 일치하나 — 이 셋이면 된다. '사전 지도'가 아니라
  '사후 소진량'이다. 그리고 (b)는 Binance metrics의 5분 OI 변화가 그대로 준다.
  OI가 줄어든 만큼이 그 구간에서 실제로 청산·축소된 포지션이다.

  이전 stopping_hazard.py는 거래량 프로파일로 지도를 '복원'하려다 실패했다
  (복원값이 거래량 노드와 구별되지 않아 정반대 메커니즘이 상쇄됨). 이건 복원이
  아니라 실측 디레버리징이라 그 교락이 없다.

핵심 정의
  캐스케이드 = 트리거 바(k시그마 급변 + OI 급감) 이후, 같은 방향 가격 진행 + OI 감소가
              이어지는 연속 구간.
  소진 강도 = |dOI_usd| / |가격변화%|   (가격 1% 움직일 때 청산된 명목가)
              선생님 표현의 "그 가격대에 쌓인 청산 물량"의 실측 대응물.

두 가지 검정
  A) 종료 바의 소진 강도가 진행 중 바보다 유의하게 낮은가?  -> "얇아지는 곳에서 멈춘다"
  B) 해저드: 현재 바의 소진 강도가 높을수록 다음 바로 계속될 확률이 높은가?
     이게 실시간으로 쓸 수 있는 형태다(현재 바까지만 보고 판단).

실행:
    python analysis/cascade_anatomy.py
    python analysis/cascade_anatomy.py --k 6 --max-len 24
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
from analysis.event_study_h2 import load, find_events, nw_tstat   # noqa: E402


def trace(df: pd.DataFrame, i0: int, side: int, max_len: int,
          retrace_stop: float) -> list[dict]:
    """트리거 바 i0에서 극단점(캐스케이드 종착점)까지의 경로를 바 단위로 기록.

    캐스케이드 정의 (수정판)
      1차 정의("매 바마다 같은 방향 + OI 감소")는 한 바만 쉬어도 끊겨 추적 길이
      중앙값이 2바에 그쳤다. 2바를 이벤트 내 평균으로 정규화하면 두 값이 기계적으로
      반대 부호가 되어 검정력이 사라진다(실측으로 확인).

      수정: 극단점에서 retrace_stop 만큼 되돌리기 전까지를 캐스케이드로 본다.
      중간에 잠시 쉬거나 반등해도 다시 밀면 계속으로 친다. 종착점 = 극단점 바.
    """
    ret = df["ret"].to_numpy()
    doi = df["doi"].to_numpy()
    oiv = df["sum_open_interest_value"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()
    contig = df["contig"].to_numpy()
    n = len(df)

    p0 = close[i0]
    if not (np.isfinite(p0) and p0 > 0):
        return []

    # 극단점 탐색: 되돌림이 retrace_stop을 넘으면 그 전까지가 캐스케이드
    ext = p0
    ext_step = 0
    end_step = 0
    for step in range(1, max_len + 1):
        t = i0 + step
        if t >= n or not contig[t]:
            break
        px_ext = low[t] if side == 1 else high[t]
        if not np.isfinite(px_ext):
            break
        better = (px_ext < ext) if side == 1 else (px_ext > ext)
        if better:
            ext, ext_step = px_ext, step
        # 극단 대비 되돌림
        move = abs(ext / p0 - 1.0)
        back = (close[t] - ext) / ext if side == 1 else (ext - close[t]) / ext
        if move > 1e-9 and back > retrace_stop * move:
            break
        end_step = step
    end_step = max(ext_step, min(end_step, max_len))
    if end_step < 2:
        return []

    out = []
    for step in range(0, end_step + 1):
        t = i0 + step
        if t + 1 >= n or not contig[t]:
            break
        r, d = ret[t], doi[t]
        if not (np.isfinite(r) and np.isfinite(d)):
            continue
        # 이 바에서 소진된 명목가 (OI 감소분). OI가 늘어난 바는 청산이 아니므로 0.
        burned = max(-d, 0.0) * (oiv[t] if np.isfinite(oiv[t]) else np.nan)
        pct = abs(r) * 100.0
        out.append({
            "step": step, "idx": t, "ret_pct": r * 100.0, "doi": d,
            "burned_usd": burned,
            # 가격 1% 움직일 때 청산된 명목가 — '그 가격대의 청산 물량'의 실측치
            "intensity": burned / pct if pct > 1e-9 else np.nan,
            "is_ext": step == ext_step,
        })
    return out


def build(symbols: list[str], k: float, doi_thr: float, min_gap: int,
          max_len: int, retrace_stop: float) -> pd.DataFrame:
    rows = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        # OI 명목가는 metrics의 sum_open_interest_value 사용
        if "sum_open_interest_value" not in df.columns:
            U.log("%s: no sum_open_interest_value -> skip" % s)
            continue
        ev = find_events(df, k, doi_thr, min_gap)
        ev = ev[ev.is_liq]
        n_traced = 0
        for r in ev.itertuples():
            path = trace(df, r.i, r.side, max_len, retrace_stop)
            if len(path) < 2:
                continue
            n_traced += 1
            last = len(path) - 1
            for p in path:
                p.update({"symbol": s, "event_i": int(r.i), "side": int(r.side),
                          "n_steps": len(path), "is_last": p["step"] == last,
                          "open_time": int(df.open_time.iat[r.i])})
                rows.append(p)
        U.log("%s: %d liq events, %d traced (len>=2)" % (s, len(ev), n_traced))
    return pd.DataFrame(rows)


def test_a(d: pd.DataFrame) -> None:
    """종료 바의 소진 강도가 진행 바보다 낮은가."""
    print("\n=== 검정 A: 캐스케이드는 '청산이 얇아지는 곳'에서 멈추는가 ===")
    d = d[np.isfinite(d["intensity"])]
    if d.empty:
        print("no data"); return
    # 이벤트 내 상대값으로 정규화 — 심볼/시기별 절대 규모 차이를 제거
    d = d.copy()
    d["rel"] = d.groupby(["symbol", "event_i"])["intensity"].transform(
        lambda s: s / s.mean() if s.mean() > 0 else np.nan)
    d = d[np.isfinite(d["rel"])]
    run = d[~d.is_ext]["rel"]
    end = d[d.is_ext]["rel"]
    print("진행 중 바: n=%5d  평균 상대강도 %.3f  중앙값 %.3f" % (len(run), run.mean(), run.median()))
    print("정지 바   : n=%5d  평균 상대강도 %.3f  중앙값 %.3f" % (len(end), end.mean(), end.median()))
    if len(run) > 10 and len(end) > 10:
        from scipy import stats
        t, p = stats.ttest_ind(end, run, equal_var=False)
        u, pu = stats.mannwhitneyu(end, run, alternative="less")
        print("Welch t = %.2f (p=%.4g) | Mann-Whitney (end<run) p=%.4g" % (t, p, pu))
        print("예측: 정지 바가 낮아야 한다 -> t 음수, p 작음")

    print("\n-- 스텝별 상대 소진강도 (캐스케이드 진행에 따른 감쇠) --")
    g = d.groupby("step")["rel"].agg(n="size", mean="mean", median="median")
    print(g[g["n"] >= 20].round(3).to_string())


def test_b(d: pd.DataFrame) -> None:
    """해저드: 현재 바의 소진 강도가 다음 바 진행을 예측하는가 (실시간 사용 가능 형태)."""
    print("\n=== 검정 B: 현재 소진강도가 '계속 갈지'를 예측하는가 (실시간형) ===")
    d = d[np.isfinite(d["intensity"])].copy()
    if d.empty:
        print("no data"); return
    d["cont"] = ~d["is_ext"]
    d["rel"] = d.groupby(["symbol", "event_i"])["intensity"].transform(
        lambda s: s / s.mean() if s.mean() > 0 else np.nan)
    d = d[np.isfinite(d["rel"])]
    if len(d) < 60:
        print("n=%d too small" % len(d)); return
    d["q"] = pd.qcut(d["rel"].rank(method="first"), 4,
                     labels=["Q1 thin", "Q2", "Q3", "Q4 thick"])
    t = d.groupby("q", observed=True).agg(n=("cont", "size"),
                                          P_continue=("cont", "mean"),
                                          mean_rel=("rel", "mean"))
    print(t.round(4).to_string())
    print("예측: thin -> P(continue) 낮음, thick -> 높음. 단조 증가해야 시나리오 성립.")


def main() -> int:
    ap = argparse.ArgumentParser(description="cascade anatomy: does it stop where liquidations thin out?")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--doi", type=float, default=-0.01)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=24, help="max bars to trace (5m each)")
    ap.add_argument("--retrace-stop", type=float, default=0.5,
                    help="cascade ends when price retraces this fraction of its move")
    a = ap.parse_args()

    U.init_stdout()
    symbols = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS
    U.log("cascade anatomy: k=%.1f dOI<=%.3f max_len=%d" % (a.k, a.doi, a.max_len))

    d = build(symbols, a.k, a.doi, a.min_gap, a.max_len, a.retrace_stop)
    if d.empty:
        U.log("no cascades traced")
        return 1
    U.atomic_write_parquet(d, os.path.join(C.DATA, "analysis", "cascade_paths.parquet"))

    ev = d.groupby(["symbol", "event_i"]).first()
    print("\ntraced cascades: %d | 길이 분포(바):" % len(ev))
    print(ev["n_steps"].describe().round(2).to_string())

    pd.set_option("display.width", 200)
    test_a(d)
    test_b(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
