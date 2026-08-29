# -*- coding: utf-8 -*-
"""지정매수 버전 백테스트 — 설계 원문의 실행 방식을 처음으로 평가한다.

왜 이것이 필요한가
  설계 원문: "캐스케이드가 바닥을 칠 가격대를 확률적으로 근사하고
              **그 자리에 지정매수를 걸어둔다**."
  그런데 analysis/backtest.py 는 **바 i+1 시가 시장가**에 왕복 10bp 정액이다
  (backtest.py:81-92). 즉 48.3bp / 샤프 1.16 은 **시장가 버전**의 수치다.
  지정가 버전 — 메이커 수수료를 벌지만 미체결과 역선택을 안는 — 은 미평가다.

무엇이 달라지는가 (셋 다 반대 방향으로 작용한다)
  (+) 더 좋은 가격에 산다        진입가가 kappa*sigma 만큼 아래
  (+) 메이커 수수료             왕복 10bp -> 7bp
  (-) **선택 편향**             도달했을 때만 거래한다. 도달 못하면 기회가 없다
  (-) **역선택**                체결됐다는 것은 가격이 거기까지 갔다는 뜻이다.
                                더 내려갈 사건에서만 체결된다.
  그래서 **체결당**이 아니라 **시도당**으로 재야 한다. 체결당만 보면 생존자 편향이다.

체결 규칙 (1분봉 해상도에서 정직하게 낼 수 있는 최대치)
  매수 지정가 p_lim 은 바 j+lat 부터 W 분간 살아 있다.
      침투:  low[j'] <= p_lim * (1 - delta)   ->  체결. 가격은 min(open[j'], p_lim)
      delta = 0  이면 '스치면 체결'(낙관).  delta > 0 이면 '뚫고 지나가야 체결'.

  *** delta 가 큐모델을 대신한다 — 왜 진짜 큐모델을 못 쓰는가 ***
  hftbacktest 의 ProbQueueModel 은
      est_front = front - (1-p)*chg + min(back - p*chg, 0),  p = f(back)/(f(back)+f(front))
  로 내 앞의 대기물량을 굴린다. 이걸 돌리려면 **내 지정가 레벨의 잔량과 그 레벨에서
  체결된 물량**이 필요한데 우리 과거 데이터에는 없다:
      binance_bulk/book_depth 는 30초 간격에 +-1,2,3,4,5% 뿐이고 (+-0.2% 는 90% 결측)
      우리 오프셋 kappa*sigma 는 대개 1% 안쪽이다.
  그래서 큐 효과를 **관측 가능한 축 하나**로 압축한다: 침투 마진 delta.
      delta = 0   ProbQueueModel 의 낙관 극한 (스치면 체결)
      delta 증가  RiskAdverseQueueModel 쪽 (내 앞이 다 없어져야 체결)
  그 사이 감도를 표로 낸다. 결론이 delta 에 뒤집히면 그 결론은 못 쓰는 것이다.

  마찬가지로 **주문 지연**도 1분봉에서는 관측되지 않는다(밀리초 단위). 대신
  '방아쇠 확인 후 몇 분 뒤에 걸었는가'(lat)를 재서 배치가 시간에 민감한지 본다.

공정 비교를 위한 규칙
  모든 설정이 **같은 시도 집합**을 쓴다. 창 [j, j+lat+W+hold] 이 1분 격자로
  이어지지 않는 방아쇠는 처음부터 전부 뺀다. 그러지 않으면 설정마다 n 이 달라져
  기준선과 비교할 수 없다. kappa=0 은 현행 backtest.py 와 같은 규칙이다.

실행:
    python analysis/limit_fill.py
    python analysis/limit_fill.py --hold 15 --window 30
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
from analysis.response_liq import cmean                         # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")


def _bars(sym: str):
    p = os.path.join(BULK1, "%s.parquet" % sym)
    if not os.path.exists(p):
        return None
    m = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close"])
    m = m.sort_values("open_time").reset_index(drop=True)
    return (m["open_time"].to_numpy(),
            m["open"].to_numpy(dtype=np.float64),
            m["high"].to_numpy(dtype=np.float64),
            m["low"].to_numpy(dtype=np.float64),
            m["close"].to_numpy(dtype=np.float64))


def attempts(symbols, k, doi_thr, gap, span) -> pd.DataFrame:
    """방아쇠 목록. 창 전체가 연속인 건만 남긴다 — 모든 설정이 이 집합을 공유한다."""
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, k, doi_thr, gap)
        if len(ev) == 0:
            continue
        b = _bars(s)
        if b is None:
            continue
        ot1, O, H, L, Cl = b
        n1 = len(ot1)
        ot5 = df["open_time"].to_numpy()
        sig5 = df["sigma"].to_numpy()
        zz = df["z"].to_numpy()
        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5):
                continue
            t_ent = int(ot5[i + 1])
            j = int(np.searchsorted(ot1, t_ent))
            if j >= n1 or ot1[j] != t_ent or j + span >= n1:
                continue
            # *** 창 전체 연속성. 설정마다 n 이 달라지면 비교가 무의미하다 ***
            if ot1[j + span] - ot1[j] != span * 60_000:
                continue
            # 창 안에 결측 봉이 있으면 설정에 따라 행이 빠져 n 이 어긋난다. 미리 뺀다.
            sl = slice(j, j + span + 1)
            if not (np.isfinite(O[sl]).all() and np.isfinite(H[sl]).all()
                    and np.isfinite(L[sl]).all() and np.isfinite(Cl[sl]).all()):
                continue
            sg = float(sig5[i])
            if not (np.isfinite(sg) and sg > 0 and O[j] > 0):
                continue
            out.append({"symbol": s, "t": t_ent, "j": j, "side": sd, "sig5": sg,
                        "z": float(zz[i]), "day": t_ent // 86_400_000,
                        "year": pd.Timestamp(t_ent, unit="ms").year})
    return pd.DataFrame(out)


def simulate_all(att: pd.DataFrame, combos, window: int, hold: int,
                 fee_maker: float, fee_taker: float) -> dict:
    """심볼을 한 번만 읽고 모든 (kappa, delta, lat) 조합을 동시에 돌린다.

    combos: [(kappa, delta_bp, lat), ...]
    반환  : {combo: DataFrame}
    """
    acc = {c: [] for c in combos}
    for s, g in att.groupby("symbol", sort=False):
        b = _bars(s)
        if b is None:
            continue
        ot1, O, H, L, Cl = b
        n1 = len(ot1)
        for r in g.itertuples():
            j, sd, sg = int(r.j), int(r.side), float(r.sig5)
            meta = {"symbol": s, "t": r.t, "side": sd, "sig5": sg, "z": r.z,
                    "day": r.day, "year": r.year}
            for c in combos:
                kappa, delta_bp, lat = c
                if kappa <= 0.0:
                    fj, p_in, fee, wait = j, O[j], 2.0 * fee_taker, 0
                else:
                    p_lim = O[j] * (1.0 - kappa * sg * sd)
                    lo, hi = j + lat, min(j + lat + window, n1)
                    dl = delta_bp * 1e-4
                    if lo >= hi:
                        acc[c].append({**meta, "filled": False, "ret": 0.0,
                                       "wait": np.nan, "mae": np.nan, "edge": np.nan})
                        continue
                    if sd == 1:
                        hit = np.flatnonzero(L[lo:hi] <= p_lim * (1.0 - dl))
                    else:
                        hit = np.flatnonzero(H[lo:hi] >= p_lim * (1.0 + dl))
                    if len(hit) == 0:
                        acc[c].append({**meta, "filled": False, "ret": 0.0,
                                       "wait": np.nan, "mae": np.nan, "edge": np.nan})
                        continue
                    fj = lo + int(hit[0])
                    # 시가가 이미 지정가를 넘었으면 그 가격에 체결된다 (더 유리)
                    p_in = (min(O[fj], p_lim) if sd == 1 else max(O[fj], p_lim))
                    fee, wait = fee_maker + fee_taker, fj - j
                e = fj + hold
                if e >= n1 or not (np.isfinite(p_in) and p_in > 0 and np.isfinite(Cl[e])):
                    continue
                ret = (Cl[e] / p_in - 1.0) * sd * 1e4 - fee
                mae = ((L[fj:e + 1].min() / p_in - 1.0) if sd == 1
                       else -(H[fj:e + 1].max() / p_in - 1.0)) * 1e4
                acc[c].append({**meta, "filled": True, "ret": ret, "wait": wait,
                               "mae": mae,
                               "edge": (O[j] / p_in - 1.0) * sd * 1e4})
    return {c: pd.DataFrame(v) for c, v in acc.items()}


def summarize(d: pd.DataFrame) -> dict:
    """시도당 / 체결당을 **둘 다** 낸다. 체결당만 보면 생존자 편향이다."""
    if len(d) == 0:
        return {}
    n = len(d)
    f = d[d["filled"]]
    r = d["ret"].to_numpy()                          # 미체결은 0
    m_a, _, t_a, _ = cmean(r, d["day"].to_numpy())
    yrs = (d["t"].max() - d["t"].min()) / (365.25 * 86_400_000)
    per_yr = n / yrs if yrs > 0 else np.nan
    eq = np.cumsum(r)
    # *** 샤프는 t / sqrt(년수) 로 낸다 ***
    # backtest.py 의 mean/sd*sqrt(연이벤트수) 는 이벤트가 iid 라고 가정한다.
    # 같은 날 여러 심볼이 동시에 터지므로 횡단면 의존이 있고, 그 공식은 샤프를
    # 부풀린다(기준선에서 2.21 vs 1.16). 일클러스터 t 를 쓰면 그 의존이 반영된다.
    out = {"n": n, "fill": len(f) / n, "att_bp": m_a, "att_t": t_a,
           "sharpe": (t_a / np.sqrt(yrs)) if yrs > 0 else np.nan,
           "maxdd": float((eq - np.maximum.accumulate(eq)).min()),
           "total": float(eq[-1]), "yrs": yrs,
           "fil_bp": np.nan, "fil_t": np.nan, "edge": np.nan,
           "wait": np.nan, "mae": np.nan}
    if len(f) > 20:
        m_f, _, t_f, _ = cmean(f["ret"].to_numpy(), f["day"].to_numpy())
        out.update({"fil_bp": m_f, "fil_t": t_f,
                    "edge": float(f["edge"].median()),
                    "wait": float(f["wait"].median()),
                    "mae": float(f["mae"].median())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="limit-order version of the cascade strategy")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--window", type=int, default=30, help="지정가 유효 시간(분)")
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    ap.add_argument("--kappas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    ap.add_argument("--lats", type=int, nargs="+", default=[0, 1, 2, 5])
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    span = max(a.lats) + a.window + a.hold

    print("=" * 104)
    print("지정매수 버전 — 설계 원문의 실행 방식. kappa=0 은 현행 시장가 기준선과 동일")
    print("=" * 104)
    print("방아쇠 K=%.0f dOI<=%.2f gap=%d | 보유 %d분 | 지정가 유효 %d분"
          % (a.k, a.doi, a.gap, a.hold, a.window))
    print("수수료 메이커 %.0fbp 테이커 %.0fbp -> 시장가 왕복 %.0fbp / 지정가 왕복 %.0fbp"
          % (a.fee_maker, a.fee_taker, 2 * a.fee_taker, a.fee_maker + a.fee_taker))
    print("모든 설정이 같은 시도 집합을 쓴다 (창 %d분 전체가 연속인 방아쇠만)" % span)

    att = attempts(syms, a.k, a.doi, a.gap, span)
    if len(att) < 50:
        print("이벤트 부족 (%d)" % len(att))
        return 1
    print("\n**사용 데이터 기간: %s ~ %s / %d종 / 방아쇠 %d건 (5분봉+1분봉 벌크)**"
          % (str(pd.Timestamp(int(att.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(att.t.max()), unit="ms"))[:10],
             att.symbol.nunique(), len(att)))

    combos = [(0.0, 0.0, 0)]
    for kp in a.kappas:
        if kp <= 0:
            continue
        for dl in a.deltas:
            combos.append((kp, dl, 0))
    kp_ref = a.kappas[min(2, len(a.kappas) - 1)]
    dl_ref = a.deltas[0] if a.deltas else 0.0
    for lt in a.lats:
        if lt != 0:
            combos.append((kp_ref, dl_ref, lt))
    res = simulate_all(att, combos, a.window, a.hold, a.fee_maker, a.fee_taker)

    print("\n" + "-" * 104)
    print("1. 지정가 오프셋 kappa 와 침투 마진 delta")
    print("-" * 104)
    print("  kappa = 진입가를 sigma(5분) 의 몇 배 아래에 두는가 | delta = 뚫어야 하는 마진(bp)")
    print("  ** att_bp(시도당, 미체결=0) 이 기준선과 비교할 값이다. fil_bp(체결당)만 보면 편향. **\n")
    print("  %-6s %-6s | %6s %6s | %9s %6s | %9s %6s | %7s %9s %8s %6s"
          % ("kappa", "delta", "n", "체결률", "att_bp", "t", "fil_bp", "t",
             "샤프", "최대낙폭", "진입이득", "대기"))
    base = summarize(res[(0.0, 0.0, 0)])
    print("  %-6.1f %-6s | %6d %6.3f | %9.2f %6.1f | %9.2f %6.1f | %7.2f %9.0f %8s %6s"
          % (0.0, "시장가", base["n"], base["fill"], base["att_bp"], base["att_t"],
             base["att_bp"], base["att_t"], base["sharpe"], base["maxdd"], "-", "-"))
    for kp in a.kappas:
        if kp <= 0:
            continue
        for dl in a.deltas:
            st = summarize(res[(kp, dl, 0)])
            if not st:
                continue
            print("  %-6.1f %-6.1f | %6d %6.3f | %9.2f %6.1f | %9.2f %6.1f | %7.2f %9.0f %8.1f %6.0f"
                  % (kp, dl, st["n"], st["fill"], st["att_bp"], st["att_t"],
                     st["fil_bp"], st["fil_t"], st["sharpe"], st["maxdd"],
                     st["edge"], st["wait"]))
    print("\n  기준선(kappa=0, 시장가) %.2f bp/시도 (t=%.1f) 샤프 %.2f"
          % (base["att_bp"], base["att_t"], base["sharpe"]))
    print("  backtest.py 보고값 48.3bp (t=2.8) 샤프 1.16 과 대조하라 (표본 필터가 달라 완전 일치는 아니다).")
    print("  ** 결론이 delta 에 뒤집히면 그 결론은 못 쓰는 것이다. **")

    print("\n" + "-" * 104)
    print("2. 배치 지연 민감도 (kappa=%.1f, delta=%.1f 고정)" % (kp_ref, dl_ref))
    print("-" * 104)
    print("  1분봉에서는 밀리초 지연이 안 보인다. '몇 분 뒤에 걸었나'로 시간민감도를 잰다.\n")
    print("  %-8s | %6s %6s | %9s %6s | %7s" % ("지연(분)", "n", "체결률", "att_bp", "t", "샤프"))
    for lt in a.lats:
        key = (kp_ref, dl_ref, lt) if lt != 0 else (kp_ref, dl_ref, 0)
        if key not in res:
            continue
        st = summarize(res[key])
        if st:
            print("  %-8d | %6d %6.3f | %9.2f %6.1f | %7.2f"
                  % (lt, st["n"], st["fill"], st["att_bp"], st["att_t"], st["sharpe"]))

    print("\n" + "-" * 104)
    print("3. 역선택 점검 — 체결된 사건과 안 된 사건이 다른가 (kappa=%.1f, delta=%.1f)"
          % (kp_ref, dl_ref))
    print("-" * 104)
    d = res[(kp_ref, dl_ref, 0)]
    if len(d) and d["filled"].any() and (~d["filled"]).any():
        f, nf = d[d["filled"]], d[~d["filled"]]
        print("  %-10s %6s %9s %11s %10s" % ("", "n", "|z| 중앙", "sigma 중앙", "수익 중앙"))
        print("  %-10s %6d %9.2f %11.5f %10.1f"
              % ("체결", len(f), f.z.abs().median(), f.sig5.median(), f.ret.median()))
        print("  %-10s %6d %9.2f %11.5f %10s"
              % ("미체결", len(nf), nf.z.abs().median(), nf.sig5.median(), "-"))
        print("  체결분 MAE 중앙 %.1f bp | 대기 중앙 %.0f 분 | 진입이득 중앙 %.1f bp"
              % (f["mae"].median(), f["wait"].median(), f["edge"].median()))
        # *** 역선택의 직접 측정 ***
        # 같은 시도 집합이므로 행이 1:1 대응한다. 기준선(시장가) 수익을
        # '지정가가 체결됐을 사건' 과 '안 됐을 사건' 으로 갈라 본다.
        # 갈라진 두 평균이 다르면, 지정가는 **원래 나빴을 사건만 골라 잡은 것**이다.
        b0 = res[(0.0, 0.0, 0)]
        same = (len(b0) == len(d) and (b0["t"].to_numpy() == d["t"].to_numpy()).all()
                and (b0["symbol"].to_numpy() == d["symbol"].to_numpy()).all())
        if same:
            msk = d["filled"].to_numpy()
            r0 = b0["ret"].to_numpy()
            mf, _, tf, _ = cmean(r0[msk], b0["day"].to_numpy()[msk])
            mn, _, tn, _ = cmean(r0[~msk], b0["day"].to_numpy()[~msk])
            print("\n  [직접 측정] 같은 사건을 **시장가로** 잡았을 때의 수익:")
            print("    지정가가 체결된 사건 (%d건)  %+8.2f bp (t=%.1f)" % (msk.sum(), mf, tf))
            print("    체결 안 된 사건   (%d건)  %+8.2f bp (t=%.1f)" % ((~msk).sum(), mn, tn))
            print("    차이 %+.2f bp — 음수면 지정가는 **원래 나쁜 사건만 골라 잡는다**(역선택)."
                  % (mf - mn))
        else:
            print("  [직접 측정] 행 정렬이 어긋나 생략")

    print("\n" + "-" * 104)
    print("4. ★ 이 실패가 알려주는 목표치 — '더 내려갈까' 를 맞히면 얼마인가")
    print("-" * 104)
    print("  3의 결과는 지정가 자체가 나쁘다는 뜻이 아니라, **어디에 걸지 못 골랐다**는 뜻이다.")
    print("  sigma 배수는 '더 내려갈 사건' 과 '즉시 튈 사건' 을 전혀 구분하지 못한다.")
    print("  설계의 ①지도 ②깊이 ③유입취소 는 정확히 그 구분을 하라고 있는 부품이다.")
    print("  그래서 **완전예지 상한**을 계산한다: 도달 여부를 미리 안다면")
    print("    도달 안 할 사건 -> 시장가 (튐을 놓치지 않는다)")
    print("    도달할 사건     -> 지정가 (싸게 산다)\n")
    print("  %-8s | %6s %6s | %10s %10s | %10s %8s"
          % ("kappa", "n_시장", "n_지정", "시장가부분", "지정가부분", "혼합bp", "vs기준선"))
    b0 = res[(0.0, 0.0, 0)]
    for kp in a.kappas:
        if kp <= 0:
            continue
        dd = res[(kp, dl_ref, 0)]
        if not (len(b0) == len(dd) and (b0["t"].to_numpy() == dd["t"].to_numpy()).all()):
            continue
        msk = dd["filled"].to_numpy()
        r0, r1 = b0["ret"].to_numpy(), dd["ret"].to_numpy()
        mix = np.where(msk, r1, r0)          # 체결될 건 지정가, 아닌 건 시장가
        mm, _, tt, _ = cmean(mix, b0["day"].to_numpy())
        print("  %-8.1f | %6d %6d | %10.2f %10.2f | %10.2f %8.2f"
              % (kp, int((~msk).sum()), int(msk.sum()),
                 float(r0[~msk].mean()), float(r1[msk].mean()), mm,
                 mm - base["att_bp"]))
    print("\n  *** 이것은 전방정보를 쓴 상한이다. 전략이 아니다. ***")
    print("  의미: ①②③ 이 '더 내려갈까' 를 완벽히 맞히면 기준선 대비 이만큼이 최대다.")
    print("  이 폭이 작으면 부품을 아무리 잘 만들어도 지정가로 얻을 게 없다는 뜻이다.")

    print("\n" + "-" * 104)
    print("5. 연도별 — 기준선 대비 (kappa=%.1f, delta=%.1f)" % (kp_ref, dl_ref))
    print("-" * 104)
    print("  %6s %7s %12s %12s %10s" % ("연도", "n", "시장가bp", "지정가bp", "차이"))
    for y in sorted(set(b0["year"])):
        g0, g1 = b0[b0.year == y], d[d.year == y]
        if len(g0) < 5:
            continue
        print("  %6d %7d %12.1f %12.1f %10.1f"
              % (y, len(g0), g0.ret.mean(), g1.ret.mean(), g1.ret.mean() - g0.ret.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
