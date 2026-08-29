# -*- coding: utf-8 -*-
"""용량·시장충격 — §6.19 설정이 실제 체결비용을 빼고도 남는가.

왜 이것부터인가
  워크포워드 OOS 건당은 **15.5bp** 다 (`overfit.py` §3). 얇다.
  왕복 슬리피지가 15bp 만 되어도 0 이 된다.
  게다가 우위가 알트에 몰려 있는데(알트 38.4 vs 메이저 26.6bp) 알트가 정확히
  호가창이 얇은 쪽이다. **여기서 죽으면 생존자 편향은 볼 필요도 없다.**

기존 `CAPACITY_FINDINGS.md` 를 그대로 못 쓰는 이유
  그 측정은 **지정가 진입 + 시장가 청산** 기준이었다(진입 슬리피지 0 가정).
  §6.19 설정은 **진입·청산 둘 다 시장가**라 양쪽에서 깊이를 먹는다.

방향 (틀리기 쉬우므로 명시)
  급락 -> 매수 진입.  시장가 **매수**는 **매도호가(ask, dp*)** 를 소모한다.
  h 분 뒤 시장가 **매도**는 **매수호가(bid, dm*)** 를 소모한다.

데이터 (추가 수집 없음. 전부 디스크에 있는 것)
  binance_bulk/book_depth  21종 x 1,318일 (2023-01-01~) x 30초 x ±1~5% 누적 명목가
  binance_bulk/klines_1m   21종 6년 (사건 정의)
  depth_ws                 21종 x 1초 x 12밴드 (30초 해상도 검증용)
  ** book_depth 는 2025년 벤더 결함이 있어 반드시 bookdepth.load_clean 경유. **

실행:
    python analysis/capacity2.py
    python analysis/capacity2.py --d 30 --x 300 --rebuild
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
from analysis.bookdepth import load_clean                              # noqa: E402
from analysis.response_liq import cmean                                # noqa: E402
from analysis.simple_bottom import prep, events, frame                 # noqa: E402
from analysis.slippage import GRID, BID_COLS, ASK_COLS, walk           # noqa: E402

W = 118
HOLDS = (15, 30, 60)
SIZES = np.array([1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6])
MAX_LAG_MS = 90_000            # 스냅샷이 이보다 오래됐으면 버린다 (30초 격자)
CACHE = os.path.join(C.DATA, "analysis", "capacity2")


def _snap(ts, mat, t_query):
    """t_query 이하의 **가장 최근** 스냅샷. 룩어헤드 없음. (행, 지연ms)."""
    k = np.searchsorted(ts, t_query, side="right") - 1
    ok = (k >= 0) & (t_query - ts[np.maximum(k, 0)] <= MAX_LAG_MS)
    out = np.full((len(t_query), mat.shape[1]), np.nan)
    lag = np.full(len(t_query), np.nan)
    idx = np.flatnonzero(ok)
    if len(idx):
        out[idx] = mat[k[idx]]
        lag[idx] = t_query[idx] - ts[k[idx]]
    return out, lag


def build(syms, D, X, gap, rebuild=False):
    """사건별로 진입·청산 시점의 깊이 프로파일을 붙여 캐시한다."""
    os.makedirs(CACHE, exist_ok=True)
    tag = "d%d_x%d_g%d" % (D, int(X), gap)
    p = os.path.join(CACHE, "%s.parquet" % tag)
    if os.path.exists(p) and not rebuild:
        return pd.read_parquet(p)

    out = []
    for s in syms:
        try:
            P1 = prep([s])
        except Exception:                                  # noqa: BLE001
            continue
        if s not in P1:
            continue
        ev = events(P1, D, X, gap, -1)
        if len(ev) < 20:
            continue
        d, LO, CL, HI = frame(P1, ev, -1)
        bd, st = load_clean(s, BID_COLS + ASK_COLS, verbose=False)
        if bd.empty:
            U.log("  %-10s bookDepth 없음" % s)
            continue
        ts = bd["ts_ms"].to_numpy()
        BIDM = bd[BID_COLS].to_numpy(dtype=np.float64)
        ASKM = bd[ASK_COLS].to_numpy(dtype=np.float64)

        # 평시 기준선: **전일** 일중앙값 (인과적. rolling median 은 360만 행에서 멈춘다)
        day = (ts // 86_400_000).astype(np.int64)
        ud, inv = np.unique(day, return_inverse=True)
        base_b = np.array([np.median(BIDM[inv == i, 0]) for i in range(len(ud))])
        base_a = np.array([np.median(ASKM[inv == i, 0]) for i in range(len(ud))])
        prev_b = np.concatenate([[np.nan], base_b[:-1]])
        prev_a = np.concatenate([[np.nan], base_a[:-1]])

        t_in = d["t"].to_numpy()
        A_in, lag_in = _snap(ts, ASKM, t_in)               # 매수 -> ask 소모
        row = {"symbol": s, "t": t_in, "day": d["day"].to_numpy(),
               "trig": d["trig"].to_numpy(), "lag_in": lag_in}
        for i, c in enumerate(ASK_COLS):
            row["in_%s" % c] = A_in[:, i]
        B_in, _ = _snap(ts, BIDM, t_in)                    # 진입 시점 bid (붕괴 진단용)
        for i, c in enumerate(BID_COLS):
            row["inb_%s" % c] = B_in[:, i]
        for h in HOLDS:
            B_out, lag_out = _snap(ts, BIDM, t_in + h * 60_000)   # 청산 -> bid 소모
            row["lag_out%d" % h] = lag_out
            for i, c in enumerate(BID_COLS):
                row["out%d_%s" % (h, c)] = B_out[:, i]
            row["gross%d" % h] = CL[:, h]
        # 전일 기준선
        di = np.searchsorted(ud, t_in // 86_400_000)
        di = np.clip(di, 0, len(ud) - 1)
        row["base_bid"] = prev_b[di]
        row["base_ask"] = prev_a[di]
        out.append(pd.DataFrame(row))
        U.log("  %-10s 사건 %d, 깊이 매칭 %d"
              % (s, len(d), int(np.isfinite(lag_in).sum())))
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    U.atomic_write_parquet(d, p)
    return d


def slip_of(prof_mat, q, cap=True):
    """행별 슬리피지(bp). 프로파일이 단조·양수가 아니면 NaN.

    cap=True 이면 **주문이 5% 격자 안의 총깊이를 넘는 경우 NaN**(=체결 불가)으로
    둔다. walk() 는 격자 밖에서 로그 기울기로 외삽하는데 beta 가 작으면 u* 가
    폭주해 −360억bp 같은 무의미한 값이 나온다. 그건 슬리피지가 아니라
    '그 규모는 이 호가창에서 소화가 안 된다' 는 뜻이므로 그렇게 표기한다.
    """
    n = len(prof_mat)
    out = np.full(n, np.nan)
    for i in range(n):
        pr = prof_mat[i]
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0) or np.any(np.diff(pr) < 0):
            continue
        if cap and q > pr[-1]:                 # 5% 안쪽 총깊이 초과 -> 체결 불가
            continue
        out[i] = walk(pr, q)[1] * 1e4
    return out


def feasible(prof_mat, q):
    """그 규모가 5% 격자 안에서 소화되는 사건 비율."""
    pr = prof_mat
    ok = np.isfinite(pr).all(axis=1) & (pr > 0).all(axis=1)
    return float(np.mean(ok & (pr[:, -1] >= q)))


def sec(n, t):
    print("\n" + "-" * W)
    print("%d. %s" % (n, t))
    print("-" * W)


def main() -> int:
    ap = argparse.ArgumentParser(description="capacity and market impact")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--d", type=int, default=15)
    ap.add_argument("--x", type=float, default=300.0)
    ap.add_argument("--gap", type=int, default=15)
    ap.add_argument("--fee", type=float, default=10.0)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * W)
    print("용량·시장충격 — §6.19 설정이 체결비용을 빼고도 남는가")
    print("=" * W)
    print("설정: %d분 누적 하락 >= %.0fbp -> 다음 봉 시가 **시장가 매수** -> h분 뒤 **시장가 매도**"
          % (a.d, a.x))
    print("방향: 매수는 매도호가(dp*) 소모 / 청산은 매수호가(dm*) 소모")
    print("수수료 왕복 %.0fbp 는 별도. 아래 slip 은 그 위에 얹힌다." % a.fee)
    U.log("사건 + 깊이 결합")
    d = build(syms, a.d, a.x, a.gap, a.rebuild)
    if d.empty:
        print("결합 결과 없음")
        return 1
    ok = np.isfinite(d["lag_in"].to_numpy())
    print("\n사건 %d건 중 깊이 매칭 %d건 (%.1f%%) | 사건 %s ~ %s"
          % (len(d), ok.sum(), 100 * ok.mean(),
             pd.to_datetime(d["t"].min(), unit="ms").strftime("%Y-%m-%d"),
             pd.to_datetime(d["t"].max(), unit="ms").strftime("%Y-%m-%d")))
    yr = pd.to_datetime(d["t"], unit="ms").dt.year.to_numpy()
    n23 = int((yr >= 2023).sum())
    print("** book_depth 는 2023-01 부터다. 2023+ 사건 %d건 중 %d건(%.1f%%) 매칭. **"
          % (n23, int(ok[yr >= 2023].sum()),
             100 * ok[yr >= 2023].mean() if n23 else 0.0))
    # 이 부분표본이 전체를 대표하는가 — 총수익을 비교해 둔다
    for lab, m in (("전체 기간", np.ones(len(d), bool)), ("2023+ (측정 가능)", yr >= 2023)):
        g = d["gross30"].to_numpy()[m]
        g = g[np.isfinite(g)]
        if len(g) < 50:
            continue
        mm, se, t, _ = cmean(g - a.fee, d["day"].to_numpy()[m][np.isfinite(d["gross30"].to_numpy()[m])])
        print("   %-18s n=%6d  총수익-수수료 %6.1fbp (t=%.1f)" % (lab, len(g), mm, t))
    print("   ** 두 값이 크게 다르면 아래 용량 결과는 그 부분표본의 것이다. **")
    d = d[ok].reset_index(drop=True)

    sec(1, "급락 순간 호가창이 얼마나 얇아지나 (전일 중앙값 대비)")
    rb = d["inb_dm1_0"].to_numpy() / d["base_bid"].to_numpy()
    ra = d["in_dp1_0"].to_numpy() / d["base_ask"].to_numpy()
    print("  1%% 이내 누적 명목가 비율. 1.0 이면 평시와 같다.\n")
    print("  %-16s | %s" % ("", " ".join("%7s" % ("p%g" % q)
                                         for q in (5, 25, 50, 75, 95))))
    for lab, v in (("매수호가(청산용)", rb), ("매도호가(진입용)", ra)):
        v = v[np.isfinite(v)]
        print("  %-16s | %s" % (lab, " ".join("%7.2f" % np.percentile(v, q)
                                              for q in (5, 25, 50, 75, 95))))
    print("\n  절대 깊이 (달러, 1%% 이내):")
    print("  %-16s | %s" % ("", " ".join("%9s" % ("p%g" % q)
                                         for q in (5, 25, 50, 75, 95))))
    for lab, c in (("매도호가 진입시", "in_dp1_0"), ("매수호가 청산시(30분)", "out30_dm1_0")):
        v = d[c].to_numpy()
        v = v[np.isfinite(v)]
        print("  %-16s | %s" % (lab, " ".join("%9.0f" % np.percentile(v, q)
                                              for q in (5, 25, 50, 75, 95))))

    sec(2, "★ 규모별 왕복 슬리피지 (bp) — 진입 ask + 청산 bid")
    AI = d[["in_%s" % c for c in ASK_COLS]].to_numpy(dtype=np.float64)
    print("  %-12s | %s" % ("주문 크기", " ".join("%22s" % ("보유 %d분" % h)
                                               for h in HOLDS)))
    print("  %-12s | %s" % ("", " ".join("%22s" % "진입 / 청산 / 왕복(중앙)"
                                         for _ in HOLDS)))
    SL = {}
    for q in SIZES:
        si = slip_of(AI, q)
        cells = []
        for h in HOLDS:
            BO = d[["out%d_%s" % (h, c) for c in BID_COLS]].to_numpy(dtype=np.float64)
            so = slip_of(BO, q)
            tot = si + so
            SL[(q, h)] = tot
            m = np.isfinite(tot)
            cells.append("%22s" % ("%5.1f /%5.1f /%6.1f"
                                   % (np.nanmedian(si), np.nanmedian(so),
                                      np.nanmedian(tot[m]) if m.any() else np.nan)))
        print("  $%-11s | %s" % ("%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6),
                                 " ".join(cells)))

    sec(3, "★★ 슬리피지를 뺀 뒤에도 남는가 — 규모별 순손익")
    print("  순 = 총수익 - 수수료 %.0fbp - 왕복 슬리피지. 일클러스터 t.\n" % a.fee)
    print("  %-12s | %s" % ("주문 크기", " ".join("%26s" % ("보유 %d분: 순bp (t)  달러/건" % h)
                                               for h in HOLDS)))
    day = d["day"].to_numpy()
    for q in SIZES:
        cells = []
        for h in HOLDS:
            g = d["gross%d" % h].to_numpy()
            net = g - a.fee - SL[(q, h)]
            m = np.isfinite(net)
            if m.sum() < 50:
                cells.append("%26s" % "(표본부족)")
                continue
            mm, se, t, _ = cmean(net[m], day[m])
            cells.append("%26s" % ("%8.1f (%5.1f) %8.0f"
                                   % (mm, t, mm * 1e-4 * q)))
        print("  $%-11s | %s" % ("%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6),
                                 " ".join(cells)))
    print("\n  ** 순bp 가 0 을 지나는 크기가 **용량**이다. 달러/건 이 최대인 크기가 최적 규모다. **")
    print("  ** 워크포워드 OOS 기준(건당 15.5bp)으로 읽으려면 위 총수익에서 인샘플 초과분을")
    print("     빼야 한다. 아래 §4 에 그 보정판을 낸다. **")

    sec(4, "★★ 과최적화 보정판 — OOS 건당으로 환산")
    g30 = d["gross30"].to_numpy()
    mm, se, t, _ = cmean(g30 - a.fee, day)
    print("  이 표본(2023-01~)의 인샘플 건당(수수료만 차감, 30분): %.1fbp" % mm)
    print("  워크포워드 OOS 건당(overfit.py §3, 전 기간): 15.5bp")
    shrink = 15.5 / mm if mm > 0 else np.nan
    print("  축소율 = 15.5 / %.1f = **%.2f**" % (mm, shrink))
    print("  아래는 총수익에 축소율을 곱한 뒤 슬리피지를 뺀 값이다 (보수적).\n")
    print("  %-12s | %s" % ("주문 크기", " ".join("%24s" % ("보유 %d분: 순bp  달러/건" % h)
                                               for h in HOLDS)))
    for q in SIZES:
        cells = []
        for h in HOLDS:
            g = d["gross%d" % h].to_numpy() * shrink
            net = g - a.fee - SL[(q, h)]
            m = np.isfinite(net)
            if m.sum() < 50:
                cells.append("%24s" % "(표본부족)")
                continue
            mv = float(np.nanmean(net[m]))
            cells.append("%24s" % ("%9.1f %11.0f" % (mv, mv * 1e-4 * q)))
        print("  $%-11s | %s" % ("%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6),
                                 " ".join(cells)))

    sec(5, "심볼별 용량 — 우위는 알트에 있고 알트가 얇다")
    maj = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}
    print("  %-10s | %6s %10s %10s | %s"
          % ("심볼", "n", "ask깊이1%", "bid깊이1%",
             " ".join("%13s" % ("$%dK 왕복slip" % (q / 1e3)) for q in (1e5, 5e5))))
    for s, g in d.groupby("symbol"):
        ii = g.index.to_numpy()
        cells = []
        for q in (1e5, 5e5):
            v = SL[(q, 30)][ii]
            cells.append("%13.1f" % np.nanmedian(v))
        print("  %-10s | %6d %10.0f %10.0f | %s"
              % (s, len(g), np.nanmedian(g["in_dp1_0"]),
                 np.nanmedian(g["out30_dm1_0"]), " ".join(cells)))
    sec(6, "★★ 군별 순손익 — 인샘플과 OOS 보정판을 나란히")
    print("  체결가능 = 그 규모가 5%% 격자 안 총깊이로 소화되는 사건 비율.")
    print("  '체결불가'는 슬리피지가 아니라 그 규모를 그 호가창에서 못 넣는다는 뜻이다.")
    print("  OOS 보정 = 총수익에 축소율 %.2f 를 곱한 뒤 비용 차감.\n" % shrink)
    AIm = d[["in_%s" % c for c in ASK_COLS]].to_numpy(dtype=np.float64)
    for lab, m in (("메이저 5종", d["symbol"].isin(maj).to_numpy()),
                   ("알트 16종", ~d["symbol"].isin(maj).to_numpy())):
        print("  ── %s (n=%d) ──" % (lab, int(m.sum())))
        print("  %-10s | %7s %9s | %11s %11s | %11s %11s"
              % ("규모", "체결률", "왕복slip", "인샘플bp", "(t)", "OOS보정bp", "달러/건"))
        for q in (1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2.5e6):
            sl = SL[(q, 30)] if (q, 30) in SL else slip_of(AIm, q)
            g = d["gross30"].to_numpy()
            net = g - a.fee - sl
            sel = m & np.isfinite(net)
            fr = feasible(AIm[m], q)
            if sel.sum() < 30:
                print("  $%-9s | %6.1f%% %9s | %11s" % (
                    "%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6),
                    100 * fr, "-", "(표본부족 %d)" % int(sel.sum())))
                continue
            mm2, _, t2, _ = cmean(net[sel], day[sel])
            oos = float(np.nanmean((g[sel] * shrink) - a.fee - sl[sel]))
            print("  $%-9s | %6.1f%% %9.1f | %11.1f %11.1f | %11.1f %11.0f"
                  % ("%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6),
                     100 * fr, float(np.nanmedian(sl[sel])), mm2, t2,
                     oos, oos * 1e-4 * q))
        print()
    print("  ** OOS보정bp 가 0 을 지나는 규모가 실질 용량이다. **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
