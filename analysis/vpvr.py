# -*- coding: utf-8 -*-
"""매물대(거래량 프로파일)로 바닥을 잡을 수 있는가 — 그리고 그게 돈이 되는가.

사용자 가설 (2026-08-16)
  "선물은 현물가격을 기초로 하니 청산 캐스케이드의 바닥은 곧 반등이고,
   이는 보통 현물의 매물대에서 저항이 일어나는 걸로 볼 수 있지 않을까?
   그러면 매물대까지 고려하면 바닥 예측력이 높아져 수익률을 높일 수 있을 것 같다."

먼저 짚을 것 — '바닥을 더 잘 맞히면 돈이 된다' 는 이미 두 번 반증됐다
  1. `dyn_entry.py`  체결후 MAE 2.8배 개선, 수익 3배 악화 (DESIGN_LOCK §3.8, §5.13)
  2. `simple_bottom.py` §3  지정가 −25~−200bp, **이벤트당** 기준으로 시장가에 못 미침
  바닥에 가까이 걸면 체결가는 좋아지나 미체결이 늘어 상쇄되고, 걸리는 건은
  '더 떨어진' 건이라 역선택이 붙는다.

그런데 용량 측정이 이 계산을 바꿨다 (`capacity2.py`)
  이전 비교는 **시장가 진입에 슬리피지를 안 물렸다.** 실측은 $250K 12.7bp,
  $500K 25.4bp 다. 지정가는 그것을 0 으로 만들고 수수료도 테이커5 -> 메이커2 다.
  -> 매물대가 유효할 수 있는 경로는 "예측이 좋아져서" 가 아니라
     **"메이커로 들어가 진입 슬리피지를 안 내서"** 다. 그 축으로 검정한다.
  용량 자체는 안 바뀐다. 청산은 여전히 시장가다.

근사 (명시)
  현물 klines 가 없어 **선물 quote_volume** 으로 매물대를 만든다. 선물·현물
  프로파일은 강하게 상관되고, 실제 체결되는 것은 선물 호가창이다.

실행:
    python analysis/vpvr.py
    python analysis/vpvr.py --lookback 30 --x 300
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
from analysis.response_liq import cmean, ols_cluster                   # noqa: E402
from analysis.simple_bottom import events, frame, NOBS                 # noqa: E402
from analysis.slippage import BID_COLS, ASK_COLS, walk                 # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
CAP = os.path.join(C.DATA, "analysis", "capacity2")
W = 118
BP = 10.0                      # 가격 버킷 폭 (bp)
SPAN = 1500.0                  # 진입가 아래 이만큼(bp)까지 매물대를 본다
SIZES = (1e5, 2.5e5, 5e5)
SHRINK = 0.39                  # overfit.py 워크포워드 축소율 (15.5 / 39.9)


def prep_v(syms):
    """simple_bottom.prep 과 같되 quote_volume 을 같이 싣는다."""
    out = {}
    for s in syms:
        p = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p):
            continue
        m = pd.read_parquet(p, columns=["open_time", "open", "high", "low",
                                        "close", "quote_volume"])
        m = m.sort_values("open_time").reset_index(drop=True)
        out[s] = (m["open_time"].to_numpy(),
                  m["open"].to_numpy(dtype=np.float64),
                  m["high"].to_numpy(dtype=np.float64),
                  m["low"].to_numpy(dtype=np.float64),
                  m["close"].to_numpy(dtype=np.float64),
                  m["quote_volume"].to_numpy(dtype=np.float64))
    return out


def profile(logp, qv, a, b, ref_log, nb):
    """[a,b) 구간 봉들의 거래대금을 진입가 기준 상대 로그가격 버킷에 담는다.

    버킷 k = 진입가 대비 -(k+0.5)*BP bp. k=0..nb-1 (진입가 아래만 본다).
    봉의 거래대금은 종가 버킷에 통째로 넣는다 (표준 VPVR 근사).
    """
    rel = (ref_log - logp[a:b]) * 1e4          # 양수 = 진입가보다 낮음
    k = np.floor(rel / BP).astype(np.int64)
    m = (k >= 0) & (k < nb)
    if not m.any():
        return np.zeros(nb)
    return np.bincount(k[m], weights=qv[a:b][m], minlength=nb)[:nb]


def build(syms, D, X, gap, lookback_d):
    """심볼별로 (사건표, 역행경로, 종가경로) 를 낸다. 사건표에 매물대 특징을 붙인다."""
    P = prep_v(syms)
    LB = lookback_d * 1440
    nb = int(SPAN / BP)
    for s, (ot, O, H, L, Cl, QV) in P.items():
        P1 = {s: (ot, O, H, L, Cl)}
        ev = events(P1, D, X, gap, -1)
        if len(ev) < 20:
            continue
        d, LO, CL, HI = frame(P1, ev, -1)
        if len(d) != len(ev):                  # frame 이 거르면 정렬이 깨진다
            raise RuntimeError("frame/events 길이 불일치: %d vs %d" % (len(d), len(ev)))
        logp = np.log(np.maximum(Cl, 1e-12))
        hvn = np.full(len(ev), np.nan)
        share = np.full(len(ev), np.nan)
        void = np.full(len(ev), np.nan)
        for r, (ss, i, j, r0) in enumerate(ev):
            a = max(0, i - LB)
            prof = profile(logp, QV, a, i + 1, np.log(O[j]), nb)
            tot = prof.sum()
            if tot <= 0:
                continue
            kmax = int(np.argmax(prof))
            cum = np.cumsum(prof) / tot
            k10 = int(np.searchsorted(cum, 0.10))
            hvn[r] = (kmax + 0.5) * BP                 # 진입가 아래 bp
            share[r] = float(prof[kmax] / tot)
            void[r] = (min(k10, nb - 1) + 0.5) * BP    # 누적 10% 지점
        d["hvn"], d["hvn_share"], d["void"] = hvn, share, void
        yield s, d, LO, CL


def slip_row(prof_mat, q):
    n = len(prof_mat)
    out = np.full(n, np.nan)
    for i in range(n):
        pr = prof_mat[i]
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0) or np.any(np.diff(pr) < 0):
            continue
        if q > pr[-1]:
            continue
        out[i] = walk(pr, q)[1] * 1e4
    return out


def sec(n, t):
    print("\n" + "-" * W)
    print("%d. %s" % (n, t))
    print("-" * W)


def main() -> int:
    ap = argparse.ArgumentParser(description="volume profile as bottom predictor")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--d", type=int, default=15)
    ap.add_argument("--x", type=float, default=300.0)
    ap.add_argument("--gap", type=int, default=15)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--wait", type=int, default=15)
    ap.add_argument("--taker", type=float, default=5.0)
    ap.add_argument("--maker", type=float, default=2.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * W)
    print("매물대(거래량 프로파일)로 바닥을 잡을 수 있는가")
    print("=" * W)
    print("방아쇠 %d분 하락>=%.0fbp | 매물대 = 직전 %d일 1분봉 quote_volume, 버킷 %.0fbp"
          % (a.d, a.x, a.lookback, BP))
    print("근사: **선물** 거래량으로 매물대를 만든다 (현물 klines 없음)")
    print("보유 %d분 | 지정가 대기 %d분 | 테이커 %.0fbp/편, 메이커 %.0fbp/편"
          % (a.hold, a.wait, a.taker, a.maker))

    D, LOa, CLa = [], [], []
    for s, d, LO, CL in build(syms, a.d, a.x, a.gap, a.lookback):
        D.append(d)
        LOa.append(LO)
        CLa.append(CL)
        U.log("  %-10s %d건" % (s, len(d)))
    if not D:
        print("사건 없음")
        return 1
    d = pd.concat(D, ignore_index=True)
    LO = np.vstack(LOa)
    CL = np.vstack(CLa)
    print("\n사건 %d건 / 심볼 %d종 | %s ~ %s"
          % (len(d), d["symbol"].nunique(),
             pd.to_datetime(d["t"].min(), unit="ms").strftime("%Y-%m-%d"),
             pd.to_datetime(d["t"].max(), unit="ms").strftime("%Y-%m-%d")))

    sec(1, "매물대는 어디에 있고 바닥은 어디인가")
    print("  진입가 아래 bp. HVN = 최대 거래량 버킷, void = 누적 거래량 10% 지점.\n")
    print("  %-16s | %s" % ("", " ".join("%8s" % ("p%g" % q)
                                         for q in (10, 25, 50, 75, 90))))
    for lab, c in (("HVN 거리", "hvn"), ("void 거리", "void"),
                   ("실제 바닥(%d분)" % a.hold, None)):
        if c is None:
            v = -LO[:, :a.hold + 1].min(axis=1)
        else:
            v = d[c].to_numpy()
        v = v[np.isfinite(v)]
        print("  %-16s | %s" % (lab, " ".join("%8.0f" % np.percentile(v, q)
                                              for q in (10, 25, 50, 75, 90))))
    bot = LO[:, :a.hold + 1].min(axis=1)
    hv = d["hvn"].to_numpy()
    m = np.isfinite(hv) & np.isfinite(bot)
    print("\n  바닥이 HVN 보다 얕은(= HVN 도달 전에 반등) 비율: %.1f%%"
          % (100 * np.mean(-bot[m] < hv[m])))
    print("  상관 corr(HVN 거리, 바닥 깊이) = %.3f"
          % np.corrcoef(hv[m], -bot[m])[0, 1])

    sec(2, "예측력 — 바닥 깊이 회귀에 넣으면 증분 R2 가 있나")
    y = -bot
    day = d["day"].to_numpy()
    base_f = ["trig", "vol1m"]
    F0 = np.column_stack([np.ones(len(d))] +
                         [d[c].to_numpy(dtype=np.float64) for c in base_f])
    ok0 = np.isfinite(F0).all(axis=1) & np.isfinite(y)
    b0, _, _ = ols_cluster(F0[ok0], y[ok0], day[ok0])
    r0 = 1 - np.var(y[ok0] - F0[ok0] @ b0) / np.var(y[ok0])
    print("  기준(방아쇠 크기 + 직전 변동성): R2 %.4f  n=%d" % (r0, int(ok0.sum())))
    print("  %-22s | %9s %7s %10s" % ("추가 변수", "계수", "t", "증분R2"))
    for c in ("hvn", "hvn_share", "void"):
        x = d[c].to_numpy(dtype=np.float64)
        ok = ok0 & np.isfinite(x)
        xs = (x[ok] - np.nanmean(x[ok])) / (np.nanstd(x[ok]) + 1e-12)
        F1 = np.column_stack([F0[ok], xs])
        b1, se1, _ = ols_cluster(F1, y[ok], day[ok])
        r1 = 1 - np.var(y[ok] - F1 @ b1) / np.var(y[ok])
        bb, _, _ = ols_cluster(F0[ok], y[ok], day[ok])
        rr0 = 1 - np.var(y[ok] - F0[ok] @ bb) / np.var(y[ok])
        print("  %-22s | %9.1f %7.1f %10.4f"
              % (c, b1[-1], b1[-1] / se1[-1] if se1[-1] > 0 else np.nan, r1 - rr0))
    print("\n  ** 증분 R2 가 0.01 수준이면 예측에 쓸 수 없다. **")

    sec(3, "★★ 슬리피지를 물린 뒤 — 시장가 vs 고정지정가 vs 매물대지정가")
    print("  이벤트당 기준(미체결=0). 청산은 셋 다 시장가라 청산 슬리피지는 공통.")
    print("  시장가만 **진입 슬리피지**를 낸다. 그게 이 비교의 핵심이다.\n")
    # capacity2 캐시에서 사건별 깊이 프로파일을 붙인다
    cp = os.path.join(CAP, "d%d_x%d_g%d.parquet" % (a.d, int(a.x), a.gap))
    if not os.path.exists(cp):
        print("  capacity2 캐시 없음(%s). 슬리피지 없는 비교만 낸다." % os.path.basename(cp))
        cap = None
    else:
        cap = pd.read_parquet(cp)
        cap = cap[np.isfinite(cap["lag_in"])]
        key = cap["symbol"].astype(str) + "_" + cap["t"].astype(str)
        cap = cap.assign(_k=key).set_index("_k")
    kd = d["symbol"].astype(str) + "_" + d["t"].astype(str)
    has = kd.isin(cap.index) if cap is not None else pd.Series(False, index=d.index)
    print("  깊이 매칭 %d/%d 건 (2023-01 이후만)" % (int(has.sum()), len(d)))
    sub = np.flatnonzero(has.to_numpy())
    if len(sub) < 100:
        print("  매칭 표본 부족")
        return 0
    cs = cap.loc[kd.iloc[sub].to_numpy()]
    AI = cs[["in_%s" % c for c in ASK_COLS]].to_numpy(dtype=np.float64)
    BO = cs[["out%d_%s" % (a.hold, c) for c in BID_COLS]].to_numpy(dtype=np.float64)
    dsub = d.iloc[sub].reset_index(drop=True)
    LOs, CLs = LO[sub], CL[sub]
    dys = dsub["day"].to_numpy()
    hvs = dsub["hvn"].to_numpy()

    for q in SIZES:
        si = slip_row(AI, q)
        so = slip_row(BO, q)
        print("  ── 규모 $%s (진입slip 중앙 %.1fbp / 청산slip 중앙 %.1fbp) ──"
              % ("%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6),
                 np.nanmedian(si), np.nanmedian(so)))
        print("  %-20s | %7s | %9s %7s | %10s"
              % ("진입 방식", "체결률", "이벤트당bp", "t", "달러/건"))
        # (a) 시장가
        net = CLs[:, a.hold] - 2 * a.taker - si - so
        mk = np.isfinite(net)
        mm, _, t, _ = cmean(net[mk], dys[mk])
        print("  %-20s | %6.1f%% | %9.1f %7.1f | %10.0f"
              % ("시장가 즉시", 100.0, mm, t, mm * 1e-4 * q))
        # (b) 고정 오프셋 지정가 / (c) 매물대 지정가
        for lab, lv in ([("-%dbp 지정가" % k, np.full(len(dsub), float(k)))
                         for k in (50, 100, 200)]
                        + [("매물대(HVN) 지정가", hvs),
                           ("매물대 x0.5", hvs * 0.5)]):
            fill = np.full(len(dsub), -1)
            good = np.isfinite(lv) & (lv > 0)
            for e in np.flatnonzero(good):
                w = LOs[e, :a.wait + 1]
                hit = np.flatnonzero(w <= -lv[e])
                if len(hit):
                    fill[e] = hit[0]
            okf = fill >= 0
            r = np.zeros(len(dsub))
            idx = np.flatnonzero(okf)
            hh = np.minimum(fill[idx] + a.hold, NOBS)
            r[idx] = (CLs[idx, hh] + lv[idx]) - a.maker - a.taker - so[idx]
            mk2 = np.isfinite(r)
            mm2, _, t2, _ = cmean(r[mk2], dys[mk2])
            print("  %-20s | %6.1f%% | %9.1f %7.1f | %10.0f"
                  % (lab, 100 * okf.mean(), mm2, t2, mm2 * 1e-4 * q))
        print()
    print("  ** 지정가가 시장가를 이기려면 (진입슬립 절약 + 메이커 수수료) 가")
    print("     (미체결 손실 + 역선택) 을 넘어야 한다. 규모가 클수록 유리해진다. **")

    sec(4, "★★ 지정가를 그 규모로 정말 채울 수 있나 — 유량 상한")
    print("  지정매수는 **테이커 매도**가 와야 채워진다. 상한은 우리 지정가 이하에서")
    print("  실제 발생한 테이커 매도 대금이다:  F = sum(quote_volume - taker_buy_quote)")
    print("  체결 봉 하나만이 아니라 **지정가 이하로 거래된 모든 봉**을 센다.")
    print("  (호가창 깊이가 아니라 유량이다 — 지정가는 깊이를 소모하지 않는다)\n")
    Pv = prep_v(syms)
    print("  %-20s | %7s | %s" % ("지정가 수준", "체결률",
                                  " ".join("%12s" % ("F p%g" % q)
                                           for q in (10, 25, 50, 75))))
    for lab, lv in (("-50bp", np.full(len(dsub), 50.0)),
                    ("-100bp", np.full(len(dsub), 100.0)),
                    ("-200bp", np.full(len(dsub), 200.0)),
                    ("매물대(HVN)", hvs)):
        F, nfill = [], 0
        for e in range(len(dsub)):
            q_ = lv[e]
            if not np.isfinite(q_) or q_ <= 0:
                continue
            w = LOs[e, :a.wait + 1]
            hit = np.flatnonzero(w <= -q_)
            if not len(hit):
                continue
            nfill += 1
            s_, i_ = dsub["symbol"].iloc[e], int(dsub["i"].iloc[e])
            ot, O, H, L, Cl, QV = Pv[s_]
            j_ = i_ + 1
            lim = O[j_] * (1.0 - q_ * 1e-4)
            u0, u1 = j_, min(j_ + a.wait, len(Cl) - 1)
            sel = np.arange(u0, u1 + 1)
            sel = sel[L[sel] <= lim]
            if not len(sel):
                continue
            # 그 봉들의 테이커 매도 대금 (근사: 봉 전체 매도 대금)
            qv = QV[sel]
            # taker_buy 는 prep_v 에 없으므로 보수적으로 절반을 매도로 본다
            F.append(float(np.sum(qv) * 0.5))
        if not F:
            print("  %-20s | %6.1f%% | (체결 없음)" % (lab, 0.0))
            continue
        F = np.array(F)
        print("  %-20s | %6.1f%% | %s"
              % (lab, 100 * nfill / len(dsub),
                 " ".join("%12.0f" % np.percentile(F, q)
                          for q in (10, 25, 50, 75))))
    print("\n  ** F 의 p10 이 목표 규모보다 크면 그 규모는 유량 제약을 받지 않는다.")
    print("     실제로는 그 유량의 일부만 우리 것이므로 alpha(=우리 몫) 를 곱해야 한다. **")

    sec(5, "★★ OOS 보정 후 사업 규모 — 지정가 진입 기준")
    print("  지정가 체결가는 한계가와 같으므로 체결 후 수익은 **순수 사후 드리프트**다:")
    print("    net = (CL[체결+보유] - CL[체결]) - 메이커 - 테이커 - 청산slip")
    print("  드리프트가 예측 성분이므로 워크포워드 축소율 %.2f 를 여기에 곱한다."
          % SHRINK)
    print("  (한계가 할인분은 CL[체결] 이 -한계가와 같아 상쇄되므로 따로 안 남는다)\n")
    yrs_s = (dsub["t"].max() - dsub["t"].min()) / (365.25 * 86_400_000)
    per_yr = len(dsub) / yrs_s
    print("  표본 %d건 / %.2f년 -> 연 %.0f건 (2023-01 이후, 21종)\n" % (len(dsub), yrs_s, per_yr))
    print("  %-14s | %-10s | %7s | %9s %9s | %11s %10s %9s"
          % ("규모", "진입", "체결률", "인샘플bp", "OOS bp", "연간$", "최대동시", "연수익률"))
    for q in SIZES:
        si = slip_row(AI, q)
        so = slip_row(BO, q)
        # 시장가
        gross_m = CLs[:, a.hold]
        net_m = gross_m * SHRINK - 2 * a.taker - si - so
        ins_m = gross_m - 2 * a.taker - si - so
        mk = np.isfinite(net_m)
        tt = dsub["t"].to_numpy()
        e = np.concatenate([tt, tt + a.hold * 60_000])
        dl = np.concatenate([np.ones(len(tt)), -np.ones(len(tt))])
        o = np.lexsort((dl, e))
        M = max(int(np.cumsum(dl[o]).max()), 1)
        mv = float(np.nanmean(net_m[mk]))
        print("  $%-13s | %-10s | %6.1f%% | %9.1f %9.1f | %11.0f %10d %8.1f%%"
              % ("%.0fK" % (q / 1e3) if q < 1e6 else "%.1fM" % (q / 1e6), "시장가",
                 100.0, float(np.nanmean(ins_m[mk])), mv,
                 mv * 1e-4 * q * per_yr, M,
                 100 * (mv * 1e-4 * q * per_yr) / (M * q)))
        for lab, lv in (("-100bp", np.full(len(dsub), 100.0)),
                        ("-200bp", np.full(len(dsub), 200.0)),
                        ("매물대", hvs)):
            fill = np.full(len(dsub), -1)
            good = np.isfinite(lv) & (lv > 0)
            for ee in np.flatnonzero(good):
                w = LOs[ee, :a.wait + 1]
                hit = np.flatnonzero(w <= -lv[ee])
                if len(hit):
                    fill[ee] = hit[0]
            okf = fill >= 0
            idx = np.flatnonzero(okf)
            hh = np.minimum(fill[idx] + a.hold, NOBS)
            drift = CLs[idx, hh] - CLs[idx, fill[idx]]
            ins = np.zeros(len(dsub))
            oos = np.zeros(len(dsub))
            ins[idx] = drift - a.maker - a.taker - so[idx]
            oos[idx] = drift * SHRINK - a.maker - a.taker - so[idx]
            mk2 = np.isfinite(oos)
            mv2 = float(np.nanmean(oos[mk2]))
            tf = tt[idx] + fill[idx] * 60_000
            e2 = np.concatenate([tf, tf + a.hold * 60_000])
            d2 = np.concatenate([np.ones(len(tf)), -np.ones(len(tf))])
            o2 = np.lexsort((d2, e2))
            M2 = max(int(np.cumsum(d2[o2]).max()), 1)
            print("  %-14s | %-10s | %6.1f%% | %9.1f %9.1f | %11.0f %10d %8.1f%%"
                  % ("", lab, 100 * okf.mean(),
                     float(np.nanmean(ins[np.isfinite(ins)])), mv2,
                     mv2 * 1e-4 * q * per_yr, M2,
                     100 * (mv2 * 1e-4 * q * per_yr) / (M2 * q)))
        print()
    print("  ** 연수익률 = 연간$ / (최대동시 x 규모). 무레버리지 피크자본 기준. **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
