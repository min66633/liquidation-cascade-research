# -*- coding: utf-8 -*-
"""D-2 — 세 항을 조립해 실제 지도 L(p,t) 를 만들고, 실현 청산 위치를 맞히는지 본다.

    L(p,t) = Σ_{τ<t} ΔOI+(τ) · f_R(p / p(τ)) · S(t-τ)
              ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^   ^^^^^^^
              언제 얼마나 열림   어디서 강제청산   아직 살아있나
              (5분 OI)         (D-1b, HL)      (D-1a, h(x))

핵심 구조 — 방향이 갈린다 (설계 원문: "상승·하락 양방향으로 쌓여 있다")
    롱 코호트: 진입가 **아래** dist 만큼에서 청산 -> 하방 연료
    숏 코호트: 진입가 **위**  dist 만큼에서 청산 -> 상방 연료
  따라서 지도는 현재가 기준 아래/위 두 장이다.

검정 (이 스크립트의 목적)
  지도가 맞다면: **아래에 연료가 많이 쌓인 상태에서 가격이 내려가면, 같은 낙폭에도
  OI 가 더 많이 파괴돼야 한다.** 연료가 없으면 가격만 내리고 OI 는 안 준다.

      ΔOI^-(t→t+k) = a + b1·(그 구간을 실제로 지나간 연료) + b2·|가격 낙폭| + ...

  b1 이 유의한 양수여야 지도가 정보를 갖는다. 가격 낙폭을 통제하는 것이 핵심 —
  통제 안 하면 "많이 내리면 많이 청산된다" 는 자명한 관계만 잡힌다.

*** 룩어헤드 없음: 지도는 t 시점까지의 OI/가격만으로 만든다. ***

실행:
    python analysis/build_map.py
    python analysis/build_map.py --symbols BTCUSDT ETHUSDT --k 12
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
from analysis.event_study_h2 import load                        # noqa: E402
from analysis.response_liq import ols_cluster                   # noqa: E402
from analysis.map_kernel import EDGES, MAXC, MINFRAC            # noqa: E402

# f_R 을 이 격자로 이산화한다 (진입가 대비 청산거리, 양수 크기)
DGRID = np.array([0.015, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50])


def load_kernels():
    """D-1 산출물 두 개를 읽어 (f_R 이산분포, **자발분** h 벡터) 로 만든다.

    *** 이중 차감 방지 (2026-08-04 수정) ***
      D-1a 의 h(x) 는 자발+강제 **합산**이다. 그런데 f_R 이 강제분을 따로 세므로,
      h 를 그대로 S 에 쓰면 강제로 사라질 포지션을 **두 번 지운다**.
      손실 구간에서 h 를 f_R 에 회귀해 강제분을 떼어내고 **자발분만** 남긴다:
          h(x) = h_vol(x) + c · f_R(x)     ->  S 에는 h_vol 만 쓴다
    """
    fp = os.path.join(C.DATA, "analysis", "fr_kernel.parquet")
    hp = os.path.join(C.DATA, "analysis", "map_kernel.parquet")
    if not (os.path.exists(fp) and os.path.exists(hp)):
        raise SystemExit("D-1 산출물이 없다. fr_kernel.py / map_kernel.py 를 먼저 실행")
    fr = pd.read_parquet(fp)
    hk = pd.read_parquet(hp)
    f_all = fr["f_R"].to_numpy()
    h_all = hk["h"].to_numpy()

    # --- 강제분 분리 (손실 구간에서만 f_R 이 0 이 아니다)
    sel = f_all > 0
    if sel.sum() >= 3:
        A = np.column_stack([np.ones(int(sel.sum())), f_all[sel]])
        cf = np.linalg.pinv(A.T @ A) @ (A.T @ h_all[sel])
        h_vol = h_all - cf[1] * f_all
        print("  h = h_vol + c·f_R 분리:  c = %.5f, 절편 = %.5f" % (cf[1], cf[0]))
        print("  강제분이 손실쪽 h 의 %.0f%% 를 설명한다."
              % (100 * cf[1] * f_all[sel].mean() / max(h_all[sel].mean(), 1e-12)))
    else:
        h_vol = h_all
        print("  f_R 구간 부족 — 분리 생략")
    h_vol = np.maximum(h_vol, 1e-5)

    lo = fr[fr["edge_hi"] <= 0].copy()
    mid = np.abs((np.maximum(lo["edge_lo"], -0.60) + lo["edge_hi"]) / 2.0)
    w = lo["f_R"].to_numpy()
    w = w / max(w.sum(), 1e-12)
    return mid.to_numpy(), w, h_vol


def run_symbol(sym, step, dmid, dw, hvec, bands, k):
    """코호트를 굴리며 지도를 만들고, 앞으로 k구간의 실현치와 짝짓는다."""
    try:
        df = load(sym)
    except FileNotFoundError:
        return None
    d = df.iloc[::step].reset_index(drop=True)
    oi = d["sum_open_interest"].to_numpy(dtype=np.float64)
    px = d["close"].to_numpy(dtype=np.float64)
    lo_ = d["low"].to_numpy(dtype=np.float64)
    ot = d["open_time"].to_numpy()
    m = (np.isfinite(oi) & (oi > 0) & np.isfinite(px) & (px > 0)
         & np.isfinite(lo_) & (lo_ > 0))
    oi, px, lo_, ot = oi[m], px[m], lo_[m], ot[m]
    n = len(oi)
    if n < 5000:
        return None
    lp = np.log(px)
    # *** 정정 (2026-08-04) ***
    # 선물 미결제약정 1계약 = 롱 1 + 숏 1. OI 가 dq 늘면 **롱도 dq, 숏도 dq** 생긴다.
    # 집계로는 롱 = 숏 = OI 이고 총 포지션 명목가는 OI 의 2배다.
    # 첫 판은 sum_toptrader_long_short_ratio 로 dq 를 **쪼갰다** — 그 비율은
    # 상위 트레이더의 편중이지 집계 구성이 아니다. 롱을 약 60%로 과소 계상했고,
    # 그 비율이 심리에 따라 변하므로 연료에 심리 잡음이 섞였다.
    nb = len(EDGES) - 1

    cp = [np.empty(MAXC), np.empty(MAXC)]
    cq = [np.zeros(MAXC), np.zeros(MAXC)]
    # *** 핵심 수정 (2026-08-04) ***
    # 연료는 나이와 무관하다 — 단 **아직 청산 안 된 것만** 연료다.
    # 진입 이후 가격이 이미 그 청산가 아래를 찍었다면 그 부분은 **이미 타버렸다.**
    # 코호트별 진입 이후 **최저가(롱) / 최고가(숏)** 를 들고 다니며 그만큼 뺀다.
    # 이것을 안 하면 오래된 코호트일수록 '이미 탄 연료' 를 잔뜩 세게 되고,
    # 그것이 첫 판의 음(-) 계수를 만들었다.
    cm = [np.empty(MAXC), np.empty(MAXC)]
    nc = [1, 1]
    for sd in (0, 1):                       # 롱·숏 각각 OI 전액
        cp[sd][0], cq[sd][0], cm[sd][0] = lp[0], oi[0], lp[0]

    rows = []
    nbd = len(bands)
    for t in range(1, n - k - 1):
        dq = oi[t] - oi[t - 1]
        tot = cq[0][:nc[0]].sum() + cq[1][:nc[1]].sum()
        if tot > 0:
            # --- 지도: 롱 코호트의 청산가는 진입가 **아래**. 하방 연료만 쓴다.
            fuel = np.zeros(nbd)
            fuel_naive = np.zeros(nbd)          # 소진 무시판 (대조용)
            if nc[0] > 0:
                # 청산 로그가격 = 진입 + log(1-d).  현재가 대비 상대거리로 환산
                lq = cp[0][:nc[0]][:, None] + np.log(1.0 - dmid)[None, :]
                rel = 1.0 - np.exp(lq - lp[t - 1])       # 현재가 대비 아래로 몇 %
                amt = cq[0][:nc[0]][:, None] * dw[None, :]
                # **이미 탄 연료 제거**: 청산가가 진입 후 최저가보다 위면 이미 터졌다
                alive = lq < cm[0][:nc[0]][:, None]
                amt_a = amt * alive
                for bi, (b0, b1) in enumerate(bands):
                    sel = (rel > b0) & (rel <= b1)
                    fuel[bi] = float(amt_a[sel].sum())
                    fuel_naive[bi] = float(amt[sel].sum())
            # --- 실현: 앞으로 k구간의 OI 파괴량과 최대 낙폭
            fut_dd = 1.0 - lo_[t:t + k + 1].min() / px[t - 1]
            fut_out = max(oi[t - 1] - oi[t:t + k + 1].min(), 0.0)
            rows.append((ot[t], oi[t - 1], fut_out, fut_dd, *fuel, *fuel_naive))

        # --- 코호트별 진입 이후 극값 갱신 (연료 소진 판정용)
        for sd in (0, 1):
            if nc[sd] > 0:
                if sd == 0:
                    np.minimum(cm[0][:nc[0]], lp[t], out=cm[0][:nc[0]])
                else:
                    np.maximum(cm[1][:nc[1]], lp[t], out=cm[1][:nc[1]])

        # --- 상태 갱신 (map_kernel 과 동일 규칙)
        if dq > 0:
            for sd in (0, 1):               # 롱·숏 **양쪽 모두** dq 만큼 생긴다
                if nc[sd] < MAXC:
                    cp[sd][nc[sd]], cq[sd][nc[sd]] = lp[t], dq
                    cm[sd][nc[sd]] = lp[t]
                    nc[sd] += 1
                else:
                    j = int(np.argmin(cq[sd][:nc[sd]]))
                    s2 = cq[sd][j] + dq
                    cp[sd][j] = (cp[sd][j] * cq[sd][j] + lp[t] * dq) / max(s2, 1e-12)
                    cq[sd][j] = s2
        elif dq < 0 and tot > 0:
            need = min(2.0 * (-dq), tot)    # 계약 1 감소 = 롱 1 + 숏 1 소멸
            wts = []
            for sd in (0, 1):
                if nc[sd] == 0:
                    wts.append(np.zeros(0))
                    continue
                x = (lp[t - 1] - cp[sd][:nc[sd]]) * (1.0 if sd == 0 else -1.0)
                wts.append(hvec[np.digitize(x, EDGES[1:-1])] * cq[sd][:nc[sd]])
            tw = wts[0].sum() + wts[1].sum()
            for sd in (0, 1):
                if nc[sd] == 0:
                    continue
                if tw > 0:
                    take = np.minimum(need * wts[sd] / tw, cq[sd][:nc[sd]])
                    cq[sd][:nc[sd]] -= take
                else:
                    cq[sd][:nc[sd]] *= max(1.0 - need / tot, 0.0)
                if nc[sd] > 200:
                    good = cq[sd][:nc[sd]] > MINFRAC * max(tot, 1e-9)
                    g = int(good.sum())
                    if 0 < g < nc[sd]:
                        cp[sd][:g] = cp[sd][:nc[sd]][good]
                        cq[sd][:g] = cq[sd][:nc[sd]][good]
                        cm[sd][:g] = cm[sd][:nc[sd]][good]
                        nc[sd] = g
    if not rows:
        return None
    A = np.array(rows, dtype=np.float64)
    out = pd.DataFrame({"t": A[:, 0].astype(np.int64), "symbol": sym,
                        "oi": A[:, 1], "out": A[:, 2], "dd": A[:, 3]})
    for bi in range(nbd):
        out["f%d" % bi] = A[:, 4 + bi]
        out["g%d" % bi] = A[:, 4 + nbd + bi]      # 소진 무시판(대조용)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="D-2 assemble map and validate")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--step", type=int, default=12, help="5분봉 몇 개마다 (12=60분)")
    ap.add_argument("--k", type=int, default=12, help="앞으로 몇 구간을 볼지")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.FULL_HISTORY_SYMBOLS

    print("=" * 80)
    print("D-2 — 지도 L(p,t) 조립과 검정")
    print("=" * 80)
    dmid, dw, hvec = load_kernels()
    print("f_R 이산화: 거리 %s" % np.round(dmid, 3))
    print("           비중 %s" % np.round(dw, 3))
    bands = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]
    print("연료 밴드(현재가 아래 %%): %s" % bands)
    print("전방 창 %d구간 (=%d분)\n" % (a.k, a.k * 5 * a.step))

    fr = []
    for s in syms:
        r = run_symbol(s, a.step, dmid, dw, hvec, bands, a.k)
        if r is not None:
            fr.append(r)
            print("  %-10s %7d 구간" % (s, len(r)))
    if not fr:
        print("표본 없음")
        return 1
    d = pd.concat(fr, ignore_index=True).sort_values("t").reset_index(drop=True)
    print("\n**사용 데이터 기간: %s ~ %s / %d종 / %d구간**"
          % (str(pd.Timestamp(d.t.min(), unit="ms"))[:10],
             str(pd.Timestamp(d.t.max(), unit="ms"))[:10], d.symbol.nunique(), len(d)))

    oi = d["oi"].to_numpy()
    y = d["out"].to_numpy() / oi                       # OI 파괴 비율
    dd = d["dd"].to_numpy()                            # 실현 낙폭
    F = np.column_stack([d["f%d" % i].to_numpy() / oi for i in range(len(bands))])
    day = d["t"].to_numpy() // 86_400_000

    print("\n실현 낙폭 중앙 %.2f%% | OI 파괴 중앙 %.2f%%"
          % (100 * np.median(dd), 100 * np.median(y)))
    print("연료(OI 대비) 밴드별 중앙: %s"
          % np.round([np.median(F[:, i]) for i in range(len(bands))], 4))

    print("\n" + "-" * 80)
    print("검정 — 낙폭을 통제하고도 연료가 OI 파괴를 설명하는가")
    print("-" * 80)
    print("  통제 없이 하면 '많이 내리면 많이 청산된다' 는 자명한 관계만 잡힌다.\n")
    ok = np.isfinite(y) & np.isfinite(dd) & np.isfinite(F).all(axis=1) & (oi > 0)
    G = np.column_stack([d["g%d" % i].to_numpy() / oi for i in range(len(bands))])
    specs = [
        ("낙폭만", np.column_stack([np.ones(ok.sum()), dd[ok]])),
        ("낙폭 + **연료(소진 반영)**",
         np.column_stack([np.ones(ok.sum()), dd[ok], F[ok].sum(axis=1)])),
        ("낙폭 + 연료(소진 무시·첫판)",
         np.column_stack([np.ones(ok.sum()), dd[ok], G[ok].sum(axis=1)])),
        ("낙폭 + 연료 밴드별(소진 반영)",
         np.column_stack([np.ones(ok.sum()), dd[ok], F[ok]])),
    ]
    for lab, X in specs:
        b, se, _ = ols_cluster(X, y[ok], day[ok])
        r2 = 1.0 - np.var(y[ok] - X @ b) / np.var(y[ok])
        print("  [%s]  R^2 = %.4f" % (lab, r2))
        print("    낙폭 계수 %.4f (t=%.1f)" % (b[1], b[1] / se[1]))
        for j in range(2, len(b)):
            nm = ("연료 전체" if len(b) == 3
                  else "연료 %.0f~%.0f%%" % (100 * bands[j - 2][0], 100 * bands[j - 2][1]))
            print("    %-14s %8.4f (t=%.1f)" % (nm, b[j], b[j] / se[j]))
        print()
    print("  연료 계수가 유의한 양수면 **지도가 낙폭 너머의 정보를 갖는다**는 뜻이다.")

    print("=" * 80)
    print("★ 본 검정 — 지도가 **가격이 어디까지 밀리는가(X)** 를 맞히는가")
    print("=" * 80)
    print("  앞의 검정은 **OI 파괴량**을 예측했다. 그건 중간 산물이지 설계의 목표가")
    print("  아니다. 설계가 예측하려는 것은 **X = 밀림 거리** 다.")
    print("  그리고 '12시간 안에' 라는 창도 임의였다 — 연료는 몇 주 안 건드려질 수")
    print("  있고 그건 지도의 실패가 아니다. **언제 터지는지는 상관없다.**")
    print("  그래서 (1) X 를 직접 예측하고 (2) 하락이 실제로 시작된 구간만 본다.\n")

    # 변동성 통제 — X 는 sigma 에 정비례한다(b1=1.017, ARCHITECTURE L3)
    sg = pd.Series(np.log(d["oi"].to_numpy())).diff().rolling(24).std().to_numpy()
    lsig = np.log(np.maximum(pd.Series(dd).rolling(168, min_periods=48)
                             .median().shift(1).to_numpy(), 1e-6))
    base = np.isfinite(dd) & (dd > 0) & np.isfinite(lsig) & np.isfinite(F).all(axis=1)

    for lab, sel in (("전체 구간", base),
                     ("**하락 상위 20%**", base & (dd > np.nanquantile(dd, 0.80))),
                     ("**하락 상위 5%** (캐스케이드)", base & (dd > np.nanquantile(dd, 0.95)))):
        nsel = int(sel.sum())
        if nsel < 500:
            continue
        y2 = np.log(dd[sel])
        X0 = np.column_stack([np.ones(nsel), lsig[sel]])
        X1 = np.column_stack([np.ones(nsel), lsig[sel], F[sel]])
        b0, s0, _ = ols_cluster(X0, y2, day[sel])
        b1, s1, _ = ols_cluster(X1, y2, day[sel])
        r20 = 1.0 - np.var(y2 - X0 @ b0) / np.var(y2)
        r21 = 1.0 - np.var(y2 - X1 @ b1) / np.var(y2)
        print("  [%s]  n=%d | R^2 기저 %.4f -> 연료추가 %.4f (증분 **%+.4f**)"
              % (lab, nsel, r20, r21, r21 - r20))
        for j, (b_0, b_1) in enumerate(bands):
            print("      연료 %3.0f~%3.0f%%  %+8.4f (t=%5.1f)"
                  % (100 * b_0, 100 * b_1, b1[2 + j], b1[2 + j] / s1[2 + j]))
        print()
    print("  증분 R^2 가 양수이고 연료 계수가 양수면 **지도가 X 를 설명한다**.")
    print("  캐스케이드 구간(상위 5%)에서 커져야 설계가 맞는 것이다.\n")

    print("-" * 80)
    print("보조 검정 — 연료가 많을 때 같은 낙폭에서 더 파괴되는가 (오분위)")
    print("-" * 80)
    tot_f = F.sum(axis=1)
    qd = pd.qcut(pd.Series(dd[ok]), 5, labels=False, duplicates="drop").to_numpy()
    qf = pd.qcut(pd.Series(tot_f[ok]), 5, labels=False, duplicates="drop").to_numpy()
    yy = y[ok]
    print("  행=낙폭 오분위, 열=연료 오분위. 값=OI 파괴 중앙(%)")
    print("  %-8s %8s %8s %8s %8s %8s" % ("낙폭\\연료", "Q0", "Q1", "Q2", "Q3", "Q4"))
    for i in range(5):
        cells = []
        for j in range(5):
            s_ = (qd == i) & (qf == j)
            cells.append("%8.2f" % (100 * np.median(yy[s_])) if s_.sum() > 30 else "       -")
        print("  %-8d %s" % (i, " ".join(cells)))
    print("\n  각 행에서 **왼쪽→오른쪽으로 증가**해야 지도가 작동하는 것이다.")
    out = os.path.join(C.DATA, "analysis", "map_panel.parquet")
    d.to_parquet(out, index=False)
    print("\n  저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
