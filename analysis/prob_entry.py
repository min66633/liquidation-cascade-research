# -*- coding: utf-8 -*-
"""확률적 바닥 추정 -> 조건부 지정매수 (+ 슈팅 레그, 상태의존 청산).

앞선 두 번의 오류를 고친다
  limit_fill.py 는 p_lim = p0 * (1 - kappa*sigma) 로 **무조건 아래**에 걸었다.
  그건 지정가지 확률모델이 아니다. 지금이 이미 바닥일 확률이 높으면 오프셋은
  0 에 가까워야 하고, 더 밀릴 확률이 높으면 깊어야 한다. kappa*sigma 는 그
  구분을 못 하므로 '더 밀릴 사건' 만 골라 잡았다(역선택 -159.45bp).

  pnl_source.py 는 보유를 15분에 고정했다. 바닥은 몇 초 만에 오기도 하고 몇 분
  걸리기도 한다. 회복 속도도 마찬가지다. 고정 보유는 그 분산을 전부 손실로 바꾼다.

이 파일이 하는 것 — 설계 원문 그대로
  1) 방아쇠 시점 t0 의 정보만으로 **앞으로 얼마나 더 밀릴지의 분포** F(X | F_t0) 를 만든다
       log X = b' f + eps,   q_alpha = exp(b'f + Q_alpha(eps))
     f 에 들어가는 것 (전부 t0 에 알려진 값):
       sigma, |z|, dOI(방아쇠 바), 직전 3/12 바 누적수익, OI 재고,
       테이커 롱숏비, 상위트레이더 포지션비, 계좌 롱숏비, 1분 실현변동성, 거래량비
     -> ① 지도(OI·포지션비·펀딩 방향), ③ 활동량(거래량·테이커) 의 과거 구간 대용치다.
  2) **워크포워드**로만 예측한다. 200건 이상 쌓인 뒤부터, 50건마다 재적합,
     훈련 끝과 시험 사건 사이에 1일 퍼지 간격. 미래 데이터를 절대 안 본다.
  3) p_lim = p0 * (1 - q_alpha) 로 지정매수. alpha 를 훑는다.
       alpha 작다 -> 얕게 (지금이 바닥이라고 보는 것. 시장가에 가깝다)
       alpha 크다 -> 깊게
     **alpha 는 고정 오프셋이 아니라 분포의 분위다.** 상태에 따라 깊이가 달라진다.
  4) 슈팅 레그(옵션): t0 에 시장가 **매도**, p_lim 에서 2배 주문으로 커버+반전.
     안 걸리면 시간정지에 시장가 커버. 밀림과 되돌림을 **둘 다** 먹는 구조다.
  5) 청산은 상태의존: 예측 밀림폭의 gamma 배를 되돌리면 익절, 아니면 시간정지.
     고정 보유도 같이 낸다 — 어느 쪽이 나은지는 표가 말하게 한다.

읽는 법
  ** 시도당(미체결=0) 이 유일하게 비교 가능한 값이다. 체결당은 생존자 편향. **
  ** 승률과 손익비를 같이 본다. 평균만 보면 소수 대박에 속는다. **

실행:
    python analysis/prob_entry.py
    python analysis/prob_entry.py --short          # 슈팅 레그 포함
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
HMAX = 60                      # 바닥을 찾는 최대 지평(분)
EMAX = 60                      # 체결 후 보유 최대(분)
FEAT = ["lsig", "absz", "doi", "r3", "r12", "loi", "tls", "ttp", "cls", "rv30", "vol"]


def build(symbols, k, doi_thr, gap) -> tuple:
    """이벤트별 특징 + 이후 1분봉 창을 통째로 들고 온다 (시뮬레이션을 즉시 반복하려고)."""
    rows, win = [], []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, k, doi_thr, gap)
        if len(ev) == 0:
            continue
        p1 = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p1):
            continue
        m = pd.read_parquet(p1, columns=["open_time", "open", "high", "low",
                                         "close", "quote_volume"])
        m = m.sort_values("open_time").reset_index(drop=True)
        ot1 = m["open_time"].to_numpy()
        O = m["open"].to_numpy(dtype=np.float64)
        H = m["high"].to_numpy(dtype=np.float64)
        L = m["low"].to_numpy(dtype=np.float64)
        Cl = m["close"].to_numpy(dtype=np.float64)
        QV = m["quote_volume"].to_numpy(dtype=np.float64)
        n1 = len(ot1)
        # 1분 실현변동성(과거 30분)과 거래량비(과거 하루 중앙 대비) — t0 에 알려진 값
        lr = np.concatenate([[np.nan], np.diff(np.log(np.maximum(Cl, 1e-12)))])
        rv30 = pd.Series(lr).rolling(30, min_periods=15).std().shift(1).to_numpy()
        vmed = pd.Series(QV).rolling(1440, min_periods=200).mean().shift(1).to_numpy()

        ot5 = df["open_time"].to_numpy()
        sig5 = df["sigma"].to_numpy()
        zz = df["z"].to_numpy()
        doi = df["doi"].to_numpy()
        ret5 = df["ret"].to_numpy()
        oi = df["sum_open_interest"].to_numpy(dtype=np.float64)
        # 정규화용 기준선이므로 평균으로 충분하다. rolling median 은 창 8640 x 63만행이라
        # 전량 실행에 수 분이 걸린다 (실측). 평균은 O(n).
        oimed = pd.Series(oi).rolling(8640, min_periods=576).mean().shift(1).to_numpy()
        tls = df["sum_taker_long_short_vol_ratio"].to_numpy(dtype=np.float64)
        ttp = df["sum_toptrader_long_short_ratio"].to_numpy(dtype=np.float64)
        cls_ = df["count_long_short_ratio"].to_numpy(dtype=np.float64)
        # 직전 3/12 바 누적수익 — 캐스케이드가 이미 얼마나 진행됐나 (현재 바 포함)
        cr = pd.Series(ret5)
        r3 = cr.rolling(3, min_periods=3).sum().to_numpy()
        r12 = cr.rolling(12, min_periods=12).sum().to_numpy()

        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5):
                continue
            t0 = int(ot5[i + 1])
            j = int(np.searchsorted(ot1, t0))
            span = HMAX + EMAX
            if j < 1 or j >= n1 or ot1[j] != t0 or j + span >= n1:
                continue
            if ot1[j + span] - ot1[j] != span * 60_000:
                continue
            sl = slice(j, j + span + 1)
            if not (np.isfinite(O[sl]).all() and np.isfinite(H[sl]).all()
                    and np.isfinite(L[sl]).all() and np.isfinite(Cl[sl]).all()):
                continue
            f = {"lsig": np.log(max(sig5[i], 1e-9)), "absz": abs(zz[i]), "doi": doi[i],
                 # 방향으로 부호를 통일한다 (숏청산이면 상승이 '진행' 이다)
                 "r3": r3[i] * sd, "r12": r12[i] * sd,
                 "loi": np.log(max(oi[i], 1e-9) / max(oimed[i], 1e-9))
                        if np.isfinite(oimed[i]) and oimed[i] > 0 else np.nan,
                 "tls": tls[i], "ttp": ttp[i], "cls": cls_[i],
                 "rv30": rv30[j], "vol": QV[j - 1] / vmed[j] if (np.isfinite(vmed[j])
                                                                and vmed[j] > 0) else np.nan}
            if not all(np.isfinite(v) for v in f.values()):
                continue
            p0 = O[j]
            if not (np.isfinite(p0) and p0 > 0):
                continue
            # 목표: t0 이후 HMAX 분 안의 **추가 밀림** (진행 방향으로) bp
            seg = (L[j:j + HMAX + 1] if sd == 1 else H[j:j + HMAX + 1])
            X = ((p0 - seg.min()) if sd == 1 else (seg.max() - p0)) / p0 * 1e4
            rows.append({"symbol": s, "t": t0, "side": sd, "p0": p0,
                         "day": t0 // 86_400_000,
                         "year": pd.Timestamp(t0, unit="ms").year,
                         "X": max(X, 0.01), "sig5": sig5[i], "z": zz[i], **f})
            win.append(np.stack([O[j:j + span + 1], H[j:j + span + 1],
                                 L[j:j + span + 1], Cl[j:j + span + 1]]))
    if not rows:
        return None, None
    d = pd.DataFrame(rows)
    o = np.argsort(d["t"].to_numpy(), kind="mergesort")
    return d.iloc[o].reset_index(drop=True), np.array(win)[o]


def walk_forward(d: pd.DataFrame, alphas, min_train=200, refit=50, purge_days=1,
                 col="X", feats=None):
    """OOS 조건부 분위. 훈련 끝과 시험 사이 purge_days 를 비운다.

    같은 날 여러 심볼이 동시에 터지므로, 퍼지 없이 자르면 같은 사건의 다른 다리를
    훈련에 넣고 시험하게 된다 (누출).

    col 로 목표를 바꿀 수 있다 — "X" 는 밀림 폭(bp), 시간 모형을 쓰려면 분 단위 열.
    """
    n = len(d)
    y = np.log(np.maximum(d[col].to_numpy(dtype=np.float64), 1e-9))
    F = d[list(feats) if feats else FEAT].to_numpy(dtype=np.float64)
    day = d["day"].to_numpy()
    Q = {a: np.full(n, np.nan) for a in alphas}
    oos = np.zeros(n, dtype=bool)
    # 특징에 결측이 있으면(오더북이 없는 날 등) 훈련·예측 양쪽에서 뺀다.
    # 빼지 않으면 OLS 가 통째로 NaN 이 되어 조용히 전부 죽는다.
    fin = np.isfinite(F).all(axis=1) & np.isfinite(y)
    beta = None
    resq = None
    last_fit = -10**9
    for i in range(min_train, n):
        if i - last_fit >= refit or beta is None:
            tr = (np.arange(n) < i) & fin
            tr &= (day < day[i] - purge_days)
            if tr.sum() < min_train:
                continue
            X = np.column_stack([np.ones(int(tr.sum())), F[tr]])
            b = np.linalg.pinv(X.T @ X) @ (X.T @ y[tr])
            e = y[tr] - X @ b
            beta, resq = b, {a: float(np.quantile(e, a)) for a in alphas}
            last_fit = i
        if beta is None or not fin[i]:
            continue
        m = float(np.concatenate([[1.0], F[i]]) @ beta)
        for a in alphas:
            Q[a][i] = np.exp(m + resq[a])
        oos[i] = True
    return Q, oos


def simulate(d, win, qlim, delta_bp, exit_rule, exit_par,
             qexit, fee_m, fee_t, use_short, taker_in=False) -> pd.DataFrame:
    # taker_in=True 는 시장가 진입(기준선). qlim=0 으로 넣으면 코드 경로는 같지만
    # 수수료는 메이커가 아니라 테이커여야 한다. 안 그러면 기준선이 3bp 유리해진다.
    """조건부 지정매수(+옵션 숏레그) 시뮬레이션. 창은 [j, j+HMAX+EMAX]."""
    out = []
    dl = delta_bp * 1e-4
    sd_ = d["side"].to_numpy()
    p0_ = d["p0"].to_numpy()
    for i in range(len(d)):
        if not np.isfinite(qlim[i]):
            continue
        sd, p0 = int(sd_[i]), float(p0_[i])
        O, H, L, Cl = win[i]
        base = {"symbol": d["symbol"].iat[i], "t": d["t"].iat[i], "day": d["day"].iat[i],
                "year": d["year"].iat[i], "side": sd, "q": qlim[i],
                "filled": False, "ret": 0.0, "wait": np.nan, "hold": np.nan,
                "leg_s": 0.0, "leg_l": 0.0}
        p_lim = p0 * (1.0 - sd * qlim[i] * 1e-4)
        # ---- 체결 탐색 (t0 부터 HMAX 분) ----
        if sd == 1:
            hit = np.flatnonzero(L[:HMAX + 1] <= p_lim * (1.0 - dl))
        else:
            hit = np.flatnonzero(H[:HMAX + 1] >= p_lim * (1.0 + dl))
        if len(hit) == 0:
            if use_short:
                # 숏레그만 열려 있다 -> HMAX 에서 시장가 커버 (손실일 수 있다)
                pc = Cl[HMAX]
                base["leg_s"] = (p0 / pc - 1.0) * sd * 1e4 - 2.0 * fee_t
                base["ret"] = base["leg_s"]
            out.append(base)
            continue
        fj = int(hit[0])
        p_in = (min(O[fj], p_lim) if sd == 1 else max(O[fj], p_lim)) if fj > 0 else p_lim
        # ---- 청산 (상태의존 또는 고정) ----
        # *** 익절 스캔은 체결된 봉의 **다음 봉**부터다 ***
        # 같은 1분봉 안의 저가와 고가는 순서를 알 수 없다. fj 를 포함시키면
        # '저가에 사서 같은 봉 고가에 판다' 가 되어 승률 100% / 낙폭 0 이 나온다
        # (실측으로 확인함). 봉내 경로를 모르는 이상 이 봉은 못 쓴다.
        lo, hi = fj + 1, min(fj + EMAX, len(Cl) - 1)
        if exit_rule == "fixed":
            ej = min(fj + int(exit_par), len(Cl) - 1)
            p_out, fee_out = Cl[ej], fee_t
        elif lo > hi:
            ej, p_out, fee_out = hi, Cl[hi], fee_t
        else:
            # 목표: 예측 밀림폭 qexit 의 gamma 배를 되돌리면 익절
            tgt = p_in * (1.0 + sd * exit_par * qexit[i] * 1e-4)
            seg = (H[lo:hi + 1] if sd == 1 else L[lo:hi + 1])
            g = np.flatnonzero(seg >= tgt) if sd == 1 else np.flatnonzero(seg <= tgt)
            if len(g):
                ej, p_out = lo + int(g[0]), tgt          # 지정가 익절 -> 메이커
                fee_out = fee_m
            else:
                ej, p_out = hi, Cl[hi]                   # 시간정지 -> 테이커
                fee_out = fee_t
        fee_in = fee_t if taker_in else fee_m
        leg_l = (p_out / p_in - 1.0) * sd * 1e4 - (fee_in + fee_out)
        leg_s = 0.0
        if use_short:
            # t0 시장가 매도 -> p_in 에서 커버 (같은 지정가 주문이 2배로 처리)
            leg_s = (p0 / p_in - 1.0) * sd * 1e4 - (fee_t + fee_m)
        base.update({"filled": True, "ret": leg_l + leg_s, "leg_l": leg_l,
                     "leg_s": leg_s, "wait": fj, "hold": ej - fj, "p_in": p_in})
        out.append(base)
    return pd.DataFrame(out)


def stat(x: pd.DataFrame) -> dict:
    if len(x) == 0:
        return {}
    r = x["ret"].to_numpy()
    m, _, t, _ = cmean(r, x["day"].to_numpy())
    yrs = (x["t"].max() - x["t"].min()) / (365.25 * 86_400_000)
    # *** 승률·손익비는 **실제 거래**에만 매긴다 ***
    # 미체결(ret=0)을 '패' 로 세면 승률이 눌리고, 0 을 손실평균에 넣으면
    # 손익비가 부풀려진다. 시도당 평균(bp) 에는 물론 0 이 그대로 들어간다.
    rt = r[r != 0.0]
    w, l = rt[rt > 0], rt[rt < 0]
    return {"n": len(x), "fill": float(x["filled"].mean()), "bp": m, "t": t,
            "med": float(np.median(rt)) if len(rt) else np.nan,
            "win": float((rt > 0).mean()) if len(rt) else np.nan,
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "hold": float(x["hold"].median()) if x["filled"].any() else np.nan,
            "wait": float(x["wait"].median()) if x["filled"].any() else np.nan,
            "maxdd": float((np.cumsum(r) - np.maximum.accumulate(np.cumsum(r))).min())}


def line(lab, s):
    if not s:
        return
    print("  %-22s | %5d %6.3f | %8.1f %5.1f %8.1f | %6.1f%% %6.2f | %6.2f %7.0f | %5.0f %5.0f"
          % (lab, s["n"], s["fill"], s["bp"], s["t"], s["med"], 100 * s["win"],
             s["pl"], s["sharpe"], s["maxdd"], s["wait"], s["hold"]))


def head():
    print("  %-22s | %5s %6s | %8s %5s %8s | %7s %6s | %6s %7s | %5s %5s"
          % ("설정", "n", "체결률", "시도당bp", "t", "중앙bp", "승률", "손익비",
             "샤프", "최대낙폭", "대기", "보유"))


def main() -> int:
    ap = argparse.ArgumentParser(description="probabilistic bottom -> conditional limit entry")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--delta", type=float, default=2.0, help="침투 마진 bp (큐 보수성)")
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.05, 0.15, 0.30, 0.50, 0.70, 0.90])
    ap.add_argument("--short", action="store_true", help="슈팅 레그 포함")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 118)
    print("확률적 바닥 추정 -> 조건부 지정매수. 오프셋은 **상태의 함수**다 (고정 kappa 아님)")
    print("=" * 118)
    d, win = build(syms, a.k, a.doi, a.gap)
    if d is None or len(d) < 300:
        print("이벤트 부족")
        return 1
    print("**사용 데이터 기간: %s ~ %s / %d종 / %d건 (5분봉+1분봉 벌크)**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d)))
    print("추가 밀림 X (t0 후 %d분): 중앙 %.0f bp | p25 %.0f | p75 %.0f | p90 %.0f"
          % (HMAX, d.X.median(), d.X.quantile(.25), d.X.quantile(.75), d.X.quantile(.90)))

    alphas = sorted(set(a.alphas + [0.5]))
    Q, oos = walk_forward(d, alphas)
    print("워크포워드: 200건 이후 50건마다 재적합, 1일 퍼지 | OOS %d건 (%s ~ %s)"
          % (oos.sum(), str(pd.Timestamp(int(d.t[oos].min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t[oos].max()), unit="ms"))[:10]))

    # --- 모형이 맞기는 하는가 (OOS 캘리브레이션) ---
    print("\n" + "-" * 118)
    print("0. 예측 분포가 맞는가 — OOS 위반율이 alpha 와 같아야 한다")
    print("-" * 118)
    Xo = d["X"].to_numpy()
    print("  %-8s %12s %12s %10s" % ("alpha", "예측q 중앙bp", "실제위반율", "편차"))
    for al in alphas:
        q = Q[al][oos]
        v = float((Xo[oos] < q).mean())
        print("  %-8.2f %12.0f %12.3f %10.3f" % (al, np.median(q), v, v - al))
    print("  위반율 = X 가 예측 분위에 못 미친 비율 = **지정가가 안 걸렸어야 할 비율**.")
    print("  alpha 와 같으면 교정된 것이고, 그때 체결률이 1-alpha 로 나와야 한다.")
    print("  편차가 크면 분포 자체가 틀린 것이므로 아래 표의 alpha 는 이름값을 못 한다.")

    n_oos = int(oos.sum())
    dd = d[oos].reset_index(drop=True)
    ww = win[oos]
    Qo = {al: Q[al][oos] for al in alphas}

    print("\n" + "-" * 118)
    print("1. 조건부 지정매수 — alpha 를 훑는다 (고정 보유 15분, 기준선 비교용)")
    print("-" * 118)
    print("  alpha 작다 = 얕게(지금이 바닥이라고 봄) | 크다 = 깊게. **깊이가 사건마다 다르다.**\n")
    head()
    # 기준선: 시장가 (q=0)
    z0 = np.zeros(n_oos)
    line("시장가 (기준선)",
         stat(simulate(dd, ww, z0, 0.0, "fixed", 15, z0,
                       a.fee_maker, a.fee_taker, False, taker_in=True)))
    for al in alphas:
        line("지정 a=%.2f" % al,
             stat(simulate(dd, ww, Qo[al], a.delta, "fixed", 15,
                           Qo[0.5], a.fee_maker, a.fee_taker, False)))

    print("\n" + "-" * 118)
    print("2. 진입 alpha x 청산 규칙 격자 — 보유 15분 고정을 버린다 (최대 %d분)" % EMAX)
    print("-" * 118)
    print("  ** 격자에서 최고 칸을 고르는 것 자체가 선택이다. 아래 3에서 안정성을 본다. **\n")
    EXITS = (("고정15", "fixed", 15), ("고정30", "fixed", 30), ("고정60", "fixed", 60),
             ("목표1.0q", "tgt", 1.0), ("목표1.5q", "tgt", 1.5), ("목표2.0q", "tgt", 2.0))
    print("  %-10s | %s" % ("alpha", " | ".join("%-14s" % e[0] for e in EXITS)))
    print("  %-10s | %s" % ("", " | ".join("%6s %7s" % ("bp", "샤프") for _ in EXITS)))
    grid = {}
    for al in alphas:
        cells = []
        for lab, rule, par in EXITS:
            s = stat(simulate(dd, ww, Qo[al], a.delta, rule, par,
                              Qo[0.5], a.fee_maker, a.fee_taker, False))
            grid[(al, lab)] = s
            cells.append("%6.1f %7.2f" % (s["bp"], s["sharpe"]))
        print("  %-10.2f | %s" % (al, " | ".join(cells)))
    best = max(grid, key=lambda kk: grid[kk]["sharpe"])
    bestbp = max(grid, key=lambda kk: grid[kk]["bp"])
    print("\n  최고 샤프 칸: alpha=%.2f / %s   |   최고 시도당bp 칸: alpha=%.2f / %s"
          % (best + bestbp))
    head()
    line("기준선 시장가15분",
         stat(simulate(dd, ww, z0, 0.0, "fixed", 15, z0, a.fee_maker, a.fee_taker, False,
                       taker_in=True)))
    line("최고샤프 a=%.2f %s" % best, grid[best])
    line("최고bp a=%.2f %s" % bestbp, grid[bestbp])
    print("  ** alpha 가 격자 끝값이면 봉우리가 아니라 경계일 수 있다. 위 표의 끝열을 보라. **")

    print("\n" + "-" * 118)
    print("3. 안정성 — 최고 칸이 OOS 기간을 반으로 갈라도 살아남는가")
    print("-" * 118)
    print("  격자에서 고른 칸은 그 기간에 맞춰진 것일 수 있다. 전/후반이 둘 다 서야 한다.\n")
    x0 = simulate(dd, ww, z0, 0.0, "fixed", 15, z0, a.fee_maker, a.fee_taker, False,
                  taker_in=True)
    h = len(x0) // 2
    cand = []
    seen = set()
    for kk in (best, bestbp):
        if kk not in seen:
            seen.add(kk)
            cand.append(kk)
    # 최고 alpha 의 청산 규칙 3종도 같이 본다 (한 칸만 보면 우연을 못 거른다)
    for lab_ in ("고정15", "고정30", "고정60"):
        kk = (max(alphas), lab_)
        if kk in grid and kk not in seen:
            seen.add(kk)
            cand.append(kk)
    head()
    line("기준선 전체", stat(x0))
    line("  기준선 전반부", stat(x0.iloc[:h]))
    line("  기준선 후반부", stat(x0.iloc[h:]))
    keep = {}
    for al_b, lab_b in cand:
        rule_b, par_b = next((r, p) for l_, r, p in EXITS if l_ == lab_b)
        xb = simulate(dd, ww, Qo[al_b], a.delta, rule_b, par_b, Qo[0.5],
                      a.fee_maker, a.fee_taker, False)
        keep[(al_b, lab_b)] = xb
        line("a=%.2f %s 전체" % (al_b, lab_b), stat(xb))
        line("  전반부", stat(xb.iloc[:h]))
        line("  후반부", stat(xb.iloc[h:]))
    print("  ** 전/후반이 둘 다 서야 한다. 한쪽만 서면 기간에 맞춰진 것이다. **")
    print("\n  연도별 (기준선 대비 차이 bp):")
    yrs = sorted(set(x0["year"]))
    print("  %-18s %s" % ("설정", " ".join("%9d" % y for y in yrs)))
    print("  %-18s %s" % ("기준선(절대)", " ".join(
        "%9.1f" % x0[x0.year == y].ret.mean() for y in yrs)))
    for kk, xb in keep.items():
        print("  %-18s %s" % ("a=%.2f %s" % kk, " ".join(
            "%9.1f" % (xb[xb.year == y].ret.mean() - x0[x0.year == y].ret.mean())
            for y in yrs)))

    print("\n" + "-" * 118)
    print("4. 슈팅 + 되돌림 — t0 에 시장가 매도, 지정가에서 2배로 커버+반전")
    print("-" * 118)
    print("  미체결이면 %d분에 시장가 커버. 밀림과 되돌림을 **둘 다** 먹는 구조다.\n" % HMAX)
    head()
    for al in alphas:
        for lab, rule, par in (("고정15", "fixed", 15), ("목표1.0q", "tgt", 1.0)):
            s = stat(simulate(dd, ww, Qo[al], a.delta, rule, par,
                              Qo[0.5], a.fee_maker, a.fee_taker, True))
            line("숏+롱 a=%.2f %s" % (al, lab), s)
    print("\n  레그 분해 (alpha=0.50, 목표1.0q):")
    x = simulate(dd, ww, Qo[0.5], a.delta, "tgt", 1.0, Qo[0.5],
                 a.fee_maker, a.fee_taker, True)
    f = x[x["filled"]]
    if len(f):
        print("    체결 %d건: 숏레그 평균 %+.1f bp | 롱레그 평균 %+.1f bp | 합 %+.1f"
              % (len(f), f.leg_s.mean(), f.leg_l.mean(), f.ret.mean()))
    nf = x[~x["filled"]]
    if len(nf):
        print("    미체결 %d건: 숏만 남아 시간정지 커버 평균 %+.1f bp"
              % (len(nf), nf.ret.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
