# -*- coding: utf-8 -*-
"""두 레그를 **독립**시킨다 — 밀림(슈팅)과 되돌림에 각각 익절·손절·시간정지.

prob_entry.py 의 설계 오류
  밀림 레그에 **청산 규칙이 없었다**. 되돌림 지정가가 체결되는 지점에서 커버하거나
  안 걸리면 60분 뒤 시장가 — 즉 밀림의 청산이 되돌림 주문에 매달려 있었다.
  그런데 밀림은 몇 분이면 끝난다 (pnl_source.py: 최저점이 0분 44.2%, 1~2분 16.8%.
  **61%가 2분 안에 끝난다**). 그걸 60분 들고 있으니 되돌림을 통째로 맞았다.
  결과: 미체결 295건 평균 -155.8bp. 이건 밀림 예측이 틀려서가 아니라 규칙이 없어서다.
  (체결된 304건의 밀림 레그는 +93.9bp 를 벌었다.)

새 규칙 — 두 레그는 서로를 기다리지 않는다
  방아쇠 (기존과 동일)
    5분봉 |z|>=K 그리고 같은 봉 dOI<=DOI. 방향 sd 는 움직임의 **반대**.
    확인은 OI 스냅샷 때문에 다음 5분봉 시각 t0 에 도착한다. p0 = t0 의 1분봉 시가.

  [1] 밀림 레그 — 움직임 **방향으로** 탄다
      진입  t0 시장가 (테이커)
      익절  p0 * (1 -+ eta * q50)      예측 밀림의 eta 배 도달 -> 지정가 (메이커)
      손절  p0 * (1 +- s_sl * q50)     반대로 가면 자른다 -> 시장가 (테이커)
      정지  tmax 분 (짧게. 2~30분 훑는다)
      셋 중 **먼저 닿는 것**.

  [2] 되돌림 레그 — 움직임 **반대로** 탄다
      진입  p_lim = p0 * (1 -+ q_alpha) 지정가, 유효 W 분 (메이커)
      익절  p_in * (1 +- gamma * q50) 지정가 (메이커)
      손절  p_in * (1 -+ r_sl * q50) 시장가 (테이커)
      정지  emax 분
      미체결이면 그 사건은 되돌림 거래 없음 (0).

  q_alpha, q50 은 prob_entry 의 워크포워드 모형에서 온다. 미래를 안 본다.

*** 봉내 순서 문제 — 반드시 보수적으로 ***
  1분봉 하나 안에서 익절선과 손절선에 **둘 다** 닿을 수 있다. 어느 쪽이 먼저였는지
  1분봉으로는 알 수 없다. 그래서 **손절이 먼저 닿았다고 본다**. 이걸 반대로 하면
  승률 100% 짜리 가짜 전략이 나온다 (prob_entry 초판에서 실제로 그랬다).

실행:
    python analysis/two_leg.py
    python analysis/two_leg.py --symbols BTCUSDT DOGEUSDT
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
from analysis.prob_entry import build, walk_forward, HMAX, EMAX      # noqa: E402
from analysis.response_liq import cmean                              # noqa: E402


def _scan(O, H, L, Cl, sd_pos, p_ent, tp, sl, t0, tmax, fee_tp, fee_sl, fee_to,
          tp_first=False):
    """t0..t0+tmax 봉을 훑어 (청산가, 수수료, 청산봉, 사유) 를 낸다.

    sd_pos = +1 이면 롱(가격이 올라야 이익), -1 이면 숏.
    tp 는 이익 쪽, sl 은 손실 쪽 가격. 같은 봉에 둘 다 닿으면 **손절 우선**.
    """
    end = min(t0 + tmax, len(Cl) - 1)
    for u in range(t0, end + 1):
        if sd_pos == 1:
            hit_sl = L[u] <= sl
            hit_tp = H[u] >= tp
        else:
            hit_sl = H[u] >= sl
            hit_tp = L[u] <= tp
        # 봉내 동시 도달의 순서는 1분봉으로 알 수 없다. 기본은 **손절 우선**(하한).
        # tp_first=True 는 익절 우선(상한). 둘을 같이 내야 진실이 그 사이에 있다.
        if tp_first:
            if hit_tp:
                return tp, fee_tp, u, "tp"
            if hit_sl:
                return sl, fee_sl, u, "sl"
        else:
            if hit_sl:
                return sl, fee_sl, u, "sl"
            if hit_tp:
                return tp, fee_tp, u, "tp"
    return Cl[end], fee_to, end, "to"


def _slip(px, sd_pos, why, slip_bp):
    """손절은 시장가로 나간다 — 캐스케이드 중이면 미끄러진다. 불리한 쪽으로 민다.

    익절은 지정가라 미끄러지지 않는다(대신 안 걸릴 뿐이고, 그건 이미 반영돼 있다).
    시간정지도 시장가지만 종가로 재고 있어 이미 실현가격이다.
    """
    if why != "sl" or slip_bp <= 0:
        return px
    # 롱의 손절 = 매도 -> 더 낮게 체결. 숏의 손절 = 매수 -> 더 높게 체결.
    return px * (1.0 - sd_pos * slip_bp * 1e-4)


def run(dd, ww, Q, alpha, cfg, fee_m, fee_t) -> pd.DataFrame:
    """cfg: dict(push_on, eta, s_sl, tmax, W, gamma, r_sl, emax, reb_exit)."""
    q50 = Q[0.5]
    qal = Q[alpha]
    out = []
    for i in range(len(dd)):
        if not (np.isfinite(qal[i]) and np.isfinite(q50[i])):
            continue
        sd = int(dd["side"].iat[i])
        O, H, L, Cl = ww[i]
        p0 = float(O[0])
        r = {"symbol": dd["symbol"].iat[i], "t": dd["t"].iat[i], "day": dd["day"].iat[i],
             "year": dd["year"].iat[i], "side": sd, "q": qal[i],
             "push": 0.0, "reb": 0.0, "ret": 0.0, "filled": False,
             "push_why": "", "reb_why": "", "push_hold": np.nan,
             "wait": np.nan, "reb_hold": np.nan}

        # ---------- [1] 밀림 레그: 움직임 방향 = -sd ----------
        if cfg["push_on"]:
            pos = -sd                                  # sd=+1(하락) -> 숏
            tp = p0 * (1.0 - sd * cfg["eta"] * q50[i] * 1e-4)
            sl = p0 * (1.0 + sd * cfg["s_sl"] * q50[i] * 1e-4)
            px, fo, u, why = _scan(O, H, L, Cl, pos, p0, tp, sl, 0, cfg["tmax"],
                                   fee_m, fee_t, fee_t, cfg["tp_first"])
            px = _slip(px, pos, why, cfg["slip"])
            r["push"] = (px / p0 - 1.0) * pos * 1e4 - (fee_t + fo)
            r["push_why"], r["push_hold"] = why, u

        # ---------- [2] 되돌림 레그: 움직임 반대 = sd ----------
        p_lim = p0 * (1.0 - sd * qal[i] * 1e-4)
        dl = cfg["delta"] * 1e-4
        w = min(cfg["W"], HMAX)
        if sd == 1:
            hit = np.flatnonzero(L[:w + 1] <= p_lim * (1.0 - dl))
        else:
            hit = np.flatnonzero(H[:w + 1] >= p_lim * (1.0 + dl))
        if len(hit):
            fj = int(hit[0])
            p_in = (min(O[fj], p_lim) if sd == 1 else max(O[fj], p_lim)) if fj > 0 else p_lim
            tp = p_in * (1.0 + sd * cfg["gamma"] * q50[i] * 1e-4)
            sl = p_in * (1.0 - sd * cfg["r_sl"] * q50[i] * 1e-4)
            # *** 체결 봉은 건너뛴다 *** 같은 봉의 저가/고가 순서를 모른다.
            if cfg["reb_exit"] == "fixed":
                e = min(fj + cfg["emax"], len(Cl) - 1)
                px, fo, ej, why = Cl[e], fee_t, e, "to"
            else:
                px, fo, ej, why = _scan(O, H, L, Cl, sd, p_in, tp, sl,
                                        fj + 1, cfg["emax"] - 1, fee_m, fee_t, fee_t,
                                        cfg["tp_first"])
                px = _slip(px, sd, why, cfg["slip"])
            r["reb"] = (px / p_in - 1.0) * sd * 1e4 - (fee_m + fo)
            r["filled"], r["wait"], r["reb_hold"], r["reb_why"] = True, fj, ej - fj, why
        r["ret"] = r["push"] + r["reb"]
        out.append(r)
    return pd.DataFrame(out)


def stat(x: pd.DataFrame, col="ret") -> dict:
    if len(x) == 0:
        return {}
    r = x[col].to_numpy()
    m, _, t, _ = cmean(r, x["day"].to_numpy())
    yrs = (x["t"].max() - x["t"].min()) / (365.25 * 86_400_000)
    rt = r[r != 0.0]
    w, l = rt[rt > 0], rt[rt < 0]
    eq = np.cumsum(r)
    return {"n": len(x), "fill": float(x["filled"].mean()), "bp": m, "t": t,
            "med": float(np.median(rt)) if len(rt) else np.nan,
            "win": float((rt > 0).mean()) if len(rt) else np.nan,
            "pl": (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan,
            "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "maxdd": float((eq - np.maximum.accumulate(eq)).min())}


def head():
    print("  %-26s | %5s %6s | %8s %5s %8s | %7s %6s | %6s %8s"
          % ("설정", "n", "체결률", "시도당bp", "t", "중앙bp", "승률", "손익비",
             "샤프", "최대낙폭"))


def line(lab, s):
    if not s:
        return
    print("  %-26s | %5d %6.3f | %8.1f %5.1f %8.1f | %6.1f%% %6.2f | %6.2f %8.0f"
          % (lab, s["n"], s["fill"], s["bp"], s["t"], s["med"],
             100 * s["win"], s["pl"], s["sharpe"], s["maxdd"]))


def base_cfg(**kw):
    c = {"push_on": False, "eta": 1.0, "s_sl": 1.0, "tmax": 5,
         "W": HMAX, "gamma": 1.0, "r_sl": 2.0, "emax": EMAX,
         "reb_exit": "tpsl", "delta": 2.0, "slip": 5.0, "tp_first": False}
    c.update(kw)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="two independent legs with TP/SL/time-stop")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--fee-maker", type=float, default=2.0)
    ap.add_argument("--fee-taker", type=float, default=5.0)
    ap.add_argument("--slip", type=float, default=5.0,
                    help="손절 시장가 슬리피지 bp (익절 지정가에는 안 붙는다)")
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    fm, ft = a.fee_maker, a.fee_taker

    print("=" * 108)
    print("두 레그 독립 — 밀림(슈팅)과 되돌림에 각각 익절·손절·시간정지")
    print("=" * 108)
    d, win = build(syms, a.k, a.doi, a.gap)
    if d is None or len(d) < 300:
        print("이벤트 부족")
        return 1
    alphas = [0.5, 0.7, 0.9]
    Q, oos = walk_forward(d, alphas)
    dd = d[oos].reset_index(drop=True)
    ww = win[oos]
    Qo = {al: Q[al][oos] for al in alphas}
    print("**사용 데이터 기간: %s ~ %s / %d종 / 전체 %d건 / OOS %d건**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d), len(dd)))

    # ---- 밀림은 언제 끝나는가 (시간정지를 데이터로 정한다) ----
    print("\n" + "-" * 108)
    print("0. 밀림은 언제 끝나는가 — 시간정지를 여기서 정한다 (앵커링 금지)")
    print("-" * 108)
    tb = []
    for i in range(len(dd)):
        sd = int(dd["side"].iat[i])
        O, H, L, Cl = ww[i]
        seg = (L[:HMAX + 1] if sd == 1 else H[:HMAX + 1])
        tb.append(int(np.argmin(seg) if sd == 1 else np.argmax(seg)))
    tb = np.array(tb)
    print("  최저(최고)점 도달 시각 분포(분): " +
          " ".join("p%02d %2.0f" % (q, np.percentile(tb, q)) for q in (10, 25, 50, 75, 90)))
    for c in (0, 1, 2, 5, 10, 30):
        print("    %2d분 이내에 끝난 비율 %.3f" % (c, float((tb <= c).mean())))

    print("\n" + "-" * 108)
    print("1. 밀림 레그 단독 — 익절 eta*q50 / 손절 s_sl*q50 / 시간정지 tmax")
    print("-" * 108)
    print("  기존(규칙 없음, 60분 보유) 은 미체결 시 -155.8bp 였다. 규칙을 붙이면?\n")
    head()
    for tmax in (2, 5, 10, 30):
        for eta, ssl in ((0.5, 1.0), (1.0, 1.0), (1.0, 0.5), (1.5, 1.0)):
            c = base_cfg(push_on=True, eta=eta, s_sl=ssl, tmax=tmax, W=0,
                         delta=a.delta, slip=a.slip)
            x = run(dd, ww, Qo, 0.9, c, fm, ft)
            s = stat(x, "push")
            s["fill"] = float((x["push_why"] == "tp").mean())     # 체결률 칸 -> 익절비율
            line("t%02d eta%.1f sl%.1f" % (tmax, eta, ssl), s)
    print("  (체결률 칸은 **익절로 끝난 비율**이다. 나머지는 손절 또는 시간정지)")

    print("\n" + "-" * 108)
    print("2. 되돌림 레그 단독 — 손절을 붙이면 (alpha=0.90)")
    print("-" * 108)
    print("  ** 봉내 동시도달은 순서를 모른다. 손절우선=하한 / 익절우선=상한. 진실은 그 사이. **")
    print()
    head()
    line("손절없음 고정60분",
         stat(run(dd, ww, Qo, 0.9, base_cfg(reb_exit="fixed", emax=60, delta=a.delta,
                                            slip=a.slip), fm, ft), "reb"))
    for gm in (1.0, 2.0, 3.0):
        for rsl in (1.0, 2.0, 4.0, 8.0):
            for tf, tag in ((False, "손절우선"), (True, "익절우선")):
                c = base_cfg(gamma=gm, r_sl=rsl, emax=60, delta=a.delta,
                             slip=a.slip, tp_first=tf)
                line("익절%.0fq 손절%.0fq %s" % (gm, rsl, tag),
                     stat(run(dd, ww, Qo, 0.9, c, fm, ft), "reb"))

    print("\n" + "-" * 108)
    print("2b. 손절 없이 가려면 **꼬리**를 알아야 한다 — 체결 건별 최악 분포")
    print("-" * 108)
    print("  평균·샤프가 좋아도 한 건이 계좌를 날리면 못 쓴다. 레버리지 한도를 여기서 정한다.\n")
    print("  %-24s %6s | %8s %8s %8s %8s | %9s"
          % ("설정", "체결n", "최악", "p01", "p05", "p10", "MAE중앙"))
    for lab, c in (("손절없음 60분", base_cfg(reb_exit="fixed", emax=60, delta=a.delta,
                                          slip=a.slip)),
                   ("익절1q 손절8q", base_cfg(gamma=1.0, r_sl=8.0, emax=60,
                                           delta=a.delta, slip=a.slip)),
                   ("익절3q 손절8q", base_cfg(gamma=3.0, r_sl=8.0, emax=60,
                                           delta=a.delta, slip=a.slip))):
        x = run(dd, ww, Qo, 0.9, c, fm, ft)
        f = x[x["filled"]]
        if not len(f):
            continue
        r = f["reb"].to_numpy()
        print("  %-24s %6d | %8.0f %8.0f %8.0f %8.0f | %9s"
              % (lab, len(f), r.min(), np.percentile(r, 1), np.percentile(r, 5),
                 np.percentile(r, 10), "-"))
    print("  ** 최악 한 건이 -X bp 면, 그 손실이 감당 가능한 크기가 되도록 레버리지를 정해야 한다. **")

    print("\n" + "-" * 108)
    print("3. 두 레그 합산 — 밀림 + 되돌림")
    print("-" * 108)
    head()
    best = None
    for tmax in (2, 5, 10):
        for al in alphas:
            c = base_cfg(push_on=True, eta=1.0, s_sl=1.0, tmax=tmax,
                         gamma=1.0, r_sl=2.0, emax=60, delta=a.delta, slip=a.slip)
            x = run(dd, ww, Qo, al, c, fm, ft)
            s = stat(x)
            line("밀림t%02d + 되돌림a=%.2f" % (tmax, al), s)
            if best is None or s["sharpe"] > best[0]["sharpe"]:
                best = (s, x, "밀림t%02d + 되돌림a=%.2f" % (tmax, al))
    print("\n  레그 분해 (%s):" % best[2])
    x = best[1]
    print("    밀림 평균 %+.1f bp (익절 %.2f / 손절 %.2f / 정지 %.2f)"
          % (x.push.mean(), (x.push_why == "tp").mean(),
             (x.push_why == "sl").mean(), (x.push_why == "to").mean()))
    f = x[x["filled"]]
    print("    되돌림 체결 %d건 평균 %+.1f bp (익절 %.2f / 손절 %.2f / 정지 %.2f)"
          % (len(f), f.reb.mean(), (f.reb_why == "tp").mean(),
             (f.reb_why == "sl").mean(), (f.reb_why == "to").mean()))

    print("\n" + "-" * 108)
    print("4. 안정성 — OOS 전/후반")
    print("-" * 108)
    head()
    h = len(x) // 2
    line(best[2] + " 전체", stat(x))
    line("  전반부", stat(x.iloc[:h]))
    line("  후반부", stat(x.iloc[h:]))
    print("\n  연도별 (밀림 / 되돌림 / 합):")
    print("  %6s %6s %10s %10s %10s" % ("연도", "n", "밀림bp", "되돌림bp", "합bp"))
    for y in sorted(set(x["year"])):
        g = x[x.year == y]
        if len(g) < 5:
            continue
        print("  %6d %6d %10.1f %10.1f %10.1f"
              % (y, len(g), g.push.mean(), g.reb.mean(), g.ret.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
