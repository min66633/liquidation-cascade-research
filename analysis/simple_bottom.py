# -*- coding: utf-8 -*-
"""원래 질문 하나만 — 급락(급등) 뒤 바닥(천장)을 잡을 수 있는가.

사용자 (2026-08-05)
  "단순 급등 혹은 급락을 잡아서 바닥을 잡을 수 있다면 어떨까에서 시작한건데
   지금 점점 처음 주제에서 벗어나서 산으로 가는 느낌"
  "로거 수집은 그만 늘리고 싶은데요"
  "캐스케이드 포착 타임프레임을 늘리는건 동의합니다"

그래서 여기서는
  - 새 데이터 없음. 기존 6년 1분봉만.
  - 방아쇠를 **절대 크기 + 긴 포착 창**으로 단순화. z 사분면·OI·오더북 전부 뺀다.
    (z 만 쓰면 조용한 구간의 작은 움직임을 캐스케이드로 오인한다 — ws_size.py 확인)
  - **가격 손절 없음.** 시간 청산만 (robust.py §5.17.2).
  - 진입 봉부터 청산 검사 (robust.py §3.11).

방아쇠
  D분 누적 하락이 X bp 이상. 즉  Cl[i]/Cl[i-D] - 1 <= -X/1e4.
  같은 심볼 5분 재진입 금지. 진입은 봉 i+1 시가.

네 가지만 묻는다
  1. 바닥은 어디에, 언제 오는가 (진입가 대비 bp, 분)
  2. 바닥 깊이를 방아쇠 시점 정보로 예측할 수 있는가
  3. 시장가 즉시 진입 vs 지정가를 q bp 아래에 — 어느 쪽이 버는가
  4. 급등(반대 방향)도 되는가

실행:
    python analysis/simple_bottom.py
    python analysis/simple_bottom.py --d 30 --x 200
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

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
NOBS = 240                    # 사건 뒤 관찰 창 (분)
W = 116


def prep(syms):
    out = {}
    for s in syms:
        p = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p):
            continue
        m = pd.read_parquet(p, columns=["open_time", "open", "high", "low", "close"])
        m = m.sort_values("open_time").reset_index(drop=True)
        out[s] = tuple(m[c].to_numpy() if c == "open_time"
                       else m[c].to_numpy(dtype=np.float64)
                       for c in ("open_time", "open", "high", "low", "close"))
    return out


def events(P, D, X, gap=5, side=-1):
    """D분 누적 |수익| >= X bp 인 봉. side=-1 급락(매수), +1 급등(매도).

    중복 제거는 **최소 D분**이다. D분 누적 조건은 크래시 뒤 D분 동안 계속 참이라
    gap<D 로 두면 한 캐스케이드가 D/gap 건으로 뻥튀기된다.
    """
    gap = max(gap, D)
    ev = []
    for s, (ot, O, H, L, Cl) in P.items():
        n = len(ot)
        r = np.full(n, np.nan)
        r[D:] = Cl[D:] / np.maximum(Cl[:-D], 1e-12) - 1.0
        cont = np.zeros(n, dtype=bool)
        cont[D:] = (ot[D:] - ot[:-D]) == D * 60_000
        hit = cont & np.isfinite(r) & ((r <= -X * 1e-4) if side < 0 else (r >= X * 1e-4))
        idx = np.flatnonzero(hit)
        last = -10**9
        for i in idx:
            if i + 1 + NOBS >= n:
                continue
            if ot[i] - last < gap * 60_000:
                continue
            j = i + 1
            if ot[j + NOBS] - ot[j] != NOBS * 60_000:
                continue
            seg = slice(j, j + NOBS + 1)
            if not (np.isfinite(O[seg]).all() and np.isfinite(H[seg]).all()
                    and np.isfinite(L[seg]).all() and np.isfinite(Cl[seg]).all()):
                continue
            last = ot[i]
            ev.append((s, i, j, float(r[i])))
    return ev


def frame(P, ev, side=-1):
    """사건별 경로 요약. sd 는 진입 방향 부호 (+1 롱, -1 숏)."""
    sd = 1 if side < 0 else -1               # 급락 -> 매수(롱)
    rows, LO, CL, HI = [], [], [], []
    for s, i, j, r0 in ev:
        ot, O, H, L, Cl = P[s]
        p0 = O[j]
        lo = (L[j:j + NOBS + 1] / p0 - 1.0) * 1e4 * sd     # 유리한 부호로: 역행이 음수
        hi = (H[j:j + NOBS + 1] / p0 - 1.0) * 1e4 * sd
        cl = (Cl[j:j + NOBS + 1] / p0 - 1.0) * 1e4 * sd
        adv = lo if sd == 1 else hi                          # 역행(= 바닥/천장) 경로
        adv = adv if sd == 1 else -( (H[j:j + NOBS + 1] / p0 - 1.0) * 1e4 )
        fav = hi if sd == 1 else -( (L[j:j + NOBS + 1] / p0 - 1.0) * 1e4 )
        tb = int(np.argmin(adv))
        # 방아쇠 시점 특징 (전부 과거)
        k = 1440
        a = max(0, i - k)
        rr = Cl[a:i + 1] / np.maximum(O[a:i + 1], 1e-12) - 1.0
        vol = float(np.nanstd(rr)) * 1e4 if len(rr) > 60 else np.nan
        rows.append({"symbol": s, "i": i, "t": int(ot[j]),
                     "day": int(ot[j]) // 86_400_000,
                     "trig": abs(r0) * 1e4, "vol1m": vol,
                     "bot": float(adv[tb]), "t_bot": tb,
                     "max_fav": float(np.max(cl))})
        LO.append(adv)
        CL.append(cl)
        HI.append(fav)
    return (pd.DataFrame(rows), np.array(LO), np.array(CL), np.array(HI))


def sec(n, t):
    print("\n" + "-" * W)
    print("%d. %s" % (n, t))
    print("-" * W)


def summ(r, day):
    m, se, t, _ = cmean(r, day)
    return m, t


def main() -> int:
    ap = argparse.ArgumentParser(description="can we catch the bottom of a sharp move")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--d", type=int, default=15)
    ap.add_argument("--x", type=float, default=200.0)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--mcost", type=float, default=7.0)   # 메이커 진입 + 테이커 청산
    ap.add_argument("--wait", type=int, default=15)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    P = prep(syms)
    lo = min(int(v[0][0]) for v in P.values())
    hi = max(int(v[0][-1]) for v in P.values())

    print("=" * W)
    print("급락 뒤 바닥을 잡을 수 있는가 — 6년 1분봉, 새 데이터 없음")
    print("=" * W)
    print("심볼 %d종 | 1분봉 %s ~ %s | 관찰 창 %d분"
          % (len(P), pd.to_datetime(lo, unit="ms").strftime("%Y-%m-%d"),
             pd.to_datetime(hi, unit="ms").strftime("%Y-%m-%d"), NOBS))
    print("방아쇠: **%d분 누적 하락 >= %.0fbp** | 같은 심볼 %d분 재진입 금지 | 진입 = 다음 봉 시가"
          % (a.d, a.x, a.gap))
    print("청산: 시간 청산만 (가격 손절 없음)")

    sec(0, "★ 포착 창 D x 크기 X — 사건 수와 바닥 깊이")
    print("  칸: 연간건수 | 바닥 중앙bp | 바닥까지 중앙분\n")
    Ds, Xs = (5, 15, 30, 60), (100, 150, 200, 300, 500)
    yrs = (hi - lo) / (365.25 * 86_400_000)
    print("  %-6s | %s" % ("D분", " ".join("%-21s" % ("%dbp" % x) for x in Xs)))
    for D in Ds:
        out = []
        for X in Xs:
            ev = events(P, D, X, a.gap, -1)
            if len(ev) < 30:
                out.append("%-21s" % ("n=%d" % len(ev)))
                continue
            d, LO, CL, HI = frame(P, ev, -1)
            out.append("%-21s" % ("%5.0f | %5.0f | %4.0f"
                                  % (len(d) / yrs, d["bot"].median(),
                                     d["t_bot"].median())))
        print("  D=%-4d | %s" % (D, " ".join(out)))

    ev = events(P, a.d, a.x, a.gap, -1)
    d, LO, CL, HI = frame(P, ev, -1)
    print("\n" + "=" * W)
    print("이하 상세: D=%d분 하락>=%.0fbp | 사건 %d건 (연 %.0f건) | 심볼 %d종"
          % (a.d, a.x, len(d), len(d) / yrs, d["symbol"].nunique()))
    print("=" * W)

    sec(1, "바닥은 어디에, 언제")
    print("  주의: '바닥' 은 **관찰 창에 의존한다.** 창을 240분으로 두면 캐스케이드 깊이가")
    print("  아니라 4시간 변동성을 재게 된다. 창별로 따로 낸다.\n")
    print("  %-10s | %8s %8s %8s | %8s %8s | %9s"
          % ("관찰 창", "바닥p25", "바닥중앙", "바닥p75", "도달중앙", "0분비율", "최대유리중앙"))
    for Wn in (5, 15, 30, 60, 120, 240):
        adv = LO[:, :Wn + 1]
        b = adv.min(axis=1)
        tb = adv.argmin(axis=1)
        mf = CL[:, :Wn + 1].max(axis=1)
        print("  %-10s | %8.0f %8.0f %8.0f | %8.0f %7.1f%% | %9.0f"
              % ("%d분" % Wn, np.percentile(b, 25), np.median(b),
                 np.percentile(b, 75), np.median(tb),
                 100 * (tb == 0).mean(), np.median(mf)))
    print("\n  ** 창이 길수록 바닥이 깊어지면 그것은 캐스케이드가 아니라 변동성이다. **")
    q = d["t_bot"].to_numpy()
    print("\n  240분 창 기준 바닥 도달 시점 누적분포:")
    print("  0분 %.1f%% | 1분 %.1f%% | 5분 %.1f%% | 15분 %.1f%% | 30분 %.1f%% | 60분 %.1f%%"
          % tuple(100 * (q <= c).mean() for c in (0, 1, 5, 15, 30, 60)))
    print("\n  평균 경로(종가 기준, bp):")
    ts = (1, 3, 5, 10, 15, 30, 60, 120, 240)
    print("  %s" % " ".join("%6s" % ("%d분" % t) for t in ts))
    print("  %s" % " ".join("%6.0f" % np.nanmean(CL[:, t]) for t in ts))
    print("  중앙:")
    print("  %s" % " ".join("%6.0f" % np.nanmedian(CL[:, t]) for t in ts))

    sec(2, "바닥 깊이를 미리 알 수 있는가")
    print("  목표: 바닥 깊이(bp). 설명변수는 방아쇠 시점까지의 정보뿐.\n")
    y = d["bot"].to_numpy()
    for nm, x in (("방아쇠 하락 크기", d["trig"].to_numpy()),
                  ("직전 1일 1분 변동성", d["vol1m"].to_numpy()),
                  ("하락/변동성", d["trig"].to_numpy() / np.maximum(d["vol1m"].to_numpy(), 1e-9))):
        m = np.isfinite(x) & np.isfinite(y)
        xs = (x[m] - x[m].mean()) / (x[m].std() + 1e-12)
        X = np.column_stack([np.ones(m.sum()), xs])
        b, se, _ = ols_cluster(X, y[m], d["day"].to_numpy()[m])
        r2 = 1 - np.var(y[m] - X @ b) / np.var(y[m])
        print("  %-22s | n=%4d 계수 %7.1f  t %5.1f  R2 %.3f"
              % (nm, int(m.sum()), b[1], b[1] / se[1] if se[1] > 0 else np.nan, r2))
    print("\n  ** R2 가 낮으면 '바닥이 얼마나 깊을지' 는 못 맞힌다는 뜻이다. **")

    sec(3, "★ 시장가 즉시 진입 vs 지정가를 q bp 아래에")
    print("  지정가는 방아쇠 다음 봉 시가 대비 q bp 아래. 대기 %d분. 체결되면 그때부터 보유." % a.wait)
    print("  시장가 왕복 %.0fbp / 지정가(메이커 진입) 왕복 %.0fbp\n" % (a.cost, a.mcost))
    Hs = (5, 15, 30, 60, 120)
    print("  %-16s | %6s | %s" % ("진입", "체결률",
                                  " ".join("%14s" % ("보유 %d분" % h) for h in Hs)))
    day = d["day"].to_numpy()
    # 시장가
    out = []
    for h in Hs:
        r = CL[:, h] - a.cost
        m, t = summ(r, day)
        out.append("%14s" % ("%6.1f (%4.1f)" % (m, t)))
    print("  %-16s | %5.0f%% | %s" % ("시장가 즉시", 100, " ".join(out)))
    # 지정가
    WAIT = a.wait
    for qq in (25, 50, 75, 100, 150, 200):
        fill = np.full(len(d), -1)
        for e in range(len(d)):
            w = LO[e, :WAIT + 1]                # 진입가 대비 역행 경로(음수)
            hitb = np.flatnonzero(w <= -qq)
            if len(hitb):
                fill[e] = hitb[0]
        ok = fill >= 0
        if ok.sum() < 30:
            print("  %-16s | %5.1f%% | (체결 표본부족 %d)"
                  % ("-%dbp 지정가" % qq, 100 * ok.mean(), int(ok.sum())))
            continue
        out = []
        for h in Hs:
            r = np.full(len(d), np.nan)
            e_i = np.flatnonzero(ok)
            hh = np.minimum(fill[e_i] + h, NOBS)
            # 체결가 = 진입가*(1 - qq/1e4). 수익은 그 기준으로 다시 잰다.
            base = -qq
            r[e_i] = (CL[e_i, hh] - base) - a.mcost
            m, t = summ(r[e_i], day[e_i])
            # 이벤트당 기대값(미체결은 0)
            ev_m = m * ok.mean()
            out.append("%14s" % ("%6.1f/%5.1f" % (m, ev_m)))
        print("  %-16s | %5.1f%% | %s" % ("-%dbp 지정가" % qq, 100 * ok.mean(),
                                          " ".join(out)))
    print("\n  시장가 칸 = 시도당bp (t). 지정가 칸 = **체결시bp / 이벤트당bp**")
    print("  ** 이벤트당bp 로 비교해야 공정하다. 미체결은 0 이다. **")

    sec(4, "급등(반대 방향)도 되는가")
    print("  같은 방아쇠를 위쪽으로. 급등 %d분 %.0fbp 이상 -> 매도.\n" % (a.d, a.x))
    ev2 = events(P, a.d, a.x, a.gap, +1)
    if len(ev2) < 30:
        print("  n=%d 표본부족" % len(ev2))
    else:
        d2, LO2, CL2, HI2 = frame(P, ev2, +1)
        print("  사건 %d건 (연 %.0f건) | 바닥(천장) 중앙 %.0fbp | 도달 중앙 %.0f분"
              % (len(d2), len(d2) / yrs, d2["bot"].median(), d2["t_bot"].median()))
        print("  %-16s | %s" % ("", " ".join("%14s" % ("보유 %d분" % h) for h in Hs)))
        out = []
        for h in Hs:
            r = CL2[:, h] - a.cost
            m, t = summ(r, d2["day"].to_numpy())
            out.append("%14s" % ("%6.1f (%4.1f)" % (m, t)))
        print("  %-16s | %s" % ("시장가 즉시(숏)", " ".join(out)))

    sec(6, "★★ 그냥 사서 들고 있는 것과 구별되는가 + 자본")
    print("  대조군: 각 사건과 **같은 심볼·같은 날**의 무작위 분에서 같은 규칙으로 매수·청산.")
    print("  심볼 구성과 달력을 동시에 통제한다. 차이가 없으면 방아쇠는 아무 일도 안 한다.\n")
    rng = np.random.default_rng(17)
    NDR = 5
    cr = {h: [] for h in Hs}
    cday = []
    for s, g in d.groupby("symbol"):
        ot, O, H, L, Cl = P[s]
        n = len(ot)
        for t_ev in g["t"].to_numpy():
            d0 = int(t_ev) // 86_400_000
            a0 = int(np.searchsorted(ot, d0 * 86_400_000, side="left"))
            b0 = min(int(np.searchsorted(ot, (d0 + 1) * 86_400_000, side="left")),
                     n - max(Hs) - 1)
            if b0 - a0 < NDR:
                continue
            for jj in rng.integers(a0, b0, NDR):
                jj = int(jj)
                p0 = O[jj]
                if not (np.isfinite(p0) and p0 > 0):
                    continue
                cday.append(d0)
                for h in Hs:
                    v = Cl[jj + h]
                    cr[h].append((v / p0 - 1.0) * 1e4 - a.cost
                                 if np.isfinite(v) else np.nan)
    cday = np.asarray(cday)
    print("  %-18s | %s" % ("", " ".join("%13s" % ("보유 %d분" % h) for h in Hs)))
    ev_row, ct_row, df_row = [], [], []
    for h in Hs:
        re_ = CL[:, h] - a.cost
        rc = np.asarray(cr[h], dtype=np.float64)
        me, te = summ(re_, day)
        mc, tc = summ(rc, cday)
        y = np.concatenate([re_, rc])
        gg = np.concatenate([day, cday])
        dm = np.concatenate([np.ones(len(re_)), np.zeros(len(rc))])
        mm = np.isfinite(y)
        b, se, _ = ols_cluster(np.column_stack([np.ones(mm.sum()), dm[mm]]),
                               y[mm], gg[mm])
        ev_row.append("%13s" % ("%6.1f (%4.1f)" % (me, te)))
        ct_row.append("%13s" % ("%6.1f (%4.1f)" % (mc, tc)))
        df_row.append("%13s" % ("%6.1f (%4.1f)" % (b[1], b[1] / se[1] if se[1] > 0 else np.nan)))
    print("  %-18s | %s" % ("사건 진입", " ".join(ev_row)))
    print("  %-18s | %s" % ("같은날 무작위(n=%d)" % len(cday), " ".join(ct_row)))
    print("  %-18s | %s" % ("★ 차이", " ".join(df_row)))
    print("\n  ** '차이' 가 '사건 진입' 보다 훨씬 작으면 성적의 대부분은 방아쇠가 아니라")
    print("     그 시기·그 종목을 들고 있었다는 사실에서 온다. **")

    print("\n  자본 — 동시보유 (진입 1단위 기준):")
    print("  %-10s | %8s %8s %10s %12s"
          % ("보유", "최대동시", "심볼최대", "평균동시", "자본정규화bp"))
    tt = d["t"].to_numpy()
    for h in Hs:
        e = np.concatenate([tt, tt + h * 60_000])
        dl = np.concatenate([np.ones(len(tt)), -np.ones(len(tt))])
        o = np.lexsort((dl, e))
        cur = np.cumsum(dl[o])
        mx = 0
        for s, g in d.groupby("symbol"):
            t2 = g["t"].to_numpy()
            e2 = np.concatenate([t2, t2 + h * 60_000])
            d2 = np.concatenate([np.ones(len(t2)), -np.ones(len(t2))])
            o2 = np.lexsort((d2, e2))
            mx = max(mx, int(np.cumsum(d2[o2]).max()))
        m, _ = summ(CL[:, h] - a.cost, day)
        yr_bp = m * len(d) / yrs
        print("  %-10s | %8d %8d %10.1f %12.0f"
              % ("%d분" % h, int(cur.max()), mx, cur.mean(),
                 yr_bp / max(int(cur.max()), 1)))

    sec(7, "★ 11.9bp 는 어떤 분포에서 나오나 — 승률인가 손익비인가")
    print("  손절이 없으므로 왼쪽 꼬리가 열려 있다. 그게 이 값의 성격을 정한다.\n")
    print("  %-8s | %6s %6s | %8s %8s %8s | %7s %8s %7s"
          % ("보유", "승률", "손익비", "평균이익", "평균손실", "중앙", "샤프", "최대낙폭", "최악1건"))
    for h in Hs:
        r = CL[:, h] - a.cost
        wn, ls = r[r > 0], r[r < 0]
        m, t = summ(r, day)
        o = np.argsort(d["t"].to_numpy())
        eq = np.cumsum(r[o])
        print("  %-8s | %5.1f%% %6.2f | %8.0f %8.0f %8.0f | %7.2f %8.0f %7.0f"
              % ("%d분" % h, 100 * (r > 0).mean(),
                 wn.mean() / abs(ls.mean()) if len(ls) and len(wn) else np.nan,
                 wn.mean(), ls.mean(), np.median(r),
                 t / np.sqrt(yrs), (eq - np.maximum.accumulate(eq)).min(), r.min()))
    print("\n  건당 수익 분위(bp), 보유 30분:")
    r30 = CL[:, 30] - a.cost
    qs = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    print("  " + "  ".join("p%g=%.0f" % (q, np.percentile(r30, q)) for q in qs))
    srt = np.sort(r30)[::-1]
    for f in (0.01, 0.05, 0.10):
        kk = int(len(r30) * f)
        print("  상위 %2.0f%% (%5d건) 이 순이익의 %5.1f%% | 제거 시 평균 %5.1fbp"
              % (100 * f, kk, 100 * srt[:kk].sum() / r30.sum(), srt[kk:].mean()))

    sec(8, "★ 방아쇠를 키우면 건당 수익이 오르나 — 크기 X 스윕 (보유 30·60분)")
    print("  건당 11.9bp 가 얇다면, 더 큰 사건만 골라 건당을 올릴 수 있는지 본다.\n")
    print("  %-10s | %7s | %s" % ("15분 하락", "연간건수",
                                  " ".join("%26s" % ("보유 %d분: 건당(t) 승률 손익비" % h)
                                           for h in (30, 60))))
    for X in (100, 150, 200, 300, 500, 800):
        ev3 = events(P, a.d, X, a.gap, -1)
        if len(ev3) < 100:
            print("  %-10s | n=%d" % ("%dbp" % X, len(ev3)))
            continue
        d3, LO3, CL3, HI3 = frame(P, ev3, -1)
        out = []
        for h in (30, 60):
            r = CL3[:, h] - a.cost
            m, t = summ(r, d3["day"].to_numpy())
            wn, ls = r[r > 0], r[r < 0]
            out.append("%26s" % ("%7.1f (%4.1f) %5.1f%% %5.2f"
                                 % (m, t, 100 * (r > 0).mean(),
                                    wn.mean() / abs(ls.mean()) if len(ls) and len(wn) else np.nan)))
        print("  %-10s | %7.0f | %s" % ("%dbp" % X, len(d3) / yrs, " ".join(out)))

    sec(9, "★★ 크기 X 별 종합 — 자본과 꼬리까지 넣고 어느 칸을 쓸 것인가")
    print("  보유 %d분 고정. 자본정규화 = 연간총bp / 최대동시보유.\n" % 60)
    HH = 60
    print("  %-8s | %7s %8s %6s | %9s %8s %6s | %8s %9s | %8s %9s"
          % ("하락", "연간", "건당bp", "t", "연간총bp", "최대동시", "샤프",
             "자본정규", "최대낙폭", "상위1%제거", "최악1건"))
    for X in (150, 200, 300, 500, 800):
        ev3 = events(P, a.d, X, a.gap, -1)
        if len(ev3) < 100:
            continue
        d3, LO3, CL3, HI3 = frame(P, ev3, -1)
        r = CL3[:, HH] - a.cost
        dy = d3["day"].to_numpy()
        m, t = summ(r, dy)
        tt3 = d3["t"].to_numpy()
        e = np.concatenate([tt3, tt3 + HH * 60_000])
        dl = np.concatenate([np.ones(len(tt3)), -np.ones(len(tt3))])
        o = np.lexsort((dl, e))
        M = max(int(np.cumsum(dl[o]).max()), 1)
        oo = np.argsort(tt3)
        eq = np.cumsum(r[oo])
        srt = np.sort(r)[::-1]
        k1 = max(1, int(0.01 * len(r)))
        yr_bp = m * len(d3) / yrs
        print("  %-8s | %7.0f %8.1f %6.1f | %9.0f %8d %6.2f | %8.0f %9.0f | %8.1f %9.0f"
              % ("%dbp" % X, len(d3) / yrs, m, t, yr_bp, M, t / np.sqrt(yrs),
                 yr_bp / M, (eq - np.maximum.accumulate(eq)).min(),
                 srt[k1:].mean(), r.min()))
    print("\n  ** 상위1%제거 = 가장 좋은 1% 를 지운 뒤의 건당 평균. 꼬리 의존도를 본다. **")

    sec(10, "★★ 짧은 반등만 먹고 나올 수 있나 — 익절 청산 (손절 없음)")
    print("  '60분 보유' 는 상한이지 목표가 아니다. 익절 지정가를 걸면 반등이 오는 즉시")
    print("  나온다. 보유가 짧아지면 자본이 덜 묶이므로 건당이 줄어도 자본효율은 오를 수 있다.")
    print("  익절은 **대기 지정가**(메이커 %.0fbp 왕복), 시간청산은 시장가(%.0fbp 왕복).\n"
          % (a.mcost, a.cost))

    def tp_run(CLx, HIx, dx, tp, tmax):
        """익절 tp bp 또는 tmax 분. 진입 봉부터 검사(시장가 진입이므로 정당)."""
        n = len(dx)
        hit = HIx[:, :tmax + 1] >= tp
        any_ = hit.any(axis=1)
        first = np.where(any_, hit.argmax(axis=1), tmax)
        r = np.where(any_, tp - a.mcost, CLx[np.arange(n), tmax] - a.cost)
        return r, first, any_

    print("  먼저 익절 없이 **시간 상한만** 줄여 본다 (자본까지 넣고).")
    print("  %-10s | %8s %6s | %6s %6s | %9s %8s %9s"
          % ("상한", "건당bp", "t", "승률", "손익비", "연간총bp", "최대동시", "자본정규"))
    for TMAX in (3, 5, 10, 15, 20, 30, 45, 60, 120):
        r = CL[:, TMAX] - a.cost
        m, t = summ(r, day)
        wn, ls = r[r > 0], r[r < 0]
        tt2 = d["t"].to_numpy()
        e = np.concatenate([tt2, tt2 + TMAX * 60_000])
        dl = np.concatenate([np.ones(len(tt2)), -np.ones(len(tt2))])
        o = np.lexsort((dl, e))
        M = max(int(np.cumsum(dl[o]).max()), 1)
        yr_bp = m * len(d) / yrs
        print("  %-10s | %8.1f %6.1f | %5.1f%% %6.2f | %9.0f %8d %9.0f"
              % ("%d분" % TMAX, m, t, 100 * (r > 0).mean(),
                 wn.mean() / abs(ls.mean()) if len(ls) and len(wn) else np.nan,
                 yr_bp, M, yr_bp / M))
    print()

    for TMAX in (15, 30, 60, 120):
        print("  ── 상한 %d분 ──" % TMAX)
        print("  %-12s | %7s %6s | %7s %6s %6s | %8s %8s %9s"
              % ("익절", "건당bp", "t", "익절률", "승률", "손익비",
                 "평균보유", "최대동시", "자본정규"))
        for tp in (0, 25, 50, 75, 100, 150, 250, 400):
            if tp == 0:
                r = CL[:, TMAX] - a.cost
                first = np.full(len(d), TMAX)
                any_ = np.zeros(len(d), dtype=bool)
            else:
                r, first, any_ = tp_run(CL, HI, d, float(tp), TMAX)
            m, t = summ(r, day)
            wn, ls = r[r > 0], r[r < 0]
            tt2 = d["t"].to_numpy()
            e = np.concatenate([tt2, tt2 + first * 60_000])
            dl = np.concatenate([np.ones(len(tt2)), -np.ones(len(tt2))])
            o = np.lexsort((dl, e))
            M = max(int(np.cumsum(dl[o]).max()), 1)
            yr_bp = m * len(d) / yrs
            print("  %-12s | %7.1f %6.1f | %6.1f%% %5.1f%% %6.2f | %8.1f %8d %9.0f"
                  % ("없음(시간)" if tp == 0 else "+%dbp" % tp, m, t,
                     100 * any_.mean(), 100 * (r > 0).mean(),
                     wn.mean() / abs(ls.mean()) if len(ls) and len(wn) else np.nan,
                     first.mean(), M, yr_bp / M))
        print()
    print("  ** 자본정규 = 연간총bp / 최대동시. 이 값이 오르면 '짧게 먹고 나오기' 가 이긴다. **")

    sec(5, "안정성 — 연도별 (시장가, 보유 30분)")
    dd = d.copy()
    dd["yr"] = pd.to_datetime(dd["t"], unit="ms").dt.year
    print("  %-8s | %6s %10s %7s" % ("연도", "n", "시도당bp", "t"))
    for y_, g in dd.groupby("yr"):
        ii = g.index.to_numpy()
        r = CL[ii, 30] - a.cost
        if len(r) < 20:
            print("  %-8d | %6d (표본부족)" % (y_, len(r)))
            continue
        m, t = summ(r, g["day"].to_numpy())
        print("  %-8d | %6d %10.1f %7.1f" % (y_, len(r), m, t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
