# -*- coding: utf-8 -*-
"""D-10 — §6.17 최선 설정의 견고성.

무엇을 의심하는가
  §6.17 의 설정(1분봉 z1<=-K, 직전 알려진 5분 dOI<=thr, 매수만, 익8s/손2s/15분)은
  **격자에서 골라낸 칸**이다. 전/후반 분할만 통과했다. 그것으로는
  아래 네 가지가 전혀 통제되지 않는다.

    1. 심볼별 분해   — 21종 중 몇 종이 우위를 만드나. 하나 빼면 무너지나
    2. 비용 민감도   — 왕복 10bp 는 가정이다. 손익분기 비용은 얼마인가
    3. 체결 가정     — 다음봉 시가 시장가 / 익절 지정가 즉시체결 / 손절 정확체결.
                       셋 다 낙관이다. 캐스케이드 직후엔 스프레드가 벌어진다
    4. 이익 집중     — 상위 5% 가 총이익의 76%. t 통계량은 정규근사에 기댄다

  추가로 5. 시간 안정성(연도별) 과 6. 파라미터 이웃(격자 최고칸인가 고원인가).

사용자 지시 반영 (2026-08-05)
  "진입하고 5분까지 같은 심볼에 진입 금지" -> **gap=5 가 기본값**이다.
  §6.17 표에 적힌 '중복제거 없음'(gap=1) 은 격자 최고칸이지 지시된 설정이 아니다.
  두 설정을 나란히 보고한다.

실행:
    python analysis/robust.py
    python analysis/robust.py --sections 1 2 4
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
from analysis.response_liq import cmean                                # noqa: E402
from analysis.fast_trigger import prep, events, concurrency            # noqa: E402

MINS_YR = 365.25 * 24 * 60
W = 122

# ---- 기본 설정 (사용자 지시 = gap 5) ----------------------------------------
BASE = dict(k=10.0, thr=-0.005, gap=5, tmax=15, tp=8.0, sl=2.0,
            cost=10.0, slip=5.0)


# ============================================================ 시뮬레이터
def run2(prepped, k, thr, gap, tmax, tp, sl, cost, slip,
         entry_slip=0.0, tp_buf=0.0, tp_first=False, gap_stop=False,
         to_slip=0.0, scan_entry_bar=False, sl_on_close=False) -> pd.DataFrame:
    """fast_trigger.run 의 확장판. 체결 가정을 전부 인자로 뺐다.

    entry_slip : 진입(매수) 시 불리한 슬리피지 bp. 시가보다 비싸게 산다
    tp_buf     : 익절 지정가가 체결되려면 고가가 목표를 이만큼(bp) **초과**해야 한다
                 (큐 뒤에 서 있어 스치기만 해선 안 채워지는 경우)
    tp_first   : 봉 안에서 익절/손절 동시 도달 시 익절을 먼저 잡는다 (낙관 상한)
    gap_stop   : 손절 봉이 손절가 **아래에서 시가**를 냈으면 시가에 체결 (갭 돌파)
    to_slip    : 시간정지 시장가 청산의 슬리피지 bp
    scan_entry_bar : 진입 봉 j 부터 청산을 본다. 진입은 O[j] 이고 그 봉의 L/H 는
        진입 이후 경로이므로 룩어헤드가 아니다. 이것을 끄면(기본, fast_trigger 와
        동일) 진입 봉 안에서 손절선을 뚫고 내려간 경우가 다음 봉으로 **미뤄져**
        갭 돌파처럼 보인다. 즉 gap_stop 의 손실 대부분은 이 스킵의 인공물이다.

    TP/SL 수준은 **실제 체결가 p_in** 기준으로 잡는다 (실거래와 동일).
    """
    rows = []
    for s, (ot1, O, H, L, Cl, z1, doi, sig) in prepped.items():
        n1 = len(ot1)
        for i in events(ot1, z1, doi, sig, k, thr, gap):
            j = i + 1
            if j + tmax >= n1 or ot1[j + tmax] - ot1[j] != tmax * 60_000:
                continue
            p0, sg = O[j], sig[i]
            if not (np.isfinite(p0) and p0 > 0 and np.isfinite(sg) and sg > 0):
                continue
            if not (np.isfinite(O[j:j + tmax + 1]).all()
                    and np.isfinite(H[j:j + tmax + 1]).all()
                    and np.isfinite(L[j:j + tmax + 1]).all()
                    and np.isfinite(Cl[j:j + tmax + 1]).all()):
                continue
            p_in = p0 * (1.0 + entry_slip * 1e-4)
            tpx, slx = p_in * (1.0 + tp * sg), p_in * (1.0 - sl * sg)
            tp_touch = tpx * (1.0 + tp_buf * 1e-4)

            p_out, why, hold, gapped = Cl[j + tmax] * (1.0 - to_slip * 1e-4), "to", tmax, 0
            u0 = j if scan_entry_bar else j + 1
            for u in range(u0, j + tmax + 1):
                hit_sl = (Cl[u] <= slx) if sl_on_close else (L[u] <= slx)
                hit_tp = H[u] >= tp_touch
                if hit_sl and hit_tp:
                    first = "tp" if tp_first else "sl"
                elif hit_sl:
                    first = "sl"
                elif hit_tp:
                    first = "tp"
                else:
                    continue
                if first == "sl":
                    if sl_on_close:
                        # 종가 기준 손절: 종가에 시장가로 나간다 (웍 손절 회피)
                        fill, gapped = Cl[u], 0
                    else:
                        # 진입 봉에서는 O[j] 에 이미 들어와 있어 갭이 있을 수 없다
                        ref = p_in if u == j else O[u]
                        gapped = int(ref < slx)
                        fill = min(ref, slx) if gap_stop else slx
                    p_out, why, hold = fill * (1.0 - slip * 1e-4), "sl", u - j
                else:
                    p_out, why, hold = tpx, "tp", u - j
                break
            rows.append({"symbol": s, "t": int(ot1[j]),
                         "t_exit": int(ot1[j]) + hold * 60_000,
                         "ret": (p_out / p_in - 1.0) * 1e4 - cost,
                         "why": why, "hold": hold, "z": z1[i], "sg": sg,
                         "gapped": gapped,
                         "day": int(ot1[j]) // 86_400_000})
    return pd.DataFrame(rows)


def summ(d, w=None):
    """w 가 주어지면 가중 수익(= 사이징) 기준으로 집계한다."""
    if d is None or len(d) < 30:
        return None
    r = d["ret"].to_numpy(dtype=np.float64)
    if w is not None:
        r = r * np.asarray(w, dtype=np.float64)
    m, se, t, _ = cmean(r, d["day"].to_numpy())
    yrs = (d["t"].max() - d["t"].min()) / (365.25 * 86_400_000)
    per_yr = len(d) / yrs if yrs > 0 else np.nan
    win, los = r[r > 0], r[r < 0]
    eq = np.cumsum(r)
    return {"n": len(d), "per_yr": per_yr, "bp": m, "se": se, "t": t,
            "win": float((r > 0).mean()),
            "pl": (win.mean() / abs(los.mean())) if len(los) and len(win) else np.nan,
            "yr_bp": m * per_yr, "sharpe": t / np.sqrt(yrs) if yrs > 0 else np.nan,
            "maxdd": float((eq - np.maximum.accumulate(eq)).min()),
            "hold": float(d["hold"].median()), "yrs": yrs}


def head(lab="설정"):
    print("  %-24s | %6s %7s | %8s %5s | %7s %6s | %10s %6s %8s"
          % (lab, "n", "연간", "시도당bp", "t", "승률", "손익비",
             "연간총bp", "샤프", "최대낙폭"))


def line(lab, s):
    if s is None:
        print("  %-24s | (표본부족)" % lab)
        return
    print("  %-24s | %6d %7.0f | %8.1f %5.1f | %6.1f%% %6.2f | %10.0f %6.2f %8.0f"
          % (lab, s["n"], s["per_yr"], s["bp"], s["t"], 100 * s["win"], s["pl"],
             s["yr_bp"], s["sharpe"], s["maxdd"]))


def sec(n, title):
    print("\n" + "-" * W)
    print("%d. %s" % (n, title))
    print("-" * W)


# ============================================================ 부트스트랩
def boot_day(r, day, reps=4000, seed=7):
    """일 블록 부트스트랩. 꼬리가 두꺼우면 t 의 정규근사가 깨진다."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(day, return_inverse=True)
    D = inv.max() + 1
    dsum = np.bincount(inv, weights=r, minlength=D)
    dcnt = np.bincount(inv, minlength=D).astype(np.float64)
    pick = rng.integers(0, D, size=(reps, D))
    tot, cnt = dsum[pick].sum(1), dcnt[pick].sum(1)
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out, int(D)


# ============================================================ 각 절
def s1_symbols(P, cfg):
    sec(1, "심볼별 분해 — 우위가 몇 종에서 나오나. 하나 빼면 무너지나")
    d = run2(P, **cfg)
    print("  전체: n=%d, 심볼 %d종\n" % (len(d), d["symbol"].nunique()))
    print("  %-10s | %6s %8s %5s | %7s %6s | %9s"
          % ("심볼", "n", "시도당bp", "t", "승률", "손익비", "총bp기여"))
    tot = d["ret"].sum()
    rows = []
    for s, g in d.groupby("symbol"):
        r = g["ret"].to_numpy()
        m, _, t, _ = cmean(r, g["day"].to_numpy())
        w, l = r[r > 0], r[r < 0]
        pl = (w.mean() / abs(l.mean())) if len(l) and len(w) else np.nan
        rows.append((s, len(g), m, t, (r > 0).mean(), pl, r.sum(), r.sum() / tot))
    rows.sort(key=lambda x: -x[6])
    for s, n, m, t, wr, pl, ssum, shr in rows:
        print("  %-10s | %6d %8.1f %5.1f | %6.1f%% %6.2f | %8.1f%%"
              % (s, n, m, t, 100 * wr, pl, 100 * shr))
    pos = sum(1 for x in rows if x[2] > 0)
    sig = sum(1 for x in rows if np.isfinite(x[3]) and x[3] > 2)
    print("\n  양(+) 심볼 %d/%d, t>2 인 심볼 %d/%d" % (pos, len(rows), sig, len(rows)))
    top3 = sum(x[7] for x in rows[:3])
    print("  상위 3종이 총이익의 %.1f%%" % (100 * top3))

    print("\n  ** 심볼 하나씩 제거 (leave-one-out) — 나빠지는 순서로 상위 6종 **")
    head("제거 심볼")
    loo = []
    for s in d["symbol"].unique():
        x = d[d["symbol"] != s]
        ss = summ(x)
        if ss:
            loo.append((s, ss))
    loo.sort(key=lambda x: x[1]["t"])
    for s, ss in loo[:6]:
        line("-%s" % s, ss)
    line("(제거 없음)", summ(d))
    print("\n  ** 최악의 leave-one-out t = %.1f (%s 제거) **"
          % (loo[0][1]["t"], loo[0][0]))

    # 메이저 vs 알트
    maj = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}
    print()
    head("군")
    line("메이저 5종", summ(d[d["symbol"].isin(maj)]))
    line("알트 16종", summ(d[~d["symbol"].isin(maj)]))
    return d


def s2_cost(P, cfg, d0):
    sec(2, "비용 민감도 — 왕복 10bp 는 가정이다. 손익분기는 어디인가")
    r0 = d0["ret"].to_numpy() + cfg["cost"]          # cost=0 환산 (상수 이동)
    m0, se0, _, _ = cmean(r0, d0["day"].to_numpy())
    print("  비용은 건당 상수 차감이므로 표준오차가 변하지 않는다. 정확히 풀린다.")
    print("  비용 0 일 때: 시도당 %.1fbp, SE %.2f" % (m0, se0))
    print("  -> 손익분기 비용 = %.1f bp (왕복)" % m0)
    print("     t>=2 유지 한계 = %.1f bp,  t>=3 = %.1f bp"
          % (m0 - 2 * se0, m0 - 3 * se0))
    print("\n  실측 대조: Binance USDT-M 테이커 4.5bp/편도 = **왕복 9bp** (VIP0, BNB 할인 없음)")
    print("            BNB 10% 할인 시 8.1bp. 메이커 2bp/편도.\n")
    head("왕복 비용")
    for c in (0, 5, 8, 10, 12, 15, 20, 25, 30):
        x = d0.copy()
        x["ret"] = r0 - c
        line("%dbp" % c, summ(x))
    print("\n  ** 손절 슬리피지 별도 스윕 (왕복 10bp 고정) — 손절은 시장가다 **")
    head("손절 슬리피지")
    for sp in (0, 5, 10, 20, 40):
        cc = dict(cfg); cc["slip"] = float(sp)
        line("%dbp" % sp, summ(run2(P, **cc)))


def s3_fill(P, cfg):
    sec(3, "체결 가정 — 캐스케이드 직후 시장가 매수는 시가에 안 채워진다")
    print("  기본 가정 3개가 전부 낙관이다:")
    print("   (a) 진입: 다음 1분봉 **시가**에 시장가 체결")
    print("   (b) 익절: 고가가 목표를 스치기만 해도 지정가 체결")
    print("   (c) 손절: 손절가에 **정확히** 체결 (갭 하락 무시)\n")
    head("진입 슬리피지")
    for es in (0, 2, 5, 10, 20):
        cc = dict(cfg); cc["entry_slip"] = float(es)
        line("+%dbp 비싸게 매수" % es, summ(run2(P, **cc)))

    print("\n  ** 익절 지정가 큐 — 고가가 목표를 이만큼 넘어야 체결 **")
    head("익절 체결 버퍼")
    for tb in (0, 2, 5, 10, 20):
        cc = dict(cfg); cc["tp_buf"] = float(tb)
        line("+%dbp 초과해야 체결" % tb, summ(run2(P, **cc)))

    print("\n  ** 손절 갭 돌파 — 손절 봉의 시가가 손절가 아래면 시가 체결 **")
    print("  주의: fast_trigger 는 **진입 봉 j 를 청산 검사에서 건너뛴다.** 그래서")
    print("  진입 봉 안에서 손절선을 뚫은 경우가 다음 봉으로 밀려 '갭' 처럼 보인다.")
    print("  진입 봉부터 검사하면(scan_entry_bar) 그 인공물이 사라진다.\n")
    head("손절 체결")
    for lab, kw in (("진입봉 건너뜀 · 정확체결", {"scan_entry_bar": False}),
                    ("진입봉 건너뜀 · 갭체결",
                     {"scan_entry_bar": False, "gap_stop": True}),
                    ("진입봉부터 검사 · 정확체결", {"scan_entry_bar": True}),
                    ("진입봉부터 검사 · 갭체결",
                     {"scan_entry_bar": True, "gap_stop": True})):
        x = run2(P, **dict(cfg, **kw))
        s = summ(x)
        line(lab, s)
        sl_n = int((x["why"] == "sl").sum())
        gp = int(x["gapped"].sum())
        print("      └ 손절 %d건 중 갭 돌파 %d건 (%.1f%%)"
              % (sl_n, gp, 100 * gp / max(sl_n, 1)))

    print("\n  ** 봉내 순서 모호성 — 1분봉으로는 익절·손절 도달 순서를 알 수 없다 **")
    head("봉내 순서")
    line("손절 우선 (하한, 기본)", summ(run2(P, **cfg)))
    cc = dict(cfg); cc["tp_first"] = True
    line("익절 우선 (상한)", summ(run2(P, **cc)))

    print("\n  ** 종합 — 진입봉부터 검사(현실적 기준선) 위에 나쁜 가정을 쌓는다 **")
    head("종합")
    base_r = dict(cfg, scan_entry_bar=True)
    line("기본 가정 (진입봉 건너뜀)",
         summ(run2(P, **dict(cfg, scan_entry_bar=False))))
    line("현실 기준선 (진입봉부터)", summ(run2(P, **base_r)))
    line("+ 갭손절", summ(run2(P, **dict(base_r, gap_stop=True))))
    line("보수 가정", summ(run2(P, **dict(base_r, entry_slip=5.0, tp_buf=5.0,
                                       gap_stop=True, slip=10.0, cost=12.0))))
    line("극단 가정", summ(run2(P, **dict(base_r, entry_slip=10.0, tp_buf=10.0,
                                       gap_stop=True, slip=20.0, cost=15.0,
                                       to_slip=5.0))))


def s4_conc(P, cfg, d0):
    sec(4, "이익 집중 — 상위 몇 건이 전부라면 t 는 믿을 수 없다")
    r = d0["ret"].to_numpy()
    srt = np.sort(r)[::-1]
    gross = srt[srt > 0].sum()
    print("  n=%d, 총합 %.0fbp, 양(+)건 합 %.0fbp" % (len(r), r.sum(), gross))
    qs = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    print("  건당 수익 분위(bp): " + " ".join(
        "p%g=%.0f" % (q, np.percentile(r, q)) for q in qs))
    print("  최악 1건 %.0fbp / 최선 1건 %.0fbp | 손절이 없으면 왼쪽 꼬리가 자본을 묶는다"
          % (r.min(), r.max()))
    print("\n  %-10s | %8s %12s %12s" % ("상위", "건수", "양건합 비중", "총합 비중"))
    for q in (0.01, 0.02, 0.05, 0.10, 0.20):
        kk = max(1, int(round(len(r) * q)))
        print("  %-10s | %8d %11.1f%% %11.1f%%"
              % ("%.0f%%" % (100 * q), kk, 100 * srt[:kk].sum() / gross,
                 100 * srt[:kk].sum() / r.sum()))

    print("\n  ** 상위 절사 — 상위 x%%를 지운 뒤에도 우위가 남나 **")
    head("절사")
    for q in (0.0, 0.01, 0.02, 0.05, 0.10):
        kk = int(round(len(r) * q))
        thr_ = srt[kk - 1] if kk > 0 else np.inf
        x = d0[d0["ret"] < thr_] if kk > 0 else d0
        line("상위 %.0f%% 제거" % (100 * q), summ(x))

    print("\n  ** 윈저화 — 지우지 않고 상단만 눌러 담는다 (실거래의 부분익절에 가깝다) **")
    head("윈저")
    for q in (1.00, 0.99, 0.95, 0.90):
        x = d0.copy()
        x["ret"] = np.minimum(r, np.quantile(r, q))
        line("상위 %.0f%% 절단" % (100 * (1 - q)), summ(x))

    print("\n  ** 일 블록 부트스트랩 — t 의 정규근사를 쓰지 않는다 **")
    cut = srt[max(1, int(0.05 * len(r))) - 1]
    for lab, dd in (("기본", d0), ("상위 5% 제거", d0[d0["ret"] < cut])):
        rr = dd["ret"].to_numpy()
        b, D = boot_day(rr, dd["day"].to_numpy())
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        p = float(np.nanmean(b <= 0))
        print("  %-14s 평균 %6.1fbp | 95%% CI [%6.1f, %6.1f] | P(<=0)=%.4f | 일수 %d"
              % (lab, np.nanmean(b), lo, hi, p, D))

    print("\n  ** 사이징 — 변동성 역가중으로 담으면 집중이 풀리나 **")
    sg = d0["sg"].to_numpy()
    w = np.clip(np.median(sg) / np.maximum(sg, 1e-12), 0.25, 4.0)
    head("사이징")
    line("동일 명목 (기본)", summ(d0))
    line("변동성 역가중", summ(d0, w=w))
    x = d0.copy(); x["ret"] = d0["ret"].to_numpy() * w
    rr = x["ret"].to_numpy(); s2 = np.sort(rr)[::-1]
    kk = max(1, int(0.05 * len(rr)))
    print("  변동성 역가중 후 상위 5%% 비중: %.1f%% (기본 %.1f%%)"
          % (100 * s2[:kk].sum() / s2[s2 > 0].sum(), 100 * srt[:kk].sum() / gross))


def s5_time(P, cfg, d0):
    sec(5, "시간 안정성 — 한 해가 만든 것인가")
    d = d0.copy()
    d["yr"] = pd.to_datetime(d["t"], unit="ms").dt.year
    head("연도")
    for y, g in d.groupby("yr"):
        line("%d" % y, summ(g))
    print()
    head("사분")
    d = d.sort_values("t").reset_index(drop=True)
    q = len(d) // 4
    for i, lab in enumerate(["1사분", "2사분", "3사분", "4사분"]):
        g = d.iloc[i * q:(i + 1) * q] if i < 3 else d.iloc[3 * q:]
        line("%s (%s~%s)" % (lab,
                             pd.to_datetime(g["t"].min(), unit="ms").strftime("%y-%m"),
                             pd.to_datetime(g["t"].max(), unit="ms").strftime("%y-%m")),
             summ(g))
    print("\n  ** 연도별 t 가 모두 양수여야 '한 해가 만든 것' 이 아니다 **")


def s6_grid(P, cfg):
    sec(6, "파라미터 이웃 — 고원인가 뾰족한 한 칸인가")
    print("  기준점: K=%.0f dOI=%.3f gap=%d 익%.0fs 손%.0fs 상한%d분\n"
          % (cfg["k"], cfg["thr"], cfg["gap"], cfg["tp"], cfg["sl"], cfg["tmax"]))
    print("  (a) 방아쇠 격자 K x dOI — 칸 값은 시도당bp (t) [n]")
    ks = [8, 9, 10, 11, 12]
    ths = [-0.002, -0.005, -0.010, -0.020]
    print("  %-6s | %s" % ("K", " ".join("%-20s" % ("dOI<=%.3f" % t) for t in ths)))
    cells = []
    for k in ks:
        out = []
        for th in ths:
            cc = dict(cfg); cc["k"] = float(k); cc["thr"] = th
            s = summ(run2(P, **cc))
            if s is None:
                out.append("%-20s" % "-")
                continue
            cells.append(s)
            out.append("%-20s" % ("%6.1f (%4.1f) [%4d]" % (s["bp"], s["t"], s["n"])))
        print("  %-6.0f | %s" % (k, " ".join(out)))
    ok = sum(1 for s in cells if s["t"] > 2)
    print("\n  t>2 인 칸: %d/%d (%.0f%%)" % (ok, len(cells), 100 * ok / max(len(cells), 1)))

    print("\n  (b) 청산 격자 익절 x 손절 (배수 x sigma5) — 시도당bp (t) [샤프]")
    tps, sls = [4, 6, 8, 12, 16], [1.5, 2.0, 3.0, 4.0]
    print("  %-6s | %s" % ("익절", " ".join("%-20s" % ("손절 %.1f" % s) for s in sls)))
    cells2 = []
    for tp in tps:
        out = []
        for sl in sls:
            cc = dict(cfg); cc["tp"] = float(tp); cc["sl"] = float(sl)
            s = summ(run2(P, **cc))
            if s is None:
                out.append("%-20s" % "-")
                continue
            cells2.append(s)
            out.append("%-20s" % ("%6.1f (%4.1f) [%4.2f]" % (s["bp"], s["t"], s["sharpe"])))
        print("  %-6.0f | %s" % (tp, " ".join(out)))
    ok2 = sum(1 for s in cells2 if s["t"] > 2)
    print("\n  t>2 인 칸: %d/%d (%.0f%%)" % (ok2, len(cells2), 100 * ok2 / max(len(cells2), 1)))

    print("\n  (c) 보유 상한")
    head("상한")
    for tm in (5, 10, 15, 20, 30, 60):
        cc = dict(cfg); cc["tmax"] = tm
        line("%d분" % tm, summ(run2(P, **cc)))


def s7_gap(P, cfg):
    sec(7, "★ 지시된 설정(gap=5) vs 격자 최고칸(gap=1) — 자본까지 맞추고 비교")
    print("  사용자 지시: '진입하고 5분까지 같은 심볼에 진입 금지' -> gap=5")
    print("  §6.17 표의 '중복제거 없음'(gap=1) 은 연간총bp 를 최대화한 칸이다.\n")
    print("  %-16s | %6s %7s | %8s %5s | %9s | %7s %7s | %10s"
          % ("설정", "n", "연간", "시도당bp", "t", "연간총bp", "최대동시", "심볼최대", "자본정규화"))
    for g in (1, 3, 5, 10, 15):
        cc = dict(cfg); cc["gap"] = g
        d = run2(P, **cc)
        s = summ(d)
        if s is None:
            continue
        c = concurrency(d)
        M = max(c["max_all"], 1)
        print("  gap=%-11dm | %6d %7.0f | %8.1f %5.1f | %9.0f | %7d %7d | %10.0f"
              % (g, s["n"], s["per_yr"], s["bp"], s["t"], s["yr_bp"],
                 c["max_all"], c["max_sym"], s["yr_bp"] / M))
    print("\n  ** 자본정규화 = 연간총bp / 최대동시보유. 같은 자본 기준의 비교다. **")


def s8_parts(P, cfg, d0):
    sec(8, "부품 기여도와 종목 선택 편향 — dOI 는 정말 필요한가, 21종은 사후선택인가")
    print("  (a) dOI 필터를 완전히 빼면? '큰 하락 뒤 매수' 와 구별되는가")
    head("dOI 조건")
    for th in (np.inf, -0.001, -0.002, -0.005, -0.010, -0.020):
        cc = dict(cfg); cc["thr"] = th
        lab = "없음 (가격만)" if not np.isfinite(th) else "dOI<=%.3f" % th
        line(lab, summ(run2(P, **cc)))
    print("\n  ** 'dOI 없음' 대비 개선이 없으면 설계의 청산 부품은 장식이다 **")

    print("\n  (b) 종목 선택 편향 — 21종은 2026년 시점에서 고른 목록이다.")
    early = sorted(s for s, v in P.items()
                   if int(v[0][0]) < pd.Timestamp("2021-01-01").value // 10**6)
    late = sorted(set(P) - set(early))
    print("      2021-01-01 이전 상장 %d종: %s" % (len(early), ", ".join(early)))
    print("      이후 상장 %d종: %s" % (len(late), ", ".join(late)))
    head("표본")
    line("전체 21종", summ(d0))
    line("2021 이전 상장만", summ(d0[d0["symbol"].isin(early)]))
    if late:
        line("2021 이후 상장만", summ(d0[d0["symbol"].isin(late)]))
    print("\n  ** '이후 상장' 이 우위를 독점하면 사후선택 편향을 배제할 수 없다 **")

    print("\n  (c) 날짜 집중 — 이익이 며칠에 몰려 있나")
    g = d0.groupby("day")["ret"].sum().sort_values(ascending=False)
    tot = g.sum()
    print("      거래 발생일 %d일 | 상위 1일 %.1f%% / 3일 %.1f%% / 10일 %.1f%% / 20일 %.1f%%"
          % (len(g), 100 * g.iloc[:1].sum() / tot, 100 * g.iloc[:3].sum() / tot,
             100 * g.iloc[:10].sum() / tot, 100 * g.iloc[:20].sum() / tot))
    top = g.index[:5]
    print("      최대 5일: %s" % ", ".join(
        "%s(%+.0fbp,%d건)" % (pd.to_datetime(dd * 86_400_000, unit="ms").strftime("%Y-%m-%d"),
                             g.loc[dd], int((d0["day"] == dd).sum())) for dd in top))
    head("표본")
    line("전체", summ(d0))
    line("상위 3일 제거", summ(d0[~d0["day"].isin(g.index[:3])]))
    line("상위 10일 제거", summ(d0[~d0["day"].isin(g.index[:10])]))


def s9_exit(P, cfg):
    sec(9, "★ 청산 규칙 재최적화 — 손절 모형을 고친 뒤에 다시 고른다")
    print("  §3 이 보인 것: 진입 봉을 청산 검사에서 건너뛰면 손절이 **미뤄지고**,")
    print("  미뤄진 손절이 이미 지나간 가격(손절가)에 체결된 것으로 계산된다.")
    print("  그 인공물이 §6.17 의 성적 전부였다. 여기서는 진입 봉부터 검사한다.\n")
    base = dict(cfg, scan_entry_bar=True)

    print("  (a) 손절 없음 — 시간정지만. **이 설정은 위 인공물의 영향을 받지 않는다**")
    head("보유 상한")
    for tm in (1, 2, 3, 5, 10, 15, 30, 60):
        line("%d분 종가 청산" % tm,
             summ(run2(P, **dict(base, tmax=tm, tp=999.0, sl=999.0))))

    print("\n  (b) 익절만 (손절 없음) — 상한 15분")
    head("익절")
    for tp in (2, 4, 6, 8, 12, 16):
        line("%.0f*sigma5" % tp,
             summ(run2(P, **dict(base, tp=float(tp), sl=999.0))))

    print("\n  (c) 손절 폭 — 얼마나 넓혀야 살아나나 (익절 8*sigma5, 상한 15분)")
    head("손절")
    for sl in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 999.0):
        lab = "없음" if sl > 100 else "%.1f*sigma5" % sl
        x = run2(P, **dict(base, sl=float(sl)))
        s = summ(x)
        line(lab, s)
        if s:
            wy = x["why"].value_counts(normalize=True)
            print("      └ 익절 %.2f / 손절 %.2f / 시간정지 %.2f"
                  % (wy.get("tp", 0), wy.get("sl", 0), wy.get("to", 0)))

    print("\n  (d) 익절 x 상한 격자 (손절 없음) — 시도당bp (t) [샤프]")
    tps, tms = [4, 6, 8, 12, 999], [3, 5, 10, 15, 30]
    print("  %-8s | %s" % ("익절", " ".join("%-20s" % ("상한 %d분" % m) for m in tms)))
    cells = []
    for tp in tps:
        out = []
        for tm in tms:
            s = summ(run2(P, **dict(base, tp=float(tp), sl=999.0, tmax=tm)))
            if s is None:
                out.append("%-20s" % "-")
                continue
            cells.append(s)
            out.append("%-20s" % ("%6.1f (%4.1f) [%4.2f]" % (s["bp"], s["t"], s["sharpe"])))
        print("  %-8s | %s" % ("없음" if tp > 100 else "%d*s" % tp, " ".join(out)))
    ok = sum(1 for s in cells if s["t"] > 2)
    print("\n  t>2 인 칸: %d/%d" % (ok, len(cells)))

    print("\n  (e) 손절을 **종가 기준**으로 — 웍에 털리지 않는다. 실행 가능한 규칙이다")
    print("      익절은 대기 지정가이므로 웍 체결이 정당하다. 손절만 종가 판정.")
    head("손절 판정")
    for sl in (1.0, 2.0, 3.0, 4.0, 6.0):
        x = run2(P, **dict(base, sl=float(sl), sl_on_close=True))
        s = summ(x)
        line("종가<= -%.1f*sigma5" % sl, s)
        if s:
            wy = x["why"].value_counts(normalize=True)
            print("      └ 익절 %.2f / 손절 %.2f / 시간정지 %.2f"
                  % (wy.get("tp", 0), wy.get("sl", 0), wy.get("to", 0)))

    print("\n  (f) 봉내 순서 상한 — 손절을 쓰는 설정에서만 문제가 된다")
    head("봉내 순서")
    line("손절 우선 (하한)", summ(run2(P, **base)))
    line("익절 우선 (상한)", summ(run2(P, **dict(base, tp_first=True))))
    line("종가손절 2s · 손절우선", summ(run2(P, **dict(base, sl_on_close=True))))
    line("종가손절 2s · 익절우선",
         summ(run2(P, **dict(base, sl_on_close=True, tp_first=True))))


def s10_hold(P, cfg):
    sec(10, "★ 보유 시간 — 60분이 또 격자 경계다. 어디서 꺾이는지 끝까지 본다")
    print("  §9(a)·§6(c) 는 60분에서 단조 증가로 끝났다. 그것은 답이 아니라")
    print("  격자의 끝이다 (§3.7 에 이미 적어 둔 오류). 240분까지 민다.\n")
    print("  주의: tmax 가 길수록 '연속된 1분봉 tmax개' 요건 때문에 표본이 줄어든다.")
    print("        n 을 같이 봐야 한다.\n")
    base = dict(cfg, scan_entry_bar=True, sl=999.0)
    for tag, tp in (("익절 8*sigma5", 8.0), ("익절 없음 (시간청산만)", 999.0)):
        print("  ** %s **" % tag)
        print("  %-10s | %6s %7s | %8s %5s | %9s %6s %8s | %8s %8s"
              % ("상한", "n", "연간", "시도당bp", "t", "연간총bp", "샤프",
                 "최대낙폭", "최대동시", "자본정규"))
        for tm in (15, 30, 45, 60, 90, 120, 180, 240):
            d = run2(P, **dict(base, tp=tp, tmax=tm))
            s = summ(d)
            if s is None:
                continue
            c = concurrency(d)
            M = max(c["max_all"], 1)
            print("  %-10s | %6d %7.0f | %8.1f %5.1f | %9.0f %6.2f %8.0f | %8d %8.0f"
                  % ("%d분" % tm, s["n"], s["per_yr"], s["bp"], s["t"],
                     s["yr_bp"], s["sharpe"], s["maxdd"], c["max_all"],
                     s["yr_bp"] / M))
        print()
    print("  ** 자본정규화(연간총bp / 최대동시보유) 가 꺾이는 지점이 실제 최적이다. **")


def s11_null(P, cfg, ndraw=20, seed=11):
    sec(11, "★★ 귀무가설 대조 — 방아쇠가 실제로 무엇을 고르는가")
    print("  §10 에서 보유 시간이 240분까지 단조 증가했다. 꺾이지 않는다.")
    print("  그렇다면 이것은 '캐스케이드 반등'이 아니라 그냥 **매수 후 보유**일 수 있다.")
    print("  표본 기간(2020-09~2026-07)은 상승장을 포함하고 종목은 고베타 알트다.\n")
    print("  대조군: 각 이벤트와 **같은 심볼·같은 날**에서 무작위 분을 %d개 뽑아" % ndraw)
    print("  똑같은 규칙(다음 봉 시가 진입, tmax분 종가 청산, 왕복 %.0fbp)으로 청산한다."
          % cfg["cost"])
    print("  심볼 구성과 달력을 동시에 통제한다. 차이가 없으면 방아쇠는 아무것도 안 한다.\n")
    rng = np.random.default_rng(seed)
    cost = cfg["cost"]
    print("  %-8s | %8s %8s %6s | %8s %8s | %10s %6s"
          % ("상한", "이벤트bp", "대조군bp", "대조n", "차이bp", "차이 t", "이벤트 t", "샤프"))
    for tm in (15, 30, 60, 120, 240):
        ev = run2(P, **dict(cfg, scan_entry_bar=True, sl=999.0, tp=999.0, tmax=tm))
        if len(ev) < 30:
            continue
        cr, cday = [], []
        for s, (ot1, O, H, L, Cl, z1, doi, sig) in P.items():
            g = ev[ev["symbol"] == s]
            if not len(g):
                continue
            n1 = len(ot1)
            for t_ev in g["t"].to_numpy():
                d0 = int(t_ev) // 86_400_000
                a = int(np.searchsorted(ot1, d0 * 86_400_000, side="left"))
                b = int(np.searchsorted(ot1, (d0 + 1) * 86_400_000, side="left"))
                b = min(b, n1 - tm - 1)
                if b - a < ndraw:
                    continue
                for jj in rng.integers(a, b, ndraw):
                    jj = int(jj)
                    if ot1[jj + tm] - ot1[jj] != tm * 60_000:
                        continue
                    p0, p1 = O[jj], Cl[jj + tm]
                    if not (np.isfinite(p0) and p0 > 0 and np.isfinite(p1)):
                        continue
                    cr.append((p1 / p0 - 1.0) * 1e4 - cost)
                    cday.append(d0)
        cr, cday = np.asarray(cr), np.asarray(cday)
        me, _, te, _ = cmean(ev["ret"].to_numpy(), ev["day"].to_numpy())
        mc, _, tc, _ = cmean(cr, cday) if len(cr) >= 30 else (np.nan,) * 4
        # 차이의 t: 두 표본을 하나의 회귀로 (더미). 일 클러스터.
        y = np.concatenate([ev["ret"].to_numpy(), cr])
        gday = np.concatenate([ev["day"].to_numpy(), cday])
        dmy = np.concatenate([np.ones(len(ev)), np.zeros(len(cr))])
        from analysis.response_liq import ols_cluster
        X = np.column_stack([np.ones(len(y)), dmy])
        bb, se, _ = ols_cluster(X, y, gday)
        s = summ(ev)
        print("  %-8s | %8.1f %8.1f %6d | %8.1f %8.1f | %10.1f %6.2f"
              % ("%d분" % tm, me, mc, len(cr), bb[1],
                 bb[1] / se[1] if se[1] > 0 else np.nan, te, s["sharpe"]))
    print("\n  ** 차이 t 가 이벤트 t 보다 크게 작으면, 성적의 상당 부분은")
    print("     '방아쇠' 가 아니라 '그 시기·그 종목을 들고 있었다' 는 사실에서 온다. **")


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description="D-10 robustness of the best config")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--gap", type=int, default=BASE["gap"])
    ap.add_argument("--k", type=float, default=BASE["k"])
    ap.add_argument("--sections", type=int, nargs="*",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--fix", action="store_true",
                    help="진입 봉부터 청산 검사 (손절 모형 정정). §1/2/4/5/7 재계산용")
    ap.add_argument("--tp", type=float, default=None)
    ap.add_argument("--sl", type=float, default=None)
    ap.add_argument("--tmax", type=int, default=None)
    a = ap.parse_args()
    U.init_stdout()
    cfg = dict(BASE); cfg["gap"] = a.gap; cfg["k"] = a.k
    cfg["scan_entry_bar"] = bool(a.fix)
    for nm in ("tp", "sl", "tmax"):
        if getattr(a, nm) is not None:
            cfg[nm] = getattr(a, nm)

    print("=" * W)
    print("D-10 — §6.17 최선 설정의 견고성")
    print("=" * W)
    print("설정: 1분봉 z1<=-%.0f **그리고** 직전 알려진 5분 dOI<=%.3f | 매수만"
          % (cfg["k"], cfg["thr"]))
    print("      같은 심볼 %d분 재진입 금지 | 다음 1분봉 시가 시장가 진입" % cfg["gap"])
    print("      익절 %.0f*sigma5 / 손절 %.0f*sigma5(+슬립 %.0fbp) / %d분 중 먼저 | 왕복 %.0fbp"
          % (cfg["tp"], cfg["sl"], cfg["slip"], cfg["tmax"], cfg["cost"]))
    print("      청산 검사 시작: %s"
          % ("진입 봉부터 (정정)" if cfg["scan_entry_bar"] else "진입 다음 봉부터 (fast_trigger 원본)"))

    syms = a.symbols if a.symbols else C.MAJORS
    P = prep(syms)
    lo = min(int(v[0][0]) for v in P.values())
    hi = max(int(v[0][-1]) for v in P.values())
    print("\n심볼 %d종 | 1분봉 패널 %s ~ %s"
          % (len(P), pd.to_datetime(lo, unit="ms").strftime("%Y-%m-%d"),
             pd.to_datetime(hi, unit="ms").strftime("%Y-%m-%d")))

    d0 = run2(P, **cfg)
    s = summ(d0)
    print("기준 성적: n=%d, 연 %.0f건, 시도당 %.1fbp (t=%.1f), 샤프 %.2f, 사건 %s ~ %s"
          % (s["n"], s["per_yr"], s["bp"], s["t"], s["sharpe"],
             pd.to_datetime(d0["t"].min(), unit="ms").strftime("%Y-%m-%d"),
             pd.to_datetime(d0["t"].max(), unit="ms").strftime("%Y-%m-%d")))

    if 1 in a.sections:
        s1_symbols(P, cfg)
    if 2 in a.sections:
        s2_cost(P, cfg, d0)
    if 3 in a.sections:
        s3_fill(P, cfg)
    if 4 in a.sections:
        s4_conc(P, cfg, d0)
    if 5 in a.sections:
        s5_time(P, cfg, d0)
    if 6 in a.sections:
        s6_grid(P, cfg)
    if 7 in a.sections:
        s7_gap(P, cfg)
    if 8 in a.sections:
        s8_parts(P, cfg, d0)
    if 9 in a.sections:
        s9_exit(P, cfg)
    if 10 in a.sections:
        s10_hold(P, cfg)
    if 11 in a.sections:
        s11_null(P, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
