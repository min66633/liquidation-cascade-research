# -*- coding: utf-8 -*-
"""호가창으로 밀림 규모를 **확률적으로** 추정한다. 웹소켓 1초 데이터.

왜 이 형태인가 (설계 원문)
  "호가는 확정 물량이 아니다. MM 이 넣었다 뺐다 한다. 따라서 '호가가 D 이니 V 를
   먹으면 여기까지 간다' 는 **결정론적 계산이 성립하지 않는다.**"
  -> 그래서 점추정이 아니라 **분포** P(X >= u | 호가상태) 를 만든다.

무엇이 바뀌었나 (앞선 실패에서 배운 것)
  1. 5분봉 OI 대신 **1초 호가**를 쓴다. 앞선 V/D 검정들이 전부 5분/30초 해상도라
     '흐름 대 재고' 를 비교하는 오류가 있었다.
  2. 분모에 **Δ(유입·취소)** 를 넣는다. depth_ws_flow 에 이미 있다.
  3. X 를 **짧은 지평(15/60/300초)** 으로 잰다. 240분은 V/D 검정에 부적합했다.
  4. 로그-로그만이 아니라 **임계형**도 같이 본다.
  5. 합격은 R^2 가 아니라 **캘리브레이션**(Kupiec / PIT / 핀볼).

모형
      log X = m_t + Z,   m_t = a + b1 log sigma + b2 log(D_eff) + b3 log(불균형)
      D_eff = D + Δ      (압력받는 쪽 깊이 + 그 구간 순유입)
      P(X >= u | F_t) = P(Z >= log u - m_t),  Z = 훈련구간 잔차 경험분포

  b2 < 0 이어야 한다 — 깊이가 두꺼우면 덜 밀린다.

*** 표본 한계: 웹소켓 2일. 캐스케이드 0건. 함수형은 검정되나 규모 외삽은 불가. ***

실행:
    python analysis/book_prob.py
    python analysis/book_prob.py --band b0_5 --hor 60
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
from analysis.response_liq import ols_cluster, cmean            # noqa: E402
from analysis.synth import kupiec, ks_unif, pinball, LEVELS     # noqa: E402

HORS = [15, 60, 300]          # 밀림 거리 지평(초)


def load_ws(band: str):
    """depth_ws + depth_ws_flow 를 (심볼, 초) 로 합친다."""
    fb = sorted(glob.glob(os.path.join(C.DATA, "depth_ws", "*", "*.parquet")))
    ff = sorted(glob.glob(os.path.join(C.DATA, "depth_ws_flow", "*", "*.parquet")))
    cb = ["ts_ms", "symbol", "mid", "bid_" + band, "ask_" + band]
    cf = ["ts_ms", "symbol", "dbid_" + band, "dask_" + band]
    b = pd.concat([pd.read_parquet(f, columns=cb) for f in fb], ignore_index=True)
    f = pd.concat([pd.read_parquet(f, columns=cf) for f in ff], ignore_index=True)
    b["sec"] = b["ts_ms"] // 1000
    f["sec"] = f["ts_ms"] // 1000
    b = b.drop_duplicates(["symbol", "sec"], keep="last")
    f = f.drop_duplicates(["symbol", "sec"], keep="last")
    d = b.merge(f[["symbol", "sec", "dbid_" + band, "dask_" + band]],
                on=["symbol", "sec"], how="left")
    return d.sort_values(["symbol", "sec"]).reset_index(drop=True)


def build(band: str, hors):
    d = load_ws(band)
    bid, ask = "bid_" + band, "ask_" + band
    dbid, dask = "dbid_" + band, "dask_" + band
    out = []
    for s, g in d.groupby("symbol"):
        g = g.sort_values("sec")
        sec = g["sec"].to_numpy()
        mid = g["mid"].to_numpy(dtype=np.float64)
        B = g[bid].to_numpy(dtype=np.float64)
        A = g[ask].to_numpy(dtype=np.float64)
        dB = np.nan_to_num(g[dbid].to_numpy(dtype=np.float64))
        dA = np.nan_to_num(g[dask].to_numpy(dtype=np.float64))
        n = len(mid)
        if n < 3000:
            continue
        lm = np.log(np.maximum(mid, 1e-12))
        r1 = np.concatenate([[np.nan], np.diff(lm)])
        sig = pd.Series(r1).rolling(120, min_periods=60).std().to_numpy()
        # *** Δ 재정의 (2026-08-04) ***
        # 1초 **순변화**는 유입과 취소가 상쇄돼 중앙이 0.0000 이었다 — 분모를 못 움직인다.
        # 설계의 Δ(v,t) 가 잡으려는 것은 **호가의 회복력**이므로,
        # 과거 60초의 **총유입 / 총취소**를 따로 누적해 쓴다(전방 정보 아님).
        addB = pd.Series(np.maximum(dB, 0)).rolling(60, min_periods=10).sum().to_numpy()
        remB = pd.Series(np.maximum(-dB, 0)).rolling(60, min_periods=10).sum().to_numpy()
        addA = pd.Series(np.maximum(dA, 0)).rolling(60, min_periods=10).sum().to_numpy()
        remA = pd.Series(np.maximum(-dA, 0)).rolling(60, min_periods=10).sum().to_numpy()
        # 격자 연속성 (1초). 끊긴 구간은 버린다.
        cont = np.concatenate([[False], np.diff(sec) == 1])
        rec = {"symbol": s, "sec": sec, "mid": mid,
               "D_bid": B, "D_ask": A, "F_bid": dB, "F_ask": dA,
               "AddB": addB, "RemB": remB, "AddA": addA, "RemA": remA,
               "sig": sig, "cont": cont}
        # 양방향 X: 아래로 밀림(롱 관점) / 위로 밀림(숏 관점)
        # **전방 롤링 극값을 벡터화한다.** 파이썬 루프면 심볼당 5천만 연산이 된다
        # (n=17만 x H=300). 배열을 뒤집어 rolling 을 걸고 다시 뒤집으면 O(n) 이다.
        rv = pd.Series(mid[::-1])
        for H in hors:
            fmin = rv.rolling(H + 1, min_periods=1).min().to_numpy()[::-1]
            fmax = rv.rolling(H + 1, min_periods=1).max().to_numpy()[::-1]
            xd = (mid - fmin) / mid * 1e4
            xu = (fmax - mid) / mid * 1e4
            # 창이 끝을 넘어가는 구간과 격자가 끊긴 구간은 버린다
            bad = ~cont
            bad[max(n - H, 0):] = True
            xd[bad] = np.nan
            xu[bad] = np.nan
            rec["xd%d" % H] = xd
            rec["xu%d" % H] = xu
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True) if out else None


def main() -> int:
    ap = argparse.ArgumentParser(description="probabilistic push-size model from book")
    ap.add_argument("--band", default="b0_5")
    ap.add_argument("--hor", type=int, default=60)
    ap.add_argument("--train", type=float, default=0.70)
    a = ap.parse_args()
    U.init_stdout()

    print("=" * 80)
    print("호가창 확률모형 — P(X >= u | 호가상태). 웹소켓 1초, 밴드 %s" % a.band)
    print("=" * 80)
    d = build(a.band, HORS)
    if d is None or len(d) < 5000:
        print("표본 부족")
        return 1
    d = d.dropna(subset=["sig", "xd%d" % a.hor])
    d = d[(d["sig"] > 0) & (d["D_bid"] > 0) & (d["D_ask"] > 0)]
    print("**사용 데이터 기간: %s ~ %s / %d종 / %d초-관측**"
          % (str(pd.Timestamp(d.sec.min() * 1000, unit="ms"))[:19],
             str(pd.Timestamp(d.sec.max() * 1000, unit="ms"))[:19],
             d.symbol.nunique(), len(d)))

    # 하방/상방을 하나의 축으로 쌓는다 — 압력받는 쪽 깊이를 분모로
    parts = []
    for lab, xc, Dc, Ac, Rc in (
            ("down", "xd%d" % a.hor, "D_bid", "AddB", "RemB"),
            ("up", "xu%d" % a.hor, "D_ask", "AddA", "RemA")):
        p = pd.DataFrame({
            "X": d[xc].to_numpy(), "D": d[Dc].to_numpy(),
            "Add": d[Ac].to_numpy(), "Rem": d[Rc].to_numpy(),
            "sig": d["sig"].to_numpy(),
            "imb": (d["D_bid"] - d["D_ask"]).to_numpy()
                   / np.maximum(d["D_bid"] + d["D_ask"], 1e-9),
            "sec": d["sec"].to_numpy(), "symbol": d["symbol"].to_numpy(),
            "side": lab})
        if lab == "up":
            p["imb"] = -p["imb"]          # 압력받는 쪽 기준으로 부호 통일
        parts.append(p)
    p = pd.concat(parts, ignore_index=True)
    p = p[np.isfinite(p["X"]) & (p["X"] > 0) & np.isfinite(p["Add"])].copy()
    # 유효깊이 = 현재 재고 + 60초간 순회복(총유입 - 총취소). 음수면 바닥을 씌운다.
    p["Deff"] = np.maximum(p["D"] + p["Add"] - p["Rem"], p["D"] * 0.1)
    p["reg"] = p["Add"] / np.maximum(p["D"], 1e-9)      # 회복력(유입/재고)
    p["cxl"] = p["Rem"] / np.maximum(p["D"], 1e-9)      # 취소압(취소/재고)
    p = p.sort_values("sec").reset_index(drop=True)
    print("X 중앙 %.2f bp | D 중앙 $%.4g | **총유입/D 중앙 %.3f | 총취소/D 중앙 %.3f**"
          % (p.X.median(), p.D.median(), p.reg.median(), p.cxl.median()))
    print("  (첫 판은 1초 **순변화**를 썼고 중앙이 0.0000 이라 분모를 못 움직였다.")
    print("   설계의 Δ 는 회복력이므로 60초 **총유입/총취소**로 다시 정의했다.)")

    y = np.log(p["X"].to_numpy())
    ls = np.log(p["sig"].to_numpy())
    lD = np.log(p["D"].to_numpy())
    lE = np.log(p["Deff"].to_numpy())
    ib = p["imb"].to_numpy()
    hr = (p["sec"].to_numpy() // 3600)
    ok = np.isfinite(y) & np.isfinite(ls) & np.isfinite(lD) & np.isfinite(lE)

    print("\n" + "-" * 80)
    print("1. 깊이가 밀림을 줄이는가 — b2 는 **음수**여야 한다")
    print("-" * 80)
    print("  %-28s %9s %7s | %9s %7s | %7s"
          % ("설정", "b1(sig)", "t", "**b2(깊이)**", "t", "R^2"))
    lg = np.log(np.maximum(p["reg"].to_numpy(), 1e-9))
    lc = np.log(np.maximum(p["cxl"].to_numpy(), 1e-9))
    ok = ok & np.isfinite(lg) & np.isfinite(lc) & np.isfinite(ib)
    specs = [("sigma 만", [ls]),
             ("+ log D (재고)", [ls, lD]),
             ("+ log D_eff = D+유입-취소", [ls, lE]),
             ("+ log D + **회복력·취소압 분리**", [ls, lD, lg, lc]),
             ("+ 위 전부 + 불균형", [ls, lD, lg, lc, ib])]
    for lab, cols in specs:
        X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
        b, se, _ = ols_cluster(X, y[ok], hr[ok])
        r2 = 1.0 - np.var(y[ok] - X @ b) / np.var(y[ok])
        c2 = ("%9.4f %7.1f" % (b[2], b[2] / se[2])) if len(b) > 2 else "%9s %7s" % ("-", "-")
        extra = ""
        if len(b) > 4:
            extra = "  [회복 %+.4f(t=%.1f) 취소 %+.4f(t=%.1f)]" % (
                b[3], b[3] / se[3], b[4], b[4] / se[4])
        print("  %-30s %9.4f %7.1f | %s | %7.4f%s"
              % (lab, b[1], b[1] / se[1], c2, r2, extra))
    print("  회복력 계수가 **음수**면 유입이 많을 때 덜 밀린다 = 설계의 Δ 가 작동한다.")
    print("  취소압 계수가 **양수**면 호가가 빠질 때 더 밀린다 = 유령 유동성.")

    print("\n" + "-" * 80)
    print("2. 비선형성 — log D 에 대해 꺾이는가 (임계형의 일반화)")
    print("-" * 80)
    print("  첫 판의 max(0, 1/D-c)*1e9 는 스케일이 터져 R^2 가 음수였다. 폐기.")
    print("  대신 log D 를 분위 매듭으로 **구간선형**화해 꺾임을 직접 본다.\n")
    kn = np.quantile(lD[ok], [0.2, 0.4, 0.6, 0.8])
    cols = [ls[ok], lD[ok]] + [np.maximum(lD[ok] - k, 0.0) for k in kn]
    X = np.column_stack([np.ones(int(ok.sum()))] + cols)
    b, se, _ = ols_cluster(X, y[ok], hr[ok])
    r2 = 1.0 - np.var(y[ok] - X @ b) / np.var(y[ok])
    print("  기울기(log D) 기저 %+.4f (t=%.1f)" % (b[2], b[2] / se[2]))
    slope = b[2]
    for i, k in enumerate(kn):
        slope += b[3 + i]
        print("    매듭 p%02d 이후 누적기울기 %+.4f   (증분 %+.4f, t=%.1f)"
              % (20 * (i + 1), slope, b[3 + i], b[3 + i] / se[3 + i]))
    print("  구간선형 R^2 = %.4f" % r2)
    print("  기울기가 얕은 쪽(작은 D)에서 **더 가파르면** 임계형 구조를 지지한다.")

    print("\n" + "-" * 80)
    print("3. 확률모형 캘리브레이션 — R^2 가 아니라 이것이 합격 기준")
    print("-" * 80)
    cut = int(len(p) * a.train)
    tr = np.zeros(len(p), dtype=bool)
    tr[:cut] = True
    Xf = np.column_stack([np.ones(len(p)), ls, lD, lg, lc, ib])
    okf = ok
    bt = np.linalg.pinv(Xf[tr & okf].T @ Xf[tr & okf]) @ (Xf[tr & okf].T @ y[tr & okf])
    m_te = Xf[~tr & okf] @ bt
    Z = y[tr & okf] - Xf[tr & okf] @ bt
    Z0 = y[tr & okf]
    xte = p["X"].to_numpy()[~tr & okf]
    print("  훈련 %d / 검정 %d | b=%s"
          % (int((tr & okf).sum()), len(xte), np.round(bt, 3)))
    from math import erf
    def p1(v):
        return np.nan if not np.isfinite(v) else float(1.0 - erf(np.sqrt(max(v, 0) / 2)))
    print("\n  *** n=%d 이면 Kupiec/KS 는 1%% 미만 편차도 기각한다." % len(xte))
    print("      **실질 지표는 위반율이 목표에서 얼마나 벗어났는가**다.\n")
    print("  %-6s | %-24s | %-24s | %s" % ("수준p", "M(호가)", "M0(무조건부)", "개선"))
    dev = {"M": [], "M0": []}
    for lv in LEVELS:
        cells, rates = [], []
        for mm, ZZ in ((m_te, Z), (np.zeros(len(xte)), Z0)):
            qq = np.exp(mm + float(np.quantile(ZZ, lv)))
            v = (xte < qq).astype(int)
            lr = kupiec(len(v), int(v.sum()), lv)
            rates.append(abs(v.mean() - lv))
            cells.append("위반%5.3f (편차%+.3f)" % (v.mean(), v.mean() - lv))
        dev["M"].append(rates[0])
        dev["M0"].append(rates[1])
        print("  %-6.2f | %-24s | %-24s | %+.3f" % (lv, *cells, rates[1] - rates[0]))
    print("  %-6s | 평균편차 %.4f%13s | 평균편차 %.4f%12s | **%.1f%% 감소**"
          % ("합계", np.mean(dev["M"]), "", np.mean(dev["M0"]), "",
             100 * (1 - np.mean(dev["M"]) / max(np.mean(dev["M0"]), 1e-12))))
    t1 = t0 = 0.0
    for lv in LEVELS:
        t1 += pinball(xte, np.exp(m_te + float(np.quantile(Z, lv))), lv)
        t0 += pinball(xte, np.full(len(xte), np.exp(float(np.quantile(Z0, lv)))), lv)
    for lab, mm, ZZ in (("M(호가)", m_te, Z), ("M0", np.zeros(len(xte)), Z0)):
        # PIT 은 Z 의 경험CDF 다. 리스트 컴프리헨션으로 하면 O(len(xte) x len(Z)) 가
        # 되어 백만 단위 표본에서 조 단위 연산이 된다(실측: 528초에 출력 0).
        # 정렬 + searchsorted 로 O(n log n) 이다.
        Zs = np.sort(ZZ)
        u = np.searchsorted(Zs, np.log(xte) - mm, side="right") / float(len(Zs))
        ks, pv = ks_unif(np.clip(u, 1e-6, 1 - 1e-6))
        print("  %-10s PIT KS D=%.4f p=%.4f %s"
              % (lab, ks, pv, "합격" if pv > 0.05 else "**불합격**"))
    print("  핀볼 합계 M %.1f | M0 %.1f -> **개선 %.1f%%**"
          % (t1, t0, 100 * (t0 - t1) / t0))
    print("\n  *** 웹소켓 2일 표본이다. 캐스케이드 0건 — **함수형만** 검정된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
