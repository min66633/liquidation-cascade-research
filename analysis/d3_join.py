# -*- coding: utf-8 -*-
"""D-3 — 설계의 세 부품을 **전부** 결합한다. 웹소켓 1초 + 지도.

    X_hat = inf{ u : ∫[D(v,t) + Δ(v,t)]dv >= ∫L(p_t(1-v))dv · φ }
              ①지도 L   ②깊이 D   ③유입 Δ

확정된 것 (여기 오기까지)
  ② 깊이       b2 = -0.067 (t=-20.8)  — 작동      book_prob.py
  ③ 회복력     -0.036 (t=-3.8)        — 작동      book_prob.py
  ③ 취소압     -0.010 (t=-1.3)        — 미검출
  ① 지도       캐스케이드 구간에서만 +1.38 (t=2.3) — 약함  build_map.py

  **셋을 같이 넣은 적이 한 번도 없다.** 그것이 이 스크립트다.

검정
      log X = a + b1 log(sigma) + b2 log(D) + b3 log(회복력) + b4 log(L)
  b4 > 0 이고 유의하면 **지도가 깊이 너머의 정보를 갖는다** = 설계 성립.
  그리고 비율 형태 log(L/D_eff) 도 같이 본다 — 설계식이 그 형태다.

구조
  지도 L 은 5분 OI 코호트로 만든다(반감기 10.5일 -> 3개월 룩백이면 99.7%).
  깊이는 웹소켓 1초. **L 을 5분 격자로 만들고 초 격자에 전방채움**한다.

*** 표본 한계: 웹소켓 약 32시간. 캐스케이드 0건. 함수형만 검정된다. ***

실행:
    python analysis/d3_join.py
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
from analysis.map_kernel import EDGES, MAXC                     # noqa: E402
from analysis.build_map import load_kernels                     # noqa: E402
from analysis.book_prob import build as build_book              # noqa: E402
from analysis.synth import kupiec, ks_unif, pinball, LEVELS     # noqa: E402

LOOKBACK_D = 100          # 코호트 룩백(일). 반감기 10.5일 -> 100일이면 사실상 전부
BANDS = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10)]


def series(sym, t0_ms, t1_ms):
    """(시각, OI계약, 가격) 5분 격자. **두 소스를 이어붙인다.**

    벌크 klines_5m/metrics 는 T-1 일별 아카이브라 최근 며칠이 비어 있다
    (실측: 2026-08-01 에서 끝나는데 웹소켓 창은 08-02~04).
    실시간 폴링 open_interest_hist 가 그 구간을 덮고, 가격도 거기서 나온다 —
    sumOpenInterestValue / sumOpenInterest = 마크가격.
    """
    lo_ms = t0_ms - LOOKBACK_D * 86_400_000
    parts = []
    try:
        df = load(sym)
        ot = df["open_time"].to_numpy()
        m = (ot >= lo_ms) & (ot <= t1_ms)
        if m.sum() > 100:
            parts.append(pd.DataFrame({
                "t": ot[m],
                "oi": df["sum_open_interest"].to_numpy(dtype=np.float64)[m],
                "px": df["close"].to_numpy(dtype=np.float64)[m]}))
    except FileNotFoundError:
        pass
    p = os.path.join(C.DATA, "binance_futures_data", "open_interest_hist",
                     "%s.parquet" % sym)
    if os.path.exists(p):
        q = pd.read_parquet(p)
        t = q["timestamp"].to_numpy()
        m = (t >= lo_ms) & (t <= t1_ms)
        if m.sum() > 10:
            oi = q["sumOpenInterest"].to_numpy(dtype=np.float64)[m]
            vv = q["sumOpenInterestValue"].to_numpy(dtype=np.float64)[m]
            with np.errstate(invalid="ignore", divide="ignore"):
                px = vv / np.where(oi > 0, oi, np.nan)
            parts.append(pd.DataFrame({"t": t[m], "oi": oi, "px": px}))
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d["oi"]) & (d["oi"] > 0)
          & np.isfinite(d["px"]) & (d["px"] > 0)]
    d = d.drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
    return d


def fuel_panel(sym, t0_ms, t1_ms, dmid, dw, hvec):
    """웹소켓 창 구간의 5분 격자 연료(하방/상방)를 만든다."""
    d = series(sym, t0_ms, t1_ms)
    if d is None or len(d) < 2000:
        return None
    oi = d["oi"].to_numpy()
    px = d["px"].to_numpy()
    ot = d["t"].to_numpy()
    n = len(oi)
    lp = np.log(px)
    nb = len(EDGES) - 1

    cp = [np.empty(MAXC), np.empty(MAXC)]
    cq = [np.zeros(MAXC), np.zeros(MAXC)]
    cm = [np.empty(MAXC), np.empty(MAXC)]
    nc = [1, 1]
    for sd in (0, 1):
        cp[sd][0], cq[sd][0], cm[sd][0] = lp[0], oi[0], lp[0]

    rows = []
    for t in range(1, n):
        dq = oi[t] - oi[t - 1]
        tot = cq[0][:nc[0]].sum() + cq[1][:nc[1]].sum()
        if tot > 0 and ot[t] >= t0_ms:
            fd = np.zeros(len(BANDS))       # 하방 연료 (롱 코호트)
            fu = np.zeros(len(BANDS))       # 상방 연료 (숏 코호트)
            for sd, tgt in ((0, fd), (1, fu)):
                if nc[sd] == 0:
                    continue
                sgn = -1.0 if sd == 0 else 1.0      # 롱은 아래, 숏은 위에서 청산
                lq = cp[sd][:nc[sd]][:, None] + sgn * np.log(1.0 - dmid)[None, :] * -1.0
                lq = cp[sd][:nc[sd]][:, None] + (np.log(1.0 - dmid) if sd == 0
                                                 else -np.log(1.0 - dmid))[None, :]
                rel = (1.0 - np.exp(lq - lp[t - 1])) if sd == 0 else \
                      (np.exp(lq - lp[t - 1]) - 1.0)
                alive = (lq < cm[sd][:nc[sd]][:, None]) if sd == 0 else \
                        (lq > cm[sd][:nc[sd]][:, None])
                amt = cq[sd][:nc[sd]][:, None] * dw[None, :] * alive
                for bi, (b0, b1) in enumerate(BANDS):
                    tgt[bi] = float(amt[(rel > b0) & (rel <= b1)].sum())
            rows.append((ot[t], oi[t - 1], *fd, *fu))

        for sd in (0, 1):
            if nc[sd] > 0:
                if sd == 0:
                    np.minimum(cm[0][:nc[0]], lp[t], out=cm[0][:nc[0]])
                else:
                    np.maximum(cm[1][:nc[1]], lp[t], out=cm[1][:nc[1]])
        if dq > 0:
            for sd in (0, 1):
                if nc[sd] < MAXC:
                    cp[sd][nc[sd]], cq[sd][nc[sd]], cm[sd][nc[sd]] = lp[t], dq, lp[t]
                    nc[sd] += 1
                else:
                    j = int(np.argmin(cq[sd][:nc[sd]]))
                    s2 = cq[sd][j] + dq
                    cp[sd][j] = (cp[sd][j] * cq[sd][j] + lp[t] * dq) / max(s2, 1e-12)
                    cq[sd][j] = s2
        elif dq < 0 and tot > 0:
            need = min(2.0 * (-dq), tot)
            wts = []
            for sd in (0, 1):
                x = (lp[t - 1] - cp[sd][:nc[sd]]) * (1.0 if sd == 0 else -1.0)
                wts.append(hvec[np.digitize(x, EDGES[1:-1])] * cq[sd][:nc[sd]])
            tw = wts[0].sum() + wts[1].sum()
            for sd in (0, 1):
                if tw > 0:
                    cq[sd][:nc[sd]] -= np.minimum(need * wts[sd] / tw, cq[sd][:nc[sd]])
                else:
                    cq[sd][:nc[sd]] *= max(1.0 - need / tot, 0.0)
    if not rows:
        return None
    A = np.array(rows, dtype=np.float64)
    out = pd.DataFrame({"t5": A[:, 0].astype(np.int64), "symbol": sym, "oi": A[:, 1]})
    for i in range(len(BANDS)):
        out["Ld%d" % i] = A[:, 2 + i]
        out["Lu%d" % i] = A[:, 2 + len(BANDS) + i]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="D-3 join map with book depth")
    ap.add_argument("--band", default="b0_5")
    ap.add_argument("--hor", type=int, default=60)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--casc-csv", default=None,
                    help="cascade segments csv (sym,def,start,dur_min) -> D-9 조건부 섹션 출력")
    a = ap.parse_args()
    U.init_stdout()

    print("=" * 82)
    print("D-3 — 지도 L + 깊이 D + 회복력 Δ 를 **전부** 결합")
    print("=" * 82)
    dmid, dw, hvec = load_kernels()
    bk = build_book(a.band, [a.hor])
    bk = bk.dropna(subset=["sig", "xd%d" % a.hor])
    bk = bk[(bk["sig"] > 0) & (bk["D_bid"] > 0) & (bk["D_ask"] > 0)]
    t0, t1 = int(bk["sec"].min() * 1000), int(bk["sec"].max() * 1000)
    print("웹소켓 창 %s ~ %s (%.1f시간) / %d종"
          % (str(pd.Timestamp(t0, unit="ms"))[:19], str(pd.Timestamp(t1, unit="ms"))[:19],
             (t1 - t0) / 3.6e6, bk["symbol"].nunique()))
    print("코호트 룩백 %d일 (반감기 10.5일 -> 사실상 전부 포함)\n" % LOOKBACK_D)

    fp = []
    for s in sorted(bk["symbol"].unique()):
        r = fuel_panel(s, t0, t1, dmid, dw, hvec)
        if r is not None:
            fp.append(r)
            print("  %-10s 연료격자 %5d" % (s, len(r)))
    if not fp:
        print("연료 없음")
        return 1
    F = pd.concat(fp, ignore_index=True).sort_values("t5")
    bk = bk.sort_values("sec")
    bk["t5"] = (bk["sec"] * 1000).astype(np.int64)
    j = pd.merge_asof(bk, F, on="t5", by="symbol", direction="backward")
    j = j.dropna(subset=["Ld0", "oi"])
    print("\n결합 %d초-관측 / %d종" % (len(j), j["symbol"].nunique()))

    parts = []
    for lab, xc, Dc, Ac, Rc, Lp in (
            ("down", "xd%d" % a.hor, "D_bid", "AddB", "RemB", "Ld"),
            ("up", "xu%d" % a.hor, "D_ask", "AddA", "RemA", "Lu")):
        q = pd.DataFrame({
            "X": j[xc].to_numpy(), "D": j[Dc].to_numpy(),
            "Add": j[Ac].to_numpy(), "sig": j["sig"].to_numpy(),
            "L": j[["%s%d" % (Lp, i) for i in range(len(BANDS))]].sum(axis=1).to_numpy(),
            "L0": j["%s0" % Lp].to_numpy(),
            "sec": j["sec"].to_numpy(), "symbol": j["symbol"].to_numpy()})
        parts.append(q)
    p = pd.concat(parts, ignore_index=True)
    p = p[np.isfinite(p["X"]) & (p["X"] > 0) & (p["L"] > 0) & (p["D"] > 0)
          & np.isfinite(p["Add"])].reset_index(drop=True)
    p["reg"] = p["Add"] / np.maximum(p["D"], 1e-9)
    print("X 중앙 %.2f bp | L 중앙 $%.4g | D 중앙 $%.4g | **L/D 중앙 %.3f**"
          % (p.X.median(), p.L.median(), p.D.median(), (p.L / p.D).median()))

    y = np.log(p["X"].to_numpy())
    ls = np.log(p["sig"].to_numpy())
    lD = np.log(p["D"].to_numpy())
    lL = np.log(p["L"].to_numpy())
    lL0 = np.log(np.maximum(p["L0"].to_numpy(), 1.0))
    lg = np.log(np.maximum(p["reg"].to_numpy(), 1e-9))
    lr = lL - lD
    hr = (p["sec"].to_numpy() // 3600)
    ok = (np.isfinite(y) & np.isfinite(ls) & np.isfinite(lD) & np.isfinite(lL)
          & np.isfinite(lg) & np.isfinite(lL0))

    print("\n" + "-" * 82)
    print("1. 지도가 깊이 너머의 정보를 갖는가 — b4(지도) 가 **양수**여야 한다")
    print("-" * 82)
    print("  %-34s %8s %6s | %8s %6s | %7s"
          % ("설정", "b(깊이)", "t", "**b(지도)**", "t", "R^2"))
    specs = [
        ("sigma", [ls], -1),
        ("+ D + 회복력", [ls, lD, lg], 1),
        ("+ D + 회복력 + **L 전체**", [ls, lD, lg, lL], 3),
        ("+ D + 회복력 + **L 근거리(0~2%)**", [ls, lD, lg, lL0], 3),
        ("**비율형 log(L/D)** (설계식)", [ls, lg, lr], 2),
    ]
    base_r2 = None
    for lab, cols, li in specs:
        X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
        b, se, _ = ols_cluster(X, y[ok], hr[ok])
        r2 = 1.0 - np.var(y[ok] - X @ b) / np.var(y[ok])
        if lab.startswith("+ D + 회복력") and base_r2 is None:
            base_r2 = r2
        cd = "%8.4f %6.1f" % (b[2], b[2] / se[2]) if len(b) > 2 else "%8s %6s" % ("-", "-")
        cl = ("%8.4f %6.1f" % (b[li], b[li] / se[li])) if li > 0 and li < len(b) \
            else "%8s %6s" % ("-", "-")
        inc = "" if base_r2 is None else "  (증분 %+.4f)" % (r2 - base_r2)
        print("  %-34s %s | %s | %7.4f%s" % (lab, cd, cl, r2, inc))
    print("\n  b(지도)>0 이고 증분 R^2 가 양수면 **설계의 분자가 값어치가 있다**.")
    print("  사전등록 기준(plan-20260804): 증분 **>= 0.02**")

    print("\n" + "-" * 82)
    print("2. 확률모형 캘리브레이션 — 지도를 넣으면 나아지는가")
    print("-" * 82)
    cut = int(len(p) * a.train)
    tr = np.zeros(len(p), dtype=bool)
    tr[:cut] = True
    mods = {
        "M0 무조건부": None,
        "M1 sigma+D+회복": np.column_stack([np.ones(len(p)), ls, lD, lg]),
        "M2 +지도": np.column_stack([np.ones(len(p)), ls, lD, lg, lL]),
    }
    print("  %-16s | %-22s | %8s | %8s" % ("모형", "위반율 편차(평균)", "PIT D", "핀볼"))
    for lab, Xf in mods.items():
        if Xf is None:
            m_te = np.zeros(int((~tr & ok).sum()))
            Z = y[tr & ok]
        else:
            bt = np.linalg.pinv(Xf[tr & ok].T @ Xf[tr & ok]) @ (Xf[tr & ok].T @ y[tr & ok])
            m_te = Xf[~tr & ok] @ bt
            Z = y[tr & ok] - Xf[tr & ok] @ bt
        xte = p["X"].to_numpy()[~tr & ok]
        devs, tot = [], 0.0
        for lv in LEVELS:
            qq = np.exp(m_te + float(np.quantile(Z, lv)))
            devs.append(abs(float((xte < qq).mean()) - lv))
            tot += pinball(xte, qq, lv)
        Zs = np.sort(Z)
        u = np.searchsorted(Zs, np.log(xte) - m_te, side="right") / float(len(Zs))
        ks, pv = ks_unif(np.clip(u, 1e-6, 1 - 1e-6))
        print("  %-16s | %22.4f | %8.4f | %8.2f"
              % (lab, float(np.mean(devs)), ks, tot))
    # ---- D-9: 캐스케이드 조건부 분리 (2026-08-18 추가) ----
    if a.casc_csv:
        seg = pd.read_csv(a.casc_csv, parse_dates=["start"])
        seg = seg[seg["def"] == "15m<=-2%"].reset_index(drop=True)
        s0 = (seg["start"].astype("int64") // 10**9).to_numpy()
        s1 = s0 + (seg["dur_min"].to_numpy() * 60).astype("int64") + 900
        sy = seg["sym"].to_numpy()
        sec_a = p["sec"].to_numpy()
        sym_a = p["symbol"].to_numpy()
        casc = np.full(len(p), -1, dtype=np.int64)
        for k in range(len(seg)):
            mm = (sym_a == sy[k]) & (sec_a >= s0[k]) & (sec_a <= s1[k])
            casc[mm] = k
        n_in = int(((casc >= 0) & ok).sum())
        print("\n" + "-" * 82)
        print("3. D-9 — 캐스케이드 조건부 분리 (세그먼트 %d건, in-casc %d초-관측)"
              % (len(seg), n_in))
        print("-" * 82)
        for nm, mask, grp in (("평시", casc < 0, hr), ("캐스케이드", casc >= 0, casc)):
            mo = mask & ok
            if int(mo.sum()) < 500:
                print("  [%s] 표본 부족 (%d)" % (nm, int(mo.sum())))
                continue
            print("  [%s] n=%d, 클러스터 %d" % (nm, int(mo.sum()), len(np.unique(grp[mo]))))
            for lab, cols, iD, iL in (("+D+회복+L전체", [ls, lD, lg, lL], 2, 4),
                                      ("비율형 log(L/D)", [ls, lg, lr], -1, 3)):
                X = np.column_stack([np.ones(int(mo.sum()))] + [c[mo] for c in cols])
                b, se, _ = ols_cluster(X, y[mo], grp[mo])
                cd = ("b(깊이)=%+.4f(t=%+.1f) " % (b[iD], b[iD] / se[iD])) if iD > 0 else ""
                print("    %-16s %sb(지도)=%+.4f (t=%+.1f)"
                      % (lab, cd, b[iL], b[iL] / se[iL]))
        print("\n  *** 창 %.1f시간 / 캐스케이드 세그먼트 %d건 — 조건부 판정 표본."
              % ((t1 - t0) / 3.6e6, len(seg)))
    else:
        print("\n  *** 캐스케이드 조건부는 --casc-csv 로 실행 (미지정 시 전체 표본 판정만).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
