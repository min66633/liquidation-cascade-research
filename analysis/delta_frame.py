# -*- coding: utf-8 -*-
"""Δ(v,t) 재측정 — 밴드 경계가 움직여서 생긴 값과 진짜 유입·취소를 분리한다.

무엇이 틀렸나 (2026-08-05 발견)
  depth_ws 의 bands() 는 **현재 mid 기준** 누적 명목가를 낸다.
      B_t(v) = sum{ bn : p >= m_t (1-v) }          (호가 전 깊이, 상한 없음)
  depth_ws_flow 의 dbid_b 는 그 차분이고 D-3a(book_prob.py) 는 그것을 Δ 로 썼다.

      B_t(v) - B_{t-1}(v)  =  [ Bhat_t(v) - B_{t-1}(v) ]  +  [ B_t(v) - Bhat_t(v) ]
                              -----------------------      ---------------------
                              호가창이 **얼어 있어도** 생김      진짜 유입·취소
                              (mid 가 움직여 경계가 이동)

  Bhat_t(v) = 시각 t-1 의 호가창을 그대로 두고 **시각 t 의 창틀**로 다시 잰 값.
  캐스케이드에서는 mid 가 빠르게 움직이므로 첫 항이 커진다. b0_5(50bp) 밴드에서
  초당 1bp 움직이면 경계가 밴드폭의 2% 만큼 이동한다. b0_05(5bp) 면 20% 다.

  D-3a 에서 취소압이 t=-1.3 에 부호까지 예상과 반대로 나왔다. 그것이 실체가
  없어서인지 이 인공물에 덮인 것인지 — 여기서 가른다.

어떻게 되돌리나 (사다리 없이)
  B_{t-1}(v) 를 12개 밴드 지점에서 알고 있다. 이것은 곧 **가격의 함수**
      G_{t-1}(p) = sum{ bn_{t-1} : p' >= p },   B_{t-1}(v_k) = G_{t-1}(m_{t-1}(1-v_k))
  를 12점에서 아는 것과 같다. v=0 에서 G=0 이다(mid 위에는 매수호가가 없다).
  시각 t 의 창틀로 다시 재려면 같은 곡선을 새 문턱에서 읽으면 된다:

      bid:  v' = 1 - m_t (1-v) / m_{t-1}
      ask:  v' = m_t (1+v) / m_{t-1} - 1
      Bhat_t(v) = G_{t-1}(v')   (13점 선형보간, [0, v_max] 로 절단)

  절단이 걸리는 비율을 같이 보고한다 — 걸리면 그만큼 과소추정이다.

*** hftbacktest 의 DiffOrderBookSnapshot 은 쓰지 않았다. ***
  그것은 고정 N레벨 스냅샷 피드용이라 OUT_OF_BOOK_DELETION 범주가 핵심인데,
  우리 depth_ws 는 바이낸스 diff 로 전 깊이 로컬 북을 유지하므로 관측창이 없다.
  구현도 for prev_lv: for curr_lv: 이중 루프라 2000레벨에는 못 쓴다.
  분류 발상(무엇이 진짜 추가·취소인가를 창틀과 분리해 센다)만 가져왔다.

실행:
    python analysis/delta_frame.py
    python analysis/delta_frame.py --band b0_5 --hor 60
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
from analysis.response_liq import ols_cluster                    # noqa: E402

# depth_ws.py 의 BANDS 와 **반드시** 같아야 한다. 바뀌면 여기도 바꾼다.
BANDS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075,
         0.01, 0.015, 0.02, 0.03, 0.05, 0.10]
BAND_NAMES = ["b%s" % ("%g" % (b * 100)).replace(".", "_") for b in BANDS]
VK0 = np.concatenate([[0.0], np.array(BANDS, dtype=np.float64)])   # 가상 매듭 v=0, G=0
VMAX = BANDS[-1]
ROLL = 60                     # book_prob.py 와 같은 총유입/총취소 누적창(초)


def load_book() -> pd.DataFrame:
    """depth_ws 북 스냅샷 전량. 밴드 24개 + mid + resyncs."""
    fs = sorted(glob.glob(os.path.join(C.DATA, "depth_ws", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("depth_ws 가 비어 있다")
    cols = (["ts_ms", "symbol", "mid", "resyncs"]
            + ["bid_" + n for n in BAND_NAMES] + ["ask_" + n for n in BAND_NAMES])
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    d["sec"] = d["ts_ms"] // 1000
    d = d.drop_duplicates(["symbol", "sec"], keep="last")
    return d.sort_values(["symbol", "sec"]).reset_index(drop=True)


def _interp_rowwise(Bprev: np.ndarray, vq: np.ndarray):
    """행마다 다른 매듭값 위에서의 선형보간.

    Bprev : (n, K)   시각 t-1 의 밴드 누적액 (v_1..v_K, 단조증가)
    vq    : (n, K)   질의점 (시각 t 의 창틀을 t-1 좌표로 옮긴 것)
    반환  : (hat (n,K), 절단마스크 (n,K))

    np.interp 는 매듭값이 행마다 다르면 못 쓴다. 매듭 **위치** VK0 는 고정이므로
    searchsorted 로 구간을 찾고 take_along_axis 로 값을 모으면 전량 벡터화된다.
    """
    n = Bprev.shape[0]
    # 단조성 보정 — 누적액이므로 원래 단조여야 하지만 부동소수 잡음을 막는다
    B0 = np.concatenate([np.zeros((n, 1)), np.maximum.accumulate(Bprev, axis=1)], axis=1)
    clipped = (vq < 0.0) | (vq > VMAX)
    q = np.clip(vq, 0.0, VMAX)
    idx = np.clip(np.searchsorted(VK0, q, side="left"), 1, len(VK0) - 1)
    lo, hi = VK0[idx - 1], VK0[idx]
    ylo = np.take_along_axis(B0, idx - 1, axis=1)
    yhi = np.take_along_axis(B0, idx, axis=1)
    w = np.where(hi > lo, (q - lo) / np.maximum(hi - lo, 1e-18), 0.0)
    return ylo + w * (yhi - ylo), clipped


class Acc:
    """밴드 x 사이드 진단 누적기. 배열을 들고 있지 않는다 (전량 보관은 2.7GB)."""

    def __init__(self):
        z = np.zeros((len(BANDS), 2))
        self.n = z.copy()
        self.s_at = z.copy()     # sum |tot|
        self.s_af = z.copy()     # sum |frm|
        self.s_t = z.copy()
        self.s_tt = z.copy()
        self.s_f = z.copy()
        self.s_ff = z.copy()
        self.s_d = z.copy()
        self.s_dd = z.copy()
        self.s_td = z.copy()
        self.s_fd = z.copy()
        self.clip = z.copy()

    def add(self, k: int, side: int, tot, frm, dlm, clipmask) -> None:
        self.n[k, side] += len(tot)
        self.s_at[k, side] += np.abs(tot).sum()
        self.s_af[k, side] += np.abs(frm).sum()
        self.s_t[k, side] += tot.sum()
        self.s_tt[k, side] += (tot * tot).sum()
        self.s_f[k, side] += frm.sum()
        self.s_ff[k, side] += (frm * frm).sum()
        self.s_d[k, side] += dlm.sum()
        self.s_dd[k, side] += (dlm * dlm).sum()
        self.s_td[k, side] += (tot * dlm).sum()
        self.s_fd[k, side] += (frm * dlm).sum()
        self.clip[k, side] += clipmask.sum()

    def _corr(self, sxy, sx, sy, sxx, syy, n):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = max(sxx / n - (sx / n) ** 2, 0.0)
        vy = max(syy / n - (sy / n) ** 2, 0.0)
        return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan

    def report(self) -> None:
        print("\n" + "-" * 92)
        print("1. 진단 — 밴드 차분에서 **프레임 이동**이 차지하는 비중")
        print("-" * 92)
        print("  총차분 = 프레임 + 진짜흐름.  프레임은 호가창이 얼어 있어도 생기는 값이다.")
        print("  Σ|프레임|/Σ|총| 이 1 에 가까우면 D-3a 의 Δ 는 사실상 mid 움직임을 잰 것이다.\n")
        print("  %-8s | %-32s | %-32s" % ("밴드", "매수(아래쪽 깊이)", "매도(위쪽 깊이)"))
        print("  %-8s | %9s %8s %11s | %9s %8s %11s"
              % ("", "Σ|프|/Σ|총|", "Var비", "corr(총,dm)",
                 "Σ|프|/Σ|총|", "Var비", "corr(총,dm)"))
        for k, nm in enumerate(BAND_NAMES):
            cells = []
            for sd in (0, 1):
                n = self.n[k, sd]
                if n < 100:
                    cells.append("%9s %8s %11s" % ("-", "-", "-"))
                    continue
                ratio = self.s_af[k, sd] / max(self.s_at[k, sd], 1e-12)
                vt = max(self.s_tt[k, sd] / n - (self.s_t[k, sd] / n) ** 2, 0.0)
                vf = max(self.s_ff[k, sd] / n - (self.s_f[k, sd] / n) ** 2, 0.0)
                vr = vf / vt if vt > 0 else np.nan
                cc = self._corr(self.s_td[k, sd], self.s_t[k, sd], self.s_d[k, sd],
                                self.s_tt[k, sd], self.s_dd[k, sd], n)
                cells.append("%9.3f %8.3f %11.3f" % (ratio, vr, cc))
            print("  %-8s | %s | %s" % (nm, cells[0], cells[1]))
        tot_n = self.n.sum(axis=0)
        print("\n  보간 절단비율: 매수 %.5f 매도 %.5f  (v' 가 [0,%.2f] 밖 — 클수록 과소추정)"
              % (self.clip[:, 0].sum() / max(tot_n[0], 1),
                 self.clip[:, 1].sum() / max(tot_n[1], 1), VMAX))
        print("  corr(총,dm): mid 가 오르면 매수문턱 m(1-v) 도 올라 매수밴드는 **줄어야** 한다.")
        print("               매수 음수 / 매도 양수 면 인공물이 지배한다는 신호다.")


def process(d: pd.DataFrame, band: str):
    """심볼별 분해. 진단은 누적기로, 회귀용 패널은 **요청 밴드만** 보관한다."""
    bcols = ["bid_" + n for n in BAND_NAMES]
    acols = ["ask_" + n for n in BAND_NAMES]
    kb = BAND_NAMES.index(band)
    acc = Acc()
    rows = []
    for s, g in d.groupby("symbol", sort=False):
        g = g.sort_values("sec")
        sec = g["sec"].to_numpy()
        mid = g["mid"].to_numpy(dtype=np.float64)
        rsy = g["resyncs"].to_numpy()
        B = g[bcols].to_numpy(dtype=np.float64)
        A = g[acols].to_numpy(dtype=np.float64)
        n = len(sec)
        if n < 300:
            continue
        # 유효한 연속쌍: 1초 간격 & 재동기화 없음 & 값 정상
        ok = np.zeros(n, dtype=bool)
        ok[1:] = ((np.diff(sec) == 1) & (np.diff(rsy) == 0)
                  & np.isfinite(mid[1:]) & np.isfinite(mid[:-1])
                  & (mid[1:] > 0) & (mid[:-1] > 0))
        ok[1:] &= (np.isfinite(B[1:]).all(1) & np.isfinite(B[:-1]).all(1)
                   & np.isfinite(A[1:]).all(1) & np.isfinite(A[:-1]).all(1))
        if ok.sum() < 200:
            continue
        i = np.flatnonzero(ok)
        j = i - 1
        r = mid[i] / mid[j]
        vb = np.array(BANDS)[None, :]
        vq_b = 1.0 - r[:, None] * (1.0 - vb)      # bid 문턱 m_t(1-v) 를 t-1 좌표로
        vq_a = r[:, None] * (1.0 + vb) - 1.0      # ask 문턱 m_t(1+v)
        hatB, cb = _interp_rowwise(B[j], vq_b)
        hatA, ca = _interp_rowwise(A[j], vq_a)
        dlm = np.log(mid[i]) - np.log(mid[j])
        for k in range(len(BANDS)):
            acc.add(k, 0, B[i, k] - B[j, k], hatB[:, k] - B[j, k], dlm, cb[:, k])
            acc.add(k, 1, A[i, k] - A[j, k], hatA[:, k] - A[j, k], dlm, ca[:, k])
        rows.append(pd.DataFrame({
            "symbol": s, "sec": sec[i], "mid": mid[i],
            "Dbid": B[i, kb], "Dask": A[i, kb],
            "rawb": B[i, kb] - B[j, kb], "trub": B[i, kb] - hatB[:, kb],
            "rawa": A[i, kb] - A[j, kb], "trua": A[i, kb] - hatA[:, kb]}))
    if not rows:
        return acc, None
    return acc, pd.concat(rows, ignore_index=True)


def build_panel(f: pd.DataFrame, hor: int) -> pd.DataFrame:
    """(심볼 x 방향) 안에서 전방 밀림 X, 변동성, 60초 총유입/총취소를 만든다.

    *** 롤링과 전방창은 반드시 심볼 안에서 돌려야 한다. ***
    심볼을 섞은 뒤 rolling 을 걸면 남의 호가로 내 회복력을 계산하게 된다.
    """
    parts = []
    for s, g in f.groupby("symbol", sort=False):
        g = g.sort_values("sec")
        sec = g["sec"].to_numpy()
        mid = g["mid"].to_numpy(dtype=np.float64)
        n = len(mid)
        if n < hor + 200:
            continue
        lm = np.log(np.maximum(mid, 1e-12))
        sig = pd.Series(np.concatenate([[np.nan], np.diff(lm)])).rolling(
            120, min_periods=60).std().to_numpy()
        rv = pd.Series(mid[::-1])
        fmin = rv.rolling(hor + 1, min_periods=1).min().to_numpy()[::-1]
        fmax = rv.rolling(hor + 1, min_periods=1).max().to_numpy()[::-1]
        # 전방 hor 초가 1초 간격으로 이어진 구간만 유효
        cont = np.zeros(n, dtype=bool)
        cont[:n - hor] = (sec[hor:] - sec[:n - hor]) == hor
        xd = np.where(cont, (mid - fmin) / mid * 1e4, np.nan)
        xu = np.where(cont, (fmax - mid) / mid * 1e4, np.nan)
        # 압력받는 쪽 깊이를 분모로: 아래로 밀리면 매수호가, 위로 밀리면 매도호가
        for lab, X_, Dc, rc, tc in (("down", xd, "Dbid", "rawb", "trub"),
                                    ("up", xu, "Dask", "rawa", "trua")):
            D = g[Dc].to_numpy(dtype=np.float64)
            rec = {"symbol": s, "side": lab, "sec": sec, "X": X_, "D": D, "sig": sig}
            for col, src in (("raw", rc), ("cor", tc)):
                v = g[src].to_numpy(dtype=np.float64)
                sv = pd.Series(v)
                rec["add_" + col] = sv.clip(lower=0).rolling(
                    ROLL, min_periods=10).sum().to_numpy()
                rec["rem_" + col] = (-sv).clip(lower=0).rolling(
                    ROLL, min_periods=10).sum().to_numpy()
                rec["v_" + col] = v
            parts.append(pd.DataFrame(rec))
    return pd.concat(parts, ignore_index=True) if parts else None


def regress(p: pd.DataFrame, band: str, hor: int) -> None:
    print("\n" + "-" * 92)
    print("2. 재측정 — 프레임 보정 전/후로 회복력·취소압이 바뀌는가 (밴드 %s, 지평 %ds)"
          % (band, hor))
    print("-" * 92)
    p = p[np.isfinite(p["X"]) & (p["X"] > 0) & (p["sig"] > 0) & (p["D"] > 0)].copy()
    p = p.sort_values("sec").reset_index(drop=True)
    if len(p) < 5000:
        print("  표본 부족 (%d)" % len(p))
        return
    print("  표본 %d 초-관측 | X 중앙 %.2f bp | D 중앙 $%.4g\n"
          % (len(p), p.X.median(), p.D.median()))
    y = np.log(p["X"].to_numpy())
    ls = np.log(p["sig"].to_numpy())
    D = np.maximum(p["D"].to_numpy(), 1e-9)
    lD = np.log(D)
    hr = p["sec"].to_numpy() // 3600
    print("  %-24s %9s %6s | %10s %6s | %11s %6s | %7s"
          % ("Δ 정의", "b(깊이)", "t", "회복력", "t", "**취소압**", "t", "R^2"))
    diag = {}
    for lab, col in (("원본(총차분)", "raw"), ("**프레임 보정**", "cor")):
        lg = np.log(np.maximum(p["add_" + col].to_numpy() / D, 1e-9))
        lc = np.log(np.maximum(p["rem_" + col].to_numpy() / D, 1e-9))
        ok = (np.isfinite(y) & np.isfinite(ls) & np.isfinite(lD)
              & np.isfinite(lg) & np.isfinite(lc))
        if ok.sum() < 1000:
            print("  %-26s 표본부족 %d" % (lab, ok.sum()))
            continue
        X = np.column_stack([np.ones(int(ok.sum())), ls[ok], lD[ok], lg[ok], lc[ok]])
        b, se, _ = ols_cluster(X, y[ok], hr[ok])
        r2 = 1.0 - np.var(y[ok] - X @ b) / np.var(y[ok])
        diag[col] = (lg, lc, ok)
        print("  %-26s %9.4f %6.1f | %10.4f %6.1f | %11.4f %6.1f | %7.4f"
              % (lab, b[2], b[2] / se[2], b[3], b[3] / se[3],
                 b[4], b[4] / se[4], r2))
    print("  회복력 음수 = 유입 많으면 덜 밀림 | 취소압 **양수** = 유령 유동성")

    # *** 식별 점검 — 유입과 취소는 둘 다 '활동량'에 비례한다 ***
    # 상관이 1 에 가까우면 두 계수는 공통 활동량을 임의로 쪼갠 값이라 못 믿는다.
    # 그래서 (합=활동량, 차=순방향) 으로 직교화해 다시 본다.
    print("\n  [식별 점검] 유입·취소는 둘 다 활동량에 비례한다 — 쪼개기가 되는가")
    for col, lab in (("raw", "원본"), ("cor", "보정")):
        if col not in diag:
            continue
        lg, lc, ok = diag[col]
        r = float(np.corrcoef(lg[ok], lc[ok])[0, 1])
        # 활동량 = (유입+취소)/2, 순방향 = 유입-취소. 서로 거의 직교한다.
        act, net = 0.5 * (lg + lc), lg - lc
        X = np.column_stack([np.ones(int(ok.sum())), ls[ok], lD[ok], act[ok], net[ok]])
        b, se, _ = ols_cluster(X, y[ok], hr[ok])
        # VIF: 회귀항끼리의 다중공선성
        Z = np.column_stack([ls[ok], lD[ok], lg[ok], lc[ok]])
        vif = []
        for c in range(Z.shape[1]):
            W = np.column_stack([np.ones(len(Z)), np.delete(Z, c, axis=1)])
            bb = np.linalg.pinv(W.T @ W) @ (W.T @ Z[:, c])
            rr = 1.0 - np.var(Z[:, c] - W @ bb) / max(np.var(Z[:, c]), 1e-18)
            vif.append(1.0 / max(1.0 - rr, 1e-9))
        print("    %s corr(유입,취소)=%+.3f | VIF[sig,D,유입,취소]=%s"
              % (lab, r, " ".join("%.1f" % v for v in vif)))
        print("        직교화: 활동량 %+.4f (t=%.1f) | **순방향(유입-취소) %+.4f (t=%.1f)**"
              % (b[3], b[3] / se[3], b[4], b[4] / se[4]))
    print("    순방향 계수가 음수·유의면 '유입이 취소를 이길수록 덜 밀린다' — Δ 의 핵심 주장이다.")

    print("\n  흐름 자체의 크기:")
    for col, lab in (("raw", "원본"), ("cor", "보정")):
        v = p["v_" + col].to_numpy()
        m = np.isfinite(v)
        print("    %s |Δ|/D 중앙 %.6f | 유입비중 %.3f | Δ 중앙 %+.4g"
              % (lab, float(np.median(np.abs(v[m]) / D[m])),
                 float((v[m] > 0).mean()), float(np.median(v[m]))))


def main() -> int:
    ap = argparse.ArgumentParser(description="frame-shift correction for Delta(v,t)")
    ap.add_argument("--bands", nargs="+", default=["b0_1", "b0_2", "b0_5", "b2"],
                    choices=BAND_NAMES,
                    help="밴드마다 재측정한다. 좁은 밴드일수록 프레임 인공물이 크다")
    ap.add_argument("--hor", type=int, default=60)
    a = ap.parse_args()
    U.init_stdout()

    print("=" * 92)
    print("Δ(v,t) 프레임 보정 — 밴드 경계 이동과 진짜 유입·취소를 분리")
    print("=" * 92)
    d = load_book()
    print("**사용 데이터 기간: %s ~ %s / %d종 / %d 초-관측 (depth_ws 1초)**"
          % (str(pd.Timestamp(int(d.ts_ms.min()), unit="ms"))[:19],
             str(pd.Timestamp(int(d.ts_ms.max()), unit="ms"))[:19],
             d.symbol.nunique(), len(d)))
    # 진단은 밴드에 무관하게 12개 전부를 한 번에 낸다 (첫 밴드 처리에 얹어서)
    first = True
    for band in a.bands:
        acc, f = process(d, band)
        if f is None or len(f) < 5000:
            print("[%s] 연속쌍 부족" % band)
            continue
        if first:
            print("연속 1초쌍 %d개" % len(f))
            acc.report()
            first = False
        p = build_panel(f, a.hor)
        del f
        if p is None:
            print("[%s] 패널 생성 실패" % band)
            continue
        regress(p, band, a.hor)
        del p
    print("\n*** 웹소켓 3일(2026-08-02~04). 캐스케이드 0건 — 함수형만 검정된다. ***")
    print("*** 좁은 밴드(b0_05/b0_1)는 차분의 60~70%가 프레임 인공물이다. 보정 없이 쓰면 안 된다. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
