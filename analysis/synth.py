# -*- coding: utf-8 -*-
"""합성 확률모형 — 캐스케이드가 어디까지 밀릴지의 **조건부 분포**. 그리고 그 검정.

설계 (사용자 원안)
  현재 상태를 보고 캐스케이드가 바닥칠 가격대를 **확률적으로 근사**해서 그 자리에
  지정매수를 건다. 필요한 것은 점추정이 아니라 **분포** P(X >= u | F_t) 다.

부품 (R-1~R-6 에서 하나씩 검증한 것들)
  제곱근 임팩트 : X = Y * sigma * sqrt(Q/ADV).  Y=1.26 (R-4), 일반흐름 1.07 (R-1)
  증폭          : Q = A * S0,  A 는 R-4 가 OI 로 직접 측정 (평균 1.443)
  꼬리          : 디클러스터 후 xi ~ 0 -> 지수 꼬리 (R-2 4절)
  깊이(D)       : **없음.** ADV 로 대리한다. 웹소켓이 쌓이면 D_eff 로 교체.

  *** A 는 룩어헤드다(사건 이후 바를 쓴다). 그래서 **예측변수로 쓰지 않고**
      확률변수로만 둔다 — 그 변동은 잔차에 흡수된다. ***

모형
  로그를 취하면 관측 가능한 부분과 확률 부분이 분리된다:
      log X = m_t + Z,     m_t = log sigma_t + 0.5 * log(S0 / ADV_t)
  m_t 는 **진입 시점에 전부 관측된다.** Z 는 훈련구간 경험분포(= Y, A, 잡음의 합성).
  따라서
      P(X >= u | F_t) = P(Z >= log u - m_t)
  이고 분위수도 닫힌 형태로 나온다: q_p(F_t) = exp(m_t + z_p).

무엇으로 합격을 판정하나 — R^2 가 아니다
  (1) Kupiec 무조건부 커버리지 LR
  (2) **m_t 삼분위별 커버리지** — 조건부 모형의 진짜 시험. 여기서 갈린다.
  (3) PIT 균등성 (KS)
  (4) Christoffersen 독립성 (위반이 뭉치는가)
  (5) 핀볼 손실 — 적정 채점규칙. M0(무조건부)와 정면 비교.
  훈련/검정은 **시간 분할**. 과거 M2 실패의 원인 중 하나가 검정구간 선택이었다.

마지막으로 이 분포를 **실제 지정가**로 바꿔서 R-5 의 고정 깊이와 비교한다.

실행:
    python analysis/synth.py
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
from analysis.scale_check import K, DOI_THR, MIN_GAP, VOL_WIN   # noqa: E402

BULK1 = os.path.join(C.DATA, "binance_bulk", "klines_1m")
WINDOW = 240           # 이벤트 창(분)
FEE_M, FEE_T = 2.0, 5.0
LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def build(symbols, window):
    out = []
    for s in symbols:
        try:
            df = load(s)
        except FileNotFoundError:
            continue
        ev = find_events(df, K, DOI_THR, MIN_GAP)
        if len(ev) == 0:
            continue
        p1 = os.path.join(BULK1, "%s.parquet" % s)
        if not os.path.exists(p1):
            continue
        k = pd.read_parquet(p1, columns=["open_time", "open", "high", "low", "close"])
        k = k.sort_values("open_time").reset_index(drop=True)
        ot1 = k["open_time"].to_numpy()
        O, H, L, Cl = (k[c].to_numpy(dtype=np.float64)
                       for c in ("open", "high", "low", "close"))
        ot5 = df["open_time"].to_numpy()
        ret = df["ret"].to_numpy(dtype=np.float64)
        qv = df["quote_volume"].to_numpy(dtype=np.float64)
        oiv = df["sum_open_interest_value"].to_numpy(dtype=np.float64)
        doi = df["doi"].to_numpy(dtype=np.float64)
        sig = (pd.Series(ret).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .std().to_numpy()) * np.sqrt(float(VOL_WIN))
        adv = (pd.Series(qv).shift(1).rolling(VOL_WIN, min_periods=VOL_WIN // 4)
               .mean().to_numpy()) * float(VOL_WIN)
        n1 = len(ot1)
        for r in ev.itertuples():
            if not r.is_liq:
                continue
            i, sd = int(r.i), int(r.side)
            if i + 1 >= len(ot5):
                continue
            if not (np.isfinite(sig[i]) and sig[i] > 0
                    and np.isfinite(adv[i]) and adv[i] > 0
                    and np.isfinite(doi[i]) and doi[i] < 0 and oiv[i] > 0):
                continue
            t0 = int(ot5[i + 1])
            s0 = int(np.searchsorted(ot1, t0))
            s1 = s0 + window
            if s0 <= 0 or s1 >= n1:
                continue
            if ot1[s1 - 1] - ot1[s0] != (window - 1) * 60_000:
                continue
            p0 = O[s0]
            if not (np.isfinite(p0) and p0 > 0):
                continue
            # X = 진입 기준가에서 **얼마나 더 밀리는가** (bp, 양수)
            x = ((p0 - L[s0:s1].min()) / p0 if sd == 1
                 else (H[s0:s1].max() - p0) / p0) * 1e4
            if not np.isfinite(x) or x <= 0:
                continue
            # 창 구간만 잘라 담는다. 심볼 전체 1분봉을 들고 있으면 21종에 2GB 다.
            out.append({"symbol": s, "side": sd, "day": int(t0 // 86_400_000),
                        "t0": t0, "X": x,
                        "S0": -doi[i] * oiv[i], "sig": sig[i], "adv": adv[i],
                        "p0": p0,
                        "lo": L[s0:s1].copy(), "hi": H[s0:s1].copy(),
                        "cl": Cl[s0:s1].copy()})
    return pd.DataFrame(out), None


def kupiec(n, x, p):
    """무조건부 커버리지 LR. p = 기대 위반율."""
    if n == 0 or x == 0 or x == n:
        return np.nan
    ph = x / n
    return -2.0 * ((n - x) * np.log((1 - p) / (1 - ph)) + x * np.log(p / ph))


def christoffersen(v):
    """위반 계열의 독립성 LR (1차 마르코프)."""
    v = np.asarray(v, dtype=int)
    if len(v) < 20:
        return np.nan
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    if min(n00 + n01, n10 + n11) == 0:
        return np.nan
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0, 1) or p01 in (0, 1) or p11 in (0, 1):
        return np.nan
    ll1 = (n00 * np.log(1 - p01) + n01 * np.log(p01)
           + n10 * np.log(1 - p11) + n11 * np.log(p11))
    ll0 = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    return -2.0 * (ll0 - ll1)


def ks_unif(u):
    u = np.sort(np.asarray(u, dtype=float))
    n = len(u)
    if n < 20:
        return np.nan, np.nan
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    j = np.arange(1, 101)
    pv = 2.0 * np.sum(((-1.0) ** (j - 1)) * np.exp(-2.0 * (j ** 2) * lam ** 2))
    return d, float(min(1.0, max(0.0, pv)))


def pinball(y, q, p):
    e = y - q
    return float(np.mean(np.where(e >= 0, p * e, (p - 1.0) * e)))


def main() -> int:
    ap = argparse.ArgumentParser(description="synthesized conditional model")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--train", type=float, default=0.70)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 80)
    print("합성 확률모형 — log X = m_t + Z,  m_t = log sigma + 0.5 log(S0/ADV)")
    print("=" * 80)
    d, _ = build(syms, a.window)
    if len(d) == 0:
        print("표본 없음")
        return 1
    d = d.sort_values("t0").reset_index(drop=True)
    t0 = pd.to_datetime(d["t0"].min(), unit="ms")
    t1 = pd.to_datetime(d["t0"].max(), unit="ms")
    print("\n**사용 데이터 기간: %s ~ %s / %d종 / 이벤트 %d건 / 창 %d분**"
          % (str(t0)[:10], str(t1)[:10], d["symbol"].nunique(), len(d), a.window))
    print("X = 진입 기준가(open[i+1]) 에서 창 안 최대 역행. 중앙 %.0fbp p90 %.0fbp"
          % (d["X"].median(), d["X"].quantile(.9)))

    d["m"] = np.log(d["sig"]) + 0.5 * np.log(d["S0"] / d["adv"])
    d["lx"] = np.log(d["X"])
    cut = int(len(d) * a.train)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    print("\n훈련 %d건 (~%s) | 검정 %d건 (%s~)"
          % (len(tr), str(pd.to_datetime(tr["t0"].iloc[-1], unit="ms"))[:10],
             len(te), str(pd.to_datetime(te["t0"].iloc[0], unit="ms"))[:10]))

    # 지수 자유추정 — 제곱근(0.5)이 맞는가
    Xd = np.column_stack([np.ones(len(tr)), np.log(tr["sig"]),
                          np.log(tr["S0"] / tr["adv"])])
    bb = np.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ tr["lx"].to_numpy())
    print("\n지수 자유추정(훈련): log sigma 계수 %.3f | log(S0/ADV) 계수 **%.3f**"
          % (bb[1], bb[2]))
    print("  제곱근 법칙은 각각 1.0 / 0.5 를 예측한다.")

    # 잔차 경험분포 = Z. M0 는 이동 없음.
    z_tr = (tr["lx"] - tr["m"]).to_numpy()
    z0_tr = tr["lx"].to_numpy()

    print("\n" + "-" * 80)
    print("검정 1~4. 캘리브레이션 (검정구간 %d건)" % len(te))
    print("-" * 80)
    print("  M1 = 조건부(m_t 로 이동) | M0 = 무조건부(훈련 분위수 고정)")
    print("  %-4s %6s | %9s %8s %8s %9s | %9s %8s %8s"
          % ("모형", "수준p", "위반율", "Kupiec", "p값", "Chris p", "위반율", "Kupiec", "p값"))
    print("  %-4s %6s | %-38s | %-28s"
          % ("", "", "            M1 (조건부)", "        M0 (무조건부)"))
    from math import erf, exp
    def chi1p(s):
        return np.nan if not np.isfinite(s) else float(1.0 - erf(np.sqrt(max(s, 0) / 2.0)))
    for p in LEVELS:
        zq = float(np.quantile(z_tr, p))
        q1 = np.exp(te["m"].to_numpy() + zq)
        q0 = float(np.exp(np.quantile(z0_tr, p)))
        for tag, q in (("M1", q1), ("M0", np.full(len(te), q0))):
            v = (te["X"].to_numpy() < q).astype(int)   # 아래로 위반 = 예측보다 덜 밀림
            rate = v.mean()
            lr = kupiec(len(v), int(v.sum()), p)
            cs = christoffersen(v)
            if tag == "M1":
                row = "  %-4s %6.2f | %9.3f %8.2f %8.3f %9.3f |" % (
                    tag, p, rate, lr, chi1p(lr), chi1p(cs))
            else:
                print(row + " %9.3f %8.2f %8.3f" % (rate, lr, chi1p(lr)))
    print("  기대 위반율 = p. Kupiec p값이 0.05 미만이면 **불합격**.")

    print("\n" + "-" * 80)
    print("검정 2b. **m_t 삼분위별 커버리지** — 조건부 모형의 진짜 시험")
    print("-" * 80)
    print("  M1 이 맞다면 어느 삼분위에서도 위반율이 p 근처여야 한다.")
    print("  M0 는 m_t 를 무시하므로 삼분위 간에 위반율이 **기울어야** 한다.")
    ter = pd.qcut(te["m"], 3, labels=False, duplicates="drop")
    for p in (0.25, 0.50, 0.75):
        zq = float(np.quantile(z_tr, p))
        q1 = np.exp(te["m"].to_numpy() + zq)
        q0 = float(np.exp(np.quantile(z0_tr, p)))
        r1 = [float((te["X"].to_numpy()[ter == g] < q1[ter == g]).mean())
              for g in (0, 1, 2)]
        r0 = [float((te["X"].to_numpy()[ter == g] < q0).mean()) for g in (0, 1, 2)]
        print("  p=%.2f | M1 %5.3f %5.3f %5.3f (폭 %.3f) | M0 %5.3f %5.3f %5.3f (폭 %.3f)"
              % (p, *r1, max(r1) - min(r1), *r0, max(r0) - min(r0)))
    print("  **폭**이 작을수록 좋다. M1 폭 < M0 폭 이면 조건부가 실제로 작동한 것이다.")

    print("\n" + "-" * 80)
    print("검정 3. PIT 균등성 / 검정 5. 핀볼 손실 (적정 채점규칙)")
    print("-" * 80)
    u1 = np.array([float((z_tr <= v).mean()) for v in (te["lx"] - te["m"]).to_numpy()])
    u0 = np.array([float((z0_tr <= v).mean()) for v in te["lx"].to_numpy()])
    for tag, u in (("M1", u1), ("M0", u0)):
        ks, pv = ks_unif(np.clip(u, 1e-6, 1 - 1e-6))
        print("  %-3s PIT KS D=%.4f  p=%.4f  %s"
              % (tag, ks, pv, "합격" if pv > 0.05 else "**불합격**"))
    print("\n  %-6s %10s %10s %10s" % ("수준p", "M1 핀볼", "M0 핀볼", "개선%"))
    tot1 = tot0 = 0.0
    for p in LEVELS:
        q1 = np.exp(te["m"].to_numpy() + float(np.quantile(z_tr, p)))
        q0 = np.full(len(te), float(np.exp(np.quantile(z0_tr, p))))
        l1, l0 = pinball(te["X"].to_numpy(), q1, p), pinball(te["X"].to_numpy(), q0, p)
        tot1 += l1
        tot0 += l0
        print("  %-6.2f %10.1f %10.1f %10.1f" % (p, l1, l0, 100 * (l0 - l1) / l0))
    print("  %-6s %10.1f %10.1f %10.1f" % ("합계", tot1, tot0, 100 * (tot0 - tot1) / tot0))
    print("  핀볼 손실은 낮을수록 좋다. 개선%가 양수여야 조건부가 이긴 것이다.")

    print("\n" + "-" * 80)
    print("검정 6. 분포를 **지정가**로 — R-5 고정깊이와 정면 비교 (검정구간)")
    print("-" * 80)
    print("  모형 분위수 q_p 만큼 아래에 건다. 이벤트마다 깊이가 **다르다**.")
    print("  %-22s %8s %10s %7s %10s" % ("방식", "체결률", "이벤트당", "t", "조건부평균"))
    LOs = te["lo"].tolist()
    HIs = te["hi"].tolist()
    CLs = te["cl"].tolist()
    P0s = te["p0"].to_numpy()
    SDs = te["side"].to_numpy()

    def run(depth_bp):
        rs, fills = [], 0
        for idx in range(len(te)):
            sd_, p0_ = int(SDs[idx]), float(P0s[idx])
            lim = p0_ * (1.0 - sd_ * depth_bp[idx] / 1e4)
            hit = (LOs[idx] <= lim) if sd_ == 1 else (HIs[idx] >= lim)
            if not hit.any():
                rs.append(0.0)
                continue
            fb = int(np.argmax(hit))
            ex = min(fb + 15, len(CLs[idx]) - 1)
            rs.append((CLs[idx][ex] / lim - 1.0) * sd_ * 1e4 - FEE_M - FEE_T)
            fills += 1
        return np.array(rs), fills

    for p in (0.10, 0.25, 0.50):
        dep = np.exp(te["m"].to_numpy() + float(np.quantile(z_tr, p)))
        rs, f = run(dep)
        m, se, t, _ = cmean(rs, te["day"].to_numpy())
        cm = rs[rs != 0].mean() if f else np.nan
        print("  %-22s %7.1f%% %10.1f %7.1f %10.1f"
              % ("모형 q%.2f (중앙깊이%3.0fbp)" % (p, np.median(dep)),
                 100 * f / len(te), m, t, cm))
    for dep0 in (0.0, 25.0, 50.0, 100.0):
        rs, f = run(np.full(len(te), dep0))
        m, se, t, _ = cmean(rs, te["day"].to_numpy())
        cm = rs[rs != 0].mean() if f else np.nan
        print("  %-22s %7.1f%% %10.1f %7.1f %10.1f"
              % ("고정 %.0fbp" % dep0, 100 * f / len(te), m, t, cm))
    print("\n  모형 기반이 고정깊이를 이겨야 '분포를 아는 것' 이 값어치가 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
