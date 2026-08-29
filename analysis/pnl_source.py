# -*- coding: utf-8 -*-
"""돈이 어디서 나오는가 — 반등인가, 밀림의 연장인가, 몇 건의 대박인가.

왜 이것을 따로 재는가
  backtest.py 는 48.3bp / t=2.8 만 낸다. 그 숫자만으로는
    (가) 캐스케이드가 끝난 뒤 **되돌림**을 먹는 것인지
    (나) 진입 직후 계속 밀리다가 우연히 회복한 것인지
    (다) 몇 건의 극단 사건이 전부를 만든 것인지
  를 구분할 수 없다. 셋은 운용상 완전히 다른 물건이다.
    (가) 면 바닥 예측이 핵심이고
    (나) 면 진입 타이밍이 핵심이고
    (다) 면 그냥 못 쓴다 (표본 밖에서 재현될 이유가 없다).

무엇을 재는가 (진입 후 1분봉 경로. hold 분 동안)
  MAE  최대 역행 (진입가 대비 최저, 롱이면 저가)
  MFE  최대 순행
  t_MAE  최저점이 몇 분째에 오는가   <- **경로 모양을 가르는 축**
      t_MAE = 0        진입 직후가 바닥. 더 안 밀렸다        -> 즉시 반등
      0 < t_MAE < hold 밀렸다가 돌아섰다                     -> V자 되돌림
      t_MAE >= hold-1  끝까지 밀리고 있다                    -> 밀림 연장(실패)
  집중도  상위 몇 건이 총이익의 몇 %를 만드는가

실행:
    python analysis/pnl_source.py
    python analysis/pnl_source.py --hold 15
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
from analysis.limit_fill import attempts, _bars                 # noqa: E402
from analysis.response_liq import cmean                         # noqa: E402


def paths(att: pd.DataFrame, hold: int, cost: float) -> pd.DataFrame:
    """시장가 진입(기준선) 후 경로. 이벤트별 MAE/MFE/t_MAE/수익."""
    rows = []
    for s, g in att.groupby("symbol", sort=False):
        b = _bars(s)
        if b is None:
            continue
        ot1, O, H, L, Cl = b
        for r in g.itertuples():
            j, sd = int(r.j), int(r.side)
            p_in = O[j]
            e = j + hold
            # 롱이면 저가가 역행, 숏이면 고가가 역행
            adv = (L[j:e + 1] / p_in - 1.0) if sd == 1 else (p_in / H[j:e + 1] - 1.0)
            fav = (H[j:e + 1] / p_in - 1.0) if sd == 1 else (p_in / L[j:e + 1] - 1.0)
            adv = adv * 1e4
            fav = fav * 1e4
            k = int(np.argmin(adv))
            rows.append({"symbol": s, "t": r.t, "side": sd, "z": r.z, "sig5": r.sig5,
                         "day": r.day, "year": r.year,
                         "ret": (Cl[e] / p_in - 1.0) * sd * 1e4 - cost,
                         "mae": float(adv.min()), "mfe": float(fav.max()),
                         "t_mae": k})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="where does the PnL come from")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--cost", type=float, default=10.0)
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS

    print("=" * 92)
    print("돈이 어디서 나오는가 — 되돌림인가 / 밀림 연장인가 / 소수 대박인가")
    print("=" * 92)
    d = paths(attempts(syms, a.k, a.doi, a.gap, a.hold), a.hold, a.cost)
    if len(d) < 50:
        print("이벤트 부족 (%d)" % len(d))
        return 1
    tot = d["ret"].sum()
    m, _, t, _ = cmean(d["ret"].to_numpy(), d["day"].to_numpy())
    print("**사용 데이터 기간: %s ~ %s / %d종 / %d건 / 보유 %d분 / 비용 %.0fbp**"
          % (str(pd.Timestamp(int(d.t.min()), unit="ms"))[:10],
             str(pd.Timestamp(int(d.t.max()), unit="ms"))[:10],
             d.symbol.nunique(), len(d), a.hold, a.cost))
    print("전체: 평균 %.1f bp (t=%.1f) | 중앙 **%.1f bp** | 승률 %.1f%% | 합계 %.0f bp"
          % (m, t, d.ret.median(), 100 * (d.ret > 0).mean(), tot))

    print("\n" + "-" * 92)
    print("1. 경로 모양 — 되돌림을 먹는가, 안 밀린 걸 먹는가")
    print("-" * 92)
    print("  t_MAE = 최저점이 진입 후 몇 분째인가. 이것이 (가)와 (나)를 가른다.\n")
    hold = a.hold
    cat = pd.cut(d["t_mae"], [-0.5, 0.5, 2.5, hold - 1.5, hold + 0.5],
                 labels=["0분(즉시 바닥)", "1~2분", "3~%d분" % (hold - 2),
                         "%d~%d분(끝까지 밀림)" % (hold - 1, hold)])
    print("  %-22s %6s %7s | %9s %8s | %9s %9s | %8s"
          % ("최저점 시각", "n", "비중", "평균bp", "중앙bp", "MAE중앙", "MFE중앙", "이익비중"))
    for lab, g in d.groupby(cat, observed=True):
        print("  %-22s %6d %6.1f%% | %9.1f %8.1f | %9.1f %9.1f | %7.1f%%"
              % (lab, len(g), 100 * len(g) / len(d), g.ret.mean(), g.ret.median(),
                 g.mae.median(), g.mfe.median(), 100 * g.ret.sum() / tot))
    print("\n  '0분' 이 이익 대부분을 만들면 -> **진입 순간이 이미 바닥**. 되돌림 예측이 아니라")
    print("     '더 안 밀릴 사건 고르기' 가 핵심이다.")
    print("  '3~%d분' 이 만들면 -> **V자 되돌림을 먹는 것**. 바닥 예측이 값어치가 있다." % (hold - 2))

    print("\n" + "-" * 92)
    print("2. 얼마나 끌려가는가 — 진입 후 역행 분포")
    print("-" * 92)
    q = [0.10, 0.25, 0.50, 0.75, 0.90]
    print("  MAE(역행) 분위: " + " ".join("p%02d %.0f" % (100 * x, d.mae.quantile(x)) for x in q))
    print("  MFE(순행) 분위: " + " ".join("p%02d %.0f" % (100 * x, d.mfe.quantile(x)) for x in q))
    print("  MAE 중앙 %.1f bp — 절반의 거래가 이만큼 물린 뒤에야 회복한다." % d.mae.median())
    print("  ** 레버리지를 쓰면 이 값이 청산선을 건드리는지 반드시 확인해야 한다. **")

    print("\n" + "-" * 92)
    print("3. 집중도 — 몇 건이 전부를 만드는가")
    print("-" * 92)
    r = np.sort(d["ret"].to_numpy())[::-1]
    cum = np.cumsum(r)
    for kk in (1, 5, 10, 25, 50):
        if kk <= len(r):
            print("  상위 %3d건 (%.1f%%) 이 총이익의 **%.1f%%**"
                  % (kk, 100 * kk / len(r), 100 * cum[kk - 1] / tot))
    for cut in (0.01, 0.05):
        nn = max(int(len(r) * cut), 1)
        rest = d["ret"].to_numpy()
        thr = np.quantile(rest, 1 - cut)
        sub = rest[rest < thr]
        mm, _, tt, _ = cmean(sub, d["day"].to_numpy()[rest < thr])
        print("  상위 %.0f%% 제거 후: 평균 **%.1f bp** (t=%.1f) — %s"
              % (100 * cut, mm, tt, "살아있다" if tt > 2 else "**무너진다**"))

    print("\n" + "-" * 92)
    print("4. 심볼별 — 어디서 나오는가")
    print("-" * 92)
    print("  %-10s %6s %9s %9s %8s %9s" % ("심볼", "n", "평균bp", "중앙bp", "승률%", "합계bp"))
    gg = d.groupby("symbol").agg(n=("ret", "size"), mean=("ret", "mean"),
                                 med=("ret", "median"), win=("ret", lambda x: (x > 0).mean()),
                                 tot=("ret", "sum")).sort_values("tot", ascending=False)
    for s, r_ in gg.iterrows():
        print("  %-10s %6d %9.1f %9.1f %8.1f %9.0f"
              % (s, r_["n"], r_["mean"], r_["med"], 100 * r_["win"], r_["tot"]))
    print("\n  상위 3종 이익비중 %.1f%% / 21종"
          % (100 * gg["tot"].head(3).sum() / tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
