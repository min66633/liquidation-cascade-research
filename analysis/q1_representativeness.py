# -*- coding: utf-8 -*-
"""Q1 — HL 청산맵 L(p) 가 시장 전체 강제흐름의 '위치'를 대표하는가. (문지기)

왜 이것이 문지기인가
  목표 설계(TARGET_DESIGN)는 가격대별 청산맵으로 도달점을 추정한다. 그런데 우리가 가진
  지도는 Hyperliquid 한 거래소의, 그것도 스윕으로 일부만 본 표본이다.
  phi ~ 0.05 (HL 글로벌 OI 15% x 스윕 커버리지 36%).
  **지도가 시장을 대표하지 못하면 아래 모든 것이 무의미하다.**

  그리고 오늘(2026-08-02) 돌린 대용품 검정들과 달리 **이것은 설계 그 자체**다.
  predict_x / x_dist / w_survival 은 전부 L(p) 가 없어서 대용품으로 한 것이고 실패했다.
  그 실패는 설계를 반증하지 않는다 — 대용품이 없다는 것만 보여준다.

무엇을 대조하는가
  지도   hl_positions 의 liquidation_px x position_value  -> 가격축 L(p)
  실현   bybit_liq 의 전건 청산 (가격 + 크기 + 방향)
  겹치는 구간 40시간 (2026-07-31 14:10 ~ 2026-08-02).

세 가지를 잰다
  (1) 커버리지  지도의 명목가 중 현재가 +-5% 안에 있는 비중. 격리/교차 분리.
                여기가 얇으면 '근접 연료' 자체가 지도에 없다는 뜻이다.
  (2) 방향성    시간당 지도의 상/하 불균형 vs 실현 청산의 롱/숏 불균형.
                지도가 '어느 쪽이 터질지' 를 맞히는가. 표본 = 심볼x시간.
  (3) 가격대별  실현 청산이 일어난 가격에서 지도가 두꺼웠는가.
                거리 통제가 필요하다 — 청산은 원래 현재가 근처에서 일어난다.

한계 (결과와 함께 반드시 보고)
  - 40시간, 롱청산 편중 구간. 레짐 일반화 불가.
  - 거래소 교차: HL 지도 vs Bybit 실현. 애초에 Q1 이 묻는 것이 바로 그 교차 타당성이다.
  - HL 포지션의 94% 가 교차마진(명목가 기준). 격리만 청산가가 가격에 고정된다.

실행:
    python analysis/q1_representativeness.py
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

BAND = 0.05                   # 근접 정의 +-5%
MAX_MAP_LAG_MS = 20 * 60_000  # 깊은 스윕이 15분 주기 -> 20분까지 허용
HOUR_MS = 3_600_000


def hl_symbol(s: str) -> str:
    """Bybit BTCUSDT -> HL BTC."""
    return s[:-4] if s.endswith("USDT") else s


def load_positions() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "hl_positions", "*", "*.parquet")))
    if not fs:
        raise FileNotFoundError("data/hl_positions 비어 있음")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[np.isfinite(d["liquidation_px"]) & (d["liquidation_px"] > 0)]
    d = d[np.isfinite(d["position_value"]) & (d["position_value"] > 0)]
    # szi > 0 이면 롱 -> 아래에서 청산(강제 매도). szi < 0 이면 숏 -> 위에서 청산.
    d["dir"] = np.where(d["szi"] > 0, "long", "short")
    return d


def load_mids() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "hl_mids", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[np.isfinite(d["mid_px"]) & (d["mid_px"] > 0)]
    return d.sort_values("ts").reset_index(drop=True)


def load_bybit() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(C.DATA, "bybit_liq", "*", "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["symbol"].isin(C.MAJORS)].copy()
    d["ntl"] = d["size"] * d["bankruptcy_px"]
    d = d[np.isfinite(d["ntl"]) & (d["ntl"] > 0)]
    d["coin"] = d["symbol"].map(hl_symbol)
    return d.sort_values("exch_ms").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Q1 representativeness of HL liquidation map")
    ap.add_argument("--band", type=float, default=BAND)
    a = ap.parse_args()
    U.init_stdout()

    pos, mids, byb = load_positions(), load_mids(), load_bybit()
    coins = sorted(set(byb["coin"]) & set(pos["coin"]))
    print("=" * 74)
    print("Q1 — HL 청산맵이 시장 강제흐름의 위치를 대표하는가  (문지기, 1차)")
    print("=" * 74)
    t0 = pd.to_datetime(byb.exch_ms.min(), unit="ms", utc=True)
    t1 = pd.to_datetime(byb.exch_ms.max(), unit="ms", utc=True)
    print("지도 스윕 %d개 / 포지션 %d행 | 실현청산 %d건 | 겹치는 심볼 %d"
          % (pos.sweep_id.nunique(), len(pos), len(byb), len(coins)))
    print("구간 %s ~ %s (%.1f시간)" % (t0, t1, (t1 - t0).total_seconds() / 3600))

    # 스윕별 mid (깊은 스윕 기준). 없으면 그 스윕의 코인은 건너뛴다.
    mid_map = mids.drop_duplicates(subset=["sweep_id", "coin"], keep="last")
    mid_map = mid_map.set_index(["sweep_id", "coin"])["mid_px"].to_dict()
    sweep_ts = pos.groupby("sweep_id")["ts"].min().to_dict()

    # ---------------------------------------------------------- (1) 커버리지
    print("\n--- 1. 커버리지: 지도 명목가 중 현재가 +-%.0f%% 안에 있는 비중 ---" % (100 * a.band))
    rows = []
    for (sid, coin), g in pos.groupby(["sweep_id", "coin"]):
        m = mid_map.get((sid, coin))
        if m is None or not np.isfinite(m) or m <= 0:
            continue
        u = (g["liquidation_px"] / m - 1.0).to_numpy()
        near = np.abs(u) <= a.band
        for lt in ("isolated", "cross"):
            sel = (g["lev_type"] == lt).to_numpy()
            tot = float(g["position_value"].to_numpy()[sel].sum())
            nr = float(g["position_value"].to_numpy()[sel & near].sum())
            rows.append({"coin": coin, "lev": lt, "tot": tot, "near": nr})
    cov = pd.DataFrame(rows)
    if cov.empty:
        print("  스윕-코인 mid 매칭 실패 — hl_mids 확인 필요")
        return 1
    g = cov.groupby("lev").agg(tot=("tot", "sum"), near=("near", "sum"))
    g["share_of_all"] = g["tot"] / g["tot"].sum()
    g["near_frac"] = g["near"] / g["tot"].replace(0, np.nan)
    print("  %-10s %14s %14s %12s %12s" % ("마진", "총명목가$", "근접$", "전체중비중", "근접비율"))
    for lt, r in g.iterrows():
        print("  %-10s %14.4g %14.4g %11.1f%% %11.1f%%"
              % (lt, r.tot, r.near, 100 * r.share_of_all, 100 * r.near_frac))
    tot_near = g["near"].sum()
    print("  => 근접(+-%.0f%%) 총 $%.4g 중 격리 $%.4g (%.1f%%)"
          % (100 * a.band, tot_near, g.loc["isolated", "near"],
             100 * g.loc["isolated", "near"] / max(tot_near, 1)))
    print("  격리만 청산가가 가격에 고정된다. 교차는 계좌 자기자본에 의존해 움직인다.")

    # ---------------------------------------------------------- (2) 방향성
    print("\n--- 2. 방향성: 지도의 상/하 불균형이 실현 청산의 방향을 맞히는가 ---")
    print("  표본 = 심볼 x 시간.  지도는 각 시간 직전 스윕.")
    sw = sorted(sweep_ts.items(), key=lambda kv: kv[1])
    sw_ids = np.array([k for k, _ in sw])
    sw_t = np.array([v for _, v in sw])

    byb["hour"] = byb["exch_ms"] // HOUR_MS
    recs = []
    for (coin, hr), gb in byb.groupby(["coin", "hour"]):
        t_end = int((hr + 1) * HOUR_MS)
        j = int(np.searchsorted(sw_t, t_end, side="right")) - 1
        if j < 0 or t_end - int(sw_t[j]) > MAX_MAP_LAG_MS:
            continue
        sid = sw_ids[j]
        m = mid_map.get((sid, coin))
        if m is None or not np.isfinite(m) or m <= 0:
            continue
        p = pos[(pos.sweep_id == sid) & (pos.coin == coin)]
        if p.empty:
            continue
        u = (p["liquidation_px"] / m - 1.0).to_numpy()
        pv = p["position_value"].to_numpy()
        isl = (p["lev_type"] == "isolated").to_numpy()
        # 롱 청산은 아래(u<0), 숏 청산은 위(u>0)
        Ld = float(pv[(u < 0) & (u >= -a.band)].sum())
        Lu = float(pv[(u > 0) & (u <= a.band)].sum())
        Ld_i = float(pv[isl & (u < 0) & (u >= -a.band)].sum())
        Lu_i = float(pv[isl & (u > 0) & (u <= a.band)].sum())
        Rd = float(gb.loc[gb.pos_side == "long", "ntl"].sum())    # 롱 청산 = 아래
        Ru = float(gb.loc[gb.pos_side == "short", "ntl"].sum())
        if (Ld + Lu) <= 0 or (Rd + Ru) <= 0:
            continue
        recs.append({"coin": coin, "hour": hr,
                     "map_imb": (Ld - Lu) / (Ld + Lu),
                     "map_imb_iso": ((Ld_i - Lu_i) / (Ld_i + Lu_i)
                                     if (Ld_i + Lu_i) > 0 else np.nan),
                     "real_imb": (Rd - Ru) / (Rd + Ru),
                     "real_usd": Rd + Ru})
    r = pd.DataFrame(recs)
    print("  표본 %d (심볼x시간), 심볼 %d" % (len(r), r.coin.nunique() if len(r) else 0))
    if len(r) >= 10:
        for lab, col in (("전체 지도", "map_imb"), ("격리만", "map_imb_iso")):
            s = r[[col, "real_imb"]].dropna()
            if len(s) < 10:
                print("  %-10s n 부족 (%d)" % (lab, len(s)))
                continue
            pe = float(s[col].corr(s["real_imb"]))
            sp = float(s[col].corr(s["real_imb"], method="spearman"))
            se = 1.0 / np.sqrt(max(len(s) - 3, 1))
            print("  %-10s n=%4d  Pearson %+.3f  Spearman %+.3f   (대략 SE %.3f)"
                  % (lab, len(s), pe, sp, se))
        big = r[r.real_usd >= 1e5]
        if len(big) >= 10:
            print("  청산 $100K 이상인 시간만: n=%d  Pearson %+.3f"
                  % (len(big), float(big["map_imb"].corr(big["real_imb"]))))
        print("  부호 일치율: %.1f%% (지도와 실현의 불균형 부호가 같은 비율, 우연=50%%)"
              % (100 * float((np.sign(r["map_imb"]) == np.sign(r["real_imb"])).mean())))
    else:
        print("  표본 부족 — 축적 필요")

    # ---------------------------------------------------------- (3) 가격대별
    print("\n--- 3. 가격대별: 실현 청산이 일어난 곳에서 지도가 두꺼웠는가 ---")
    print("  거리 통제 필수 — 청산은 원래 현재가 근처에서 일어난다.")
    print("  실현 청산가의 |u| 와 같은 거리의 '반대편' 지도 두께를 대조군으로 쓴다.")
    hits = []
    for coin, gb in byb.groupby("coin"):
        for rr in gb.itertuples():
            j = int(np.searchsorted(sw_t, rr.exch_ms, side="right")) - 1
            if j < 0 or rr.exch_ms - int(sw_t[j]) > MAX_MAP_LAG_MS:
                continue
            sid = sw_ids[j]
            m = mid_map.get((sid, coin))
            if m is None or not np.isfinite(m) or m <= 0:
                continue
            p = pos[(pos.sweep_id == sid) & (pos.coin == coin)]
            if p.empty:
                continue
            u_real = rr.bankruptcy_px / m - 1.0
            if not np.isfinite(u_real) or abs(u_real) > a.band or abs(u_real) < 1e-4:
                continue
            u = (p["liquidation_px"] / m - 1.0).to_numpy()
            pv = p["position_value"].to_numpy()
            w = 0.2 * abs(u_real)                       # +-20% 상대폭 밴드
            same = float(pv[(u >= u_real - w) & (u <= u_real + w)].sum())
            opp = float(pv[(u >= -u_real - w) & (u <= -u_real + w)].sum())
            hits.append({"coin": coin, "ntl": rr.ntl, "u": u_real,
                         "L_same": same, "L_opp": opp})
    h = pd.DataFrame(hits)
    print("  매칭된 실현청산 %d건" % len(h))
    if len(h) >= 20:
        both = h[(h.L_same + h.L_opp) > 0]
        win = float((both.L_same > both.L_opp).mean())
        print("  실현 위치 지도두께 > 반대편 두께 인 비율: %.1f%%  (우연=50%%, n=%d)"
              % (100 * win, len(both)))
        print("  L_same 중앙 $%.4g  vs  L_opp 중앙 $%.4g"
              % (both.L_same.median(), both.L_opp.median()))
        z = (win - 0.5) / (0.5 / np.sqrt(len(both)))
        print("  이항검정 z = %+.2f  (|z|>1.96 이면 우연 아님)" % z)
        print("  L_same=0 인 비율: %.1f%%  <- 지도가 그 가격대를 아예 못 본 비율"
              % (100 * float((h.L_same == 0).mean())))
    else:
        print("  표본 부족 — 축적 필요")

    print("\n판정: (1)이 얇으면 지도에 근접 연료가 없다.")
    print("      (2)(3)이 우연 수준이면 Q1 불합격 -> 설계의 L(p) 분기가 막힌다.")
    print("      40시간 1차이므로 '방향 확인' 용도다. 확정 판정은 축적 후.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
