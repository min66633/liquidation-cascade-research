# -*- coding: utf-8 -*-
"""pi(u) 대리변수 검정 — 청산맵 없이 '상대가 누구인가'를 잴 수 있나.

배경
  EV(u) = S(u) * E[r | fill(u)] - c 에서 S 쪽 조건부화는 전부 실패했다
  (고정 offset / 손수 규칙 / 적합된 조건부 해저드 모두 고정을 못 이김).
  이유도 구조적이다 — 무차별성이 **곱**에서 성립하므로 S 를 개선하면 E[r|fill] 이 상쇄한다.

  남은 항은 pi(u) = U/(U+I) 하나다. U = L(강제청산) + Sigma(재량 손절).
  지금까지 pi 의 대리변수는 **dOI <= -2% 이진값 하나**뿐이었고, 그것만으로
  LIQ - CTRL = 35.2bp 가 나왔다. 더 나은 대리변수면 더 나올 수 있다.

이 스크립트가 쓰는 대리변수 (전부 트리거 시점 관측 가능, 청산맵 불필요)
  crowd_lsr     상위트레이더 롱숏비율의 자기 과거 대비 z 점수  -> 포지션 쏠림
  crowd_cnt     계정수 기준 롱숏비율 z 점수
  taker_ls      테이커 롱숏 거래량비 z 점수                     -> 흐름 쏠림
  funding_z     펀딩비 z 점수                                  -> 쏠림의 '가격'
  doi_mag       |dOI| 연속값                                    -> 청산 강도(이진 아님)

방향 가설
  롱청산 이벤트(가격 하락)에서는 **롱이 붐빌수록** 강제 롱청산이 많다
  -> pi 가 높다 -> 체결 상대가 정보 없는 쪽일 확률이 높다 -> 수익이 높다.
  숏청산 이벤트에서는 부호가 뒤집힌다. 그래서 방향을 곱해 '유리한 쏠림'으로 정규화한다.

검정
  (1) 대리변수 5분위별 이벤트당 EV — 단조인가
  (2) 훈련 절반에서 상위 구간을 고르고 평가 절반에서 확인 (표본 외)
  (3) 기존 이진 필터(dOI<=-2%) 위에 더 얹었을 때 개선되는가

실행:
    python analysis/pi_proxy.py
    python analysis/pi_proxy.py --offset 0.02 --hold-min 15
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
from analysis.event_study_h2 import load, find_events, nw_tstat   # noqa: E402

BULK = os.path.join(C.DATA, "binance_bulk")
MIN_MS = 60_000
BAR_MS = 300_000
ZWIN = 288 * 7          # z 점수 창 = 7일 (5분봉)


def load_1m(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "klines_1m", "%s.parquet" % symbol)
    if not os.path.exists(p):
        raise FileNotFoundError("missing 1m klines for %s" % symbol)
    return (pd.read_parquet(p)[["open_time", "high", "low", "close"]]
              .sort_values("open_time").reset_index(drop=True))


def load_funding(symbol: str) -> pd.DataFrame:
    p = os.path.join(BULK, "funding", "%s.parquet" % symbol)
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_parquet(p)[["ts_ms", "funding"]].sort_values("ts_ms")


def zscore(s: pd.Series, win: int) -> pd.Series:
    """자기 과거 대비 z. 현재 값 제외(shift)해서 룩어헤드를 막는다."""
    m = s.shift(1).rolling(win, min_periods=win // 4).mean()
    sd = s.shift(1).rolling(win, min_periods=win // 4).std()
    return (s - m) / sd.replace(0.0, np.nan)


def build(symbol: str, offset: float, ttl_min: int, hold_min: int,
          k: float, doi_thr: float, min_gap: int, cost: float) -> pd.DataFrame:
    df5 = load(symbol)
    m1 = load_1m(symbol)
    if df5.empty or m1.empty:
        return pd.DataFrame()

    # ---- pi 대리변수 (전부 사전 관측 가능)
    df5 = df5.copy()
    df5["z_lsr"] = zscore(df5["sum_toptrader_long_short_ratio"], ZWIN)
    df5["z_cnt"] = zscore(df5["count_long_short_ratio"], ZWIN)
    df5["z_tls"] = zscore(df5["sum_taker_long_short_vol_ratio"], ZWIN)
    fr = load_funding(symbol)
    if not fr.empty:
        # 펀딩은 8시간 간격 -> 5분 격자에 직전값으로 채운다(미래값 금지)
        idx = np.searchsorted(fr["ts_ms"].to_numpy(),
                              df5["open_time"].to_numpy(), side="right") - 1
        fv = np.where(idx >= 0, fr["funding"].to_numpy()[np.clip(idx, 0, None)], np.nan)
        df5["funding"] = fv
        df5["z_fund"] = zscore(df5["funding"], 21 * 3)      # 21일 x 하루 3회
    else:
        df5["funding"] = np.nan
        df5["z_fund"] = np.nan

    ev = find_events(df5, k, doi_thr, min_gap)
    ev = ev[ev.is_liq]
    if ev.empty:
        return pd.DataFrame()

    ot = m1["open_time"].to_numpy()
    lo, hi, cl = (m1[c].to_numpy() for c in ("low", "high", "close"))
    n1 = len(ot)
    t5, close5 = df5["open_time"].to_numpy(), df5["close"].to_numpy()
    cols = {c: df5[c].to_numpy() for c in
            ("z_lsr", "z_cnt", "z_tls", "z_fund", "funding", "doi", "sigma")}

    out = []
    for r in ev.itertuples():
        i = r.i
        p0 = close5[i]
        if not (np.isfinite(p0) and p0 > 0):
            continue
        limit = p0 * (1 - offset) if r.side == 1 else p0 * (1 + offset)
        a = int(np.searchsorted(ot, int(t5[i]) + BAR_MS, side="left"))
        b = int(np.searchsorted(ot, int(t5[i]) + BAR_MS + ttl_min * MIN_MS, side="left"))
        if a >= n1 or b <= a or b + hold_min >= n1:
            continue
        rec = {"symbol": symbol, "trig_ms": int(t5[i]), "side": int(r.side),
               "doi_mag": abs(float(cols["doi"][i])),
               "sigma": float(cols["sigma"][i])}
        # 방향 정규화: 롱청산(side=+1, 하락)이면 롱 쏠림(+)이 유리.
        # 숏청산(side=-1, 상승)이면 숏 쏠림(즉 LSR 낮음)이 유리 -> 부호를 뒤집는다.
        for nm, key in (("crowd_lsr", "z_lsr"), ("crowd_cnt", "z_cnt"),
                        ("crowd_tls", "z_tls"), ("crowd_fund", "z_fund")):
            v = cols[key][i]
            rec[nm] = float(v) * r.side if np.isfinite(v) else np.nan
        rec["funding_raw"] = float(cols["funding"][i]) * r.side \
            if np.isfinite(cols["funding"][i]) else np.nan

        seg = (lo[a:b] <= limit) if r.side == 1 else (hi[a:b] >= limit)
        idx = np.flatnonzero(seg)
        if idx.size == 0:
            rec.update({"filled": False, "ret": np.nan, "mae": np.nan})
        else:
            j = a + int(idx[0])
            e = min(j + hold_min, n1 - 1)
            rec.update({"filled": True,
                        "ret": float((cl[e] / limit - 1.0) * r.side - cost),
                        "mae": float((lo[j:e + 1].min() / limit - 1.0) if r.side == 1
                                     else (1.0 - hi[j:e + 1].max() / limit))})
        out.append(rec)
    return pd.DataFrame(out)


def agg(h: pd.DataFrame) -> dict:
    f = h[h["filled"] & np.isfinite(h["ret"])]
    n = len(h)
    fr = len(f) / max(n, 1)
    if not len(f):
        return {"n_ev": n, "fill%": 0.0, "cond_bp": np.nan, "per_ev_bp": np.nan,
                "win%": np.nan, "mae_p05": np.nan, "t_NW": np.nan}
    r = f["ret"].to_numpy()
    return {"n_ev": n, "fill%": 100 * fr, "cond_bp": 1e4 * r.mean(),
            "per_ev_bp": 1e4 * r.mean() * fr, "win%": 100 * (r > 0).mean(),
            "mae_p05": 1e4 * np.nanpercentile(f["mae"], 5), "t_NW": nw_tstat(r, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description="pi proxies without the liquidation map")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--offset", type=float, default=0.02)
    ap.add_argument("--ttl-min", type=int, default=60)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--doi", type=float, default=-0.02)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--cost-bps", type=float, default=7.0)
    ap.add_argument("--bins", type=int, default=5)
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 220)
    syms = a.symbols if a.symbols else C.MAJORS
    cost = a.cost_bps / 1e4

    frames = []
    for s in syms:
        try:
            d = build(s, a.offset, a.ttl_min, a.hold_min, a.k, a.doi, a.min_gap, cost)
        except FileNotFoundError as e:
            U.log(str(e))
            continue
        if not d.empty:
            frames.append(d)
    if not frames:
        U.log("no events")
        return 1
    d = pd.concat(frames, ignore_index=True)
    d["dt"] = pd.to_datetime(d["trig_ms"], unit="ms", utc=True)
    d = d.sort_values("dt").reset_index(drop=True)
    U.atomic_write_parquet(d.drop(columns=["dt"]),
                           os.path.join(C.DATA, "analysis", "pi_proxy.parquet"))

    base = agg(d)
    print("\n=== 표본 ===")
    print("이벤트 %d | 심볼 %d | %s ~ %s | offset %.1f%% HOLD %d분"
          % (len(d), d.symbol.nunique(), d["dt"].min().date(), d["dt"].max().date(),
             100 * a.offset, a.hold_min))
    print("기준선(전체): 체결률 %.1f%% | 조건부 %.1fbp | 이벤트당 %.1fbp | t %.1f"
          % (base["fill%"], base["cond_bp"], base["per_ev_bp"], base["t_NW"]))

    PROXIES = [("crowd_lsr", "상위트레이더 롱숏비 z"),
               ("crowd_cnt", "계정수 롱숏비 z"),
               ("crowd_tls", "테이커 롱숏비 z"),
               ("crowd_fund", "펀딩비 z"),
               ("funding_raw", "펀딩비 원값"),
               ("doi_mag", "|dOI| 연속")]

    # ---------------------------------------------------------- (1) 5분위
    print("\n=== (1) 대리변수 5분위별 이벤트당 EV — 단조인가 ===")
    print("  (방향 정규화됨: 값이 클수록 '우리에게 유리한 쏠림')")
    for col, lab in PROXIES:
        v = d[col]
        ok = d[np.isfinite(v)]
        if len(ok) < a.bins * 20:
            print("  %-18s 표본부족 n=%d" % (lab, len(ok)))
            continue
        ok = ok.copy()
        ok["q"] = pd.qcut(ok[col].rank(method="first"), a.bins, labels=False)
        row = []
        for q in range(a.bins):
            r = agg(ok[ok["q"] == q])
            row.append(r["per_ev_bp"])
        rho = float(pd.Series(range(a.bins)).corr(pd.Series(row), method="spearman"))
        print("  %-18s n=%4d  " % (lab, len(ok))
              + "  ".join("%+6.1f" % x if np.isfinite(x) else "   n/a" for x in row)
              + "   |  Q5-Q1 %+6.1f  rho %+.2f"
              % ((row[-1] - row[0]) if np.isfinite(row[-1]) and np.isfinite(row[0])
                 else np.nan, rho))

    # ---------------------------------------------------------- (2) 표본 외
    print("\n=== (2) 표본 외 — 훈련에서 상위 40%% 구간을 고르고 평가에서 확인 ===")
    cut = len(d) // 2
    tr, te = d.iloc[:cut], d.iloc[cut:]
    print("  훈련 %d (%s~%s) / 평가 %d (%s~%s)"
          % (len(tr), tr["dt"].min().date(), tr["dt"].max().date(),
             len(te), te["dt"].min().date(), te["dt"].max().date()))
    b_te = agg(te)
    print("  평가구간 전체(필터 없음): 이벤트당 %.1fbp (n=%d, t=%.1f)"
          % (b_te["per_ev_bp"], b_te["n_ev"], b_te["t_NW"]))
    print("\n  %-18s %8s %8s %10s %10s %8s %8s"
          % ("필터", "임계", "이벤트", "체결률", "이벤트당bp", "승률", "t"))
    for col, lab in PROXIES:
        s_tr = tr[np.isfinite(tr[col])]
        if len(s_tr) < 40:
            continue
        thr = float(s_tr[col].quantile(0.60))          # 상위 40%
        sel = te[np.isfinite(te[col]) & (te[col] >= thr)]
        if len(sel) < 20:
            print("  %-18s %8.2f %8s" % (lab, thr, "평가표본부족"))
            continue
        r = agg(sel)
        print("  %-18s %8.2f %8d %9.1f%% %10.1f %7.1f%% %8.1f"
              % (lab, thr, r["n_ev"], r["fill%"], r["per_ev_bp"], r["win%"], r["t_NW"]))
    print("  평가구간 전체 %.1fbp 를 **의미 있게** 넘어야 대리변수에 정보가 있는 것이다."
          % b_te["per_ev_bp"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
