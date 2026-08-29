# -*- coding: utf-8 -*-
"""대표성 검정 — Hyperliquid 청산맵이 시장 전체 청산을 예측하는가.

이 연구 전체의 핵심 미검증 가정
  실측 청산가 분포는 온체인에서만 얻을 수 있고, 사실상 Hyperliquid 하나다.
  그런데 HL은 BTC perp OI의 약 15%(실측: Binance 47 / Bybit 24.5 / OKX 13.5 / HL 15)이고,
  그중에서도 이 수집기가 잡는 것은 HL 포지션의 일부다. 지도가 시장의 작은 조각인 셈이다.

  작은 표본이어도 **편향이 없으면** 쓸 수 있다(여론조사와 같은 논리). 편향되면 못 쓴다.
  이 스크립트는 그 편향 여부를 직접 측정한다.

측정 방법
  가격이 어떤 구간을 통과할 때마다:
    예측 = 통과 직전 HL 스냅샷의 그 구간 청산 연료(명목가)
    실현 = 같은 구간에서 Bybit이 실제로 찍은 청산 명목가(전건 피드, 파산가 기준)
  둘의 관계를 본다. 대형 캐스케이드가 필요 없고 평상시 가격 움직임만으로 축적된다.

판정
  - 상관이 유의하게 양수  -> HL 지도가 시장 전체 연료 위치를 담고 있다. 연구 진행.
  - 상관이 0             -> HL 표본이 편향됐거나 지도가 무정보. 접근 재설계 필요.
  - 회귀계수(실현/예측)   -> 시장 전체로 투영할 스케일 계수의 추정치.

실행:
    python analysis/representativeness.py
    python analysis/representativeness.py --bin-bps 25 --min-obs 30
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

def build_symbol_map(hl_coins, bybit_symbols) -> dict[str, str]:
    """HL 코인명 -> Bybit 심볼 매핑을 데이터에서 직접 만든다.

    규칙은 단순하다: COIN + "USDT". 예외는 HL의 k접두(kPEPE = 1000PEPE) 정도다.
    실측(2026-07-31): HL 177코인 중 170개가 Bybit에 대응 종목을 가진다.
    하드코딩하면 코인이 추가될 때마다 조용히 표본에서 빠지므로 자동화한다.
    """
    bset = set(map(str, bybit_symbols))
    out = {}
    for c in hl_coins:
        c = str(c)
        for cand in ([c + "USDT"] + (["1000" + c[1:] + "USDT"] if c.startswith("k") else [])):
            if cand in bset:
                out[c] = cand
                break
    return out


def load_hl_snapshots() -> pd.DataFrame:
    """전체 스윕의 포지션 + 해당 스윕의 mid를 결합해 청산가/명목가 표로."""
    pos_files = sorted(glob.glob(os.path.join(C.HL_DIR_POSITIONS, "*", "positions_*.parquet")))
    mid_files = {os.path.basename(f).split("_")[-1].split(".")[0]: f
                 for f in glob.glob(os.path.join(C.HL_DIR_MIDS, "*", "mids_*.parquet"))}
    frames = []
    for f in pos_files:
        sid = os.path.basename(f).split("_")[-1].split(".")[0]
        mf = mid_files.get(sid)
        if mf is None:
            continue
        try:
            p = pd.read_parquet(f, columns=["sweep_id", "address", "coin", "szi",
                                            "liquidation_px"])
            m = pd.read_parquet(mf)
        except Exception:
            continue
        if p.empty:
            continue
        m = m[m["phase"] == "start"].set_index("coin")["mid_px"]
        p = p[p["liquidation_px"].notna()].copy()
        p["mark"] = p["coin"].map(m)
        p = p[p["mark"].notna() & (p["mark"] > 0)]
        if p.empty:
            continue
        p["notional"] = p["szi"].abs() * p["mark"]
        p["pos_side"] = np.where(p["szi"] > 0, "long", "short")
        frames.append(p[["sweep_id", "coin", "pos_side", "liquidation_px",
                         "notional", "mark"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_bybit_liq() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(C.BYBIT_LIQ_DIR, "*", "liq_*.parquet")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d["bankruptcy_px"] > 0]
    d["notional"] = d["size"] * d["bankruptcy_px"]
    return d


def price_path(coin: str) -> pd.DataFrame:
    """모든 스냅샷(깊은+핫)의 mid를 모아 가격 경로를 만든다.

    핫 스윕이 60초마다 mid를 남기므로, 깊은 스윕 간격(15분)보다 훨씬 촘촘한
    경로를 얻는다. 창 안에서 가격이 실제로 지나간 구간을 판정하는 데 쓴다.
    """
    frames = []
    for f in glob.glob(os.path.join(C.HL_DIR_MIDS, "*", "mids_*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["ts", "coin", "mid_px"])
        except Exception:
            continue
        d = d[d["coin"] == coin]
        if not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["ts", "mid_px"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["mid_px"] > 0].sort_values("ts").reset_index(drop=True)
    return out[["ts", "mid_px"]]


def build_pairs(hl: pd.DataFrame, by: pd.DataFrame, bin_bps: float,
                smap: dict[str, str] | None = None,
                pad_pct: float = 2.0, max_bins: int = 4000) -> pd.DataFrame:
    """스윕 구간 x 가격빈 단위로 (예측 연료, 실현 청산)을 짝짓는다.

    스윕 t의 지도는 [t, t+1) 구간에 발생한 청산을 예측해야 한다 -> 룩어헤드 없음.

    중요 — **가격이 실제로 지나간 구간만** 검정 대상이다. 가격이 근처에도 안 간 곳에
    연료가 있다고 예측했는데 청산이 안 나온 것은 예측 실패가 아니다. 그런 빈을 넣으면
    '예측했는데 안 터졌다'가 대량으로 섞여 상관이 인위적으로 0에 끌려간다.
    가격 경로는 핫 스윕이 60초마다 남긴 mid로 만든다.
    """
    if hl.empty or by.empty:
        return pd.DataFrame()

    if smap is None:
        smap = build_symbol_map(hl["coin"].unique(), by["symbol"].unique())
    sweeps = np.sort(hl["sweep_id"].unique())
    rows = []
    for coin, byb_sym in smap.items():
        h_coin = hl[hl["coin"] == coin]
        b_coin = by[by["symbol"] == byb_sym]
        if h_coin.empty or b_coin.empty:
            continue
        path = price_path(coin)
        if path.empty:
            continue
        pts = path["ts"].to_numpy()
        pxs = path["mid_px"].to_numpy()

        for si, sid in enumerate(sweeps[:-1]):
            nxt = sweeps[si + 1]
            hsnap = h_coin[h_coin["sweep_id"] == sid]
            if hsnap.empty:
                continue
            bwin = b_coin[(b_coin["exch_ms"] >= sid) & (b_coin["exch_ms"] < nxt)]
            if bwin.empty:
                continue
            mark = float(hsnap["mark"].iloc[0])
            if not np.isfinite(mark) or mark <= 0:
                continue

            # 창 안에서 가격이 지나간 범위 (여유 pad_pct% 를 준다)
            sel = (pts >= sid) & (pts <= nxt)
            seg = pxs[sel]
            if len(seg) < 2:
                continue
            lo = float(seg.min()) * (1.0 - pad_pct / 100.0)
            hi = float(seg.max()) * (1.0 + pad_pct / 100.0)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue

            step = mark * bin_bps / 1e4
            n_bins = int((hi - lo) / step) + 1
            # 배열을 만들기 '전에' 크기를 검사한다. 안 그러면 극단적 청산가 하나가
            # 수십억 개짜리 배열을 요구해 MemoryError로 죽는다(실제로 발생).
            if n_bins < 3 or n_bins > max_bins:
                continue
            edges = lo + step * np.arange(n_bins + 1)

            for side in ("long", "short"):
                hs = hsnap[hsnap.pos_side == side]
                bs = bwin[bwin.pos_side == side]
                pred, _ = np.histogram(hs["liquidation_px"], bins=edges,
                                       weights=hs["notional"])
                real, _ = np.histogram(bs["bankruptcy_px"], bins=edges,
                                       weights=bs["notional"])
                for j in range(len(pred)):
                    rows.append({"coin": coin, "sweep_id": int(sid), "side": side,
                                 "bin_px": float(0.5 * (edges[j] + edges[j + 1])),
                                 "pred_notional": float(pred[j]),
                                 "real_notional": float(real[j])})
    return pd.DataFrame(rows)


def report(pairs: pd.DataFrame, min_obs: int) -> None:
    if pairs.empty:
        print("no paired observations yet — need more overlapping HL sweeps and Bybit liquidations")
        return
    print("paired bins: %d | coins: %d | sweeps: %d"
          % (len(pairs), pairs.coin.nunique(), pairs.sweep_id.nunique()))
    if len(pairs) < min_obs:
        print("below --min-obs (%d) — collect longer before judging" % min_obs)
        return

    x = pairs["pred_notional"].to_numpy(dtype="float64")
    y = pairs["real_notional"].to_numpy(dtype="float64")
    lx, ly = np.log1p(x), np.log1p(y)
    r = np.corrcoef(lx, ly)[0, 1] if len(x) > 2 else np.nan
    print("corr(log1p pred, log1p real) = %.3f" % r)

    # 예측 연료 유무로 나눠 실현 청산 비교 — 가장 단순한 정보성 검정
    has = pairs[pairs.pred_notional > 0]["real_notional"]
    non = pairs[pairs.pred_notional <= 0]["real_notional"]
    print("bins WITH predicted fuel: n=%d, mean realized $%.0f, P(any)=%.1f%%"
          % (len(has), has.mean() if len(has) else 0,
             100 * (has > 0).mean() if len(has) else 0))
    print("bins WITHOUT predicted fuel: n=%d, mean realized $%.0f, P(any)=%.1f%%"
          % (len(non), non.mean() if len(non) else 0,
             100 * (non > 0).mean() if len(non) else 0))

    pos = pairs[(pairs.pred_notional > 0) & (pairs.real_notional > 0)]
    if len(pos) >= 10:
        ratio = pos["real_notional"] / pos["pred_notional"]
        print("scale factor (realized/predicted) median %.2f  IQR %.2f~%.2f  n=%d"
              % (ratio.median(), ratio.quantile(0.25), ratio.quantile(0.75), len(pos)))
        print("  -> HL 지도를 시장 전체로 투영할 때의 배율 추정치")


def main() -> int:
    ap = argparse.ArgumentParser(description="is the HL liquidation map representative?")
    ap.add_argument("--bin-bps", type=float, default=25.0, help="price bin width in bps of mark")
    ap.add_argument("--min-obs", type=int, default=50)
    a = ap.parse_args()

    U.init_stdout()
    hl = load_hl_snapshots()
    by = load_bybit_liq()
    U.log("HL rows %d (sweeps %d) | Bybit liq rows %d"
          % (len(hl), hl.sweep_id.nunique() if not hl.empty else 0, len(by)))
    if hl.empty or by.empty:
        U.log("need both feeds to have overlapping data")
        return 1

    smap = build_symbol_map(hl["coin"].unique(), by["symbol"].unique())
    U.log("symbol map: %d HL coins matched to Bybit symbols" % len(smap))
    pairs = build_pairs(hl, by, a.bin_bps, smap)
    if not pairs.empty:
        U.atomic_write_parquet(pairs, os.path.join(C.DATA, "analysis", "representativeness_pairs.parquet"))
    print()
    report(pairs, a.min_obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
