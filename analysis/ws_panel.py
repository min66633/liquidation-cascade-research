# -*- coding: utf-8 -*-
"""웹소켓 1초 패널 구축 + 사건 census — 봉을 버리기 위한 첫 단계.

왜 필요한가 (사용자, 2026-08-05)
  "웹소켓으로 밀리초 데이터 다 받아놨는데 왜 안 쓰는지 모르겠어요"
  "봉으로 계산하는 게 맞아요?"

  맞지 않다. 1분봉 방아쇠는 봉이 **닫혀야** 판정되므로 캐스케이드가 봉 앞쪽에서
  터지면 최대 59초를 버린다. 1~3분 보유의 t 가 낮았던 것이 알파 부재인지
  이 지연 때문인지 봉으로는 구분되지 않는다.

  depth_ws 는 21종 × 1초 간격으로 mid 와 양쪽 12개 밴드 누적 명목가를 담고 있다.
  즉 **가격도 깊이도 1초**다. depth_ws_flow 는 그 밴드들의 차분(= 설계의 ③),
  oi_fast 는 고빈도 OI(= 설계의 dOI 를 5분 지연 없이), bybit_liq 는 밀리초 청산.

이 스크립트가 하는 일
  1. depth_ws / depth_ws_flow / oi_fast 를 심볼별 1초 격자로 합쳐 캐시
  2. **결측·재동기화 구간을 명시**한다 (연속성이 없으면 60초 창이 거짓말을 한다)
  3. 봉을 쓰지 않는 방아쇠로 사건을 센다:
       z60(t) = [mid(t)/mid(t-60s) - 1] / sigma60,  **매 초 평가**
     봉 격자가 없으므로 확인 지연이 0 이다.
  4. K 별 사건 수를 보고한다 — 표본이 몇 건인지 먼저 알아야 다음을 정한다

실행:
    python analysis/ws_panel.py            # 캐시 만들고 census
    python analysis/ws_panel.py --rebuild  # 캐시 재생성
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

CACHE = os.path.join(C.DATA, "analysis", "ws_panel")
SEC = 1000
W = 118


def _load_dir(sub, cols, ts="ts_ms"):
    fs = sorted(glob.glob(os.path.join(C.DATA, sub, "*", "*.parquet")))
    if not fs:
        return None
    out = []
    for f in fs:
        try:
            out.append(pd.read_parquet(f, columns=cols))
        except Exception:
            continue
    if not out:
        return None
    d = pd.concat(out, ignore_index=True)
    return d.sort_values([ "symbol", ts]).reset_index(drop=True)


def build(rebuild=False):
    """심볼별 1초 격자 패널을 만들어 캐시. 각 심볼 -> DataFrame(초 인덱스)."""
    os.makedirs(CACHE, exist_ok=True)
    done = sorted(glob.glob(os.path.join(CACHE, "*.parquet")))
    if done and not rebuild:
        return {os.path.basename(p)[:-8]: p for p in done}

    U.log("depth_ws 로드")
    bk = _load_dir("depth_ws", ["ts_ms", "symbol", "mid", "resyncs",
                                "bid_b0_5", "bid_b1", "bid_b2", "bid_b5",
                                "ask_b0_5", "ask_b1", "ask_b2", "ask_b5"])
    if bk is None:
        raise FileNotFoundError("depth_ws 가 비어 있다")
    U.log("depth_ws_flow 로드")
    fl = _load_dir("depth_ws_flow", ["ts_ms", "symbol",
                                     "dbid_b0_5", "dbid_b1", "dbid_b2",
                                     "dask_b0_5", "dask_b1", "dask_b2"])
    U.log("oi_fast 로드")
    oi = _load_dir("oi_fast", ["ts_ms", "symbol", "oi_usd"])

    out = {}
    for s, g in bk.groupby("symbol"):
        g = g.drop_duplicates("ts_ms").sort_values("ts_ms")
        sec = (g["ts_ms"].to_numpy() // SEC).astype(np.int64)
        g = g.assign(sec=sec).drop_duplicates("sec", keep="last")
        idx = np.arange(int(g["sec"].min()), int(g["sec"].max()) + 1, dtype=np.int64)
        d = g.set_index("sec").reindex(idx)
        d.index.name = "sec"

        if fl is not None:
            f = fl[fl["symbol"] == s].copy()
            if len(f):
                f["sec"] = (f["ts_ms"].to_numpy() // SEC).astype(np.int64)
                # 같은 초에 여러 갱신이 있으면 **합**이 유량의 정의다
                f = f.groupby("sec")[["dbid_b0_5", "dbid_b1", "dbid_b2",
                                      "dask_b0_5", "dask_b1", "dask_b2"]].sum()
                d = d.join(f, how="left")
        if oi is not None:
            o = oi[oi["symbol"] == s].copy()
            if len(o):
                o["sec"] = (o["ts_ms"].to_numpy() // SEC).astype(np.int64)
                o = o.groupby("sec")["oi_usd"].last()
                # OI 는 상태량이므로 앞으로 채운다 (마지막 알려진 값)
                d = d.join(o, how="left")
                d["oi_usd"] = d["oi_usd"].ffill()
        d = d.drop(columns=["symbol", "ts_ms"], errors="ignore")
        p = os.path.join(CACHE, "%s.parquet" % s)
        d.reset_index().to_parquet(p, index=False)
        out[s] = p
        U.log("  %-10s %7d초 (관측 %6d, 결측 %5.1f%%)"
              % (s, len(d), int(d["mid"].notna().sum()),
                 100 * d["mid"].isna().mean()))
    return out


def load(sym):
    p = os.path.join(CACHE, "%s.parquet" % sym)
    d = pd.read_parquet(p).set_index("sec")
    return d


def gaps(d, max_gap=5):
    """연속 관측 구간(run)으로 쪼갠다. 결측이 max_gap 초를 넘으면 끊는다."""
    ok = d["mid"].notna().to_numpy()
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    brk = np.flatnonzero(np.diff(idx) > max_gap)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [len(idx) - 1]])
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def census(syms, ks=(3, 4, 5, 6, 8, 10), win=60, vol=3600):
    """봉 없는 방아쇠로 사건을 센다. 매 초 평가, 확인 지연 0."""
    print("=" * W)
    print("웹소켓 1초 패널 — 사건 census (봉 격자 없음, 매 초 평가)")
    print("=" * W)
    print("방아쇠: z60(t) = [mid(t)/mid(t-%ds) - 1] / sigma60,  sigma60 = 과거 %d초의"
          % (win, vol))
    print("        %ds 수익 표준편차(현재 제외). 하락 방향(z60 <= -K)만 센다.\n" % win)
    tot_sec, rows = 0, []
    print("  %-10s | %8s %8s %7s | %s"
          % ("심볼", "관측초", "연속구간", "결측%", "  ".join("K=%g" % k for k in ks)))
    cnt_all = {k: 0 for k in ks}
    for s in syms:
        try:
            d = load(s)
        except FileNotFoundError:
            continue
        runs = gaps(d)
        mid = d["mid"].to_numpy(dtype=np.float64)
        n_ok = int(np.isfinite(mid).sum())
        tot_sec += n_ok
        m = pd.Series(mid)
        r = (m / m.shift(win) - 1.0)
        sd = r.rolling(vol, min_periods=vol // 4).std().shift(1)
        z = (r / sd).to_numpy()
        # 연속 구간 안에서만 유효 (창이 결측을 건너뛰면 거짓 신호)
        valid = np.zeros(len(mid), dtype=bool)
        for a, b in runs:
            if b - a >= win + vol // 4:
                valid[a + win + vol // 4:b + 1] = True
        cs = []
        for k in ks:
            hit = valid & np.isfinite(z) & (z <= -k)
            # 같은 사건의 연속 초를 1건으로: 60초 이내 재발동은 제외
            ii = np.flatnonzero(hit)
            c, last = 0, -10**9
            for i in ii:
                if i - last < win:
                    continue
                last = i
                c += 1
            cs.append(c)
            cnt_all[k] += c
        rows.append((s, n_ok, len(runs), 100 * (1 - n_ok / max(len(mid), 1)), cs))
        print("  %-10s | %8d %8d %6.1f%% | %s"
              % (s, n_ok, len(runs), 100 * (1 - n_ok / max(len(mid), 1)),
                 "  ".join("%4d" % c for c in cs)))
    print("  " + "-" * (W - 2))
    print("  %-10s | %8d %8s %7s | %s"
          % ("합계", tot_sec, "", "", "  ".join("%4d" % cnt_all[k] for k in ks)))
    print("\n  총 관측 %.1f 심볼-시간 (심볼당 %.1f 시간)"
          % (tot_sec / 3600, tot_sec / 3600 / max(len(rows), 1)))
    print("\n  ** 이 표가 다음 단계를 정한다. K=8~10 에서 두 자릿수가 안 되면")
    print("     1초 표본으로 손익을 추정할 수 없고, 6년 봉 데이터로 지연 기울기를")
    print("     재는 쪽(analysis/latency.py)이 유일한 경로다. **")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="websocket 1s panel + event census")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    a = ap.parse_args()
    U.init_stdout()
    build(rebuild=a.rebuild)
    syms = a.symbols if a.symbols else C.MAJORS
    census(syms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
