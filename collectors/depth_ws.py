# -*- coding: utf-8 -*-
"""Binance 선물 호가 diff 스트림 — 로컬 북 유지 + 추가/취소 이벤트 관측.

왜 이것이 필요한가 (2026-08-02 실측)
  현행 depth_poll 은 REST 스냅샷이라 세 가지가 막혀 있다.

  | REST 30초 (depth_poll)          | 웹소켓 diff (이 파일)        |
  |---------------------------------|------------------------------|
  | 도달범위 BTC 0.19% / ETH 0.56%  | **전 깊이** (로컬 북 유지)   |
  | 30초 (캐스케이드는 몇 초)       | 100ms                        |
  | **취소를 못 봄** (차이만 보임)  | **추가/취소를 이벤트로 관측**|

  착수 근거는 **도달범위**다. liq_cluster.py 에서 BTC 청산 769건 중 유효 클러스터가
  **1건**이었다 — 도달범위가 0.19% 라 밴드 3개를 못 채워 전부 탈락했다.
  메이저 2종의 가격대별 분석이 사실상 불가능한 상태다.

  (W = 호가 잔존율 예보는 QW 검정에서 떨어졌다. 그쪽 근거는 약하다.
   다만 Queue-reactive 모듈의 Delta(v,t) 는 이 데이터로만 관측된다.)

*** L(p) 와 같은 성질: 과거 백필이 불가능하다. 지금부터 쌓는 수밖에 없다. ***

Binance diff depth 프로토콜 (공식 문서)
  스트림 <symbol>@depth@100ms 는 {U, u, pu, b, a} 를 준다.
    U  = 이 이벤트의 첫 update id
    u  = 마지막 update id
    pu = **직전 이벤트의 u**   <- 연속성 검사에 쓴다 (선물 전용, 현물에는 없다)
  동기화 절차:
    1) 스트림 구독 시작, 이벤트를 버퍼에 쌓는다
    2) REST /fapi/v1/depth?limit=1000 으로 스냅샷 (lastUpdateId)
    3) u < lastUpdateId 인 버퍼 이벤트는 버린다
    4) 첫 적용 이벤트는 U <= lastUpdateId+1 <= u 를 만족해야 한다
    5) 이후 각 이벤트의 pu 가 직전 u 와 같아야 한다. 어긋나면 **재동기화**
  이 절차를 어기면 북이 조용히 틀어진다 — 그래서 pu 검사를 반드시 넣는다.

저장 (2종)
  depth_ws/<날짜>/book_<ts>.parquet   1초 간격 밴드 집계 (누적 명목가)
  depth_ws_flow/<날짜>/flow_<ts>.parquet
      1초 구간의 **추가/취소/체결 소진** 명목가. 이것이 Delta(v,t) 의 원재료다.
      호가가 줄어든 원인을 '체결' 과 '취소' 로 나누려면 aggTrade 가 필요하므로
      여기서는 **순변화만** 기록하고, 체결 분리는 분석 단계에서 aggTrade 와 조인한다.

실행:
    python collectors/depth_ws.py
    python collectors/depth_ws.py --symbols BTCUSDT ETHUSDT --seconds 120
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

WS_BASE = "wss://fstream.binance.com/stream?streams="
REST_DEPTH = "https://fapi.binance.com/fapi/v1/depth"
OUT_BOOK = os.path.join(C.DATA, "depth_ws")
OUT_FLOW = os.path.join(C.DATA, "depth_ws_flow")
SNAP_SEC = 1.0                 # 밴드 집계 주기
FLUSH_SEC = 60
RESYNC_COOLDOWN = 2.0          # 재동기화 최소 간격 (레이트리밋 보호)
STREAMS_PER_CONN = 8           # 연결당 스트림 수 (메시지량 분산)
# 밴드: 근접을 촘촘히. REST 와 달리 도달범위 제약이 없다.
BANDS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075,
         0.01, 0.015, 0.02, 0.03, 0.05, 0.10]
BAND_NAMES = ["b%s" % ("%g" % (b * 100)).replace(".", "_") for b in BANDS]


class Book:
    """한 심볼의 로컬 오더북. 가격->수량 dict 로 유지한다."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_u: int | None = None
        self.synced = False
        self.buf: list[dict] = []
        self.last_resync = 0.0
        self.resyncs = 0
        self.snap_id: int | None = None   # 스냅샷 수신 후 스트림 따라잡기 대기
        # 직전 집계 시점의 밴드값 — 순변화(Delta) 계산용
        self.prev_bands: np.ndarray | None = None

    def apply(self, side: dict, levels) -> None:
        for px_s, qty_s in levels:
            px, qty = float(px_s), float(qty_s)
            if qty == 0.0:
                side.pop(px, None)
            else:
                side[px] = qty

    def handle(self, ev: dict) -> bool:
        """이벤트 적용. 연속성이 깨지면 False 를 돌려 재동기화를 요청한다."""
        if not self.synced:
            # 스냅샷을 이미 받아뒀다면, 스트림이 그 지점을 따라잡는지 본다.
            # (REST 가 스트림보다 앞설 수 있다. 그때 straddle 이벤트를 기다리지 않고
            #  synced=True 로 만들면 pu 가 영원히 안 맞는다 — 갭 4.6% 의 원인이었다)
            if self.snap_id is not None:
                u, Uu = ev.get("u"), ev.get("U")
                if u is None or Uu is None or u < self.snap_id:
                    return True               # 스냅샷보다 과거 — 버린다
                if Uu <= self.snap_id + 1 <= u:
                    self.apply(self.bids, ev.get("b", []))
                    self.apply(self.asks, ev.get("a", []))
                    self.last_u = u
                    self.synced = True
                    self.snap_id = None
                    self.buf.clear()
                    return True
                # 스냅샷 지점을 건너뛴 이벤트 — 스냅샷이 낡았다. 다시 받는다.
                self.snap_id = None
                self.buf = [ev]
                return False
            self.buf.append(ev)
            if len(self.buf) > 5000:          # 스냅샷이 계속 실패하는 상황
                self.buf = self.buf[-2000:]
            return True
        pu, u = ev.get("pu"), ev.get("u")
        if pu is None or u is None:
            return True
        if self.last_u is not None and pu != self.last_u:
            # 갭. **버퍼링을 다시 켜야** REST 스냅샷과 스트림 위치를 맞출 수 있다.
            # (이걸 빼면 재동기화 후 last_u 가 REST 의 lastUpdateId 로 고정되는데
            #  스트림은 그 사이에도 흘러가 pu 가 영원히 안 맞는다 — 갭 99% 의 원인이었다)
            self.synced = False
            self.buf = [ev]
            return False
        self.apply(self.bids, ev.get("b", []))
        self.apply(self.asks, ev.get("a", []))
        self.last_u = u
        return True

    def resync(self, snap: dict) -> None:
        last_id = int(snap["lastUpdateId"])
        self.bids = {float(p): float(q) for p, q in snap["bids"] if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snap["asks"] if float(q) > 0}
        applied = False
        for ev in self.buf:
            u, Uu = ev.get("u"), ev.get("U")
            if u is None or Uu is None or u < last_id:
                continue
            if not applied:
                # 첫 적용 이벤트는 U <= lastUpdateId+1 <= u 를 만족해야 한다
                if not (Uu <= last_id + 1 <= u):
                    continue
                applied = True
            elif self.last_u is not None and ev.get("pu") != self.last_u:
                continue
            self.apply(self.bids, ev.get("b", []))
            self.apply(self.asks, ev.get("a", []))
            self.last_u = u
        self.resyncs += 1
        self.prev_bands = None                # 재동기화 후 Delta 는 끊어 읽는다
        if applied:
            self.buf.clear()
            self.snap_id = None
            self.synced = True
        else:
            # 버퍼에 스냅샷을 straddle 하는 이벤트가 없었다 = REST 가 스트림보다 앞섰다.
            # synced 로 만들지 **않고** snap_id 를 남겨 handle() 이 따라잡게 한다.
            self.buf.clear()
            self.snap_id = last_id
            self.synced = False

    def mid(self) -> float:
        if not self.bids or not self.asks:
            return float("nan")
        return 0.5 * (max(self.bids) + min(self.asks))

    def bands(self) -> np.ndarray:
        """[bid 밴드..., ask 밴드...] 누적 명목가. 도달범위 제약 없음."""
        m = self.mid()
        out = np.full(2 * len(BANDS), np.nan)
        if not np.isfinite(m) or m <= 0:
            return out
        bp = np.fromiter(self.bids.keys(), float, len(self.bids))
        bq = np.fromiter(self.bids.values(), float, len(self.bids))
        ap = np.fromiter(self.asks.keys(), float, len(self.asks))
        aq = np.fromiter(self.asks.values(), float, len(self.asks))
        bn, an = bp * bq, ap * aq
        for i, b in enumerate(BANDS):
            out[i] = float(bn[bp >= m * (1 - b)].sum())
            out[len(BANDS) + i] = float(an[ap <= m * (1 + b)].sum())
        return out


def _snap_sync(session, symbol: str) -> dict:
    # get_json 은 timeout/max_retry/backoff_base 가 키워드 필수다.
    return U.get_json(session, REST_DEPTH, {"symbol": symbol, "limit": 1000},
                      timeout=5.0, max_retry=2, backoff_base=0.5)


async def snapshot(session, symbol: str) -> dict | None:
    try:
        return await asyncio.to_thread(_snap_sync, session, symbol)
    except Exception as e:                    # noqa: BLE001 — 재시도로 처리
        U.log("%s 스냅샷 실패: %s" % (symbol, e))
        return None


def flush(rows: list, cols: list, outdir: str, prefix: str) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=cols)
    day = pd.to_datetime(int(df["ts_ms"].iloc[0]), unit="ms", utc=True).strftime("%Y-%m-%d")
    d = os.path.join(outdir, day)
    os.makedirs(d, exist_ok=True)
    U.atomic_write_parquet(df, os.path.join(d, "%s_%d.parquet" % (prefix, U.utc_now_ms())))
    return len(df)


async def run_conn(session, symbols: list[str], books: dict, stop_at: float,
                   stats: dict) -> None:
    import websockets
    streams = "/".join("%s@depth@100ms" % s.lower() for s in symbols)
    url = WS_BASE + streams
    while time.time() < stop_at:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                          max_queue=4096) as ws:
                for s in symbols:                     # 연결 직후 전부 재동기화
                    books[s].synced = False
                    books[s].buf.clear()
                pending = set(symbols)
                while time.time() < stop_at:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    msg = json.loads(raw)
                    ev = msg.get("data") or msg
                    sym = ev.get("s")
                    if sym is None or sym not in books:
                        continue
                    bk = books[sym]
                    stats["msgs"] += 1
                    if not bk.handle(ev):
                        pending.add(sym)
                        stats["gaps"] += 1
                    if sym in pending and (time.time() - bk.last_resync) > RESYNC_COOLDOWN:
                        bk.last_resync = time.time()
                        snap = await snapshot(session, sym)
                        if snap:
                            bk.resync(snap)
                            pending.discard(sym)
        except asyncio.TimeoutError:
            U.log("WS 무응답 — 재연결 (%s)" % symbols[0])
        except Exception as e:                        # noqa: BLE001
            U.log("WS 오류(%s): %s — 3초 후 재연결" % (symbols[0], e))
            await asyncio.sleep(3)


async def aggregate(books: dict, stop_at: float, stats: dict) -> None:
    bcols = (["ts_ms", "symbol", "mid", "n_bid", "n_ask", "resyncs"]
             + ["bid_" + n for n in BAND_NAMES] + ["ask_" + n for n in BAND_NAMES])
    fcols = (["ts_ms", "symbol", "mid"]
             + ["dbid_" + n for n in BAND_NAMES] + ["dask_" + n for n in BAND_NAMES])
    brows, frows = [], []
    last_flush = time.time()
    while time.time() < stop_at:
        await asyncio.sleep(SNAP_SEC)
        ts = U.utc_now_ms()
        for s, bk in books.items():
            if not bk.synced:
                continue
            v = bk.bands()
            if not np.any(np.isfinite(v)):
                continue
            m = bk.mid()
            brows.append([ts, s, m, len(bk.bids), len(bk.asks), bk.resyncs] + list(v))
            if bk.prev_bands is not None:
                frows.append([ts, s, m] + list(v - bk.prev_bands))
            bk.prev_bands = v
        if time.time() - last_flush >= FLUSH_SEC:
            n1 = flush(brows, bcols, OUT_BOOK, "book")
            n2 = flush(frows, fcols, OUT_FLOW, "flow")
            stats["book"] += n1
            stats["flow"] += n2
            U.log("flush book=%d flow=%d | 누적 book=%d msgs=%d gaps=%d"
                  % (n1, n2, stats["book"], stats["msgs"], stats["gaps"]))
            brows, frows = [], []
            last_flush = time.time()
    flush(brows, bcols, OUT_BOOK, "book")
    flush(frows, fcols, OUT_FLOW, "flow")


async def main_async(symbols: list[str], seconds: float) -> None:
    session = U.make_session("liq-research-depth-ws/1.0")
    books = {s: Book(s) for s in symbols}
    stop_at = time.time() + seconds
    stats = defaultdict(int)
    chunks = [symbols[i:i + STREAMS_PER_CONN]
              for i in range(0, len(symbols), STREAMS_PER_CONN)]
    U.log("depth_ws 시작: %d종 / %d연결 / 밴드 %d개 / %.0f초"
          % (len(symbols), len(chunks), len(BANDS), seconds))
    await asyncio.gather(aggregate(books, stop_at, stats),
                         *[run_conn(session, c, books, stop_at, stats) for c in chunks])
    U.log("종료: book %d행 / flow %d행 / 메시지 %d / 갭 %d"
          % (stats["book"], stats["flow"], stats["msgs"], stats["gaps"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance futures diff-depth local book")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=float("inf"))
    a = ap.parse_args()
    U.init_stdout()
    syms = a.symbols if a.symbols else C.MAJORS
    lock = U.acquire_single_instance(os.path.join(C.DATA, ".locks"), "depth_ws")
    if lock is None:
        U.log("이미 실행 중 — 종료")
        return 0
    try:
        asyncio.run(main_async(syms, a.seconds))
    except KeyboardInterrupt:
        U.log("중단됨")
    return 0


if __name__ == "__main__":
    sys.exit(main())
