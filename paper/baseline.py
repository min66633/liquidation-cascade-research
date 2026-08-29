# -*- coding: utf-8 -*-
"""기준선 전략 페이퍼 트레이딩 — L(p) 없이 돌아가는 완결 전략.

*** 이것은 목표 설계가 아니다. 넘어야 할 기준선이다. ***
목표 설계는 TARGET_DESIGN.md 에 있다 — 가격대별 OI 히트맵 x 오더북으로 캐스케이드
바닥을 확률 추정해 거기 걸고 1~5분 짧게 보유하는 것. 실측 격차는 크다:
    완벽 예지 바닥 진입   1분 +156.1bp (승률 99%) / 15분 +258bp
    현행 트리거 -2%       1분   -9.9bp          / 15분  +88bp
바닥에서 1% 만 벗어나도 수익의 2/3 가 사라진다. 되돌림의 61% 가 첫 1분에 들어온다.
목표 설계를 구현하려면 가격대별 OI(=L(p))가 필요한데 과거 아카이브가 없어
2026-07-31 부터 hl_positions 로 수집 중이다. 그때까지 이 기준선을 실측해 둔다.
**여기 HOLD_MIN=15 를 목표 설계의 '1~5분'으로 바꾸면 안 된다** — 짧은 보유는
정확한 진입과 묶여서만 성립하고, 현행 진입에서 1분 보유는 -9.9bp 로 손실이다.

전략 사양 (백테스트에서 확정된 그대로. 임의 변형 금지)
  이벤트   5분봉 |z| >= K sigma  AND  dOI <= DOI_THR
             z      = 바 수익률 / 직전 288바(1일) 표준편차   (현재 바 제외)
             dOI    = OI[T+5m] / OI[T] - 1                   (바 T 구간의 OI 변화)
             side   = +1 롱청산(급락) -> 아래에 매수 지정가
                      -1 숏청산(급등) -> 위에 매도 지정가
  행동     트리거 바 종가에서 OFFSET 만큼 떨어진 곳에 지정가, TTL 동안 유효
  청산     체결 후 HOLD_MIN 분 경과 시 시장가 청산 (손절 없음)
  제약     무레버리지. 심볼당 동시 1포지션.

백테스트 근거 (analysis/maker_1m.py, k=8σ, offset 2%, 1분봉 체결판정, 15분 보유)
  체결률 28.4% / 조건부 +129.6bp / 승률 70.7% / 이벤트당 ~37bp
  전 연도 양수(2024 +28.7bp), 2025-10-10 제외해도 +89.8bp

보유시간을 15분으로 정한 근거 (2026-08-01 실측, 위와 같은 표본)
  이벤트당 bp / MAE 5%분위 / EV÷|MAE| , offset 2%:
      15분   LIQ 36.8  CTRL  0.0   t=2.6   -984bp   0.0374
      60분   LIQ 40.8  CTRL +3.3   t=2.2  -1370bp   0.0298   <- 지배당함
     240분   LIQ 51.9  CTRL -1.3   t=2.8  -1382bp   0.0376
  CTRL이 세 지평 모두 0 근처이므로 증분은 드리프트가 아니라 청산 채널 고유다.
  그러나 60분은 MAE가 이미 39% 커진 뒤에도 수익이 안 들어와 셋 중 유일하게 지배당한다.
  240분은 EV가 가장 높지만 (a) 4시간 점유가 MIN_GAP_BARS(60분) 후속 이벤트를 막고
  (b) summarize()의 NW가 시간순이 아닌 심볼블록 순서에 걸려 있어 동일자 심볼간 상관을
  못 잡는다 — t 우위가 그 결함에 기대고 있어 채택하지 않았다.
  TTL(주문 유효시간)은 60분 그대로다. maker_1m.py --live-min 60 과 같은 값이며
  보유시간과는 다른 파라미터다. 이전 판에서 이 둘을 혼동해 HOLD_MIN=60 이었다.

이 페이퍼가 검증하려는 것 (백테스트가 못 한 것)
  1) 실제 체결 — 백테스트는 '저가가 닿으면 체결'로 판정했다. 큐 우선순위는 반영 못 한다.
     여기서는 '뚫고 지나감'(관통)과 '닿기만 함'을 나눠 기록해 낙관 편향을 실측한다.
  2) 탐지 지연 — 바 종료 후 OI 갱신까지 걸리는 실제 시간
  3) MAE 실감내 — 5%분위 -790~-1,059bp를 실제로 견딜 수 있는지
  4) 현 레짐의 이벤트 빈도
  5) **용량** — "체결됐다"는 판정은 가격에 대한 진술이지 물량이 아니다.
     체결 시점의 테이커 흐름과 청산 시점의 반대편 호가 깊이를 함께 기록한다.
         용량 = ALPHA x min( 유량 F,  청산시점 깊이 D )
     과거분 실측(analysis/capacity.py, 5종 72건): **99% 가 깊이에 구속**되고,
     이벤트당 중앙 용량은 \\$373K (BTC \\$5.0M ~ 알트 \\$225K, 20배 편차).
     "큰 수익 사건일수록 용량이 작다"는 확인되지 않았다 — 용량가중 − 균등가중 =
     −7.9bp, 일자 클러스터 부트스트랩 95% CI [−75.7, +52.1] 로 0 을 포함한다.

용량 관측의 한계 (실측 2026-07-31)
  depth_poll 은 REST limit=1000 이 최대라 도달 범위가 종목마다 다르다.
      BTCUSDT 0.18%   ETHUSDT 0.56%   SOLUSDT 13.7%   WLDUSDT 99.9%
  따라서 **BTC/ETH 는 -1% 깊이를 실시간 관측할 수 없다.** 여기서는 도달한 것 중
  가장 깊은 밴드를 쓰고 어느 밴드였는지(depth_band)를 함께 남긴다. BTC/ETH 의
  용량 수치는 하한이며 과거분(bookDepth -1%)과 직접 비교하면 안 된다.

주문은 전부 가상이다. 거래소로 나가는 주문은 없다.

실행:
    python paper/baseline.py
    python paper/baseline.py --status        # 현재 상태만 출력
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from collectors.depth_poll import summarize as depth_summarize   # noqa: E402

# ------------------------------------------------------------------ 전략 상수
K_SIGMA = 8.0                 # 이벤트 임계 (백테스트와 동일)
DOI_THR = -0.02               # OI 급감 임계
OFFSET = 0.02                 # 트리거 종가 대비 지정가 거리
TTL_MIN = 60                  # 주문 유효 시간(분) — 백테스트 --live-min 60 과 동일
HOLD_MIN = 15                 # 보유 시간(분) — 위 도스트링의 지평 비교표 참조
COST_BPS = 7.0                # 왕복 비용 가정 (maker-in / taker-out)
VOL_WIN = 288                 # 변동성 창(5분봉 1일)
MIN_GAP_BARS = 12             # 같은 심볼 이벤트 최소 간격(5분봉)
BAR_MS = 300_000
ALPHA = 0.10                  # 흐름/호가의 몇 %까지 차지해도 시장을 안 바꾸는가 (가정)
# depth_poll 밴드 — 깊은 것부터. 도달 못 한 밴드는 NaN 이므로 첫 유효값을 쓴다.
DEPTH_BANDS = ["2_0", "1_0", "0_5", "0_2", "0_1", "0_05"]
DEPTH_MAX_LAG_S = 300.0       # 스냅샷이 이보다 오래되면 쓰지 않는다

PAPER_DIR = os.path.join(C.DATA, "paper")
STATE_PATH = os.path.join(PAPER_DIR, "state.json")

ORDER_COLS = {
    "order_id": "string", "symbol": "string", "side": "int64",
    "signal_ms": "int64", "ref_px": "float64", "limit_px": "float64",
    "z": "float64", "doi": "float64", "sigma": "float64",
    "expire_ms": "int64", "status": "string",
    "fill_ms": "int64", "fill_px": "float64", "touched_only": "bool",
    "exit_ms": "int64", "exit_px": "float64",
    "ret_bp": "float64", "mae_bp": "float64", "detect_lag_s": "float64",
    # 용량
    "flow_fill": "float64",       # 체결 1분봉의 우리를 체결시키는 방향 테이커 대금
    "flow_win": "float64",        # 보유구간 중 지정가 너머에 머문 바들의 같은 흐름 합
    "depth_fill": "float64",      # 체결 시점 반대편(청산할 쪽) 호가 명목가
    "depth_exit": "float64",      # 청산 시점 같은 쪽 호가 명목가  <- 실제 구속 조건
    "depth_band": "string",       # depth_fill 이 나온 밴드 (도달 한계 때문에 종목마다 다름)
    "depth_band_exit": "string",  # depth_exit 이 나온 밴드 (덮어쓰면 안 됨 — 다를 수 있다)
    "depth_lag_s": "float64",     # 청산 시점 스냅샷의 신선도 (직접 조회 성공이면 0)
    "cap_usd": "float64",         # ALPHA x min(flow_win, depth_exit)
}


def _empty(cols: dict) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})


def coerce(rows: list[dict], cols: dict) -> pd.DataFrame:
    if not rows:
        return _empty(cols)
    df = pd.DataFrame(rows)
    for c, dt in cols.items():
        if c not in df.columns:
            df[c] = pd.NA
        try:
            df[c] = df[c].astype(dt)
        except (TypeError, ValueError):
            if dt == "string":
                df[c] = df[c].astype("string")
            elif dt == "bool":
                df[c] = df[c].map({True: True, False: False}).fillna(False).astype("bool")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                if dt == "int64":
                    df[c] = df[c].fillna(-1).astype("int64")
    return df[list(cols.keys())]


# ------------------------------------------------------------------ 데이터
def fetch_klines(session, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """OHLC + 테이커 흐름. qv/tbqv 는 용량의 유량 상한 계산에 쓴다.

    sell_qv = 전체 대금 - 테이커 매수 대금 = 테이커 **매도** 대금.
    우리가 매수 지정가를 깔았으면 이 흐름이 우리를 체결시킨다(반대도 대칭).
    """
    d = U.get_json(session, C.BINANCE_FAPI + "/fapi/v1/klines",
                   {"symbol": symbol, "interval": interval, "limit": limit},
                   timeout=C.BINANCE_HTTP_TIMEOUT_S, max_retry=C.BINANCE_MAX_RETRY,
                   backoff_base=C.BINANCE_BACKOFF_BASE_S)
    if not isinstance(d, list) or not d:
        return pd.DataFrame()
    df = pd.DataFrame(d, columns=["open_time", "open", "high", "low", "close", "volume",
                                  "close_time", "qv", "n", "tbv", "tbqv", "ig"])
    for c in ("open_time", "open", "high", "low", "close", "qv", "tbqv"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open_time", "open", "high", "low", "close", "qv", "tbqv"]].dropna(
        subset=["open_time", "open", "high", "low", "close"])
    df["sell_qv"] = (df["qv"] - df["tbqv"]).clip(lower=0.0)
    df["buy_qv"] = df["tbqv"].clip(lower=0.0)
    return df


def _depth_files(ts_ms: int) -> list[str]:
    """ts_ms 근처에 flush 된 depth_poll 파일들. 파일명이 flush 시각이다.

    depth_poll 은 FLUSH_SEC(120초)마다 쓰고 한 파일이 그 직전 구간을 담으므로,
    ts_ms 이후에 flush 된 파일에 ts_ms 시점 행이 들어 있다.
    """
    out = []
    for off in (-86_400_000, 0, 86_400_000):     # UTC 일 경계를 걸치는 경우 대비
        day = time.strftime("%Y-%m-%d", time.gmtime((ts_ms + off) / 1000.0))
        dd = os.path.join(C.DEPTH_DIR, day)
        if not os.path.isdir(dd):
            continue
        for fn in os.listdir(dd):
            if not (fn.startswith("depth_") and fn.endswith(".parquet")):
                continue
            try:
                fts = int(fn[len("depth_"):-len(".parquet")])
            except ValueError:
                continue
            # 파일은 [fts - flush주기, fts] 구간을 담는다. 여유 있게 잡는다.
            if -600_000 <= fts - ts_ms <= 900_000:
                out.append(os.path.join(dd, fn))
    return sorted(out)


def depth_at(symbol: str, ts_ms: int, side: int) -> tuple[float, str, float]:
    """ts_ms 에 가장 가까운 호가 깊이. 반환 (명목가, 밴드, 지연초).

    side=+1 : 우리가 매수로 진입 -> 청산은 시장가 매도 -> **매수호가**(bid)를 먹는다
    side=-1 : 우리가 매도로 진입 -> 청산은 시장가 매수 -> **매도호가**(ask)를 먹는다
    도달하지 못한 밴드는 depth_poll 이 NaN 으로 남기므로 첫 유효 밴드를 쓴다.
    """
    pre = "bid_" if side == 1 else "ask_"
    frames = []
    for p in _depth_files(ts_ms):
        # 여기서 read_parquet_or_quarantine 을 쓰면 안 된다 — 그 함수는 실패 시 파일을
        # .corrupt 로 **改名**한다. Defender 스캔이나 depth_poll 의 동시 쓰기로 생긴
        # 일시적 PermissionError 에도 멀쩡한 스냅샷을 파괴하게 된다. 여기 파일은
        # 우리 소유가 아니라 남이 쓰는 읽기 전용 자료다.
        try:
            d = pd.read_parquet(p)
        except Exception:                        # noqa: BLE001 — 한 파일 실패는 무시
            continue
        if d.empty or "symbol" not in d.columns:
            continue
        frames.append(d[d["symbol"] == symbol])
    if not frames:
        return float("nan"), "", float("nan")
    d = pd.concat(frames, ignore_index=True)
    if d.empty:
        return float("nan"), "", float("nan")
    lag = (ts_ms - d["ts_ms"]).abs()
    i = int(lag.idxmin())
    lag_s = float(lag.loc[i]) / 1000.0
    if lag_s > DEPTH_MAX_LAG_S:
        return float("nan"), "", lag_s
    for band in DEPTH_BANDS:
        col = pre + band
        if col in d.columns:
            v = d.loc[i, col]
            if pd.notna(v) and float(v) > 0:
                return float(v), band, lag_s
    return float("nan"), "", lag_s


def depth_live(session, symbol: str, side: int) -> tuple[float, str, float]:
    """지금 이 순간의 호가 깊이를 직접 조회.

    청산은 '지금' 시장가로 나가는 것이므로 depth_poll 의 캐시(최대 120초 지연)로는
    캐스케이드 구간에서 의미가 없다 — 깊이가 분 단위로 무너지기 때문이다.
    실패하면 캐시로 폴백한다. 밴드 산출은 depth_poll 과 같은 함수를 재사용한다.
    """
    ts = U.utc_now_ms()
    try:
        # 재시도/타임아웃을 짧게 잡는다. 기본값(3회 x 15초 + 백오프)이면 청산 1건당
        # 최대 52초가 걸려 --interval 60 짜리 루프를 무너뜨린다. 캐시 폴백이 있으므로
        # 느린 정답보다 빠른 NaN 이 낫다.
        book = U.get_json(session, C.BINANCE_FAPI + "/fapi/v1/depth",
                          {"symbol": symbol, "limit": 1000},
                          timeout=3.0, max_retry=1, backoff_base=0.5)
        rec = depth_summarize(symbol, book, ts)
    except Exception as e:                       # noqa: BLE001 — 폴백이 있으므로 삼킨다
        U.log("depth_live %s failed: %s: %s" % (symbol, type(e).__name__, e))
        rec = None
    if rec:
        pre = "bid_" if side == 1 else "ask_"
        for band in DEPTH_BANDS:
            v = rec.get(pre + band)
            if v is not None and v == v and float(v) > 0:
                return float(v), band, 0.0
    return depth_at(symbol, ts, side)


def fetch_oi(session, symbol: str, limit: int = 3) -> pd.DataFrame:
    """5분 OI 이력. 갱신이 늦다 — 실측 탐지 지연 230초(바 하나에 육박)."""
    d = U.get_json(session, C.BINANCE_FAPI + "/futures/data/openInterestHist",
                   {"symbol": symbol, "period": "5m", "limit": limit},
                   timeout=C.BINANCE_HTTP_TIMEOUT_S, max_retry=C.BINANCE_MAX_RETRY,
                   backoff_base=C.BINANCE_BACKOFF_BASE_S)
    if not isinstance(d, list) or not d:
        return pd.DataFrame()
    df = pd.DataFrame(d)
    df["open_time"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["oi"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df[["open_time", "oi"]].dropna()


def fetch_oi_now(session, symbol: str) -> float:
    """현재 OI 즉시 조회. 5분 경계 직후에 찍으면 hist 와 같은 값을 몇 초 만에 얻는다.

    왜 필요한가 — openInterestHist 는 갱신이 느려 실측 탐지 지연이 230초였다.
    되돌림은 5분 대기에 엣지의 90%가 소멸하므로(analysis/event_study_h2.py),
    230초 지연은 백테스트와 사실상 다른 전략을 돌리는 것이 된다.
    """
    d = U.get_json(session, C.BINANCE_FAPI + "/fapi/v1/openInterest",
                   {"symbol": symbol},
                   timeout=C.BINANCE_HTTP_TIMEOUT_S, max_retry=C.BINANCE_MAX_RETRY,
                   backoff_base=C.BINANCE_BACKOFF_BASE_S)
    if not isinstance(d, dict):
        return float("nan")
    return U.to_float(d.get("openInterest"))


# ------------------------------------------------------------------ 엔진
class Paper:
    def __init__(self, symbols: list[str], k: float = K_SIGMA, offset: float = OFFSET,
                 ttl_min: int = TTL_MIN, hold_min: int = HOLD_MIN,
                 paper_dir: str = PAPER_DIR) -> None:
        self.symbols = symbols
        # 사양은 상수로 고정하되, 스모크 테스트용 오버라이드를 허용한다.
        # 운영에서는 기본값(백테스트와 동일)만 쓴다.
        self.k = k
        self.offset = offset
        self.ttl_min = ttl_min
        self.hold_min = hold_min
        self.doi_thr = DOI_THR
        self.verbose = False
        self.dir = paper_dir
        self.session = U.make_session(C.USER_AGENT)
        self.orders: list[dict] = []
        self.last_event_bar: dict[str, int] = {}
        # 5분 경계별 실시간 OI 스냅샷 {symbol: {bar_open_ms: oi}}
        self.oi_snap: dict[str, dict[int, float]] = {}
        self.last_boundary = -1
        self.seq = 0
        os.makedirs(self.dir, exist_ok=True)
        self._load()

    # -------------------------------------------------------- 상태 저장/복구
    def _load(self) -> None:
        d = U.read_parquet_or_quarantine(os.path.join(self.dir, "orders.parquet"))
        if d is not None and not d.empty:
            self.orders = d.to_dict("records")
            self.seq = len(self.orders)
            U.log("state: %d orders restored" % len(self.orders))
        try:
            with open(os.path.join(self.dir, "state.json"), encoding="utf-8") as f:
                self.last_event_bar = {k: int(v) for k, v
                                       in json.load(f).get("last_event_bar", {}).items()}
        except (OSError, ValueError, TypeError):
            pass

    def _save(self) -> None:
        U.atomic_write_parquet(coerce(self.orders, ORDER_COLS),
                               os.path.join(self.dir, "orders.parquet"))
        sp = os.path.join(self.dir, "state.json")
        tmp = sp + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"last_event_bar": self.last_event_bar}, f)
            os.replace(tmp, sp)
        except OSError as e:
            U.log("state save failed: %s" % e)

    # -------------------------------------------------------- OI 경계 스냅샷
    def snap_oi(self) -> int:
        """5분 경계를 막 넘었으면 전 심볼 OI를 즉시 찍는다.

        openInterestHist 는 갱신이 느려(실측 230초) 그대로 쓰면 백테스트와 다른
        전략이 된다. 경계 직후 실시간 OI 를 직접 찍어 지연을 초 단위로 줄인다.
        반환: 이번에 찍은 경계 타임스탬프 (안 찍었으면 -1)
        """
        now = U.utc_now_ms()
        boundary = (now // BAR_MS) * BAR_MS
        if boundary == self.last_boundary:
            return -1
        # 경계 직후 몇 초는 거래소 집계가 안정될 시간을 준다
        if now - boundary < 3000:
            return -1
        self.last_boundary = boundary
        t0 = time.monotonic()
        n = 0
        for sym in self.symbols:
            v = fetch_oi_now(self.session, sym)
            if v == v and v > 0:                      # NaN 아님
                self.oi_snap.setdefault(sym, {})[boundary] = float(v)
                n += 1
            time.sleep(0.05)
        # 오래된 스냅샷 정리 (하루치면 충분)
        cutoff = boundary - 288 * BAR_MS
        for sym in self.oi_snap:
            self.oi_snap[sym] = {k: v for k, v in self.oi_snap[sym].items() if k >= cutoff}
        U.log("oi snapshot @%d: %d/%d symbols in %.1fs"
              % (boundary, n, len(self.symbols), time.monotonic() - t0))
        return boundary

    # -------------------------------------------------------- 신호
    def scan(self, symbol: str) -> None:
        """직전에 닫힌 5분봉으로 이벤트를 판정한다. 룩어헤드 없음."""
        k = fetch_klines(self.session, symbol, "5m", VOL_WIN + 5)
        if len(k) < VOL_WIN + 2:
            return
        # 마지막 행은 미완성 봉 -> 버린다
        k = k.iloc[:-1].reset_index(drop=True)
        oi = fetch_oi(self.session, symbol, limit=3)
        if oi.empty or len(oi) < 2:
            return

        bar = int(k["open_time"].iloc[-1])            # 방금 닫힌 봉
        if self.last_event_bar.get(symbol, -1) >= bar - MIN_GAP_BARS * BAR_MS:
            return
        # 사양의 '심볼당 동시 1포지션'을 실제로 강제한다. last_event_bar 만으로는
        # MIN_GAP(60분) == TTL(60분) 이라 미체결 주문과 새 신호가 겹칠 수 있다.
        if any(o["symbol"] == symbol and o["status"] in ("open", "filled")
               for o in self.orders):
            return

        ret = k["close"] / k["open"] - 1.0
        sigma = float(ret.iloc[:-1].tail(VOL_WIN).std())   # 현재 봉 제외
        if not np.isfinite(sigma) or sigma <= 0:
            return
        r = float(ret.iloc[-1])
        z = r / sigma

        # dOI = OI[bar+5m] / OI[bar] - 1.
        # 1순위: 우리가 5분 경계마다 직접 찍어둔 실시간 OI (지연 수 초)
        # 2순위: openInterestHist (지연 실측 230초 — 되돌림이 5분이면 90% 소멸하므로
        #        이걸로 돌리면 백테스트와 다른 전략을 검증하는 셈이 된다)
        snap = self.oi_snap.get(symbol, {})
        o0 = snap.get(bar)
        o1 = snap.get(bar + BAR_MS)
        src = "live"
        if o0 is None or o1 is None:
            o = oi.set_index("open_time")["oi"]
            if bar not in o.index or (bar + BAR_MS) not in o.index:
                return
            o0, o1 = float(o[bar]), float(o[bar + BAR_MS])
            src = "hist"
        if not (o0 and o0 > 0):
            return
        doi = o1 / o0 - 1.0

        if self.verbose:
            U.log("  scan %-9s z=%+6.2f dOI=%+.4f  (need |z|>=%.1f, dOI<=%.3f)"
                  % (symbol, z, doi, self.k, self.doi_thr))
        if abs(z) < self.k or doi > self.doi_thr:
            return

        side = 1 if z < 0 else -1
        ref = float(k["close"].iloc[-1])
        limit_px = ref * (1 - self.offset) if side == 1 else ref * (1 + self.offset)
        now = U.utc_now_ms()
        self.seq += 1
        self.orders.append({
            "order_id": "%s-%d" % (symbol, self.seq),
            "symbol": symbol, "side": side, "signal_ms": bar + BAR_MS,
            "ref_px": ref, "limit_px": limit_px, "z": z, "doi": doi, "sigma": sigma,
            "expire_ms": now + self.ttl_min * 60_000, "status": "open",
            "fill_ms": -1, "fill_px": np.nan, "touched_only": False,
            "exit_ms": -1, "exit_px": np.nan,
            "ret_bp": np.nan, "mae_bp": np.nan,
            "detect_lag_s": (now - (bar + BAR_MS)) / 1000.0,
        })
        self.last_event_bar[symbol] = bar
        U.log("SIGNAL %s side=%+d z=%.1f dOI=%.3f[%s] ref=%.6f limit=%.6f lag=%.0fs"
              % (symbol, side, z, doi, src, ref, limit_px, (now - (bar + BAR_MS)) / 1000.0))

    # -------------------------------------------------------- 체결/청산
    def update(self, symbol: str) -> None:
        live = [o for o in self.orders
                if o["symbol"] == symbol and o["status"] in ("open", "filled")]
        if not live:
            return
        k = fetch_klines(self.session, symbol, "1m", 200)
        if k.empty:
            return
        ot = k["open_time"].to_numpy()
        low = k["low"].to_numpy()
        high = k["high"].to_numpy()
        close = k["close"].to_numpy()
        sell_qv = k["sell_qv"].to_numpy()
        buy_qv = k["buy_qv"].to_numpy()
        now = U.utc_now_ms()

        for o in live:
            if o["status"] == "open":
                if now > o["expire_ms"]:
                    o["status"] = "expired"
                    U.log("EXPIRE %s" % o["order_id"])
                    continue
                sel = ot >= o["signal_ms"]
                if not sel.any():
                    continue
                lp = o["limit_px"]
                hit = ((low[sel] <= lp) if o["side"] == 1 else (high[sel] >= lp))
                if hit.any():
                    idx = int(np.flatnonzero(sel)[int(np.argmax(hit))])
                    o["status"] = "filled"
                    o["fill_ms"] = int(ot[idx])
                    o["fill_px"] = lp
                    # 관통 여부: 저가가 지정가를 넘어섰나(닿기만 했나)
                    o["touched_only"] = bool(
                        (low[idx] >= lp * 0.9999) if o["side"] == 1
                        else (high[idx] <= lp * 1.0001))
                    # 용량 — 우리를 체결시킨 방향의 테이커 대금
                    o["flow_fill"] = float(sell_qv[idx] if o["side"] == 1 else buy_qv[idx])
                    dv, band, lag = depth_at(o["symbol"], int(ot[idx]), int(o["side"]))
                    o["depth_fill"] = dv
                    o["depth_band"] = band
                    U.log("FILL   %s @ %.6f%s  flow=%.2fM depth=%.2fM[%s]"
                          % (o["order_id"], lp,
                             "  [touched-only]" if o["touched_only"] else "",
                             o["flow_fill"] / 1e6,
                             dv / 1e6 if dv == dv else float("nan"), band or "-"))
            elif o["status"] == "filled":
                if now < o["fill_ms"] + self.hold_min * 60_000:
                    continue
                # 창의 **상한**이 있어야 한다. 없으면 재기동/정체로 루프가 늦어졌을 때
                # 보유기간 밖의 움직임이 MAE 와 수익에 섞인다(리뷰 실측: MAE 280배 오차).
                end_ms = o["fill_ms"] + self.hold_min * 60_000
                sel = (ot >= o["fill_ms"]) & (ot <= end_ms)
                if not sel.any():
                    continue
                seg_lo, seg_hi = low[sel], high[sel]
                px = float(close[sel][-1])
                o["status"] = "closed"
                o["exit_ms"] = int(ot[sel][-1]) + 60_000
                o["exit_px"] = px
                if now > end_ms + 120_000:
                    U.log("LATE EXIT %s: %.0fs 늦게 청산 판정 (창은 잘라 씀)"
                          % (o["order_id"], (now - end_ms) / 1000.0))
                e = o["fill_px"]
                o["ret_bp"] = 1e4 * ((px / e - 1.0) * o["side"]) - COST_BPS
                o["mae_bp"] = 1e4 * ((seg_lo.min() / e - 1.0) if o["side"] == 1
                                     else (1.0 - seg_hi.max() / e))
                # 용량 — 유량은 지정가 너머에 머문 바들만, 깊이는 청산 시점 기준
                beyond = (seg_lo <= e) if o["side"] == 1 else (seg_hi >= e)
                flow = (sell_qv if o["side"] == 1 else buy_qv)[sel]
                o["flow_win"] = float(np.nansum(np.where(beyond, flow, 0.0)))
                dv, band, lag = depth_live(self.session, o["symbol"], int(o["side"]))
                o["depth_exit"] = dv
                o["depth_lag_s"] = lag
                o["depth_band_exit"] = band
                o["cap_usd"] = (ALPHA * min(o["flow_win"], dv)
                                if dv == dv and dv > 0 else np.nan)
                U.log("CLOSE  %s ret=%+.1fbp mae=%.1fbp cap=%s"
                      % (o["order_id"], o["ret_bp"], o["mae_bp"],
                         ("$%.0fK" % (o["cap_usd"] / 1e3)) if o["cap_usd"] == o["cap_usd"]
                         else "n/a"))

    # -------------------------------------------------------- 보고
    def report(self) -> None:
        d = coerce(self.orders, ORDER_COLS)
        if d.empty:
            print("주문 없음")
            return
        n_ev = len(d)
        n_fill = int((d.status.isin(["filled", "closed"])).sum())
        cl = d[d.status == "closed"]
        print("\n=== 페이퍼 현황 ===")
        print("이벤트 %d | 체결 %d (%.0f%%) | 청산완료 %d | 진행중 %d"
              % (n_ev, n_fill, 100 * n_fill / max(n_ev, 1), len(cl),
                 int((d.status.isin(["open", "filled"])).sum())))
        if len(cl):
            r = cl["ret_bp"].astype("float64")
            print("평균 %+.1fbp | 중앙 %+.1fbp | 승률 %.0f%% | MAE 5%%분위 %.0fbp"
                  % (r.mean(), r.median(), 100 * (r > 0).mean(),
                     np.nanpercentile(cl["mae_bp"].astype("float64"), 5)))
            print("관통체결 %d / 닿기만 %d" % (int((~cl.touched_only).sum()),
                                              int(cl.touched_only.sum())))
        if n_ev:
            print("탐지 지연 중앙 %.0f초" % d["detect_lag_s"].astype("float64").median())

        # ---- 용량
        cap = cl[cl["cap_usd"].notna()] if len(cl) else cl
        if len(cap):
            c = cap["cap_usd"].astype("float64")
            fw = cap["flow_win"].astype("float64")
            de = cap["depth_exit"].astype("float64")
            print("\n=== 용량 (alpha=%.2f) ===" % ALPHA)
            print("측정 %d/%d건 | 이벤트당 p10 $%.0fK  중앙 $%.0fK  p90 $%.0fK"
                  % (len(cap), len(cl), np.nanpercentile(c, 10) / 1e3,
                     np.nanmedian(c) / 1e3, np.nanpercentile(c, 90) / 1e3))
            print("유량 중앙 $%.1fM | 청산시점 깊이 중앙 $%.1fM | 구속: 깊이 %.0f%%"
                  % (np.nanmedian(fw) / 1e6, np.nanmedian(de) / 1e6,
                     100 * float((de < fw).mean())))
            print("청산시점 깊이 밴드: %s   (REST limit=1000 도달 한계. BTC/ETH 는 하한값이라"
                  % cap["depth_band_exit"].astype("string").value_counts().to_dict())
            print("  종목 간 용량을 그대로 합치면 안 되고 capacity.py 의 -1% 수치와도 다르다)")
            print("스냅샷 지연 중앙 %.0f초" % np.nanmedian(cap["depth_lag_s"].astype("float64")))
            w = c.to_numpy(dtype="float64")
            rr = cap["ret_bp"].astype("float64").to_numpy()
            ok = np.isfinite(w) & np.isfinite(rr) & (w > 0)
            if ok.sum() >= 3:
                print("균등 가중 %+.1fbp  vs  용량 가중 %+.1fbp"
                      % (rr[ok].mean(), float((w[ok] * rr[ok]).sum() / w[ok].sum())))
        elif len(cl):
            print("\n용량: 아직 측정된 건 없음 (depth_poll 스냅샷 매칭 실패)")


def main() -> int:
    ap = argparse.ArgumentParser(description="baseline paper trading (no real orders)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--interval", type=float, default=60.0, help="scan loop seconds")
    ap.add_argument("--status", action="store_true", help="print status and exit")
    ap.add_argument("--k", type=float, default=K_SIGMA, help="TEST ONLY: sigma threshold")
    ap.add_argument("--offset", type=float, default=OFFSET, help="TEST ONLY: limit offset")
    ap.add_argument("--ttl-min", type=int, default=TTL_MIN)
    ap.add_argument("--hold-min", type=int, default=HOLD_MIN)
    ap.add_argument("--paper-dir", default=PAPER_DIR, help="TEST ONLY: separate output dir")
    ap.add_argument("--doi", type=float, default=DOI_THR, help="TEST ONLY: OI drop threshold")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    symbols = a.symbols if a.symbols else C.MAJORS
    p = Paper(symbols, a.k, a.offset, a.ttl_min, a.hold_min, a.paper_dir)
    p.doi_thr = a.doi
    p.verbose = a.verbose
    if (a.k, a.offset) != (K_SIGMA, OFFSET):
        U.log("WARNING: non-spec parameters (k=%.1f offset=%.3f) — smoke test only"
              % (a.k, a.offset))
    if a.status:
        p.report()
        return 0

    lock = U.acquire_single_instance(C.LOGS, "paper_baseline")
    if lock is None:
        U.log("another paper instance is already running -> exit")
        return 0

    U.log("paper baseline start: %d symbols | k=%.1f offset=%.1f%% ttl=%dm hold=%dm | dir=%s"
          % (len(symbols), a.k, 100 * a.offset, a.ttl_min, a.hold_min,
             os.path.basename(a.paper_dir)))
    try:
        while True:
            slot = time.monotonic()
            # 1) 5분 경계면 전 심볼 OI 를 먼저 찍는다 (지연 최소화)
            try:
                fresh = p.snap_oi()
            except Exception as e:
                U.log("snap_oi failed: %s: %s" % (type(e).__name__, e))
                fresh = -1
            # 2) 체결/청산은 매 사이클, 신호 판정은 경계 직후에만
            for sym in symbols:
                try:
                    p.update(sym)
                    if fresh > 0:
                        p.scan(sym)
                except Exception as e:
                    U.log("%s failed: %s: %s" % (sym, type(e).__name__, e))
                time.sleep(0.2)
            p._save()
            el = time.monotonic() - slot
            if el < a.interval:
                time.sleep(a.interval - el)
    except KeyboardInterrupt:
        U.log("interrupted")
    finally:
        p._save()
        p.report()
        p.session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
