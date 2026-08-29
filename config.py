# -*- coding: utf-8 -*-
"""청산 캐스케이드 연구 — 공통 설정.

핵심 상수는 2026-07-31 실측값에 근거한다(README.md의 '실측 검증' 절 참조).
  - HL info 엔드포인트 가중치 상한 1200/분, clearinghouseState = weight 2 → 600주소/분
  - 리더보드 40,976주소 중 상위 1,000계좌가 총 계좌가치의 87.5%
  - batchClearinghouseStates는 공개 API에서 HTTP 500 (노드 RPC 전용) → 미사용
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")

# ------------------------------------------------------------------ Hyperliquid
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

HL_WEIGHT_PER_MIN = 1200          # IP당 info 가중치 상한 (공식 문서)
HL_CLEARINGHOUSE_WEIGHT = 2       # clearinghouseState 1회당 가중치
HL_SAFETY = 0.80                  # 안전마진 — 상한의 80%만 사용

# 요청 간 최소 간격(초). 600/분 * 0.8 = 480/분 → 0.125초
HL_REQ_INTERVAL_S = 60.0 / (HL_WEIGHT_PER_MIN / HL_CLEARINGHOUSE_WEIGHT * HL_SAFETY)

# 커버리지 vs 신선도 트레이드오프 (2026-07-31 실측 근거로 커버리지 쪽을 택함)
#
# top_n=1000 / 300초일 때 실측: 스윕 명목가 $3.39B. HL 전체 OI는 $6.72B이고 포지션은
# 롱+숏 양변이라 OI의 2배이므로, 실제로 잡은 것은 HL 포지션의 약 22%였다.
# BTC perp OI에서 HL 비중이 15%(Binance 47 / Bybit 24.5 / OKX 13.5)이므로,
# 연쇄하면 지도가 시장 전체 포지션의 약 3%만 본 셈이다 — 그대로 쓸 수 없다.
#
# 반대로 지도는 매우 안정적이다: 40분 8스윕에서 가격축 프로파일 상관 0.97~1.00,
# 총 연료 불변, 빈 구간 22/50개가 계속 빔. 즉 신선도를 커버리지와 바꿔도 손해가 적다.
# 캐스케이드 '중'의 빠른 갱신은 Bybit 실현 청산 피드가 대신 담당한다.
#
# 4000계좌 x 0.125초 = 약 500초 소요. 900초 간격이면 마감(810초) 안에 여유 있게 들어간다.
HL_TOP_N = 4000
HL_SWEEP_INTERVAL_S = 900
HL_UNIVERSE_MAX_AGE_S = 86400     # 유니버스 갱신 주기(초)

# 탐색 예산: 매 스윕 슬롯의 일부를 랭킹 밖(꼬리) 주소 순회에 할당한다.
# 이유 — 계좌가치 랭킹만 쓰면 '자본은 작지만 고레버리지'인 계좌를 영원히 못 본다.
# 관측되지 않으면 observed_ntl이 생기지 않아 랭킹에 진입할 수도 없는 순환 문제가
# 생기므로, 꼬리를 회전 스캔해 청산 연료를 실제로 들고 있는 계좌를 발굴한다.
# 1000슬롯 * 0.15 = 150개/스윕 → 약 4만 주소 1바퀴에 22시간(5분 간격 기준).
HL_EXPLORE_FRAC = 0.15
HL_OBSERVED_LOOKBACK_FILES = 300  # 관측 랭킹 산출 시 훑을 최근 스냅샷 수(약 25시간)

# 랭킹 하한 가중치. rank_key = max(관측명목가, 계좌가치 * 이 값).
# 이유 — 지금 포지션이 없는 대형 계좌를 관측명목가 0으로 강등하면 폴링 대상에서
# 빠지고, 빠지면 다시 관측될 수 없어 영구 사각지대가 되는 자기강화 루프가 생긴다.
# (실측: 관측된 46계좌 중 35개가 순간 플랫. $4.6B 계좌가 랭킹 2위→37,273위로 추락.)
# 플랫 고래는 청산 연료가 아니지만 '언제든 될 수 있는' 후보이므로 하한만 준다.
HL_FLAT_ACCOUNT_WEIGHT = 0.25

# 유니버스 갱신 '시도' 최소 간격. 성공이 아니라 시도 기준으로 막아야 한다 —
# 33MB 다운로드 후 저장에서 실패하면 매 스윕(300초)마다 재다운로드해 하루 9.5GB가 된다.
HL_UNIVERSE_RETRY_MIN_S = 1800

# 스윕 벽시계 마감 = 스윕 간격의 이 비율. 초과 시 남은 주소를 건너뛰고 untrusted 처리.
# 이유 — 주소당 최악 55초(3회 재시도 x 15초 타임아웃 + 백오프)라 1,000주소면
# 이론상 한 스윕이 15시간까지 늘어난다. 프로세스는 살아있고 로그도 조용하다.
HL_SWEEP_DEADLINE_FRAC = 0.90
HL_HTTP_TIMEOUT_S = 15
HL_MAX_RETRY = 3
HL_BACKOFF_BASE_S = 1.5           # 재시도 지수 백오프 기저

# 스윕 실패율이 이 값을 넘으면 해당 스윕을 신뢰 불가로 표시한다.
HL_MAX_FAIL_RATE = 0.30

# 2계층 스윕 — 깊은 스윕(전체 지도)과 핫 스윕(근접 연료)을 분리한다.
#
# 왜: 전체 4000계좌 스윕은 레이트리밋상 물리적으로 500초가 걸린다. 간격을 줄이려 해도
# 상한에 막힌다. 그런데 변동성 급등 시 빠르게 변하는 것은 현재가 근처 연료뿐이고,
# 먼 연료는 천천히 변한다. 실측(2026-07-31): 청산가가 현재가 ±20% 안에 있는 주소는
# 151개(19초), ±30%면 241개(30초) — 전체의 4~6% 비용으로 근접 연료를 추적할 수 있다.
#
# 핫리스트는 매 사이클 '현재 mark' 기준으로 다시 계산한다. 가격이 밀리면 멀리 있던
# 포지션이 자동으로 핫리스트에 들어온다.
HL_HOT_BAND_PCT = 25.0            # 현재가 대비 이 % 이내에 청산가가 있으면 핫
HL_HOT_INTERVAL_S = 60            # 핫 스윕 주기(초)
HL_HOT_MAX_ADDR = 600             # 폭주 방지 상한(변동성 급등 시 핫리스트가 커진다)

HL_DIR_POSITIONS = os.path.join(DATA, "hl_positions")
HL_DIR_HOT = os.path.join(DATA, "hl_hot")
HL_DIR_ACCOUNTS = os.path.join(DATA, "hl_accounts")
HL_DIR_MIDS = os.path.join(DATA, "hl_mids")
HL_DIR_UNIVERSE = os.path.join(DATA, "hl_universe")
HL_DIR_SWEEPLOG = os.path.join(DATA, "hl_sweeplog")

# ------------------------------------------------------------------ Binance OI
# /futures/data/* 계열은 전부 '최근 30일'만 보관한다 → 지금부터 적재해야 영구 보존.
BINANCE_FAPI = "https://fapi.binance.com"

# 연구 유니버스 — 순수 크립토 메이저 21종.
# 토큰화 주식(SOXL/SKHYNIX/KORU/EWY/SNDK/MU/SPCX)과 상품(XAU/XAG), 저유동성 밈,
# 거래소 자체 토큰(HYPE)은 제외한다. 미시구조가 달라 표본을 오염시킨다.
MAJORS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
    "DOTUSDT", "TRXUSDT", "BCHUSDT", "UNIUSDT", "NEARUSDT",
    "ATOMUSDT", "ETCUSDT", "FILUSDT", "SUIUSDT", "AAVEUSDT",
    "WLDUSDT",
]

# 6년치 5m/1m 봉과 metrics를 전부 받아둔 심볼. 장기 이벤트 스터디(H2, maker)의 기본값.
# 21종 전체로 늘리면 1분봉만 3GB 가까이 되므로, 장기 분석은 이 5종으로 유지하고
# Tardis 청산 기반 분석(impact/continuity)만 21종으로 넓힌다.
FULL_HISTORY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

BINANCE_SYMBOLS = MAJORS          # 실시간 OI 수집 대상
BINANCE_PERIOD = "5m"
BINANCE_LIMIT = 500               # 5m * 500 = 약 41시간 — 중복 구간은 병합 시 제거
BINANCE_POLL_INTERVAL_S = 300
BINANCE_HTTP_TIMEOUT_S = 15
BINANCE_MAX_RETRY = 3
BINANCE_BACKOFF_BASE_S = 2.0
BINANCE_REQ_INTERVAL_S = 0.25     # 요청 간 페이싱

# (하위폴더명, 엔드포인트경로, 타임스탬프필드, 값필드들)
BINANCE_SERIES = [
    ("open_interest_hist", "/futures/data/openInterestHist", "timestamp",
     ["sumOpenInterest", "sumOpenInterestValue"]),
    ("top_ls_position_ratio", "/futures/data/topLongShortPositionRatio", "timestamp",
     ["longShortRatio", "longAccount", "shortAccount"]),
    ("top_ls_account_ratio", "/futures/data/topLongShortAccountRatio", "timestamp",
     ["longShortRatio", "longAccount", "shortAccount"]),
    ("global_ls_account_ratio", "/futures/data/globalLongShortAccountRatio", "timestamp",
     ["longShortRatio", "longAccount", "shortAccount"]),
    ("taker_ls_ratio", "/futures/data/takerlongshortRatio", "timestamp",
     ["buySellRatio", "buyVol", "sellVol"]),
]

BINANCE_DIR = os.path.join(DATA, "binance_futures_data")

# ------------------------------------------------------------------ Bybit 청산
# 실측(2026-07-31): Binance !forceOrder@arr 전종목 90초 구독 -> 0건.
# Bybit allLiquidation 4종목 90초 -> 380 레코드. 가격+크기 전건 제공.
# 히스토리가 없으므로 지금부터 축적해야 한다.
# 심볼은 기동 시 거래대금 상위로 동적 선택한다(고정 목록은 폴백).
# 이유 — 대표성 검정에 필요한 것은 거래대금이 아니라 '독립 관측 수'다. HL 지도가
# 177개 코인을 덮는데 15개만 수집하면 짝지을 코인이 9개뿐이라 검정력이 안 나온다.
# 실측: HL 177코인 중 170개가 Bybit에 대응 종목이 있다.
BYBIT_TOP_N = 150
BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers?category=linear"
BYBIT_SYMBOLS = [                      # 티커 조회 실패 시 폴백
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "LTCUSDT", "TRXUSDT", "DOTUSDT", "APTUSDT", "NEARUSDT",
]
BYBIT_LIQ_DIR = os.path.join(DATA, "bybit_liq")

# ------------------------------------------------------------------ 호가 깊이
# 충격의 분모. 실측(2026-07-31): 청산 물량이 대기 지정매수의 0.1~1%(비율 중앙 0.0016,
# 최대 0.31)이고 통상 구간에서는 가격을 밀지 못한다(모든 분모에서 R2~0.000).
# 모델 충격항이 임계형 max(0, V/D - c)^beta 가 되었으므로 D 없이는 임계 초과를
# 판정할 수 없다. 과거분은 data.binance.vision/bookDepth 로 T-1 까지 받을 수 있지만
# 극단 이벤트(연 수 회)는 실시간으로 잡지 않으면 그 회차가 영구 결손이다.
DEPTH_DIR = os.path.join(DATA, "depth")

USER_AGENT = "liq-cascade-research/0.1 (academic; contact: minuk3614@gmail.com)"


def ensure_dirs() -> None:
    """적재 디렉터리 일괄 생성."""
    dirs = [DATA, LOGS, HL_DIR_POSITIONS, HL_DIR_HOT, HL_DIR_ACCOUNTS, HL_DIR_MIDS,
            HL_DIR_UNIVERSE, HL_DIR_SWEEPLOG, BINANCE_DIR, BYBIT_LIQ_DIR,
            DEPTH_DIR]
    for name, _, _, _ in BINANCE_SERIES:
        dirs.append(os.path.join(BINANCE_DIR, name))
    for d in dirs:
        os.makedirs(d, exist_ok=True)
