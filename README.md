# 청산 캐스케이드 연구

강제청산이 만드는 가격 움직임과 그 되돌림을 정량화한다.

> # ⛔ 연구 종료 — 2026-08-22
>
> ## → **[`FINAL_REPORT.md`](FINAL_REPORT.md) 부터 읽으세요.** 결론과 근거가 그 한 문서에 닫혀 있습니다.
>
> **판정**: 우위는 실재하나(워크포워드 OOS 건당 **15.5bp**, $t$=2.4)
> **호가창 깊이보다 작다.** 급락 순간 매도호가 1% 이내 중앙 **\$983K**,
> \$100K 시장가 진입만으로 왕복 슬리피지 **8.9bp**.
> 무레버리지 피크자본 기준 **연 3.4%** 가 상한 — 자본 투입 근거 없음.
>
> **설계 부품 3개(①청산맵 ②오더북 깊이 ③유입·취소)는 전부 기여 0.**
> 남은 것은 "15분에 3% 넘게 떨어지면 산다" 하나.
>
> 상시 수집 로거 8종 전부 중지·비활성화 (`C:\Quant\polymarket\start_loggers.py`).
> **쌓인 데이터는 삭제하지 않았다** — `data/` 그대로. 되살리려면 LIQ 8종 `False` → `True`.
>
> **아래 내용은 종료 전에 쓰인 것이라 상당수가 낡았다.** 수치가 충돌하면
> `FINAL_REPORT.md` → `STATUS.md` 뒤쪽 절 순으로 우선한다.

> ## ⚠ 설계의 정본은 `TARGET_DESIGN.md`
>
> **목표 설계**: 가격대별 OI 히트맵 × 가격대별 오더북을 **실시간 대조**해
> 캐스케이드 바닥을 **확률적으로 근사** → 거기 지정매수 → 체결 시 **1~5분 짧게 보유**.
>
> **근거**: 바닥에서 1%만 벗어나도 수익의 **2/3가 사라진다**(1분 156 → 56bp).
> 현행 고정 2% 진입은 1분 **−9.9bp**. **진입 정밀도가 전부다.**
>
> ~~완벽 예지 1분 +156bp(승률 99%)~~ — 이 절대수치는 **2026-08-02 플라시보로
> 대부분 기각**. 청산 고유 기여 **16%**, 승률 99%는 **정의 인공물**(무작위도 81%).
>
> **막는 것 하나**: 가격대별 OI가 과거 데이터에 없다.
> Hyperliquid `liquidationPx` 뿐이고 아카이브가 없어 **2026-07-31부터 수집 중**.
> 단 **들어갈 자리는 특정됐다** — 조건부 분포의 $\mu$ 하나(`PROB_MODEL.md` §6).
>
> **구조식은 확정**: $\log X = a + \gamma\log(V/D)$, $\gamma = 0.875 \pm 0.078$,
> 표본 외 $R^2 = 0.338$ (21종 639이벤트). **오더북 모양에서 독립 추정한 0.861 과 일치.**
> 사전 관측치만으로는 0.014.
>
> **단 "예측력의 전부가 $V$ 에 있다"로 읽지 말 것** — $D$ 를 관측 상수로 놓았을 때만
> 그렇다. 실제 흡수 깊이는 $D_t\cdot W$ 이고 호가는 캐스케이드 중 변한다
> (`PROB_MODEL.md` §9). 그리고 **HL 실측 청산맵 경로는 70% 공백으로 막혔다** —
> 지금은 공개 데이터로 지도를 **재구성**하는 대안 A 로 간다 (`TARGET_DESIGN.md` §3.2).
>
> 지금 돌아가는 페이퍼(고정 2% / 15분)는 **넘어야 할 기준선**이지 목표 설계가 아니다.
> **이미 실패한 시도는 `TARGET_DESIGN.md` §5에 있다 — 반복하지 말 것.**
> 단 그중 `predict_x.py` 는 **점 추정**이 실패한 것이지 확률모델이 실패한 게 아니다.
**모델 v2 (2026-07-31 개정)** — v1의 핵심 항이 실측으로 기각되어 인과 구조를 바꿨다.

| 문서 | 내용 |
|---|---|
| **`STATUS.md`** | **지금 위치 한 장.** 무엇이 검정됐고 무엇이 대용품이었나 — **먼저 읽을 것** |
| **`TARGET_DESIGN.md`** | **설계의 정본.** 만들려는 것, 실패 목록 |
| **`PROB_MODEL.md`** | **확률모델의 정본.** 점 추정과의 차이, 구조식, $L(p)$ 삽입점, 합격 기준 |
| `MODEL.md` / `MODEL.png` | 측정된 것의 정본 — 수식 정의, 모수, Q1~Q4. **확률은 전부 무조건부** |
| `MATH.png` | 수식 해설판 (읽는 법 / 기호 / 실측) |
| `EXPLAINER.md` | 비전공자용 전체 설명서 |
| `MECHANISM.md` | 현행 기준선의 로직 — 시각 t에 무엇을 계산해 무슨 결정을 하는가 |
| `analysis/*_FINDINGS.md` | 개별 검정 결과 원본 |
| `RESEARCH_LOG.md` | 시간순 기록 |
| `.claude/plans/` | consensus-plan 산출물 (계획·ADR) |

---

## 핵심 결과 한 줄

> 청산은 가격을 밀지 않는다(대기 지정매수의 0.1~1%, R²≈0).
> 그러나 **"이 움직임이 정보가 아니라 포지셔닝에서 나왔다"는 표지**이고,
> 정보 없는 매도의 반대편에 서면 유동성 공급 프리미엄을 받는다.

전략은 시점을 맞추는 것이 아니라 **정보 없는 매도가 나올 가격대에 지정가를 미리 까는 것**이다.

---

## 실시간 수집기 (4종, 전부 가동)

바탕화면 `start_loggers.bat` → `C:\Quant\polymarket\start_loggers.py` 의 JOBS에 등록되어 있다.
상태 확인 `python start_loggers.py --list`, 중지는 해당 최소화 콘솔 창을 닫는다.

| 수집기 | 대상 | 주기 | 역할 |
|---|---|---|---|
| `hl_positions` | HL 상장 **177종** 전부 | 깊은 15분 + **핫 60초** | $\hat L(u)$ 사전 청산맵. **대체 불가** |
| `bybit_liq_ws` | 거래대금 상위 **150종** | 실시간 전건 | 실현 청산(가격+크기). $\hat L$ 보정·검증 |
| `binance_oi_poll` | 메이저 **21종** × 5시리즈 | 5분 | 레짐. `/futures/data/*` 30일 보관 구제 |
| `depth_poll` | 메이저 **21종** | 30초 | 충격의 분모 $D(u)$ |

### 왜 이 넷인가

- **`hl_positions`**: `liquidationPx`는 라이브 쿼리에만 존재하고 과거 스냅샷 아카이브가
  세상에 없다. "무엇이 청산됐는가"는 백필되지만 **"어디에 쌓여 있었는가"는 지금부터
  쌓아야만** 얻는다.
- **`bybit_liq_ws`**: Binance `forceOrder`는 심볼당 초당 1건에 걸려 있다(실측: 전 종목
  90초 구독에 **0건**, 초당 2건 이상 0.38% vs Bybit 32.3%·최대 170건/초). 히스토리 없음.
- **`depth_poll`**: 모델 충격항이 임계형 $\max(0, V/D-c)^\beta$ 라 $D$ 없이는 임계 초과를
  판정할 수 없다. 극단 이벤트는 연 수 회뿐이라 놓치면 그 회차가 영구 결손.
- **`binance_oi_poll`**: 과거분은 `data.binance.vision/metrics`(2020-09~)로 전부 받을 수
  있어 긴급하지 않다. T-1 갭 메우기 + 벌크 데이터셋 중단 대비용.

### 2계층 스윕 (hl_positions)

4000계좌 전체 스윕은 레이트리밋상 물리적으로 500초가 걸린다. 간격 단축으로는 못 푼다.
대신 계층을 나눈다.

- **깊은 스윕** 4000계좌 / 900초 — 전체 지도
- **핫 스윕** 청산가가 현재가 ±25% 이내인 주소(약 215개) / **60초** — 27초 비용

핫리스트는 매 사이클 현재 mark 기준으로 재계산한다. 가격이 밀리면 멀리 있던 포지션이
자동으로 들어와 **변동성 급등 시 추적 대상이 스스로 넓어진다.**

---

## 관측 한계 (전부 실측)

| 항목 | 수치 | 함의 |
|---|---|---|
| HL 커버리지 $\phi$ | ≈ **0.05** (HL 15% × 스윕 36%) | 지도는 시장의 작은 표본. Q1이 문지기 |
| Binance 청산 피드 | 초당 1건, 편향이 **강도 의존(185배)** | 거래소 합산 불가. 물량은 Bybit 전건만 |
| OKX 청산 `amount` | 코인이 아니라 **계약 수** | ctVal 미적용 시 명목가 40배 과대 |
| REST 호가 도달 범위 | BTC **0.19%** / ETH 0.56% / 나머지 −1% 이상 | BTC·ETH의 −1% 깊이는 실시간 관측 불가 |
| WS `depth20` | ±0.1% | −0.5% 이상 전부 NaN. 사용 불가 |
| 재량 손절 층 $\Sigma(u)$ | 관측 수단 없음 | 근접 격리 비중 21.9% → 나머지 78%의 손절이 지도에 없음 |

**도달 못 한 깊이 구간은 0이 아니라 NaN으로 남긴다.** 0으로 채우면 "깊이가 없다"로 오독된다.

---

## 데이터 레이아웃

```
data/
  hl_positions/<날짜>/positions_<sweep>.parquet   # 깊은 스윕 (L-hat 원재료)
  hl_hot/<날짜>/hot_<id>.parquet                  # 핫 스윕 (근접 연료 60초)
  hl_accounts/ hl_mids/ hl_sweeplog/ hl_universe/
  bybit_liq/<날짜>/liq_<ts>.parquet               # 실현 청산 전건
  depth/<날짜>/depth_<ts>.parquet                 # 호가 깊이 (구간 누적 명목가)
  binance_futures_data/<series>/<SYMBOL>.parquet  # 5분 OI·롱숏비율
  tardis_multi/liquidations.parquet               # 과거 청산 3거래소 x 47종 x 95일
  binance_bulk/{klines_1m, klines_5m, metrics, book_depth}/
  analysis/                                       # 검정 산출물
```

`hl_sweeplog`의 `trusted=False` 스윕은 실패율 30% 초과로 $\hat L$ 이 과소집계된 구간이다.
**분석 시 반드시 제외하거나 별도 취급할 것.**

---

## 연구 유니버스

메이저 21종 — 순수 크립토, 2021년부터 이력, perp 유동성 상위:

```
BTC ETH SOL XRP BNB DOGE ADA AVAX LINK LTC DOT
TRX BCH UNI NEAR ATOM ETC FIL SUI AAVE WLD
```

토큰화 주식(SOXL·SKHYNIX·KORU·EWY·SNDK·MU·SPCX), 상품(XAU·XAG), 저유동성 밈,
거래소 자체 토큰(HYPE)은 제외. 미시구조가 달라 표본을 오염시킨다.

`hl_positions`(177종)와 `bybit_liq_ws`(150종)는 넓게 받아두고 분석에서 21종으로 거른다.
받는 비용이 무시할 수준이고 나중에 확장할 수 있기 때문이다.

---

## 실행

```bash
# 상시 (start_loggers.bat 이 자동 기동)
python collectors/hl_positions.py
python collectors/bybit_liq_ws.py
python collectors/binance_oi_poll.py
python collectors/depth_poll.py

# 과거 데이터
python downloaders/binance_bulk.py                              # 5m + metrics (2020-09~)
python downloaders/binance_bulk.py --interval 1m --no-metrics   # 1m 전체 이력
python downloaders/binance_days.py                              # 1m, 필요한 날짜만
python downloaders/tardis_multi.py --top 40                     # 청산 3거래소
python downloaders/binance_depth.py                             # bookDepth (2023-01~)

# 검정
python analysis/impact_depth.py        # 충격함수 (깊이 정규화)  <- v2 근거
python analysis/feed_bias.py           # 거래소 피드 스로틀 편향
python analysis/maker_1m.py            # 지정가 배치 + 플라시보
python analysis/stopping_hazard.py     # S(u), EV(u) 무차별성
python analysis/representativeness.py  # Q1 대표성 (데이터 대기)
python analysis/render_model.py        # MODEL.png 재생성
```

---

## Windows 주의사항 (실측으로 물린 것들)

1. JSON 읽기에 `encoding='utf-8'` 필수. 기본 cp949라 리더보드 파싱이 죽는다.
2. 런처가 `>> x.out` 으로 리다이렉트하면 stdout이 cp949가 되어 한글 print가 실패한다
   → `common.init_stdout()` 이 utf-8 + `errors='replace'` 로 재설정한다.
3. 저장은 전부 `atomic_write_parquet`(PID별 tmp → `os.replace`). 대상 파일을 다른
   프로세스가 잡고 있으면 `PermissionError` 가 나므로 짧게 재시도한다.
4. **읽기 실패한 파일은 절대 덮어쓰지 않는다.** 격리(`.corrupt.<ts>`) 후 새로 시작.
5. 수집기 4종 모두 `msvcrt` 파일잠금으로 중복 실행을 차단한다.
6. parquet append 시 `pd.concat` 후 **스키마를 다시 고정**한다(int64가 double로 드리프트).
7. `render_model.py` 를 **셸 heredoc으로 패치하지 말 것** — 백슬래시 한 겹이 소실되면
   `\beta` → 백스페이스+`eta` 로 조용히 깨진다. Edit 도구로 직접 고칠 것.
8. matplotlib 텍스트에서 통화 `$` 는 반드시 이스케이프. 안 하면 mathtext 구간이 열려
   뒤따르는 한글이 두부가 된다.

---

## 검정 상태

| # | 문항 | 상태 |
|---|---|---|
| **Q1** | 실현 청산이 $\hat L$ 두꺼운 가격대에 집중되는가 (대표성) | **데이터 대기** — 문지기 |
| **Q2** | $\mathbb{E}[r\mid\text{fill}(u)]$ 가 $\hat L(u)$ 의 증가함수인가 | v2 핵심, Q1 통과 후 |
| Q3 | $\pi$ 조건부 EV 가 무조건부 42~52bp 를 넘는가 | Q2 통과 후 |
| Q4 | $V/D>c$ 극단에서 충격항이 살아나는가 | 표본 밖 (Tardis 유료) |

**넘어야 할 기준선: EV 42~52bp / 이벤트당 33~38bp.**
못 넘으면 고정 offset 2~3%와 다를 바 없고, $\hat L$ 을 볼 이유가 없다.

---

## 참고문헌

- Coval & Stafford (2007), JFE — fire sale 가격압력과 반전, 유동성 공급자 수익
- Nagel (2012), RFS — 단기 반전은 유동성 공급의 대가
- Brunnermeier & Pedersen (2009), RFS — margin/loss spiral
- Brunnermeier & Pedersen (2005), JF — Predatory Trading
- Osler (2005), JIMF — FX 스탑 클러스터와 가격 캐스케이드 ($\Sigma(u)$ 층)
- Bian, He, Shue, Zhou (2018), NBER WP 25040 — 2015 중국 폭락, 40영업일 되돌림
- arXiv 2607.27070 (2026) — BTC 캐스케이드 7건, 조기경보 신호의 이벤트 이질성
