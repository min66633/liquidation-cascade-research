# -*- coding: utf-8 -*-
"""MODEL.md(v3)의 수식을 PNG로 렌더링.

matplotlib mathtext 사용(별도 LaTeX 설치 불필요). mathtext는 LaTeX의 부분집합이라
\\boxed, \\underbrace, \\big, \\dfrac, \\!, align 환경을 지원하지 않는다.
박스는 실제 텍스트 bbox를 잡아 patch로 깐다.

이 파일에서 반복적으로 밟은 지뢰 — 고치기 전에 읽을 것
  1) 셸 heredoc으로 패치하지 말 것. 백슬래시 한 겹이 소실되면 \\beta -> \\b(백스페이스)
     +eta 처럼 조용히 깨진다(실제 3회 발생). Edit/Write 도구로 직접 고칠 것.
  2) 통화 기호 \\$ 를 반드시 이스케이프할 것. 안 하면 mathtext 구간이 열려서
     그 뒤 한글이 수식 폰트로 넘어가 두부(tofu)가 된다.
  3) 한글을 $...$ 안에 넣지 말 것. mathtext(CM 폰트)에 한글 글리프가 없다.
     \\mathrm{상시} 같은 것도 안 된다 — 설명은 전부 수식 밖으로.
  4) U+2212(진짜 마이너스)는 Malgun Gothic 에 없다. 본문에는 ASCII 하이픈을 쓴다.
  5) 레이아웃은 정규화 좌표(0~1)에 gap 을 빼 나가는 방식이라, 내용이 늘면 y 가 0
     아래로 내려가 잘린다. 내용을 늘릴 때는 GS(gap scale)를 줄이고 figsize 높이를
     같은 비율로 키워 물리적 밀도를 유지한다.

실행:
    python analysis/render_model.py
    python analysis/render_model.py --out model.png --dpi 220
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C            # noqa: E402

KR = "Malgun Gothic"
plt.rcParams["font.family"] = KR
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

INK = "#12161c"
MUTED = "#5b6572"
ACCENT = "#0b6bcb"
WARN = "#b3261e"
OK = "#1b7f4b"
BOXBG = "#eef4fb"
BOXED = "#c8dcf2"
DEADBG = "#fdeeec"
DEADED = "#f3c9c4"
RULE = "#d6dbe2"

# 레이아웃 산수 (v3 는 v2 대비 내용이 약 3배다)
#   내용 총량 = 2.94 정규화단위 (GS=1 기준). y 는 0.982 에서 시작하므로 캔버스 아래로
#   넘치지만, savefig(bbox_inches="tight") 가 넘친 아티스트까지 포함해 저장하므로
#   잘리지는 않는다(박스는 clip_on=False 필요). 실제 출력 크기를 결정하는 것은
#   **물리 밀도** PPU = GS x FIG_H (정규화 1단위당 인치)다.
#     줄간격(인치) = note gap(0.0152) x PPU,   본문 폰트 9.3pt = 0.129 인치
#   v2 는 PPU=18.5 -> 줄간격/글자높이 = 2.18 로 매우 성기다. v3 는 내용이 3배라
#   그대로 두면 1:5.5 짜리 띠가 된다. PPU=13.2 (비율 1.55) 로 조이고 폭을 넓힌다.
GS = 0.74
PPU = 13.2
FIG_W = 11.5
FIG_H = PPU / GS


def render(path: str, dpi: int) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = [0.982]
    boxed = []          # (artist, dead?)

    def drop(v: float) -> None:
        y[0] -= v * GS

    def head(txt: str, size: float = 12.5, gap: float = 0.020) -> None:
        drop(gap)
        ax.text(0.065, y[0], txt, fontsize=size, color=ACCENT,
                family=KR, weight="bold", va="top")
        drop(0.015)          # 0.011 이면 '(v3)' 같은 디센더가 첫 줄에 닿는다

    def note(txt: str, size: float = 9.3, gap: float = 0.0152,
             color: str = MUTED, x: float = 0.065) -> None:
        ax.text(x, y[0], txt, fontsize=size, color=color, family=KR, va="top")
        drop(gap)

    def eq(txt: str, size: float = 14, gap: float = 0.036,
           box: bool = False, dead: bool = False) -> None:
        if box:
            drop(0.012)
        t = ax.text(0.5, y[0] - 0.006 * GS, txt, fontsize=size,
                    color=(MUTED if dead else INK), ha="center", va="top", zorder=3)
        if box:
            boxed.append((t, dead))
            gap += 0.015
        drop(gap)

    def rule(gap: float = 0.011) -> None:
        drop(gap * 0.4)
        ax.plot([0.065, 0.935], [y[0], y[0]], lw=0.8, color=RULE, clip_on=False)
        drop(gap * 0.6)

    # ---------------------------------------------------------------- 제목
    ax.text(0.5, y[0], "청산 캐스케이드 확률모델  v3", fontsize=21, color=INK,
            family=KR, weight="bold", ha="center", va="top")
    drop(0.028)
    ax.text(0.5, y[0], "Liquidation Cascade — Probabilistic Model (revised 2026-08-01)",
            fontsize=10.5, color=MUTED, ha="center", va="top")
    drop(0.019)
    rule()

    # ---------------------------------------------------------------- 0
    head("0.  v1 충격항 — 이제는 '미검정'이 아니라 임계형이 반증됐다")
    eq(r"$dM \;=\; \kappa\left[\max\left(0,\;\frac{V(u)}{D(u)} - c\right)\right]^{\beta} du"
       r"\;-\; \lambda M\,du$", size=13.5, gap=0.040, box=True, dead=True)
    note("v2 는 이 임계형을 놓고 'c > 0.31, 그 위는 미검정'이라 했다.", color=INK)
    note(r"그 0.31 이 나온 표본(`impact_depth.py` 1,423버스트)의 실제 일수를 확인하니:")
    y[0] -= 0.005 * GS
    note("   표본 일수                       6일   (2026년 매월 1일)", color=WARN)
    note("   그 6일의 max|z| 백분위 중앙   52.2%   정확히 평범한 날", color=WARN)
    note("   그 6일의 청산성 이벤트          0건   <- 사건 자체가 부재", color=WARN)
    y[0] -= 0.005 * GS
    note("전 기간 1,307일 중 청산성 이벤트가 있는 날은 148일(11.3%). 6일이 전부 비껴갈")
    note("확률은 0.887^6 = 49%. 즉 '임계 미달'이 아니라 지지집합 문제이며,")
    note("같은 날들에서 관측을 늘려도 해결되지 않는다. 필요한 건 다른 '날'이었다.")
    y[0] -= 0.006 * GS
    note("bookDepth 는 2023-01 이후 전 일자 무료 -> 청산성 148일 추가 수집 후 재측정:",
         color=INK)
    note("   V/D 지지집합    <= 0.31   ->   청산이벤트 88.3%가 0.31 초과, 최대 42배",
         color=OK)
    note("   형태            임계형(볼록) 가정   ->   꺾임 없음. 완만한 단조 증가", color=OK)
    note("   지수            미추정   ->   b = 0.04  (V/D 100배 -> 변위 1.22배)", color=OK)
    y[0] -= 0.005 * GS
    note("벤더 결함 주의 — bookDepth 는 2025년에 특정 밴드가 몇 시간씩 고정된다", color=WARN)
    note("(2025-10-10 BTC -1%: 동일 notional 440회 연속). 안 거르면 V/D 꼬리가 날조된다 —",
         color=WARN)
    note("이 문서 초판이 '2,471배'라 적은 것이 그 때문이다. 필터: analysis/bookdepth.py",
         color=WARN)
    y[0] -= 0.005 * GS
    note("'기각'이라 쓰지 않는 이유: (a) V 대용치(OI 급감액)가 자발적 청산을 포함해")
    note("감쇠편의로 계수를 0 쪽으로 민다. (b) 고정값을 걸러내면 V/D >= 10 구간이 n=31 로")
    note("비어, 임계가 있다면 있을 만한 곳을 여전히 못 본다.")
    rule()

    # ---------------------------------------------------------------- 1
    head("1.  좌표계")
    eq(r"$u \;=\; 1 - p/p_{0} \;\;(\geq 0)$"
       r"$\qquad p_{0}:\ \mathrm{start},\;\; u:\ \mathrm{displacement}$",
       size=13, gap=0.032)
    rule()

    # ---------------------------------------------------------------- 2
    head("2.  정보 없는 매도 흐름 밀도 — 두 층")
    eq(r"$U(u) \;=\; L(u) \;+\; \Sigma(u)$", size=16, gap=0.034, box=True)
    note(r"$L(u)$ 강제청산 : 관측 가능 (Hyperliquid liquidationPx x 명목가)")
    note(r"$\Sigma(u)$ 재량 손절 : 관측 불가 (Osler 2005 스탑 캐스케이드 층)")
    note("실측 — 현재가에 가까울수록 격리(isolated) 비중 상승:", color=INK)
    note("   0-5%: 21.9%     5-10%: 10.5%     10-20%: 9.6%     50%+: 1.1%   (전체 4.9%)")
    note("   교차마진(97%)은 청산가가 멀어(중앙 진입가 -38%) 강제청산 대상이 아니다.")
    note("   격리가 스퀴즈되면 교차 보유자가 '재량으로' 손절한다 -> 지도에 안 잡힌다.")
    eq(r"$\hat{L}(u) \;=\; \phi\,L(u), \qquad \phi \approx 0.05$", size=13, gap=0.030)
    rule()

    # ---------------------------------------------------------------- 3
    head("3.  가격 경로 — 문턱 없는 오목 멱함수, 그리고 D 는 상수가 아니다")
    eq(r"$dM \;=\; \kappa\left(\frac{V(u)}{D(u)}\right)^{b} du \;-\; \lambda M\,du,"
       r"\qquad b \approx 0.04$", size=13.5, gap=0.040, box=True)
    note("임계형을 폐기하고 문턱 없는 형태로 되돌린다. b = 0.04 는 실효적으로 0 이므로")
    eq(r"$\Rightarrow\;\; dM \;\approx\; -\lambda M\,du$", size=13.5, gap=0.030)
    note("=> 가격 경로 S(u) 는 청산 분포로 조건부화되지 않는다. (v1 과 가장 크게 다른 점)",
         color=INK)
    y[0] -= 0.006 * GS
    note("3.1  위험한 변수는 분자가 아니라 분모다", color=ACCENT, size=10.5)
    note("     레짐 대비 깊이 붕괴 (D_ref / D_min, 방향별 -1% 누적 명목가):")
    note("        |z| 0-2      중앙 1.09배     p90  1.55배")
    note("        |z| 8-12     중앙 2.16배     p90  8.92배")
    note("        |z| 20+      중앙 7.74배     p90 37.7배", color=WARN)
    note("     2025-10-10 최저:  BTC 109배    ETH 195배    SOL 523배", color=WARN)
    note("     그날의 '일중앙' 깊이는 정상이었다 -> 붕괴는 지속이 아니라 순간이고,")
    note("     하필 V 가 가장 큰 그 몇 분에 아래로 튄다.", color=INK)
    eq(r"$R_{0} \;\approx\; \frac{\partial(\mathrm{displacement})}{\partial V}"
       r"\times\frac{\partial L}{\partial u}, \qquad"
       r"\frac{\partial(\mathrm{displacement})}{\partial V} \propto \frac{1}{D}$",
       size=12.5, gap=0.034)
    note("     R0 를 1 위로 미는 것은 V 의 증가가 아니라 D 의 붕괴다. D 는 상태변수다.",
         color=INK)
    rule()

    # ---------------------------------------------------------------- 3.5
    head("3.5  채널 A 의 지위 — 게이트가 아니라 형태가 틀렸다")
    eq(r"$\mathrm{reversal}\;=\;\mathrm{A}\;+\;\mathrm{B}$", size=15, gap=0.034, box=True)
    note("   채널 A  충격 후 반동          문턱 없음, b = 0.04      큰 사건에서 부호 미확정",
         color=INK)
    note("   채널 B  유동성 공급 프리미엄   상시                     LIQ-CTRL = 65.7bp 확인",
         color=INK)
    y[0] -= 0.004 * GS
    note("v2 는 'A 에 임계 게이트가 달렸을 뿐 임계 위에서는 살아 있다'고 했다. 지금은")
    note("'임계는 없었고, 문턱 없는 오목 효과만 있으며, 크기가 경제적으로 무의미하다'.")
    note("=> 설계 결론은 같지만 근거가 강해졌다: B 를 기본으로 깔고 A 에 의존하지 않는다.",
         color=INK)
    note("   단 A 구간은 수익이 아니라 위험이 결정되는 구간이다 — 캐스케이드(되돌림 있음)와")
    note("   진짜 리프라이싱(되돌림 없음)이 거기서 갈리고, 동시에 깊이가 500배 증발한다.",
         color=WARN)
    rule()

    # ---------------------------------------------------------------- 4
    head("4.  채널 B — 체결 상대의 정보성")
    eq(r"$\pi(u) \;=\; \frac{U(u)}{U(u) + I(u)}"
       r"\qquad\quad"
       r"\mathbb{E}\left[\,r \mid \mathrm{fill}(u)\,\right] \;=\; r_{0} + \rho\,\pi(u)$",
       size=14, gap=0.042, box=True)
    note("실증 — 두 연구가 독립적으로 같은 결론. (v2 문서는 이 둘을 한 표에 섞어",
         color=INK)
    note(" taker 수치에 maker 체결률을 붙여 놨다. 아래가 정정판이다.)", color=INK)
    y[0] -= 0.004 * GS
    note("   (1) taker   event_study_h2.py   전량 시장가 진입 (체결률 1.00), 비용 10bp")
    note("        LIQ  n=433   +41.8bp   t= 2.8   승률 61.2%   ->  이벤트당 +41.8bp")
    note("        CTRL n=561   -23.9bp   t=-2.5   승률 49.7%   ->  이벤트당 -23.9bp")
    y[0] -= 0.003 * GS
    note("   (2) maker   maker_1m.py   offset 2%, 지정가, 비용 7bp")
    note("        LIQ  체결률 28.4%   +129.6bp   t=2.6   승률 70.7%  ->  이벤트당 +36.8bp")
    note("        CTRL 체결률 28.9%     +0.1bp   t=0.0   승률 58.0%  ->  이벤트당   +0.0bp")
    y[0] -= 0.004 * GS
    note("   => 충격 크기가 아니라 체결 상대의 정보성이 수익을 만든다.", color=INK)
    note(r"   rho 로 인용하는 65.7bp = 41.8 - (-23.9) 는 (1) taker 판의 격차다.")
    y[0] -= 0.005 * GS
    note("4.1  식별의 한계 — rho 는 하한만 안다", color=ACCENT, size=10.5)
    note(r"     관측된 것은 rho 가 아니라 rho x (pi_LIQ - pi_CTRL) = 65.7bp 다.")
    note("     pi 는 관측된 적이 없으므로 rho 와 pi 의 스케일은 분리 식별되지 않는다.")
    note(r"     pi_LIQ - pi_CTRL <= 1 이므로  rho >= 65.7bp  (하한).", color=INK)
    note("     현재 결정에는 무해하다 — 'LIQ 는 거래, CTRL 은 안 함'에 필요한 것은")
    note("     식별되는 차이뿐이다. 분해는 pi 를 다른 레짐으로 외삽할 때만 필요하다.")
    rule()

    # ---------------------------------------------------------------- 5
    head("5.  의사결정")
    eq(r"$\mathrm{EV}(u) \;=\; S(u)\left[\,r_{0} + \rho\,\pi(u)\,\right] \;-\; c_{\mathrm{fee}}$",
       size=16, gap=0.036, box=True)
    eq(r"$w(u) \;\propto\; \max\left(\mathrm{EV}(u),\,0\right)$", size=13, gap=0.030)
    note("양방향 사다리. 방향 예측 없음 — 비대칭은 예측이 아니라 지도에서 관측된다.")
    note("사다리의 근거는 '분산'이 아니라 '분포 형태 개선'이다 (7절).", color=INK)
    rule()

    # ---------------------------------------------------------------- 6
    head("6.  무차별성 — 1차뿐 아니라 2차 모멘트에서도 성립한다")
    eq(r"$S(u)\,\mathbb{E}[\,r \mid u\,] \;\approx\; \mathrm{const}$",
       size=15, gap=0.032, box=True)
    eq(r"$\mathrm{Var}(Br) \;=\; S s^{2} + S(1-S)m^{2},\qquad"
       r"\mathrm{Sharpe}(u) \;=\; \frac{\sqrt{S}\,m}{\sqrt{s^{2}+(1-S)m^{2}}}$",
       size=12.5, gap=0.036)
    note("가정하지 않고 s(u) 를 직접 쟀다 (295이벤트 x 40레벨, 15분, 비용 7bp):", color=INK)
    note("      u       S       m(bp)    s(bp)    s/m      EV(bp)    sd(bp)    Sharpe")
    note("   0.25%    0.790      58       390    6.68       46.1       348     0.133")
    note("   1.00%    0.495      79       491    6.24       38.9       347     0.112")
    note("   2.00%    0.288     163       640    3.94       46.8       351     0.133")
    note("   4.00%    0.122     425       997    2.35       51.9       375     0.138")
    note("   6.00%    0.071     720     1,296    1.80       51.2       392     0.131")
    y[0] -= 0.005 * GS
    note("시도당 표준편차도 전 구간 ~350bp 로 EV(45bp)만큼이나 평평하다. s 는 m 에", color=INK)
    note("비례하지도 상수도 아니고(s/m 6.68 -> 1.80), S 하락과 s 상승이 Var 의 두 항에서",
         color=INK)
    note("정확히 상쇄된다. => Sharpe = 0.13 이 0.25~6% 전 구간 평평.", color=OK)
    note("목적함수를 평균-분산으로 바꿔도 배치가 얕은 쪽으로 쏠리지 않는다.", color=OK)
    y[0] -= 0.004 * GS
    note("넘어야 할 기준선(Q3): 이벤트당 33~38bp (비용 후). 진짜 대조군 CTRL 은 0bp.",
         color=INK)
    note("주의: 이 표본은 k=8시그마 dOI<=-2% 롱청산 295건이다. '무조건부'는 배치거리에")
    note("대해서지 이벤트 선택에 대해서가 아니다.")
    rule()

    # ---------------------------------------------------------------- 7
    head("7.  사다리는 분산되지 않는다 — 체결이 완전 중첩이기 때문")
    eq(r"$S(u) \;=\; \mathbb{P}(X \geq u), \qquad"
       r"X \;=\; \max\ \mathrm{drawdown\ in\ horizon}$", size=13.5, gap=0.036, box=True)
    note("실측 확인: max |S_실측(u) - P(X >= u)| = 0.0000  (완전 중첩)", color=OK)
    note("X 분위: 25%=0.29%   50%=0.99%   75%=2.23%   90%=4.64%   99%=19.40%")
    y[0] -= 0.004 * GS
    note("   사다리              자본투입률   평균(bp)   Sharpe   승률    p05(bp)")
    note("   균등 40레벨            25.1%       47.0      0.13    67.1%     -18")
    note("   얕은쪽만 (~2%)         48.7%       43.4      0.13    59.7%    -175")
    note("   깊은쪽만 (2%~)         14.2%       48.7      0.14    26.4%       0")
    note("   단일 2.0%              28.8%       46.8      0.13    21.4%     -86")
    y[0] -= 0.004 * GS
    note("=> 사다리는 Sharpe 를 개선하지 못한다(0.13 = 단일 레벨). 레벨을 넓게 깔아도",
         color=INK)
    note("   X 한 변수에 대한 노출만 커진다. 개선되는 것은 좌측 꼬리(p05 -86 -> -18)와",
         color=INK)
    note("   승률(21% -> 67%)이다.", color=INK)
    y[0] -= 0.005 * GS
    note("숏풋 가설은 확인되지 않았다", color=ACCENT, size=10.5)
    # 적분 기호는 상하한 때문에 보통 식보다 훨씬 높다 — gap 을 넉넉히 줘야 아래 표를
    # 침범하지 않는다(0.032 에서 실제로 겹쳤다).
    eq(r"$\Pi(X) \;=\; \int_{0}^{X} w(u)\,(u-v)\,du \;=\; w\left[\frac{X^{2}}{2}-vX\right]$",
       size=13, gap=0.055)
    note("     X 구간        n   자본투입률   평균(bp)   중앙(bp)   최악(bp)")
    note("   [0, 0.5%)     108      1.8%        2.0        0.0       -0.1")
    note("   [3%, 5%)       27     62.3%       74.1       29.2       -83")
    note("   [10%, inf)     10    100.0%     +550.7       -7.2      -526", color=WARN)
    y[0] -= 0.004 * GS
    note("이론상 되돌림이 없으면 -wX^2/2 인 숏풋이 된다. 실측은 다르다 — X 가 커질수록")
    note("평균이 '증가'한다(되돌림이 자주 일어나 v << X). 대신 최심 구간에서 평균 +550.7 /")
    note("중앙 -7.2 — 손실 구조가 아니라 복권 구조다. 최악 5%(14건)의 기여는 총 13,862bp")
    note("중 -1,770bp(-13%)로 파괴적이지 않다.")
    note("=> pi 의 역할은 이 복권의 당첨 여부를 사전에 가르는 꼬리 분류기다.", color=INK)
    rule()

    # ---------------------------------------------------------------- 8
    head("8.  검정 문항 (v3)")
    note("Q1   실현 청산이 L-hat 두꺼운 가격대에 집중되는가 (대표성)     <- 문지기, 대기중")
    note(r"Q2   $\mathbb{E}[r\mid \mathrm{fill}(u)]$ 가 $\hat{L}(u)$ 의 증가함수인가      <- 핵심",
         color=INK)
    note("        반드시 '같은 u 안에서' 비교할 것. 레벨 간 비교는 phi(u) 와 혼동된다.")
    note("Q3   pi 조건부 EV 가 이벤트당 33~38bp(비용후) 를 넘는가        <- Q2 통과 후")
    note("Q4   극단 V/D 에서 채널 A 가 살아나는가            <- 부분 검정: 임계형 반증",
         color=OK)
    note("Q5   위험조정수익 Sharpe(u) 도 평평한가            <- 통과: 0.13 평평(6절)",
         color=OK)
    note("Q6   사다리 손익의 X 분포가 숏풋인가              <- 반증: 평균이 증가(7절)",
         color=OK)
    rule()

    # ---------------------------------------------------------------- 9
    head("9.  제약")
    note("무레버리지 필수 — MAE 5%분위 15분 -790~-1,059bp / 60분 -2,507bp.", color=WARN)
    note("   체결된 거래의 5%가 -8~-25% 끌려간다. 레버리지를 쓰면 전략 자체가 청산된다.",
         color=WARN)
    note("maker 전용 — 왕복 50bp에서 엣지 소멸. 시장가는 받으려던 프리미엄을 지불하는 것.")
    note("보유 15분 — 이벤트당 15/60/240분 = 36.8 / 40.8 / 51.9bp,")
    note("   MAE -984 / -1,370 / -1,382bp. 60분은 MAE 가 39% 커진 뒤에도 수익이 안 들어와")
    note("   셋 중 유일하게 지배당한다. TTL(주문 유효 60분)과는 다른 파라미터다.")
    note("극단일 수익은 계획에 넣지 않는다 — 최대 수익일이 거래소 API 정지일이었다.")
    rule()

    # ------------------------------------------------------------- 10 용량
    head("10.  용량 — EV 와 별개의 제약")
    eq(r"$\mathrm{capacity} \;=\; \alpha\cdot\min\left(F,\; D_{exit}\right),"
       r"\qquad \alpha = 0.10$", size=14, gap=0.034, box=True)
    note("F = 지정가 이하에서 발생한 테이커 매도 대금 (지정매수는 이것이 와야만 체결된다)")
    note("D_exit = 청산 시각의 반대편 호가 명목가 (HOLD 후 시장가로 나간다)")
    y[0] -= 0.004 * GS
    note("   5종 72건 (offset 2%, HOLD 15분)      p10        중앙        p90")
    note("   유량 F (보유구간)                  \\$12.2M     \\$79.7M    \\$497.3M")
    note("   청산 깊이 D_exit                    \\$1.4M      \\$3.7M     \\$54.3M", color=WARN)
    note("   용량                                \\$137K      \\$373K      \\$5.4M", color=INK)
    y[0] -= 0.004 * GS
    note("=> 99% 의 사건에서 깊이가 구속한다. 문제는 '체결되느냐'가 아니라 '나올 수 있느냐'다.",
         color=INK)
    note("   심볼 편차 20배 — BTC \\$5.0M / ETH \\$2.9M / 알트 \\$225K. 자본배분을 나눠야 한다.")
    note("   체결 이벤트 연 22건(5종) x \\$373K = 연 회전 \\$8.0M. 작은 전략이다.", color=INK)
    y[0] -= 0.004 * GS
    note("'큰 수익 사건일수록 용량이 작다'는 확인되지 않았다 — 용량가중 - 균등가중 = -7.9bp,")
    note("일자 클러스터 부트스트랩 95% CI [-75.7, +52.1] 로 0 을 포함한다.")
    note("가장 큰 미지수: 2025-10-10 의 두 체결 이벤트 모두 청산 시각 ±2분에 쓸 수 있는",
         color=WARN)
    note("깊이 스냅샷이 없다 — 고정 필드가 걸러낸 구간이 정확히 캐스케이드 구간이다.",
         color=WARN)
    note("즉 캐스케이드 당일의 용량은 현재 데이터로 측정 불가다.", color=WARN)

    # 실제 bbox 확정 후 박스를 뒤에 깐다
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t, dead in boxed:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(inv)
        pady = 0.007 * GS
        ax.add_patch(FancyBboxPatch(
            (0.075, bb.y0 - pady), 0.85, bb.height + 2 * pady,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=1.0,
            edgecolor=(DEADED if dead else BOXED),
            facecolor=(DEADBG if dead else BOXBG),
            zorder=0, clip_on=False))
        if dead:      # 폐기된 식에 취소선
            ax.plot([bb.x0 - 0.012, bb.x1 + 0.012],
                    [bb.y0 + bb.height / 2] * 2,
                    lw=1.5, color=WARN, alpha=0.8, zorder=4, clip_on=False)

    span = 0.982 - y[0]
    print("레이아웃: 내용 %.3f 정규화단위 x PPU %.1f = %.1f 인치 (폭 %.1f, 비율 1:%.2f)"
          % (span, PPU, span * PPU, FIG_W, span * PPU / FIG_W))
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="render MODEL.md (v3) equations to PNG")
    ap.add_argument("--out", default=os.path.join(C.ROOT, "MODEL.png"))
    ap.add_argument("--dpi", type=int, default=210)
    a = ap.parse_args()
    render(a.out, a.dpi)
    print("wrote %s (%.0f KB)" % (a.out, os.path.getsize(a.out) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
