# -*- coding: utf-8 -*-
"""수식 해설 PNG — EXPLAINER.md 의 수식을 읽을 수 있게 렌더링.

왜 별도 파일인가
  markdown 뷰어에서 LaTeX 가 안 보인다. MODEL.png(render_model.py)은 모델 정본의
  '참조 카드'라 조밀한데, 이 파일은 수식마다 읽는 법 / 기호 뜻 / 실측값을 붙인
  해설판이다. 대상 독자가 다르다.

이 파일에서 반복적으로 밟은 지뢰 (render_model.py 와 동일)
  1) 셸 heredoc 으로 패치하지 말 것. 백슬래시 한 겹이 소실되면 \\beta -> 백스페이스+eta
     로 조용히 깨진다(실제 3회 발생). Edit/Write 도구로 직접 고칠 것.
  2) 통화 기호 \\$ 를 반드시 이스케이프. 안 하면 mathtext 구간이 열려서 뒤의 한글이
     수식 폰트로 넘어가 두부(tofu)가 된다.
  3) 한글을 $...$ 안에 넣지 말 것. mathtext(CM 폰트)에 한글 글리프가 없다.
  4) mathtext 는 \\boxed, \\underbrace, \\big, \\dfrac, \\!, align 을 지원하지 않는다.
  5) U+2212(진짜 마이너스)는 Malgun Gothic 에 없다. 본문에는 ASCII 하이픈.

실행:
    python analysis/render_math.py
    python analysis/render_math.py --out MATH.png --dpi 220
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
LBL = "#8b95a3"

GS = 0.62          # gap scale
PPU = 15.0         # 정규화 1단위당 인치 — 해설판이라 MODEL.png(13.2)보다 성기게
FIG_W = 11.5
FIG_H = PPU / GS


def render(path: str, dpi: int) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = [0.985]
    boxed = []

    def drop(v):
        y[0] -= v * GS

    def head(txt, size=13.0, gap=0.022):
        drop(gap)
        ax.text(0.055, y[0], txt, fontsize=size, color=ACCENT,
                family=KR, weight="bold", va="top")
        drop(0.016)

    def note(txt, size=9.2, gap=0.0145, color=MUTED, x=0.075):
        ax.text(x, y[0], txt, fontsize=size, color=color, family=KR, va="top")
        drop(gap)

    def label(txt):
        ax.text(0.055, y[0], txt, fontsize=8.2, color=LBL, family=KR,
                weight="bold", va="top")
        drop(0.013)

    def eq(txt, size=15, gap=0.040, box=True, dead=False):
        if box:
            drop(0.014)
        t = ax.text(0.5, y[0] - 0.006 * GS, txt, fontsize=size,
                    color=(MUTED if dead else INK), ha="center", va="top", zorder=3)
        if box:
            boxed.append((t, dead))
            gap += 0.018
        drop(gap)

    def rule(gap=0.014):
        drop(gap * 0.4)
        ax.plot([0.055, 0.945], [y[0], y[0]], lw=0.8, color=RULE, clip_on=False)
        drop(gap * 0.6)

    # ------------------------------------------------------------- 제목
    ax.text(0.5, y[0], "청산 캐스케이드 — 수식 해설", fontsize=22, color=INK,
            family=KR, weight="bold", ha="center", va="top")
    drop(0.030)
    ax.text(0.5, y[0], "수식마다  '읽는 법' / '기호' / '실측'  을 붙였습니다.  "
                       "정본은 MODEL.md, 서술은 EXPLAINER.md",
            fontsize=10, color=MUTED, ha="center", va="top")
    drop(0.020)
    rule()

    # ------------------------------------------------------------- 1
    head("1.  좌표계 — 시간이 아니라 가격을 축으로 쓴다")
    eq(r"$u \;=\; 1 - \frac{p}{p_{0}}$", size=17)
    label("읽는 법")
    note("트리거 봉 종가에서 몇 % 내려왔는가. u = 0.02 면 2% 아래.")
    label("왜 시간축이 아닌가")
    note("캐스케이드는 몇 분 만에 끝나 5분봉으로 보면 점 2~3개로 뭉개진다.")
    note("반면 '어느 가격대를 지났나'는 선명하다.")
    label("기호")
    note("p0 = 사건이 일어난 가격(트리거 봉 종가)    p = 현재가    u >= 0")
    rule()

    # ------------------------------------------------------------- 2
    head("2.  체결확률 (무조건부) — 지정가는 가격이 와야만 체결된다")
    eq(r"$S(u) \;=\; \mathbb{P}\,(X \geq u)$", size=17)
    label("읽는 법")
    note("u 아래에 깔아둔 지정매수가 체결될 확률 = 가격이 u 까지 내려올 확률.")
    note("X 는 그 사건에서 실제로 도달한 최대 낙폭이다.")
    label("*** 이것은 무조건부다 — 배치 결정에 쓸 수 없다 ***")
    note("모든 사건을 합친 '하나의' 곡선이다. '청산 사건 일반에서 2% 까지 갈 확률 29%'", color=WARN)
    note("라는 문장이지 '지금 이 사건에서' 의 확률이 아니다. 아래 2.6 을 볼 것.", color=WARN)
    label("중요 — 체결은 완전히 중첩된다")
    note("가격은 위에서 아래로 내려오므로 u <= X 인 레벨은 전부 체결되고", color=INK)
    note("u > X 는 하나도 체결되지 않는다. 40개 레벨에 나눠 깔아도 독립적인", color=INK)
    note("40개 베팅이 아니라 X 라는 단 하나의 확률변수에 대한 노출이다.", color=INK)
    label("실측")
    note("max |S(u) 실측 - P(X >= u)| = 0.0000   (완전 일치, 295 이벤트)", color=OK)
    note("S: 0.79 (u=0.25%)  ->  0.29 (2%)  ->  0.07 (6%)")
    note("X 분위: 25%=0.29%   50%=0.99%   75%=2.23%   90%=4.64%   99%=19.40%")
    rule()

    # ------------------------------------------------------------- 2.6
    head("2.6  체결확률 (조건부) — 설계가 실제로 요구하는 것  [미검정]")
    eq(r"$\mathbb{P}\,\left(X \geq u \mid \mathcal{F}_{t}\right)"
       r"\;=\; \Phi\!\left(\frac{\mu(\mathcal{F}_{t}) - \log D_{t}"
       r" - (\log u - a)/\gamma}{\sigma(\mathcal{F}_{t})}\right)$", size=15)
    label("무엇이 다른가")
    note("2번은 사건을 구분하지 않는 '하나의' 곡선. 이것은 사건마다 '다른' 곡선이다.", color=INK)
    note("배치는 '이 사건에서' 어디에 걸지를 정하는 것이므로 이쪽이 필요하다.", color=INK)
    label("*** 핵심 — mu 와 sigma 를 둘 다 모형화한다 ***")
    note("sigma 를 상수로 두면 그건 다시 점추정이다. 사건마다 '퍼짐이 다르다'는 것이", color=WARN)
    note("확률모델의 존재 이유다 — 어떤 캐스케이드는 바닥이 뻔하고 어떤 건 아무데나 간다.", color=WARN)
    label("무차별성과의 관계 — 이게 중요하다")
    note("4번의 '어디 걸어도 같다'는 무조건부 곡선의 성질이다. 사건별 곡선이 서로 다르면", color=INK)
    note("사건마다 다른 u 를 고를 수 있고 그때는 평평하지 않을 수 있다.", color=INK)
    note("=> 무차별성은 조건부 세계를 반증하지 않는다. 조건부는 아직 미검정이다.", color=OK)
    label("합격 기준 — R2 가 아니다")
    note("(1) 보정: '확률 30%' 라 했을 때 실제로 30% 가 체결되는가 (Kupiec)")
    note("(2) 판별: 넓다고 예측한 사건이 실제로 넓은가 (조건부 IQR 비 >= 1.5)")
    note("R2 는 mu 만 잰다. 배치에 쓰이는 것은 분위수이고 그건 sigma 가 정한다.", color=INK)
    label("이미 실패한 것 — 점추정은 확률모델이 아니다")
    note("predict_x.py 는 log X 를 OLS 로 회귀해 '숫자 하나' 를 내고 배수를 곱했다.", color=WARN)
    note("표본외 R2=0.029, 짝지은 차이 27개 중 유의 1개(우연 기대 1.35). 정본 PROB_MODEL.md",
         color=WARN)
    label("*** 이 형태로 검정했고 떨어졌다 (x_dist.py, 21종 639이벤트, 평가 320) ***")
    note("  보정   Kupiec 9개 중 4개 위반. 9개 p 전부 실제 < 명목 (한 방향 계통오차)", color=WARN)
    note("  PIT    KS 기각 p=0.029.  무조건부판(M0)은 p=0.243 으로 통과", color=WARN)
    note("  판별   예측 sigma 3분위 -> 실제 log X 의 IQR 비 = 0.78  (역방향!)", color=WARN)
    note("왜 떨어졌나 -> 18절. D 를 관측 상수로 놓은 것이 원인이다.", color=OK)
    rule()

    # ------------------------------------------------------------- 3
    head("3.  기대값 (무조건부 S 기반) — 배치 결정의 목적함수")
    eq(r"$\mathrm{EV}(u) \;=\; S(u)\;\cdot\;\mathbb{E}\left[\,r \mid \mathrm{fill}(u)\,\right] \;-\; c_{\mathrm{fee}}$",
       size=16)
    label("읽는 법")
    note("(체결될 확률) x (체결됐을 때 버는 것의 '평균') - 수수료.")
    note("E[r | fill(u)] 의 세로줄은 '조건부'다 — 체결된 경우만 센다.")
    note("여기 S 는 2번의 '무조건부' S 다. 조건부판은 2.6 을 볼 것.", color=WARN)
    label("기호")
    note("c_fee = 왕복 비용. Binance USD-M 선물 maker 0.02% + taker 0.05% = 7bp")
    label("실측 (295 이벤트 x 40 레벨, 15분 보유)")
    note("   u        S      E[r|fill]     EV")
    note("  0.25%   0.790     +58bp      46.1bp")
    note("  1.00%   0.495     +79bp      38.9bp")
    note("  2.00%   0.288    +163bp      46.8bp")
    note("  6.00%   0.071    +720bp      51.2bp")
    label("주의 — E[r|fill] 은 '항상 번다'가 아니다. 손실 건이 다 들어간 평균이다")
    note("청산은 HOLD 후 무조건 시장가다. 손절도 익절도 없어서 수익은 그대로 음수가", color=INK)
    note("될 수 있다. 21종 offset 2% 체결 324건의 실제 분포:", color=INK)
    note("     1%분위  -888bp     25%분위   -77bp     75%분위  +181bp")
    note("     5%분위  -355bp     50%분위   +59bp     99%분위 +1512bp")
    note("     평균 +87.6bp   표준편차 453.6bp   <- 변동이 평균의 5.2배", color=WARN)
    note("     손실 119건(36.7%) 평균 -202.5bp  /  수익 205건(63.3%) 평균 +255.9bp",
         color=WARN)
    note("     최악: FIL -1,531bp, WLD -1,007bp, SUI -960bp")
    note("     보유 중 최대 평가손실(MAE) 5%분위 -1,104bp", color=WARN)
    note("=> 3분의 1 이상이 손실이고, 그것을 포함한 평균이 +87.6bp 다.", color=INK)
    rule()

    # ------------------------------------------------------------- 4
    head("4.  무차별성 정리 — 이 연구의 중심 문제")
    eq(r"$S(u)\;\mathbb{E}[\,r \mid u\,] \;\approx\; \mathrm{const} \;\approx\; 46 \sim 52\,\mathrm{bp}$",
       size=16)
    label("읽는 법")
    note("체결확률은 11배 떨어지고 조건부 수익은 12배 오르는데 곱이 평평하다.", color=INK)
    note("=> 조건부 정보가 없으면 어느 가격대에 깔든 똑같다.", color=WARN)
    label("두 계열은 정확히 멱함수이고 지수가 대칭이다")
    eq(r"$S(u)\approx 0.79\left(\frac{u}{0.25\%}\right)^{-0.763}"
       r"\qquad\quad"
       r"\mathbb{E}[r|u]\approx 65\,\mathrm{bp}\left(\frac{u}{0.25\%}\right)^{+0.760}$",
       size=13, gap=0.036, box=False)
    label("왜 그런가 — 우연이 아니라 균형이다")
    note("어떤 거리의 기대값이 높으면 마켓메이커가 거기로 몰려 지정가를 깐다.")
    note("경쟁이 붙어 수익이 깎이고, 모든 거리의 기대값이 같아지는 데서 멈춘다.")
    note("평평한 수준 ~51bp 가 시장이 매기는 유동성 공급의 가격이다.")
    label("이것이 왜 심각한가")
    note("이 무차별성을 못 깨면 연구는 '고정 2%' 와 다를 게 없고 청산맵을 볼 이유가 없다.",
         color=WARN)
    label("흔한 오해 — '평평하다'는 '캐스케이드가 가속하지 않는다'가 아니다")
    note("위 S(u) 는 사건 간 '평균'이다. 개별 사건의 S 는 사후적으로 0/1 이고,", color=INK)
    note("실제 낙폭 X 의 변동계수는 1.77 로 평균이 대표성을 잃는 수준이다.", color=INK)
    note("그리고 캐스케이드는 실제로 가속한다 — 조건부 도달 확률(다음 절 참조)이")
    note("깊어질수록 오른다. 가속이 없어서 평평한 것이 아니라,")
    note("가속과 되돌림 크기가 같은 원인에서 나와 곱에서 상쇄되기 때문이다.", color=INK)
    note("=> 무차별성은 '가속이 없다'가 아니라 '가속을 사전에 구별할 수 없다'이다.", color=WARN)
    rule()

    # ------------------------------------------------------------- 4.5
    head("4.5  해저드 — 캐스케이드는 가속한다")
    eq(r"$h(u) \;=\; \mathbb{P}\left(\,\mathrm{stop\ at\ }u \mid X \geq u\,\right),"
       r"\qquad S(u) \;=\; \prod_{v<u}\left(1 - h(v)\right)$", size=14)
    label("읽는 법")
    note("h 는 '거기까지 왔을 때 거기서 멈출 확률'. S 는 그것들을 곱해 쌓은 것이다.")
    note("h 가 상수면 S 는 지수감쇠, h 가 감소하면 S 의 꼬리가 두꺼워진다.")
    label("실측 — 조건부 도달 확률 P(X >= u + d | X >= u), 295 이벤트")
    note("   도달한 u      +0.5%p     +1%p      +2%p     잔여 n")
    note("     0.25%       0.712     0.541     0.309      233")
    note("     1.00%       0.753     0.582     0.363      146")
    note("     2.00%       0.765     0.624     0.424       85")
    note("     3.00%       0.811     0.679     0.491       53")
    note("     5.00%       0.885     0.808     0.769       26", color=WARN)
    note("완전 단조 증가 — 0.25% 까지 온 사건이 2%p 더 갈 확률은 31% 인데,", color=INK)
    note("5% 까지 온 사건은 77% 다.  '한 번 뚫리면 멀리 간다' 가 성립한다.", color=INK)
    label("아직 모르는 것 — 가속의 '위치'")
    note("풀링 데이터는 평균적으로 매끄러운 가속만 보여준다. 두꺼운 청산 군집 위에서")
    note("h 가 높고 뚫린 뒤 얇은 구간에서 h 가 급락하는 **절벽 구조**가 개별 사건에")
    note("있는지, 그 위치를 L-hat 이 알려주는지는 미검정이다. 그것이 Q2c 다.", color=INK)
    rule()

    # ------------------------------------------------------------- 5
    head("5.  위험까지 봐도 평평하다 — 2차 모멘트")
    eq(r"$\mathrm{Var}(Br) \;=\; S\,s^{2} \;+\; S(1-S)\,m^{2}"
       r"\qquad\quad"
       r"\mathrm{Sharpe}(u) \;=\; \frac{\sqrt{S}\;m}{\sqrt{s^{2}+(1-S)m^{2}}}$",
       size=14)
    label("읽는 법")
    note("체결을 동전던지기 B ~ Bernoulli(S), 조건부 수익을 평균 m 표준편차 s 로 둔다.")
    note("분산의 첫 항은 '체결됐을 때의 변동', 둘째 항은 '체결되냐 마냐의 변동'이다.")
    label("실측 — 표준편차도 평평하다")
    note("   u        S     m(bp)   s(bp)   s/m    EV(bp)   sd(bp)   Sharpe")
    note("  0.25%   0.790     58     390    6.68    46.1      348     0.133")
    note("  2.00%   0.288    163     640    3.94    46.8      351     0.133")
    note("  6.00%   0.071    720   1,296    1.80    51.2      392     0.131")
    label("결론")
    note("S 가 떨어지는 만큼 s 가 올라 분산의 두 항이 상쇄된다. Sharpe = 0.13 이", color=OK)
    note("0.25~6% 전 구간 평평 => 평균-분산으로 바꿔도 배치가 쏠리지 않는다.", color=OK)
    rule()

    # ------------------------------------------------------------- 6
    head("6.  사다리 손익 — 왜 분산되지 않는가")
    eq(r"$\Pi(X) \;=\; \int_{0}^{X} w(u)\,(u-v)\,du \;=\; w\left[\frac{X^{2}}{2} - vX\right]$",
       size=15, gap=0.052)
    label("읽는 법")
    note("X 까지 내려갔으면 [0, X] 의 모든 레벨이 체결돼 있다. 각 레벨은 진입가 u 와")
    note("청산 시점 낙폭 v 의 차이만큼 번다. 그 합이 손익이고 X 에 2차다.")
    label("이론상 예측")
    note("되돌림이 없으면(v = X) 손익이 -wX^2/2 인 숏풋(short put) 페이오프가 된다.")
    label("실측 — 그렇지 않다")
    note("   X 구간         n    자본투입률   평균(bp)   중앙(bp)")
    note("  [0, 0.5%)     108      1.8%        2.0        0.0")
    note("  [3%, 5%)       27     62.3%       74.1       29.2")
    note("  [10%, inf)     10    100.0%     +550.7       -7.2", color=WARN)
    note("X 가 커질수록 평균이 오른다 (되돌림이 자주 일어나 v << X).")
    note("다만 최심 구간에서 평균 +550.7 / 중앙 -7.2 — 손실 구조가 아니라 복권 구조다.",
         color=INK)
    label("사다리 vs 단일 레벨")
    note("균등 40레벨 Sharpe 0.13  =  단일 2% Sharpe 0.13   (개선 없음)")
    note("개선되는 것은 좌측 꼬리(p05 -86 -> -18bp)와 양수 비율(21% -> 67%)뿐이다.")
    rule()

    # ------------------------------------------------------------- 7
    head("7.  정보 없는 매도 흐름 — 두 층으로 되어 있다")
    eq(r"$U(u) \;=\; L(u) \;+\; \Sigma(u)$", size=17)
    label("읽는 법")
    note("정보 없는 매도 = 강제청산 + 재량 손절.")
    note("L 은 볼 수 있고 Sigma 는 볼 수 없다.")
    label("기호")
    note("L(u)     강제청산 밀도.  관측 가능 — Hyperliquid 가 계좌별 청산가를 공개", color=INK)
    note("Sigma(u) 재량 손절 밀도. 관측 불가 — 어디에도 기록이 없다", color=WARN)
    label("왜 두 층이 구조적으로 분리되나 (실측 50,551 포지션)")
    note("   현재가 대비 거리      격리마진 비중")
    note("      0-5%                21.9%")
    note("      5-10%               10.5%")
    note("      50%+                 1.1%      (전체 평균 4.9%)")
    note("가까울수록 격리·고레버리지 비중이 오른다. 교차마진은 청산가가 멀어")
    note("(중앙 진입가 -38%) 사정거리 밖이다. 즉 근접 강제 물량은 격리에서 나오고,")
    note("교차 보유자는 그 움직임을 보고 재량으로 손절한다 — 그 층이 Sigma 다.")
    rule()

    # ------------------------------------------------------------- 8
    head("8.  관측 커버리지 — 우리가 보는 것은 일부다")
    eq(r"$\hat{L}(u) \;=\; \phi \, L(u), \qquad \phi \approx 0.05$", size=16)
    label("읽는 법")
    note("실제 청산맵 L 중 우리가 보는 비율이 phi 다. 약 5%.")
    label("어디서 5% 가 나오나")
    note("Hyperliquid 의 시장 점유율 15%  x  우리 스윕이 덮는 계좌 36%  =  약 5%")
    label("위험")
    note("phi 가 거리 u 에 따라 다르면 L-hat 의 형태 자체가 왜곡된다.", color=WARN)
    note("HL 포지션의 97% 가 교차마진인데 CEX 리테일은 격리·고레버리지 비중이 높다.")
    note("=> 검정을 '같은 u 안에서' 짜야 phi(u) 가 상쇄된다.", color=INK)
    rule()

    # ------------------------------------------------------------- 9
    head("9.  상대의 정보성 — 무차별성을 깨는 유일한 항")
    eq(r"$\pi(u) \;=\; \frac{U(u)}{U(u) + I(u)}"
       r"\qquad\quad"
       r"\mathbb{E}\left[\,r \mid \mathrm{fill}(u)\,\right] \;=\; r_{0} + \rho\,\pi(u)$",
       size=15)
    label("읽는 법")
    note("pi = 내 지정가를 체결시킨 상대가 '정보 없는 쪽'일 비중.")
    note("정보 없는 상대와 체결하면 이득, 정보 있는 상대면 손해(역선택)다.")
    label("기호")
    note("I(u)  정보 기반 매도 흐름 (뭔가 알고 파는 쪽)")
    note("r0    순수 역선택 손실. 상대가 전부 정보 있는 쪽일 때의 수익 (음수여야 정상)")
    note("rho   두 유형 사이의 스프레드 = 정보 없는 상대와 거래할 때의 프리미엄")
    label("실측 — 대리변수로 가른 결과 (플라시보 통과)")
    note("   LIQ  (급변 + OI급감)   n=433   +41.8bp   t= 2.8   승률 61.2%", color=OK)
    note("   CTRL (같은 급변, OI유지) n=561   -23.9bp   t=-2.5   승률 49.7%")
    note("   격차 65.7bp — 같은 크기의 하락이라도 강제 흐름이 동반된 쪽만 되돌린다.", color=INK)
    label("식별의 한계 — rho 는 하한만 안다")
    eq(r"$\rho\,(\pi_{\mathrm{LIQ}} - \pi_{\mathrm{CTRL}}) = 65.7\,\mathrm{bp}"
       r"\;\; \Rightarrow \;\; \rho \geq 65.7\,\mathrm{bp}$", size=13, gap=0.034, box=False)
    note("pi 를 절대 눈금으로 잰 적이 없어 rho 와 pi 는 분리 식별되지 않는다.")
    note("(pi_LIQ - pi_CTRL <= 1 이므로 하한만 나온다.)")
    note("실무적으로는 무해하다 — 'LIQ 는 거래, CTRL 은 안 함' 에 필요한 것은 차이뿐이다.")
    rule()

    # ------------------------------------------------------------- 10
    head("10.  충격항 (폐기) — 임계형은 지지되지 않았다")
    eq(r"$dM \;=\; \kappa\left[\max\left(0,\;\frac{V(u)}{D(u)} - c\right)\right]^{\beta} du"
       r"\;-\; \lambda M\,du$", size=14, gap=0.044, dead=True)
    label("읽는 법 (원래 가정)")
    note("청산액 V 가 호가 깊이 D 대비 임계 c 를 넘으면 그때부터 가격을 민다.")
    note("둘째 항은 밀린 것이 시간이 지나면 되돌아온다는 감쇠항이다.")
    label("왜 폐기했나")
    note("임계 c 는 6일치 표본(청산 이벤트 0건)에서 '> 0.31' 로만 알려져 있었다.", color=WARN)
    note("148일을 추가 수집해 지지집합을 42배 넓혔는데 꺾임이 없었다.", color=WARN)
    note("V/D 30배당 변위 2.8배로 완만한 단조 증가일 뿐이다.")
    rule()

    # ------------------------------------------------------------- 11
    head("11.  충격항 (현행) — 문턱 없는 오목 멱함수, 그리고 실효적으로 0")
    eq(r"$dM \;=\; \kappa\left(\frac{V(u)}{D(u)}\right)^{b} du \;-\; \lambda M\,du,"
       r"\qquad b \approx 0.04$", size=14)
    label("읽는 법")
    note("V/D 가 100배 커져도 변위는 1.22배. 표준 제곱근 법칙(b=0.5)의 1/11 이다.")
    note("=> 청산은 규모와 무관하게 가격을 거의 밀지 못한다.", color=INK)
    label("따라서")
    eq(r"$dM \;\approx\; -\lambda M \, du$", size=14, gap=0.032, box=False)
    note("가격 경로 S(u) 는 청산 분포로 조건부화되지 않는다 — 무차별성을 S 로는 못 깬다.")
    label("주의 — 통계적 유의성이지 경제적 유의성이 아니다")
    note("전체 표본 t = 8.04 (일자 클러스터). iid 표준오차를 쓰면 15.95 로 2배 부풀려진다.")
    note("11만 바가 만든 유의성이며 효과 크기는 무의미하다.")
    label("*** 표기 경고 — 이 b 는 16번의 gamma 가 아니다 ***")
    note("b ~ 0.04 : 충격함수 지수. 청산량이 가격을 미는 탄력성. 사실상 0, 기각됨.", color=WARN)
    note("gamma = 1.011 : 구조식 기울기. log X 의 log(V/D) 탄력성. 살아있음.", color=WARN)
    note("둘은 다른 양이다. 섞어 쓰지 말 것 (PROB_MODEL.md 0절).", color=WARN)
    rule()

    # ------------------------------------------------------------- 12
    head("12.  분기비 — 캐스케이드를 만드는 것은 분자가 아니라 분모다")
    eq(r"$R_{0} \;\approx\; \frac{\partial(\mathrm{displacement})}{\partial V}"
       r"\;\times\; \frac{\partial L}{\partial u},"
       r"\qquad \frac{\partial(\mathrm{displacement})}{\partial V} \;\propto\; \frac{1}{D}$",
       size=13.5)
    label("읽는 법")
    note("R0 = 청산 1건이 낳는 2차 청산의 기댓값. 전염병의 재생산지수와 같은 구조다.")
    note("R0 < 1 이면 몇 개 넘어지다 멈추고, R0 > 1 이면 폭주한다.")
    label("핵심")
    note("R0 를 1 위로 미는 것은 청산액 V 의 증가가 아니라 호가 깊이 D 의 붕괴다.", color=INK)
    label("실측 — 깊이는 순간적으로 무너진다")
    note("   |z| 0-2     중앙 1.09배      p90  1.39배")
    note("   |z| 8-12    중앙 1.64배      p90  5.19배")
    note("   |z| 20+     중앙 6.10배      p90 36.4배", color=WARN)
    note("   2025-10-10 최저:  BTC 114배   ETH 206배   SOL 561배", color=WARN)
    note("그날의 일중앙 깊이는 정상이었다 => 붕괴는 지속이 아니라 순간이고,")
    note("하필 V 가 가장 큰 그 몇 분에 아래로 튄다. D 는 상수가 아니라 상태변수다.", color=INK)
    rule()

    # ------------------------------------------------------------- 13
    head("13.  수급 청산 회계 — v1(정지점)의 올바른 형태")
    eq(r"$\int_{0}^{u^{*}}\left[\,D(v) + M(v)\,\right] dv"
       r"\;=\; \int_{0}^{u^{*}} L(v)\,dv \;+\; E$", size=15, gap=0.050)
    label("읽는 법")
    note("좌변 = 아래에서 받아주는 물량(대기 지정매수 D + 도착하는 시장매수 M)")
    note("우변 = 위에서 쏟아지는 물량(강제매도 L + 외생 매도 E)")
    note("둘이 같아지는 u* 가 캐스케이드가 멈추는 곳이다.")
    label("검정 결과 (2026-08-01, 253 이벤트)")
    note("정의 범위:  표본의 67% 에서만 성립. 33% 는 구간 순매수라 정의 불가", color=WARN)
    note("사후 회계:  흡수율 중앙 1.47 — 자릿수는 맞는다", color=OK)
    note("깊이 형태:  심볼 내부에서 u* (0.610) = NS/D 스칼라 (0.601) < NS/OI (0.625)", color=WARN)
    note("            => 호가 프로파일의 밴드 구조는 아무것도 더하지 않는다", color=WARN)
    note("사전 예측:  R2 = -0.17.  상수(-0.11) 도 변동성(+0.05) 도 못 이긴다", color=WARN)
    label("실패 원인")
    note("깊이 지도가 아니라 도착 유량 M 을 예보할 수 없어서다.", color=INK)
    note("사후 Spearman 0.613 -> 사전 0.251 의 낙차 전부가 그 대체 비용이다.")
    label("아직 안 해본 형태 (Q2c)")
    note("공급을 가격대별 히트맵 L-hat(p) 으로 쪼개고, 가격이 내려가는 동안", color=INK)
    note("순차 갱신하면 총 유량을 미리 맞힐 필요가 없다. Q1 통과 후 검정한다.", color=INK)
    rule()

    # ------------------------------------------------------------- 14
    head("14.  용량 — EV 와 별개의 제약")
    eq(r"$\mathrm{capacity} \;=\; \alpha \cdot \min\left(\,F,\; D_{\mathrm{exit}}\,\right),"
       r"\qquad \alpha = 0.10$", size=15)
    label("읽는 법")
    note("F        진입 상한. 지정매수는 테이커 매도가 와야만 체결된다")
    note("D_exit   청산 상한. HOLD 후 시장가로 나갈 때 대기 중인 반대편 호가")
    note("alpha    우리가 차지해도 시장을 안 바꾸는 비중. 가정이며 결과는 여기 선형")
    label("실측 (5종 72건, offset 2%, HOLD 15분)")
    note("                        p10        중앙        p90")
    note("  유량 F              \\$12.2M     \\$79.7M    \\$497.3M")
    note("  청산 깊이 D_exit     \\$1.4M      \\$3.7M     \\$54.3M", color=WARN)
    note("  용량                 \\$137K      \\$373K      \\$5.4M", color=INK)
    label("결론")
    note("99% 의 사건에서 깊이가 구속한다. 문제는 '체결되느냐'가 아니라", color=INK)
    note("'나올 수 있느냐'다. 심볼 편차 20배 (BTC \\$5.0M / 알트 \\$225K).", color=INK)
    note("체결 이벤트 연 22건(5종) x \\$373K = 연 회전 \\$8.0M. 작은 전략이다.")
    rule()

    # ------------------------------------------------------------- 16
    head("16.  구조식 — 밀린 거리는 강제매도량 / 호가두께 다  [2026-08-02 확정]")
    eq(r"$\log X \;=\; a \;+\; \gamma\,\log\frac{V}{D},"
       r"\qquad a = -7.324,\quad \gamma = 1.011$", size=15)
    label("읽는 법")
    note("X 는 캐스케이드 바닥까지의 거리. V 는 진행방향 테이커 물량, D 는 대기물량.")
    note("gamma = 1 이므로 '밀리는 거리가 V/D 에 1:1 로 비례' 한다.", color=INK)
    label("이론과의 대조 — 형태까지 맞는다")
    note("D(u) = A u^beta 를 흡수하며 밀린다면 gamma = 1/(1+beta) 여야 한다.")
    note("gamma 는 (0,1] 안에 있어야 하고, beta=0(가격축 균일)이면 gamma=1.")
    label("*** 독립 2회 측정이 일치한다 ***")
    note("  수익 회귀     gamma = 0.875 +- 0.078   21종 639이벤트 3.5년 (일자클러스터)", color=OK)
    note("  오더북 모양   gamma = 0.861            27클러스터 40시간 라이브 호가", color=OK)
    note("      Cum(u) = B u^kappa 를 밴드에 적합, gamma = 1/kappa, 적합 R2 = 0.99", color=OK)
    note("데이터도 방법도 완전히 다른데 0.875 vs 0.861 이다.", color=INK)
    note("=> 구조식의 **함수 형태**는 검증됐다. 실패한 것은 확률화 방식이다(18절).", color=INK)
    label("실측 (5종 252이벤트, 2023-01-02~2026-05-23, 표본외 126, 창길이 순환 제거)")
    note("  변수                              표본외 R2     Spearman")
    note("  사전 관측치만                        +0.009       +0.238", color=WARN)
    note("  log(V/D)  60분 고정창                +0.423       +0.676", color=OK)
    note("  log(V/D)   5분 고정창                +0.353       +0.630")
    note("  실현 OI감소 60분 고정                 +0.203       +0.487")
    label("=> 예측력의 전부가 V 에 있다")
    note("D 는 이미 30초 해상도로 보고 있다. 모르는 것은 V 하나이고,", color=INK)
    note("가격대별 청산맵 L(p) 가 그것을 주는 유일한 물건이다. 들어갈 자리는", color=INK)
    note("조건부 분포의 mu 하나로 특정됐다 (2.6절, PROB_MODEL.md 6절).", color=OK)
    label("한계 — 역인과")
    note("V 는 강제 + 재량 매도의 합이고 '가격이 밀려서 매도가 나왔다' 가 섞여 있다.", color=WARN)
    note("L(p) 는 강제분만 예보하므로 실현 성능은 0.009 와 0.423 사이 어딘가다.", color=WARN)
    rule()

    # ------------------------------------------------------------- 18
    head("18.  D 는 관측값이 아니라 확률변수다  [2026-08-02, 확률모델 실패의 원인]")
    eq(r"$\log X \;=\; a + \gamma\left(\log V - \log D_{\mathrm{eff}}\right),"
       r"\qquad D_{\mathrm{eff}} \;=\; D_{t}\cdot W$", size=15)
    label("무엇이 빠져 있었나")
    note("2.6 절 모델은 D 를 '관측된 상수' 로 놓고 불확실성을 전부 V 로 몰았다.")
    note("그런데 화면의 호가는 확정 물량이 아니다. 마켓메이커/봇이 넣었다 뺐다 한다.", color=INK)
    note("W = 캐스케이드 중 실제로 남아 있는 비율. 실측 1/W: 중앙 3.4배 / p90 17배,", color=WARN)
    note("     극단일(2025-10-10)에는 109~523배. 앞서 극단일 수치를 전형값처럼", color=WARN)
    note("     인용했던 것을 정정한다. W_pre 중앙 0.997 -> 트리거 시점은 안 얇다.", color=WARN)
    eq(r"$\log X = a + \gamma\log V - \gamma\log D_{t} - \gamma\log W$",
       size=14, gap=0.034, box=False)
    note("                                              ^^^^^^^^^^^  빠져 있던 항", color=WARN)
    label("분산도 두 성분이다")
    eq(r"$\sigma^{2} \;=\; \gamma^{2}\left(\sigma^{2}_{\log V}"
       r"\;+\; \sigma^{2}_{\log W} \;-\; 2\,\mathrm{Cov}\right)$",
       size=14, gap=0.034, box=False)
    note("내가 모형화한 것은 첫 항뿐이고, 둘째 항은 2~3자릿수로 변동한다.", color=WARN)
    label("나누기 형태가 무엇을 숨기는가 — 물리는 순차 흡수다")
    eq(r"$X = \inf\left\{u : \int_{0}^{u}\left[d_{\mathrm{rest}}(v)"
       r" + \Delta(v,t)\right]dv \;\geq\; V\right\}$", size=14, gap=0.034, box=False)
    note("V/D 는 이 식의 닫힌 해이고, 두 가정 위에 선다:")
    note("  (1) 호가 모양이 멱함수  -> 검증됨 (적합 R2 = 0.99, 16절)", color=OK)
    note("  (2) 캐스케이드 중 호가가 안 변한다 (delta=0)  -> 틀림", color=WARN)
    note("gamma 가 두 방법에서 일치한 것은 (1) 이 맞다는 증거다.", color=INK)
    note("(2) 를 뒤늦게 W 로 넣었는데 **스칼라 W 는 조잡하다** — 실제 대상은", color=INK)
    note("delta(v,t), 가격대마다 다르고 시간에 따라 변하는 함수다.", color=INK)
    note("=> 모형화할 확률적 대상은 delta(v,t) 다. 틱 예측이 아니라 분포면 된다.", color=OK)
    label("이 한 가지가 관측된 실패 둘을 설명한다")
    note("  sigma 모형이 역방향(0.78)  -> 지배적 불확실성이 delta 인데 사전특징에 없다", color=INK)
    note("  보정이 한 방향으로 깨짐     -> 그 항을 빼먹었으니 위치가 계통적으로 어긋난다", color=INK)
    label("*** 정정 — 9.2배를 W 로 귀속한 것은 과잉주장이었다 ***")
    note("u_pred 계산에 쓴 V 는 Bybit 청산만이다. Bybit 는 perp 의 10~15%,", color=WARN)
    note("청산은 전체 테이커 물량의 ~10%. 실제로 호가를 먹은 양은 한두 자릿수 크다.", color=WARN)
    note("9.2배는 호가철수와 V 과소측정이 섞인 값이고, 나는 전부 W 탓으로 돌렸다.", color=WARN)
    note("같은 이유로 'V/D=0.65% 라 분모가 허수' 도 성립하지 않는다.", color=WARN)
    note("재구성 지도로 '서 있는 연료' 를 재면 BTC 현재가 +-1% 안 \\$10.5M,", color=OK)
    note("같은 구간 호가 \\$20~40M -> V/D ~ 0.3. 두 자릿수 다르다. (20절)", color=OK)
    label("gamma 만 두 방법에서 일치한 것도 일관된다")
    note("W 는 곱셈 상수라 기울기를 안 건드리고 절편과 분산만 옮긴다.", color=OK)
    label("실측 근거 (Bybit 전건 39.9시간, 클러스터 27개, 2026-07-31~08-02)")
    note("  V/D(도달최심)  중앙 0.0065  p90 0.035  최대 0.076")
    note("  u_pred 중앙 0.018%   u_act 중앙 0.223%   배수 중앙 9.2")
    note("  corr(log V/D, log u_act) = -0.045  (n=21)")
    note("  *** 단 이 40시간에 캐스케이드가 없었다(최대 클러스터 \\$2.26M).", color=WARN)
    note("      V/D 범위가 좁아 상관 0 은 '관계 없음'이 아니라 '범위 제약'이다.", color=WARN)
    note("      Bybit 한 거래소분이라 전 거래소 합산이면 V/D 는 5~6% 일 수 있다.", color=WARN)
    label("=> 질문이 바뀐다")
    note("  이전:  V(강제매도량)를 예보할 수 있는가   -> L(p) 가 필요")
    note("  지금:  W(호가 잔존율)를 예보할 수 있는가  -> 30초 호가로 관측 가능", color=OK)
    note("그리고 이쪽이 '실시간으로 오더북을 대조한다' 는 목표 설계와 더 맞는다.", color=OK)
    rule()

    # ------------------------------------------------------------- 18.5
    head("18.5  W 는 잡음이 아니다 — L 과 얽혀 있다  [미검정]")
    eq(r"$\log X = a + \gamma\left(\log V(u) - \log D_{t} - \log W(u)\right),"
       r"\qquad V(u)=\int L(p)\,dp,\quad \partial W/\partial L < 0$", size=13)
    label("가설")
    note("시장에 'OI 가 두껍게 쌓인 쪽을 터뜨리려는' 유인이 있다면,", color=INK)
    note("L 이 두꺼운 가격대에서 호가가 더 많이 빠진다. 즉 W 와 L 이 독립이 아니다.", color=INK)
    label("그러면 두 항이 같은 방향으로 민다")
    note("  물량은 더 쏟아지고(V 증가)  동시에  받아줄 호가는 더 사라진다(W 감소).")
    note("  곱해진다. 이것이 V/D_t = 0.65% 인데 9.2배 밀리는 이유의 유력한 후보다.", color=OK)
    note("  그리고 sigma 모형이 실패한 이유이기도 하다 — W 가 L 에 조건부인데", color=OK)
    note("  내 사전 특징에 L 이 없었으니 잡힐 리가 없다.", color=OK)
    label("문헌 (둘 다 이미 참고문헌에 있다)")
    note("  Brunnermeier & Pedersen (2005)  Predatory Trading")
    note("  Osler (2005)  FX 스탑 클러스터와 가격 캐스케이드")
    label("비판적 단서 — 의도는 다를 수 있으나 예측은 같다")
    note("MM 은 스프레드로 벌지 방향으로 벌지 않는다. '터뜨린다'는 의도는 방향성", color=WARN)
    note("트레이더에 더 맞고, MM 쪽은 역선택 회피로 호가를 뺀다.", color=WARN)
    note("그런데 두 이야기의 관측 예측이 같다(OI 두꺼운 쪽 호가가 사라진다).", color=INK)
    note("어느 쪽이 맞는지 몰라도 검정은 성립한다 — 오히려 유리하다.", color=OK)
    label("=> L(p) 의 가치가 되살아난다")
    note("  이전: L(p) 는 V 만 예보. 그런데 실현청산은 호가의 0.65% 뿐이라 약하다", color=WARN)
    note("  지금: L(p) 가 V 와 W 를 동시에 예보. 곱해지므로 훨씬 강하다", color=OK)
    note("단 미검정이고, L(p) 와 전 깊이 북이 둘 다 쌓여야 검정할 수 있다.")
    label("전 깊이 북 — 웹소켓 diff 가 REST 의 세 한계를 동시에 푼다")
    note("  도달범위  BTC 0.19% / ETH 0.56% 너머 NaN   ->  전 깊이 (로컬 북 유지)")
    note("  주기      30초 (캐스케이드는 몇 초)          ->  100ms")
    note("  취소      스냅샷 차이만 보임                 ->  호가 철수를 직접 관측", color=OK)
    note("셋째가 핵심이다. W 를 추정하는 게 아니라 실제로 보게 된다.", color=OK)
    note("L(p) 와 같은 성질 — 과거 백필 불가, 지금부터 쌓아야 한다.", color=WARN)
    rule()

    # ------------------------------------------------------------- 20
    head("20.  대안 A — 공개 데이터로 청산맵을 재구성한다  [지금 만드는 것]")
    eq(r"$\hat{L}(p,t) \;=\; \sum_{\tau<t}\;\Delta OI^{+}(\tau)\,p(\tau)"
       r"\;\times\; f_{R}\!\left(\frac{p}{p(\tau)}\right)\;\times\; S(t-\tau)$", size=14)
    label("읽는 법")
    note("시각 tau 에 가격 p(tau) 에서 열린 물량이 청산거리 분포 f_R 에 따라 흩어진다.")
    note("그걸 과거 전체에 대해 더하면 현재 시점의 지도가 된다.")
    label("왜 이 경로인가 — HL 실측 경로가 막혔다")
    note("Q1 실측(44.7시간, 실현청산 3,578건): 실현 청산 가격대에서 지도가 빈 비율", color=WARN)
    note("**70.0%**. phi~0.05 와 정확히 일치한다. 표본 문제가 아니라 구조적 커버리지다.", color=WARN)
    note("재구성은 바이낸스 자체 OI/가격을 쓰므로 시장 전체를 보고, 2020-09 로 소급된다.", color=OK)
    label("교과서 공식을 쓰지 않는다")
    note("p_liq = p_entry (1 - 1/L) 은 |오차|<10% 가 19.3% 뿐이다(유지증거금/펀딩).", color=WARN)
    note("f_R 경험분포를 통째로 쓰면 레버리지를 명시적으로 다룰 필요가 없다.", color=OK)
    label("f_R 실측 (HL 격리, 명목가 가중, 243K행)")
    note("  격리 중앙 청산거리 15.7%  vs  교차 35.3%   -> 격리가 근접 연료")
    note("  롱 22.3%  vs  숏 7.9%    -> 따로 모형화해야 한다")
    note("  다봉이다 — 이산 레버리지 계층(3~5배=20~35%, 20~33배=3~5%)에 뭉친다", color=INK)
    note("  시간 변동계수 0.136 (통과)  /  코인 변동계수 0.583 (코인별로 재서 쓰면 됨)")
    note("  근접 물량은 고레버리지: 0~2% 구간 중앙 30배, 20%+ 구간 3배 (단조)", color=OK)
    label("정규화")
    note("sum_p L_hat = OI(t) x 격리비중.  이 제약이 S 의 수준을 고정한다.")
    label("첫 결과 (BTC 2026-07-30, 현재가 64,750)")
    note("  현재가 -1% 안 롱청산 연료 \\$10.5M,  같은 구간 호가 \\$20~40M -> V/D ~ 0.3", color=OK)
    note("  Bybit 실현 클러스터의 0.0065 와 두 자릿수 다르다 — 당연하다.", color=INK)
    note("  지도는 '터질 수 있는 물량', 실현치는 '한 거래소에서 실제 터진 것'.", color=INK)
    label("최대 위험 — f_R 이 바이낸스를 대표하나")
    note("HL 은 교차마진 94.5% 로 정교한 고래 중심이다. 코인별 형태가 뒤집혀 있다:", color=WARN)
    note("  HL: BTC/ETH 7.9~8.7%(고레버리지)  SOL/LIT/FARTCOIN 31~47%(저레버리지)", color=WARN)
    note("  바이낸스 리테일은 알트에 고배율이 흔하다 — 반대일 가능성이 높다.", color=WARN)
    note("=> 정본은 Bybit 실현청산 역산. f_R 만 갈아끼우면 된다. (축적 중 44시간)", color=OK)
    rule()

    # ------------------------------------------------------------- 15
    head("21.  전체를 한 장으로")
    note("EV(u) = S(u) [ r0 + rho x pi(u) ] - c_fee   에서 깰 수 있는 자리는 두 곳뿐이다.",
         color=INK, size=9.8)
    drop(0.006)
    note("  S(u)   체결확률(무조건부) ->  불가. 청산이 경로를 안 민다 (11절, b=0.04)", color=WARN)
    note("  pi(u)  상대 정보성        ->  경로 하나. L-hat 으로 추정한다 (9절)", color=OK)
    drop(0.006)
    note("2026-08-02 에 셋째 자리를 열었다가 다시 닫혔다:", color=ACCENT, size=9.8)
    note("  P(X>=u | F_t)  조건부 체결확률 -> 검정함. 보정/판별 둘 다 불합격 (2.6절)",
         color=WARN)
    note("  단 실패 원인이 진단됐다 -> D 를 상수로 놓은 것 (18절). 형태는 살아있다", color=OK)
    note("  구조식 log X = a + gamma log(V/D) 는 독립 2회로 검증 (16절)", color=OK)
    drop(0.006)
    note("pi 를 쓰려면 먼저 L-hat 이 시장 전체를 대표하는지 확인해야 한다 (8절).")
    note("그것이 Q1 이고, 지금 hl_positions 가 그 데이터를 모으는 중이다.")
    drop(0.008)
    note("검정 현황:", color=ACCENT, size=9.8)
    note("  Q1        L-hat 대표성                              문지기. 데이터 대기")
    note("  Q2a       수급 청산 회계 (일회 예측)                  실패 (13절)", color=WARN)
    note("  Q2b       E[r|fill] 이 L-hat 의 증가함수인가          핵심. Q1 통과 후", color=INK)
    note("  Q2c       가격대별 히트맵 + 순차 갱신                 미검정. Q1 통과 후")
    note("  Q3        pi 조건부 EV > 33~38bp/이벤트              Q2 통과 후")
    note("  Q4        극단 V/D 에서 충격항이 살아나는가           꺾임 증거 없음 (10-11절)", color=OK)
    note("  Q5        Sharpe 도 평평한가                         통과 (5절)", color=OK)
    note("  Q6        사다리 손익이 숏풋인가                      반증 (6절)", color=OK)
    note("  QX-점     X 의 점추정으로 배치                        실패 (2.6절)", color=WARN)
    note("  QX-분포   X 의 조건부 분포 (D 상수 가정)              실패. 보정/판별 (2.6절)", color=WARN)
    note("  QW        W 를 사전 관측치로 예보                     실패. 예보분이 이미 특징에 있음",
         color=WARN)
    note("  Q-delta   delta(v,t) 분포를 모형화                    웹소켓 필요. 미착수", color=INK)
    note("  A-1       재구성 지도가 실현청산 위치를 맞히는가        **다음 검정** (20절)", color=OK)
    note("  A-2       재구성 지도로 X 예보 (6년 백테스트)          A-1 통과 후", color=OK)
    drop(0.006)
    note("*** 위 실패 셋(QX-점/QX-분포/QW)은 전부 L(p) 가 없어서 **대용품**으로 한 것이다.",
         color=INK)
    note("    그 실패는 설계를 반증하지 않는다 — '대용품이 없다' 는 것만 보여준다.", color=INK)
    note("    설계 자체는 아직 검정된 적이 없다. 지금 위치는 STATUS.md.", color=INK)

    # 박스 렌더
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t, dead in boxed:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(inv)
        pady = 0.008 * GS
        ax.add_patch(FancyBboxPatch(
            (0.070, bb.y0 - pady), 0.86, bb.height + 2 * pady,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=1.0,
            edgecolor=(DEADED if dead else BOXED),
            facecolor=(DEADBG if dead else BOXBG),
            zorder=0, clip_on=False))
        if dead:
            ax.plot([bb.x0 - 0.012, bb.x1 + 0.012],
                    [bb.y0 + bb.height / 2] * 2,
                    lw=1.5, color=WARN, alpha=0.8, zorder=4, clip_on=False)

    span = 0.985 - y[0]
    print("레이아웃: 내용 %.3f x PPU %.1f = %.1f 인치 (폭 %.1f, 비율 1:%.2f)"
          % (span, PPU, span * PPU, FIG_W, span * PPU / FIG_W))
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="render annotated math sheet to PNG")
    ap.add_argument("--out", default=os.path.join(C.ROOT, "MATH.png"))
    ap.add_argument("--dpi", type=int, default=210)
    a = ap.parse_args()
    render(a.out, a.dpi)
    print("wrote %s (%.0f KB)" % (a.out, os.path.getsize(a.out) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
