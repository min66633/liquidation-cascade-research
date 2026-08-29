# -*- coding: utf-8 -*-
"""D-9 핵심: ②③ 판정을 캐스케이드 구간 조건부로 분리 추정.

book_prob.build() 를 그대로 임포트해 동일 사양(스태킹·Deff·reg·cxl)으로
패널을 만들고, cascade_segments.csv(15m<=-2%, 23건)의 [시작, 시작+지속+15분]
구간을 in-cascade 로 마킹. 평시 vs 캐스케이드 부분표본에서 같은 회귀:
    log X = a + b1 log sig + b2 log D + b3 reg + b4 cxl + b5 imb
클러스터: 평시=일자, 캐스케이드=세그먼트ID (중첩 전방창 보정).
추가: 평시로 학습한 분위모형의 캐스케이드 구간 위반율(꼬리 이전성).
"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LIQ = r"C:\Quant\final\strateges\liquidation_cascade_research"
sys.path.insert(0, LIQ)
os.chdir(LIQ)

import numpy as np
import pandas as pd
from analysis.book_prob import build
from analysis.response_liq import ols_cluster

SCRATCH = r"C:\Users\user\AppData\Local\Temp\claude\C--Quant-final-strateges-volatility-trading\6bb26e42-6852-4d76-990a-6f7a86eb0145\scratchpad"
seg = pd.read_csv(os.path.join(SCRATCH, "cascade_segments.csv"), parse_dates=["start"])
seg = seg[seg["def"] == "15m<=-2%"].reset_index(drop=True)
seg["t0"] = seg["start"].astype("int64") // 10**9
seg["t1"] = seg["t0"] + (seg["dur_min"] * 60).astype(int) + 900
print(f"cascade segments: {len(seg)}", flush=True)

d = build("b0_5", [60])
d = d.dropna(subset=["sig", "xd60"])
d = d[(d["sig"] > 0) & (d["D_bid"] > 0) & (d["D_ask"] > 0)]
print(f"panel: {len(d):,} sec-obs", flush=True)

parts = []
for lab, xc, Dc, Ac, Rc in (("down", "xd60", "D_bid", "AddB", "RemB"),
                            ("up", "xu60", "D_ask", "AddA", "RemA")):
    p = pd.DataFrame({
        "X": d[xc].to_numpy(), "D": d[Dc].to_numpy(),
        "Add": d[Ac].to_numpy(), "Rem": d[Rc].to_numpy(),
        "sig": d["sig"].to_numpy(),
        "imb": (d["D_bid"] - d["D_ask"]).to_numpy()
               / np.maximum((d["D_bid"] + d["D_ask"]).to_numpy(), 1e-9),
        "sec": d["sec"].to_numpy(), "symbol": d["symbol"].to_numpy(),
    })
    if lab == "up":
        p["imb"] = -p["imb"]
    parts.append(p)
del d
p = pd.concat(parts, ignore_index=True)
del parts
p = p[np.isfinite(p["X"]) & (p["X"] > 0) & np.isfinite(p["Add"])]
p["reg"] = p["Add"] / np.maximum(p["D"], 1e-9)
p["cxl"] = p["Rem"] / np.maximum(p["D"], 1e-9)

# cascade marking (symbol + time interval)
p["casc"] = -1
sec = p["sec"].to_numpy()
sym = p["symbol"].to_numpy()
casc = np.full(len(p), -1, np.int64)
for k, r in seg.iterrows():
    m = (sym == r["sym"]) & (sec >= r["t0"]) & (sec <= r["t1"])
    casc[m] = k
p["casc"] = casc
n_c = int((casc >= 0).sum())
print(f"in-cascade obs: {n_c:,} ({n_c/len(p):.3%})", flush=True)


def run(sub, groups, name):
    y = np.log(sub["X"].to_numpy())
    X = np.column_stack([
        np.ones(len(sub)),
        np.log(sub["sig"].to_numpy()),
        np.log(sub["D"].to_numpy()),
        sub["reg"].to_numpy(),
        sub["cxl"].to_numpy(),
        sub["imb"].to_numpy(),
    ])
    b, se, _ = ols_cluster(X, y, groups)
    t = b / se
    names = ["const", "log sig", "log D", "회복력", "취소압", "불균형"]
    print(f"\n[{name}] n={len(sub):,} clusters={len(np.unique(groups))}")
    for i in range(1, 6):
        print(f"  {names[i]:>6}: b={b[i]:+.4f} t={t[i]:+6.2f}")
    return b


calm = p[p["casc"] < 0]
cas = p[p["casc"] >= 0]
b_calm = run(calm, (calm["sec"].to_numpy() // 86400), "평시 (일자 클러스터)")
b_cas = run(cas, cas["casc"].to_numpy(), "캐스케이드 (세그먼트 클러스터)")

# tail transfer: calm-trained quantile model -> cascade violation rates
ycalm = np.log(calm["X"].to_numpy())
Xcalm = np.column_stack([np.ones(len(calm)), np.log(calm["sig"]), np.log(calm["D"]),
                         calm["reg"], calm["cxl"], calm["imb"]])
resid = ycalm - Xcalm @ b_calm
qs = np.quantile(resid, [0.10, 0.25, 0.50, 0.75, 0.90])
ycas = np.log(cas["X"].to_numpy())
Xcas = np.column_stack([np.ones(len(cas)), np.log(cas["sig"]), np.log(cas["D"]),
                        cas["reg"], cas["cxl"], cas["imb"]])
mcas = Xcas @ b_calm
print("\n[꼬리 이전성] 평시 학습 분위 -> 캐스케이드 위반율 (목표=1-p):")
for pv, q in zip([0.10, 0.25, 0.50, 0.75, 0.90], qs):
    viol = float((ycas > mcas + q).mean())
    print(f"  p={pv:.2f}: 위반 {viol:.3f} (목표 {1-pv:.2f}, 편차 {viol-(1-pv):+.3f})")
