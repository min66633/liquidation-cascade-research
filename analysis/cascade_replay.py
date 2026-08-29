# -*- coding: utf-8 -*-
"""First-pass D-9-lite on accumulated cascades (n=23, 15m<=-2% definition).

Per event: entry = mid at trigger moment t0 (first -2%/15m hit).
Outcomes: ret @ t0+15m, +60m; max rebound and max further drawdown within 60m.
Onset book state: bid_b2 / ask_b2 USD depth at t0, and bid_b2 change over
[t0, t0+60s] (crude inflow proxy - NOT the formal kernel; label first-pass).
Correlations (Spearman) of onset depth/inflow vs outcomes.
NOTE: informal first pass; the formal re-judgment is the project's own
pipeline (DESIGN_LOCK Section 5). n=23 - directional evidence only.
"""
import io, sys, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

BASE = r"C:\Quant\final\strateges\liquidation_cascade_research\data\depth_ws"
ev = pd.read_csv("cascade_segments.csv", parse_dates=["start"])
ev = ev[ev["def"] == "15m<=-2%"].reset_index(drop=True)
print(f"events: {len(ev)}")

need = {}
for _, r in ev.iterrows():
    d = r["start"].strftime("%Y-%m-%d")
    need.setdefault(d, set()).add(r["sym"])
    d2 = (r["start"] + pd.Timedelta(minutes=75)).strftime("%Y-%m-%d")
    need.setdefault(d2, set()).add(r["sym"])

cols = ["ts_ms", "symbol", "mid", "bid_b2", "ask_b2"]
frames = []
for d, syms in need.items():
    p = os.path.join(BASE, d)
    if not os.path.isdir(p):
        continue
    for f in sorted(glob.glob(os.path.join(p, "*.parquet"))):
        try:
            df = pd.read_parquet(f, columns=cols)
            frames.append(df[df["symbol"].isin(syms)])
        except Exception:
            pass
data = pd.concat(frames, ignore_index=True)
print(f"loaded rows: {len(data):,}")

rows = []
for _, r in ev.iterrows():
    sym = r["sym"]
    t0 = int(r["start"].value // 10**6)
    d = data[data["symbol"] == sym].drop_duplicates("ts_ms").sort_values("ts_ms")
    t = d["ts_ms"].to_numpy(np.int64)
    m = d["mid"].to_numpy(float)
    bb = d["bid_b2"].to_numpy(float)
    i0 = np.searchsorted(t, t0)
    if i0 >= len(t) or abs(t[min(i0, len(t)-1)] - t0) > 5000:
        continue
    e = m[i0]

    def at(ms):
        j = np.searchsorted(t, t0 + ms)
        if j >= len(t) or t[j] - (t0 + ms) > 10000:
            return np.nan
        return m[j]

    m15, m60 = at(900_000), at(3_600_000)
    j60 = np.searchsorted(t, t0 + 3_600_000)
    seg = m[i0:min(j60, len(m))]
    rows.append({
        "sym": sym, "t0": r["start"],
        "ret15": m15 / e - 1 if np.isfinite(m15) else np.nan,
        "ret60": m60 / e - 1 if np.isfinite(m60) else np.nan,
        "max_reb": seg.max() / e - 1 if len(seg) else np.nan,
        "max_dd": seg.min() / e - 1 if len(seg) else np.nan,
        "depth0": bb[i0],
        "dflow60": (bb[min(np.searchsorted(t, t0 + 60_000), len(bb) - 1)] - bb[i0]) / max(bb[i0], 1e-9),
    })

df = pd.DataFrame(rows)
print(f"\nreplay n={len(df)}")


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


for k, nm in [("ret15", "+15min"), ("ret60", "+60min"),
              ("max_reb", "max rebound(60m)"), ("max_dd", "max drawdown(60m)")]:
    x = df[k].dropna()
    print(f"  {nm:>18}: mean={x.mean()*1e4:+7.1f}bp med={x.median()*1e4:+7.1f}bp "
          f"t={tstat(x):+5.2f} win={(x>0).mean():.0%} n={len(x)}")

from scipy.stats import spearmanr
for a, b in [("depth0", "ret60"), ("depth0", "max_dd"), ("dflow60", "ret60"), ("dflow60", "max_dd")]:
    s = df[[a, b]].dropna()
    if len(s) > 8:
        rho, p = spearmanr(s[a], s[b])
        print(f"  spearman({a},{b}): rho={rho:+.2f} p={p:.2f} n={len(s)}")

print("\nper-event (worst-to-best by ret60):")
for _, r in df.sort_values("ret60").iterrows():
    print(f"  {r['sym']:>9} {r['t0']}  +15m {r['ret15']*1e4 if np.isfinite(r['ret15']) else float('nan'):+7.1f}bp "
          f"+60m {r['ret60']*1e4 if np.isfinite(r['ret60']) else float('nan'):+7.1f}bp "
          f"reb {r['max_reb']*1e4:+6.1f} dd {r['max_dd']*1e4:+7.1f}")
