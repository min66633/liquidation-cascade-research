# -*- coding: utf-8 -*-
"""Scan accumulated depth_ws (1s mids) for cascade segments.

Definitions from STATUS.md usage:
  A) 60s return <= -100bp   ("급락 구간" validation definition)
  B) 15min return <= -2%    (baseline trigger scale)
Dedupe: merge hits within 10 min per symbol. Output counts + worst events.
"""
import io, sys, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

BASE = r"C:\Quant\final\strateges\liquidation_cascade_research\data\depth_ws"
days = sorted(d for d in os.listdir(BASE) if d.startswith("2026"))
print(f"days: {days[0]} ~ {days[-1]} ({len(days)})", flush=True)

frames = []
for day in days:
    for f in sorted(glob.glob(os.path.join(BASE, day, "*.parquet"))):
        try:
            frames.append(pd.read_parquet(f, columns=["ts_ms", "symbol", "mid"]))
        except Exception:
            pass
data = pd.concat(frames, ignore_index=True)
print(f"rows: {len(data):,}, symbols: {data['symbol'].nunique()}", flush=True)

events = []
for sym, d in data.groupby("symbol"):
    d = d.drop_duplicates("ts_ms").sort_values("ts_ms")
    t = d["ts_ms"].to_numpy(np.int64)
    m = d["mid"].to_numpy(float)
    n = len(d)
    if n < 5000:
        continue
    for label, win_ms, thr in [("60s<=-100bp", 60_000, -0.01),
                               ("15m<=-2%", 900_000, -0.02)]:
        j = np.searchsorted(t, t - win_ms)
        jc = np.clip(j, 0, n - 1)
        valid = (t - t[jc]) <= win_ms + 5000
        ret = np.where(valid & (m[jc] > 0), m / np.maximum(m[jc], 1e-12) - 1, np.nan)
        hit = np.where(np.isfinite(ret) & (ret <= thr))[0]
        if len(hit) == 0:
            continue
        # dedupe within 10 min
        groups = []
        start = hit[0]
        prev = hit[0]
        for i in hit[1:]:
            if t[i] - t[prev] > 600_000:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        for a, b in groups:
            seg = ret[a:b + 1]
            worst = float(np.nanmin(seg))
            events.append({"sym": sym, "def": label,
                           "start": pd.to_datetime(t[a], unit="ms"),
                           "worst": worst,
                           "dur_min": (t[b] - t[a]) / 60000})

ev = pd.DataFrame(events)
if len(ev) == 0:
    print("NO cascade segments under either definition.")
else:
    for label, g in ev.groupby("def"):
        print(f"\n[{label}] segments: {len(g)} across {g['sym'].nunique()} symbols")
        top = g.nsmallest(12, "worst")
        for _, r in top.iterrows():
            print(f"  {r['sym']:>10} {r['start']}  worst={r['worst']*100:+.2f}% "
                  f"dur={r['dur_min']:.0f}min")
    ev.to_csv("cascade_segments.csv", index=False)
    print("\nsaved cascade_segments.csv")
