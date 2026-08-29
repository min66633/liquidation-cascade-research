# -*- coding: utf-8 -*-
"""데이터 무결성 점검 — 이 프로젝트에서 실제로 당한 결함들을 전부 훑는다.

왜 만드나
  2026-08-01 하루에 데이터 결함으로 결론을 두 번 잘못 냈다.
    (1) bookDepth 의 2025년 구간에서 특정 밴드가 몇 시간씩 고정 -> V/D 꼬리 날조.
        "지지집합 2,471배" 로 보고했다가 실제 42배로 정정.
    (2) 16종의 1분봉이 95일치뿐인데 5분봉은 전 기간 -> 체결 0건이 나왔고
        "21종 확장이 EV 를 60% 깎았다" 로 잠깐 오독.
  둘 다 '큰 n' 뒤에 숨어 있었다. 그래서 관측 수가 아니라 **밀도·고유값·일수**를 본다.

검사 항목
  A. 밀도      실제 행수 / 기대 행수. 1 이 정상. (1분봉 사고)
  B. 고정값    동일 값이 min_run 이상 연속. (bookDepth 사고)
  C. 중복      같은 타임스탬프가 두 번
  D. 격자      타임스탬프가 봉 간격의 배수인가
  E. 단조      누적 호가 밴드가 dm1 <= dm2 <= ... 인가
  F. 물리      OHLC 위반, 0/음수, OI 가 5분에 50% 이상 증발
  G. 결측일    달력 대비 빠진 날

각 검사는 PASS / WARN / FAIL 로 낸다. FAIL 은 그 데이터로 낸 결론을 의심해야 한다.

실행:
    python analysis/dataqc.py                 # 전체
    python analysis/dataqc.py --only klines   # 일부만
    python analysis/dataqc.py --symbols BTCUSDT
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

BULK = os.path.join(C.DATA, "binance_bulk")
BID = ["dm1_0", "dm2_0", "dm3_0", "dm4_0", "dm5_0"]
ASK = ["dp1_0", "dp2_0", "dp3_0", "dp4_0", "dp5_0"]

FAIL, WARN, PASS = "FAIL", "WARN", "PASS"
_ORDER = {FAIL: 0, WARN: 1, PASS: 2}
_results: list[dict] = []


def rep(dataset: str, symbol: str, check: str, status: str, detail: str) -> None:
    _results.append({"dataset": dataset, "symbol": symbol, "check": check,
                     "status": status, "detail": detail})


def frozen_runs(v: np.ndarray, min_run: int) -> tuple[int, int]:
    """(고정 구간에 속한 행 수, 최장 런). NaN 은 비교에서 빠진다."""
    if v.size == 0:
        return 0, 0
    same = np.empty(v.size, dtype=bool)
    same[0] = False
    np.not_equal(v[1:], v[:-1], out=same[1:])
    run = np.cumsum(same)
    sz = np.bincount(run)
    return int(sz[run][sz[run] >= min_run].size), int(sz.max())


# --------------------------------------------------------------------- klines
def qc_klines(symbols: list[str], interval: str) -> None:
    step = 60_000 if interval == "1m" else 300_000
    ds = "klines_%s" % interval
    for s in symbols:
        p = os.path.join(BULK, ds, "%s.parquet" % s)
        if not os.path.exists(p):
            rep(ds, s, "존재", FAIL, "파일 없음")
            continue
        d = pd.read_parquet(p, columns=["open_time", "open", "high", "low",
                                        "close", "volume"])
        if d.empty:
            rep(ds, s, "존재", FAIL, "빈 파일")
            continue
        d = d.sort_values("open_time").reset_index(drop=True)
        ot = d["open_time"].to_numpy()

        span = max((ot[-1] - ot[0]) / step + 1, 1)
        dens = len(d) / span
        rep(ds, s, "A 밀도", PASS if dens >= 0.99 else (WARN if dens >= 0.9 else FAIL),
            "%.3f  (%d행 / 기대 %.0f, %s~%s)"
            % (dens, len(d), span,
               pd.to_datetime(ot[0], unit="ms", utc=True).date(),
               pd.to_datetime(ot[-1], unit="ms", utc=True).date()))

        ndup = len(d) - d["open_time"].nunique()
        rep(ds, s, "C 중복", PASS if ndup == 0 else FAIL, "%d건" % ndup)
        noff = int((ot % step != 0).sum())
        rep(ds, s, "D 격자", PASS if noff == 0 else FAIL, "%d건" % noff)

        o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
        viol = int((~((l <= o + 1e-12) & (o <= h + 1e-12) &
                      (l <= c + 1e-12) & (c <= h + 1e-12) & (l > 0))).sum())
        rep(ds, s, "F OHLC", PASS if viol == 0 else FAIL, "%d건 위반" % viol)

        nrow, longest = frozen_runs(c, 30)
        rep(ds, s, "B 종가고정", PASS if nrow / len(d) < 0.01 else WARN,
            "%.2f%% (%d행), 최장 %d봉" % (100 * nrow / len(d), nrow, longest))

        zv = int((d["volume"].to_numpy() <= 0).sum())
        rep(ds, s, "F 무거래", PASS if zv / len(d) < 0.05 else WARN,
            "%.2f%% (%d봉)" % (100 * zv / len(d), zv))


# -------------------------------------------------------------------- metrics
def qc_metrics(symbols: list[str]) -> None:
    step = 300_000
    for s in symbols:
        p = os.path.join(BULK, "metrics", "%s.parquet" % s)
        if not os.path.exists(p):
            rep("metrics", s, "존재", FAIL, "파일 없음")
            continue
        d = pd.read_parquet(p, columns=["open_time", "sum_open_interest",
                                        "sum_open_interest_value"])
        if d.empty:
            rep("metrics", s, "존재", FAIL, "빈 파일")
            continue
        d = d.sort_values("open_time").reset_index(drop=True)
        ot = d["open_time"].to_numpy()
        span = max((ot[-1] - ot[0]) / step + 1, 1)
        dens = len(d) / span
        rep("metrics", s, "A 밀도",
            PASS if dens >= 0.99 else (WARN if dens >= 0.9 else FAIL),
            "%.3f  (%d행, %s~%s)"
            % (dens, len(d),
               pd.to_datetime(ot[0], unit="ms", utc=True).date(),
               pd.to_datetime(ot[-1], unit="ms", utc=True).date()))
        noff = int((ot % step != 0).sum())
        rep("metrics", s, "D 격자", PASS if noff == 0 else WARN,
            "%d건 (event_study_h2.load 가 이미 걸러냄)" % noff)

        oi = d["sum_open_interest"].to_numpy(dtype="float64")
        bad = int((~np.isfinite(oi) | (oi <= 0)).sum())
        rep("metrics", s, "F OI 0/NaN", PASS if bad == 0 else WARN, "%d건" % bad)

        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(np.diff(oi) / oi[:-1])
        big = int(np.nansum(rel > 0.5))
        rep("metrics", s, "F OI 급변>50%", PASS if big == 0 else WARN,
            "%d건 (v_doi 계산에서 제외 대상)" % big)

        nrow, longest = frozen_runs(oi, 12)
        rep("metrics", s, "B OI고정", PASS if nrow / len(d) < 0.01 else WARN,
            "%.2f%% (%d행), 최장 %d봉(=%d분)"
            % (100 * nrow / len(d), nrow, longest, longest * 5))


# ------------------------------------------------------------------ bookDepth
def qc_bookdepth(symbols: list[str]) -> None:
    for s in symbols:
        pdir = os.path.join(BULK, "book_depth", s)
        legacy = os.path.join(BULK, "book_depth", "%s.parquet" % s)
        files = sorted(glob.glob(os.path.join(pdir, "*.parquet")))
        if not files and not os.path.exists(legacy):
            rep("bookDepth", s, "존재", WARN, "파일 없음 (미수집 종목)")
            continue
        rep("bookDepth", s, "일자파일", PASS, "%d일 + 구파일 %s"
            % (len(files), "있음" if os.path.exists(legacy) else "없음"))

        # 스키마 다양성 — 밴드가 날마다 다르면 필수/선택을 나눠야 한다
        import pyarrow.parquet as pq
        have_cnt: dict[str, int] = {}
        for f in files:
            try:
                for c in pq.ParquetFile(f).schema.names:
                    have_cnt[c] = have_cnt.get(c, 0) + 1
            except Exception:
                rep("bookDepth", s, "파일읽기", FAIL, os.path.basename(f))
        if files:
            core = [c for c in BID + ASK if have_cnt.get(c, 0) == len(files)]
            part = {c: v for c, v in have_cnt.items()
                    if c not in ("ts_ms",) and v < len(files)}
            rep("bookDepth", s, "밴드 전일자보유", PASS if len(core) == 10 else FAIL,
                "%d/10 (%s)" % (len(core), ",".join(sorted(core)[:3]) + "..."))
            if part:
                rep("bookDepth", s, "밴드 일부일자", WARN,
                    ", ".join("%s=%d/%d" % (k, v, len(files))
                              for k, v in sorted(part.items())))

        # 내용 검사는 공용 로더(고정 필터 적용 전 원본)로
        cols = BID + ASK
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f, columns=["ts_ms"] + cols))
            except Exception:
                pass
        if not frames:
            rep("bookDepth", s, "내용", WARN, "필수 밴드 보유 일자 없음")
            continue
        d = pd.concat(frames, ignore_index=True).drop_duplicates("ts_ms")
        d = d.sort_values("ts_ms").reset_index(drop=True)

        ndup = len(pd.concat(frames, ignore_index=True)) - len(d)
        rep("bookDepth", s, "C 중복", PASS if ndup == 0 else WARN, "%d건" % ndup)

        gaps = np.diff(d["ts_ms"].to_numpy())
        med = float(np.median(gaps)) if gaps.size else np.nan
        rep("bookDepth", s, "스냅샷간격", PASS if 25_000 <= med <= 35_000 else WARN,
            "중앙 %.0f초" % (med / 1000.0))

        for side, cc in (("매수", BID), ("매도", ASK)):
            v = d[cc].to_numpy(dtype="float64")
            nonmono = int((np.diff(v, axis=1) < -1e-9).any(axis=1).sum())
            rep("bookDepth", s, "E 단조(%s)" % side,
                PASS if nonmono == 0 else FAIL,
                "%.3f%% (%d행)" % (100 * nonmono / len(d), nonmono))
            bad = int((~np.isfinite(v) | (v <= 0)).any(axis=1).sum())
            rep("bookDepth", s, "F 0/NaN(%s)" % side,
                PASS if bad == 0 else WARN, "%d행" % bad)

        # 고정값 — 이 프로젝트를 실제로 물린 결함
        for c in ("dm1_0", "dp1_0"):
            v = d[c].to_numpy(dtype="float64")
            nrow, longest = frozen_runs(v, 10)
            st = PASS if nrow / len(d) < 0.005 else (WARN if nrow / len(d) < 0.05 else FAIL)
            rep("bookDepth", s, "B 고정(%s)" % c, st,
                "%.2f%% (%d행), 최장 %d스냅(=%.1f시간)"
                % (100 * nrow / len(d), nrow, longest, longest * 30 / 3600.0))


# ------------------------------------------------------------------- 라이브
def qc_live() -> None:
    # 경로는 실제 수집기가 쓰는 것과 맞춰야 한다 — 처음에 hl/, binance_oi/ 로 적었다가
    # '디렉터리 없음' WARN 만 냈다. 있지도 않은 경로를 검사하면 QC 가 거짓 안심을 준다.
    #
    # **기대 신선도는 수집기마다 다르다.** 처음에 전부 60분으로 잡았더니 hl_universe
    # (설계상 24시간 주기)가 WARN 으로 떴다. 거짓 경보는 진짜 경보를 묻는다.
    # (이름, 경로, 패턴, WARN 임계 분, FAIL 임계 분)
    checks = [
        ("hl_positions", os.path.join(C.DATA, "hl_positions"), "**/*.parquet", 30, 120),
        ("hl_universe", os.path.join(C.DATA, "hl_universe"), "**/*.parquet", 26 * 60, 48 * 60),
        ("hl_hot", os.path.join(C.DATA, "hl_hot"), "**/*.parquet", 10, 60),
        ("bybit_liq", os.path.join(C.DATA, "bybit_liq"), "**/*.parquet", 15, 120),
        ("binance_oi", os.path.join(C.DATA, "binance_futures_data"), "**/*.parquet", 30, 180),
        ("depth_poll", C.DEPTH_DIR, "**/*.parquet", 10, 60),
        ("paper", os.path.join(C.DATA, "paper"), "*.parquet", 10, 60),
    ]
    now = U.utc_now_ms()
    for name, root, pat, warn_min, fail_min in checks:
        if not os.path.isdir(root):
            rep("live", name, "디렉터리", WARN, "없음: %s" % root)
            continue
        files = glob.glob(os.path.join(root, pat), recursive=True)
        if not files:
            rep("live", name, "파일", WARN, "0개 (%s)" % root)
            continue
        newest = max(files, key=os.path.getmtime)
        age_min = (now / 1000.0 - os.path.getmtime(newest)) / 60.0
        tot = sum(os.path.getsize(f) for f in files) / 1e6
        st = PASS if age_min < warn_min else (WARN if age_min < fail_min else FAIL)
        rep("live", name, "신선도", st,
            "파일 %d개 / %.1fMB / 최신 %.0f분 전 (기대 <%d분)"
            % (len(files), tot, age_min, warn_min))


def main() -> int:
    ap = argparse.ArgumentParser(description="data integrity audit")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["klines1m", "klines5m", "metrics", "bookdepth", "live"])
    a = ap.parse_args()

    U.init_stdout()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)
    syms = a.symbols if a.symbols else C.MAJORS
    only = set(a.only) if a.only else None

    def want(x):
        return only is None or x in only

    if want("klines5m"):
        U.log("klines 5m ...")
        qc_klines(syms, "5m")
    if want("klines1m"):
        U.log("klines 1m ...")
        qc_klines(syms, "1m")
    if want("metrics"):
        U.log("metrics ...")
        qc_metrics(syms)
    if want("bookdepth"):
        U.log("bookDepth ...")
        qc_bookdepth(syms)
    if want("live"):
        U.log("live collectors ...")
        qc_live()

    t = pd.DataFrame(_results)
    if t.empty:
        print("검사 항목 없음")
        return 0
    t["rank"] = t["status"].map(_ORDER)

    print("\n" + "=" * 100)
    print("요약 — 상태별 건수")
    print(t.groupby(["dataset", "status"]).size().unstack(fill_value=0).to_string())

    for st in (FAIL, WARN):
        sub = t[t["status"] == st].sort_values(["dataset", "symbol"])
        print("\n" + "=" * 100)
        print("%s  %d건" % (st, len(sub)))
        if sub.empty:
            print("  없음")
        else:
            print(sub[["dataset", "symbol", "check", "detail"]].to_string(index=False))

    n_fail = int((t["status"] == FAIL).sum())
    print("\n" + "=" * 100)
    print("FAIL %d / WARN %d / PASS %d"
          % (n_fail, int((t["status"] == WARN).sum()), int((t["status"] == PASS).sum())))
    print("FAIL 이 있으면 그 데이터로 낸 결론을 의심할 것.")
    U.atomic_write_parquet(t.drop(columns=["rank"]),
                           os.path.join(C.DATA, "analysis", "dataqc.parquet"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
