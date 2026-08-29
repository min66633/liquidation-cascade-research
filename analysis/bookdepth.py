# -*- coding: utf-8 -*-
"""Binance bookDepth 정제 로더 — 2025년 고정(frozen) 필드를 걸러낸다.

왜 필요한가 (2026-08-01 코드리뷰에서 발견)
  data.binance.vision 의 bookDepth 는 2025년 구간에서 특정 percentage 행의 notional 이
  **몇 시간씩 동일 값으로 고정**된다. 실증:
    - 2025-10-10 BTCUSDT, percentage=-1: notional=2142547.0931 이 440회 연속,
      2142311.3113 이 512회 연속 (전체 2,879 스냅샷 중 952개)
    - 같은 구간에서 이를 포함하는 dm2_0 은 3배(XRP 는 219배) 움직였다.
      포함되는 밴드가 컨테이너가 219배 흔들리는 동안 비트 단위로 동일할 수는 없다.
    - 연도별 고정 비중: 2023 0.01%, 2024 0.00%, **2025 6.3~22.0%**, 2026 0.00%
  일부는 하루 전체가 상수다 — 2025-05-01(5종 전부), ETH 2025-11-01/03/21,
  XRP 2025-08-01(24시간 내내 $100.00399).

이걸 안 거르면 V/D 의 분모가 고정되어 비율 꼬리가 통째로 날조된다. 실측 영향:
  V/D>=10 구간 678건 중 648건이 고정 분모, "평상시 최대 = 상한의 523,277배"가 실제 640배.

거르는 기준 (둘 다 보수적으로 잡았다)
  1) 동일 값이 min_run(기본 10, = 5분) 이상 연속  -> 그 구간 전체 제거
  2) 하루 안에서 서로 다른 값이 min_uniq(기본 3) 미만 -> 그 날 전체 제거
  요청한 컬럼 중 **하나라도** 고정이면 그 행을 버린다(분모를 섞어 쓰기 때문).

비용은 정상 구간에서 1.3% 수준이다.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C

DEPTH_DIR = os.path.join(C.DATA, "binance_bulk", "book_depth")


def _read_raw(symbol: str, cols: list[str], optional: list[str]
              ) -> tuple[pd.DataFrame, list[str], int]:
    """일자 분할 파일 + (있으면) 구 단일 파일을 합쳐 읽는다.

    저장 구조가 2026-08-01 에 심볼당 단일 파일 -> 일자 분할로 바뀌었다.
    둘 다 읽고 ts_ms 로 중복 제거하므로 마이그레이션은 필요 없다.
    분할본을 뒤에 두어(keep="last") 같은 시각이면 분할본이 이긴다.

    **선택적 컬럼**: Binance 는 대부분의 날에 정수 percentage(-5..-1, 1..5)만 주고
    ±0.2 는 소수의 날에만 붙는다(실측 1,304일 중 197일). 단일 파일 시절에는 concat 이
    합집합을 만들어 이 사실이 안 보였는데, 일자 분할로 바꾸니 드러났다. ±0.2 를
    필수로 요구하면 85% 의 날이 버려진다 — 그래서 optional 로 받는다.
    """
    req = ["ts_ms"] + list(cols)
    full = req + list(optional)
    frames, bad, n_opt = [], [], 0
    legacy = os.path.join(DEPTH_DIR, "%s.parquet" % symbol)
    paths = ([legacy] if os.path.exists(legacy) else []) + \
            sorted(glob.glob(os.path.join(DEPTH_DIR, symbol, "*.parquet")))
    for p in paths:
        # 스키마를 먼저 한 번만 읽어 어떤 컬럼을 요청할지 정한다. 실패-후-재시도로
        # 하면 선택 컬럼이 없는 날마다 파일을 두 번 연다(실측: 심볼당 1,304 -> 2,400회).
        try:
            have = set(pq.ParquetFile(p).schema.names)
        except Exception:                      # noqa: BLE001 — 못 여는 파일은 버림
            bad.append(os.path.basename(p))
            continue
        if not set(req).issubset(have):
            bad.append(os.path.basename(p))
            continue
        cols_here = req + [c for c in optional if c in have]
        if len(cols_here) > len(req):
            n_opt += 1
        try:
            frames.append(pd.read_parquet(p, columns=cols_here))
        except Exception:                      # noqa: BLE001
            bad.append(os.path.basename(p))
    if not frames:
        return pd.DataFrame(), bad, 0
    # 컬럼이 다른 프레임을 concat 하면 없는 쪽은 NaN 으로 채워진다(의도된 동작)
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates(subset="ts_ms", keep="last"), bad, n_opt


def frozen_mask(d: pd.DataFrame, col: str, min_run: int, min_uniq: int,
                day_codes=None) -> np.ndarray:
    """True = 버려야 할 행.

    groupby.transform 대신 bincount 를 쓴다 — 360만 행 x 10컬럼에서 transform 은
    컬럼당 수 초가 걸려 로딩보다 오래 걸렸다(실측 20분 실행의 주범).
    """
    v = d[col].to_numpy()
    same = np.empty(len(v), dtype=bool)
    same[0] = False
    np.not_equal(v[1:], v[:-1], out=same[1:])
    run = np.cumsum(same)                      # 동일값 연속 구간 id
    stuck_run = np.bincount(run)[run] >= min_run

    if day_codes is None:
        day_codes = pd.factorize(
            pd.to_datetime(d["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d"))[0]
    nu = pd.Series(v).groupby(day_codes).nunique().to_numpy()
    stuck_day = nu[day_codes] < min_uniq
    return stuck_run | stuck_day


def load_clean(symbol: str, cols: list[str], *, optional: list[str] | None = None,
               min_run: int = 10, min_uniq: int = 3,
               verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """ts_ms + 요청 컬럼. 고정 구간 제거 후 반환.

    cols     : 필수. 하나라도 없는 파일은 버린다.
    optional : 있으면 싣고 없으면 NaN. 고정 필터와 dropna 의 대상이 아니다.
    반환: (df, stats).  파일이 없거나 필수 컬럼이 없으면 (빈 DataFrame, stats).
    """
    optional = list(optional or [])
    stats = {"symbol": symbol, "n_raw": 0, "n_drop": 0, "n_keep": 0,
             "days_raw": 0, "days_keep": 0, "missing": [], "n_optional": 0}
    d, bad, n_opt = _read_raw(symbol, cols, optional)
    stats["n_optional"] = n_opt
    if d.empty:
        stats["missing"] = bad or ["<no file>"]
        return pd.DataFrame(), stats
    if bad and verbose:
        print("  [bookDepth] %-9s 필수 컬럼 없어 건너뛴 파일 %d개" % (symbol, len(bad)))

    # 선택 컬럼은 NaN 이 정상이므로 dropna 대상에서 뺀다
    d = d.dropna(subset=["ts_ms"] + list(cols))
    for c in cols:
        d = d[d[c] > 0]
    d = d.sort_values("ts_ms").reset_index(drop=True)
    if d.empty:
        return d, stats

    day_all = pd.to_datetime(d["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    stats["n_raw"] = len(d)
    stats["days_raw"] = int(day_all.nunique())

    day_codes = pd.factorize(day_all)[0]
    bad_rows = np.zeros(len(d), dtype=bool)
    for c in cols:
        bad_rows |= frozen_mask(d, c, min_run, min_uniq, day_codes)
    d = d[~bad_rows].reset_index(drop=True)

    stats["n_drop"] = stats["n_raw"] - len(d)
    stats["n_keep"] = len(d)
    if not d.empty:
        stats["days_keep"] = int(pd.to_datetime(d["ts_ms"], unit="ms", utc=True)
                                   .dt.strftime("%Y-%m-%d").nunique())
    if verbose and stats["n_drop"]:
        print("  [bookDepth] %-9s 고정 제거 %d/%d (%.1f%%), 일수 %d -> %d"
              % (symbol, stats["n_drop"], stats["n_raw"],
                 100.0 * stats["n_drop"] / max(stats["n_raw"], 1),
                 stats["days_raw"], stats["days_keep"]))
    return d, stats
