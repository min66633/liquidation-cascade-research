# -*- coding: utf-8 -*-
"""Hyperliquid 주소 유니버스 구축.

stats-data 리더보드(약 33MB, 무료 GET 1회)에서 전체 주소 + 계좌가치를 받아
순위를 매긴다. 2026-07-31 실측: 40,976주소, 총 계좌가치 $375.4억,
상위 1,000계좌가 87.5%를 차지 → 상위 N만 폴링해도 대부분을 커버한다.

랭킹 기준
  1순위: 과거 스냅샷에서 관측된 실제 포지션 명목가(total_ntl_pos)의 최근 최대값
  2순위: 리더보드 계좌가치(accountValue)   ← 미관측 주소의 대체값
계좌가치가 커도 무레버리지면 청산 연료가 아니므로, 관측이 쌓일수록 1순위가 정확하다.

단독 실행:
    python collectors/hl_universe.py           # 갱신 후 요약 출력
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402

_UNIVERSE_GLOB = "universe_*.parquet"


def _observed_ranking(lookback_files: int | None = None) -> pd.Series:
    """최근 스냅샷들에서 주소별 관측 포지션 명목가의 최대값.

    반환: index=address, value=max(total_ntl_pos). 파일이 없으면 빈 Series.
    """
    if lookback_files is None:
        lookback_files = C.HL_OBSERVED_LOOKBACK_FILES
    pattern = os.path.join(C.HL_DIR_ACCOUNTS, "*", "accounts_*.parquet")
    files = sorted(glob.glob(pattern))[-lookback_files:]
    if not files:
        return pd.Series(dtype="float64")

    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=["address", "total_ntl_pos"]))
        except Exception as e:
            U.log("universe: skip unreadable %s (%s)" % (os.path.basename(f), e))
    if not frames:
        return pd.Series(dtype="float64")

    df = pd.concat(frames, ignore_index=True)
    df["total_ntl_pos"] = pd.to_numeric(df["total_ntl_pos"], errors="coerce")
    return df.groupby("address")["total_ntl_pos"].max().dropna()


def fetch_leaderboard(session, max_retry: int = 3) -> pd.DataFrame:
    """리더보드 원본 → DataFrame(address, account_value). 33MB GET, 재시도 포함."""
    last = "unknown"
    payload = None
    for attempt in range(max_retry):
        try:
            r = session.get(C.HL_LEADERBOARD_URL, timeout=180)
            r.raise_for_status()
            # requests의 인코딩 추정에 맡기지 않는다 — cp949 환경에서 깨진다.
            payload = json.loads(r.content.decode("utf-8"))
            break
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
            if attempt < max_retry - 1:
                time.sleep(5.0 * (attempt + 1))
    if payload is None:
        raise U.FetchError("leaderboard fetch failed: %s" % last)

    rows = payload["leaderboardRows"] if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError("leaderboard returned no rows")

    df = pd.DataFrame({
        "address": [str(x.get("ethAddress", "")).lower() for x in rows],
        "account_value": [U.to_float(x.get("accountValue")) for x in rows],
    })
    df = df[df["address"].str.startswith("0x")]
    df = df.dropna(subset=["account_value"])
    # 잔고 0 이하 계좌는 청산 연료가 될 수 없다. 실측 40,976행 중 3,688행(9%)이
    # 여기 해당 — 남겨두면 탐색 예산의 9%를 확정적으로 빈 계좌에 쓴다.
    df = df[df["account_value"] > 0]
    df = df.drop_duplicates(subset=["address"], keep="first")
    if df.empty:
        raise ValueError("leaderboard parsed to zero valid rows")
    return df


def refresh_universe(session=None) -> pd.DataFrame:
    """리더보드를 새로 받아 랭킹을 만들고 parquet으로 저장. 저장한 DF를 반환."""
    C.ensure_dirs()
    own = session is None
    if own:
        session = U.make_session(C.USER_AGENT)
    try:
        df = fetch_leaderboard(session)
    finally:
        if own:
            session.close()

    observed = _observed_ranking()
    df["observed_ntl"] = df["address"].map(observed).astype("float64")
    # rank_key = max(관측 명목가, 계좌가치 * 하한가중치).
    # fillna만 쓰면 '관측됐지만 그 순간 플랫'인 계좌가 0.0으로 확정 강등된다
    # (0.0은 NaN이 아니라 fillna가 안 걸린다). 그러면 폴링에서 빠지고,
    # 빠지면 다시 관측될 수 없어 영구 사각지대가 되는 자기강화 루프가 생긴다.
    df["rank_key"] = np.maximum(df["observed_ntl"].fillna(0.0),
                                df["account_value"] * C.HL_FLAT_ACCOUNT_WEIGHT)
    df = df.sort_values("rank_key", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["built_at_ms"] = U.utc_now_ms()

    day = time.strftime("%Y%m%d", time.gmtime())
    path = os.path.join(C.HL_DIR_UNIVERSE, "universe_%s.parquet" % day)
    U.atomic_write_parquet(df, path)

    n_obs = int(df["observed_ntl"].notna().sum())
    U.log("universe: %d addrs, %d with observed positions, top%d covers %.1f%% of account value -> %s"
          % (len(df), n_obs, C.HL_TOP_N,
             100.0 * df["account_value"].head(C.HL_TOP_N).sum() / max(df["account_value"].sum(), 1e-9),
             os.path.basename(path)))
    return df


def load_latest_universe() -> tuple[pd.DataFrame | None, float]:
    """가장 최근 유니버스 parquet과 그 나이(초)를 반환. 없으면 (None, inf)."""
    files = sorted(glob.glob(os.path.join(C.HL_DIR_UNIVERSE, _UNIVERSE_GLOB)))
    if not files:
        return None, float("inf")
    path = files[-1]
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        U.log("universe: latest file unreadable (%s) -> forcing refresh" % e)
        return None, float("inf")
    # built_at_ms가 없거나 비정상이면 나이를 계산할 수 없다. 여기서 KeyError가 나면
    # 매 스윕 같은 파일을 다시 골라 같은 예외를 내므로 영구 정지가 된다.
    if df.empty or not {"address", "built_at_ms"}.issubset(df.columns):
        U.log("universe: %s missing required columns -> forcing refresh" % os.path.basename(path))
        return None, float("inf")
    built = U.to_float(df["built_at_ms"].iloc[0])
    if math.isnan(built) or built <= 0:
        # max(0.0, nan)은 0.0을 돌려준다 → 나이 0으로 오인해 영원히 갱신하지 않게 된다.
        U.log("universe: %s has invalid built_at_ms -> forcing refresh" % os.path.basename(path))
        return None, float("inf")
    age = max(0.0, (U.utc_now_ms() - built) / 1000.0)
    return df, age


_last_refresh_attempt = 0.0


def get_universe(max_age_s: float | None = None, session=None) -> pd.DataFrame:
    """캐시가 신선하면 그대로, 오래됐으면 갱신해서 반환. 갱신 실패 시 캐시 폴백."""
    global _last_refresh_attempt
    max_age_s = C.HL_UNIVERSE_MAX_AGE_S if max_age_s is None else max_age_s
    df, age = load_latest_universe()
    if df is not None and age <= max_age_s:
        return df

    # 갱신 '시도' 자체를 스로틀한다. 성공 기준으로만 막으면 33MB를 받은 뒤
    # 저장에서 실패하는 경우 매 스윕(300초)마다 재다운로드해 하루 9.5GB가 된다.
    since = time.monotonic() - _last_refresh_attempt
    if df is not None and since < C.HL_UNIVERSE_RETRY_MIN_S:
        return df
    _last_refresh_attempt = time.monotonic()

    try:
        return refresh_universe(session=session)
    except Exception as e:
        if df is not None:
            U.log("universe: refresh failed (%s) -> using cache aged %.0fs" % (e, age))
            return df
        raise


_CURSOR_PATH = os.path.join(C.HL_DIR_UNIVERSE, "explore_cursor.json")


def _load_cursor() -> int:
    try:
        with open(_CURSOR_PATH, encoding="utf-8") as f:
            return max(0, int(json.load(f).get("cursor", 0)))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def _save_cursor(value: int) -> None:
    tmp = _CURSOR_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"cursor": int(value)}, f)
        os.replace(tmp, _CURSOR_PATH)
    except OSError as e:
        U.log("universe: cursor save failed (%s)" % e)


def select_addresses(uni: pd.DataFrame, top_n: int,
                     explore_frac: float | None = None) -> tuple[list[str], int, int]:
    """스윕 대상 선택 = 코어(랭킹 상위) + 탐색(꼬리 회전 스캔).

    꼬리를 회전 스캔하지 않으면 '자본은 작지만 고레버리지'인 계좌가 관측되지 않고,
    관측되지 않으면 관측 랭킹에도 못 올라오는 순환 사각지대가 생긴다.

    반환: (주소 리스트, 코어 수, 탐색 수)
    """
    if explore_frac is None:
        explore_frac = C.HL_EXPLORE_FRAC
    # NaN은 min/max를 그대로 통과한다(클램프가 클램프하지 않는다). 이후
    # int(round(top_n * nan))이 ValueError를 내고 매 스윕 반복된다.
    if not (explore_frac >= 0.0):
        explore_frac = C.HL_EXPLORE_FRAC
    explore_frac = min(explore_frac, 0.9)

    addrs = uni["address"].tolist()
    if not addrs:
        return [], 0, 0
    top_n = max(1, min(top_n, len(addrs)))

    n_core = int(round(top_n * (1.0 - explore_frac)))
    n_core = min(max(n_core, 1), top_n)
    core = addrs[:n_core]

    # 꼬리는 주소 사전순으로 고정한다. rank_key 순서를 그대로 쓰면 유니버스가
    # 갱신될 때마다 리스트가 재정렬되어, 위치 기반 커서가 다른 순열을 가리키게 된다
    # (한 바퀴 22시간 vs 갱신 24시간이라 매 사이클 중간에 어긋난다).
    # 주소 사전순은 갱신에 불변이라 커서가 계속 의미를 갖는다.
    tail = sorted(addrs[n_core:])
    n_explore = min(top_n - n_core, len(tail))
    explore: list[str] = []
    if n_explore:
        cur = _load_cursor() % len(tail)
        # 꼬리 끝에서 앞으로 되감으며 순환한다.
        idx = [(cur + i) % len(tail) for i in range(n_explore)]
        explore = [tail[i] for i in idx]
        _save_cursor((cur + n_explore) % len(tail))

    return core + explore, len(core), len(explore)


if __name__ == "__main__":
    U.init_stdout()
    out = refresh_universe()
    top = out.head(C.HL_TOP_N)
    print("\ntop%d account_value sum: $%.1fM" % (C.HL_TOP_N, top["account_value"].sum() / 1e6))
    print("median account_value (all): $%.0f" % out["account_value"].median())
    print(out.head(10)[["rank", "address", "account_value", "observed_ntl"]].to_string(index=False))
