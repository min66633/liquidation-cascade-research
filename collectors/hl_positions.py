# -*- coding: utf-8 -*-
"""Hyperliquid 청산맵 L(p) 스냅샷 수집기 — 이 연구의 X변수 생성기.

왜 이 수집기가 대체 불가인가
  HL S3 아카이브는 L2북/체결/asset_ctxs를 제공하지만 '계좌별 포지션 상태'는 없다.
  liquidationPx는 clearinghouseState 라이브 쿼리에만 존재하고 과거 스냅샷이
  세상에 존재하지 않는다. 즉 "무엇이 청산됐는가"(사후)는 백필되지만
  "어디에 청산이 쌓여 있었는가"(사전 L(p))는 지금부터 쌓아야만 얻는다.

수집 단위 (스윕)
  allMids(시작) → 상위 N계좌 clearinghouseState 순차 조회 → allMids(종료)
  스윕은 순간이 아니라 약 125초에 걸쳐 번진다. 캐스케이드 중에는 이 번짐이
  유의미하므로 계좌별 조회시각(ts)과 시작/종료 시점의 mid를 모두 기록해
  분석 단계에서 번짐 보정이 가능하게 한다.

실행:
    python collectors/hl_positions.py              # 상시 루프
    python collectors/hl_positions.py --once       # 1회만 (점검용)
    python collectors/hl_positions.py --top-n 50 --once
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from collectors import hl_universe  # noqa: E402

# parquet 스키마 고정 — 어떤 스윕에서 전부 NaN이 나와도 컬럼 타입이 흔들리지 않게 한다.
# (고정하지 않으면 pyarrow가 null 타입으로 추론해 이후 concat이 깨진다.)
POS_DTYPES = {
    "sweep_id": "int64", "ts": "int64", "address": "string", "coin": "string",
    "szi": "float64", "entry_px": "float64", "liquidation_px": "float64",
    "position_value": "float64", "margin_used": "float64", "unrealized_pnl": "float64",
    "lev_type": "string", "lev_value": "float64", "max_leverage": "float64",
    "funding_since_open": "float64",
}
ACC_DTYPES = {
    "sweep_id": "int64", "ts": "int64", "address": "string", "hl_time": "int64",
    "account_value": "float64", "total_ntl_pos": "float64", "total_raw_usd": "float64",
    "total_margin_used": "float64", "cross_maint_margin_used": "float64",
    "withdrawable": "float64", "n_positions": "int64",
}
MID_DTYPES = {
    "sweep_id": "int64", "phase": "string", "ts": "int64",
    "coin": "string", "mid_px": "float64",
}
SWEEP_DTYPES = {
    "sweep_id": "int64", "started_ms": "int64", "ended_ms": "int64",
    "n_target": "int64", "n_core": "int64", "n_explore": "int64",
    "n_ok": "int64", "n_fail": "int64", "n_skipped": "int64", "n_429": "int64",
    "n_positions": "int64", "req_interval_s": "float64", "trusted": "bool",
}


def _empty(dtypes: dict) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})


def _coerce(rows: list[dict], dtypes: dict) -> pd.DataFrame:
    """행 리스트를 고정 스키마 DataFrame으로. 빈 리스트도 스키마를 유지한다.

    폴백 경로에서도 선언 타입을 반드시 지킨다. 예전처럼 float로 우회해 두면
    같은 파일군인데 어떤 파일은 int64, 어떤 파일은 double이 되고, 나중에
    디렉터리 통째 읽기가 ArrowInvalid로 터진다(소수값이 하나라도 섞이는 순간).
    """
    if not rows:
        return _empty(dtypes)
    df = pd.DataFrame(rows)
    for col, dt in dtypes.items():
        if col not in df.columns:
            df[col] = pd.NA
        try:
            df[col] = df[col].astype(dt)
        except (TypeError, ValueError):
            U.log("schema: column %s did not fit %s -> coercing with sentinel" % (col, dt))
            df[col] = _force_dtype(df[col], dt)
    return df[list(dtypes.keys())]


def _force_dtype(s: pd.Series, dt: str) -> pd.Series:
    """어떤 값이 들어와도 선언 타입으로 떨어뜨린다. int는 -1, bool은 False가 결측 표식."""
    if dt == "int64":
        return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64")
    if dt == "bool":
        return s.map({True: True, False: False}).fillna(False).astype("bool")
    if dt == "float64":
        return pd.to_numeric(s, errors="coerce").astype("float64")
    return s.astype("string")


class Pacer:
    """요청 간 최소 간격 유지. 429가 보이면 간격을 늘리고 깨끗하면 되돌린다.

    완화 조건을 '실패 0건'으로 두면 안 된다 — 항상 실패하는 주소가 하나만 있어도
    relax가 영원히 호출되지 않고, 429가 한 번씩 낄 때마다 8배 상한까지 한 방향으로
    래칫된다. 그러면 1,000주소 스윕이 1,000초가 되어 5분 주기가 조용히 17분이 된다
    (가중치는 상한의 10%만 쓰면서). 소량 실패는 정상으로 보고 완화한다.
    """

    def __init__(self, interval_s: float):
        self.base = interval_s
        self.interval = interval_s
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()

    def penalize(self) -> None:
        self.interval = min(self.interval * 1.5, self.base * 8)

    def relax(self) -> None:
        self.interval = max(self.base, self.interval * 0.8)

    def update(self, n_429: int, n_fail: int, n_target: int) -> None:
        if n_429 > 0:
            self.penalize()
        elif n_fail <= max(1, int(0.01 * n_target)):
            self.relax()


def fetch_mids(session, sweep_id: int, phase: str, stats: dict) -> list[dict]:
    """allMids 스냅샷. 실패해도 스윕 전체를 죽이지 않는다(빈 리스트 반환)."""
    try:
        data = U.post_json(session, C.HL_INFO_URL, {"type": "allMids"},
                           timeout=C.HL_HTTP_TIMEOUT_S, max_retry=C.HL_MAX_RETRY,
                           backoff_base=C.HL_BACKOFF_BASE_S, stats=stats)
    except U.FetchError as e:
        U.log("mids(%s) failed: %s" % (phase, e))
        return []
    if not isinstance(data, dict):
        return []
    ts = U.utc_now_ms()
    return [{"sweep_id": sweep_id, "phase": phase, "ts": ts,
             "coin": str(k), "mid_px": U.to_float(v)} for k, v in data.items()]


def parse_state(state: dict, sweep_id: int, ts: int, address: str
                ) -> tuple[list[dict], dict | None]:
    """clearinghouseState 응답 → (포지션 행들, 계좌 요약 행)."""
    if not isinstance(state, dict):
        return [], None

    ms = state.get("marginSummary") or {}
    positions: list[dict] = []
    for item in state.get("assetPositions") or []:
        pos = (item or {}).get("position") or {}
        szi = U.to_float(pos.get("szi"))
        # NaN은 truthy라 'not szi'로는 걸러지지 않는다 — 명시적으로 검사한다.
        if math.isnan(szi) or szi == 0.0:
            continue
        lev = pos.get("leverage") or {}
        funding = pos.get("cumFunding") or {}
        positions.append({
            "sweep_id": sweep_id, "ts": ts, "address": address,
            "coin": str(pos.get("coin", "")),
            "szi": szi,
            "entry_px": U.to_float(pos.get("entryPx")),
            # None인 경우가 흔하다(교차마진에서 담보가 충분하면 청산가 없음) → NaN 유지
            "liquidation_px": U.to_float(pos.get("liquidationPx")),
            "position_value": U.to_float(pos.get("positionValue")),
            "margin_used": U.to_float(pos.get("marginUsed")),
            "unrealized_pnl": U.to_float(pos.get("unrealizedPnl")),
            "lev_type": str(lev.get("type", "")),
            "lev_value": U.to_float(lev.get("value")),
            "max_leverage": U.to_float(pos.get("maxLeverage")),
            "funding_since_open": U.to_float(funding.get("sinceOpen")),
        })

    hl_time = U.to_float(state.get("time"))
    account = {
        "sweep_id": sweep_id, "ts": ts, "address": address,
        # 거래소가 찍어준 상태 시각. 없으면 0 (조회시각 ts로 대체 가능).
        "hl_time": 0 if math.isnan(hl_time) else int(hl_time),
        "account_value": U.to_float(ms.get("accountValue")),
        "total_ntl_pos": U.to_float(ms.get("totalNtlPos")),
        "total_raw_usd": U.to_float(ms.get("totalRawUsd")),
        "total_margin_used": U.to_float(ms.get("totalMarginUsed")),
        "cross_maint_margin_used": U.to_float(state.get("crossMaintenanceMarginUsed")),
        "withdrawable": U.to_float(state.get("withdrawable")),
        "n_positions": len(positions),
    }
    return positions, account


class HotTracker:
    """근접 연료 추적기 — 깊은 스윕이 만든 지도에서 '지금 사정거리 안'인 주소를 고른다.

    핫리스트는 매 사이클 현재 mark 기준으로 다시 계산한다. 가격이 밀리면 멀리 있던
    포지션이 자동으로 들어오므로, 변동성 급등 시 추적 대상이 스스로 넓어진다.
    """

    def __init__(self) -> None:
        self.liq: pd.DataFrame = pd.DataFrame(columns=["address", "coin", "liquidation_px"])
        self.last_run = 0.0

    def update_map(self, pos_rows: list[dict]) -> None:
        """깊은 스윕 결과로 지도를 갱신."""
        rows = [{"address": r["address"], "coin": r["coin"],
                 "liquidation_px": r["liquidation_px"]}
                for r in pos_rows if not math.isnan(r["liquidation_px"])]
        self.liq = pd.DataFrame(rows) if rows else self.liq.iloc[0:0]

    def select(self, mids: dict[str, float], band_pct: float, cap: int) -> list[str]:
        if self.liq.empty or not mids:
            return []
        d = self.liq.copy()
        d["mark"] = d["coin"].map(mids)
        d = d[d["mark"].notna() & (d["mark"] > 0) & (d["liquidation_px"] > 0)]
        if d.empty:
            return []
        d["dist"] = (d["liquidation_px"] / d["mark"] - 1.0).abs() * 100.0
        d = d[d["dist"] <= band_pct].sort_values("dist")
        # 가장 가까운 것부터. 상한을 넘으면 먼 쪽을 버린다.
        return list(dict.fromkeys(d["address"].tolist()))[:cap]


def run_hot(session, tracker: HotTracker, pacer: Pacer, band_pct: float,
            cap: int) -> int:
    """핫 스윕 1회. 현재 mid를 받아 핫리스트를 정하고 그 주소만 다시 조회한다."""
    stats = {"n_429": 0}
    hot_id = U.utc_now_ms()
    day = time.strftime("%Y-%m-%d", time.gmtime(hot_id / 1000.0))

    mid_rows = fetch_mids(session, hot_id, "hot", stats)
    mids = {r["coin"]: r["mid_px"] for r in mid_rows
            if not math.isnan(r.get("mid_px", float("nan")))}
    addrs = tracker.select(mids, band_pct, cap)
    if not addrs:
        # 조용히 반환하면 '핫이 안 돌았다'와 '돌았는데 사정거리에 아무것도 없다'가
        # 구분되지 않는다. 후자는 정상 상태이므로 명시적으로 남긴다.
        U.log("hot %d: no positions within +-%.0f%% of mark (map=%d rows)"
              % (hot_id, band_pct, len(tracker.liq)))
        return 0

    pos_rows: list[dict] = []
    n_fail = 0
    for addr in addrs:
        pacer.wait()
        try:
            state = U.post_json(session, C.HL_INFO_URL,
                                {"type": "clearinghouseState", "user": addr},
                                timeout=C.HL_HTTP_TIMEOUT_S, max_retry=C.HL_MAX_RETRY,
                                backoff_base=C.HL_BACKOFF_BASE_S, stats=stats)
        except U.FetchError:
            n_fail += 1
            continue
        p, _ = parse_state(state, hot_id, U.utc_now_ms(), addr)
        pos_rows.extend(p)

    try:
        _write(_coerce(pos_rows, POS_DTYPES), C.HL_DIR_HOT, "hot", hot_id, day)
        _write(_coerce(mid_rows, MID_DTYPES), C.HL_DIR_MIDS, "mids", hot_id, day)
    except Exception as e:
        U.log("hot write failed %d: %s: %s" % (hot_id, type(e).__name__, e))
        return 0

    U.log("hot %d: addrs=%d pos=%d fail=%d 429=%d" %
          (hot_id, len(addrs), len(pos_rows), n_fail, stats.get("n_429", 0)))
    return len(pos_rows)


def _write(df: pd.DataFrame, base_dir: str, prefix: str, sweep_id: int, day: str) -> None:
    path = os.path.join(base_dir, day, "%s_%d.parquet" % (prefix, sweep_id))
    U.atomic_write_parquet(df, path)


def append_sweeplog(row: dict, day: str) -> None:
    """일별 스윕 로그에 한 행 추가.

    concat 후 반드시 스키마를 다시 고정한다. 안 하면 SWEEP_DTYPES에 필드가
    추가되는 순간 pd.concat이 새 컬럼을 끝에 붙이고 기존 행을 NaN으로 채워
    int64가 double로 변한다. 그 드리프트는 파일에 눌어붙어 이후 append마다
    재생산되고, 소수값이 하나 섞이면 읽기가 ArrowInvalid로 터진다.
    """
    path = os.path.join(C.HL_DIR_SWEEPLOG, "sweeplog_%s.parquet" % day)
    new = _coerce([row], SWEEP_DTYPES)
    old = U.read_parquet_or_quarantine(path)      # 손상 시 격리(덮어쓰지 않음)
    if old is not None:
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["sweep_id"], keep="last")
    new = new.reindex(columns=list(SWEEP_DTYPES))
    for col, dt in SWEEP_DTYPES.items():
        if str(new[col].dtype) != dt:
            new[col] = _force_dtype(new[col], dt)
    U.atomic_write_parquet(new, path)


def run_sweep(session, addresses: list[str], pacer: Pacer,
              n_core: int = 0, n_explore: int = 0,
              sweep_interval_s: float | None = None,
              on_tick=None) -> dict:
    """1회 깊은 스윕 수행 후 결과 요약 반환.

    on_tick: 주소 루프 중간에 호출되는 콜백. 소비한 초를 반환해야 한다(마감 보정용).
             핫 스윕을 끼워 넣는 데 쓴다.
    """
    sweep_id = U.utc_now_ms()
    day = time.strftime("%Y-%m-%d", time.gmtime(sweep_id / 1000.0))
    stats = {"n_429": 0}
    interval = C.HL_SWEEP_INTERVAL_S if sweep_interval_s is None else sweep_interval_s
    # 벽시계 마감. requests의 timeout은 소켓 read 타임아웃이지 전체 요청 마감이
    # 아니라서, 느리게 흘리는 서버 하나가 요청을 한없이 붙잡을 수 있다.
    deadline = time.monotonic() + C.HL_SWEEP_DEADLINE_FRAC * interval

    mids = fetch_mids(session, sweep_id, "start", stats)
    pos_rows: list[dict] = []
    acc_rows: list[dict] = []
    n_ok = n_fail = n_skipped = 0

    for i, addr in enumerate(addresses):
        # 핫 스윕을 깊은 스윕 중간에 끼워 넣는다. 깊은 스윕은 500초가 걸리는데
        # 그동안 근접 연료를 놓치면 변동성 급등에 대응하지 못한다.
        # 핫 스윕에 쓴 시간만큼 마감을 늘려준다 — 아니면 핫이 깊은 스윕을 굶긴다.
        if on_tick is not None:
            deadline += on_tick()
        if time.monotonic() > deadline:
            n_skipped = len(addresses) - i
            U.log("sweep %d: deadline hit -> skipping %d remaining addresses"
                  % (sweep_id, n_skipped))
            break
        pacer.wait()
        try:
            state = U.post_json(session, C.HL_INFO_URL,
                                {"type": "clearinghouseState", "user": addr},
                                timeout=C.HL_HTTP_TIMEOUT_S, max_retry=C.HL_MAX_RETRY,
                                backoff_base=C.HL_BACKOFF_BASE_S, stats=stats)
        except U.FetchError:
            n_fail += 1
            continue
        p, a = parse_state(state, sweep_id, U.utc_now_ms(), addr)
        if a is None:
            n_fail += 1
            continue
        pos_rows.extend(p)
        acc_rows.append(a)
        n_ok += 1

    mids.extend(fetch_mids(session, sweep_id, "end", stats))
    ended = U.utc_now_ms()

    n_target = len(addresses)
    fail_rate = (n_fail + n_skipped) / max(n_target, 1)
    trusted = fail_rate <= C.HL_MAX_FAIL_RATE

    # 세 파일을 독립적으로 쓴다. 하나가 실패해도 나머지는 남기고, 무엇보다
    # 스윕로그 행은 반드시 남겨야 한다 — 안 그러면 positions 파일만 덩그러니
    # 남아 trusted 필터로도 걸러지지 않는 고아 데이터가 된다.
    writes = ((C.HL_DIR_POSITIONS, "positions", _coerce(pos_rows, POS_DTYPES)),
              (C.HL_DIR_ACCOUNTS, "accounts", _coerce(acc_rows, ACC_DTYPES)),
              (C.HL_DIR_MIDS, "mids", _coerce(mids, MID_DTYPES)))
    for base_dir, prefix, frame in writes:
        try:
            _write(frame, base_dir, prefix, sweep_id, day)
        except Exception as e:
            trusted = False
            U.log("write failed %s sweep %d: %s: %s" % (prefix, sweep_id, type(e).__name__, e))

    summary = {
        "sweep_id": sweep_id, "started_ms": sweep_id, "ended_ms": ended,
        "n_target": n_target, "n_core": n_core, "n_explore": n_explore,
        "n_ok": n_ok, "n_fail": n_fail, "n_skipped": n_skipped,
        "n_429": int(stats.get("n_429", 0)), "n_positions": len(pos_rows),
        "req_interval_s": pacer.interval, "trusted": trusted,
    }
    try:
        append_sweeplog(summary, day)
    except Exception as e:
        U.log("sweeplog write failed sweep %d: %s: %s" % (sweep_id, type(e).__name__, e))

    pacer.update(int(stats.get("n_429", 0)), n_fail, n_target)
    summary["_pos_rows"] = pos_rows          # 핫리스트 갱신용 (parquet에는 안 들어간다)

    n_liq = sum(1 for r in pos_rows if not math.isnan(r["liquidation_px"]))
    U.log("sweep %d: %ds core=%d expl=%d ok=%d fail=%d skip=%d 429=%d pos=%d (liqPx=%d) pace=%.3fs%s"
          % (sweep_id, (ended - sweep_id) // 1000, n_core, n_explore, n_ok, n_fail,
             n_skipped, stats.get("n_429", 0), len(pos_rows), n_liq, pacer.interval,
             "" if trusted else "  [UNTRUSTED]"))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Hyperliquid liquidation-map snapshot collector")
    ap.add_argument("--once", action="store_true", help="single sweep then exit")
    ap.add_argument("--top-n", type=int, default=C.HL_TOP_N)
    ap.add_argument("--interval", type=float, default=C.HL_SWEEP_INTERVAL_S)
    ap.add_argument("--explore-frac", type=float, default=C.HL_EXPLORE_FRAC,
                    help="fraction of each sweep spent scanning the ranking tail")
    ap.add_argument("--hot-interval", type=float, default=C.HL_HOT_INTERVAL_S,
                    help="seconds between near-money (hot) sweeps; 0 disables")
    ap.add_argument("--hot-band", type=float, default=C.HL_HOT_BAND_PCT,
                    help="a position is hot if its liq price is within this %% of mark")
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    lock = U.acquire_single_instance(C.LOGS, "hl_positions")
    if lock is None:
        U.log("another hl_positions instance is already running -> exit")
        return 0
    session = U.make_session(C.USER_AGENT)
    pacer = Pacer(C.HL_REQ_INTERVAL_S)

    U.log("hl_positions start: top_n=%d explore=%.0f%% interval=%ds req_pace=%.3fs"
          % (a.top_n, 100 * a.explore_frac, a.interval, C.HL_REQ_INTERVAL_S))

    tracker = HotTracker()

    def hot_tick() -> float:
        """핫 스윕이 필요하면 실행하고 소비한 초를 반환. 실패해도 루프를 죽이지 않는다."""
        if a.hot_interval <= 0 or tracker.liq.empty:
            return 0.0
        if time.monotonic() - tracker.last_run < a.hot_interval:
            return 0.0
        t0 = time.monotonic()
        try:
            run_hot(session, tracker, pacer, a.hot_band, C.HL_HOT_MAX_ADDR)
        except Exception as e:
            U.log("hot aborted: %s: %s" % (type(e).__name__, e))
        tracker.last_run = time.monotonic()
        return time.monotonic() - t0

    try:
        while True:
            slot = time.monotonic()
            try:
                uni = hl_universe.get_universe(session=session)
                addresses, n_core, n_expl = hl_universe.select_addresses(
                    uni, a.top_n, a.explore_frac)
                if not addresses:
                    raise ValueError("universe produced no addresses")
                res = run_sweep(session, addresses, pacer, n_core, n_expl,
                                a.interval, on_tick=hot_tick)
                tracker.update_map(res.get("_pos_rows") or [])
            except Exception as e:
                # 한 스윕의 실패로 루프가 죽으면 안 된다 — 기록하고 다음 슬롯으로.
                U.log("sweep aborted: %s: %s" % (type(e).__name__, e))

            if a.once:
                return 0

            # 깊은 스윕 사이의 대기 시간에도 핫 스윕은 계속 돈다.
            while True:
                elapsed = time.monotonic() - slot
                if elapsed >= a.interval:
                    if elapsed > a.interval * 1.05:
                        U.log("overrun: cycle took %.0fs (interval %.0fs)"
                              % (elapsed, a.interval))
                    break
                if hot_tick() > 0.0:
                    continue
                time.sleep(min(1.0, a.interval - elapsed))
    except KeyboardInterrupt:
        U.log("interrupted -> exit")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
