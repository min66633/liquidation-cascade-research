# -*- coding: utf-8 -*-
"""Binance bookDepth 다운로더 — 충격의 '분모' D(u).

왜 필요한가
  변위 = g( V_청산 / Depth(u) ) 에서 분모가 D 다. 강제청산은 시장가로 나가 아래쪽에
  대기 중인 지정매수를 먹어치우므로, 얼마나 밀리는지는 청산액 자체가 아니라
  '청산액 대비 그 구간 호가 명목가'가 결정한다.

collectors/depth_poll.py 와 무엇이 다른가 (헷갈리기 쉬움)
  depth_poll : REST /fapi/v1/depth?limit=1000 을 30초마다. **실시간**이지만 도달 범위가
               종목마다 다르다 — BTC 0.18%, ETH 0.56%. -1% 깊이는 관측 불가.
  이 파일    : data.binance.vision 일별 아카이브. **T-1 까지만** 받을 수 있지만
               ±0.2/1/2/3/4/5% 를 전부 준다. 과거 분석의 분모는 전부 이쪽이다.
  둘 다 필요하다. 라이브 용량은 depth_poll, 과거 분석은 bookDepth.

데이터 형태 (2023-01-01~, 무료)
  timestamp, percentage, depth, notional
  percentage = 현재가 대비 %, notional = 그 구간까지의 누적 호가 명목가.
  하루 34,560행 (12개 percentage x 2,880 스냅샷 = 30초 간격).

저장 구조 — **일자 분할** (2026-08-01 변경)
  data/binance_bulk/book_depth/<SYMBOL>/<YYYY-MM-DD>.parquet
  이전에는 심볼당 파일 하나였는데, 그러면 하루치를 추가할 때마다 수백 MB 를 통째로
  다시 써야 해서 일일 잡으로 돌릴 수가 없다. 구 단일 파일도 로더가 계속 읽으므로
  (analysis/bookdepth.py 가 ts_ms 로 중복 제거) 마이그레이션은 필요 없다.

주의 — 2025년 벤더 결함
  특정 percentage 행의 notional 이 몇 시간씩 고정된다. 이 다운로더는 원본을 그대로
  저장하고, 걸러내는 것은 analysis/bookdepth.py 의 몫이다.

실행:
    python downloaders/binance_depth.py                       # Tardis 보유일 (기본)
    python downloaders/binance_depth.py --since 2023-01-01    # 구간 전체 백필
    python downloaders/binance_depth.py --days 2025-10-10
    python downloaders/binance_depth.py --days-file stress_days.txt
    python downloaders/binance_depth.py --daemon              # 매일 T-1 수집 (로거용)
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as U            # noqa: E402
import config as C            # noqa: E402
from downloaders.binance_days import tardis_days   # noqa: E402

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
RAW = os.path.join(C.DATA, "binance_bulk", "_raw")
OUT = os.path.join(C.DATA, "binance_bulk", "book_depth")
FIRST_DAY = "2023-01-01"          # 이 이전은 bookDepth 미제공
DAEMON_UTC_HOUR = 3               # T-1 파일이 확실히 올라온 뒤 (UTC 03:00)
DAEMON_SLEEP_S = 900


def fetch_day(session: requests.Session, symbol: str, day: str) -> str | None:
    dest = os.path.join(RAW, symbol, "bookDepth", "%s.zip" % day)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = "%s/%s/%s-bookDepth-%s.zip" % (BASE, symbol, symbol, day)
    try:
        r = session.get(url, timeout=120)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        # testzip() 은 손상 멤버의 이름을 돌려준다. None 이 아니면 받지 않는다.
        if zipfile.ZipFile(io.BytesIO(r.content)).testzip() is not None:
            return None
    except zipfile.BadZipFile:
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = "%s.%d.tmp" % (dest, os.getpid())
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def _col_name(c) -> str:
    """percentage -> 컬럼명. dm/dp + 절대값, 소수점은 언더바.

    float 로 강제하지 않으면 정수 percentage 만 있는 날에 "dm1" 이 나와서
    하위 분석이 전부 '컬럼 없음'으로 조용히 죽는다(2026-08-01 리뷰).
    """
    v = float(c)
    tail = ("%g" % abs(v)).replace(".", "_") if abs(v) % 1 else "%d_0" % int(abs(v))
    return "d%s%s" % ("m" if v < 0 else "p", tail)


def build_one(path: str) -> pd.DataFrame:
    """zip 하나 -> 넓은 형식(시각 x percentage 컬럼)."""
    try:
        z = zipfile.ZipFile(path)
        d = pd.read_csv(z.open(z.namelist()[0]))
    except Exception:                            # noqa: BLE001 — 하루 실패는 건너뛴다
        return pd.DataFrame()
    if d.empty or "percentage" not in d.columns:
        return pd.DataFrame()

    ts = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    d = d[ts.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    # NaT 를 그대로 astype("int64") 하면 -9223372036854775808 이 되어 dropna 를 통과한다
    d["ts_ms"] = ts[ts.notna()].astype("int64").to_numpy() // 10 ** 6
    for c in ("percentage", "depth", "notional"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["ts_ms", "percentage", "notional"])
    if d.empty:
        return pd.DataFrame()

    w = (d.pivot_table(index="ts_ms", columns="percentage", values="notional",
                       aggfunc="last")
           .sort_index())
    w.columns = [_col_name(c) for c in w.columns]
    return w.reset_index()


def part_path(symbol: str, day: str) -> str:
    return os.path.join(OUT, symbol, "%s.parquet" % day)


def resolve_days(a: argparse.Namespace) -> list[str]:
    if a.days_file:
        with open(a.days_file, encoding="utf-8-sig") as f:   # BOM 이면 첫 날짜가 404
            raw = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    elif a.days:
        raw = list(a.days)
    elif a.since:
        end = dt.date.today() - dt.timedelta(days=1)         # T-1 까지만 존재
        cur = dt.date.fromisoformat(a.since)
        raw = []
        while cur <= end:
            raw.append(cur.isoformat())
            cur += dt.timedelta(days=1)
    else:
        raw = list(tardis_days())
    days = sorted({d for d in raw if d >= FIRST_DAY})
    dropped = len(set(raw)) - len(days)
    if dropped:
        U.log("skip %d day(s) before %s (bookDepth 미제공)" % (dropped, FIRST_DAY))
    return days


def sync(session: requests.Session, symbols: list[str], days: list[str],
         workers: int, force: bool) -> tuple[int, int]:
    """필요한 날만 받아 일자 파일로 쓴다. 반환 (새로 쓴 파일 수, 실패 수)."""
    n_new = n_fail = 0
    for s in symbols:
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
        todo = [d for d in days if force or not os.path.exists(part_path(s, d))]
        if not todo:
            continue
        got = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_day, session, s, d): d for d in todo}
            for fu in as_completed(futs):
                p = fu.result()
                if p:
                    got.append((futs[fu], p))
                else:
                    n_fail += 1
        wrote = 0
        for day, p in sorted(got):
            w = build_one(p)
            if w.empty:
                n_fail += 1
                continue
            U.atomic_write_parquet(w, part_path(s, day))
            wrote += 1
        n_new += wrote
        if wrote:
            have = len([f for f in os.listdir(os.path.join(OUT, s))
                        if f.endswith(".parquet")])
            U.log("%-9s 요청 %d일 -> 신규 %d, 보유 총 %d일" % (s, len(todo), wrote, have))
    return n_new, n_fail


def daemon(session: requests.Session, symbols: list[str], workers: int) -> int:
    """매일 UTC 03:00 이후 T-1 을 받는다. 놓친 날은 최근 7일을 훑어 메운다."""
    U.log("bookDepth daemon: %d symbols, UTC %02d:00 이후 T-1 수집"
          % (len(symbols), DAEMON_UTC_HOUR))
    last_done = None
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        target = (now.date() - dt.timedelta(days=1)).isoformat()
        if now.hour >= DAEMON_UTC_HOUR and last_done != target:
            # 최근 7일을 함께 확인해 정지/실패 구간을 자동으로 메운다
            days = [(now.date() - dt.timedelta(days=i)).isoformat()
                    for i in range(1, 8)]
            days = [d for d in sorted(days) if d >= FIRST_DAY]
            try:
                n, f = sync(session, symbols, days, workers, force=False)
                U.log("daily sync %s: 신규 %d, 실패 %d" % (target, n, f))
                if f == 0 or n > 0:
                    last_done = target
            except Exception as e:                # noqa: BLE001 — 데몬은 죽지 않는다
                U.log("daily sync failed: %s: %s" % (type(e).__name__, e))
        time.sleep(DAEMON_SLEEP_S)


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance bookDepth downloader (day-partitioned)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--days", nargs="*", default=None, help="YYYY-MM-DD ...")
    ap.add_argument("--days-file", default=None, help="한 줄에 하나씩 적은 날짜 파일")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD 부터 T-1 까지 전부")
    ap.add_argument("--daemon", action="store_true", help="매일 T-1 수집 (로거용)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="이미 있는 일자도 다시 만든다")
    a = ap.parse_args()

    U.init_stdout()
    C.ensure_dirs()
    symbols = a.symbols if a.symbols else C.MAJORS
    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})

    try:
        if a.daemon:
            lock = U.acquire_single_instance(C.LOGS, "binance_depth")
            if lock is None:
                U.log("another binance_depth daemon is already running -> exit")
                return 0
            return daemon(session, symbols, a.workers)

        days = resolve_days(a)
        if not days:
            U.log("no days to fetch")
            return 1
        U.log("bookDepth: %d symbols x %d days (%s ~ %s)"
              % (len(symbols), len(days), days[0], days[-1]))
        t0 = time.monotonic()
        n, f = sync(session, symbols, days, a.workers, a.force)
        U.log("done: 신규 %d 파일, 실패 %d, %.0f초" % (n, f, time.monotonic() - t0))
    except KeyboardInterrupt:
        U.log("interrupted")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
