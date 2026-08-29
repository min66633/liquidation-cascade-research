# -*- coding: utf-8 -*-
"""공용 유틸 — HTTP 재시도, 원자적 parquet 쓰기, 로깅.

Windows 주의사항 두 가지가 반영되어 있다.
  1) 기본 인코딩이 cp949라 JSON을 읽을 때 encoding='utf-8'을 반드시 명시해야 한다
     (미명시 시 리더보드 파싱이 UnicodeDecodeError로 실패한다 — 2026-07-31 실측).
  2) 로거는 .bat에서 '>> x.out'으로 리다이렉트되어 실행된다. 리다이렉트된 stdout은
     로케일 인코딩(cp949)을 쓰므로 한글 print가 UnicodeEncodeError를 낸다.
     init_stdout()으로 utf-8 재설정 + errors='replace'를 건다.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any

import pandas as pd
import requests


class FetchError(Exception):
    """재시도를 모두 소진한 뒤의 최종 실패."""


# ------------------------------------------------------------------ 부트스트랩
def init_stdout() -> None:
    """리다이렉트된 stdout/stderr를 utf-8로 재설정. 실패해도 진행."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def log(msg: str) -> None:
    """UTC 타임스탬프가 붙은 한 줄 로그. 즉시 flush."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print("[%sZ] %s" % (ts, msg), flush=True)


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Content-Type": "application/json"})
    return s


# ------------------------------------------------------------------ HTTP
def _sleep_backoff(attempt: int, base: float, retry_after: str | None) -> None:
    """지수 백오프 + 지터. Retry-After 헤더가 있으면 우선한다.

    Retry-After는 신뢰할 수 없는 외부 입력이다. 음수(시계 오차로 흔함)를 그대로
    time.sleep에 넘기면 ValueError가 나고, 그건 FetchError가 아니라 호출부의
    except를 빠져나가 스윕 전체를 죽인다. 반드시 [0, 120]으로 클램프한다.
    """
    if retry_after:
        try:
            wait = float(retry_after)
        except (TypeError, ValueError):
            wait = float("nan")
        if wait == wait:                       # NaN이 아니면 사용
            time.sleep(min(max(wait, 0.0), 120.0))
            return
    wait = base * (2 ** attempt) + random.uniform(0, 0.5)
    time.sleep(min(wait, 60.0))


def _request_json(session: requests.Session, method: str, url: str,
                  *, payload: dict | None = None, params: dict | None = None,
                  timeout: float, max_retry: int, backoff_base: float,
                  stats: dict | None = None) -> Any:
    """재시도 포함 JSON 요청.

    재시도 대상: 429, 5xx, 연결/타임아웃 오류, JSON 파싱 실패.
    재시도 안 함: 그 외 4xx (요청 자체가 잘못된 것이므로 반복해도 동일).

    stats dict를 주면 n_429 / n_5xx / n_neterr 카운터를 누적한다. 호출부가
    레이트리밋 초과(429)와 단순 네트워크 불안정을 구분해 페이싱을 조절하는 데 쓴다.
    """
    last = "unknown"
    for attempt in range(max_retry):
        is_final = attempt == max_retry - 1     # 마지막 시도 뒤에는 자지 않는다
        try:
            if method == "POST":
                r = session.post(url, data=json.dumps(payload), timeout=timeout)
            else:
                r = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            last = "%s: %s" % (type(e).__name__, e)
            if stats is not None:
                stats["n_neterr"] = stats.get("n_neterr", 0) + 1
            if not is_final:
                _sleep_backoff(attempt, backoff_base, None)
            continue

        if r.status_code == 429 or r.status_code >= 500:
            last = "HTTP %d" % r.status_code
            if stats is not None:
                key = "n_429" if r.status_code == 429 else "n_5xx"
                stats[key] = stats.get(key, 0) + 1
            if not is_final:
                _sleep_backoff(attempt, backoff_base, r.headers.get("Retry-After"))
            continue
        if r.status_code >= 400:
            raise FetchError("HTTP %d (non-retryable): %s" % (r.status_code, r.text[:200]))

        try:
            return r.json()
        except ValueError as e:
            last = "json decode: %s" % e
            if not is_final:
                _sleep_backoff(attempt, backoff_base, None)

    raise FetchError("%d attempts failed, last=%s" % (max_retry, last))


def post_json(session: requests.Session, url: str, payload: dict, *,
              timeout: float, max_retry: int, backoff_base: float,
              stats: dict | None = None) -> Any:
    return _request_json(session, "POST", url, payload=payload, timeout=timeout,
                         max_retry=max_retry, backoff_base=backoff_base, stats=stats)


def get_json(session: requests.Session, url: str, params: dict, *,
             timeout: float, max_retry: int, backoff_base: float,
             stats: dict | None = None) -> Any:
    return _request_json(session, "GET", url, params=params, timeout=timeout,
                         max_retry=max_retry, backoff_base=backoff_base, stats=stats)


# ------------------------------------------------------------------ 저장
def atomic_write_parquet(df: pd.DataFrame, path: str, *, replace_retry: int = 4) -> None:
    """임시파일에 쓴 뒤 os.replace로 교체. 크래시 시 반쪽 파일이 남지 않는다.

    tmp 이름에 PID를 넣는다 — 같은 수집기가 두 번 떠도 서로의 임시파일을 밟지 않는다.
    Windows에서 대상 파일을 다른 프로세스(분석 노트북, Defender, 탐색기 미리보기)가
    열고 있으면 os.replace가 PermissionError를 낸다. 이건 대개 일시적이라 짧게 재시도한다.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    try:
        df.to_parquet(tmp, index=False, compression="snappy")
        for attempt in range(replace_retry):
            try:
                os.replace(tmp, path)      # Windows에서도 원자적 교체
                return
            except PermissionError:
                if attempt == replace_retry - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def quarantine(path: str) -> str:
    """읽기 실패한 파일을 .corrupt.<ts>로 격리하고 새 경로를 반환.

    격리에 실패하면 예외를 그대로 올린다 — 읽지 못한 파일을 덮어쓰는 것보다
    쓰기를 포기하는 편이 낫다. 이 파일들은 유일본이라 덮어쓰면 영구 손실이다.
    """
    dest = "%s.corrupt.%d" % (path, utc_now_ms())
    os.replace(path, dest)
    return dest


def read_parquet_or_quarantine(path: str) -> pd.DataFrame | None:
    """기존 parquet을 읽는다. 손상됐으면 격리하고 None을 반환.

    반환 None = "기존 데이터 없음(새로 시작해도 안전)". 격리조차 실패하면 예외가
    올라가므로, 호출부는 그 경우 쓰기를 건너뛰어야 한다.
    """
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        dest = quarantine(path)
        log("quarantined unreadable %s (%s) -> %s"
            % (os.path.basename(path), e, os.path.basename(dest)))
        return None


def acquire_single_instance(lock_dir: str, name: str):
    """중복 실행 방지. 잠금을 못 잡으면 None을 반환한다.

    반환된 파일 객체는 프로세스가 살아 있는 동안 참조를 유지해야 한다(GC되면
    잠금이 풀린다). OS가 프로세스 종료 시 잠금을 자동 해제하므로 크래시 후
    잠금이 남는 문제는 없다.

    같은 수집기가 둘 뜨면 merge_write / append_sweeplog가 read-modify-write
    경합을 일으켜 한쪽 갱신이 소실된다.
    """
    os.makedirs(lock_dir, exist_ok=True)
    path = os.path.join(lock_dir, "%s.lock" % name)
    try:
        f = open(path, "w")
    except OSError as e:
        log("lock: cannot open %s (%s) -> proceeding without guard" % (path, e))
        return True                     # 잠금 불가 시 기동을 막지는 않는다
    try:
        import msvcrt
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        return f                        # 비Windows: 잠금 없이 진행
    except OSError:
        f.close()
        return None                     # 다른 인스턴스가 보유 중
    return f


def to_float(v: Any) -> float:
    """HL/Binance는 수치를 문자열로 준다. None/빈값/파싱실패는 NaN."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
