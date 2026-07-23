from __future__ import annotations

import ctypes
import os
import threading


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

_lock = threading.Lock()
_condition = threading.Condition(_lock)
_count = 0
_worker: threading.Thread | None = None
_REFRESH_SECONDS = 30.0


def _set_execution_state(flags: int) -> bool:
    if os.name != "nt":
        return True
    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    return bool(result)


def _request_keep_awake() -> None:
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    if not _set_execution_state(flags):
        _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def _power_worker() -> None:
    active = False
    while True:
        with _condition:
            while _count <= 0:
                if active:
                    _set_execution_state(ES_CONTINUOUS)
                    active = False
                _condition.wait()

        _request_keep_awake()
        active = True

        with _condition:
            if _count > 0:
                _condition.wait(timeout=_REFRESH_SECONDS)


def _ensure_worker_locked() -> None:
    global _worker
    if os.name != "nt":
        return
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=_power_worker, name="ARPKeepAwake", daemon=True)
    _worker.start()


def keep_awake(reason: str = "") -> None:
    """Prevent Windows system sleep while ARP has an active long-running process."""
    global _count
    if os.name != "nt":
        return
    with _lock:
        _count += 1
        _ensure_worker_locked()
        _condition.notify_all()


def release_keep_awake() -> None:
    global _count
    if os.name != "nt":
        return
    with _lock:
        if _count > 0:
            _count -= 1
        if _count == 0:
            _condition.notify_all()
