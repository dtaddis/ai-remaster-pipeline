from __future__ import annotations

import ctypes
import os
import threading


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

_lock = threading.Lock()
_count = 0


def _set_execution_state(flags: int) -> bool:
    if os.name != "nt":
        return True
    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    return bool(result)


def keep_awake(reason: str = "") -> None:
    """Prevent Windows system sleep while ARP has an active long-running process."""
    global _count
    with _lock:
        _count += 1
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        if not _set_execution_state(flags):
            _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def release_keep_awake() -> None:
    global _count
    with _lock:
        if _count > 0:
            _count -= 1
        if _count == 0:
            _set_execution_state(ES_CONTINUOUS)
