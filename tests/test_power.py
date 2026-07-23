from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from ai_remaster_gui import power


class PowerTests(unittest.TestCase):
    def test_windows_keep_awake_request_is_owned_by_worker_thread(self) -> None:
        calls: list[tuple[int, int]] = []
        changed = threading.Event()
        caller_thread = threading.get_ident()
        awake_flags = power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED | power.ES_AWAYMODE_REQUIRED

        def fake_set_execution_state(flags: int) -> bool:
            calls.append((threading.get_ident(), flags))
            changed.set()
            return True

        def wait_for_flags(flags: int, thread_id: int | None = None) -> int:
            deadline = time.time() + 2.0
            while time.time() < deadline:
                for seen_thread, seen_flags in reversed(calls):
                    if seen_flags == flags and (thread_id is None or seen_thread == thread_id):
                        return seen_thread
                changed.clear()
                changed.wait(0.05)
            self.fail(f"Timed out waiting for SetThreadExecutionState({flags:#x}); calls={calls!r}")

        with power._condition:
            power._count = 0
            power._condition.notify_all()

        try:
            with mock.patch.object(power.os, "name", "nt"), mock.patch.object(power, "_set_execution_state", side_effect=fake_set_execution_state):
                power.keep_awake("test")
                worker_thread = wait_for_flags(awake_flags)
                self.assertNotEqual(worker_thread, caller_thread)

                power.release_keep_awake()
                self.assertEqual(wait_for_flags(power.ES_CONTINUOUS, worker_thread), worker_thread)
        finally:
            with power._condition:
                power._count = 0
                power._condition.notify_all()


if __name__ == "__main__":
    unittest.main()
