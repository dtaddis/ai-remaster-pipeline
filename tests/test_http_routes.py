"""Routing/behaviour tests for the GUI request handler.

These pin down the set of routes the handler answers and the response shapes produced by the two
shared helpers (_send_result and _send_action), so the consolidation of the per-endpoint
boilerplate cannot silently drop a route or change a payload.
"""

from __future__ import annotations

import inspect
import re
import threading
import unittest
from types import MethodType
from unittest import mock

from ai_remaster_gui import http_handler, state
from ai_remaster_gui import server  # noqa: F401  (imported for its bind_context side effect)

EXPECTED_GET_ROUTES = {
    "/",
    "/site.webmanifest",
    "/api/state",
    "/api/command",
    "/api/existing-outputs",
    "/api/media-status",
    "/api/comfy",
    "/api/logfile",
    "/api/openai-models",
    "/api/shot-preview",
    "/api/upscale-shot-comparison",
    "/api/aspect-preview",
    "/api/outpaint-auto-crop",
    "/api/outpaint-chunk-preview",
    "/api/outpaint-guide-preview",
    "/media",
}

EXPECTED_POST_ROUTES = {
    "/api/settings", "/api/run", "/api/upscale-preview", "/api/stop", "/api/quit",
    "/api/shot-scrub", "/api/shot-prompt", "/api/shot-enabled", "/api/shot-merge",
    "/api/shot-split", "/api/shot-boundary", "/api/shot-fade", "/api/shot-upscale-strength", "/api/reference-regenerate",
    "/api/reference-delete", "/api/reference-custom", "/api/reference-mask-sam",
    "/api/guide-frame-mask-sam", "/api/reference-edit-preview", "/api/reference-edit-accept",
    "/api/reference-edit-revert", "/api/reference-paint-save", "/api/export-media",
    "/api/outpaint-chunk", "/api/outpaint-chunk-regenerate", "/api/outpaint-anchor",
    "/api/outpaint-anchor-clear", "/api/outpaint-anchor-generate", "/api/outpaint-end-anchor",
    "/api/outpaint-end-anchor-clear", "/api/outpaint-end-anchor-generate", "/api/guide-frame-add",
    "/api/guide-frame-remove", "/api/guide-frame-save", "/api/guide-frame-upload",
    "/api/guide-frame-clear", "/api/guide-frame-generate", "/api/guide-frame-edit-preview",
    "/api/guide-frame-edit-accept", "/api/guide-frame-edit-revert", "/api/guide-paint-save",
    "/api/browse", "/api/browse-global-source", "/api/overview-clear", "/api/project-save",
    "/api/project-load", "/api/cache-delete",
}


def _routes_in(method_name: str) -> set[str]:
    source = inspect.getsource(getattr(http_handler.Handler, method_name))
    return set(re.findall(r'parsed\.path == "([^"]+)"', source))


class RouteCoverageTests(unittest.TestCase):
    def test_get_routes_unchanged(self) -> None:
        self.assertEqual(_routes_in("do_GET"), EXPECTED_GET_ROUTES)

    def test_post_routes_unchanged(self) -> None:
        self.assertEqual(_routes_in("do_POST"), EXPECTED_POST_ROUTES)


class FakeApp:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.settings: dict = {"global": {}}

    def state(self, view: str = "") -> dict:
        return {"_view": view}


class _Handler(http_handler.Handler):
    def __init__(self, path: str, body: dict | None = None) -> None:
        self.path = path
        self.headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload):  # noqa: D102
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None):  # noqa: D102
        self.responses["error"] = code


class HelperBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_app = state.APP
        state.APP = FakeApp()

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_send_result_success_merges_dict_and_state(self) -> None:
        with mock.patch.object(http_handler, "merge_manifest_shots", return_value={"merged": 2}):
            handler = _Handler("/api/shot-merge", {"manifest": "m.csv", "index": 0})
            handler.do_POST()
        self.assertEqual(handler.responses["json"], {"ok": True, "merged": 2, "state": {"_view": "shots"}})

    def test_send_result_failure_logs_and_reports(self) -> None:
        boom = RuntimeError("kaboom")
        with mock.patch.object(http_handler, "merge_manifest_shots", side_effect=boom):
            handler = _Handler("/api/shot-merge", {"manifest": "m.csv", "index": 0})
            handler.do_POST()
        self.assertEqual(handler.responses["json"], {"ok": False, "error": "kaboom"})
        self.assertTrue(any("Shot merge failed: kaboom" in line for line in state.APP.log))

    def test_send_result_side_effect_only_returns_ok(self) -> None:
        with mock.patch.object(http_handler, "update_manifest_row", return_value=None) as row:
            handler = _Handler("/api/shot-prompt", {"manifest": "m.csv", "index": 1, "prompt": "hi"})
            handler.do_POST()
        self.assertEqual(handler.responses["json"], {"ok": True})
        row.assert_called_once()

    def test_shot_boundary_returns_row_patch_instead_of_full_state(self) -> None:
        with (
            mock.patch.object(http_handler, "update_shot_boundary", return_value={"frame": "12"}) as boundary,
            mock.patch.object(http_handler, "shot_rows_for_indices", return_value=[{"index": 0}, {"index": 1}]) as rows,
        ):
            handler = _Handler("/api/shot-boundary", {"manifest": "m.csv", "index": 0, "edge": "end", "frame": 12})
            handler.do_POST()

        payload = handler.responses["json"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["rows"], [{"index": 0}, {"index": 1}])
        self.assertNotIn("state", payload)
        boundary.assert_called_once_with("m.csv", 0, "end", 0.0, 12)
        rows.assert_called_once_with("m.csv", (-2, -1, 0, 1))

    def test_state_route_can_request_cached_shot_previews(self) -> None:
        state.APP.state = mock.Mock(return_value={"ok": True})
        handler = _Handler("/api/state?active=shots&shot_previews=cached")
        handler.do_GET()

        self.assertEqual(handler.responses["json"], {"ok": True})
        state.APP.state.assert_called_once_with("shots", generate_shot_previews=False)

    def test_upscale_shot_comparison_returns_cached_frame_pair(self) -> None:
        state.APP.upscale_shot_comparison_frames = mock.Mock(
            return_value={"before": "before.jpg", "after": "after.jpg", "stale": "false"}
        )
        handler = _Handler("/api/upscale-shot-comparison?index=3&time=4.25")
        handler.do_GET()

        self.assertEqual(
            handler.responses["json"],
            {"ok": True, "before": "before.jpg", "after": "after.jpg", "stale": "false"},
        )
        state.APP.upscale_shot_comparison_frames.assert_called_once_with(3, 4.25)

    def test_send_action_returns_ok_message_state(self) -> None:
        state.APP.run_reference_regeneration = lambda *a: (True, "done")
        handler = _Handler("/api/reference-regenerate", {"manifest": "m.csv", "index": 0})
        handler.do_POST()
        self.assertEqual(
            handler.responses["json"],
            {"ok": True, "message": "done", "state": {"_view": ""}, "error": ""},
        )

    def test_send_action_failure_has_null_state(self) -> None:
        state.APP.run_reference_regeneration = lambda *a: (False, "nope")
        handler = _Handler("/api/reference-regenerate", {"manifest": "m.csv", "index": 0})
        handler.do_POST()
        payload = handler.responses["json"]
        self.assertEqual(payload["ok"], False)
        self.assertIsNone(payload["state"])
        self.assertEqual(payload["error"], "nope")

    def test_send_action_extra_key_carries_preview(self) -> None:
        state.APP.run_guide_edit_preview = lambda *a: (True, "edited", "preview.png")
        handler = _Handler("/api/guide-frame-edit-preview", {"chunk_index": 0, "guide_index": 0})
        handler.do_POST()
        self.assertEqual(handler.responses["json"]["preview"], "preview.png")


class StateLockTests(unittest.TestCase):
    def test_shot_view_builds_after_app_lock_is_released(self) -> None:
        app = server.PipelineApp.__new__(server.PipelineApp)
        app.settings = {"global": {"colorize": "true"}}
        app.project_path = None
        app.log = []
        app.process = None
        app.running_stage = ""
        app.running_stage_key = ""
        app.running_reference_manifest = ""
        app.running_reference_index = None
        app.run_started_at = 0.0
        app.lock = threading.Lock()

        app.source_media_state = MethodType(lambda self, source: {
            "previews": [],
            "info": {},
            "monochrome": True,
            "analysis": {},
            "aspect_preview": "",
        }, app)
        app.files_for = MethodType(lambda self, stage: [], app)
        app.active_stages = MethodType(lambda self: (), app)
        app.progress = MethodType(lambda self: [], app)
        app.phase_progress = MethodType(lambda self: {"global": {"percent": 0, "label": "Waiting"}, "stages": []}, app)
        app.expected_outputs = MethodType(lambda self, key: [], app)
        app.existing_outputs = MethodType(lambda self, key: [], app)
        app.upscale_preview_state = MethodType(lambda self: {}, app)
        app.output_selection_state = MethodType(lambda self: {}, app)

        def fake_shot_views(settings, generate_previews=True):
            self.assertTrue(app.lock.acquire(blocking=False))
            app.lock.release()
            return {"shots": [{"index": 0}], "shots_manifest": "m.csv"}

        with (
            mock.patch.object(server, "shot_views", side_effect=fake_shot_views),
            mock.patch.object(server, "source_section_state", return_value={}),
            mock.patch.object(server, "system_status", return_value={}),
        ):
            payload = app.state("shots")

        self.assertEqual(payload["shot_views"]["shots"], [{"index": 0}])


if __name__ == "__main__":
    unittest.main()
