from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

from . import state
from .cache import delete_cache_category, delete_cache_file
from .config import STATIC_DIR
from .file_dialogs import browse_path
from .manifests import update_manifest_row
from .media import (
    aspect_preview_at_for_settings,
    auto_crop_for_settings,
    export_media_file,
    media_clip_path,
    pipeline_source_text,
)
from .outpaint_guides import (
    _guide_source_seconds,
    accept_guide_edit,
    add_guide_frame,
    chunk_frame_preview,
    clear_guide_frame_image,
    remove_guide_frame,
    revert_guide_edit,
    sam_guide_mask,
    save_guide_frame,
    save_guide_paint,
    upload_guide_frame_image,
)
from .paths import resolve, resolve_served
from .references import (
    accept_reference_edit,
    delete_color_reference,
    extract_reference_frame,
    install_custom_color_reference,
    merge_manifest_shots,
    preview_reference_frame,
    revert_reference_edit,
    sam_reference_mask,
    save_reference_paint,
    split_manifest_shot,
    shot_rows_for_indices,
    update_shot_boundary,
    update_shot_fade,
)

# These outpaint helpers stay in server.py because they are wired into its chunk/guide internals;
# importing server here would be circular (see state.py's note). server.py injects them at startup
# via bind_context. Declared here so the names resolve statically and tooling can see them.
ensure_outpaint_prepared_canvas = None
outpaint_chunks_state = None
outpaint_chunk_preview = None
update_outpaint_chunk = None
install_outpaint_guide = None
clear_outpaint_guide = None
install_outpaint_end_guide = None
clear_outpaint_end_guide = None

_SERVER_OUTPAINT_OPS = (
    "ensure_outpaint_prepared_canvas",
    "outpaint_chunks_state",
    "outpaint_chunk_preview",
    "update_outpaint_chunk",
    "install_outpaint_guide",
    "clear_outpaint_guide",
    "install_outpaint_end_guide",
    "clear_outpaint_end_guide",
)


def bind_context(context: dict) -> None:
    """Wire in the few server.py-defined outpaint helpers the request handlers call (see above)."""
    globals().update({name: context[name] for name in _SERVER_OUTPAINT_OPS})


# The GUI only ever serves the local user through their own browser. These are the host names a
# loopback address answers to; anything else in the Host header means a remote site is trying to
# reach us by name (a DNS-rebinding attack), so we refuse it.
LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def served_source_paths() -> list[str]:
    """The selected source video lives anywhere on disk, so its folder is added to the set of
    directories the file-serving endpoints are allowed to read from."""
    source = state.APP.settings.get("global", {}).get("source", "")
    return [source] if source else []


class Handler(BaseHTTPRequestHandler):
    server_version = "AIRemasterGUI/1.0"

    def request_host_is_local(self) -> bool:
        """Reject requests whose Host header is not a loopback name. Blocks DNS rebinding, where a
        page on attacker.com re-points its own domain at 127.0.0.1 to talk to this server."""
        host = self.headers.get("Host", "")
        if not host:
            return True  # a missing Host cannot carry an attacker-controlled domain
        return (urlparse("//" + host).hostname or "").lower() in LOOPBACK_HOSTNAMES

    def request_origin_is_local(self) -> bool:
        """Reject state-changing requests carrying a cross-origin Origin/Referer. Blocks CSRF, where
        another site silently POSTs to our localhost endpoints from the user's browser."""
        for header in ("Origin", "Referer"):
            value = self.headers.get(header, "")
            if value:
                return (urlparse(value).hostname or "").lower() in LOOPBACK_HOSTNAMES
        return True  # same-origin requests may omit both; the Host check still applies

    def do_GET(self) -> None:  # noqa: N802
        if not self.request_host_is_local():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path.startswith("/static/"):
            static_path = STATIC_DIR / unquote(parsed.path.removeprefix("/static/"))
            try:
                static_path.resolve().relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(404)
                return
            self.send_static(static_path)
        elif parsed.path == "/site.webmanifest":
            self.send_json(
                {
                    "name": "ARP - AI Remaster Pipeline",
                    "short_name": "ARP",
                    "start_url": "/",
                    "display": "standalone",
                    "background_color": "#101316",
                    "theme_color": "#2d8f7d",
                    "icons": [
                        {"src": "/media?path=assets/branding/arp-app-icon-192.png", "sizes": "192x192", "type": "image/png"},
                        {"src": "/media?path=assets/branding/arp-app-icon-512.png", "sizes": "512x512", "type": "image/png"},
                    ],
                }
            )
        elif parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            view = query.get("active", [""])[0]
            generate_shot_previews = query.get("shot_previews", ["generate"])[0] != "cached"
            self.send_json(state.APP.state(view, generate_shot_previews=generate_shot_previews))
        elif parsed.path == "/api/command":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"command": state.APP.command_for(stage) if stage else []})
        elif parsed.path == "/api/existing-outputs":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"paths": state.APP.existing_outputs(stage) if stage else []})
        elif parsed.path == "/api/media-status":
            query = parse_qs(parsed.query)
            path_text = query.get("path", [""])[0]
            path = resolve_served(path_text, served_source_paths())
            exists = bool(path and path.is_file())
            self.send_json(
                {
                    "ok": True,
                    "path": path_text,
                    "exists": exists,
                    "mtime": path.stat().st_mtime if exists else 0,
                    "running": state.APP.process is not None and state.APP.process.poll() is None,
                    "running_stage": state.APP.running_stage,
                    "log_count": len(state.APP.log),
                }
            )
        elif parsed.path == "/api/comfy":
            url = parse_qs(parsed.query).get("url", ["http://127.0.0.1:8188"])[0].rstrip("/")
            # Only reach out over HTTP(S). urlopen also speaks file://, ftp://, etc., which would
            # turn this connectivity check into a way to read local files or probe other services.
            if urlparse(url).scheme not in ("http", "https"):
                self.send_json({"ok": False, "error": "ComfyUI URL must be http or https."})
                return
            try:
                with urlopen(url + "/queue", timeout=3) as response:
                    self.send_json({"ok": True, "queue": json.loads(response.read().decode("utf-8"))})
            except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/logfile":
            path = resolve_served(parse_qs(parsed.query).get("path", [""])[0], served_source_paths())
            text = path.read_text(encoding="utf-8", errors="replace")[-12000:] if path and path.exists() else ""
            self.send_json({"text": text})
        elif parsed.path == "/api/openai-models":
            token = state.APP.settings.get("references", {}).get("openai_api_key", "").strip()
            if not token:
                self.send_json({"ok": False, "error": "Add your OpenAI API key first."})
                return
            try:
                request = Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {token}"})
                with urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                models = sorted(
                    item.get("id", "")
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and model_looks_image_capable(str(item.get("id", "")))
                )
                self.send_json({"ok": True, "models": models})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-preview":
            query = parse_qs(parsed.query)
            try:
                frame_text = query.get("frame", [""])[0]
                frame = int(frame_text) if str(frame_text).strip() != "" else None
                path = preview_reference_frame(query.get("manifest", [""])[0], int(query.get("index", ["0"])[0]), float(query.get("time", ["0"])[0]), frame=frame)
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/aspect-preview":
            query = parse_qs(parsed.query)
            try:
                path = aspect_preview_at_for_settings(state.APP.settings, float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-auto-crop":
            query = parse_qs(parsed.query)
            try:
                result = auto_crop_for_settings(state.APP.settings, float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, **result, "state": state.APP.state("outpaint")})
            except Exception as exc:
                state.APP.log.append(f"Auto Crop failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk-preview":
            query = parse_qs(parsed.query)
            try:
                preview = outpaint_chunk_preview(
                    state.APP.settings,
                    int(query.get("chunk_index", ["0"])[0]),
                    query.get("kind", ["source"])[0],
                    query.get("position", ["middle"])[0],
                )
                self.send_json({"ok": True, "preview": preview})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-guide-preview":
            query = parse_qs(parsed.query)
            try:
                chunk_index = int(query.get("chunk_index", ["0"])[0])
                frame_idx = int(query.get("frame_idx", ["0"])[0])
                source_text = state.APP.outpaint_source_for()
                if not source_text:
                    self.send_json({"ok": False, "error": "No source material"})
                else:
                    chunks_state = outpaint_chunks_state(state.APP.settings)
                    rows = chunks_state.get("rows", [])
                    row = next((r for r in rows if r.get("index") == chunk_index), None)
                    if row is None:
                        self.send_json({"ok": False, "error": "Chunk not found"})
                    else:
                        range_source = ensure_outpaint_prepared_canvas(source_text, state.APP.settings.get("outpaint", {}))
                        fps = float(row.get("fps", 24) or 24)
                        secs = _guide_source_seconds(row, frame_idx, fps)
                        cache_key = f"gfprev_{chunk_index}_{frame_idx}"
                        preview = chunk_frame_preview(range_source, secs, cache_key)
                        self.send_json({"ok": True, "preview": preview})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/media":
            query = parse_qs(parsed.query)
            path = resolve_served(unquote(query.get("path", [""])[0]), served_source_paths())
            if path is None:
                self.send_error(404)
                return
            if "clip_start" in query or "clip_end" in query:
                try:
                    start = float(query.get("clip_start", ["0"])[0])
                    end = float(query.get("clip_end", [str(start + 0.041)])[0])
                    path = media_clip_path(path, start, end, query.get("clip_key", [""])[0])
                except FileNotFoundError:
                    self.send_error(404)
                    return
                except Exception as exc:
                    state.APP.log.append(f"Shot video preview failed: {exc}")
                    self.send_error(404)
                    return
            self.send_media(path)
        else:
            self.send_error(404)

    def _send_result(self, label: str, produce) -> None:
        """Success path for endpoints answering {"ok": True, **<dict>}. produce() returns that
        dict; on any error, log "<label> failed: …" and answer {"ok": False, "error": …}."""
        try:
            self.send_json({"ok": True, **produce()})
        except Exception as exc:
            state.APP.log.append(f"{label} failed: {exc}")
            self.send_json({"ok": False, "error": str(exc)})

    def _send_action(self, label: str, run, view: str = "", extra_key: str | None = None) -> None:
        """Success path for APP.run_* endpoints returning (ok, message) or (ok, message, value):
        answer {ok, message, state, error}, including fresh state only when ok and the optional
        third value under extra_key. On error, same log + error response as _send_result."""
        try:
            result = run()
            ok, message = result[0], result[1]
            payload = {"ok": ok, "message": message, "state": state.APP.state(view) if ok else None, "error": "" if ok else message}
            if extra_key is not None:
                payload[extra_key] = result[2]
            self.send_json(payload)
        except Exception as exc:
            state.APP.log.append(f"{label} failed: {exc}")
            self.send_json({"ok": False, "error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if not self.request_host_is_local() or not self.request_origin_is_local():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        data = self.read_json()
        if parsed.path == "/api/settings":
            state.APP.update_settings(str(data.get("stage", "")), data.get("values", {}))
            self.send_json({"ok": True})
        elif parsed.path == "/api/run":
            ok, message = state.APP.run_all() if data.get("all") else state.APP.run_stage(str(data.get("stage", "")))
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/upscale-preview":
            ok, message = state.APP.run_upscale_preview()
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/stop":
            state.APP.stop()
            self.send_json({"ok": True})
        elif parsed.path == "/api/quit":
            # Kill any running stage, answer, then stop the server so the launching shell
            # returns to its prompt with nothing left running. Answer before shutdown so
            # this response still reaches the browser. request_quit is imported here rather
            # than at module load because lifecycle imports http_handler (a cycle).
            from .lifecycle import request_quit

            state.APP.stop_for_quit()
            self.send_json({"ok": True})
            request_quit(self.server)
        elif parsed.path == "/api/shot-scrub":
            self._send_result("Shot scrub", lambda: {
                **extract_reference_frame(
                    str(data.get("manifest", "")),
                    int(data.get("index", 0)),
                    float(data.get("time", 0)),
                    int(data.get("frame")) if data.get("frame") is not None and str(data.get("frame")).strip() != "" else None,
                ),
                "state": state.APP.state("references"),
            })
        elif parsed.path == "/api/shot-prompt":
            self._send_result("Shot prompt save", lambda: update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"prompt": str(data.get("prompt", ""))}) or {})
        elif parsed.path == "/api/shot-enabled":
            self._send_result("Shot enabled save", lambda: update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"enabled": "true" if data.get("enabled") else "false"}) or {})
        elif parsed.path == "/api/shot-merge":
            self._send_result("Shot merge", lambda: {**merge_manifest_shots(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state("shots")})
        elif parsed.path == "/api/shot-split":
            self._send_result("Shot split", lambda: {**split_manifest_shot(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state("shots")})
        elif parsed.path == "/api/shot-boundary":
            def update_boundary_result():
                manifest = str(data.get("manifest", ""))
                index = int(data.get("index", 0))
                edge = str(data.get("edge", ""))
                result = update_shot_boundary(
                    manifest,
                    index,
                    edge,
                    float(data.get("time", 0)),
                    int(data.get("frame")) if data.get("frame") is not None and str(data.get("frame")).strip() != "" else None,
                )
                return {
                    **result,
                    "rows": shot_rows_for_indices(manifest, (index - 2, index - 1, index, index + 1)),
                    "log": "\n".join(state.APP.log[-800:]),
                    "log_count": len(state.APP.log),
                }
            self._send_result("Shot boundary update", update_boundary_result)
        elif parsed.path == "/api/shot-fade":
            self._send_result("Shot fade update", lambda: {**update_shot_fade(str(data.get("manifest", "")), int(data.get("index", 0)), bool(data.get("enabled")), str(data.get("crossfade_seconds", ""))), "state": state.APP.state("shots")})
        elif parsed.path == "/api/shot-upscale-strength":
            def update_upscale_strength():
                manifest = str(data.get("manifest", ""))
                index = int(data.get("index", 0))
                raw = str(data.get("strength", "")).strip()
                if raw:
                    strength = max(0.0, min(100.0, float(raw)))
                    raw = f"{strength:.3f}".rstrip("0").rstrip(".")
                update_manifest_row(resolve(manifest), index, {"upscale_strength": raw})
                return {"strength": raw, "state": state.APP.state("upscale")}
            self._send_result("Shot upscale strength update", update_upscale_strength)
        elif parsed.path == "/api/reference-regenerate":
            self._send_action("Reference regeneration", lambda: state.APP.run_reference_regeneration(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("provider", "qwen"))))
        elif parsed.path == "/api/reference-delete":
            self._send_result("Reference delete", lambda: {**delete_color_reference(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state()})
        elif parsed.path == "/api/reference-custom":
            self._send_result("Custom reference install", lambda: {**install_custom_color_reference(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state()})
        elif parsed.path == "/api/reference-mask-sam":
            self._send_result("Reference smart mask", lambda: {**sam_reference_mask(
                str(data.get("manifest", "")),
                int(data.get("index", 0)),
                data.get("points", []) if isinstance(data.get("points", []), list) else [],
                int(data.get("width", 1)),
                int(data.get("height", 1)),
                int(data.get("tolerance", 10)),
            )})
        elif parsed.path == "/api/guide-frame-mask-sam":
            self._send_result("Guide smart mask", lambda: {**sam_guide_mask(
                int(data.get("chunk_index", 0)),
                int(data.get("guide_index", 0)),
                data.get("points", []) if isinstance(data.get("points", []), list) else [],
                int(data.get("width", 1)),
                int(data.get("height", 1)),
                str(data.get("fallback_path", "")),
            )})
        elif parsed.path == "/api/reference-edit-preview":
            self._send_action("Reference edit preview", lambda: state.APP.run_reference_edit_preview(
                str(data.get("manifest", "")),
                int(data.get("index", 0)),
                str(data.get("instruction", "")),
                str(data.get("mask", "")),
                str(data.get("sampled_color", "")),
            ), view="references", extra_key="preview")
        elif parsed.path == "/api/reference-edit-accept":
            self._send_result("Reference edit accept", lambda: {**accept_reference_edit(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("preview", ""))), "state": state.APP.state("references")})
        elif parsed.path == "/api/reference-edit-revert":
            self._send_result("Reference edit revert", lambda: {**revert_reference_edit(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state("references")})
        elif parsed.path == "/api/reference-paint-save":
            self._send_result("Reference paint save", lambda: {**save_reference_paint(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("image", ""))), "state": state.APP.state("references")})
        elif parsed.path == "/api/export-media":
            self._send_result("Media export", lambda: {**export_media_file(str(data.get("path", "")))})
        elif parsed.path == "/api/outpaint-chunk":
            self._send_result("Outpaint chunk save", lambda: update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")), str(data.get("guide_strength", "")), str(data.get("guide_end_strength", "")), data.get("custom_length", None), str(data.get("offset_x", "0")), str(data.get("offset_y", "0")), data.get("auto_start_guide", True)) or {"state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-chunk-regenerate":
            self._send_action("Outpaint chunk regeneration", lambda: (
                update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")), str(data.get("guide_strength", "")), str(data.get("guide_end_strength", "")), data.get("custom_length", None), str(data.get("offset_x", "0")), str(data.get("offset_y", "0")), data.get("auto_start_guide", True)),
                state.APP.run_outpaint_chunk(int(data.get("index", 0))),
            )[1], view="outpaint")
        elif parsed.path == "/api/outpaint-anchor":
            self._send_result("Outpaint guide install", lambda: {**install_outpaint_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-anchor-clear":
            self._send_result("Outpaint guide clear", lambda: {**clear_outpaint_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-anchor-generate":
            self._send_action("Outpaint guide generation", lambda: state.APP.run_outpaint_guide_generation(int(data.get("index", 0)), str(data.get("prompt", ""))), view="outpaint")
        elif parsed.path == "/api/outpaint-end-anchor":
            self._send_result("Outpaint end guide install", lambda: {**install_outpaint_end_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-end-anchor-clear":
            self._send_result("Outpaint end guide clear", lambda: {**clear_outpaint_end_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-end-anchor-generate":
            self._send_action("Outpaint end guide generation", lambda: state.APP.run_outpaint_end_guide_generation(int(data.get("index", 0)), str(data.get("prompt", ""))), view="outpaint")
        elif parsed.path == "/api/guide-frame-add":
            self._send_result("Guide frame add", lambda: {**add_guide_frame(int(data.get("chunk_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-remove":
            self._send_result("Guide frame remove", lambda: {**remove_guide_frame(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-save":
            self._send_result("Guide frame save", lambda: {**save_guide_frame(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), int(data.get("frame_idx", 0)), float(data.get("strength", 0.7))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-upload":
            self._send_result("Guide frame upload", lambda: {**upload_guide_frame_image(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-clear":
            self._send_result("Guide frame clear", lambda: {**clear_guide_frame_image(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-generate":
            self._send_action("Guide frame generation", lambda: state.APP.run_guide_frame_generation(
                int(data.get("chunk_index", 0)),
                int(data.get("guide_index", 0)),
                int(data.get("frame_idx", 0)),
                str(data.get("prompt", "")),
            ), view="outpaint")
        elif parsed.path == "/api/guide-frame-edit-preview":
            self._send_action("Guide frame edit preview", lambda: state.APP.run_guide_edit_preview(
                int(data.get("chunk_index", 0)),
                int(data.get("guide_index", 0)),
                str(data.get("instruction", "")),
                str(data.get("mask", "")),
                str(data.get("sampled_color", "")),
            ), view="outpaint", extra_key="preview")
        elif parsed.path == "/api/guide-frame-edit-accept":
            self._send_result("Guide frame edit accept", lambda: {**accept_guide_edit(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), str(data.get("preview", ""))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-edit-revert":
            self._send_result("Guide frame edit revert", lambda: {**revert_guide_edit(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-paint-save":
            self._send_result("Guide paint save", lambda: {**save_guide_paint(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), str(data.get("image", ""))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/browse":
            self._send_result("Browse", lambda: {"path": browse_path(str(data.get("kind", "file")), str(data.get("current", "")))})
        elif parsed.path == "/api/browse-global-source":
            def browse_global_source() -> dict:
                selected = browse_path("file", str(data.get("current", "")))
                if selected:
                    state.APP.update_settings("global", {"source": selected})
                return {"path": selected, "state": state.APP.state("global")}

            self._send_result("Browse", browse_global_source)
        elif parsed.path == "/api/overview-clear":
            state.APP.clear_overview()
            self.send_json({"ok": True, "state": state.APP.state("global")})
        elif parsed.path == "/api/project-save":
            self._send_result("Project save", lambda: {**state.APP.save_project(bool(data.get("save_as"))), "state": state.APP.state("global")})
        elif parsed.path == "/api/project-load":
            self._send_result("Project load", lambda: {**state.APP.load_project(), "state": state.APP.state("global")})
        elif parsed.path == "/api/cache-delete":
            def cache_delete() -> dict:
                if data.get("all"):
                    result = delete_cache_category("all")
                elif data.get("category"):
                    result = delete_cache_category(str(data.get("category", "")))
                else:
                    result = delete_cache_file(str(data.get("path", "")))
                return {**result, "state": state.APP.state()}

            self._send_result("Cache delete", cache_delete)
        else:
            self.send_error(404)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            self.send_error(404)
            return
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = 200
        if range_header.startswith("bytes="):
            raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
            left, _, right = raw_range.partition("-")
            try:
                if left:
                    start = int(left)
                    if right:
                        end = int(right)
                elif right:
                    suffix = int(right)
                    start = max(0, file_size - suffix)
                if start < 0 or end < start or start >= file_size:
                    raise ValueError
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except FileNotFoundError:
            return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def model_looks_image_capable(model_id: str) -> bool:
    text = model_id.lower()
    return text.startswith("gpt-image") or text.startswith("dall-e")

