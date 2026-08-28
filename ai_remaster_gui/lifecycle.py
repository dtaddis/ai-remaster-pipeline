from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .comfy import comfy_busy_message, comfy_is_running, comfy_queue, discover_comfy_instances
from .config import CONFIG_FILE, ROOT, current_config
from .http_handler import Handler

STARTED_COMFY_PROCESS: subprocess.Popen | None = None
COMFY_STARTUP_LOG = ROOT / "output" / "logs" / "comfyui-startup.log"


def port_can_bind(host: str, port: int) -> bool:
    """Return whether a fresh local server can bind this endpoint now."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, int(port)))
        return True
    except OSError:
        return False


def persist_comfy_endpoint(config: dict[str, str], host: str, port: int) -> dict[str, str]:
    """Persist a managed ComfyUI endpoint without discarding installer settings."""
    updated = dict(config)
    parsed = urlparse(updated.get("comfy_url", "http://127.0.0.1:8188"))
    scheme = parsed.scheme or "http"
    updated["comfy_host"] = host
    updated["comfy_port"] = str(port)
    updated["comfy_url"] = f"{scheme}://{host}:{port}"
    stored: dict[str, object] = {}
    if CONFIG_FILE.exists():
        try:
            loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                stored.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    stored.update(updated)
    CONFIG_FILE.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    return updated


def ensure_bindable_managed_comfy_endpoint(config: dict[str, str]) -> dict[str, str]:
    """Move ARP-managed ComfyUI away from occupied or Windows-reserved ports.

    Hyper-V/WSL can reserve large dynamic port ranges after an update. A bind
    probe catches both those exclusions and ordinary port conflicts without
    encoding any Windows-specific range assumptions.
    """
    if str(config.get("comfy_managed_by_arp", "true")).lower() != "true":
        return config
    host = config.get("comfy_host", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        return config
    try:
        configured_port = int(config.get("comfy_port", "8188"))
    except (TypeError, ValueError):
        configured_port = 8188
    if port_can_bind(host, configured_port):
        return config

    for candidate in range(8188, 9000):
        if candidate != configured_port and port_can_bind(host, candidate):
            updated = persist_comfy_endpoint(config, host, candidate)
            startup_log(
                f"ComfyUI port {configured_port} is unavailable or reserved; "
                f"ARP will use {updated['comfy_url']} instead."
            )
            return updated
    raise RuntimeError("ARP could not find a bindable local port for managed ComfyUI.")


def ensure_comfy_available_for_stage(stage_title: str) -> tuple[bool, str]:
    if os.environ.get("AI_REMASTER_NO_COMFY_AUTOSTART") == "1":
        return True, ""
    config = current_config()
    url = config.get("comfy_url", "http://127.0.0.1:8188")
    instances = discover_comfy_instances(url)
    if len(instances) > 1:
        message = "Multiple ComfyUI instances appear to be running: " + ", ".join(instances) + ". Close extras or update .ai_remaster_config.json to the one ARP should use."
        startup_log(message)
        return False, message
    if instances:
        queue = comfy_queue(url)
        message = comfy_busy_message(url, queue)
        if message:
            startup_log(message)
            return False, message
        startup_log(f"Found ComfyUI already running at {url}; queue is idle.")
        return True, ""
    if STARTED_COMFY_PROCESS and STARTED_COMFY_PROCESS.poll() is None:
        startup_log(f"ComfyUI launch is already in progress at {url}.")
        if wait_for_comfy_ready(url, STARTED_COMFY_PROCESS, float(os.environ.get("AI_REMASTER_COMFY_START_TIMEOUT", "180"))):
            return True, ""
        message = comfy_startup_failure_message(url, stage_title)
        startup_log(message)
        return False, message
    startup_log(f"ComfyUI is not running at {url}; launching it now.")
    if not start_comfy_if_needed(monitor=False):
        message = comfy_startup_failure_message(url, stage_title)
        return False, message
    url = current_config().get("comfy_url", url)
    if wait_for_comfy_ready(url, STARTED_COMFY_PROCESS, float(os.environ.get("AI_REMASTER_COMFY_START_TIMEOUT", "180"))):
        return True, ""
    message = comfy_startup_failure_message(url, stage_title)
    startup_log(message)
    return False, message

def startup_log(message: str) -> None:
    app = globals().get("APP")
    if app is not None:
        app.log.append(message)
    else:
        print(message, flush=True)

def comfy_startup_log_path() -> Path:
    return COMFY_STARTUP_LOG

def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])

def torch_cuda_warning() -> str:
    try:
        import torch
    except Exception as exc:
        return f"PyTorch could not be imported from ARP's Python environment: {exc}. Re-run install_windows.bat."
    version = getattr(torch, "__version__", "unknown")
    cuda_build = getattr(torch.version, "cuda", None)
    if not cuda_build:
        return f"ARP's Python environment has a CPU-only PyTorch build ({version}). Re-run install_windows.bat so it can install CUDA PyTorch."
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return f"PyTorch CUDA probe failed for torch {version} / CUDA {cuda_build}: {exc}. Check your NVIDIA driver and rerun install_windows.bat."
    if not available:
        return f"PyTorch has CUDA support (torch {version}, CUDA {cuda_build}), but no CUDA device is visible. Check your NVIDIA driver/GPU before running ComfyUI."
    return ""

def wait_for_comfy_ready(url: str, process: subprocess.Popen | None, timeout_seconds: float = 180.0) -> bool:
    global STARTED_COMFY_PROCESS
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if comfy_is_running(url):
            startup_log(f"ComfyUI is ready at {url}")
            return True
        if process and process.poll() is not None:
            startup_log(f"ComfyUI exited before it became ready. Exit code: {process.returncode}")
            startup_log("Check the ComfyUI console window for the error.")
            if process is STARTED_COMFY_PROCESS:
                STARTED_COMFY_PROCESS = None
            return False
        time.sleep(2)
    startup_log(f"Timed out waiting for ComfyUI to become ready at {url}")
    startup_log("Check the ComfyUI console window for details.")
    return False

def monitor_comfy_startup(url: str, process: subprocess.Popen | None) -> None:
    wait_for_comfy_ready(url, process, float(os.environ.get("AI_REMASTER_COMFY_START_TIMEOUT", "180")))

def comfy_startup_failure_message(url: str, stage_title: str) -> str:
    tail = tail_text(comfy_startup_log_path(), max_lines=20)
    detail = f" Recent startup log:\n{tail}" if tail else ""
    return (
        f"ComfyUI did not become ready at {url}, so {stage_title} was not started. "
        f"Close any stuck ComfyUI window and retry. If it fails again, check the ComfyUI console "
        f"and {comfy_startup_log_path()} for the startup error."
        f"{detail}"
    )

def start_comfy_if_needed(monitor: bool = True) -> bool:
    global STARTED_COMFY_PROCESS
    config = current_config()
    url = config.get("comfy_url", "http://127.0.0.1:8188")
    if STARTED_COMFY_PROCESS and STARTED_COMFY_PROCESS.poll() is None:
        startup_log(f"ComfyUI launch is already in progress at {url}.")
        return True
    instances = discover_comfy_instances(url)
    if len(instances) > 1:
        startup_log("Multiple ComfyUI instances appear to be running: " + ", ".join(instances) + ". Close extras or update .ai_remaster_config.json to the one ARP should use.")
        return False
    if instances:
        if instances[0].rstrip("/") == url.rstrip("/"):
            startup_log(f"ComfyUI already running at {url}")
            return True
        else:
            startup_log(f"ComfyUI appears to be running at {instances[0]}, but ARP is configured for {url}. Close it or update .ai_remaster_config.json.")
            return False
    config = ensure_bindable_managed_comfy_endpoint(config)
    url = config.get("comfy_url", url)
    comfy_dir = Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    if str(config.get("comfy_managed_by_arp", "true")).lower() != "true":
        startup_log("Using an external ComfyUI checkout. ARP can start it, but install_windows.bat will not update ComfyUI core for this path.")
    warning = torch_cuda_warning()
    if warning:
        startup_log("Warning: " + warning)
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        if CONFIG_FILE.exists():
            startup_log(f"ComfyUI is configured but main.py was not found: {main_py}")
            startup_log("Run install_windows.bat again and choose your ComfyUI directory.")
        else:
            startup_log("ComfyUI is not configured yet.")
            startup_log("Run install_windows.bat again and choose whether to clone ComfyUI or use an existing ComfyUI directory.")
        return False
    host = config.get("comfy_host", "127.0.0.1")
    port = str(config.get("comfy_port", "8188"))
    command = [sys.executable, "main.py", "--listen", host, "--port", port]
    log_path = comfy_startup_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 72 + "\n")
        log.write(f"Starting ComfyUI: {' '.join(command)}\n")
        log.write(f"Working directory: {comfy_dir}\n")
        log.flush()
    # Launch ComfyUI in its own console window so its live progress (model load, sampling /
    # progress bars) is visible — writing straight to that console keeps the bars as in-place
    # updates. We deliberately do NOT redirect stdout/stderr to a file: that would capture the log
    # but blank out the window, which is the experience we want back. The banner above records the
    # launch command for the GUI's log-file viewer. CREATE_NEW_PROCESS_GROUP scopes our taskkill to
    # that tree on shutdown.
    kwargs: dict = {"cwd": str(comfy_dir)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    STARTED_COMFY_PROCESS = subprocess.Popen(command, **kwargs)
    startup_log(f"Started ComfyUI in a new console window at {url}")
    startup_log("ComfyUI is still starting; the pipeline will wait before queueing prompts.")
    if monitor:
        threading.Thread(target=monitor_comfy_startup, args=(url, STARTED_COMFY_PROCESS), daemon=True).start()
    return True

def stop_started_comfy() -> None:
    global STARTED_COMFY_PROCESS
    process = STARTED_COMFY_PROCESS
    if not process or process.poll() is not None:
        STARTED_COMFY_PROCESS = None
        return
    print("Stopping ComfyUI started by ARP...")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except Exception as exc:
        print(f"Could not stop ComfyUI cleanly: {exc}")
    finally:
        STARTED_COMFY_PROCESS = None

def install_shutdown_handlers() -> None:
    atexit.register(stop_started_comfy)

    def handle_signal(signum, _frame) -> None:
        stop_started_comfy()
        raise SystemExit(0 if signum in (signal.SIGINT, signal.SIGTERM) else 1)

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handle_signal)

def request_quit(server: ThreadingHTTPServer) -> None:
    """Stop the GUI server shortly after the current response is flushed.

    serve_forever() runs on the main thread, so shutdown() has to come from another
    thread; the brief delay lets the /api/quit response reach the browser before the
    socket closes. main()'s finally block then runs server_close() and stops ComfyUI.
    """
    def _shutdown() -> None:
        time.sleep(0.3)
        try:
            server.shutdown()
        except Exception as exc:
            print(f"Could not stop ARP cleanly: {exc}")

    threading.Thread(target=_shutdown, daemon=True).start()

def create_server(host: str, requested_port: int) -> ThreadingHTTPServer:
    ports = [requested_port, 0] if requested_port != 0 else [0]
    last_error: OSError | None = None
    for port in ports:
        try:
            return ThreadingHTTPServer((host, port), Handler)
        except OSError as exc:
            last_error = exc
            if port != 0:
                print(f"GUI port {port} was unavailable ({exc}); trying a free port.")
    assert last_error is not None
    raise last_error
