from __future__ import annotations

import shutil
import subprocess
import sys
import os
import json
from pathlib import Path

from . import state
from .config import IMAGE_EXTS, ROOT, VIDEO_EXTS
from .paths import rel, resolve
from .project_io import last_browse_dir


def parse_duration(value: str | None) -> float | None:
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return None

def browse_path(kind: str, current: str = "") -> str:
    initial = browse_initial_path(kind, current)
    if os.name == "nt":
        selected = browse_path_windows(kind, initial)
    elif sys.platform == "darwin":
        selected = browse_path_macos(kind, initial)
    else:
        selected = browse_path_linux(kind, initial)
    remember_browse_dir(selected)
    return selected


def browse_source_paths(current: str = "") -> list[str]:
    """Choose one video or one-or-more still images for the global source."""
    initial = browse_initial_path("file", current)
    if os.name == "nt":
        selected = browse_source_paths_windows(initial)
    elif sys.platform == "darwin":
        selected = browse_source_paths_macos(initial)
    else:
        selected = browse_source_paths_linux(initial)
    if selected:
        remember_browse_dir(selected[0])
    return selected

def browse_initial_path(kind: str, current: str = "") -> Path:
    save_kinds = {"save", "save_image", "project_save"}
    last_dir = last_browse_dir()
    if not current:
        return last_dir or ROOT

    current_path = resolve(current)
    if kind in save_kinds:
        if last_dir:
            return last_dir / (current_path.name or "output")
        if current_path.parent.exists():
            return current_path
        return current_path

    if last_dir:
        return last_dir
    if current_path.exists():
        return current_path if current_path.is_dir() else current_path.parent
    if current_path.parent.exists():
        return current_path.parent
    return ROOT

def remember_browse_dir(selected: str) -> None:
    if not selected or state.APP is None:
        return
    path = resolve(selected)
    folder = path if path.is_dir() else path.parent
    if not folder.exists():
        return
    state.APP.settings.setdefault("global", {})["last_browse_dir"] = str(folder)
    state.APP.save()

def browse_path_windows(kind: str, initial: Path) -> str:
    initial_dir = initial if initial.is_dir() else initial.parent
    initial_text = str(initial_dir).replace("'", "''")
    initial_file = "" if initial.is_dir() else initial.name.replace("'", "''")
    owner_script = """
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.WindowState = 'Minimized'
$owner.Add_Shown({ $owner.Hide() })
$owner.Show()
$owner.Activate()
"""
    show_script = """
try {
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dialog.FileName) }
} finally {
    $owner.Dispose()
}
"""
    if kind == "folder":
        script = f"""
{owner_script}
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.SelectedPath = '{initial_text}'
try {{
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($dialog.SelectedPath) }}
}} finally {{
    $owner.Dispose()
}}
"""
    elif kind in {"save", "save_image", "project_save"}:
        filter_text = (
            "ARP project files (*.arpp)|*.arpp|All files (*.*)|*.*"
            if kind == "project_save"
            else "Image files (*.png;*.jpg;*.jpeg;*.webp)|*.png;*.jpg;*.jpeg;*.webp|All files (*.*)|*.*"
            if kind == "save_image"
            else "Video files (*.mp4;*.mov;*.mkv;*.webm)|*.mp4;*.mov;*.mkv;*.webm|All files (*.*)|*.*"
        )
        script = f"""
{owner_script}
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.FileName = '{initial_file}'
$dialog.Filter = '{filter_text}'
{show_script}
"""
    else:
        filter_text = (
            "ARP project files (*.arpp)|*.arpp|All files (*.*)|*.*"
            if kind == "project_open"
            else "Media/workflow files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.jpg;*.jpeg;*.json;*.csv)|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.jpg;*.jpeg;*.json;*.csv|All files (*.*)|*.*"
        )
        script = f"""
{owner_script}
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.Filter = '{filter_text}'
{show_script}
"""
    if state.APP is not None:
        state.APP.log.append(f"Opening Windows browse dialog for {kind}: {initial_dir}")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Windows file dialog failed.").strip())
    selected = result.stdout.strip()
    if selected:
        state.APP.log.append(f"Browse selected: {selected}")
    else:
        state.APP.log.append("Browse cancelled.")
    return rel(Path(selected)) if selected else ""


def browse_source_paths_windows(initial: Path) -> list[str]:
    initial_dir = initial if initial.is_dir() else initial.parent
    initial_text = str(initial_dir).replace("'", "''")
    initial_file = "" if initial.is_dir() else initial.name.replace("'", "''")
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.WindowState = 'Minimized'
$owner.Add_Shown({{ $owner.Hide() }})
$owner.Show()
$owner.Activate()
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.InitialDirectory = '{initial_text}'
$dialog.FileName = '{initial_file}'
$dialog.Multiselect = $true
$dialog.Filter = 'Video or image files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.bmp;*.jpg;*.jpeg;*.dpx;*.webp;*.tif;*.tiff)|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.png;*.bmp;*.jpg;*.jpeg;*.dpx;*.webp;*.tif;*.tiff|All files (*.*)|*.*'
try {{
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {{
        [Console]::Out.Write(($dialog.FileNames | ConvertTo-Json -Compress))
    }}
}} finally {{
    $owner.Dispose()
}}
"""
    if state.APP is not None:
        state.APP.log.append(f"Opening Windows source browse dialog: {initial_dir}")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Windows file dialog failed.").strip())
    if not result.stdout.strip():
        if state.APP is not None:
            state.APP.log.append("Browse cancelled.")
        return []
    decoded = json.loads(result.stdout)
    paths = decoded if isinstance(decoded, list) else [decoded]
    selected = [rel(Path(path)) for path in paths]
    if state.APP is not None:
        state.APP.log.append(f"Browse selected {len(selected)} source item(s).")
    return selected

def browse_path_macos(kind: str, initial: Path) -> str:
    initial_dir = initial if initial.is_dir() else initial.parent
    initial_script = applescript_quote(str(initial_dir))
    if kind == "folder":
        script = f'set chosen to choose folder with prompt "Choose folder" default location POSIX file {initial_script}\nPOSIX path of chosen'
    elif kind in {"save", "save_image", "project_save"}:
        default_name = applescript_quote("" if initial.is_dir() else initial.name)
        script = f'set chosen to choose file name with prompt "Choose output path" default location POSIX file {initial_script} default name {default_name}\nPOSIX path of chosen'
    else:
        script = f'set chosen to choose file with prompt "Choose file" default location POSIX file {initial_script}\nPOSIX path of chosen'
    result = subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "User canceled" in stderr or "(-128)" in stderr:
            state.APP.log.append("Browse cancelled.")
            return ""
        raise RuntimeError(stderr or "macOS file dialog failed.")
    selected = result.stdout.strip()
    if selected:
        state.APP.log.append(f"Browse selected: {selected}")
    else:
        state.APP.log.append("Browse cancelled.")
    return rel(Path(selected)) if selected else ""


def browse_source_paths_macos(initial: Path) -> list[str]:
    initial_dir = initial if initial.is_dir() else initial.parent
    initial_script = applescript_quote(str(initial_dir))
    script = f'''set chosen to choose file with prompt "Choose a video or image sequence" default location POSIX file {initial_script} with multiple selections allowed
set output to ""
repeat with itemPath in chosen
    set output to output & POSIX path of itemPath & linefeed
end repeat
output'''
    result = subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "User canceled" in stderr or "(-128)" in stderr:
            if state.APP is not None:
                state.APP.log.append("Browse cancelled.")
            return []
        raise RuntimeError(stderr or "macOS file dialog failed.")
    selected = [rel(Path(line.strip())) for line in result.stdout.splitlines() if line.strip()]
    if selected and state.APP is not None:
        state.APP.log.append(f"Browse selected {len(selected)} source item(s).")
    return selected

def applescript_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

def browse_path_linux(kind: str, initial: Path) -> str:
    if shutil.which("zenity"):
        return browse_path_zenity(kind, initial)
    if shutil.which("kdialog"):
        return browse_path_kdialog(kind, initial)
    raise RuntimeError("No native file picker found. Install zenity or kdialog, or paste the path into the field.")


def browse_source_paths_linux(initial: Path) -> list[str]:
    initial_dir = initial if initial.is_dir() else initial.parent
    if shutil.which("zenity"):
        command = [
            "zenity", "--file-selection", "--multiple", "--separator=\n",
            f"--filename={str(initial_dir) + os.sep}",
            "--file-filter=Video and images | *.mp4 *.mov *.mkv *.avi *.webm *.m4v *.png *.bmp *.jpg *.jpeg *.dpx *.webp *.tif *.tiff",
        ]
    elif shutil.which("kdialog"):
        command = [
            "kdialog", "--getopenfilename", str(initial_dir),
            "Video and images (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.png *.bmp *.jpg *.jpeg *.dpx *.webp *.tif *.tiff)",
            "--multiple", "--separate-output",
        ]
    else:
        raise RuntimeError("No native file picker found. Install zenity or kdialog, or paste a video path into the field.")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if state.APP is not None:
            state.APP.log.append("Browse cancelled.")
        return []
    selected = [rel(Path(line.strip())) for line in result.stdout.splitlines() if line.strip()]
    if selected and state.APP is not None:
        state.APP.log.append(f"Browse selected {len(selected)} source item(s).")
    return selected

def browse_path_zenity(kind: str, initial: Path) -> str:
    save_kind = kind in {"save", "save_image", "project_save"}
    filename = str(initial if save_kind and not initial.is_dir() else initial) + ("" if save_kind and not initial.is_dir() else os.sep)
    command = ["zenity", "--file-selection", f"--filename={filename}"]
    if kind == "folder":
        command.append("--directory")
    elif kind in {"save", "save_image", "project_save"}:
        command.append("--save")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        state.APP.log.append("Browse cancelled.")
        return ""
    selected = result.stdout.strip()
    if selected:
        state.APP.log.append(f"Browse selected: {selected}")
    return rel(Path(selected)) if selected else ""

def browse_path_kdialog(kind: str, initial: Path) -> str:
    if kind == "folder":
        command = ["kdialog", "--getexistingdirectory", str(initial)]
    elif kind in {"save", "save_image", "project_save"}:
        command = ["kdialog", "--getsavefilename", str(initial)]
    else:
        command = ["kdialog", "--getopenfilename", str(initial)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        state.APP.log.append("Browse cancelled.")
        return ""
    selected = result.stdout.strip()
    if selected:
        state.APP.log.append(f"Browse selected: {selected}")
    return rel(Path(selected)) if selected else ""
