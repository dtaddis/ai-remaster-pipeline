from __future__ import annotations

import re
import shutil
import sys
import threading
from typing import TextIO


PERCENT_RE = re.compile(r"(?<!\d)(100|\d{1,2})(?:\.\d+)?\s*%")
COUNT_RE = re.compile(
    r"\b(?:chunk|segment|shot|frame|cue|window)\s+(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
FFMPEG_NOISE_RE = re.compile(
    r"^\s*(?:ffmpeg version|built with |configuration:|libav\w+\s|libsw\w+\s|"
    r"metadata:|major_brand\s*:|minor_version\s*:|compatible_brands\s*:|encoder\s*:|"
    r"handler_name\s*:|stream mapping:|press \[q\]|side data:|cpb properties:|"
    r"input #\d|output #\d|duration:|stream #\d|frame=|video:|audio:|"
    r"\[(?:libx26[45]|aac|out#\d|mp4)\b)",
    re.IGNORECASE,
)


class ConsoleLog(list[str]):
    """In-memory run history which also writes concise live output to ARP's console.

    The list behaviour is retained for progress estimation, error notifications, and tests. Lines
    which already describe progress are rendered in-place on an interactive terminal instead of
    producing hundreds of near-identical console rows.
    """

    def __init__(self, values=(), stream: TextIO | None = None) -> None:
        super().__init__(values)
        self.stream = stream or sys.stdout
        self._lock = threading.RLock()
        self._progress_active = False
        self._progress_signature: tuple[int, str] | None = None
        self._progress_width = 0

    def append(self, value: str) -> None:
        text = str(value)
        super().append(text)
        self._write_message(text)

    def extend(self, values) -> None:
        for value in values:
            self.append(value)

    def progress(self, percent: int | float, label: str) -> None:
        value = max(0, min(100, int(round(float(percent)))))
        clean_label = " ".join(str(label or "Working").replace("\r", " ").replace("\n", " ").split())
        signature = (value, clean_label)
        with self._lock:
            if signature == self._progress_signature:
                return
            self._progress_signature = signature
            if not self._interactive():
                # Redirected output should remain line-oriented and is normally used for tests or
                # diagnostics. Only explicit 0/100 updates are useful there.
                if value in {0, 100}:
                    print(f"{clean_label}: {value}%", file=self.stream, flush=True)
                return
            columns = max(50, shutil.get_terminal_size(fallback=(100, 24)).columns)
            bar_width = max(12, min(32, columns - 58))
            filled = int(round(bar_width * value / 100))
            bar = "#" * filled + "-" * (bar_width - filled)
            available = max(12, columns - bar_width - 10)
            rendered = f"[{bar}] {value:3d}% {clean_label[:available]}"
            padding = " " * max(0, self._progress_width - len(rendered))
            self.stream.write("\r" + rendered + padding)
            self.stream.flush()
            self._progress_width = len(rendered)
            self._progress_active = True

    def finish_progress(self) -> None:
        with self._lock:
            if self._interactive() and self._progress_active:
                self.stream.write("\n")
                self.stream.flush()
            self._progress_active = False
            self._progress_signature = None
            self._progress_width = 0

    def _write_message(self, text: str) -> None:
        # Tqdm and a few native tools place several carriage-return updates in one captured line.
        # The final update is the useful one.
        clean = text.replace("\x1b[K", "").split("\r")[-1].rstrip()
        if "\n" in clean:
            for line in clean.splitlines():
                self._write_message(line)
            return
        if not clean:
            return
        percent = PERCENT_RE.search(clean)
        count = COUNT_RE.search(clean)
        progress_words = ("progress", "%|", "frames:", "install", "download")
        if percent and any(word in clean.lower() for word in progress_words):
            self.progress(float(percent.group(1)), clean)
            return
        if count:
            current, total = int(count.group(1)), int(count.group(2))
            if total > 0:
                self.progress(current * 100 / total, clean)
                return
        # Native FFmpeg banners, codec capability dumps, and per-encoder statistics obscure the
        # useful ARP messages. They remain in this list for diagnostics/error excerpts; only their
        # console rendering is suppressed. Warnings and errors always pass through.
        lower = clean.lower()
        if FFMPEG_NOISE_RE.match(clean) and not any(word in lower for word in ("error", "warning", "failed")):
            return
        with self._lock:
            self.finish_progress()
            print(clean, file=self.stream, flush=True)

    def _interactive(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, OSError):
            return False
