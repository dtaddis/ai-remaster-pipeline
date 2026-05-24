from __future__ import annotations

# Compatibility wrapper: older launchers and tests import ai_remaster_gui.app.
# The implementation lives in server.py so the app boundary is easier to see.
from .server import *  # noqa: F401,F403
