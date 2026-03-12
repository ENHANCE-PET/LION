#!/usr/bin/env python3
"""
LION Telemetry
--------------

Anonymous usage telemetry for LION. Helps us understand how the tool is used
and prioritize development.

Disable with: LIONZ_TELEMETRY=0

What we collect:
- LION version
- Model used (fdg, psma)
- Platform (linux, darwin, win32)
- Accelerator (cpu, cuda, mps)
- Number of subjects processed
- Success/failure

What we DON'T collect:
- File paths
- Patient data
- IP addresses (Supabase doesn't log by default)
- Any identifiable information
"""

import os
import sys
import threading
from urllib import request, error
import json

from lionz.constants import VERSION

SUPABASE_URL = "https://oxgwmxvezejtibfpglgn.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94Z3dteHZlemVqdGliZnBnbGduIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMzMTk1MzgsImV4cCI6MjA4ODg5NTUzOH0.q86BNdgG49j9tkUkn7ueQoR3LgwWdhbPinU4sEHnINg"


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled (opt-out via environment variable)."""
    env_value = os.environ.get("LIONZ_TELEMETRY", "1").lower()
    return env_value not in ("0", "false", "no", "off")


def send_telemetry(
    model: str,
    accelerator: str,
    n_subjects: int,
    success: bool,
) -> None:
    """
    Send anonymous telemetry data to Supabase.

    Runs in a background thread, fails silently, never blocks the main process.
    """
    if not is_telemetry_enabled():
        return

    def _send():
        try:
            data = {
                "version": VERSION,
                "model": model,
                "platform": sys.platform,
                "accelerator": accelerator or "unknown",
                "n_subjects": n_subjects,
                "success": success,
            }

            payload = json.dumps(data).encode("utf-8")

            req = request.Request(
                f"{SUPABASE_URL}/rest/v1/lionz_telemetry",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Prefer": "return=minimal",
                },
                method="POST",
            )

            with request.urlopen(req, timeout=5):
                pass

        except Exception:
            # Fail silently - telemetry should never break the tool
            pass

    # Run in background thread to not block
    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
