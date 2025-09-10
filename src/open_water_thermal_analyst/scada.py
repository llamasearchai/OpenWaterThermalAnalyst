from __future__ import annotations

import os
from typing import Dict, Any, Optional
import requests


def send_setpoint_adjustment(payload: Dict[str, Any], endpoint: Optional[str] = None) -> bool:
    """Send a control payload to a SCADA HTTP endpoint if available.

    If no endpoint is provided, attempts to use the SCADA_ENDPOINT env var. Returns True on success.
    """
    url = endpoint or os.environ.get("SCADA_ENDPOINT")
    if not url:
        # No endpoint configured; we consider this a no-op success in non-production contexts.
        return True
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception:
        return False

