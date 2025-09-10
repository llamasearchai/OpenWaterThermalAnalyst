from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
import requests


def send_alert(level: str, message: str, context: Optional[Dict[str, Any]] = None, webhook_url: Optional[str] = None) -> None:
    """Send alert to a webhook if available, otherwise log to stdout.

    Level: INFO, WARNING, CRITICAL
    """
    payload = {
        "level": level.upper(),
        "message": message,
        "context": context or {},
    }
    url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
    if url:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:  # Fallback to stdout if webhook fails
            print(json.dumps({"alert": payload, "error": str(e)}))
    else:
        print(json.dumps({"alert": payload}))

