from __future__ import annotations

from typing import Dict, Any, Optional
import requests


def fetch_weather(lat: float, lon: float, hourly: Optional[str] = None) -> Dict[str, Any]:
    """Fetch current/hourly weather from Open-Meteo (no API key required).

    hourly: comma-separated variables, defaults to relativehumidity_2m,temperature_2m,windspeed_10m,winddirection_10m
    """
    variables = (
        hourly
        or "relativehumidity_2m,temperature_2m,windspeed_10m,winddirection_10m,dewpoint_2m"
    )
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&hourly={variables}&timezone=UTC"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

