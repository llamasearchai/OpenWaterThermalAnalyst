from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import os
import yaml


@dataclass
class ComplianceThresholds:
    warning_celsius: float = 28.0
    critical_celsius: float = 32.0
    rolling_minutes: int = 60


@dataclass
class AppConfig:
    thresholds: ComplianceThresholds = field(default_factory=ComplianceThresholds)
    alert_webhook_url: Optional[str] = None
    scada_endpoint: Optional[str] = None


def _env_override(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.environ.get(key, "")
    return value


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML if provided, otherwise return defaults.

    Supports environment-variable interpolation with ${VAR_NAME} strings.
    """
    if path is None or not os.path.exists(path):
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Apply simple env interpolation
    def interpolate(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: interpolate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [interpolate(v) for v in obj]
        return _env_override(obj)

    data = interpolate(raw)

    thresholds = data.get("thresholds", {})
    return AppConfig(
        thresholds=ComplianceThresholds(
            warning_celsius=float(thresholds.get("warning_celsius", 28.0)),
            critical_celsius=float(thresholds.get("critical_celsius", 32.0)),
            rolling_minutes=int(thresholds.get("rolling_minutes", 60)),
        ),
        alert_webhook_url=data.get("alert_webhook_url"),
        scada_endpoint=data.get("scada_endpoint"),
    )

