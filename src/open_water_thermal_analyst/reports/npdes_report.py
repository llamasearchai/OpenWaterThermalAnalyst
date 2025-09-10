from __future__ import annotations

from typing import Dict, Any
from datetime import datetime, timezone


def generate_markdown_report(results: Dict[str, Any], output_path: str) -> str:
    """Generate a Markdown report summarizing compliance results.

    results should contain keys like 'stats', 'thresholds', 'warning_exceedances', 'critical_exceedances'.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stats = results.get("stats", {})
    thresholds = results.get("thresholds", {})

    md = [
        f"# NPDES Thermal Discharge Compliance Report",
        "",
        f"Generated: {now}",
        "",
        "## Summary Statistics",
        f"- Min Temperature: {stats.get('min', 'n/a'):.3f} °C" if isinstance(stats.get('min'), (int, float)) else f"- Min Temperature: {stats.get('min', 'n/a')}",
        f"- Mean Temperature: {stats.get('mean', 'n/a'):.3f} °C" if isinstance(stats.get('mean'), (int, float)) else f"- Mean Temperature: {stats.get('mean', 'n/a')} ",
        f"- Max Temperature: {stats.get('max', 'n/a'):.3f} °C" if isinstance(stats.get('max'), (int, float)) else f"- Max Temperature: {stats.get('max', 'n/a')} ",
        f"- Count: {stats.get('count', 0)}",
        "",
        "## Thresholds",
        f"- Warning Threshold: {thresholds.get('warning_celsius', 'n/a')} °C",
        f"- Critical Threshold: {thresholds.get('critical_celsius', 'n/a')} °C",
        "",
        "## Exceedances",
        f"- Warning Exceedances: {results.get('warning_exceedances', 0)}",
        f"- Critical Exceedances: {results.get('critical_exceedances', 0)}",
        "",
        "## Notes",
        "These results are produced by OpenWaterThermalAnalyst and are intended for regulatory compliance documentation.",
    ]

    content = "\n".join(md) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path

