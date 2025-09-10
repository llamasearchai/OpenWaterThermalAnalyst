from __future__ import annotations

import pandas as pd
from typing import Dict, Any


def correlate_with_fish_kill_events(temp_series: pd.Series, fish_events_csv: str) -> Dict[str, Any]:
    """Correlate rolling mean temperature with fish kill events.

    fish_events_csv must contain columns: timestamp, events (count or severity)
    temp_series must be indexed by datetime and represent water temperature in Celsius.
    """
    df_events = pd.read_csv(fish_events_csv, parse_dates=["timestamp"])  # type: ignore
    df_events = df_events.set_index("timestamp").sort_index()

    temp = temp_series.sort_index().rolling("7D").mean().dropna()
    events = df_events["events"].resample("D").sum().rolling(7).mean().dropna()

    # Align indices
    joined = pd.concat([temp, events], axis=1, join="inner")
    joined.columns = ["temp_c", "events"]

    corr = joined.corr().iloc[0, 1]
    return {
        "pearson_corr": float(corr),
        "n_samples": int(len(joined)),
    }

