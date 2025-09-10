from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any
import numpy as np

from .config import ComplianceThresholds


def evaluate_compliance(temp_c: np.ndarray, thresholds: ComplianceThresholds) -> Dict[str, Any]:
    """Compute statistics and exceedances against thresholds.

    Returns a dictionary containing summary statistics, exceedance masks, and counts.
    """
    valid = np.isfinite(temp_c)
    vals = temp_c[valid]
    stats = {
        "min": float(np.min(vals)) if vals.size else float("nan"),
        "mean": float(np.mean(vals)) if vals.size else float("nan"),
        "max": float(np.max(vals)) if vals.size else float("nan"),
        "count": int(vals.size),
    }
    warn_mask = (temp_c >= thresholds.warning_celsius) & valid
    crit_mask = (temp_c >= thresholds.critical_celsius) & valid
    return {
        "stats": stats,
        "thresholds": asdict(thresholds),
        "warning_exceedances": int(np.count_nonzero(warn_mask)),
        "critical_exceedances": int(np.count_nonzero(crit_mask)),
        "warning_mask": warn_mask,
        "critical_mask": crit_mask,
    }

