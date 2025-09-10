from __future__ import annotations

from typing import Dict
import math


def estimate_correction_params(relative_humidity: float, aerosol_optical_depth: float, precip_water_cm: float) -> Dict[str, float]:
    """Estimate simple atmospheric correction parameters for TIR.

    Returns a dictionary with approximated values:
    - tau: atmospheric transmittance (0..1)
    - l_up: upwelling path radiance surrogate (W/m^2/sr/um)
    - l_down: downwelling sky radiance surrogate (W/m^2/sr/um)

    This is a simplified parameterization intended to distinguish natural variability
    and provide robustness; for regulatory workflows consider using MODTRAN/6S-based
    coefficients if available.
    """
    rh = max(0.0, min(100.0, relative_humidity)) / 100.0
    aod = max(0.0, aerosol_optical_depth)
    w = max(0.0, precip_water_cm)

    # Heuristic transmittance model: drier air => higher tau, more aerosols => lower tau
    tau = max(0.3, min(1.0, 0.95 - 0.2 * rh - 0.15 * aod - 0.05 * w))

    # Radiance surrogates increase with humidity and water vapor
    l_up = 0.2 + 0.8 * rh + 0.3 * w
    l_down = 0.3 + 1.0 * rh + 0.4 * w

    return {"tau": tau, "l_up": l_up, "l_down": l_down}

