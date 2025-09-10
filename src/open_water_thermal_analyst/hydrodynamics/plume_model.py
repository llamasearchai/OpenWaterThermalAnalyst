from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def simulate_plume(
    T0: np.ndarray,
    u: float,
    v: float,
    kappa: float,
    dx: float,
    dy: float,
    dt: float,
    steps: int,
    source: Optional[Tuple[int, int, float]] = None,
) -> np.ndarray:
    """Explicit 2D advection-diffusion of temperature anomaly.

    T0: initial temperature field (Celsius or Kelvin consistently)
    u, v: mean flow velocities (m/s)
    kappa: thermal diffusivity (m^2/s)
    dx, dy: grid spacing (m)
    dt: time step (s) - must satisfy stability (CFL and diffusion constraints)
    steps: number of time steps
    source: optional (iy, ix, S) adding S each step at a grid cell
    """
    T = T0.astype(float).copy()
    ny, nx = T.shape

    # Stability checks (not enforcing but helpful for users)
    cfl_x = abs(u) * dt / max(dx, 1e-9)
    cfl_y = abs(v) * dt / max(dy, 1e-9)
    diff = 2 * kappa * dt * (1 / max(dx, 1e-9) ** 2 + 1 / max(dy, 1e-9) ** 2)
    if cfl_x + cfl_y + diff > 1.0:
        # Conservative stabilization by reducing effective dt contribution
        scale = max(1.0, cfl_x + cfl_y + diff)
        dt_eff = dt / scale
    else:
        dt_eff = dt

    for _ in range(steps):
        Tn = T.copy()
        # Second-order central differences for diffusion
        T_xx = (np.roll(Tn, -1, axis=1) - 2 * Tn + np.roll(Tn, 1, axis=1)) / (dx * dx)
        T_yy = (np.roll(Tn, -1, axis=0) - 2 * Tn + np.roll(Tn, 1, axis=0)) / (dy * dy)
        # Upwind for advection
        T_x = (Tn - np.roll(Tn, 1, axis=1)) / dx if u >= 0 else (np.roll(Tn, -1, axis=1) - Tn) / dx
        T_y = (Tn - np.roll(Tn, 1, axis=0)) / dy if v >= 0 else (np.roll(Tn, -1, axis=0) - Tn) / dy

        T += dt_eff * (-u * T_x - v * T_y + kappa * (T_xx + T_yy))

        if source is not None:
            iy, ix, S = source
            if 0 <= iy < ny and 0 <= ix < nx:
                T[iy, ix] += S

    return T

