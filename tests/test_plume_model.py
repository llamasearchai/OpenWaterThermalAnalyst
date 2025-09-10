import numpy as np
import pytest

from open_water_thermal_analyst.hydrodynamics.plume_model import simulate_plume


class TestPlumeSimulation:
    """Test hydrodynamic plume simulation functions."""

    def test_simulate_plume_basic(self):
        """Test basic plume simulation with simple initial conditions."""
        ny, nx = 10, 10
        T0 = np.zeros((ny, nx), dtype=float)
        T0[5, 5] = 10.0  # Initial hotspot

        u, v = 0.1, 0.0  # Flow velocity (eastward)
        kappa = 1e-4  # Thermal diffusivity
        dx = dy = 10.0  # Grid spacing
        dt = 1.0  # Time step
        steps = 50

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # Check basic properties
        assert T_final.shape == (ny, nx)
        assert np.isfinite(T_final).all()
        assert T_final.max() < T0.max()  # Should decrease due to diffusion
        assert T_final.min() >= 0.0  # Should not go negative

        # Check mass conservation (approximately)
        initial_mass = np.sum(T0)
        final_mass = np.sum(T_final)
        assert abs(final_mass - initial_mass) / initial_mass < 0.1  # Within 10%

    def test_simulate_plume_no_flow(self):
        """Test plume simulation with no flow (pure diffusion)."""
        ny, nx = 8, 8
        T0 = np.zeros((ny, nx), dtype=float)
        T0[4, 4] = 20.0

        u, v = 0.0, 0.0  # No flow
        kappa = 1e-3  # Higher diffusivity for faster diffusion
        dx = dy = 5.0
        dt = 0.5
        steps = 20

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # With diffusion only, temperature should spread symmetrically
        assert T_final.shape == (ny, nx)
        assert T_final[4, 4] < T0[4, 4]  # Center should decrease

        # Check symmetry (approximately)
        assert abs(T_final[4, 3] - T_final[4, 5]) < 0.1  # Left-right symmetry
        assert abs(T_final[3, 4] - T_final[5, 4]) < 0.1  # Up-down symmetry

    def test_simulate_plume_with_source(self):
        """Test plume simulation with continuous source term."""
        ny, nx = 6, 6
        T0 = np.zeros((ny, nx), dtype=float)

        u, v = 0.0, 0.0
        kappa = 1e-4
        dx = dy = 10.0
        dt = 1.0
        steps = 10
        source = (3, 3, 1.0)  # Add 1.0°C per step at center

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps, source=source)

        # Source should increase temperature at injection point
        assert T_final[3, 3] > T0[3, 3]
        # Allow small numerical tolerance due to diffusion
        expected_temp = steps * source[2]
        assert abs(T_final[3, 3] - expected_temp) < 0.01  # Within 1% of expected

    def test_simulate_plume_boundary_conditions(self):
        """Test that plume simulation handles boundaries correctly."""
        ny, nx = 5, 5
        T0 = np.ones((ny, nx), dtype=float) * 25.0  # Uniform temperature

        u, v = 0.1, 0.0
        kappa = 1e-4
        dx = dy = 10.0
        dt = 1.0
        steps = 10

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # Check that boundaries are handled (no NaN or extreme values)
        assert np.isfinite(T_final).all()
        assert T_final.min() >= 0.0
        assert T_final.max() <= 30.0  # Should not increase significantly

    def test_simulate_plume_stability(self):
        """Test numerical stability of plume simulation."""
        ny, nx = 10, 10
        T0 = np.random.rand(ny, nx) * 10.0  # Random initial conditions

        u, v = 0.05, 0.05
        kappa = 1e-4
        dx = dy = 5.0
        dt = 0.1  # Small time step for stability
        steps = 100

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # Check stability
        assert np.isfinite(T_final).all()
        assert not np.any(np.isnan(T_final))
        assert not np.any(np.isinf(T_final))

        # Check that solution remains bounded
        assert T_final.min() >= -1.0  # Small negative values possible due to numerics
        assert T_final.max() <= 20.0

    def test_simulate_plume_advection_dominance(self):
        """Test plume simulation with advection-dominated transport."""
        ny, nx = 12, 12
        T0 = np.zeros((ny, nx), dtype=float)
        T0[6, 2] = 10.0  # Initial condition near left boundary

        u, v = 0.5, 0.0  # Strong eastward flow
        kappa = 1e-6  # Very low diffusivity
        dx = dy = 5.0
        dt = 0.5
        steps = 20

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # With strong advection, plume should move eastward
        # Center of mass should shift to the right
        y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        total_mass = np.sum(T_final)
        center_x = np.sum(x_coords * T_final) / total_mass
        center_y = np.sum(y_coords * T_final) / total_mass

        assert center_x > 2.0  # Should move right from initial x=2
        assert abs(center_y - 6.0) < 1.0  # Should stay near y=6

    def test_simulate_plume_large_time_steps(self):
        """Test plume simulation behavior with larger time steps."""
        ny, nx = 8, 8
        T0 = np.zeros((ny, nx), dtype=float)
        T0[4, 4] = 5.0

        u, v = 0.1, 0.1
        kappa = 1e-4
        dx = dy = 10.0
        dt = 5.0  # Larger time step
        steps = 10

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # Should still produce reasonable results
        assert np.isfinite(T_final).all()
        assert T_final.max() > 0.0
        assert T_final.min() >= -0.1  # Allow small negative values

    def test_simulate_plume_zero_initial_condition(self):
        """Test plume simulation with zero initial condition."""
        ny, nx = 6, 6
        T0 = np.zeros((ny, nx), dtype=float)

        u, v = 0.0, 0.0
        kappa = 1e-4
        dx = dy = 10.0
        dt = 1.0
        steps = 10

        T_final = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps)

        # Should remain zero (or very close to zero)
        assert np.allclose(T_final, 0.0, atol=1e-10)
