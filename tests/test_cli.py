import sys
import subprocess


def _run_module_help() -> str:
    """Run the CLI module help and return its output."""
    r = subprocess.run([sys.executable, "-m", "open_water_thermal_analyst.cli", "-h"], capture_output=True, text=True)
    assert r.returncode == 0
    return r.stdout


def test_cli_help_shows_program_name():
    out = _run_module_help()
    assert "OpenWaterThermalAnalyst" in out or "openwater" in out


def test_check_compliance_help():
    r = subprocess.run([sys.executable, "-m", "open_water_thermal_analyst.cli", "check-compliance", "-h"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Temperature raster" in r.stdout
