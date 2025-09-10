# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository overview
- OpenWaterThermalAnalyst integrates thermal infrared remote sensing, hydrodynamic plume simulation, regulatory compliance checks, optional ML prediction, and alerting to analyze thermal impacts in open water systems.

Common commands
- Prerequisites (macOS): GDAL is required by rasterio.
  ```bash
  brew install gdal
  ```
- Create a virtualenv and install (editable) with optional extras:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  python -m pip install -U pip
  pip install -e .           # core
  pip install -e .[ml]       # optional ML (PyTorch etc.)
  pip install -e .[hydro]    # optional hydro tools
  ```
- CLI usage (installed entry point is `openwater`):
  ```bash
  openwater --version
  openwater -h
  openwater process-imagery -h
  openwater check-compliance -h
  openwater model-plume -h
  ```
- Dev tooling (uv, Hatch, tox):
  ```bash
  # Install with uv and include dev/ai/viz extras
  uv pip install -e .[dev,ai,viz]

  # Run tests with tox
  tox -q

  # Hatch convenience scripts
  hatch run dev:cli
  hatch run test:unit
  hatch run dev:build
  ```
- Quickstart (examples from README):
  ```bash
  # 1) Process thermal imagery to surface temperature (Celsius)
  openwater process-imagery input.tif output_c.tif \
    --emissivity 0.99 --wavelength-um 10.9 \
    --k1 774.8853 --k2 1321.0789 --to-celsius

  # 2) Check compliance, write a Markdown report + JSON summary,
  #    generate an AI narrative, and export to SQLite for Datasette
  openwater check-compliance output_c.tif \
    --warning 28 --critical 32 \
    --report-md report.md -o summary.json \
    --explain --export-sqlite outputs/compliance.sqlite

  # Explore results with Datasette (if installed)
  datasette serve outputs/compliance.sqlite

  # 3) Run a 2D advection–diffusion plume simulation and save .npy
  openwater model-plume 200 300 \
    --u 0.2 --v 0.05 --kappa 1e-4 \
    --dx 10 --dy 10 --dt 1 --steps 600 \
    --init-hotspot 100 50 5 \
    --output plume.npy
  ```
- Run from source (without installing the entry point):
  ```bash
  python -m open_water_thermal_analyst.cli -h
  ```
- Build distributable artifacts (wheel/sdist):
  ```bash
  python -m pip install build
  python -m build  # outputs dist/*.whl and dist/*.tar.gz
  ```
- Testing and linting:
  - Run smoke tests via tox (editable install + pytest):
    ```bash
    tox -q
    ```
  - Run tests via Hatch or directly:
    ```bash
    hatch run test:unit
    pytest -q
    ```
  - Run a single test with pytest:
    ```bash
    pytest tests/test_cli.py::test_check_compliance_help -q
    ```

Configuration and integrations
- Config file (optional): pass via --config to CLI; example at config/example.yaml. Supports ${VAR} environment interpolation.
- Environment variables used:
  - ALERT_WEBHOOK_URL: webhook receiving alerts.
  - SCADA_ENDPOINT: HTTP endpoint for SCADA setpoint adjustments.
  - OPENAI_API_KEY: used by AI narrative generation (openai SDK).
  - OPENAI_MODEL: optional model name used for AI summaries (default: gpt-4o-mini).

High-level architecture (big picture)
- Packaging/entry point
  - pyproject.toml uses setuptools with src/ layout and declares the console script `openwater` -> open_water_thermal_analyst.cli:main.
  - Version is defined in src/open_water_thermal_analyst/__init__.py.
- CLI orchestration (src/open_water_thermal_analyst/cli.py)
  - process-imagery: reads a single-band thermal raster, optionally converts radiance→brightness temperature (Planck K1/K2), applies single-channel emissivity correction to estimate surface temperature (Kelvin or Celsius), and writes a GeoTIFF.
  - check-compliance: loads thresholds from YAML (with env interpolation), reads temperature raster (°C by default or Kelvin with --kelvin), evaluates exceedances, optionally emits a Markdown NPDES report, and sends webhook alerts on exceedances.
  - model-plume: explicit 2D advection–diffusion on a grid with optional source/hotspot; saves results to .npy.
- Thermal imagery processing (src/open_water_thermal_analyst/imagery.py)
  - Raster IO via rasterio (GDAL required). Utilities for Planck inversion (brightness_temperature_from_radiance), single-channel LST (surface_temperature_from_bt), Kelvin↔Celsius conversion, and GeoTIFF writing.
- Compliance (src/open_water_thermal_analyst/compliance.py, config.py)
  - Dataclasses for thresholds and app config. evaluate_compliance returns stats, masks, and exceedance counts. Config loader supports ${VAR} substitution.
- Reporting and alerting (src/open_water_thermal_analyst/reports/npdes_report.py, alerts.py)
  - Markdown NPDES summary report generator. Webhook-based alert sender with stdout fallback on failure.
- AI narrative generation (src/open_water_thermal_analyst/ai.py)
  - Summarizes compliance results using OpenAI SDK or 'llm' CLI fallback; integrated via --explain/--explain-to flags.
- Results export for Datasette (src/open_water_thermal_analyst/datasette_export.py)
  - Persists compliance runs and stats to SQLite; explore with 'datasette serve path/to.db'.
- Hydrodynamics (src/open_water_thermal_analyst/hydrodynamics/plume_model.py)
  - Explicit scheme combining upwind advection and central-difference diffusion with a conservative effective dt scaling when stability limits are exceeded.
- Machine learning (src/open_water_thermal_analyst/ml/models.py)
  - RandomForest regressor training utility (joblib persistence). Optional LSTM model and trainer require the [ml] extra (PyTorch).
- External data hooks (src/open_water_thermal_analyst/weather.py, scada.py)
  - Weather via Open‑Meteo (no API key). SCADA setpoint POST helper honoring SCADA_ENDPOINT.

Repository conventions
- Do not use emojis, stubs, or placeholders in code. Ensure all code is complete.

