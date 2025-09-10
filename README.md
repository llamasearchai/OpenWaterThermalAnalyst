# OpenWaterThermalAnalyst

OpenWaterThermalAnalyst integrates thermal infrared remote sensing, hydrodynamic plume modeling, regulatory compliance checking, machine learning prediction, and real-time alerting to analyze and manage thermal impacts in open water systems.

## Features
- Thermal imagery processing (e.g., Landsat 8/9 TIRS, MODIS, ECOSTRESS) via rasterio/GDAL
- Emissivity-corrected surface temperature mapping (single-channel method)
- Atmospheric parameter estimation utilities
- 2D advection–diffusion plume simulation
- Automated regulatory compliance evaluation against temperature thresholds
- Machine learning (Random Forest; optional LSTM with PyTorch extra) for thermal impact prediction
- Real-time alerting via webhook
- Ecological correlation with fish kill events
- NPDES-ready Markdown report generation
- SCADA integration hook for operational adjustments

## Install
Python >= 3.10 is recommended.

Note: rasterio/GDAL may require system libraries. On macOS, install GDAL via Homebrew before Python deps:

- brew install gdal

Install the package (editable is convenient during development):

- pip install -e .

Optional extras:

- pip install -e .[ml]

## Quickstart

1) Process thermal imagery to surface temperature (Celsius):

- openwater process-imagery input.tif output_c.tif --emissivity 0.99 --wavelength-um 10.9 --k1 774.8853 --k2 1321.0789 --to-celsius

If your input is already brightness temperature in Kelvin, omit --k1/--k2.

2) Check compliance and generate a report:

- openwater check-compliance output_c.tif --warning 28 --critical 32 --report-md report.md -o summary.json

3) Run a plume simulation and save the resulting temperature field:

- openwater model-plume 200 300 --u 0.2 --v 0.05 --kappa 1e-4 --dx 10 --dy 10 --dt 1 --steps 600 --init-hotspot 100 50 5 --output plume.npy

## Configuration
You can supply a YAML config with threshold defaults and integration endpoints. See config/example.yaml.

Environment variables supported via ${VAR_NAME} syntax:
- ALERT_WEBHOOK_URL: webhook to receive alerts
- SCADA_ENDPOINT: HTTP endpoint for sending setpoint adjustments

## Machine Learning
- Random Forest training utility is available in the library API (open_water_thermal_analyst.ml.models.train_random_forest_regressor).
- LSTM model requires the optional 'ml' extra.

## Reporting
- NPDES Markdown reports can be generated from compliance results using the library API or via the CLI check-compliance --report-md flag.

## License
MIT

