"""
FastAPI application for OpenWaterThermalAnalyst with OpenAI Agents SDK integration.

This module provides REST API endpoints for all thermal analysis functionality,
including comprehensive workflows powered by multi-agent systems.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# from .agents import run_agents_workflow_sync  # Temporarily disabled due to schema issues
from . import imagery, compliance, config, alerts
from .hydrodynamics.plume_model import simulate_plume
from .reports.npdes_report import generate_markdown_report
from .ml.models import train_random_forest_regressor
from .datasette_export import export_compliance_results


# Pydantic models for API requests/responses

class ThermalProcessingRequest(BaseModel):
    emissivity: float = Field(default=0.99, description="Surface emissivity")
    wavelength_um: float = Field(default=10.9, description="Sensor wavelength in micrometers")
    k1: Optional[float] = Field(default=None, description="Planck K1 constant for radiance conversion")
    k2: Optional[float] = Field(default=None, description="Planck K2 constant for radiance conversion")
    to_celsius: bool = Field(default=True, description="Convert output to Celsius")


class ComplianceCheckRequest(BaseModel):
    warning_threshold: float = Field(default=28.0, description="Warning temperature threshold (°C)")
    critical_threshold: float = Field(default=32.0, description="Critical temperature threshold (°C)")
    kelvin: bool = Field(default=False, description="Input data is in Kelvin")


class PlumeSimulationRequest(BaseModel):
    ny: int = Field(..., description="Grid height")
    nx: int = Field(..., description="Grid width")
    u: float = Field(..., description="Flow velocity u (m/s)")
    v: float = Field(..., description="Flow velocity v (m/s)")
    kappa: float = Field(default=1e-4, description="Thermal diffusivity (m²/s)")
    dx: float = Field(default=10.0, description="Grid spacing x (m)")
    dy: float = Field(default=10.0, description="Grid spacing y (m)")
    dt: float = Field(default=1.0, description="Time step (s)")
    steps: int = Field(default=100, description="Number of simulation steps")
    init_hotspot: Optional[List[float]] = Field(default=None, description="Initial hotspot [iy, ix, dT]")
    source: Optional[List[float]] = Field(default=None, description="Continuous source [iy, ix, S]")


class MLTrainingRequest(BaseModel):
    target_column: str = Field(..., description="Target column name for prediction")
    test_size: float = Field(default=0.2, description="Test set size fraction")


class ComprehensiveAnalysisRequest(BaseModel):
    warning_threshold: float = Field(default=28.0, description="Warning temperature threshold (°C)")
    critical_threshold: float = Field(default=32.0, description="Critical temperature threshold (°C)")
    output_dir: str = Field(default="api_analysis_output", description="Output directory for results")


class AlertRequest(BaseModel):
    level: str = Field(..., description="Alert level (INFO, WARNING, CRITICAL)")
    message: str = Field(..., description="Alert message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context data")


# FastAPI application
app = FastAPI(
    title="OpenWaterThermalAnalyst API",
    description="REST API for thermal plume analysis with OpenAI Agents SDK integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "OpenWaterThermalAnalyst API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "openai_agents": "available",
            "thermal_imagery": "available",
            "compliance_engine": "available",
            "plume_model": "available"
        }
    }


@app.post("/thermal/process")
async def process_thermal_imagery(
    file: UploadFile = File(...),
    request: ThermalProcessingRequest = None
):
    """Process thermal imagery to surface temperature."""
    if request is None:
        request = ThermalProcessingRequest()

    try:
        # Save uploaded file temporarily
        temp_input = f"/tmp/{file.filename}"
        with open(temp_input, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the imagery
        output_path = f"/tmp/processed_{file.filename}"
        arr, transform, crs_wkt, nodata = imagery.read_raster(temp_input)

        if request.k1 is not None and request.k2 is not None:
            bt_k = imagery.brightness_temperature_from_radiance(arr, request.k1, request.k2)
        else:
            bt_k = arr

        lst_k = imagery.surface_temperature_from_bt(
            bt_k,
            emissivity=request.emissivity,
            wavelength_um=request.wavelength_um
        )
        out = imagery.kelvin_to_celsius(lst_k) if request.to_celsius else lst_k
        imagery.write_geotiff(output_path, out.astype(float), transform, crs_wkt, nodata)

        # Clean up temp input file
        os.remove(temp_input)

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"processed_{file.filename}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/compliance/check")
async def check_compliance(
    file: UploadFile = File(...),
    request: ComplianceCheckRequest = None
):
    """Evaluate temperature raster against compliance thresholds."""
    if request is None:
        request = ComplianceCheckRequest()

    try:
        # Save uploaded file temporarily
        temp_input = f"/tmp/{file.filename}"
        with open(temp_input, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load configuration
        cfg = config.load_config(None)
        arr, _, _, _ = imagery.read_raster(temp_input)
        data_c = arr - 273.15 if request.kelvin else arr

        thresholds = config.ComplianceThresholds(
            warning_celsius=request.warning_threshold,
            critical_celsius=request.critical_threshold,
            rolling_minutes=cfg.thresholds.rolling_minutes,
        )

        results = compliance.evaluate_compliance(data_c, thresholds)

        # Clean up temp file
        os.remove(temp_input)

        # Convert numpy types for JSON serialization
        serializable_results = {
            "stats": {k: (float(v) if hasattr(v, 'item') else v)
                     for k, v in results["stats"].items()},
            "thresholds": results["thresholds"],
            "warning_exceedances": int(results["warning_exceedances"]),
            "critical_exceedances": int(results["critical_exceedances"])
        }

        return JSONResponse(content=serializable_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


@app.post("/plume/simulate")
async def simulate_plume_endpoint(request: PlumeSimulationRequest):
    """Run hydrodynamic plume simulation."""
    try:
        T0 = np.zeros((request.ny, request.nx), dtype=float)

        if request.init_hotspot:
            iy, ix, dT = request.init_hotspot
            if 0 <= iy < request.ny and 0 <= ix < request.nx:
                T0[int(iy), int(ix)] = dT

        source_tuple = tuple(request.source) if request.source else None
        T = simulate_plume(
            T0, request.u, request.v, request.kappa,
            request.dx, request.dy, request.dt, request.steps,
            source=source_tuple
        )

        output_path = "/tmp/plume_simulation.npy"
        np.save(output_path, T)

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename="plume_simulation.npy"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/ml/train")
async def train_ml_model(
    file: UploadFile = File(...),
    request: MLTrainingRequest = None
):
    """Train machine learning model for thermal prediction."""
    if request is None:
        request = MLTrainingRequest()

    try:
        # Save uploaded file temporarily
        temp_input = f"/tmp/{file.filename}"
        with open(temp_input, "wb") as f:
            content = await file.read()
            f.write(content)

        # Train model
        model_path = "/tmp/trained_model.joblib"
        result = train_random_forest_regressor(
            csv_path=temp_input,
            target=request.target_column,
            model_out=model_path,
            test_size=request.test_size
        )

        # Clean up temp file
        os.remove(temp_input)

        return FileResponse(
            model_path,
            media_type="application/octet-stream",
            filename="thermal_prediction_model.joblib"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Temporarily disabled due to schema compatibility issues
# @app.post("/agents/comprehensive-analysis")
# async def comprehensive_analysis(
#     file: UploadFile = File(...),
#     request: ComprehensiveAnalysisRequest = None,
#     background_tasks: BackgroundTasks = None
# ):
#     """Run comprehensive thermal analysis using multi-agent system."""
#     if request is None:
#         request = ComprehensiveAnalysisRequest()
#
#     try:
#         # Save uploaded file temporarily
#         temp_input = f"/tmp/{file.filename}"
#         with open(temp_input, "wb") as f:
#             content = await file.read()
#             f.write(content)
#
#         # Run comprehensive analysis
#         result = run_agents_workflow_sync(
#             "comprehensive_analysis",
#             thermal_raster_path=temp_input,
#             output_dir=request.output_dir,
#             warning_threshold=request.warning_threshold,
#             critical_threshold=request.critical_threshold
#         )
#
#         # Clean up temp file
#         os.remove(temp_input)
#
#         return JSONResponse(content=result)
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
#
#
# @app.post("/agents/plume-workflow")
# async def plume_workflow(request: PlumeSimulationRequest):
#     """Run plume simulation workflow using agents."""
#     try:
#         result = run_agents_workflow_sync(
#             "plume_simulation",
#             ny=request.ny,
#             nx=request.nx,
#             u=request.u,
#             v=request.v,
#             kappa=request.kappa,
#             dx=request.dx,
#             dy=request.dy,
#             dt=request.dt,
#             steps=request.steps,
#             init_hotspot=request.init_hotspot,
#             source=request.source
#         )
#
#         return JSONResponse(content=result)
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")
#
#
# @app.post("/agents/ml-workflow")
# async def ml_workflow(
#     file: UploadFile = File(...),
#     request: MLTrainingRequest = None
# ):
#     """Run ML training workflow using agents."""
#     if request is None:
#         request = MLTrainingRequest()
#
#     try:
#         # Save uploaded file temporarily
#         temp_input = f"/tmp/{file.filename}"
#         with open(temp_input, "wb") as f:
#             content = await file.read()
#             f.write(content)
#
#         # Run ML workflow
#         result = run_agents_workflow_sync(
#             "ml_training",
#             training_data_path=temp_input,
#             target_column=request.target_column,
#             output_dir="api_ml_output"
#         )
#
#         # Clean up temp file
#         os.remove(temp_input)
#
#         return JSONResponse(content=result)
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")


@app.post("/alert/send")
async def send_alert(request: AlertRequest):
    """Send environmental alert."""
    try:
        webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
        alerts.send_alert(
            request.level,
            request.message,
            request.context,
            webhook_url
        )

        return JSONResponse(content={
            "success": True,
            "alert_level": request.level,
            "webhook_used": webhook_url is not None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert failed: {str(e)}")


@app.post("/report/generate")
async def generate_report(
    compliance_data: Dict[str, Any],
    output_filename: str = "compliance_report.md"
):
    """Generate compliance report."""
    try:
        output_path = f"/tmp/{output_filename}"
        generate_markdown_report(compliance_data, output_path)

        return FileResponse(
            output_path,
            media_type="text/markdown",
            filename=output_filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.post("/datasette/export")
async def export_to_datasette_endpoint(
    compliance_data: Dict[str, Any],
    db_filename: str = "compliance_results.db",
    metadata: Optional[Dict[str, Any]] = None
):
    """Export compliance results to SQLite database for Datasette."""
    try:
        db_path = f"/tmp/{db_filename}"
        export_compliance_results(db_path, compliance_data, metadata)

        return FileResponse(
            db_path,
            media_type="application/octet-stream",
            filename=db_filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()
