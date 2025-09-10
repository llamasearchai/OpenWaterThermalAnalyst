"""
OpenAI Agents SDK integration for OpenWaterThermalAnalyst.

This module implements a multi-agent architecture for comprehensive thermal plume analysis
with specialized agents for different aspects of the workflow.
"""

from __future__ import annotations

import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import os

from agents import Agent, Runner, function_tool, handoff, trace
import numpy as np

from . import imagery, compliance, config, alerts
from .hydrodynamics.plume_model import simulate_plume
from .reports.npdes_report import generate_markdown_report
from .ml.models import train_random_forest_regressor
from .datasette_export import export_compliance_results


class ThermalAnalysisAgent(Agent):
    """Specialized agent for thermal imagery analysis and processing."""

    def __init__(self):
        super().__init__(
            name="Thermal Analysis Agent",
            instructions="""
            You are a thermal remote sensing specialist. Your role is to:
            1. Analyze thermal infrared imagery from satellites (Landsat, MODIS, ECOSTRESS)
            2. Convert radiance/brightness temperature to surface temperature
            3. Apply atmospheric corrections and emissivity adjustments
            4. Provide detailed analysis of thermal patterns and anomalies
            5. Explain thermal signatures and their environmental implications

            Always provide scientific justification for your analysis and recommendations.
            """,
            model="gpt-4o-mini"
        )


class ComplianceAgent(Agent):
    """Specialized agent for regulatory compliance evaluation."""

    def __init__(self):
        super().__init__(
            name="Compliance Agent",
            instructions="""
            You are an environmental compliance specialist focused on water quality regulations.
            Your role is to:
            1. Evaluate temperature data against NPDES and other regulatory thresholds
            2. Identify exceedances and potential violations
            3. Assess environmental impact significance
            4. Recommend mitigation strategies and monitoring protocols
            5. Generate compliance documentation and reports

            Always reference specific regulatory requirements and provide actionable recommendations.
            """,
            model="gpt-4o-mini"
        )


class PlumeModelingAgent(Agent):
    """Specialized agent for hydrodynamic plume modeling and simulation."""

    def __init__(self):
        super().__init__(
            name="Plume Modeling Agent",
            instructions="""
            You are a hydrodynamic modeling specialist for thermal plumes in water bodies.
            Your role is to:
            1. Design and execute 2D advection-diffusion plume simulations
            2. Interpret plume behavior and thermal transport patterns
            3. Assess mixing zones and dilution effects
            4. Predict plume evolution under different flow conditions
            5. Provide insights for operational decision-making

            Always explain the physics behind your modeling choices and validate assumptions.
            """,
            model="gpt-4o-mini"
        )


class AlertingAgent(Agent):
    """Specialized agent for real-time alerting and notification management."""

    def __init__(self):
        super().__init__(
            name="Alerting Agent",
            instructions="""
            You are an environmental monitoring alert specialist.
            Your role is to:
            1. Evaluate alert conditions based on thermal thresholds
            2. Determine appropriate alert levels (INFO, WARNING, CRITICAL)
            3. Craft clear, actionable alert messages for different stakeholders
            4. Coordinate multi-channel notification delivery
            5. Track alert response and follow-up actions

            Always prioritize critical environmental concerns and ensure timely communication.
            """,
            model="gpt-4o-mini"
        )


class ReportingAgent(Agent):
    """Specialized agent for generating comprehensive reports and documentation."""

    def __init__(self):
        super().__init__(
            name="Reporting Agent",
            instructions="""
            You are a technical report writer specializing in environmental assessments.
            Your role is to:
            1. Synthesize complex technical data into clear, comprehensive reports
            2. Generate NPDES-compliant documentation
            3. Create executive summaries for different stakeholder audiences
            4. Include appropriate visualizations and data exports
            5. Ensure reports meet regulatory and professional standards

            Always structure reports logically with clear findings, implications, and recommendations.
            """,
            model="gpt-4o-mini"
        )


class MLAgent(Agent):
    """Specialized agent for machine learning analysis and prediction."""

    def __init__(self):
        super().__init__(
            name="ML Analysis Agent",
            instructions="""
            You are a machine learning specialist for environmental data analysis.
            Your role is to:
            1. Train predictive models for thermal impact forecasting
            2. Analyze patterns in historical thermal data
            3. Generate predictions for future thermal conditions
            4. Validate model performance and uncertainty estimates
            5. Provide insights from ML analysis for decision-making

            Always explain model choices, performance metrics, and limitations clearly.
            """,
            model="gpt-4o-mini"
        )


# Tool functions for agents to use

@function_tool
def process_thermal_imagery(
    input_path: str,
    output_path: str,
    emissivity: float = 0.99,
    wavelength_um: float = 10.9,
    k1: Optional[float] = None,
    k2: Optional[float] = None,
    to_celsius: bool = True
):
    """Process thermal imagery to surface temperature."""
    try:
        arr, transform, crs_wkt, nodata = imagery.read_raster(input_path)

        if k1 is not None and k2 is not None:
            bt_k = imagery.brightness_temperature_from_radiance(arr, k1, k2)
        else:
            bt_k = arr  # assume brightness temperature is provided in Kelvin

        lst_k = imagery.surface_temperature_from_bt(bt_k, emissivity=emissivity, wavelength_um=wavelength_um)
        out = imagery.kelvin_to_celsius(lst_k) if to_celsius else lst_k
        imagery.write_geotiff(output_path, out.astype(np.float32), transform, crs_wkt, nodata)

        stats = {
            "min": float(np.min(out)),
            "max": float(np.max(out)),
            "mean": float(np.mean(out)),
            "std": float(np.std(out))
        }

        return json.dumps({
            "success": True,
            "output_path": output_path,
            "statistics": stats,
            "processing_parameters": {
                "emissivity": emissivity,
                "wavelength_um": wavelength_um,
                "to_celsius": to_celsius
            }
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def run_compliance_check(
    temp_raster_path: str,
    warning_threshold: float = 28.0,
    critical_threshold: float = 32.0,
    kelvin: bool = False
):
    """Evaluate temperature raster against compliance thresholds."""
    try:
        cfg = config.load_config(None)  # Load default config
        arr, _, _, _ = imagery.read_raster(temp_raster_path)
        data_c = arr - 273.15 if kelvin else arr

        thresholds = config.ComplianceThresholds(
            warning_celsius=warning_threshold,
            critical_celsius=critical_threshold,
            rolling_minutes=cfg.thresholds.rolling_minutes,
        )

        results = compliance.evaluate_compliance(data_c, thresholds)

        # Convert numpy types for JSON serialization
        serializable_results = {
            "stats": {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                     for k, v in results["stats"].items()},
            "thresholds": results["thresholds"],
            "warning_exceedances": int(results["warning_exceedances"]),
            "critical_exceedances": int(results["critical_exceedances"])
        }

        return json.dumps({
            "success": True,
            "results": serializable_results
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def simulate_thermal_plume(
    ny: int,
    nx: int,
    u: float,
    v: float,
    kappa: float = 1e-4,
    dx: float = 10.0,
    dy: float = 10.0,
    dt: float = 1.0,
    steps: int = 100,
    init_hotspot: Optional[List[float]] = None,
    source: Optional[List[float]] = None,
    output_path: str = "plume_simulation.npy"
):
    """Run hydrodynamic plume simulation."""
    try:
        T0 = np.zeros((ny, nx), dtype=float)

        if init_hotspot:
            iy, ix, dT = init_hotspot
            if 0 <= iy < ny and 0 <= ix < nx:
                T0[int(iy), int(ix)] = dT

        source_tuple = tuple(source) if source else None
        T = simulate_plume(T0, u, v, kappa, dx, dy, dt, steps, source=source_tuple)

        np.save(output_path, T)

        stats = {
            "final_min": float(np.min(T)),
            "final_max": float(np.max(T)),
            "final_mean": float(np.mean(T)),
            "simulation_steps": steps,
            "grid_size": f"{ny}x{nx}"
        }

        return json.dumps({
            "success": True,
            "output_path": output_path,
            "statistics": stats,
            "simulation_parameters": {
                "velocity": [u, v],
                "diffusivity": kappa,
                "grid_spacing": [dx, dy],
                "time_step": dt,
                "steps": steps
            }
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def generate_compliance_report(
    compliance_results: Dict[str, Any],
    output_path: str,
    include_ai_analysis: bool = True
):
    """Generate comprehensive compliance report."""
    try:
        generate_markdown_report(compliance_results, output_path)

        return json.dumps({
            "success": True,
            "report_path": output_path,
            "ai_analysis_included": include_ai_analysis
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def export_to_datasette(
    compliance_results: Dict[str, Any],
    db_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Export compliance results to SQLite database for Datasette."""
    try:
        export_compliance_results(db_path, compliance_results, metadata)

        return json.dumps({
            "success": True,
            "database_path": db_path,
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def train_ml_model(
    csv_path: str,
    target_column: str,
    model_output_path: str,
    test_size: float = 0.2
):
    """Train machine learning model for thermal prediction."""
    try:
        result = train_random_forest_regressor(
            csv_path=csv_path,
            target=target_column,
            model_out=model_output_path,
            test_size=test_size
        )

        return json.dumps({
            "success": True,
            "model_path": result.model_path,
            "performance": {
                "mae": result.mae,
                "training_samples": result.n_train,
                "test_samples": result.n_test
            }
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@function_tool
def send_environmental_alert(
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    webhook_url: Optional[str] = None
):
    """Send environmental alert notification."""
    try:
        url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
        alerts.send_alert(level, message, context, url)

        return json.dumps({
            "success": True,
            "alert_level": level,
            "webhook_used": url is not None
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


class OpenWaterThermalAnalystOrchestrator:
    """Main orchestrator for the multi-agent thermal analysis system."""

    def __init__(self):
        # Initialize specialized agents
        self.thermal_agent = ThermalAnalysisAgent()
        self.compliance_agent = ComplianceAgent()
        self.plume_agent = PlumeModelingAgent()
        self.alert_agent = AlertingAgent()
        self.report_agent = ReportingAgent()
        self.ml_agent = MLAgent()

        # Set up handoffs between agents
        self.thermal_agent.handoffs = [
            handoff(self.compliance_agent, "For regulatory compliance evaluation"),
            handoff(self.plume_agent, "For hydrodynamic plume modeling"),
            handoff(self.ml_agent, "For machine learning analysis")
        ]

        self.compliance_agent.handoffs = [
            handoff(self.alert_agent, "For sending alerts about compliance issues"),
            handoff(self.report_agent, "For generating compliance reports"),
            handoff(self.thermal_agent, "For additional thermal analysis")
        ]

        self.plume_agent.handoffs = [
            handoff(self.compliance_agent, "For compliance assessment of plume impacts"),
            handoff(self.report_agent, "For plume modeling reports")
        ]

        self.alert_agent.handoffs = [
            handoff(self.report_agent, "For documenting alert responses")
        ]

        self.ml_agent.handoffs = [
            handoff(self.compliance_agent, "For predictive compliance analysis"),
            handoff(self.report_agent, "For ML analysis reports")
        ]

        # Add function tools to agents
        self.thermal_agent.tools = [process_thermal_imagery]
        self.compliance_agent.tools = [run_compliance_check, generate_compliance_report]
        self.plume_agent.tools = [simulate_thermal_plume]
        self.alert_agent.tools = [send_environmental_alert]
        self.report_agent.tools = [generate_compliance_report, export_to_datasette]
        self.ml_agent.tools = [train_ml_model]

    async def run_comprehensive_analysis(
        self,
        thermal_raster_path: str,
        output_dir: str = "analysis_output",
        warning_threshold: float = 28.0,
        critical_threshold: float = 32.0
    ) -> Dict[str, Any]:
        """Run complete thermal analysis workflow using multi-agent system."""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Thermal imagery processing
        processed_path = os.path.join(output_dir, "processed_temperature.tif")

        thermal_result = await Runner.run(
            self.thermal_agent,
            f"Process thermal imagery from {thermal_raster_path} and save to {processed_path}. "
            "Analyze thermal patterns and identify potential hotspots."
        )

        # Step 2: Compliance evaluation
        compliance_result = await Runner.run(
            self.compliance_agent,
            f"Evaluate compliance of processed temperature data at {processed_path} "
            f"against warning threshold {warning_threshold}°C and critical threshold {critical_threshold}°C. "
            "Identify any exceedances and assess environmental impact."
        )

        # Step 3: Generate comprehensive report
        report_path = os.path.join(output_dir, "compliance_report.md")

        report_result = await Runner.run(
            self.report_agent,
            f"Generate comprehensive NPDES compliance report based on the compliance evaluation results. "
            f"Save the report to {report_path} and include detailed analysis and recommendations."
        )

        # Step 4: Check if alerts are needed
        alert_check = await Runner.run(
            self.alert_agent,
            "Review the compliance results and determine if any alerts should be sent. "
            "If critical exceedances are found, prepare appropriate alert messages."
        )

        return {
            "thermal_processing": thermal_result.final_output,
            "compliance_evaluation": compliance_result.final_output,
            "report_generation": report_result.final_output,
            "alert_assessment": alert_check.final_output,
            "output_files": {
                "processed_raster": processed_path,
                "compliance_report": report_path
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def run_plume_simulation_workflow(
        self,
        ny: int,
        nx: int,
        u: float,
        v: float,
        output_dir: str = "plume_output"
    ) -> Dict[str, Any]:
        """Run plume modeling workflow."""

        os.makedirs(output_dir, exist_ok=True)
        plume_path = os.path.join(output_dir, "plume_simulation.npy")

        # Run plume simulation
        plume_result = await Runner.run(
            self.plume_agent,
            f"Simulate thermal plume with grid size {ny}x{nx}, flow velocity ({u}, {v}) m/s. "
            f"Save results to {plume_path} and analyze plume behavior and mixing characteristics."
        )

        # Evaluate compliance of plume
        compliance_result = await Runner.run(
            self.compliance_agent,
            f"Evaluate compliance of plume simulation results from {plume_path}. "
            "Assess thermal impact and potential regulatory concerns."
        )

        # Generate plume analysis report
        report_path = os.path.join(output_dir, "plume_report.md")

        report_result = await Runner.run(
            self.report_agent,
            f"Generate detailed report on plume simulation and compliance assessment. "
            f"Save to {report_path} with visualizations and recommendations."
        )

        return {
            "plume_simulation": plume_result.final_output,
            "compliance_assessment": compliance_result.final_output,
            "report": report_result.final_output,
            "output_files": {
                "plume_data": plume_path,
                "analysis_report": report_path
            }
        }

    async def run_ml_training_workflow(
        self,
        training_data_path: str,
        target_column: str,
        output_dir: str = "ml_output"
    ) -> Dict[str, Any]:
        """Run machine learning training workflow."""

        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "thermal_prediction_model.joblib")

        # Train ML model
        ml_result = await Runner.run(
            self.ml_agent,
            f"Train machine learning model using data from {training_data_path} "
            f"to predict {target_column}. Save model to {model_path} and evaluate performance."
        )

        # Generate ML analysis report
        report_path = os.path.join(output_dir, "ml_analysis_report.md")

        report_result = await Runner.run(
            self.report_agent,
            f"Generate comprehensive report on ML model training and performance analysis. "
            f"Save to {report_path} with model evaluation metrics and insights."
        )

        return {
            "ml_training": ml_result.final_output,
            "report": report_result.final_output,
            "output_files": {
                "trained_model": model_path,
                "analysis_report": report_path
            }
        }


# Global orchestrator instance
orchestrator = OpenWaterThermalAnalystOrchestrator()


async def run_agents_workflow(workflow_type: str, **kwargs) -> Dict[str, Any]:
    """Main entry point for running agent workflows."""

    if workflow_type == "comprehensive_analysis":
        return await orchestrator.run_comprehensive_analysis(**kwargs)
    elif workflow_type == "plume_simulation":
        return await orchestrator.run_plume_simulation_workflow(**kwargs)
    elif workflow_type == "ml_training":
        return await orchestrator.run_ml_training_workflow(**kwargs)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")


def run_agents_workflow_sync(workflow_type: str, **kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for agent workflows."""
    return asyncio.run(run_agents_workflow(workflow_type, **kwargs))
