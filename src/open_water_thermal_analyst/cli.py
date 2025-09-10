import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__
from . import imagery
from .config import load_config, ComplianceThresholds
from .compliance import evaluate_compliance
from .alerts import send_alert
from .hydrodynamics.plume_model import simulate_plume
from .reports.npdes_report import generate_markdown_report


def cmd_process_imagery(args: argparse.Namespace) -> None:
    arr, transform, crs_wkt, nodata = imagery.read_raster(args.input)

    if args.k1 is not None and args.k2 is not None:
        bt_k = imagery.brightness_temperature_from_radiance(arr, args.k1, args.k2)
    else:
        bt_k = arr  # assume brightness temperature is provided in Kelvin

    lst_k = imagery.surface_temperature_from_bt(bt_k, emissivity=args.emissivity, wavelength_um=args.wavelength_um)
    out = imagery.kelvin_to_celsius(lst_k) if args.to_celsius else lst_k
    imagery.write_geotiff(args.output, out.astype(np.float32), transform, crs_wkt, nodata)


def cmd_model_plume(args: argparse.Namespace) -> None:
    ny, nx = args.ny, args.nx
    T0 = np.zeros((ny, nx), dtype=float)
    if args.init_hotspot is not None:
        iy, ix, dT = args.init_hotspot
        if 0 <= iy < ny and 0 <= ix < nx:
            T0[iy, ix] = dT
    source = tuple(args.source) if args.source else None
    T = simulate_plume(T0, args.u, args.v, args.kappa, args.dx, args.dy, args.dt, args.steps, source=source)  # type: ignore
    np.save(args.output, T)


def cmd_check_compliance(args: argparse.Namespace) -> None:
    cfg = load_config(args.config) if args.config else load_config(None)
    arr, _, _, _ = imagery.read_raster(args.temp)
    data_c = arr - 273.15 if args.kelvin else arr

    thresholds = ComplianceThresholds(
        warning_celsius=args.warning if args.warning is not None else cfg.thresholds.warning_celsius,
        critical_celsius=args.critical if args.critical is not None else cfg.thresholds.critical_celsius,
        rolling_minutes=cfg.thresholds.rolling_minutes,
    )

    results = evaluate_compliance(data_c, thresholds)

    # Alerting
    if results["critical_exceedances"] > 0:
        send_alert("CRITICAL", "Thermal critical threshold exceeded", {"count": results["critical_exceedances"]}, cfg.alert_webhook_url)
    elif results["warning_exceedances"] > 0:
        send_alert("WARNING", "Thermal warning threshold exceeded", {"count": results["warning_exceedances"]}, cfg.alert_webhook_url)

    # Optional report
    if args.report_md:
        generate_markdown_report(results, args.report_md)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in results.items() if not k.endswith("_mask")}, f, indent=2)
    else:
        print(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in results.items() if not k.endswith("_mask")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openwater",
        description=(
            "OpenWaterThermalAnalyst: Thermal remote sensing, hydrodynamics, compliance checks, ML, and alerts"
        ),
    )
    parser.add_argument("--version", action="version", version=f"openwater {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # process-imagery
    p_proc = subparsers.add_parser("process-imagery", help="Process thermal imagery to surface temperature")
    p_proc.add_argument("input", help="Input single-band thermal raster (radiance or BT)")
    p_proc.add_argument("output", help="Output GeoTIFF path")
    p_proc.add_argument("--emissivity", type=float, default=0.99)
    p_proc.add_argument("--wavelength-um", dest="wavelength_um", type=float, default=10.9)
    p_proc.add_argument("--k1", type=float, default=None, help="Planck K1 (if input is radiance)")
    p_proc.add_argument("--k2", type=float, default=None, help="Planck K2 (if input is radiance)")
    p_proc.add_argument("--to-celsius", action="store_true", help="Write output in Celsius")
    p_proc.set_defaults(func=cmd_process_imagery)

    # model-plume
    p_plume = subparsers.add_parser("model-plume", help="Simulate 2D advection-diffusion plume")
    p_plume.add_argument("ny", type=int)
    p_plume.add_argument("nx", type=int)
    p_plume.add_argument("--u", type=float, required=True, help="Flow velocity u (m/s)")
    p_plume.add_argument("--v", type=float, required=True, help="Flow velocity v (m/s)")
    p_plume.add_argument("--kappa", type=float, default=1e-4, help="Thermal diffusivity (m^2/s)")
    p_plume.add_argument("--dx", type=float, default=10.0)
    p_plume.add_argument("--dy", type=float, default=10.0)
    p_plume.add_argument("--dt", type=float, default=1.0)
    p_plume.add_argument("--steps", type=int, default=100)
    p_plume.add_argument("--init-hotspot", nargs=3, type=float, metavar=("IY","IX","dT"))
    p_plume.add_argument("--source", nargs=3, type=float, metavar=("IY","IX","S"))
    p_plume.add_argument("--output", required=True, help="Output .npy file path")
    p_plume.set_defaults(func=cmd_model_plume)

    # check-compliance
    p_comp = subparsers.add_parser("check-compliance", help="Evaluate temperature raster against thresholds")
    p_comp.add_argument("temp", help="Temperature raster (Celsius by default)")
    p_comp.add_argument("--kelvin", action="store_true", help="Interpret input as Kelvin")
    p_comp.add_argument("--warning", type=float, default=None, help="Warning threshold (°C)")
    p_comp.add_argument("--critical", type=float, default=None, help="Critical threshold (°C)")
    p_comp.add_argument("--report-md", dest="report_md", default=None, help="Optional Markdown report output path")
    p_comp.add_argument("-o", "--output", default=None, help="Optional JSON output path")
    p_comp.add_argument("--config", default=None, help="Optional YAML config path")
    p_comp.set_defaults(func=cmd_check_compliance)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()

