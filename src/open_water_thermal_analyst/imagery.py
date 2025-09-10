from __future__ import annotations

import math
from typing import Tuple, Optional
import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
except Exception as e:  # pragma: no cover - optional import
    rasterio = None  # type: ignore
    Affine = object  # type: ignore


def read_raster(path: str) -> Tuple[np.ndarray, Optional[Affine], Optional[str], Optional[float]]:
    """Read a single-band raster into a numpy array.

    Returns (array, transform, crs, nodata).
    """
    if rasterio is None:
        raise RuntimeError(
            "rasterio is required for reading rasters. Please install the 'rasterio' package."
        )
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        return data, src.transform, src.crs.to_wkt() if src.crs else None, src.nodata


def brightness_temperature_from_radiance(L: np.ndarray, K1: float, K2: float) -> np.ndarray:
    """Convert spectral radiance to brightness temperature in Kelvin.

    BT = K2 / ln(K1/L + 1)
    """
    L_clip = np.clip(L, 1e-12, np.inf)
    return K2 / np.log((K1 / L_clip) + 1.0)


def surface_temperature_from_bt(
    bt_kelvin: np.ndarray,
    emissivity: float = 0.99,
    wavelength_um: float = 10.9,
) -> np.ndarray:
    """Estimate land/water surface temperature (Kelvin) from brightness temperature.

    Applies emissivity correction using the single-channel approach.
    wavelength_um = sensor effective wavelength (e.g., Landsat 8 TIRS band ~10.9 µm)
    """
    # c2 in um*K (second radiation constant ~ 1.4388e4 um*K)
    c2 = 1.4388e4
    with np.errstate(divide="ignore", invalid="ignore"):
        lst_k = bt_kelvin / (1 + (wavelength_um * bt_kelvin / c2) * np.log(emissivity))
    return lst_k


def kelvin_to_celsius(K: np.ndarray) -> np.ndarray:
    return K - 273.15


def write_geotiff(path: str, data: np.ndarray, transform: Affine, crs_wkt: Optional[str], nodata: Optional[float] = None) -> None:  # type: ignore
    if rasterio is None:
        raise RuntimeError(
            "rasterio is required for writing rasters. Please install the 'rasterio' package."
        )
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "transform": transform,
        "crs": crs_wkt,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)

