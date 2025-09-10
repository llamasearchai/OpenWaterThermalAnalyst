import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from open_water_thermal_analyst.imagery import (
    read_raster,
    brightness_temperature_from_radiance,
    surface_temperature_from_bt,
    kelvin_to_celsius,
    write_geotiff
)


class TestImageryProcessing:
    """Test thermal imagery processing functions."""

    def test_brightness_temperature_from_radiance(self):
        """Test brightness temperature conversion from radiance."""
        # Test data: typical Landsat 8 TIRS radiance values
        L = np.array([0.1, 1.0, 10.0])
        K1 = 774.8853  # Landsat 8 TIRS K1
        K2 = 1321.0789  # Landsat 8 TIRS K2

        bt_k = brightness_temperature_from_radiance(L, K1, K2)

        # Check that output is reasonable (should be in Kelvin)
        assert bt_k.shape == L.shape
        assert np.all(bt_k > 100)  # Should be above absolute zero (adjusted for test data)
        assert np.all(bt_k < 400)  # Should be reasonable temperature

        # Test monotonicity
        assert bt_k[0] < bt_k[1] < bt_k[2]

    def test_surface_temperature_from_bt(self):
        """Test surface temperature estimation from brightness temperature."""
        bt_k = np.array([300.0, 310.0, 320.0])
        emissivity = 0.99
        wavelength_um = 10.9

        lst_k = surface_temperature_from_bt(bt_k, emissivity, wavelength_um)

        assert lst_k.shape == bt_k.shape
        assert np.all(lst_k > 290)  # Should be close to input but corrected
        # Note: For high emissivity (0.99), correction might be minimal or slightly increase temperature
        assert np.allclose(lst_k, bt_k, rtol=0.01)  # Should be very close to input

    def test_kelvin_to_celsius(self):
        """Test temperature unit conversion."""
        kelvin_temps = np.array([273.15, 293.15, 313.15])
        celsius_temps = kelvin_to_celsius(kelvin_temps)

        expected = np.array([0.0, 20.0, 40.0])
        np.testing.assert_array_almost_equal(celsius_temps, expected, decimal=2)

    def test_write_geotiff(self):
        """Test GeoTIFF writing functionality."""
        try:
            from rasterio.transform import Affine
        except ImportError:
            pytest.skip("rasterio not available")

        # Create test data
        data = np.random.rand(10, 10).astype(np.float32)
        transform = Affine(1, 0, 0, 0, -1, 0)  # Proper affine transform
        crs_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            try:
                write_geotiff(tmp.name, data, transform, crs_wkt)

                # Verify file was created
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0

                # Try to read it back
                data_read, transform_read, crs_read, nodata_read = read_raster(tmp.name)

                assert data_read.shape == data.shape
                np.testing.assert_array_equal(data_read, data)

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_read_raster_nonexistent_file(self):
        """Test error handling for non-existent files."""
        try:
            import rasterio
            with pytest.raises((RuntimeError, FileNotFoundError, rasterio.errors.RasterioIOError)):
                read_raster("nonexistent_file.tif")
        except ImportError:
            pytest.skip("rasterio not available")

    def test_temperature_conversion_pipeline(self):
        """Test complete temperature conversion pipeline."""
        # Start with radiance values
        radiance = np.array([5.0, 10.0, 15.0])

        # Convert to brightness temperature
        K1, K2 = 774.8853, 1321.0789
        bt_k = brightness_temperature_from_radiance(radiance, K1, K2)

        # Apply emissivity correction
        lst_k = surface_temperature_from_bt(bt_k, emissivity=0.98, wavelength_um=10.9)

        # Convert to Celsius
        lst_c = kelvin_to_celsius(lst_k)

        # Verify pipeline results
        assert lst_c.shape == radiance.shape
        assert np.all(lst_c > -50)  # Reasonable temperature range
        assert np.all(lst_c < 100)

        # Verify monotonicity is preserved
        assert lst_c[0] < lst_c[1] < lst_c[2]
