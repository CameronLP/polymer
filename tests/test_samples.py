from pathlib import Path

import pytest
from core import env

"""
The path to these products should be provided in the file `.env`

Example in .env:
    LEVEL1_SAMPLE_<sensor> = /path/to/sensor_product
"""

expected_products = {
    "LEVEL1_SAMPLE_PRISMA": "PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5",
    "LEVEL1_SAMPLE_HYPSO": "aeronetvenice_2025-03-04T10-38-05Z-l1c.nc",
    "LEVEL1_SAMPLE_PACE_OCI": "PACE_OCI.20250101T000738.L1B.V3.nc"
}

def sample(sensor) -> Path:
    return Path(env.getvar(sensor))

@pytest.mark.parametrize("sensor", list(expected_products.keys()))
def test_samples(sensor):
    """Verify that sample Level-1 data files exist and have the expected filenames.

    For each sensor, this test checks that the environment variable
    LEVEL1_SAMPLE_<SENSOR> points to an existing file whose name matches
    the expected product name in ``expected_products``.
    """
    assert sample(sensor).exists()
    assert sample(sensor).name == expected_products[sensor]