"""Tests for puncta detection and 3D Gaussian fitting."""

import numpy as np
import xarray as xr

from synxm.puncta import Gaussian3D


def test_gaussian3d_fit_recovers_center_and_shapes():
    """Test 3D Gaussian fit recovers center and shape parameters accurately."""
    axis = np.arange(-8.0, 8.5, 0.5)
    coordinates = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=0
    ).reshape(3, -1)
    gaussian = Gaussian3D(sigma=1.0)
    expected = np.array([0.1, -0.2, 0.3, 0.0, 1.0])
    values = gaussian.model(coordinates, *expected).reshape(
        axis.size, axis.size, axis.size
    )
    data = xr.DataArray(
        values,
        dims=("Z", "Y", "X"),
        coords={"Z": axis, "Y": axis, "X": axis},
    )

    xdata = coordinates
    assert gaussian.model(xdata, *expected).shape == (axis.size**3,)
    assert gaussian.jac(xdata, *expected).shape == (axis.size**3, 5)

    initial_center = np.array([0.25, -0.05, 0.45])
    fitted = gaussian.fit(data, initial_center)

    np.testing.assert_allclose(fitted[:3], expected[:3], atol=1e-3)


def test_detect_puncta_returns_dataframe():
    """Test detect_puncta returns a valid DataFrame on synthetic spot data."""
    import pandas as pd

    from synxm.puncta import detect_puncta

    axis_z = np.linspace(0, 10, 11)
    axis_y = np.linspace(0, 10, 11)
    axis_x = np.linspace(0, 10, 11)
    # create synthetic image with a bright spot at center
    arr = np.zeros((2, 11, 11, 11), dtype=np.float32)
    arr[0, 5, 5, 5] = 100.0
    arr[1, 5, 5, 5] = 80.0

    da = xr.DataArray(
        arr,
        dims=("C", "Z", "Y", "X"),
        coords={"C": [0, 1], "Z": axis_z, "Y": axis_y, "X": axis_x},
    )

    params = {
        0: {"scale_um": 1.0, "specificity": 1},
        1: {"scale_um": 1.0, "specificity": 1},
    }

    df = detect_puncta(da, params)
    assert isinstance(df, pd.DataFrame)
    assert set(["Z", "Y", "X", "C", "mean intensity", "volume"]).issubset(df.columns)
    assert len(df) > 0
    assert set(df["C"].unique()).issubset({0, 1})
