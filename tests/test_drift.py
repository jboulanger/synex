"""Tests for z-drift estimation and correction."""

import numpy as np
import xarray as xr

from prrt_clustering import correct_z_drift


def test_correct_z_drift():
    """Test drift correction."""
    np.random.seed(42)
    true_shift = np.array([0.5, 0.25])
    shape = (1, 1, 10, 20, 30)

    # Create a grid (2,H,D)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape[3:]], indexing="ij"))

    # Create a set of points (some might be on the border and disappear,
    # we want to be robust to that)
    pts = np.random.uniform(low=[0, 0], high=[shape[3], shape[4]], size=(30, 2))

    # Create a stack with smooth particles shifting by a half a pixel
    _data = np.zeros(shape)
    for k in range(shape[2]):
        for p in pts:
            _data[0, 0, k] += np.exp(
                -0.5 * np.square(grid - p.reshape(2, 1, 1)).sum(axis=0)
            )
        pts += true_shift
    assert _data.shape == shape

    # Create the xr.dataarray
    data = xr.DataArray(
        data=_data,
        dims=["T", "C", "Z", "Y", "X"],
        coords={
            "T": np.array([0]),
            "C": np.array([0]),
            "Z": np.arange(shape[2]),
            "Y": np.arange(shape[3]),
            "X": np.arange(shape[4]),
        },
    )

    # Estimate/correct drift
    shifts, corrected = correct_z_drift(data)

    # Check correctness
    assert len(shifts) == shape[2]
    errors = np.array([[d["VY"], d["VX"]] for d in shifts.iloc]) - true_shift
    assert np.mean(np.abs(errors[1:])) < 0.1
    assert corrected.data.shape == shape


if __name__ == "__main__":
    test_correct_z_drift()
