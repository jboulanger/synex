import numpy as np
import xarray as xr

from prrt_clustering.puncta import Gaussian3D


def test_gaussian3d_fit_recovers_center_and_shapes():
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
