"""Puncta detection module."""

from itertools import product

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from scipy.optimize import curve_fit


class Gaussian3D:
    """3D Gaussian."""

    def __init__(self, sigma, shift=0.5):
        self.shift = shift
        self.d = 2 * sigma
        self.sigma = sigma

    def model(self, x, p0, p1, p2, p3, p4):
        """3D Gaussian function with fixed sigma."""
        p = np.array([p0, p1, p2]).reshape(3, 1)
        delta = (x - p - self.shift) / self.sigma
        return p3 + p4 * np.exp(-np.sum(np.square(delta), axis=0))

    def jac(self, x, p0, p1, p2, p3, p4):
        """3D Gaussian function Jacobian."""
        # Recompute the core Gaussian part
        p = np.array([p0, p1, p2]).reshape(3, 1)
        delta = (x - p - self.shift) / self.sigma
        g_core = np.exp(-np.sum(np.square(delta), axis=0))
        # Partial derivatives:
        factor = (2 * p4 * g_core) / (self.sigma**2)
        df_dp0 = factor * (x[0] - p0 - self.shift)
        df_dp1 = factor * (x[1] - p1 - self.shift)
        df_dp2 = factor * (x[2] - p2 - self.shift)
        df_dp3 = np.ones_like(g_core)
        df_dp4 = g_core
        return np.stack([df_dp0, df_dp1, df_dp2, df_dp3, df_dp4], axis=1)

    def fit(self, data: xr.DataArray, p):
        """Fit the model.

        Parameters
        ----------
        data: ndarray
            2D image [H,W]
        p: ndarray
            Initial parameters [z,y,x,a,b]

        Returns
        -------
        ndarray:
            Estimated parameters [z,y,x,a,b]
        """
        crop = data.sel(
            X=slice(p[2] - self.d, p[2] + self.d),
            Y=slice(p[1] - self.d, p[1] + self.d),
            Z=slice(p[0] - 3 * self.d, p[0] + 3 * self.d),
        )
        axes = [crop[a].values for a in ["Z", "Y", "X"]]
        xdata = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).T
        ydata = np.array(crop.data).astype(float).ravel()
        a, b = np.percentile(ydata, [50, 100])
        ydata = (ydata - a) / (b - a)
        p0 = np.array([p[0], p[1], p[2], 0, 1])
        try:
            popt = curve_fit(self.model, xdata, ydata, p0, jac=self.jac)[0]
        except Exception:
            popt = p0
        return popt


def _detect_puncta_log(
    data: xr.DataArray, scale_um: float = 0.1, specificity: float = 3
):
    """Detect puncta with Laplacian of Gaussians.

    Parameters
    ----------
    data: xr.DataArray
        Input array
    scale_um: float
    specificity: float
    """
    voxel = np.array([(data[a][1].values - data[a][0].values) for a in ["Z", "Y", "X"]])
    origin = np.array([data[a][0].values for a in ["Z", "Y", "X"]])
    sigma = scale_um / voxel
    sigma[0] *= 3
    score = np.sqrt(data.data.astype(np.float32))
    score = ndi.gaussian_filter(score, sigma)
    score = -ndi.laplace(score)
    beta = np.median(np.abs(score))
    t = 2.302 * specificity * beta  # log(10)
    mask = ndi.median_filter(score > t, 3)

    lm = np.abs(ndi.maximum_filter(score, size=3) - score) < 1e-6
    # import matplotlib.pyplot as plt

    # plt.imshow((4 * score / score.max() + mask + 3 * lm).max(0))
    # plt.show()
    mask *= lm
    coords = origin + voxel * np.stack(np.where(mask), axis=1)
    num_points = coords.shape[0]
    # gauss = Gaussian3D(sigma=scale_um)
    # new_coords = []
    # for p in coords:
    #     new_coords.append(gauss.fit(data, p))
    # new_coords = np.array(new_coords)
    print(num_points)
    points = xr.Dataset(
        data_vars={"intensity": ("point", np.ones(num_points))},
        coords={
            "point": np.arange(num_points),
            "X": ("point", coords[:, 2]),
            "Y": ("point", coords[:, 1]),
            "Z": ("point", coords[:, 0]),
            "C": ("point", [data.C.values] * coords.shape[0]),
        },
    )
    return points


def detect_puncta(
    data: xr.DataArray, scale_um: float = 0.1, specificity: float = 3
) -> xr.Dataset:
    """Detect puncta in an multi-channel dataset.

    Parameters
    ----------
    data: xr.DataArray
        Input array
    scale_um: float
    specificity: float

    Returns
    -------
    list[dict]
        List of dictionary with point coordinates
            [{"C":channel,"coordinates":array([N,3])}].
    """
    points = []
    dims = [data[d] for d in data.dims if d not in ["Z", "Y", "X"]]
    for coords in product(*dims):
        cdict = {c.name: c.data.item() for c in coords}
        points.append(_detect_puncta_log(data.sel(cdict), scale_um, specificity))
    return xr.concat(points, dim="point")
