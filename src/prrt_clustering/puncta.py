"""Puncta detection module."""

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from scipy.optimize import curve_fit


class Gaussian3D:
    """3D Gaussian."""

    def __init__(self, sigma):
        self.d = 2 * sigma
        self.sigma = sigma

    def model(self, x, p0, p1, p2, p3, p4):
        """3D Gaussian function with fixed sigma."""
        p = np.array([p0, p1, p2]).reshape(3, 1)
        delta = (x - p) / self.sigma
        return p3 + p4 * np.exp(-0.5 * np.sum(np.square(delta), axis=0))

    def jac(self, x, p0, p1, p2, p3, p4):
        """3D Gaussian function Jacobian."""
        # Recompute the core Gaussian part
        p = np.array([p0, p1, p2]).reshape(3, 1)
        delta = (x - p) / self.sigma
        g_core = np.exp(-0.5 * np.sum(np.square(delta), axis=0))
        # Partial derivatives:
        factor = (p4 * g_core) / (self.sigma**2)
        df_dp0 = factor * (x[0] - p0)
        df_dp1 = factor * (x[1] - p1)
        df_dp2 = factor * (x[2] - p2)
        df_dp3 = np.ones_like(g_core)
        df_dp4 = g_core
        return np.stack([df_dp0, df_dp1, df_dp2, df_dp3, df_dp4], axis=1)

    def simple_fit(self, data: xr.DataArray, p: np.ndarray, niter: int = 3):
        """Fit the model using a mean-shift.

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
            Z=slice(p[0] - 3 * self.d, p[0] + 3 * self.d),
            Y=slice(p[1] - self.d, p[1] + self.d),
            X=slice(p[2] - self.d, p[2] + self.d),
        )
        axes = [crop[a].values for a in ["Z", "Y", "X"]]
        xdata = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).T
        ydata = np.array(crop.data).astype(float).ravel()
        popt = p.copy()
        for _ in range(niter):
            delta = (xdata - popt.reshape(3, 1)) / self.sigma
            weights = ydata * np.exp(-0.5 * np.sum(delta**2, axis=0))
            popt = (xdata @ weights) / np.sum(weights)
        return popt

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
            Z=slice(p[0] - 3 * self.d, p[0] + 3 * self.d),
            Y=slice(p[1] - self.d, p[1] + self.d),
            X=slice(p[2] - self.d, p[2] + self.d),
        )
        axes = [crop[a].values for a in ["Z", "Y", "X"]]
        xdata = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).T
        ydata = np.array(crop.data).astype(float).ravel()
        a, b = np.percentile(ydata, [0, 100])
        ydata = (ydata - a) / (b - a)
        p0 = np.array([p[0], p[1], p[2], 0, 1])
        try:
            popt = curve_fit(self.model, xdata, ydata, p0, jac=self.jac)[0]
            popt = curve_fit(self.model, xdata, ydata, p0, jac=self.jac)[0]

            if np.linalg.norm(p0[:3] - popt[:3]) > 2:
                popt = p0

        except Exception:
            popt = p0
        popt[3] = popt[3] * (b - a) + a
        popt[4] = popt[4] * (b - a)

        return popt


def _detect_puncta_log(
    data: xr.DataArray,
    scale_um: float = 0.1,
    specificity: float = 3,
    subpixel: bool = True,
) -> pd.DataFrame:
    """Detect puncta with Laplacian of Gaussians.

    Parameters
    ----------
    data: xr.DataArray
        Input array
    scale_um: float
    specificity: float

    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates (Z, Y, X), channel (C), and intensity.
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
    mask = score > t
    lm = np.abs(ndi.maximum_filter(score, size=3) - score) < 1e-6
    mask *= lm
    coords = origin + voxel * np.stack(np.where(mask), axis=1)
    intensity = data.values[mask]
    num_points = coords.shape[0]
    if subpixel is True:
        gauss = Gaussian3D(sigma=scale_um)
        new_coords = []
        for p in coords:
            # new_coords.append(gauss.fit(data, p))
            new_coords.append(gauss.simple_fit(data, p))
        coords = np.array(new_coords)
    channel_val = data.C.item() if "C" in data.coords else 0
    if num_points == 0:
        return pd.DataFrame(columns=["Z", "Y", "X", "C", "intensity"])
    return pd.DataFrame(
        {
            "Z": coords[:, 0],
            "Y": coords[:, 1],
            "X": coords[:, 2],
            "C": channel_val,
            "intensity": intensity,
        }
    )


def detect_puncta(
    data: xr.DataArray,
    scale_um: float = 0.1,
    specificity: float = 3,
    subpixel: bool = False,
) -> pd.DataFrame:
    """Detect puncta in a multi-channel dataset.

    Parameters
    ----------
    data: xr.DataArray
        Input array
    scale_um: float
    specificity: float

    Returns
    -------
    pd.DataFrame
        DataFrame containing detected puncta coordinates and attributes.
    """
    points = []
    dims = [data[d] for d in data.dims if d not in ["Z", "Y", "X"]]
    if not dims:
        return _detect_puncta_log(
            data, scale_um=scale_um, specificity=specificity, subpixel=subpixel
        )
    for coords in product(*dims):
        cdict = {c.name: c.data.item() for c in coords}
        points.append(
            _detect_puncta_log(
                data.sel(cdict),
                scale_um=scale_um,
                specificity=specificity,
                subpixel=subpixel,
            )
        )
    if len(points) == 0:
        return pd.DataFrame(columns=["Z", "Y", "X", "C", "intensity"])
    return pd.concat(points, ignore_index=True)
