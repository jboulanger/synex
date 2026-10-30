"""Puncta detection module."""

from itertools import product

import numpy as np
import pandas as pd
import skimage as ski
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

    def _prepare_data(self, data: xr.DataArray, p: np.ndarray):
        crop = data.sel(
            Z=slice(p[0] - 3 * self.d, p[0] + 3 * self.d),
            Y=slice(p[1] - self.d, p[1] + self.d),
            X=slice(p[2] - self.d, p[2] + self.d),
        )

        axes = [crop[a].values for a in ["Z", "Y", "X"]]
        xdata = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).T
        ydata = np.asarray(crop.data, dtype=float, copy=False).ravel()
        ymin, ymax = ydata.min(), ydata.max()
        ydata = (ydata - ymin) / (ymax - ymin)
        return xdata, ydata, ymin, ymax

    def simple_fit(self, data: xr.DataArray, p: np.ndarray, niter: int = 3):
        """Fit the model location parameters using a mean-shift.

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
        xdata, ydata, _, _ = self._prepare_data(data, p)
        popt = p.copy()
        for _ in range(niter):
            delta = (xdata - popt[:3].reshape(3, 1)) / self.sigma
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
        xdata, ydata, ymin, ymax = self._prepare_data(data, p)
        p0 = np.array([p[0], p[1], p[2], 0, 1])
        try:
            popt = curve_fit(self.model, xdata, ydata, p0, jac=self.jac)[0]
            if np.linalg.norm(p0[:3] - popt[:3]) > 2:
                popt = p0

        except Exception:
            popt = p0
        popt[3] = popt[3] * (ymax - ymin) + ymin
        popt[4] = popt[4] * (ymax - ymin)

        return popt


def _detect_puncta_log(
    data: xr.DataArray,
    scale_um: float = 0.1,
    specificity: float = 3,
    subpixel: bool = False,
) -> pd.DataFrame:
    """Detect puncta with Laplacian of Gaussians.

    Parameters
    ----------
    data: xr.DataArray
        Input array
    scale_um: float
    specificity: float
    subpixel:bool
        activate subpixel localization

    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates (Z, Y, X), channel (C), and intensity.
    """
    # Compute scale in pixels
    voxel = np.array([(data[a][1].values - data[a][0].values) for a in ["Z", "Y", "X"]])
    origin = np.array([data[a][0].values for a in ["Z", "Y", "X"]])
    sigma = scale_um / voxel
    sigma[0] *= 3
    # Compute a score using the laplacian of Gaussian filter
    score = np.asarray(data.data, dtype=np.float32)
    score = np.sqrt(np.maximum(0, score))
    score = ndi.gaussian_filter(score, sigma)
    score = -ndi.laplace(score)

    # Threshold the score assuming the score follows a laplace distribution
    m = np.median(score)
    beta = np.median(np.abs(score - m))
    t = m + 2.302 * specificity * beta  # log(10)
    mask = score > t

    # Compute the distance function
    distance = ndi.distance_transform_edt(mask, sampling=voxel)
    distance = ndi.gaussian_filter(distance, 1)

    # Define markers as local maxima of the score within the mask
    peak_coords = ski.feature.peak_local_max(
        score,
        labels=mask,
        exclude_border=False,
    )
    peak_mask = np.zeros_like(mask, dtype=bool)
    if len(peak_coords) > 0:
        peak_mask[tuple(peak_coords.T)] = True
    markers = ski.morphology.label(peak_mask, connectivity=3)

    # Split regions using a watershed
    labels = ski.segmentation.watershed(-distance, markers, mask=mask)

    # Extract centroid, area and mean intensity
    props = ski.measure.regionprops(labels, np.asarray(data.data))

    if len(props) == 0:
        return pd.DataFrame(columns=["Z", "Y", "X", "C", "intensity", "area"])
    coords = np.array(
        [[*p.centroid_weighted, p.area, p.intensity_mean] for p in props if p.area > 9]
    )
    if len(coords) == 0:
        return pd.DataFrame(columns=["Z", "Y", "X", "C", "intensity", "area"])

    # Scale spatial coordinates
    coords[:, :3] = origin + voxel * coords[:, :3]

    # Subpixel localization
    if subpixel is True:
        gauss = Gaussian3D(sigma=scale_um)
        new_coords = []
        for p in coords:
            popt = gauss.fit(data, p)
            new_coords.append([popt[0], popt[1], popt[2], p[3], p[4]])
        coords = np.array(new_coords)

    # Format the resutls as a dataframe
    return pd.DataFrame(
        {
            "Z": coords[:, 0],
            "Y": coords[:, 1],
            "X": coords[:, 2],
            "C": data.C.item() if "C" in data.coords else 0,
            "intensity": coords[:, 3] * coords[:, 4],
            "area": coords[:, 3],
        }
    )


def detect_puncta(
    data: xr.DataArray,
    scale_um: list[float] = [0.1, 0.1, 0.1, 0.1],
    specificity: list[float] = [3, 3, 3, 3],
    subpixel: bool = False,
) -> pd.DataFrame:
    """Detect puncta in a multi-channel dataset.

    Parameters
    ----------
    data: xr.DataArray
        Input array
    scale_um: list[float] | float
    specificity: list[float] | float

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
    for index, coords in enumerate(product(*dims)):
        cdict = {c.name: c.data.item() for c in coords}
        points.append(
            _detect_puncta_log(
                data.sel(cdict),
                scale_um=scale_um[index],
                specificity=specificity[index],
                subpixel=subpixel,
            )
        )
        points[-1]["index"] = index
        points[-1]["index"] = points[-1]["index"].astype(int)

    if len(points) == 0:
        return pd.DataFrame(columns=["Z", "Y", "X", "C", "intensity", "area", "index"])
    return pd.concat(points, ignore_index=True)
