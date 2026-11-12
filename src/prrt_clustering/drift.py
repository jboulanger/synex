"""Drift correction."""

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi


def estimate_translation(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Estimate the global translation between two images.

    Uses Lucas-Kanade optical flow method.

    Parameters
    ----------
    reference : np.ndarray
        The reference (template) image (2D).
    moving : np.ndarray
        The moving image to align with the reference, same shape as reference.

    Returns
    -------
    np.ndarray
        A 1D translation vector [dy, dx] in pixels that registers the moving image.
    """
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    # Ensure floating-point precision
    ref = np.array(reference).astype(np.float64, copy=False)
    mov = np.array(moving).astype(np.float64, copy=False)

    # Filtering
    ref = ndi.gaussian_filter(ref, 1)
    mov = ndi.gaussian_filter(mov, 1)

    # Spatial gradients
    I1, I2 = np.gradient(ref)
    # Temporal gradient
    I0 = mov - ref

    # Structure tensor components (mean over pixels)
    a = np.square(I1).mean()
    b = (I1 * I2).mean()
    c = np.square(I2).mean()
    d = -(I1 * I0).mean()
    e = -(I2 * I0).mean()

    # Construct system matrix and vector
    A = np.array([[a, b], [b, c]])
    B = np.array([d, e])

    # Least-squares solve to handle textureless/flat regions gracefully
    translation, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return translation


def correct_z_drift(data: xr.DataArray) -> tuple[pd.DataFrame, xr.DataArray]:
    """Correct the drift across the Z-axis using all channels.

    Parameters
    ----------
    data: np.ndarray

    Returns
    -------
    list[dict]: list of shifts {"T": t, "C": c, "Z": z, "shift":[dy,dx]}
    np.ndarray: corrected image
    """
    shifts = []
    stabilized = xr.DataArray(np.zeros_like(data.data), data.coords)

    z0 = data["Z"][0]
    # Estimation
    for t in data["T"]:
        shifts.append(
            {
                "T": t.item(),
                "Z": z0.item(),
                "VX": 0,
                "VY": 0,
            }
        )
        reference = data.sel(T=t, Z=z0).mean("C").data
        for z in data["Z"][1:]:
            moving = data.sel(T=t, Z=z).mean("C").data
            v = estimate_translation(reference, moving)
            shifts.append({"T": t.item(), "Z": z.item(), "VY": v[0], "VX": v[1]})
            reference, moving = moving, reference

    # Create a DataFrame and compute the cumulative shifts
    shifts = pd.DataFrame.from_records(shifts, index=["T", "Z"])
    shifts["DX"] = shifts["VX"].cumsum()
    shifts["DY"] = shifts["VY"].cumsum()

    # Correction
    for t in data["T"]:
        for c in data["C"]:
            stabilized.loc[{"T": t, "C": c, "Z": z0}] = data.sel(T=t, C=c, Z=z0)
            for z in data["Z"][1:]:
                moving = data.sel(T=t, C=c, Z=z).data
                matrix = np.eye(3, 3)
                matrix[:2, -1] = shifts.loc[t.item(), z.item()][["DY", "DX"]]
                registered = ndi.affine_transform(moving, matrix)
                stabilized.loc[{"T": t, "C": c, "Z": z}] = registered

    # Convert shift to micron
    for dim, col in product(["X", "Y"], ["D", "V"]):
        s = data[dim][1].item() - data[dim][0].item()
        shifts[f"{col}{dim} [um]"] = shifts[f"{col}{dim}"] * s
        shifts[f"{col}{dim} [um]"] = shifts[f"{col}{dim}"] * s

    return shifts, stabilized
