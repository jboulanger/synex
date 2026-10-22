"""Drift correction."""

import numpy as np
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
        A 1D translation vector [dy, dx] that registers the moving image.
    """
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    # Ensure floating-point precision
    ref = reference.astype(np.float32, copy=False)
    mov = moving.astype(np.float32, copy=False)

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


def correct_z_drift(data: xr.DataArray) -> tuple[list[dict], xr.DataArray]:
    """Correct the drift across the Z-axis.

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
    matrix = np.eye(3, 3)
    z0 = data["Z"][0]
    for t in data["T"]:
        for c in data["C"]:
            shifts.append({"T": t, "C": c, "Z": z0, "shift": np.array([0.0, 0.0])})
            stabilized.loc[{"T": t, "C": c, "Z": z0}] = data.sel(T=t, C=c, Z=z0)
            sv = np.array([0.0, 0.0])
            reference = data.sel(T=t, C=c, Z=z0).data
            for z in data["Z"][1:]:
                moving = data.sel(T=t, C=c, Z=z).data
                v = estimate_translation(reference, moving)
                sv += v
                shifts.append(
                    {"T": t, "C": c, "Z": z, "shift": v, "cumulated shifts": sv}
                )
                # register
                matrix[:2, -1] = sv
                registered = ndi.affine_transform(moving, matrix)
                stabilized.loc[{"T": t, "C": c, "Z": z}] = registered
                # swap planes
                reference, moving = moving, reference

    return shifts, stabilized
