"""Drift correction."""

from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi


import numpy as np
import scipy.ndimage as ndi

import dask.array as da


def _cpd_e_step(
    ref_lm: np.ndarray,
    mov_lm: np.ndarray,
    t: np.ndarray,
    sigma2: float,
    outlier_weight: float = 0.2,
    image_area: float = 1e5,
) -> np.ndarray:
    """Compute soft correspondence matrix P of shape (N_ref, M_mov)."""
    # Shifted reference landmarks: (N, 2)
    transformed_ref = ref_lm + t

    # Pairwise squared Euclidean distances: (N, M)
    diff = mov_lm[None, :, :] - transformed_ref[:, None, :]
    dist2 = np.sum(diff**2, axis=-1)

    # Gaussian kernel
    P = np.exp(-dist2 / (2.0 * sigma2))

    # Uniform outlier noise term (prevents distant/unmatched points from forcing bad matches)
    N = len(ref_lm)
    if outlier_weight > 0.0:
        c = (
            (2.0 * np.pi * sigma2)
            * (outlier_weight / (1.0 - outlier_weight))
            * (N / image_area)
        )
    else:
        c = 1e-12

    denom = np.sum(P, axis=0, keepdims=True) + c
    P /= denom
    return P


def estimate_translation_lk_cpd(
    reference: np.ndarray,
    moving: np.ndarray,
    ref_landmarks: np.ndarray | None = None,
    mov_landmarks: np.ndarray | None = None,
    landmark_weight: float = 1.0,
    initial_sigma: float = 20.0,
    min_sigma: float = 1.0,
    anneal_rate: float = 0.85,
    outlier_weight: float = 0.2,
    max_iters: int = 30,
    tol: float = 1e-4,
) -> np.ndarray:
    """Joint Lucas-Kanade and Coherent Point Drift (CPD) translation estimator.

    Does NOT require 1-to-1 landmark correspondences or equal landmark counts.

    Parameters
    ----------
    reference : np.ndarray
        The reference (template) image (2D).
    moving : np.ndarray
        The moving image to align with the reference, same shape as reference.
    ref_landmarks : np.ndarray, shape (N, 2), optional
        Unordered landmark coordinates in reference image as [y, x].
    mov_landmarks : np.ndarray, shape (M, 2), optional
        Unordered landmark coordinates in moving image as [y, x].
    landmark_weight : float, default=1.0
        Weight (lambda) balancing landmark alignment against pixel intensity error.
    initial_sigma : float, default=20.0
        Initial Gaussian capture radius (in pixels) for soft landmark matching.
    min_sigma : float, default=1.0
        Minimum Gaussian radius floor (in pixels) as annealing progresses.
    anneal_rate : float, default=0.85
        Multiplicative decay applied to sigma each iteration.
    outlier_weight : float, default=0.2
        Expected fraction [0, 1) of landmarks without a true counterpart.
    max_iters : int, default=30
        Maximum joint Gauss-Newton / EM iterations.
    tol : float, default=1e-4
        Convergence tolerance on step norm ||delta_t||.

    Returns
    -------
    np.ndarray
        A 1D translation vector [dy, dx] in pixels.
    """
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    ref = ndi.gaussian_filter(np.asarray(reference, dtype=np.float64), 1.0)
    mov = ndi.gaussian_filter(np.asarray(moving, dtype=np.float64), 1.0)
    image_area = float(ref.shape[0] * ref.shape[1])

    has_landmarks = (
        ref_landmarks is not None
        and mov_landmarks is not None
        and len(ref_landmarks) > 0
        and len(mov_landmarks) > 0
    )

    if has_landmarks:
        ref_lm = np.asarray(ref_landmarks, dtype=np.float64)
        mov_lm = np.asarray(mov_landmarks, dtype=np.float64)
        if (
            ref_lm.ndim != 2
            or ref_lm.shape[1] != 2
            or mov_lm.ndim != 2
            or mov_lm.shape[1] != 2
        ):
            raise ValueError("Landmarks must have shape (N, 2) and (M, 2) as [y, x].")

        # Initialize translation using centroid difference of point clouds
        t = mov_lm.mean(axis=0) - ref_lm.mean(axis=0)
        sigma2 = float(initial_sigma**2)
        min_sigma2 = float(min_sigma**2)
    else:
        t = np.zeros(2, dtype=np.float64)

    # Precompute image gradients and Hessian
    I1, I2 = np.gradient(ref)
    H_img = np.array(
        [
            [np.square(I1).mean(), (I1 * I2).mean()],
            [(I1 * I2).mean(), np.square(I2).mean()],
        ],
        dtype=np.float64,
    )

    for _ in range(max_iters):
        mov_warped = ndi.shift(mov, shift=-t, order=1, mode="nearest")

        # Valid interior mask
        valid = np.ones_like(ref, dtype=bool)
        dy_int, dx_int = int(np.ceil(abs(t[0]))), int(np.ceil(abs(t[1])))
        if dy_int > 0:
            valid[:dy_int, :] = False
            valid[-dy_int:, :] = False
        if dx_int > 0:
            valid[:, :dx_int] = False
            valid[:, -dx_int:] = False
        if not np.any(valid):
            break

        I0 = mov_warped - ref
        B = np.array(
            [
                -(I1[valid] * I0[valid]).mean(),
                -(I2[valid] * I0[valid]).mean(),
            ],
            dtype=np.float64,
        )
        H = H_img.copy()

        if has_landmarks:
            # 1. E-step: compute soft correspondence probabilities P of shape (N, M)
            P = _cpd_e_step(
                ref_lm=ref_lm,
                mov_lm=mov_lm,
                t=t,
                sigma2=sigma2,
                outlier_weight=outlier_weight,
                image_area=image_area,
            )

            M = len(mov_lm)
            # Total inlier assignment mass normalized by M
            Np_norm = np.sum(P) / M

            # Probability-weighted mean displacement vector
            # Sum_{n,m} P_{nm} * (mov_m - ref_n - t) / M
            weighted_mov = P.T @ ref_lm  # (M, 2) -> sum_n P_{nm} * ref_n
            col_sums = np.sum(P, axis=0)  # (M,)
            expected_residual = (
                np.sum(col_sums[:, None] * (mov_lm - t) - weighted_mov, axis=0) / M
            )

            H += landmark_weight * Np_norm * np.eye(2)
            B += landmark_weight * expected_residual

            # Anneal sigma^2 for sharper matching on the next iteration
            sigma2 = max(min_sigma2, sigma2 * (anneal_rate**2))

        delta_t, _, _, _ = np.linalg.lstsq(H, B, rcond=None)
        t += delta_t

        if np.linalg.norm(delta_t) < tol and (
            not has_landmarks or sigma2 <= min_sigma2 * 1.05
        ):
            break

    return t


def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    scale: float = 1.5,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> np.ndarray:
    """Estimate the global translation between two images.

    Uses an iterative Inverse-Compositional Lucas-Kanade optical flow method.

    Parameters
    ----------
    reference : np.ndarray
        The reference (template) image (2D).
    moving : np.ndarray
        The moving image to align with the reference, same shape as reference.
    scale : float, default=1.0
        Standard deviation for Gaussian smoothing before gradient computation.
    max_iters : int, default=20
        Maximum number of Gauss-Newton iterations.
    tol : float, default=1e-4
        Convergence threshold on the update step norm (in pixels).

    Returns
    -------
    np.ndarray
        A 1D translation vector [dy, dx] in pixels that registers the moving image.
    """
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving images must have the same shape.")

    # Ensure floating-point precision
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)

    # Filtering
    if scale > 0:
        ref = ndi.gaussian_filter(ref, scale)
        mov = ndi.gaussian_filter(mov, scale)

    # Spatial gradients on reference (constant across iterations)
    I1, I2 = np.gradient(ref)

    # Precompute structure tensor components on the full image as fallback
    # (Recomputed over valid interior pixels inside the loop when shifted)
    translation = np.zeros(2, dtype=np.float64)

    for _ in range(max_iters):
        # Warp moving image toward reference using current estimate
        # Note: shift=-translation samples moving at (y + dy, x + dx)
        mov_warped = ndi.shift(mov, shift=-translation, order=1, mode="nearest")

        # Mask out border pixels pulled in from outside the image bounds
        valid = np.ones_like(ref, dtype=bool)
        dy_pad = int(np.ceil(abs(translation[0])))
        dx_pad = int(np.ceil(abs(translation[1])))

        if dy_pad > 0:
            valid[:dy_pad, :] = False
            valid[-dy_pad:, :] = False
        if dx_pad > 0:
            valid[:, :dx_pad] = False
            valid[:, -dx_pad:] = False

        # Stop if shift exceeds image dimensions
        if not np.any(valid):
            break

        # Temporal gradient on valid overlapping region
        I0 = mov_warped[valid] - ref[valid]
        I1_v = I1[valid]
        I2_v = I2[valid]

        # Structure tensor components (mean over valid pixels)
        a = np.square(I1_v).mean()
        b = (I1_v * I2_v).mean()
        c = np.square(I2_v).mean()
        d = -(I1_v * I0).mean()
        e = -(I2_v * I0).mean()

        # Construct system matrix and vector
        A = np.array([[a, b], [b, c]], dtype=np.float64)
        B = np.array([d, e], dtype=np.float64)

        # Solve for incremental update
        delta, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        translation += delta

        if np.linalg.norm(delta) < tol:
            break

    return translation


def estimate_translation_old(
    reference: np.ndarray, moving: np.ndarray, scale: float = 1
) -> np.ndarray:
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
    ref = ndi.gaussian_filter(ref, scale)
    mov = ndi.gaussian_filter(mov, scale)

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
        if isinstance(reference, da.Array):
            # reference.compute_chunk_sizes()
            reference = reference.compute()
        for z in data["Z"][1:]:
            moving = data.sel(T=t, Z=z).mean("C").data
            if isinstance(moving, da.Array):
                # moving.compute_chunk_sizes()
                moving = moving.compute()
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
