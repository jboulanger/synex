"""Plotting and visualization utilities for microscopy images and puncta."""

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt


def plot_data(data: xr.DataArray, split: bool = True, fig=None, axes=None):
    """Display the image data.

    Parameters
    ----------
    data : xr.DataArray
        Image array.
    split : bool, optional
        Split each channel into a different axes, by default True.
    fig: matplotlib.Figure, optional
        Existing figure to reuse.
    axes : matplotlib.Axes or array of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig, axes
        The matplotlib figure and axes objects.
    """
    extent = (
        float(data["X"][0].values),
        float(data["X"][-1].values),
        float(data["Y"][-1].values),
        float(data["Y"][0].values),
    )

    cmap = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(4, 1, 3)
    if fig is None:
        fig = plt.figure()

    if "C" in data.sizes:
        groups = list(data.groupby("C"))
        n_groups = len(groups)

        # Set up axes based on the split parameter or provided axes
        if axes is None:
            if split:
                axes = fig.subplots(1, n_groups, squeeze=False)
                axes = axes[0]  # Flatten 2D array to 1D
            else:
                ax = fig.subplots()
                axes = [ax] * n_groups  # Broadcast single axis
        else:
            if not isinstance(axes, (list, np.ndarray, tuple)):
                axes = [axes] * n_groups if not split else [axes]
            fig = axes[0].figure

        # If split display each channel
        if split:
            for k, (name, sub_data) in enumerate(groups):
                plane = np.array(sub_data.max("Z")).astype(float).squeeze()
                a, b = plane.min(), plane.max()
                if b > a:
                    plane = (plane - a) / (b - a)
                axes[k].imshow(plane, extent=extent, cmap="gray")
                axes[k].set(title=name)
        else:
            rgb = np.zeros((data.sizes["Y"], data.sizes["X"], 3))
            for k, (_, sub_data) in enumerate(groups):
                plane = np.array(sub_data.max("Z")).astype(float).squeeze()
                a, b = plane.min(), plane.max()
                if b > a:
                    plane = (plane - a) / (b - a)
                rgb = np.maximum(rgb, np.expand_dims(plane, 2) @ cmap[k % len(cmap)])

            axes[0].imshow(rgb, extent=extent)
    else:
        # Only one channel
        if axes is None:
            ax = fig.subplots()
            axes = [ax]
        else:
            ax = axes[0] if isinstance(axes, (list, np.ndarray, tuple)) else axes
            axes = [ax]

        plane = np.array(data.max("Z")).astype(float)
        axes[0].imshow(plane, extent=extent, cmap="gray")

    for ax in axes:
        ax.axis("off")

    if split:
        fig.tight_layout()

    return fig, axes


def plot_points(points: pd.DataFrame, ms: float = 40, split: bool = False, axes=None):
    """Display points.

    Parameters
    ----------
    points : pd.DataFrame
        Table or dataset of coordinates (X, Y, Z, C) for each point.
    ms : float, optional
        Marker size passed to plt.plot, by default 40.
    split : bool, optional
        Split each channel into a different axes, by default False.
    axes : matplotlib.Axes or array of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig, axes
        The matplotlib figure and axes objects.
    """
    if points is None:
        return None

    if split:
        cmap = np.array([[1, 0, 0]] * 4)
    else:
        cmap = np.array([[1, 0.1, 1], [1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 1]])

    groups = list(points.groupby("index", sort=False))
    n_groups = len(groups)

    # Set up axes based on the split parameter or provided axes
    if axes is None:
        if split:
            fig, axes = plt.subplots(1, n_groups, squeeze=False)
            axes = axes[0]  # Flatten 2D array to 1D
        else:
            fig, ax = plt.subplots()
            axes = [ax] * n_groups  # Broadcast single axis
    else:
        if not isinstance(axes, (list, np.ndarray, tuple)):
            axes = [axes] * n_groups if not split else [axes]
        if len(axes) < n_groups:
            raise ValueError(
                f"Number of provided axes {len(axes)} does not match number "
                f"of channel groups ({n_groups})."
            )
        fig = axes[0].figure

    # Unified plotting loop
    for k, pts in groups:
        channel_index = int(np.asarray(k).item())
        axes[channel_index].plot(
            pts["X"],
            pts["Y"],
            ".",
            color=cmap[channel_index % len(cmap)],
            mfc="none",
            ms=ms,
            alpha=0.7,
        )

    if split:
        fig.tight_layout()

    return fig, axes


def plot_clusters(points: pd.DataFrame, ms: float = 40, split: bool = False, axes=None):
    """Display points.

    Parameters
    ----------
    points : pd.DataFrame
        Table or dataset of coordinates (X, Y, Z, C,cluster) for each point.
    ms : float, optional
        Marker size passed to plt.plot, by default 40.
    split : bool, optional
        Split each channel into a different axes, by default False.
    axes : matplotlib.Axes or array of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig, axes
        The matplotlib figure and axes objects.
    """
    if points is None:
        return None

    if split:
        cmap = np.array([[1, 0, 0]] * 4)
    else:
        cmap = np.array([[1, 0.1, 1], [1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 1]])

    groups = list(points.groupby("index", sort=False))
    n_groups = len(groups)

    # Set up axes based on the split parameter or provided axes
    if axes is None:
        if split:
            fig, axes = plt.subplots(1, n_groups, squeeze=False)
            axes = axes[0]  # Flatten 2D array to 1D
        else:
            fig, ax = plt.subplots()
            axes = [ax] * n_groups  # Broadcast single axis
    else:
        if not isinstance(axes, (list, np.ndarray, tuple)):
            axes = [axes] * n_groups if not split else [axes]
        if len(axes) < n_groups:
            raise ValueError(
                f"Number of provided axes {len(axes)} does not match number "
                f"of channel groups ({n_groups})."
            )
        fig = axes[0].figure

    # Unified plotting loop
    for k, pts in groups:
        channel_index = int(np.asarray(k).item())
        axes[channel_index].scatter(
            pts["X"],
            pts["Y"],
            c=pts["cluster"],
            # color=cmap[channel_index % len(cmap)],
            # mfc="none",
            s=ms,
            alpha=0.5,
        )

    if split:
        fig.tight_layout()

    return fig, axes
