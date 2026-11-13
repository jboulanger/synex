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

        axes_dict = {n: axes[k] for k, (n, _) in enumerate(groups)}

        # If split display each channel
        if split:
            for name, sub_data in groups:
                plane = np.array(sub_data.max("Z")).astype(float).squeeze()
                a, b = plane.min(), plane.max()
                if b > a:
                    plane = (plane - a) / (b - a)
                axes_dict[name].imshow(plane, extent=extent, cmap="gray")
                axes_dict[name].set(title=name)
        else:
            rgb = np.zeros((data.sizes["Y"], data.sizes["X"], 3))
            for k, (name, sub_data) in enumerate(groups):
                plane = np.array(sub_data.max("Z")).astype(float).squeeze()
                a, b = plane.min(), plane.max()
                if b > a:
                    plane = (plane - a) / (b - a)
                rgb = np.maximum(rgb, np.expand_dims(plane, 2) @ cmap[k % len(cmap)])

            axes_dict[name].imshow(rgb, extent=extent)
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

    return fig, axes_dict


def plot_points(
    points: pd.DataFrame, ms: float = 40, split: bool = False, axes_dict=None
):
    """Display points.

    Parameters
    ----------
    points : pd.DataFrame
        Table or dataset of coordinates (X, Y, Z, C) for each point.
    ms : float, optional
        Marker size passed to plt.plot, by default 40.
    split : bool, optional
        Split each channel into a different axes, by default False.
    axes_dict : dict(str,matplotlib.Axes), optional
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
    groups = list(points.groupby("C"))

    cmap_dict = {n: cmap[k % len(cmap)] for k, (n, _) in enumerate(groups)}

    # Set up axes based on the split parameter or provided axes
    if axes_dict is None:
        n_groups = len(groups)
        if split:
            fig, axes = plt.subplots(1, n_groups, squeeze=False)
            axes = axes[0]  # Flatten 2D array to 1D
        else:
            fig, ax = plt.subplots()
            axes = [ax] * n_groups  # Broadcast single axis
        axes_dict = {n: axes[k] for k, (n, _) in enumerate(groups)}

    # Unified plotting loop
    for name, pts in groups:
        if name in axes_dict.keys():
            axes_dict[name].plot(
                pts["X"],
                pts["Y"],
                ".",
                color=cmap_dict[name],
                mfc="none",
                ms=ms,
                alpha=0.7,
            )

    fig = axes_dict[list(axes_dict.keys())[0]].figure

    if split:
        fig.tight_layout()

    return fig, axes_dict


def plot_drift_correction(data, stabilized, shifts):
    """Plot drift correction."""
    plt.figure()
    plt.subplot(311)
    plt.plot(shifts.reset_index()["Z"], shifts["DX [um]"])
    plt.plot(shifts.reset_index()["Z"], shifts["DY [um]"])
    plt.legend(["X", "Y"])
    plt.xlabel("Z [um]")
    plt.ylabel("shift [um]")
    n = 500
    a = data[0, 2, :, :n, :n].max(axis=1)
    b = stabilized[0, 2, :, :n, :n].max(axis=1)
    c = data[0, 2, :, :n, :n].max(axis=2)
    d = stabilized[0, 2, :, :n, :n].max(axis=2)
    plt.subplot(323)
    a.plot(add_colorbar=False, cmap="gray")
    plt.subplot(324)
    c.plot(add_colorbar=False, cmap="gray")
    plt.subplot(325)
    b.plot(add_colorbar=False, cmap="gray")
    plt.subplot(326)
    c.plot(add_colorbar=False, cmap="gray")
    plt.suptitle("Z drift correction")
    plt.tight_layout()


def plot_clusters(
    points: pd.DataFrame, ms: float = 40, split: bool = False, axes_dict: dict = {}
):
    """Display points.

    Parameters
    ----------
    points : pd.DataFrame
        Table or dataset of coordinates (X, Y, Z, C,cluster) for each point.
    ms : float, optional
        Marker size passed to plt.plot, by default 40.
    split : bool, optional
        Split each channel into a different axes, by default False.
    axes : dictmatplotlib.Axes or array of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig, axes
        The matplotlib figure and axes objects.
    """
    if points is None:
        return None

    # n_groups = len(groups)

    # # Set up axes based on the split parameter or provided axes
    # if axes is None:
    #     if split:
    #         fig, axes = plt.subplots(1, n_groups, squeeze=False)
    #         axes = axes[0]  # Flatten 2D array to 1D
    #     else:
    #         fig, ax = plt.subplots()
    #         axes = [ax] * n_groups  # Broadcast single axis
    # else:
    #     if not isinstance(axes, (list, np.ndarray, tuple)):
    #         axes = [axes] * n_groups if not split else [axes]
    #     if len(axes) < n_groups:
    #         raise ValueError(
    #             f"Number of provided axes {len(axes)} does not match number "
    #             f"of channel groups ({n_groups})."
    #         )
    #     fig = axes[0].figure

    # Unified plotting loop
    for n, pts in points.groupby("C"):
        axes_dict[n].scatter(
            pts["X"],
            pts["Y"],
            c=pts["cluster"],
            # color=cmap[channel_index % len(cmap)],
            # mfc="none",
            s=ms,
            alpha=0.5,
        )

    fig = axes_dict[list(axes_dict.keys())[0]].figure

    if split:
        fig.tight_layout()

    return fig, axes_dict


def bagel_chart(
    data: dict,
    center_text: str | None = None,
    colors: np.ndarray | None = None,
    ax=None,
):
    """Create a compact, modern bagel chart with tightly aligned callout labels.

    Parameters
    ----------
    data : dict
        Dictionary with key-value pairs to plot.
    center_text : str, optional
        Text to display in the center hole of the bagel.
    colors : np.ndarray, optional
        Custom color palette for the wedges.
    ax: axes, optional
        Optional axes
    """
    values = list(data.values())
    total = sum(values)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))

    wedges, texts = ax.pie(
        values,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        startangle=90,
        colors=colors,
    )

    bbox_props = dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="lightgray",
        lw=1,  # shadow=True
    )
    kw = dict(
        arrowprops=dict(
            arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90", lw=1.1
        ),
        bbox=bbox_props,
        zorder=3,
        va="center",
    )

    annotations = [f"{k}\n{val} ({100 * val / total:.1f}%)" for k, val in data.items()]

    for k, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        # 2. Reduced multiplier (1.15 / 1.25) to bring labels closer to the chart
        ax.annotate(
            annotations[k],
            xy=(x, y),
            xytext=(1.1 * np.sign(x), 1.1 * y),
            horizontalalignment=horizontalalignment,
            multialignment="center",  # Keeps lines centered relative to each other
            fontsize=11,
            weight="semibold",
            **kw,
        )

    if center_text:
        ax.text(
            0,
            0,
            center_text,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="#333333",
        )


def make_the_bagel_plot(gst: dict, sst: pd.DataFrame):
    """Create a bagel plot."""
    data1 = dict()
    data1["synpatic"] = gst["Number of PRRT+GluA1 synpases"]
    data1["extra synaptic"] = (
        gst["Number of PRRT+GluA1 total"] - gst["Number of PRRT+GluA1 synpases"]
    )

    data2 = dict()
    data2["PRRT1-containing"] = (sst["Number of PRRT puncta"] >= 0).sum()
    data2["PRRT1-free"] = (
        (sst["Number of PRRT puncta"] == 0) & (sst["Number of A1 puncta"] > 0)
    ).sum()
    fig, ax = plt.subplots(2, 1, figsize=(5, 5))
    bagel_chart(data1, "% Total \n GluA1+PRRT1", ax=ax[0])
    bagel_chart(data2, "% Total \n GluA1 synapses", ax=ax[1])


def make_the_histogram_distance(points):
    """Compute the histogram of distances."""
    import seaborn as sns

    dist_prrt_psd = points[(points["C"] == "PRRT") & (points["synapse_id"] >= 0)][
        "distance to PSD"
    ]

    # Create figure with publication-friendly dimensions
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # 1. Use Seaborn's histplot with kde=True to automatically add and scale the curve
    sns.histplot(
        dist_prrt_psd,
        bins=25,
        kde=True,
        color="#4c72b0",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        ax=ax,
        line_kws={
            "color": "#1b263b",
            "linewidth": 2.2,
        },  # Customizes the KDE line appearance
    )

    # 2. Emphasized threshold line
    threshold = 0.4
    ax.axvline(x=threshold, color="#9F9E9E", linestyle="--", linewidth=2)

    # 3. Add text annotation for the threshold line
    ax.text(
        threshold + 0.02,
        ax.get_ylim()[1] * 0.85,
        f"Cutoff: {threshold} μm",
        color="#9F9E9E",
        weight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9F9E9E", alpha=0.8),
    )

    # Axis limits and labels with clean typography
    ax.set_xlim(0, 1)
    ax.set_title(
        "PRRT1 Distance Distribution to PSD95", fontsize=13, weight="bold", pad=15
    )
    ax.set_xlabel("Distance to PSD95 [μm]", fontsize=11, weight="semibold")
    ax.set_ylabel("Number of Detected Synaptic PRRT1", fontsize=11, weight="semibold")

    # 4. Subtle background grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="gray")

    # 5. Minimalist styling: remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return dist_prrt_psd
