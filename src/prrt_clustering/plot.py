from matplotlib import pyplot as plt
import xarray as xr
import numpy as np


def plot_data(data: xr.DataArray):
    """Display the image data.

    Parameters
    ----------
    data:xr.DataArray
        Image array
    """
    extent = (
        float(data["X"][0].values),
        float(data["X"][-1].values),
        float(data["Y"][-1].values),
        float(data["Y"][0].values),
    )
    if "C" in data.sizes:
        rgb = np.zeros((data.sizes["Y"], data.sizes["X"], 3))
        cmap = np.array([[0.5, 0, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(
            4, 1, 3
        )
        for c in range(data.sizes["C"]):
            plane = np.array(data.isel(C=c).max("Z")).astype(float)
            a, b = plane.min(), plane.max()
            plane = (plane - a) / (b - a)
            rgb = np.maximum(rgb, np.expand_dims(plane, 2) @ cmap[c])
        plt.imshow(rgb, extent=extent)
    else:
        plt.imshow(data.max("Z"), extent=extent, cmap="gray")
    plt.axis("off")


def plot_points(points: xr.Dataset, field="coordinates", ms=40):
    """Display points.

    Parameters
    ----------
    points: list[dict]
        List of coordinates for each channel
    """
    if points is not None:
        cmap = np.array([[1, 0.1, 1], [1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 1]])

        for k, name_pts in enumerate(points.groupby("C")):
            name, pts = name_pts
            plt.plot(
                pts["X"],
                pts["Y"],
                ".",
                color=cmap[k],
                mfc="none",
                ms=ms,
                alpha=0.7,
            )
