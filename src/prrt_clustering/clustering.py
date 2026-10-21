"""Clustering of labeled points."""

import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from sklearn.neighbors import NearestNeighbors


def compute_nearest_neighbor(points):
    """Compute nearest neighbor.

    Parameters
    ----------
    points: xr.Dataset
        Dataset with coordinates X,Y,Z and data_vars:intensity
    """
    nn = NearestNeighbors()
    for n, p in points.groupby("C"):
        X = np.stack((p["Z"], p["Y"], p["X"])).T
        distance, indices = nn.kneighbors(X)


def cluster(points, eps=2, min_samples=5):
    """Cluster points for each channel C.

    Parameters
    ----------
    points: xr.Dataset
        Dataset with coordinates: X,Y,Z,C and data_vars:intensity
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    # model = MeanShift(bandwidth=eps)
    # model = OPTICS(min_samples=min_samples, max_eps=eps)
    new_points = []
    for n, p in points.groupby("C"):
        X = np.stack((p["Z"], p["Y"], p["X"])).T
        y = model.fit_predict(X)
        coords = X[y >= 0, :]
        num_points = coords.shape[0]
        if num_points > 0:
            new_points.append(
                xr.Dataset(
                    data_vars={
                        "intensity": ("point", np.ones(num_points)),
                        "class": ("point", y[y >= 0]),
                    },
                    coords={
                        # "point": np.arange(num_points),
                        "X": ("point", coords[:, 2]),
                        "Y": ("point", coords[:, 1]),
                        "Z": ("point", coords[:, 0]),
                        "C": ("point", [n] * coords.shape[0]),
                    },
                )
            )
    # Compute cluster parameters
    centroids = []
    for channel, pchannel in new_points.groupby("C"):
        for cls, pcls in pchannel.groupby("class"):
            X = np.stack((pcls["Z"], pcls["Y"], pcls["X"])).T
            y = np.mean.axis(0)

            centroid.append(
                xr.Dataset(
                    data_vars={
                        "intensity": ("point", np.ones(num_points)),
                        "class": ("point", [cls] * num_points),
                    },
                    coords={
                        # "point": np.arange(num_points),
                        "X": ("point", coords[:, 2]),
                        "Y": ("point", coords[:, 1]),
                        "Z": ("point", coords[:, 0]),
                        "C": ("point", [n] * coords.shape[0]),
                    },
                )
            )

    if len(new_points) > 0:
        return xr.concat(new_points, dim="point")
    else:
        return None
