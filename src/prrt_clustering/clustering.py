"""Clustering of labeled points."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gamma


def compute_nearest_neighbor(points: pd.DataFrame, eps: float, min_samples: int = 5):
    """Clean point cloud using nearest neighbors.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns X, Y, Z, C, intensity
    """
    nnbors = NearestNeighbors(metric="euclidean")
    clean = []
    for n, p in points.groupby("C"):
        X = p[["Z", "Y", "X"]].to_numpy()
        nnbors.fit(X)
        distance, indices = nnbors.kneighbors(X, 10)
        num = np.sum(distance[:, 1:] < eps, axis=1)
        clean.append(p[num >= min_samples])
        # import matplotlib.pyplot as plt

        # # plt.hist(distance)
        # plt.scatter(p["X"], p["Y"], s=10, c=np.mean(distance[:, 1:], 1))
        # plt.show()
    return pd.concat(clean)


def cluster(points: pd.DataFrame, eps=2, min_samples=5):
    """Cluster points for each channel C.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns: X, Y, Z, C and intensity
    """
    if points is None or points.empty:
        return None

    model = OPTICS(max_eps=eps, min_samples=min_samples)
    clusters = []

    for c, p in points.groupby("C"):
        coords = p[["Z", "Y", "X"]].to_numpy()
        labels = model.fit_predict(coords)
        mask = labels >= 0
        if np.any(mask):
            clustered = p.loc[mask].copy()
            clustered["class"] = labels[mask]
            clusters.append(clustered)

    if not clusters:
        return None

    return pd.concat(clusters, ignore_index=True)
