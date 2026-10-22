"""Clustering of labeled points."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_synthetic_point_cloud():
    """Create a synthetic point clouds."""
    D, H, W = 4, 20, 20
    clst = np.random.uniform(low=[0, 0, 0, 0.5], high=[D, H, W, 1], size=(50, 4))
    points = pd.concat(
        [
            pd.DataFrame(
                np.vstack(
                    [
                        *[
                            np.random.normal(loc=x, scale=0.1, size=(10, 4))
                            for x in clst
                        ],
                        np.random.uniform(
                            low=[0, 0, 0, 0.5], high=[D, H, W, 1], size=(1000, 4)
                        ),
                    ]
                ),
                columns=["Z", "Y", "X", "intensity"],
            )
            .assign(C=channel)
            .assign(label=np.arange(500 + 1000) < 500)
            for channel in "ABCD"
        ]
    )
    return points


class OrderedGaussianMixture(GaussianMixture):
    """Gaussian Mixture Model with sorted cluster labels.

    Sorts predicted component labels by the Euclidean norm of their cluster
    means in ascending order. Label 0 corresponds to the cluster with the
    smallest mean norm.
    """

    def fit_predict(self, X, y=None):
        """Fit model to X and predict sorted cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        ndarray of shape (n_samples,)
            Component labels sorted by cluster mean norm in ascending order.
        """
        labels = super().fit_predict(X, y)
        # Compute norms of the fitted GMM means_ (or sample means)
        norms = np.linalg.norm(self.means_, axis=1)
        order = np.argsort(norms)  # order[0] is the component with smallest norm

        # Remap labels so 0 is smallest norm, 1 is next, etc.
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        return rank[labels]


class NNDiffFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer computing square-root differences between sorted neighbor distances.

    Fits a nearest-neighbors index on input coordinates and transforms them
    into square-root differences of sequential neighbor distances.

    Parameters
    ----------
    n_neighbors : int, default=4
        Number of neighbors to query (including the point itself).
    metric : str, default="euclidean"
        Distance metric for nearest neighbor queries.
    """

    def __init__(self, n_neighbors: int = 4, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y=None):
        """Fit the NearestNeighbors estimator on the input coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training coordinate points.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : NNDiffFeatureExtractor
            Fitted transformer instance.
        """
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn_.fit(X)
        return self

    def transform(self, X):
        """Transform coordinate points into distance difference features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query coordinate points.

        Returns
        -------
        ndarray of shape (n_samples, n_neighbors - 2)
            Square root of differences between sequential nearest-neighbor distances.
        """
        # kneighbors returns (distances, indices)
        distances, _ = self.nn_.kneighbors(X)
        return np.sqrt(np.diff(distances, axis=1))


def clean_points(points: pd.DataFrame):
    """Clean point cloud using nearest neighbors.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns X, Y, Z, C, intensity
    """
    clean = []
    import matplotlib.pyplot as plt

    gmm_pipeline = Pipeline(
        [
            ("nn_features", NNDiffFeatureExtractor()),
            ("scaler", StandardScaler()),
            # Retain 95% of the data's variance
            ("pca", PCA(n_components=0.95, random_state=42)),
            # Fit the GMM on the reduced dimensions
            (
                "gmm",
                OrderedGaussianMixture(
                    n_components=2, covariance_type="full", random_state=42
                ),
            ),
        ]
    )

    for _, p in points.groupby("C"):
        X = p[["Z", "Y", "X"]].to_numpy()
        labels = gmm_pipeline.fit_predict(X)
        clean.append(p.iloc[labels == 1])
        # Visualization
        # Z = gmm_pipeline.named_steps["nn_features"].transform(X)
        # label_keep = 1
        # plt.plot(Z[labels == label_keep, 0], Z[labels == label_keep, 1], "g.")
        # plt.plot(Z[labels != label_keep, 0], Z[labels != label_keep, 1], "r.")
        # plt.show()
    return pd.concat(clean)


def cluster(points: pd.DataFrame, eps: float = 2.0, min_samples: int = 5):
    """Cluster points for each channel C using OPTICS.

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
