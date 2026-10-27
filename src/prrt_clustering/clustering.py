"""Clustering of labeled points."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_synthetic_point_cloud(
    D: int = 4,
    H: int = 20,
    W: int = 20,
    N: int = 50,
    K: int = 10,
    M: int = 5000,
    C: int = 4,
):
    """Create a synthetic point clouds.

    Parameters
    ----------
    D: int
        Depth
    H: int
        Height
    W: int
        Width
    N: int
        Number of clusters
    K: int
        Number of point per cluster
    M: int
        Number of noise points
    C: int
        Number of channels
    """
    clst = np.random.uniform(low=[0, 0, 0, 0.5], high=[D, H, W, 1], size=(N, 4))

    points = pd.concat(
        [
            pd.DataFrame(
                np.vstack(
                    [
                        *[
                            np.random.normal(loc=x, scale=0.1, size=(K, 4))
                            for x in clst
                        ],
                        np.random.uniform(
                            low=[0, 0, 0, 0.5], high=[D, H, W, 1], size=(M, 4)
                        ),
                    ]
                ),
                columns=["Z", "Y", "X", "intensity"],
            )
            .assign(C=f"CH-{channel}")
            .assign(noise=np.arange(N * K + M) > N * K)
            for channel in range(C)
        ]
    )
    clusters = pd.DataFrame(
        clst,
        columns=["Z", "Y", "X", "intensity"],
    ).assign(C="cluster")

    return points, clusters


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


class BesagFeature(BaseEstimator, TransformerMixin):
    """Transformer computing 3D local Besag's L-function features.

    Computes neighbor counts within radial balls and compares them to Complete
    Spatial Randomness (CSR) in 3D:
        L_i(r) = ( (3 * N_i(r)) / (4 * pi * lambda) )^(1/3) - r
    where lambda is the estimated point density across the bounding volume.

    Parameters
    ----------
    max_radius : float, default=1.0
        Maximum radial search distance for neighbor queries.
    num_radius : int, default=10
        Number of radial evaluation points linearly spaced in (0, max_radius].
    metric : str, default="euclidean"
        Distance metric for nearest neighbor queries.
    """

    def __init__(
        self,
        max_radius: float = 1.0,
        num_radius: int = 10,
        metric: str = "euclidean",
    ):
        self.max_radius = max_radius
        self.num_radius = num_radius
        self.metric = metric

    def fit(self, X, y=None):
        """Fit the spatial radius index and estimate global point density.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training coordinate points (e.g. Z, Y, X).
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : CSRFeature
            Fitted transformer instance.
        """
        X = np.asarray(X)
        self.radii_ = np.linspace(
            self.max_radius / self.num_radius, self.max_radius, self.num_radius
        )
        self.nn_ = NearestNeighbors(radius=self.max_radius, metric=self.metric)
        self.nn_.fit(X)

        # Estimate global density lambda = N / V
        # Use bounding box volume with guard against 0 volume
        ranges = np.ptp(X, axis=0)
        volume = float(np.prod(np.maximum(ranges, 1e-6)))
        self.lambda_ = max(len(X) / volume, 1e-12)
        return self

    def transform(self, X):
        """Transform coordinate points into 3D Besag L-function features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query coordinate points.

        Returns
        -------
        ndarray of shape (n_samples, num_radius)
            Besag L-function values L_i(r) across radii.
        """
        X = np.asarray(X)
        # Query all neighbor distances within max_radius
        distances_list = self.nn_.radius_neighbors(X, return_distance=True)[0]

        # Vectorized counting of neighbors within each radius r
        # (excluding point itself at distance 0)
        counts = np.empty((len(X), len(self.radii_)), dtype=np.float64)
        for i, dists in enumerate(distances_list):
            if len(dists) <= 1:
                counts[i, :] = 0.0
            else:
                # Exclude self-distance (distance ~ 0)
                dists_no_self = dists[dists > 1e-12]
                counts[i, :] = np.searchsorted(
                    np.sort(dists_no_self), self.radii_, side="right"
                )

        # 3D Besag L-function: (3 * N_i(r) / (4 * pi * lambda))^(1/3) - r
        # Under CSR, expected value is 0. Clustered points produce L(r) > 0.
        l_r = np.cbrt((3.0 * counts) / (4.0 * np.pi * self.lambda_)) - self.radii_
        return l_r


def clean_points_nn_gmm(points: pd.DataFrame, n_neighbors=10):
    """Clean point cloud using nearest neighbors.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns X, Y, Z, C, intensity

    Returns
    -------
    pd.DataFrame
        Copy of points with an added boolean 'noise' column.
    """
    clean = []
    import matplotlib.pyplot as plt

    pipeline = Pipeline(
        [
            ("nn_features", NNDiffFeatureExtractor(n_neighbors)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            (
                "gmm",
                OrderedGaussianMixture(
                    n_components=2, covariance_type="full", random_state=42
                ),
            ),
        ]
    )

    for _, p in points.groupby("C"):
        p = p.copy()
        X = p[["Z", "Y", "X"]].to_numpy()
        labels = pipeline.fit_predict(X)
        p["noise"] = labels == 0
        clean.append(p)
        # Visualization
        Z = pipeline.named_steps["nn_features"].transform(X)
        label_keep = 0
        plt.plot(Z[labels == label_keep, 0], Z[labels == label_keep, 1], "g.")
        plt.plot(Z[labels != label_keep, 0], Z[labels != label_keep, 1], "r.")
        plt.show()
    return pd.concat(clean)


def clean_points_nn_if(points: pd.DataFrame, contamination=0.1):
    """Clean point cloud using nearest neighbors and IsolationForest.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns X, Y, Z, C, intensity

    Returns
    -------
    pd.DataFrame
        Copy of points with an added boolean 'noise' column.
    """
    clean = []
    # import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest

    pipeline = Pipeline(
        [
            ("nn_features", NNDiffFeatureExtractor(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            (
                "isolation_forest",
                IsolationForest(contamination=contamination),
            ),
        ]
    )
    # pipeline = IsolationForest(contamination=0.001)

    for _, p in points.groupby("C"):
        p = p.copy()
        X = p[["Z", "Y", "X"]].to_numpy()
        labels = pipeline.fit_predict(X)
        p["noise"] = labels == 1
        clean.append(p)
        # Visualization
        # Z = pipeline.named_steps["nn_features"].transform(X)
        # plt.plot(Z[labels == label_keep, 0], Z[labels == label_keep, 1], "g.")
        # plt.plot(Z[labels != label_keep, 0], Z[labels != label_keep, 1], "r.")
    # plt.show()
    return pd.concat(clean)


def clean_points_besag_gmm(
    points: pd.DataFrame, max_radius: float = 1.5, num_radius: int = 11
):
    """Clean point cloud using besag features.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns X, Y, Z, C, intensity

    Returns
    -------
    pd.DataFrame
        Copy of points with an added boolean 'noise' column.
    """
    clean = []
    import matplotlib.pyplot as plt

    pipeline = Pipeline(
        [
            ("csr_features", BesagFeature(max_radius, num_radius)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
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
        labels = pipeline.fit_predict(X)
        p["noise"] = labels == 0
        clean.append(p)
        # Visualization

        # Z = pipeline.named_steps["csr_features"].transform(X)
        # plt.imshow(
        #     Z, aspect="auto", interpolation="none", extent=(0, 0.5, Z.shape[0], 0)
        # )
        # plt.plot(Z[labels == label_keep, 0], Z[labels == label_keep, 1], "g.")
        # plt.plot(Z[labels != label_keep, 0], Z[labels != label_keep, 1], "r.")
    plt.show()
    return pd.concat(clean)


def cluster(points: pd.DataFrame, eps: float = 2.0, min_samples: int = 5):
    """Cluster points for each channel C using DBSCAN.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns: X, Y, Z, C and intensity
    """
    if points is None or points.empty:
        return None

    model = DBSCAN(eps=eps, min_samples=min_samples)
    new_points = []

    for _, p in points.groupby("C"):
        X = p[["Z", "Y", "X"]].to_numpy()
        labels = model.fit_predict(X)
        p["cluster"] = labels
        p["noise"] = labels == -1
        new_points.append(p)

    if not new_points:
        return None, None

    new_points = pd.concat(new_points, ignore_index=True)

    clusters = new_points.groupby(["C", "cluster"]).mean()

    return new_points, clusters
