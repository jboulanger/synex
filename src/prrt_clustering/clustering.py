"""Clustering of labeled points."""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


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

    Returns
    -------
    pd.DataFrame: points
    pd.DataFrame: clusters
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
    """Transformer computing 3D local Besag's L-function weighted features.

    Uses the 4th column as weight.

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
            Training coordinate points (Z, Y, X, W) with W a weight.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : CSRFeature
            Fitted transformer instance.
        """
        # Compute spatial nearest neighbors.
        Xs = np.asarray(X)[:, :3]
        W = np.asarray(X)[:, -1]
        self.radii_ = np.linspace(
            self.max_radius / self.num_radius, self.max_radius, self.num_radius
        )
        self.nn_ = NearestNeighbors(radius=self.max_radius, metric=self.metric)
        self.nn_.fit(Xs)

        # Estimate global density lambda = N / V
        ranges = np.ptp(Xs, axis=0)
        volume = float(np.prod(np.maximum(ranges, 1e-6)))
        self.lambda_ = max(W.sum() / volume, 1e-12)
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
        Xs = np.asarray(X)[:, :3]
        W = np.asarray(X)[:, -1]

        # Query all neighbor distances within max_radius
        nn_dist, nn_ind = self.nn_.radius_neighbors(
            Xs, return_distance=True, sort_results=True
        )

        # Vectorized counting of neighbors within each radius r
        # (excluding point itself at distance 0)
        counts = np.empty((len(X), len(self.radii_)), dtype=np.float64)
        bins = np.concatenate([[0], self.radii_])
        for i, (dists, inds) in enumerate(zip(nn_dist, nn_ind)):
            if len(dists) <= 1:
                counts[i, :] = 0.0
            else:
                # Exclude self-distance (distance ~ 0) and self-weight
                dists_no_self = dists[1:]
                weights_no_self = W[inds[1:]]
                # Accumulate weighs into counts
                h = np.histogram(dists, bins, weights=W[inds])[0]
                counts[i, :] = np.cumsum(h)

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
    # import matplotlib.pyplot as plt

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
        # Z = pipeline.named_steps["nn_features"].transform(X)
        # label_keep = 0
        # plt.plot(Z[labels == label_keep, 0], Z[labels == label_keep, 1], "g.")
        # plt.plot(Z[labels != label_keep, 0], Z[labels != label_keep, 1], "r.")
        # plt.show()
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

    pipeline = Pipeline(
        [
            ("csr_features", BesagFeature(max_radius, num_radius)),
            ("scaler", RobustScaler()),
            # ("pca", PCA(n_components=2, random_state=42)),
            (
                "gmm",
                OrderedGaussianMixture(
                    n_components=2, covariance_type="full", random_state=42
                ),
            ),
        ]
    )

    for _, p in points.groupby("C"):
        X = p[["Z", "Y", "X", "intensity"]].to_numpy()
        labels = pipeline.fit_predict(X)
        p["noise"] = labels == 0
        clean.append(p)
        # Visualization
        # Z = pipeline.named_steps["csr_features"].transform(X)
        # Z = pipeline.named_steps["scaler"].transform(Z)
        # Z = pipeline.named_steps["pca"].transform(Z)
        # label_keep = 1
        # plt.imshow(
        #     Z, aspect="auto", interpolation="none", extent=(0, 0.5, Z.shape[0], 0)
        # )
        # plt.plot(Z[labels == label_keep, 0], Z[labels == label_keep, 1], "g.")
        # plt.plot(Z[labels != label_keep, 0], Z[labels != label_keep, 1], "r.")
        # plt.show()
    return pd.concat(clean)


def clean_points_besag(
    points: pd.DataFrame,
    max_radius: float = 1.5,
    num_radius: int = 11,
    threshold: float = 1,
):
    """Clean point cloud using by looking how much.

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

    pipeline = BesagFeature(max_radius, num_radius)

    for _, p in points.groupby("C", sort=False):
        X = p[["Z", "Y", "X", "intensity"]].to_numpy()
        labels = pipeline.fit_transform(X)
        # import matplotlib.pyplot as plt

        # plt.imshow(labels < threshold)
        # plt.show()
        p["noise"] = (labels < threshold).min(axis=1)
        clean.append(p)
    return pd.concat(clean)


def cluster_dbscan(points: pd.DataFrame, eps: float = 2.0, min_samples: int = 5):
    """Cluster points for each channel C using DBSCAN.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns: X, Y, Z, C and intensity
    """
    if points is None or points.empty:
        return None, None

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
    clusters = clusters[clusters["noise"] < 0.5]
    clusters = clusters.reset_index()
    clusters.drop(columns=["noise"], inplace=True)

    return new_points, clusters


def find_clusters(
    df, max_distance, threshold, x_col="X", y_col="Y", z_col="Z", v_col="V"
):
    """Find distance-connected clusters in a point cloud DataFrame where the sum of V exceeds threshold.

    Parameters
    ----------
    - df: pd.DataFrame containing the point cloud data
    - max_distance: float, maximum distance d for neighborhood connection (eps)
    - threshold: float, minimum sum of V required (t)
    - x_col, y_col, z_col, v_col: str, column names for coordinates and value

    Returns
    -------
    - valid_clusters: list of dicts containing cluster metadata
    - df_annotated: pd.DataFrame with an added 'cluster_id' column (-1 for noise/filtered out)
    """
    # 1. Extract coordinates and values as NumPy arrays for speed
    points_xyz = df[[x_col, y_col, z_col]].to_numpy()
    values = df[v_col].to_numpy()
    n_points = len(df)

    if n_points == 0:
        print("DataFrame is empty.")
        return [], df.copy()

    print("Building KD-Tree...")
    tree = KDTree(points_xyz)

    print("Finding neighboring point pairs...")
    pairs = tree.query_pairs(r=max_distance, output_type="set")

    if not pairs:
        print("No pairs found within max_distance.")
        df_out = df.copy()
        df_out["cluster_id"] = -1
        return [], df_out

    # Build adjacency structure for connected components
    row = np.array([p[0] for p in pairs])
    col = np.array([p[1] for p in pairs])
    data = np.ones(len(pairs))

    # Make symmetric
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data_sym = np.concatenate([data, data])

    graph = coo_matrix((data_sym, (row_sym, col_sym)), shape=(n_points, n_points))

    print("Computing connected components...")
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    # Evaluate clusters against threshold
    valid_clusters = []
    final_labels = np.full(n_points, -1, dtype=int)  # -1 represents filtered-out points
    valid_cluster_counter = 0

    for cluster_id in range(n_components):
        mask = labels == cluster_id
        cluster_sum_v = np.sum(values[mask])

        if cluster_sum_v > threshold:
            final_labels[mask] = valid_cluster_counter
            cluster_points = points_xyz[mask]

            valid_clusters.append(
                {
                    "valid_cluster_id": valid_cluster_counter,
                    "original_component_id": cluster_id,
                    "sum_v": cluster_sum_v,
                    "point_count": np.sum(mask),
                    "point_indices": np.where(mask)[0],
                    "centroid": np.mean(cluster_points, axis=0),
                }
            )
            valid_cluster_counter += 1

    # Add cluster labels back to a copy of the DataFrame
    df_annotated = df.copy()
    df_annotated["cluster_id"] = final_labels

    print(
        f"Found {len(valid_clusters)} valid clusters out of {n_components} total components."
    )
    return valid_clusters, df_annotated


def pair_cluster(points: pd.DataFrame, channels: list[str], distance: float):
    """Pair clusters that are less than the distance away.

    Parameters
    ----------
    points: pd.Dataframe
        Dataframe with X,Y,Z,C containing cluster centers
    C1:  str
        name of the first channel
    C2: str
        name of the second channel
    distance: float
        Distance threshold

    Returns
    -------
    pd.DataFrame
        List of pairs
    """
    X1 = points[points["C"] == channels[0]][["Z", "Y", "X"]].to_numpy()
    X2 = points[points["C"] == channels[1]][["Z", "Y", "X"]].to_numpy()

    from scipy.spatial.distance import cdist

    D = cdist(X1, X2) < distance

    pairs = np.where(D)
    return pd.DataFrame({channels[0]: pairs[0], channels[1]: pairs[1]})
