"""Clustering of labeled points."""

from itertools import permutations, product

import numpy as np
import pandas as pd

# from scipy.sparse import coo_matrix
# from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from scipy.stats import poisson
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
    # initial clusters positions
    low, high = [0, 0, 0, 0.5, 10], [D, H, W, 1, 50]
    clst = np.random.uniform(low=low, high=high, size=(N, 5))

    points = pd.concat(
        [
            pd.DataFrame(
                np.vstack(
                    [
                        # point belonging to clusters
                        *[
                            np.random.normal(loc=x, scale=0.1, size=(K, 5))
                            for x in clst
                        ],
                        # add noise
                        np.random.uniform(
                            low=low,
                            high=high,
                            size=(M, 5),
                        ),
                    ]
                ),
                columns=["Z", "Y", "X", "V", "A"],
            )
            .assign(C=f"CH-{channel}")
            .assign(noise=np.arange(N * K + M) > N * K)
            for channel in range(C)
        ]
    )
    clusters = pd.DataFrame(
        clst,
        columns=["Z", "Y", "X", "V", "A"],
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
                # dists_no_self = dists[1:]
                # weights_no_self = W[inds[1:]]
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


def cluster_dbscan(
    points: pd.DataFrame,
    eps: float = 2.0,
    min_samples: int = 5,
    cluster_column: str = "cluster_id",
):
    """Cluster points for each channel C using DBSCAN.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns: X, Y, Z, C and intensity
    eps: float, optional
    min_samples: int , optional
    cluster_column:str, optional
        "cluster_id column name

    Returns
    -------
    pd.DataFrame
        points with X,Y,Z,C,cluster aligned with points
    pd.DataFrame
        cluster centroids
    """
    cols = ["X", "Y", "Z", "C", "V", "A", "cluster"]
    if points is None or points.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    new_points = []

    for _, p in points.groupby("C"):
        X = p[["Z", "Y", "X"]].to_numpy()
        p[cluster_column] = model.fit_predict(X)
        new_points.append(p)

    if not new_points:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    new_points = pd.concat(new_points)

    # Compute cluster centroids
    clusters = new_points.groupby(["C", cluster_column]).mean()
    clusters = clusters.reset_index()
    clusters = clusters[clusters[cluster_column] >= 0]

    return new_points, clusters


def pair_cluster(
    points: pd.DataFrame,
    channels: list[str],
    distance: float,
    cluster_column: str = "cluster_id",
    pair_column: str = "synapse_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair clustered points across selected channels (C).

    Parameters
    ----------
    points: pd.DataFrame
        Points with X, Y, Z, C, cluster
    channels: list[str]
        List of channels used to find clusters and pairs of clusters
    distance: float
        Distance threshold for the pairs

    Returns
    -------
    pd.DataFrame
        DataFrame with a row per point with column 'synpase' added to the input points.
    pd.DataFrame
        DataFrame with 1 row per pair cluster ID for each channel and the minimum
        distance.
    """
    cols_output = channels + ["distance"]
    if points.empty or not set(channels).issubset(points["C"].unique()):
        return points, pd.DataFrame(columns=cols_output)

    # Filter points belonging to valid clusters and requested channels
    mask = (points[cluster_column] >= 0) & (points["C"].isin(channels))
    filtered = points[mask]

    if filtered.empty:
        return points, pd.DataFrame(columns=cols_output)

    # Extract coordinates, channels, and cluster IDs as NumPy arrays
    coords = filtered[["X", "Y", "Z"]].to_numpy()
    chs = filtered["C"].to_numpy()
    ids = filtered[cluster_column].to_numpy()

    # Build the KD tree and query neighbors
    tree = KDTree(coords)
    neighbors = tree.query_ball_point(coords, r=distance)

    lens = [len(nb) for nb in neighbors]
    total_neighbors = sum(lens)
    if total_neighbors == 0:
        return points, pd.DataFrame(columns=cols_output)

    i_indices = np.repeat(np.arange(len(neighbors)), lens)
    j_indices = np.concatenate(neighbors)

    # Filter out self-pairs and same-channel pairs
    valid_mask = (i_indices != j_indices) & (chs[i_indices] != chs[j_indices])
    if not np.any(valid_mask):
        return points, pd.DataFrame(columns=cols_output)

    i_valid = i_indices[valid_mask]
    j_valid = j_indices[valid_mask]

    # Compute exact Euclidean distances vectorially
    dists = np.linalg.norm(coords[i_valid] - coords[j_valid], axis=1)

    # Build a temporary link DataFrame for vectorized grouping
    link_df = pd.DataFrame(
        {
            "ch1": chs[i_valid],
            "id1": ids[i_valid],
            "ch2": chs[j_valid],
            "id2": ids[j_valid],
            "dist": dists,
        }
    )

    # Vectorized canonicalization to ensure undirected uniqueness (A-B matches B-A)
    key1 = link_df["ch1"].astype(str) + ":" + link_df["id1"].astype(str)
    key2 = link_df["ch2"].astype(str) + ":" + link_df["id2"].astype(str)
    swap = key1 > key2

    ch1_fin = np.where(swap, link_df["ch2"], link_df["ch1"])
    id1_fin = np.where(swap, link_df["id2"], link_df["id1"])
    ch2_fin = np.where(swap, link_df["ch1"], link_df["ch2"])
    id2_fin = np.where(swap, link_df["id1"], link_df["id2"])

    link_df["ch1"] = ch1_fin
    link_df["id1"] = id1_fin
    link_df["ch2"] = ch2_fin
    link_df["id2"] = id2_fin

    # Group by unique cluster pairs to find the global minimum distance
    grouped = link_df.groupby(["ch1", "id1", "ch2", "id2"], as_index=False)[
        "dist"
    ].min()

    # Build final records using only unique cluster pairs
    records = [
        {row.ch1: row.id1, row.ch2: row.id2, "distance": row.dist}
        for row in grouped.itertuples(index=False)
    ]

    df = pd.DataFrame.from_records(records)

    # Ensure all requested channels are present as columns, plus 'distance'
    for ch in channels:
        if ch not in df.columns:
            df[ch] = np.nan

    # Add a new columns synapse to the original dataframe
    new_points = points.copy()
    for syn_id, s in df.reset_index(drop=True).iterrows():
        for ch in channels:
            mask = (filtered["C"] == ch) & (
                filtered[cluster_column].astype(int) == int(s[ch])
            )
            new_points.loc[filtered.index[mask], pair_column] = int(
                np.asarray(syn_id).item()
            )

    return new_points, df[cols_output]


def associate_to_paired_cluster(
    points: pd.DataFrame, distance: float = 1, column_id: str = "synapse_id"
) -> pd.DataFrame:
    """Associate points to closest paired clusters.

    The points dataframe has the columns X, Y, Z, pair_cluster. For each row of points,
    assigns the 'pair_column' to the closest the 'pair_column' value of the row within
    max_distance.

    Parameters
    ----------
    points: pd.DataFrame
        Points with X, Y, Z, (and pair_column if it exists)
    distance: float, optional
        Maximum distance for association
    pair_column: str, optional
        The name of the column to use (default "synapse_id")

    Returns
    -------
    pd.DataFrame
        With the 'pair_column' computed for each reachable point.
    """
    pos = ["X", "Y", "Z"]
    new_points = points.copy()

    # Ensure the pair_column exists in the dataframe
    if column_id not in new_points.columns:
        new_points[column_id] = -1

    # Filter synapse points (synapse >= 0)
    synapse_mask = new_points[column_id] >= 0
    synapse_points = new_points[synapse_mask]

    if synapse_points.empty or new_points.empty:
        return new_points

    # Extract coordinates
    syn_coords = synapse_points[pos].to_numpy()
    all_coords = new_points[pos].to_numpy()

    # Build KD-tree from synapse-labeled points
    tree = KDTree(syn_coords)

    # Query the closest synapse for every point (k=1)
    distances, indices = tree.query(all_coords, k=1)

    # Filter points within max_distance threshold
    valid_mask = np.logical_and(distances <= distance, np.logical_not(synapse_mask))

    # Get the actual synapse IDs corresponding to the nearest tree indices
    synapse_ids = synapse_points[column_id].to_numpy()
    closest_synapse_ids = synapse_ids[indices]

    # Assign the closest synapse ID to valid points
    new_points.loc[valid_mask, column_id] = closest_synapse_ids[valid_mask]

    return new_points


def pair_puncta(
    points: pd.DataFrame,
    channels: list[str],
    distance: float,
    pair_column: str = "pair_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair points across selected channels (C).

    Parameters
    ----------
    points: pd.DataFrame
        Points with X, Y, Z, C, cluster
    channels: list[str]
        List of channels used to find clusters and pairs of clusters
    distance: float
        Distance threshold for the pairs
    pair_column: str
        Name of the added column with pair id

    Returns
    -------
    pd.DataFrame
        DataFrame with a row per point with column  added to the input points.
    pd.DataFrame
        DataFrame with 1 row per pair cluster ID for each channel and the minimum
        distance.
    """
    cols_output = channels + ["distance"]
    if points.empty or not set(channels).issubset(points["C"].unique()):
        return points, pd.DataFrame(columns=cols_output)

    # Filter points belonging to valid clusters and requested channels
    mask = points["C"].isin(channels)
    filtered = points[mask]

    if filtered.empty:
        return points, pd.DataFrame(columns=cols_output)

    # Extract coordinates, channels, and IDs as NumPy arrays
    coords = filtered[["X", "Y", "Z"]].to_numpy()
    chs = filtered["C"].to_numpy()
    ids = filtered.index.to_numpy()

    # Build the KD tree and query neighbors
    tree = KDTree(coords)
    neighbors = tree.query_ball_point(coords, r=distance)

    lens = [len(nb) for nb in neighbors]
    total_neighbors = sum(lens)
    if total_neighbors == 0:
        return points, pd.DataFrame(columns=cols_output)

    i_indices = np.repeat(np.arange(len(neighbors)), lens)
    j_indices = np.concatenate(neighbors)

    # Filter out self-pairs and same-channel pairs
    valid_mask = (i_indices != j_indices) & (chs[i_indices] != chs[j_indices])
    if not np.any(valid_mask):
        return points, pd.DataFrame(columns=cols_output)

    i_valid = i_indices[valid_mask]
    j_valid = j_indices[valid_mask]

    # Compute exact Euclidean distances vectorially
    dists = np.linalg.norm(coords[i_valid] - coords[j_valid], axis=1)

    # Build a temporary link DataFrame for vectorized grouping
    link_df = pd.DataFrame(
        {
            "ch1": chs[i_valid],
            "id1": ids[i_valid],
            "ch2": chs[j_valid],
            "id2": ids[j_valid],
            "dist": dists,
        }
    )

    # Vectorized canonicalization to ensure undirected uniqueness (A-B matches B-A)
    key1 = link_df["ch1"].astype(str) + ":" + link_df["id1"].astype(str)
    key2 = link_df["ch2"].astype(str) + ":" + link_df["id2"].astype(str)
    swap = key1 > key2

    ch1_fin = np.where(swap, link_df["ch2"], link_df["ch1"])
    id1_fin = np.where(swap, link_df["id2"], link_df["id1"])
    ch2_fin = np.where(swap, link_df["ch1"], link_df["ch2"])
    id2_fin = np.where(swap, link_df["id1"], link_df["id2"])

    link_df["ch1"] = ch1_fin
    link_df["id1"] = id1_fin
    link_df["ch2"] = ch2_fin
    link_df["id2"] = id2_fin

    # Group by unique pairs to find the global minimum distance
    grouped = link_df.groupby(["ch1", "id1", "ch2", "id2"], as_index=False)[
        "dist"
    ].min()

    # Build final records using only unique pairs
    records = [
        {row.ch1: row.id1, row.ch2: row.id2, "distance": row.dist}
        for row in grouped.itertuples(index=False)
    ]

    df = pd.DataFrame.from_records(records)

    # Ensure all requested channels are present as columns, plus 'distance'
    for ch in channels:
        if ch not in df.columns:
            df[ch] = np.nan

    # Add a new columns synapse to the original dataframe
    new_points = points.copy()
    for syn_id, s in df.reset_index(drop=True).iterrows():
        for ch in channels:
            mask = (filtered["C"] == ch) & (filtered.index.astype(int) == int(s[ch]))
            new_points.loc[filtered.index[mask], pair_column] = int(
                np.asarray(syn_id).item()
            )

    return new_points, df[cols_output]


def compute_distances(points: pd.DataFrame) -> pd.DataFrame:
    """Compute the distance from each point to the closest point in other channels (C).

    Parameters
    ----------
    points: pd.DataFrame
        Data frame with columns X, Y, Z, C

    Returns
    -------
    pd.DataFrame:
        The input dataframe with the distance to the closest point in each of the other
        channels.
    """
    pos = ["X", "Y", "Z"]
    channels = points["C"].unique()

    # List directed pair (c_trg -> c_ref)
    channel_pairs = permutations(channels, 2)
    new_points = points.copy()

    for c_trg, c_ref in channel_pairs:
        trg_mask = points["C"] == c_trg
        ref_mask = points["C"] == c_ref

        # If either channel is empty, skip
        if not trg_mask.any() or not ref_mask.any():
            continue

        trg = points[trg_mask][pos].to_numpy()
        ref = points[ref_mask][pos].to_numpy()
        i_trg = points[trg_mask].index.to_numpy()

        # Build KD-tree on the reference channel and query from the target channel
        tree = KDTree(ref)
        dist, _ = tree.query(trg, k=1)

        # Assign distance for points in c_trg to the closest point in c_ref
        new_points.loc[i_trg, f"distance to {c_ref}"] = dist

    return new_points


def quantification(
    points: pd.DataFrame, distance: float, synapse_column: str = "synapse_id"
) -> pd.DataFrame:
    """Quantify presence of markers for each synapse.

    Group the points by synapse.

    Parameters
    ----------
    points: pd.DataFrame
        Data frame with columns X, Y, Z, C, synapse
    distance: float
        Distance threshold

    Returns
    -------
    pd.DataFrame
        Dataframe with a column per unique C value and a row per unique synapse value
        reporting the number of points in the vicinity of the synapse.
    """
    pos = ["X", "Y", "Z"]

    # Filter synapse points (synapse >= 0)
    synapse_mask = points[synapse_column] >= 0
    synapse_points = points[synapse_mask]

    if synapse_points.empty or points.empty:
        return pd.DataFrame(columns=points["C"].unique())

    # Mappings for unique synapses and channels
    unique_synapses = np.sort(synapse_points[synapse_column].unique())
    synapse_to_idx = {syn: i for i, syn in enumerate(unique_synapses)}

    unique_channels = points["C"].unique()
    channel_to_idx = {c: i for i, c in enumerate(unique_channels)}

    # Extract coordinates of both point clouds
    syn_coords = synapse_points[pos].to_numpy()
    all_coords = points[pos].to_numpy()

    # Build KD-tree from synpases labeled points
    tree = KDTree(syn_coords)

    # Query all synapses neighbors
    neighbors = tree.query_ball_point(all_coords, r=distance)

    # Initialize accumulation matrix
    acc = np.zeros((len(unique_synapses), len(unique_channels)), dtype=np.int64)

    # Create index mapping
    channels_arr = points["C"].map(channel_to_idx).to_numpy()
    synapse_arr = synapse_points[synapse_column].map(synapse_to_idx).to_numpy()

    # Aggregate counts using numpy vectorized accumulation
    for i, nb_indices in enumerate(neighbors):
        if not nb_indices:
            continue
        m = channels_arr[i]
        syn_idxs = synapse_arr[nb_indices]
        np.add.at(acc[:, m], syn_idxs, 1)

    return pd.DataFrame(
        data=acc, index=unique_synapses.astype(np.uint64), columns=unique_channels
    )


def compute_union_of_balls_volume(
    points: pd.DataFrame, ball_radius: float, voxel_size: float
) -> float:
    """Estimate the volume of the union of overlapping balls centered at `coords`.

    Parameters
    ----------
    points : pd.DataFrame
        Dataframe with Z, Y, X coordinates.
    ball_radius : float
        Radius of the ball around each point.
    voxel_size : float, optional
        Resolution of grid. Defaults to ball_radius / 3 for accuracy.

    Returns
    -------
    float
        Total volume occupied by the union of balls.
    """
    coords = points[["Z", "Y", "X"]].to_numpy()

    if voxel_size is None:
        voxel_size = ball_radius / 3.0

    # Get bounding box limits expanded by the ball radius

    min_coords = coords.min(axis=0) - ball_radius
    max_coords = coords.max(axis=0) + ball_radius

    # Create 3D grid axes
    z_range = np.arange(min_coords[0], max_coords[0], voxel_size)
    y_range = np.arange(min_coords[1], max_coords[1], voxel_size)
    x_range = np.arange(min_coords[2], max_coords[2], voxel_size)

    grid = (
        np.vstack(np.meshgrid(z_range, y_range, x_range, indexing="ij"))
        .reshape(3, -1)
        .T
    )

    # Check which grid voxels fall within ball_radius of ANY point
    tree = KDTree(coords)

    # query_ball_point in the grid
    occupied_voxels = tree.query_ball_point(grid, r=ball_radius, return_length=True)

    # restrict to the inside of the domain
    inside_box = np.logical_and(
        np.all(grid >= coords.min(axis=0), axis=1),
        np.all(grid <= coords.max(axis=0), axis=1),
    )

    occupied_voxels = np.logical_and(occupied_voxels > 0, inside_box)

    num_occupied = np.count_nonzero(occupied_voxels)

    voxel_volume = voxel_size**3

    return float(num_occupied * voxel_volume)


def expected_number_of_pair(
    points: pd.DataFrame,
    channels: list[str],
    max_distance: float,
    window_volume=None,
) -> float:
    """Compute the expected number of point pairs at less than a max_distance.

    Parameters
    ----------
    points: pd.DataFrame
        Data with columns "Z","Y","X","C","distance to {channels[1]}"
    channels: list[str]
        Value of C to filter on
    max_distance:float
        Maximum distance
    window_volume:float
        Volume of the analysis window.

    Returns
    -------
    float
        Expected number of points at a distance less than max_distance.
    """
    # Bounding box volume VW
    if window_volume is None:
        window_volume = np.prod(np.ptp(points[["Z", "Y", "X"]].to_numpy(), axis=0))

    # Correct 3D ball volume: (4/3) * pi * R^3
    ball_volume = (4.0 / 3.0) * np.pi * (max_distance**3)

    n1 = len(points[points["C"] == channels[0]])
    n2 = len(points[points["C"] == channels[1]])

    # Expected number of pairs
    mu = (n1 * n2 / window_volume) * ball_volume
    return mu


def compute_poisson_pair_pvalue(
    points: pd.DataFrame,
    channels: list[str],
    max_distance: float,
    volume_window: float | None = None,
) -> float:
    """Compute the p-value to observe pairs assuming two Poisson point processes.

    Parameters
    ----------
    points: pd.DataFrame
        Data with columns "Z","Y","X","C","distance to {channels[1]}"
    channels: list[str]
        Value of C to filter on
    max_distance:float
        Maximum distance
    volume_window: float, optional
        Volume of the analysis window

    Returns
    -------
    float
        pvalue to observe the points at a distance less than max_distance.
    """
    mu = expected_number_of_pair(points, channels, max_distance, volume_window)

    # Extract coordinates for each channel dynamically based on parameters
    pts1 = points[points["C"] == channels[0]][["Z", "Y", "X"]].to_numpy()
    pts2 = points[points["C"] == channels[1]][["Z", "Y", "X"]].to_numpy()

    if len(pts1) == 0 or len(pts2) == 0:
        return 1.0

    # Build KD-trees and count neighboring pairs efficiently
    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)
    n0 = tree1.count_neighbors(
        tree2, max_distance
    )  # Counts pairs (x in pts1, y in pts2) with dist <= max_distance

    # Correct Poisson upper-tail p-value: P(N >= n0) = 1 - P(N <= n0 - 1)
    p_val = 1 - poisson.cdf(n0 - 1, mu)
    return float(p_val)


def compute_monte_carlo_pair_pvalue(
    points: pd.DataFrame,
    channels: list[str],
    max_distance: float,
    num_shifts: int = 100,
    shift_range: np.ndarray | None = None,
    min_shift: float | None = None,
):
    """Compute empirical p-value for spatial co-localization using random shifts.

    Parameters
    ----------
    points : pd.DataFrame
        Dataframe with "Z", "Y", "X", "C" columns.
    channels : list[str]
        Two channel names [channel_1, channel_2] to compare.
    max_distance : float
        Maximum distance threshold to count a valid pair.
    num_shifts : int, optional
        Number of random Monte Carlo shifts (default: 100).
    shift_range : np.ndarray, optional
        Maximum shift magnitude along [Z, Y, X] (default to to 10 x max_distance)
    min_shift : float, optional
        Minimum shift distance to prevent near-zero shifts (defaults to max_distance).

    Returns
    -------
    pvalue : float
        One-tailed p-value testing if observed pair count is significantly higher
        than expected by chance.
    n0 : int
        Observed pair count at zero shift.
    counts : np.ndarray
        Array of pair counts from the randomized shifts.
    """
    pts1 = points[points["C"] == channels[0]][["Z", "Y", "X"]].to_numpy()
    pts2 = points[points["C"] == channels[1]][["Z", "Y", "X"]].to_numpy()

    if len(pts1) == 0 or len(pts2) == 0:
        return 1.0, 0, np.zeros(num_shifts)

    if shift_range is None:
        shift_range = np.array([0, 10 * max_distance, 10 * max_distance])

    if min_shift is None:
        min_shift = max_distance

    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)
    n0 = tree1.count_neighbors(tree2, max_distance)

    counts = np.zeros(num_shifts, dtype=int)

    # Generate shifts, discarding near-zero shifts that don't decorrelate
    shifts = []
    while len(shifts) < num_shifts:
        s = np.random.uniform(-shift_range, shift_range, size=(3,))
        if np.linalg.norm(s) >= min_shift:
            shifts.append(s)
    shifts = np.array(shifts)

    # Compute pair counts under random shifts
    for k in range(num_shifts):
        shifted_tree2 = KDTree(pts2 + shifts[k])
        counts[k] = tree1.count_neighbors(shifted_tree2, max_distance)

    # Standard Monte Carlo p-value formulation (including observed data in baseline)
    pvalue = (np.sum(counts >= n0) + 1) / (num_shifts + 1)

    return float(pvalue), int(n0), counts


def synapse_stats(points):
    """Compute statistics for each synapse."""
    records = []
    for n, g in points[points["synapse_id"] >= 0].groupby("synapse_id"):
        records.append(
            {
                "synapse": int(n),
                "Number of RIM puncta": len(g[g["C"] == "RIM"]),
                "Number of PSD puncta": len(g[g["C"] == "PSD"]),
                "Number of A1 puncta": len(g[g["C"] == "A1"]),
                "Number of PRRT puncta": len(g[g["C"] == "PRRT"]),
                "Number of A1-PRRT complex": len(g[g["pair_id"] >= 0]),
                "Mean distance PRRT:PSD": g[g["C"] == "PRRT"]["distance to PSD"].mean(),
                "Mean distance PRRT:RIM": g[g["C"] == "PRRT"]["distance to RIM"].mean(),
                "Mean distance A1:PSD": g[g["C"] == "A1"]["distance to PSD"].mean(),
                "Mean distance A1:RIM": g[g["C"] == "A1"]["distance to RIM"].mean(),
                "Mean distance PRRT:A1": g[g["C"] == "PRRT"]["distance to A1"].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def global_stats(
    points: pd.DataFrame, distance: float, ball_radius: float = 2, voxel_size: float = 1
):
    """Compute global statistics at the image level.

    Parameters
    ----------
    points: pd.DataFrame
    """
    # Precompute the volume of the analysis window for efficiency
    volume_window = compute_union_of_balls_volume(
        points, ball_radius, voxel_size=voxel_size
    )
    return {
        **{
            f"Number of {c} puncta": len(points[points["C"] == c])
            for c in points["C"].unique()
        },
        **{
            f"Number of {c} cluster": len(
                (points[(points["C"] == c) & (points["cluster_id"] >= 0)])
            )
            for c in ["PSD", "RIM"]
        },
        "Number of synpases": points["synapse_id"].nunique(),
        "Number of GluA1 synpases": points[
            (points["C"] == "A1") & (points["synapse_id"] >= 0)
        ]["synapse_id"].nunique(),
        "Number of PRRT+GluA1 synpases": points[
            (points["pair_id"] >= 0) & (points["synapse_id"] >= 0)
        ]["synapse_id"].nunique(),
        "Number of PRRT+GluA1 total": int((points["pair_id"] >= 0).sum()),
        **{
            f"Average distance of {p[0]} to {p[1]}": float(
                points[(points["C"] == p[0]) & (points["synapse_id"] >= 0)][
                    f"distance to {p[1]}"
                ].mean()
            )
            for p in product(["A1", "PRRT"], ["RIM", "PSD"])
        },
        **{
            f"poisson pvalue {p[0]}:{p[1]}": compute_poisson_pair_pvalue(
                points, list(p), distance, volume_window
            )
            for p in permutations(points["C"].unique(), 2)
        },
        **{
            f"monte carlo pvalue {p[0]}:{p[1]}": compute_monte_carlo_pair_pvalue(
                points, list(p), distance
            )[0]
            for p in permutations(points["C"].unique(), 2)
        },
    }
