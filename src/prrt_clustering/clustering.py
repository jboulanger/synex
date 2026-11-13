"""Clustering of labeled points."""

from itertools import permutations, product

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# from scipy.sparse import coo_matrix
# from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
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
    columns = ["Z", "Y", "X", "mean intensity", "volume"]
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
                columns=columns,
            )
            .assign(C=f"CH-{channel}")
            .assign(noise=np.arange(N * K + M) > N * K)
            for channel in range(C)
        ]
    )
    clusters = pd.DataFrame(
        clst,
        columns=columns,
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
    channels: list[str],
    eps: float = 2.0,
    min_samples: int = 5,
    cluster_column: str = "cluster_id",
):
    """Cluster points for each channel C using DBSCAN.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns: X, Y, Z, C and intensity
    channels: list[str]
        List of channels to use
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
    print("--")
    cols = ["X", "Y", "Z", "C", "V", "A", "cluster"]
    if points is None or points.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    new_points = points.copy()
    new_points[cluster_column] = -1
    for C, p in points.groupby("C"):
        if C in channels:
            X = p[["Z", "Y", "X"]].to_numpy()
            new_points.loc[p.index, cluster_column] = model.fit_predict(X)

    # Compute cluster centroids
    clusters = new_points.groupby(["C", cluster_column]).mean()
    clusters = clusters.reset_index()
    clusters = clusters[clusters[cluster_column] >= 0]

    return new_points, clusters


def sample_in_union_of_balls(
    coords: np.ndarray,
    n_samples: int,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample n_samples inside the union of balls.

    Parameters
    ----------
    coords: np.ndarray
    n_samples: int
    radius: float
    rng: np.random.Generator

    Returns
    -------
    np.ndarray:
        List of random coordinates in the domain.
    """
    tree = KDTree(coords)
    low = np.min(coords, axis=0)
    high = np.max(coords, axis=0)

    accepted = []
    n_collected = 0
    # Batch size scaled up to account for rejection rate
    batch_size = max(n_samples * 2, 1000)

    while n_collected < n_samples:
        candidates = rng.uniform(low, high, size=(batch_size, 3))
        dists, _ = tree.query(candidates, k=1, workers=-1)
        valid = candidates[dists <= radius]
        if len(valid) > 0:
            accepted.append(valid)
            n_collected += len(valid)

    return np.vstack(accepted)[:n_samples]


def compute_dbscan_clustering_pvalue(
    points: pd.DataFrame,
    channels: list[str],
    eps: float = 2.0,
    min_samples: int = 5,
    cluster_column: str = "cluster_id",
    n_tests: int = 1000,
    radius: float = 5,
    seed: int | None = None,
) -> pd.DataFrame:
    """Test statistical significance of DBSCAN cluster counts.

    Parameters
    ----------
    points: pd.DataFrame
        DataFrame with columns: X, Y, Z, C and cluster_column
    channels: list[str]
        List of channels to use
    eps: float, optional
        Maximum distance between points in the cluster.
    min_samples: int, optional
        Minimum of samples per cluster
    cluster_column: str, optional
        cluster_id column name
    n_tests: int, optional
        Number of tests
    radius: float, optional
        Radius used to estimate the support of the point process.

    Returns
    -------
    pd.DataFrame
        Table with columns "C", "number of clusters" and "p-value"
    """
    cols = ["C", "number of clusters", "p-value"]
    if points is None or points.empty:
        return pd.DataFrame(columns=cols)

    rng = np.random.default_rng(seed)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    Xref = points[["Z", "Y", "X"]].to_numpy()

    pvalues = []
    for C, p in points.groupby("C"):
        if C in channels:
            X = p[["Z", "Y", "X"]].to_numpy()
            n_points = len(X)

            # Observed number of non-noise clusters
            clid = p[cluster_column].to_numpy()
            n_obs_clusters = len(np.unique(clid[clid >= 0]))

            # Bounding box (replace with true acquisition bounds if known)
            low, high = np.min(X, axis=0), np.max(X, axis=0)

            sim_clusters = np.zeros(n_tests, dtype=int)
            for k in range(n_tests):
                X_sim = rng.uniform(low, high, size=(n_points, 3))
                X_sim = sample_in_union_of_balls(Xref, n_points, radius, rng)
                lbl = model.fit_predict(X_sim)
                sim_clusters[k] = len(np.unique(lbl[lbl >= 0]))

            # One-tailed empirical p-value: P(N_sim >= N_obs)
            p_val = (1 + np.sum(sim_clusters >= n_obs_clusters)) / (n_tests + 1)
            pvalues.append(
                {
                    "C": C,
                    "number of clusters": n_obs_clusters,
                    "p-value": p_val,
                }
            )

    return pd.DataFrame.from_records(pvalues, columns=cols)


def pair_cluster(
    points: pd.DataFrame,
    channels: tuple[str, str] | list[str],
    distance: float,
    cluster_column: str = "cluster_id",
    pair_column: str = "synapse_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair clustered points between two channels (C) using optimal 1-to-1 matching.

    - Clusters are identified by `cluster_column` (value >= 0).
    - Distance between clusters is the minimum Euclidean distance between their points.
    - Matches clusters 1-to-1 via the Hungarian algorithm (`linear_sum_assignment`).

    Parameters
    ----------
    points : pd.DataFrame
        Points with X, Y, Z, C, and `cluster_column`.
    channels : sequence of 2 str
        Pair of channels [ch1, ch2] to match across.
    distance : float
        Maximum distance threshold for valid pairs.
    cluster_column : str, default="cluster_id"
        Column containing cluster IDs.
    pair_column : str, default="synapse_id"
        Output column name for assigned pair IDs (-1 if unpaired).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Copy of `points` with `pair_column` added.
        - DataFrame with columns `[ch1, ch2, "distance"]` for each matched pair.
    """
    if len(channels) != 2 or channels[0] == channels[1]:
        raise ValueError("`channels` must contain exactly two distinct channel names.")

    ch1, ch2 = channels
    cols_output = [ch1, ch2, "distance"]

    new_points = points.copy()
    new_points[pair_column] = pd.Series(
        -1, index=new_points.index, dtype=pd.Int64Dtype()
    )

    if points.empty:
        return new_points, pd.DataFrame(columns=cols_output)

    # Filter valid clustered points separately for each channel
    valid_cluster = points[cluster_column] >= 0
    pts1 = points[valid_cluster & (points["C"] == ch1)]
    pts2 = points[valid_cluster & (points["C"] == ch2)]

    if pts1.empty or pts2.empty:
        return new_points, pd.DataFrame(columns=cols_output)

    coords1 = pts1[["X", "Y", "Z"]].to_numpy()
    coords2 = pts2[["X", "Y", "Z"]].to_numpy()
    ids1 = pts1[cluster_column].to_numpy()
    ids2 = pts2[cluster_column].to_numpy()

    # Bipartite spatial query: only searches pairs between ch1 and ch2
    tree1 = KDTree(coords1)
    tree2 = KDTree(coords2)
    neighbors = tree1.query_ball_tree(tree2, r=distance)

    lens = [len(nb) for nb in neighbors]
    if sum(lens) == 0:
        return new_points, pd.DataFrame(columns=cols_output)

    i_idx = np.repeat(np.arange(len(pts1)), lens)
    j_idx = np.concatenate([np.array(n, dtype=np.uint64) for n in neighbors])
    dists = np.linalg.norm(coords1[i_idx] - coords2[j_idx], axis=1)

    # Minimum distance between every candidate (id1, id2) cluster pair
    grouped = (
        pd.DataFrame({"id1": ids1[i_idx], "id2": ids2[j_idx], "dist": dists})
        .groupby(["id1", "id2"], as_index=False)["dist"]
        .min()
    )

    # Build cost matrix for Hungarian matching
    u_ids1 = np.sort(grouped["id1"].unique())
    u_ids2 = np.sort(grouped["id2"].unique())
    map1 = {cid: i for i, cid in enumerate(u_ids1)}
    map2 = {cid: j for j, cid in enumerate(u_ids2)}

    # Finite penalty strictly larger than any valid sum of distances
    penalty = (grouped["dist"].max() + 1.0) * max(len(u_ids1), len(u_ids2)) + 1.0
    cost_matrix = np.full((len(u_ids1), len(u_ids2)), penalty, dtype=float)

    cost_matrix[grouped["id1"].map(map1), grouped["id2"].map(map2)] = grouped["dist"]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_costs = cost_matrix[row_ind, col_ind]
    valid = matched_costs <= distance

    if not np.any(valid):
        return new_points, pd.DataFrame(columns=cols_output)

    df = (
        pd.DataFrame(
            {
                ch1: u_ids1[row_ind[valid]],
                ch2: u_ids2[col_ind[valid]],
                "distance": matched_costs[valid],
            }
        )
        .sort_values("distance", ascending=True)
        .reset_index(drop=True)
    )
    df[pair_column] = df.index.astype("Int64")

    # Map synapse IDs back to pts1 and pts2 directly
    for ch, pts_ch in ((ch1, pts1), (ch2, pts2)):
        id_to_syn = df.set_index(ch)[pair_column]
        mapped_syn = pts_ch[cluster_column].map(id_to_syn)
        matched_mask = mapped_syn.notna()
        new_points.loc[pts_ch.index[matched_mask], pair_column] = mapped_syn[
            matched_mask
        ].astype("Int64")

    return new_points, df[cols_output]


def associate_to_paired_cluster(
    points: pd.DataFrame,
    channels: list[str],
    distance: float = 1.0,
    column_id: str = "synapse_id",
) -> pd.DataFrame:
    """Associate points in `channels` to the closest paired cluster (`column_id >= 0`).

    Each unassigned point in `channels` is matched (many-to-one) to the nearest
    point belonging to an existing paired cluster within `distance`.

    Parameters
    ----------
    points : pd.DataFrame
        Points with X, Y, Z, C, and `column_id`.
    channels : list[str]
        Target channel(s) whose unassigned points should be associated to paired clusters.
    distance : float, default=1.0
        Maximum Euclidean distance threshold for association.
    column_id : str, default="synapse_id"
        Name of the column storing paired cluster / synapse IDs (-1 if unassigned).

    Returns
    -------
    pd.DataFrame
        Copy of `points` with `column_id` updated for reachable target points.
    """
    required_columns = set(["X", "Y", "Z", "C", column_id])
    assert required_columns.issubset(points.columns)
    pos = ["X", "Y", "Z"]
    new_points = points.copy()

    if column_id not in new_points.columns:
        new_points[column_id] = -1
        return new_points

    if new_points.empty or not channels:
        return new_points

    # Reference points: already assigned to a paired cluster and outside `channels`
    synapse_mask = (new_points[column_id] >= 0) & (~new_points["C"].isin(channels))
    # Target points: inside `channels` and not yet assigned
    target_mask = new_points["C"].isin(channels) & (new_points[column_id] < 0)

    synapse_points = new_points[synapse_mask]
    target_points = new_points[target_mask]

    if synapse_points.empty or target_points.empty:
        return new_points

    syn_coords = synapse_points[pos].to_numpy(dtype=float)
    target_coords = target_points[pos].to_numpy(dtype=float)

    # Query only target points against reference synapse points within `distance`
    tree = KDTree(syn_coords)
    distances, indices = tree.query(target_coords, k=1, distance_upper_bound=distance)

    # `tree.query` returns `inf` (and index == len(syn_coords)) when no neighbor is within `distance`
    valid = np.isfinite(distances)
    if not np.any(valid):
        return new_points

    synapse_ids = synapse_points[column_id].to_numpy()
    new_points.loc[target_points.index[valid], column_id] = synapse_ids[indices[valid]]

    return new_points


def pair_puncta(
    points: pd.DataFrame,
    channels: tuple[str, str] | list[str],
    distance: float,
    pair_column: str = "pair_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair individual puncta across two channels (C) using 1-to-1 linear assignment.

    Parameters
    ----------
    points : pd.DataFrame
        Points with X, Y, Z, and C columns.
    channels : sequence of 2 str
        Pair of distinct channels [ch1, ch2] to match across.
    distance : float
        Maximum distance threshold for valid pairs.
    pair_column : str, default="pair_id"
        Name of the added column with assigned pair IDs (-1 if unpaired).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Copy of `points` with `pair_column` added (-1 for unpaired points).
        - DataFrame with columns `[ch1, ch2, "distance"]` containing the matched
            point indices from `points` and their Euclidean distance.
    """
    required_columns = set(["X", "Y", "Z", "C"])
    assert required_columns.issubset(points.columns)

    if len(channels) != 2 or channels[0] == channels[1]:
        raise ValueError("`channels` must contain exactly two distinct channel names.")

    ch1, ch2 = channels
    cols_output = [ch1, ch2, "distance"]

    new_points = points.copy()
    new_points[pair_column] = pd.Series(
        -1, index=new_points.index, dtype=pd.Int64Dtype()
    )

    if points.empty:
        return new_points, pd.DataFrame(columns=cols_output)

    pts1 = points[points["C"] == ch1]
    pts2 = points[points["C"] == ch2]

    if pts1.empty or pts2.empty:
        return new_points, pd.DataFrame(columns=cols_output)

    coord_cols = [c for c in ("X", "Y", "Z") if c in points.columns]
    coords1 = pts1[coord_cols].to_numpy(dtype=float)
    coords2 = pts2[coord_cols].to_numpy(dtype=float)

    # Bipartite spatial query between ch1 and ch2 puncta
    tree1 = KDTree(coords1)
    tree2 = KDTree(coords2)
    neighbors = tree1.query_ball_tree(tree2, r=distance)

    lens = [len(nb) for nb in neighbors]
    if sum(lens) == 0:
        return new_points, pd.DataFrame(columns=cols_output)

    i_idx = np.repeat(np.arange(len(pts1)), lens)
    j_idx = np.concatenate(neighbors).astype(int)
    dists = np.linalg.norm(coords1[i_idx] - coords2[j_idx], axis=1)

    # Remap to compact active indices
    active_i, row_pos = np.unique(i_idx, return_inverse=True)
    active_j, col_pos = np.unique(j_idx, return_inverse=True)
    row_pos = row_pos.ravel()
    col_pos = col_pos.ravel()

    n_a, n_b = len(active_i), len(active_j)

    # Build sparse bipartite graph WITH slack nodes so unmatched puncta never block valid pairs.
    # Real edges have weight = (dist - distance - 1.0) < 0, ensuring maximum cardinality first,
    # then minimum total Euclidean distance!
    edge_weights = dists - (distance + 1.0)
    graph = csr_array((edge_weights, (row_pos, col_pos)), shape=(n_a, n_b))

    row_ind, col_ind = min_weight_full_bipartite_matching(graph)

    # Recover original distances from matched edges
    matched_dists = np.asarray(graph[row_ind, col_ind]).ravel() + (distance + 1.0)

    matched_idx1 = pts1.index[active_i[row_ind]]
    matched_idx2 = pts2.index[active_j[col_ind]]

    df = (
        pd.DataFrame(
            {
                ch1: matched_idx1,
                ch2: matched_idx2,
                "distance": matched_dists,
            }
        )
        .sort_values("distance", ascending=True)
        .reset_index(drop=True)
    )
    df[pair_column] = df.index.astype("Int64")

    new_points.loc[df[ch1], pair_column] = df[pair_column].to_numpy()
    new_points.loc[df[ch2], pair_column] = df[pair_column].to_numpy()

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
    required_columns = set(["X", "Y", "Z", "C"])
    assert required_columns.issubset(points.columns)

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
    required_columns = set(["X", "Y", "Z"])
    assert required_columns.issubset(points.columns)

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
    required_columns = set(["X", "Y", "Z", "C"])
    assert required_columns.issubset(points.columns)

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
    num_shifts: int = 1000,
    shift_range: np.ndarray | None = None,
    min_shift: float | None = None,
    box_size: np.ndarray | None = None,
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
        Maximum shift magnitude along [Z, Y, X] (default to 10 x max_distance).
    min_shift : float, optional
        Minimum shift distance to prevent near-zero shifts (defaults to max_distance).
    box_size : np.ndarray, optional
        Bounding box dimensions [Z, Y, X] for toroidal boundary conditions.
        If None, computed from the coordinate ranges of the combined points.

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

    all_pts = np.vstack([pts1, pts2])
    min_bound = all_pts.min(axis=0)
    if box_size is None:
        box_size = (all_pts.max(axis=0) - min_bound) + 1e-5
    else:
        box_size = np.asarray(box_size, dtype=float)

    p1 = pts1 - min_bound
    p2 = pts2 - min_bound

    if shift_range is None:
        shift_range = np.array([0, 10 * max_distance, 10 * max_distance])

    if min_shift is None:
        min_shift = max_distance

    tree1 = KDTree(p1, boxsize=box_size)
    tree2 = KDTree(p2, boxsize=box_size)
    n0 = tree1.count_neighbors(tree2, max_distance)

    counts = np.zeros(num_shifts, dtype=int)

    # Generate shifts, discarding near-zero shifts that don't decorrelate
    shifts = []
    while len(shifts) < num_shifts:
        s = np.random.uniform(-shift_range, shift_range, size=(3,))
        if np.linalg.norm(s) >= min_shift:
            shifts.append(s)
    shifts = np.array(shifts)

    # Compute pair counts under random shifts with toroidal boundary conditions
    for k in range(num_shifts):
        shifted_p2 = (p2 + shifts[k]) % box_size
        shifted_tree2 = KDTree(shifted_p2, boxsize=box_size)
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
    points: pd.DataFrame,
    distance: float,
    ball_radius: float = 5.0,
    voxel_size: float = 1.0,
    channels=None,
):
    """Compute global statistics at the image level.

    Perform randomization for pairing.


    Parameters
    ----------
    points: pd.DataFrame
    distance: float
        Distance threshold for association between pair of puncta

    TODO
    ----
    - move here the cluster randomization or compute the pairing randomization elsewhere.
    """
    required_columns = set(["X", "Y", "Z", "C", "synapse_id", "pair_id"])
    assert required_columns.issubset(points.columns)
    print("ok 1")
    # Precompute the volume of the analysis window for efficiency
    volume_window = compute_union_of_balls_volume(
        points, ball_radius, voxel_size=voxel_size
    )
    print("ok 2")
    return {
        "Volume window": volume_window,
        **{
            f"Number of {c} puncta": len(points[points["C"] == c])
            for c in points["C"].unique()
        },
        **{
            f"Number of {c} cluster": points[
                (points["C"] == c) & (points["cluster_id"] >= 0)
            ]["cluster_id"].nunique()
            for c in ["PSD", "RIM"]
        },
        "Number of synpases": points[points["synapse_id"] >= 0]["synapse_id"].nunique(),
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
