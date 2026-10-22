"""Tests for clustering and nearest neighbor functions."""

import numpy as np
import pandas as pd

from prrt_clustering.clustering import cluster, compute_nearest_neighbor


def test_cluster_and_nearest_neighbor():
    """Test clustering and nearest neighbor computation on synthetic puncta data."""
    # synthetic DataFrame with 2 channels
    np.random.seed(42)
    # 2 tightly packed clusters for channel 0
    c0_pts = np.vstack([
        np.random.normal(loc=[1.0, 1.0, 1.0], scale=0.1, size=(10, 3)),
        np.random.normal(loc=[10.0, 10.0, 10.0], scale=0.1, size=(10, 3)),
    ])
    c1_pts = np.random.normal(loc=[5.0, 5.0, 5.0], scale=0.1, size=(8, 3))

    df0 = pd.DataFrame(c0_pts, columns=["Z", "Y", "X"])
    df0["C"] = 0
    df0["intensity"] = 1.0

    df1 = pd.DataFrame(c1_pts, columns=["Z", "Y", "X"])
    df1["C"] = 1
    df1["intensity"] = 1.0

    points = pd.concat([df0, df1], ignore_index=True)

    # test compute_nearest_neighbor
    compute_nearest_neighbor(points, eps=1.0)

    # test cluster
    res = cluster(points, eps=0.5, min_samples=3)
    assert isinstance(res, pd.DataFrame)
    assert "class" in res.columns
    assert len(res) == len(points)
    # channel 0 should have 2 distinct clusters
    c0_classes = set(res[res["C"] == 0]["class"].unique())
    assert len(c0_classes) == 2
