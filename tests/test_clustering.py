"""Tests for clustering and nearest neighbor functions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from prrt_clustering.clustering import (
    associate_to_paired_cluster,
    clean_points_nn_gmm,
    compute_monte_carlo_pair_pvalue,
    create_synthetic_point_cloud,
    pair_cluster,
)


def test_clean_points():
    """Test clustering and nearest neighbor computation on synthetic puncta data."""
    # synthetic point cloud with 4 channels and background
    np.random.seed(42)
    points, _ = create_synthetic_point_cloud()

    # test compute_nearest_neighbor
    clean = clean_points_nn_gmm(points)

    assert len(clean) == len(points)
    assert "noise" in clean.columns
    assert clean["noise"].dtype == bool
    assert clean["noise"].sum() > 0

    y_true = points["noise"].to_numpy()
    y_pred = clean["noise"].to_numpy()

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # cm is [[TN, FP], [FN, TP]] where positive is noise
    assert cm.shape == (2, 2)
    assert f1 > 0.90


def test_compute_monte_carlo_pair_pvalue():
    """Test Monte Carlo shift test with toroidal boundary conditions."""
    import pandas as pd

    # Well-separated points across CH1 and CH2 within a 20x20x20 box
    coords = np.array(
        [
            [2.0, 2.0, 2.0],
            [6.0, 6.0, 6.0],
            [10.0, 10.0, 10.0],
            [14.0, 14.0, 14.0],
        ]
    )
    df1 = pd.DataFrame(coords, columns=["Z", "Y", "X"]).assign(C="CH1")
    df2 = pd.DataFrame(coords + 0.1, columns=["Z", "Y", "X"]).assign(C="CH2")
    points = pd.concat([df1, df2], ignore_index=True)

    pval, n0, counts = compute_monte_carlo_pair_pvalue(
        points,
        channels=["CH1", "CH2"],
        max_distance=0.5,
        num_shifts=50,
        min_shift=1.0,
        box_size=np.array([20.0, 20.0, 20.0]),
    )

    assert n0 == 4
    assert pval <= 0.05
    assert len(counts) == 50


def test_associate_to_paired_cluster_preserves_existing_ids():
    """Test that associate_to_paired_cluster preserves existing IDs."""
    import pandas as pd

    # Point 0: already in synapse 1 at X=0.0
    # Point 1: already in synapse 2 at X=0.05 (near point 0)
    # Point 2: unassigned (-1) at X=0.04 (closest to point 1: dist 0.01 vs 0.04)
    # Point 3: unassigned (-1) at X=10.0 (far away)
    df = pd.DataFrame(
        {
            "X": [0.0, 0.05, 0.04, 10.0],
            "Y": [0.0, 0.0, 0.0, 10.0],
            "Z": [0.0, 0.0, 0.0, 10.0],
            "C": ["A", "A", "B", "C"],
            "synapse_id": [1, 2, -1, -1],
        },
        index=[10, 20, 30, 40],  # Non-default index to also test index stability
    )

    result = associate_to_paired_cluster(
        df, channels=["B", "C"], distance=1.0, column_id="synapse_id"
    )

    # Pre-existing positive IDs must remain unchanged
    assert result.loc[10, "synapse_id"] == 1
    assert result.loc[20, "synapse_id"] == 2

    # Unassigned point at X=0.04 is nearest to point 1 (X=0.05, synapse 2)
    assert result.loc[30, "synapse_id"] == 2

    # Unassigned distant point must remain unassigned (-1)
    assert result.loc[40, "synapse_id"] == -1


def test_pair_cluster_greedy_one_to_one():
    """Test that pair_cluster performs greedy 1-to-1 matching and avoids collision."""
    import pandas as pd

    # Cluster 0 in PSD is 0.5 um from Cluster 0 in RIM
    # Cluster 1 in PSD is 0.2 um from Cluster 0 in RIM (closer!)
    # Cluster 2 in PSD is 0.4 um from Cluster 1 in RIM
    # Under greedy 1-to-1 matching:
    # (PSD 1, RIM 0) is paired first (dist 0.2)
    # (PSD 2, RIM 1) is paired next (dist 0.4)
    # (PSD 0, RIM 0) cannot be paired because RIM 0 is already taken.
    points = pd.DataFrame(
        {
            "X": [0.0, 0.3, 0.1, 1.0, 1.4],
            "Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "C": ["PSD", "RIM", "PSD", "RIM", "PSD"],
            "cluster_id": [0, 0, 1, 1, 2],
        }
    )

    new_points, pairs = pair_cluster(points, channels=["PSD", "RIM"], distance=0.6)

    # There should be exactly 2 synapses, neither of which has missing markers
    assert len(pairs) == 2
    synapses_in_points = new_points[new_points["synapse_id"] >= 0]
    grouped = synapses_in_points.groupby("synapse_id")["C"].nunique()
    assert (grouped == 2).all()

    # PSD cluster 0 must not overwrite synapse_id of RIM cluster 0
    assert (new_points[new_points["cluster_id"] == 0]["synapse_id"] == -1).any()


def make_points(rows: list[tuple]) -> pd.DataFrame:
    """Helper to build a points DataFrame from (X, Y, Z, C, cluster_id) tuples."""
    return pd.DataFrame(rows, columns=["X", "Y", "Z", "C", "cluster_id"])


def test_hungarian_beats_greedy():
    """
    Classic case where greedy fails but Hungarian finds the global minimum:
      - A0 at (0, 0, 0), A1 at (2, 0, 0)
      - B0 at (1.9, 0, 0), B1 at (5, 0, 0)

    Distances:
      d(A0, B0) = 1.9, d(A0, B1) = 5.0
      d(A1, B0) = 0.1, d(A1, B1) = 3.0

    Greedy picks (A1, B0) = 0.1 first, forcing (A0, B1) = 5.0 -> total = 5.1.
    Hungarian picks (A0, B0) = 1.9 and (A1, B1) = 3.0 -> total = 4.9.
    """
    pts = make_points(
        [
            (0.0, 0.0, 0.0, "A", 0),
            (2.0, 0.0, 0.0, "A", 1),
            (1.9, 0.0, 0.0, "B", 0),
            (5.0, 0.0, 0.0, "B", 1),
        ]
    )

    out_pts, pairs = pair_cluster(pts, channels=["A", "B"], distance=6.0)

    assert len(pairs) == 2
    # Pairs are sorted by ascending distance
    assert pairs.iloc[0].to_dict() == pytest.approx({"A": 0, "B": 0, "distance": 1.9})
    assert pairs.iloc[1].to_dict() == pytest.approx({"A": 1, "B": 1, "distance": 3.0})

    # Verify synapse_id assignment back onto points
    expected_synapses = [0, 1, 0, 1]
    assert out_pts["synapse_id"].tolist() == expected_synapses
    assert out_pts["synapse_id"].dtype == pd.Int64Dtype()


def test_disconnected_components_with_penalty():
    """Tests rectangular / disconnected bipartite graphs where some clusters."""
    pts = make_points(
        [
            # Component 1: A0 near B0 (dist 1.0)
            (0.0, 0.0, 0.0, "A", 0),
            (1.0, 0.0, 0.0, "B", 0),
            # Component 2: A1 and B1 are close to each other (dist 1.5), far from Component 1
            (50.0, 0.0, 0.0, "A", 1),
            (51.5, 0.0, 0.0, "B", 1),
            # Unpaired outlier in channel A
            (100.0, 0.0, 0.0, "A", 2),
        ]
    )

    out_pts, pairs = pair_cluster(pts, channels=["A", "B"], distance=2.0)

    assert len(pairs) == 2
    assert set(zip(pairs["A"], pairs["B"])) == {(0, 0), (1, 1)}
    assert pairs["distance"].tolist() == pytest.approx([1.0, 1.5])

    # Cluster A2 should remain unpaired (-1)
    assert out_pts.loc[4, "synapse_id"] == -1


def test_noise_and_ignored_channels():
    """Points with cluster_id < 0 or from unselected channels must get synapse_id = -1."""
    pts = make_points(
        [
            (0.0, 0.0, 0.0, "A", -1),  # Noise point right on top of B0
            (0.0, 0.0, 0.0, "B", 0),
            (0.0, 0.0, 0.0, "C", 0),  # Unselected channel
            (1.0, 0.0, 0.0, "A", 0),  # Valid cluster A0
        ]
    )

    out_pts, pairs = pair_cluster(pts, channels=["A", "B"], distance=2.0)

    assert len(pairs) == 1
    assert pairs.iloc[0].to_dict() == pytest.approx({"A": 0, "B": 0, "distance": 1.0})
    assert out_pts["synapse_id"].tolist() == [-1, 0, -1, 0]


def test_no_pairs_within_threshold():
    """Returns empty pairs DataFrame and all -1 synapse IDs when nothing is within `distance`."""
    pts = make_points(
        [
            (0.0, 0.0, 0.0, "A", 0),
            (10.0, 0.0, 0.0, "B", 0),
        ]
    )

    out_pts, pairs = pair_cluster(pts, channels=["A", "B"], distance=5.0)

    assert pairs.empty
    assert list(pairs.columns) == ["A", "B", "distance"]
    assert (out_pts["synapse_id"] == -1).all()


def test_empty_or_missing_channel():
    """Handles empty DataFrames or missing channels gracefully without mutating input."""
    empty_pts = make_points([])
    out_empty, pairs_empty = pair_cluster(empty_pts, channels=["A", "B"], distance=5.0)
    assert out_empty.empty
    assert "synapse_id" in out_empty.columns
    assert pairs_empty.empty

    single_ch_pts = make_points([(0.0, 0.0, 0.0, "A", 0)])
    out_single, pairs_single = pair_cluster(
        single_ch_pts, channels=["A", "B"], distance=5.0
    )
    assert "synapse_id" not in single_ch_pts.columns  # Original input not mutated
    assert out_single["synapse_id"].tolist() == [-1]
    assert pairs_single.empty


def test_custom_column_names():
    """Respects custom `cluster_column` and `pair_column` names."""
    pts = pd.DataFrame(
        {
            "X": [0.0, 0.5],
            "Y": [0.0, 0.0],
            "Z": [0.0, 0.0],
            "C": ["pre", "post"],
            "obj_id": [10, 20],
        }
    )

    out_pts, pairs = pair_cluster(
        pts,
        channels=["pre", "post"],
        distance=1.0,
        cluster_column="obj_id",
        pair_column="matched_pair",
    )

    assert "matched_pair" in out_pts.columns
    assert out_pts["matched_pair"].tolist() == [0, 0]
    assert pairs.iloc[0].to_dict() == pytest.approx(
        {"pre": 10, "post": 20, "distance": 0.5}
    )


@pytest.mark.parametrize("bad_channels", [[], ["A"], ["A", "B", "C"], ["A", "A"]])
def test_invalid_channels_raises(bad_channels):
    """Raises ValueError if channels does not have 2 distinct entries."""
    pts = make_points([(0.0, 0.0, 0.0, "A", 0)])
    with pytest.raises(ValueError, match="two distinct"):
        pair_cluster(pts, channels=bad_channels, distance=1.0)


if __name__ == "__main__":
    test_clean_points()
    # test_clean_points_outliers()
