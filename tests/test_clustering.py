"""Tests for clustering and nearest neighbor functions."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from prrt_clustering.clustering import (
    clean_points,
    clean_points_outliers,
    create_synthetic_point_cloud,
)


def test_clean_points():
    """Test clustering and nearest neighbor computation on synthetic puncta data."""
    # synthetic point cloud with 4 channels and background
    np.random.seed(42)
    points = create_synthetic_point_cloud()

    # test compute_nearest_neighbor
    clean = clean_points(points)

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


def test_clean_points_outliers():
    """Test noise classification using isolation forest on synthetic puncta data."""
    np.random.seed(42)
    points = create_synthetic_point_cloud()

    clean = clean_points_outliers(points)

    assert len(clean) == len(points)
    assert "noise" in clean.columns
    assert clean["noise"].dtype == bool

    y_true = points["noise"].to_numpy()
    y_pred = clean["noise"].to_numpy()

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert cm.shape == (2, 2)
    assert f1 > 0.70


if __name__ == "__main__":
    test_clean_points()
    test_clean_points_outliers()
