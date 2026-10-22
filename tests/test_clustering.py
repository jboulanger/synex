"""Tests for clustering and nearest neighbor functions."""

import numpy as np
import pandas as pd

from prrt_clustering.clustering import (
    cluster,
    clean_points,
    create_synthetic_point_cloud,
)


def test_clean_points():
    """Test clustering and nearest neighbor computation on synthetic puncta data."""
    # synthetic point cloud with 4 channels and background
    np.random.seed(42)
    points = create_synthetic_point_cloud()

    # test compute_nearest_neighbor
    clean = clean_points(points)

    assert len(clean) < len(points)


if __name__ == "__main__":
    test_clean_points()
