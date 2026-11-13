"""Tests for the SynXM command-line configuration."""

from synxm.__main__ import load_parameters


def test_default_parameters_include_dbscan_channels():
    """Default CLI settings configure DBSCAN for both scaffold channels."""
    parameters = load_parameters(None)

    assert parameters["dbscan"] == {
        "channels": ["RIM", "PSD"],
        "eps": 1.0,
        "min_samples": 3,
    }
