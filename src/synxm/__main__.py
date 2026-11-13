"""Command-line interface for SynXM batch processing."""

import argparse
from pathlib import Path

import yaml

from .workflow import process


def load_parameters(path: Path | None) -> dict:
    """Load configuration parameters from YAML or return default parameters."""
    if path is None:
        return {
            "channels": ["PRRT", "A1", "PSD", "RIM"],
            "detection": {
                "PRRT": {"scale_um": 0.4, "specificity": 6},
                "A1": {"scale_um": 0.3, "specificity": 6},
                "PSD": {"scale_um": 0.1, "specificity": 8},
                "RIM": {"scale_um": 0.1, "specificity": 6},
            },
            "dbscan": {
                "channels": ["RIM", "PSD"],
                "eps": 1.0,
                "min_samples": 3,
            },
            "synapse": {"distance": 1.0, "channels": ["RIM", "PSD"]},
            "pairs": {"distance": 0.5, "channels": ["A1", "PRRT"]},
        }

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run CLI parsing and processing."""
    parser = argparse.ArgumentParser(
        description="Analyze synaptic markers in 3D expansion microscopy images."
    )
    parser.add_argument("data", type=Path, help="Path to input image file (e.g. .czi).")
    parser.add_argument("output", type=Path, help="Output directory for results.")
    parser.add_argument(
        "--parameters",
        "-p",
        type=Path,
        default=None,
        help="Path to YAML parameters file (uses defaults if omitted).",
    )
    args = parser.parse_args()

    parameters = load_parameters(args.parameters)
    process(args.data, args.output, parameters)


if __name__ == "__main__":
    main()
