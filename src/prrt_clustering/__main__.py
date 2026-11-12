"""Command-line interface for batch processing in prrt_clustering."""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from bioio import BioImage

import prrt_clustering as pclst


def run_checks(points: pd.DataFrame, parameters: dict) -> None:
    """Check that the number of puncta in each component of the synapse."""
    min_samples = parameters["dbscan"]["min_samples"]
    synapse_counts = (
        points.groupby(["synapse_id", "C"], dropna=False)[["cluster_id"]]
        .count()
        .pivot_table(index="synapse_id", columns="C", values="cluster_id")
    )
    channels = [c for c in ["PSD", "RIM"] if c in synapse_counts.columns]
    if len(channels) == 2:
        n = (synapse_counts[channels].min(axis=1) < min_samples).sum()
        if n > 0:
            print(f"Number of puncta in {n} synapse lower than {min_samples}.")


def process(data_path: Path, output_folder: Path, parameters: dict) -> None:
    """Run drift correction, puncta detection, clustering, and pairing for an image."""
    output_folder.mkdir(parents=True, exist_ok=True)
    name = data_path.stem
    with open(output_folder / f"{name}_parameters.yaml", "w") as f:
        yaml.safe_dump(parameters, f)

    image = BioImage(data_path)
    data = image.xarray_dask_data.assign_coords({"C": parameters["channels"]})
    shift, stabilized = pclst.correct_z_drift(data)
    points = pclst.detect_puncta(stabilized[0], parameters["detection"])
    points, clusters = pclst.cluster_dbscan(points, **parameters["dbscan"])
    points, synapses = pclst.pair_cluster(points, **parameters["synapse"])
    points, pairs = pclst.pair_puncta(points, **parameters["pairs"])
    points = pclst.associate_to_paired_cluster(points, **parameters["synapse"])
    points = pclst.compute_distances(points)
    run_checks(points, parameters)
    g_stats = pclst.global_stats(points, **parameters["synapse"])
    s_stats = pclst.synapse_stats(points)

    s_stats.to_csv(output_folder / f"{name}_s_stats.csv")
    points.to_csv(output_folder / f"{name}_points.csv")
    synapses.to_csv(output_folder / f"{name}_synapses.csv")
    clusters.to_csv(output_folder / f"{name}_clusters.csv")
    pairs.to_csv(output_folder / f"{name}_pairs.csv")
    with open(output_folder / f"{name}_g_stats.yaml", "w") as f:
        yaml.safe_dump(g_stats, f)


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
            "dbscan": {"eps": 1.0, "min_samples": 3},
            "synapse": {"distance": 1.0, "channels": ["RIM", "PSD"]},
            "pairs": {"distance": 0.5, "channels": ["A1", "PRRT"]},
        }

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run CLI parsing and processing."""
    parser = argparse.ArgumentParser(
        description="Process 3D multi-channel microscopy image for PRRT clustering."
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
