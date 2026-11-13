"""Command-line interface for batch processing in prrt_clustering."""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from bioio import BioImage

import prrt_clustering as pclst


def process(data_path: Path, output_folder: Path, parameters: dict) -> None:
    """Run drift correction, puncta detection, clustering, and pairing for an image."""
    # Create output folder if needed.
    output_folder.mkdir(parents=True, exist_ok=True)
    name = data_path.stem

    # Save parameters
    with open(output_folder / f"{name}_parameters.yaml", "w") as f:
        yaml.safe_dump(parameters, f)
    # Load data
    image = BioImage(data_path)
    data = image.xarray_dask_data.assign_coords({"C": parameters["channels"]})
    # Correct drift using subpixelic LK
    shifts, stabilized = pclst.correct_z_drift(data)
    pclst.plot_drift_correction(data, stabilized, shifts)
    plt.savefig(output_folder / "drift.jpg")
    shifts.to_csv(output_folder / "drift.csv")
    # Detect puncta in all channels
    points = pclst.detect_puncta(stabilized[0], parameters["detection"])
    # Cluster puncta using DBSCAN
    points, clusters = pclst.cluster_dbscan(points, **parameters["dbscan"])
    # Compute a p-values for the number of observed clusters
    pval_clusters = pclst.compute_dbscan_clustering_pvalue(
        points, **parameters["dbscan"]
    )
    pval_clusters.to_csv(output_folder / "dbscan_pvalues.csv")
    # Pair clusters in target channels
    points, synapses = pclst.pair_cluster(points, **parameters["synapse"])
    # Pair puncta in target channels
    points, pairs = pclst.pair_puncta(points, **parameters["pairs"])
    #
    points = pclst.associate_to_paired_cluster(points, **parameters["synapse"])
    points = pclst.compute_distances(points)
    g_stats = pclst.global_stats(points, **parameters["synapse"])
    s_stats = pclst.synapse_stats(points)

    s_stats.to_csv(output_folder / f"{name}_s_stats.csv")
    points.to_csv(output_folder / f"{name}_points.csv")
    synapses.to_csv(output_folder / f"{name}_synapses.csv")
    clusters.to_csv(output_folder / f"{name}_clusters.csv")
    pairs.to_csv(output_folder / f"{name}_pairs.csv")
    with open(output_folder / f"{name}_g_stats.yaml", "w") as f:
        yaml.safe_dump(g_stats, f)
