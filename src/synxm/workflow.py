"""SynXM image-processing workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from bioio import BioImage

import synxm as synxm


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
    shifts, stabilized = synxm.correct_z_drift(data)

    # Detect puncta in all channels
    points = synxm.detect_puncta(stabilized[0], parameters["detection"])

    # Cluster puncta using DBSCAN
    points, clusters = synxm.cluster_dbscan(points, **parameters["dbscan"])

    # Compute a p-values for the number of observed clusters
    pval_clusters = synxm.compute_dbscan_clustering_pvalue(
        points, **parameters["dbscan"]
    )

    # Pair clusters in target channels
    points, synapses = synxm.pair_cluster(points, **parameters["synapse"])

    # Pair puncta in target channels
    points, pairs = synxm.pair_puncta_many_to_many(points, **parameters["pairs"])

    # Associate puncta to synapse
    points = synxm.associate_to_paired_cluster(points, **parameters["synapse"])

    # Compute nearest cross channel pair-wise distances
    points = synxm.compute_distances(points)

    # Compute statistics for the image
    g_stats = synxm.global_stats(points, **parameters["synapse"])

    # Compute statistics per synapse
    s_stats = synxm.synapse_stats(points)

    # Save results
    pval_clusters.to_csv(output_folder / "dbscan_pvalues.csv")
    s_stats.to_csv(output_folder / f"{name}_s_stats.csv")
    points.to_csv(output_folder / f"{name}_points.csv")
    synapses.to_csv(output_folder / f"{name}_synapses.csv")
    clusters.to_csv(output_folder / f"{name}_clusters.csv")
    pairs.to_csv(output_folder / f"{name}_pairs.csv")
    with open(output_folder / f"{name}_g_stats.yaml", "w") as f:
        yaml.safe_dump(g_stats, f)

    synxm.plot_drift_correction(data, stabilized, shifts)
    plt.savefig(output_folder / f"{name}_drift.jpg")
    shifts.to_csv(output_folder / f"{name}_drift.csv")
