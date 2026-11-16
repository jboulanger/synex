"""SynXM image-processing workflow."""

from itertools import permutations, product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from bioio import BioImage

from .clustering import (
    associate_to_paired_cluster,
    cluster_dbscan,
    compute_dbscan_clustering_pvalue,
    compute_distances,
    compute_monte_carlo_pair_pvalue,
    compute_poisson_pair_pvalue,
    compute_union_of_balls_volume,
    pair_cluster,
    pair_puncta_many_to_many,
)
from .drift import correct_z_drift
from .plot import plot_drift_correction
from .puncta import detect_puncta


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
    - move here the cluster randomization or compute the pairing randomization
      elsewhere.
    """
    required_columns = set(["X", "Y", "Z", "C", "synapse_id", "pair_id"])
    assert required_columns.issubset(points.columns)

    # Precompute the volume of the analysis window for efficiency
    volume_window = compute_union_of_balls_volume(
        points, ball_radius, voxel_size=voxel_size
    )

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


def process(data_path: Path, output_folder: Path, parameters: dict) -> str:
    """Run drift correction, puncta detection, clustering, and pairing for an image."""
    try:
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
        shifts, stabilized = correct_z_drift(data)

        # Detect puncta in all channels, there is only one time points here (ExM).
        points = detect_puncta(stabilized[0], parameters["detection"])

        # Cluster puncta using DBSCAN
        points, clusters = cluster_dbscan(points, **parameters["dbscan"])

        # Compute a p-values for the number of observed clusters
        pval_clusters = compute_dbscan_clustering_pvalue(points, **parameters["dbscan"])

        # Pair clusters in target channels
        points, synapses = pair_cluster(points, **parameters["synapse"])

        # Pair puncta in target channels
        points, pairs = pair_puncta_many_to_many(points, **parameters["pairs"])

        # Associate puncta to synapse
        points = associate_to_paired_cluster(points, **parameters["association"])

        # Compute nearest cross channel pair-wise distances
        points = compute_distances(points)

        # Compute statistics for the image
        g_stats = global_stats(points, **parameters["synapse"])

        # Compute statistics per synapse
        s_stats = synapse_stats(points)

        # Save results
        pval_clusters.to_csv(output_folder / f"{name}_dbscan_pvalues.csv")
        s_stats.to_csv(output_folder / f"{name}_s_stats.csv")
        points.to_csv(output_folder / f"{name}_points.csv")
        synapses.to_csv(output_folder / f"{name}_synapses.csv")
        clusters.to_csv(output_folder / f"{name}_clusters.csv")
        pairs.to_csv(output_folder / f"{name}_pairs.csv")
        with open(output_folder / f"{name}_g_stats.yaml", "w") as f:
            yaml.safe_dump(g_stats, f)

        plot_drift_correction(data, stabilized, shifts)
        plt.savefig(output_folder / f"{name}_drift.jpg")
        shifts.to_csv(output_folder / f"{name}_drift.csv")
        return "OK"
    except Exception as e:
        return f"Fail: {e}"
