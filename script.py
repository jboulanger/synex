"""Batch-process the configured SynXM image folders."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

import synxm

MAX_WORKERS = 4

DATA_ROOT = Path(
    "/media/pandora/Greger_group/Xinyao/PRRT Revision/"
    "PRRTm750 A1rb647 PSDck555 RIMgp488"
)
FOLDERS = [
    DATA_ROOT / "261006 Re-image",
    DATA_ROOT / "261016 Repeat",
    DATA_ROOT / "261007 Repeat",
    DATA_ROOT / "261020 Repeat",
]


def process_image(
    task: tuple[Path, Path, dict],
) -> None:
    """Run SynXM on one image."""
    filename, output_folder, parameters = task
    synxm.process(filename, output_folder, parameters)


def main() -> None:
    """Process all configured images in parallel and aggregate their results."""
    parameters = {
        "channels": ["PRRT", "A1", "PSD", "RIM"],
        "detection": {
            "PRRT": {"scale_um": 0.4, "specificity": 6},
            "A1": {"scale_um": 0.3, "specificity": 8},
            "PSD": {"scale_um": 0.1, "specificity": 8},
            "RIM": {"scale_um": 0.1, "specificity": 6},
        },
        "dbscan": {"channels": ["RIM", "PSD"], "eps": 1.5, "min_samples": 3},
        "synapse": {"distance": 1.0, "channels": ["RIM", "PSD"]},
        "pairs": {"distance": 0.5, "channels": ["A1", "PRRT"]},
        "association": {"distance": 1.0, "channels": ["A1", "PRRT"]},
    }

    output_folder = Path("./results")
    filelist = [
        filename
        for data_path in FOLDERS
        for filename in Path(data_path).glob("**/*.czi")
    ]

    stems = [filename.stem for filename in filelist]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "Input images must have unique filenames because results use "
            "the input filename as their output prefix."
        )

    if filelist:
        tasks = ((filename, output_folder, parameters) for filename in filelist)
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(filelist))) as pool:
            list(pool.map(process_image, tasks))

    # sst = pd.concat(
    #     [
    #         pd.read_csv(x, index_col=0).assign(file=x.stem)
    #         for x in output_folder.glob("*_s_stats.csv")
    #     ],
    #     ignore_index=True,
    # )
    # points = pd.concat(
    #     [
    #         pd.read_csv(x, index_col=0).assign(file=x.stem)
    #         for x in output_folder.glob("*_points.csv")
    #     ],
    #     ignore_index=True,
    # )

    gst = []
    for x in output_folder.glob("*_g_stats.yaml"):
        with open(x, "r") as f:
            gst.append(yaml.safe_load(f))
            gst[-1]["file"] = x.stem
    gst = pd.DataFrame.from_records(gst)
    gst.to_csv(output_folder / "global_stats.csv")


if __name__ == "__main__":
    main()
