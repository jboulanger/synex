from pathlib import Path

import pandas as pd
import yaml

import synxm

parameters = {
    "channels": ["PRRT", "A1", "PSD", "RIM"],
    "detection": {
        "PRRT": {"scale_um": 0.4, "specificity": 6},
        "A1": {"scale_um": 0.3, "specificity": 6},
        "PSD": {"scale_um": 0.1, "specificity": 8},
        "RIM": {"scale_um": 0.1, "specificity": 6},
    },
    "dbscan": {"channels": ["RIM", "PSD"], "eps": 1.0, "min_samples": 3},
    "synapse": {"distance": 1.0, "channels": ["RIM", "PSD"]},
    "pairs": {"distance": 0.5, "channels": ["A1", "PRRT"]},
}

output_folder = Path("./results")

folder = [
    "/media/pandora/Greger_group/Xinyao/PRRT Revision/PRRTm750 A1rb647 PSDck555 RIMgp488/261006 Re-image",
    "/media/pandora/Greger_group/Xinyao/PRRT Revision/PRRTm750 A1rb647 PSDck555 RIMgp488/261016 Repeat",
    "/media/pandora/Greger_group/Xinyao/PRRT Revision/PRRTm750 A1rb647 PSDck555 RIMgp488/261007 Repeat",
    "/media/pandora/Greger_group/Xinyao/PRRT Revision/PRRTm750 A1rb647 PSDck555 RIMgp488/261020 Repeat",
]

for data_path in folder:
    for filename in Path(data_path).glob("**/*.czi"):
        print(filename)
        try:
            synxm.process(filename, output_folder, parameters)
        except Exception as e:
            print(e)
            print("fail")


sst = pd.concat(
    [
        pd.read_csv(x, index_col=0).assign(file=x.stem)
        for x in output_folder.glob("*_s_stats.csv")
    ],
    ignore_index=True,
)
points = pd.concat(
    [
        pd.read_csv(x, index_col=0).assign(file=x.stem)
        for x in output_folder.glob("*_points.csv")
    ],
    ignore_index=True,
)


gst = []
for x in output_folder.glob("*_g_stats.yaml"):
    with open(x, "r") as f:
        gst.append(yaml.safe_load(f))
        gst[-1]["file"] = x.stem
gst = pd.DataFrame.from_records(gst)
gst.to_csv(output_folder / "global_stats.csv")
