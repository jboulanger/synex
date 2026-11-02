# Clustering of pre-post synaptic puncta imaged in expanded brain slide imaged in confocal microscopy.

Data
Location: /pandora/Greger_group/Xinyao/PRRT Revision/PRRTm750 A1rb647 PSDck555 RIMgp488
Dimensions: 3D / 4 channels
Channel: 1 pre synpatic marker and 3 post synaptic ones.

Images show pre-post synaptic clusters of fluorescently tagged proteins among a high level of non-specific labelling.

## Installation
```bash
git clone
cd prrt_clustering
uv sync
```
## Testing 
```bash
uv run --extra cu126 --extra test pytest
```

## Workflow
1. Load images with bioio
1. de-drift z stacks
1. Detect puncta in each 3/4 channels, options:
    1. Deconvolve + spot detection?
    1. Franke-Wolfe for 3d spike localization?
1. Cluster markers and discard noise (non-specific label)
1. Compute the distance between clusters across channels and compute the number of 4 synapses showing all 4 markers.

