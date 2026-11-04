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
1. Detect puncta in each channels
1. Cluster (DBSCAN) markers in RIM and PSD channels -> cluster id
1. Pair clusters in RIM and PSD
1. Pair points in A1 and PRRT
1. Associate PRRT and A1 puncta to synapse
1. Compute the distance between clusters across channels and compute the number of 4 synapses showing all 4 markers.
1. Summarize counts per synpases
1. Summarize distances 

## Metrics
For each detected puncta:
- channel C
- position X,Y,X
- cluster id
- PSD-RIM cluster pair id
- A1-PRRT pair id
- Distance to A1
- Distance to PSD in synapse
- Distance to RIM in synapse

For each synapse:
- Number of puncta in each channel
- Distances A1 to 

Summary metrics:
- Number of puncta per channel
- Number of clusters in RIM and PSD channels
- Number of pair of clusters in RIM and PSD channels synapses.
- Number of synpase with A1
- Number of synpase with PRRT
