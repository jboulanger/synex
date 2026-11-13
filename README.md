# SynXM

SynXM is a Python package for detecting and characterizing synaptic-marker
puncta in three-dimensional, multichannel expansion-microscopy images. It
provides axial-drift correction, puncta detection, spatial clustering,
cross-channel association, summary statistics, and diagnostic plots.

The current analysis configuration is intended for four channels:
PRRT1, GluA1 (A1), PSD-95 (PSD), and RIM 1/2 (RIM). RIM is the presynaptic
scaffold marker; the other three markers are analyzed as postsynaptic markers.

> **Publication/reproducibility note:** Confirm that the code, parameter values,
> and manuscript methods describe the same analysis before using results in a
> publication. In particular, this workflow groups PRRT1 and GluA1 puncta as
> many-to-many connected components, while the current methods text describes
> one-to-one assignment. The workflow also does not export a dedicated
> four-marker-positive synapse count.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- Four-channel 3D image data readable by BioIO (CZI is the configured example
  format)

The workflow uses physical image coordinates for spatial measurements. Ensure
that the input image has valid spatial calibration and that its channel order
matches the channel names supplied to SynXM. The configured workflow processes
the first time point when a time dimension is present.

## Installation

Obtain the source code from the project repository, then install the project
and its dependencies from the repository root:

```bash
uv sync
```

The public repository URL is not configured in this checkout. Add the released
repository URL here before publication.

## Quick start

Run the command-line workflow on one image:

```bash
uv run synxm path/to/image.czi path/to/output
```

To supply analysis parameters:

```bash
uv run synxm path/to/image.czi path/to/output --parameters parameters.yaml
```

The channel names are assigned to image channels in the order listed. Change
the order to match the acquisition metadata when needed.

Example `parameters.yaml`:

```yaml
channels: [PRRT, A1, PSD, RIM]
detection:
  PRRT: {scale_um: 0.4, specificity: 6}
  A1: {scale_um: 0.3, specificity: 6}
  PSD: {scale_um: 0.1, specificity: 8}
  RIM: {scale_um: 0.1, specificity: 6}
dbscan:
  channels: [RIM, PSD]
  eps: 1.0
  min_samples: 3
synapse:
  channels: [RIM, PSD]
  distance: 1.0
pairs:
  channels: [A1, PRRT]
  distance: 0.5
```

Distances and detection scales are in micrometres. The defaults above reflect
the current analysis configuration; they are not universal settings for other
microscopes, specimens, or expansion protocols. Use the same validated
configuration for all samples in a study, and report any changes.

## Analysis workflow

1. Read the image with BioIO and assign the configured channel labels.
2. Estimate and correct lateral shifts between successive Z planes.
3. Detect puncta independently in each channel with a 3D Laplacian-of-Gaussian
   filter, local-maximum detection, and watershed segmentation.
4. Cluster RIM and PSD puncta separately with DBSCAN. The default radius is
   1.0 µm, and the minimum cluster size is three puncta.
5. Match RIM and PSD clusters one-to-one using their minimum inter-punctum
   distance, with a default maximum distance of 1.0 µm. Matched pairs define
   synapses.
6. Group PRRT and A1 puncta within 0.5 µm as many-to-many connected components.
7. Associate puncta with nearby synapses and calculate cross-channel distances.
8. Export per-punctum, per-synapse, and specimen-level results, along with
   drift-correction diagnostics and spatial-randomization statistics.

The complete parameter values and method descriptions are in
[`methods/methods.md`](methods/methods.md). Review that document alongside the
code before reporting the randomization tests: the analysis estimates spatial
support from detected puncta, and those null-model assumptions should be
assessed for each dataset.

## Output files

The output directory contains the following files for an input named
`image.czi`:

| File | Contents |
| --- | --- |
| `image_parameters.yaml` | Parameters used for the run |
| `image_points.csv` | Detected puncta, channel, coordinates, intensity, volume, cluster and association IDs, and calculated distances |
| `image_clusters.csv` | RIM and PSD cluster summaries |
| `image_synapses.csv` | Matched RIM–PSD cluster pairs and distances |
| `image_pairs.csv` | PRRT–A1 puncta associations and distances |
| `image_s_stats.csv` | Puncta counts and mean distances for each synapse |
| `image_g_stats.yaml` | Specimen-level counts, distances, and pair-association statistics |
| `dbscan_pvalues.csv` | Monte Carlo significance estimates for DBSCAN cluster counts |
| `image_drift.csv` | Estimated per-plane shifts |
| `image_drift.jpg` | Drift-correction diagnostic plot |

Coordinates in the puncta table are `Z`, `Y`, and `X`, in physical units
provided by the image metadata. The `C` column contains the configured channel
name. Unassigned cluster, synapse, and pair IDs use `-1`. The association table
for PRRT and A1 represents connected groups, not necessarily one-to-one pairs.

The specimen-level summary includes marker counts and counts of synapses with
A1 and PRRT–A1 associations. It does not currently provide a dedicated count
of synapses positive for all four markers. Calculate and validate that metric
from the per-synapse table if it is required.

`dbscan_pvalues.csv` uses a fixed filename, so a later run in the same output
directory replaces the earlier file. Keep separate output directories for
independent runs if you need to preserve each run's randomization results.

## Reproducibility and interpretation

- Detection settings were selected and visually checked on representative
  data. Confirm detection quality for each new imaging setup and report the
  settings used.
- Puncta coordinates are not corrected for chromatic aberration. Consider
  this when interpreting cross-channel distances.
- Distances are based on the spatial calibration in the image. Report whether
  distances represent post-expansion or pre-expansion scale.
- Randomized p-values depend on the null model and random sampling. Report the
  model, simulation count, and analysis parameters; do not treat puncta or
  synapses within one specimen as independent biological replicates.
- The package reports image/specimen-level summaries. Define the biological
  replicate and perform group-level statistical analyses at the appropriate
  experimental-unit level.

## Tests

Run the test suite with:

```bash
uv run pytest
```

## Citation, license, and data

- **Citation:** Citation metadata is provided in
  [`CITATION.cff`](./CITATION.cff). No publication DOI or public repository URL
  is configured in this checkout; add these when the software is released.
- **License:** SynXM is distributed under the GNU General Public License,
  version 3. See [`LICENSE`](./LICENSE) for the full license text.
- **Code availability:** Add the public repository URL and a release/tag for
  the version used in the publication.
- **Data availability:** No public example dataset or data-access link is
  included here. Add a repository/DOI or a clear access procedure for the
  image data and any derived data needed to reproduce the figures.
