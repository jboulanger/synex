# Methods

### Image registration and drift correction
Four-channel three-dimensional (3D) confocal image volumes were acquired and imported with native physical voxel coordinates using BioIO. All stated distances are reported in the native sample without taking into account the expansion factor.

To eliminate inter-plane axial and lateral sample displacement introduced during sequential optical sectioning along the $Z$-axis, adjacent optical planes were aligned via Lucas–Kanade optical flow estimation. For each successive plane pair, images were smoothed with a spatial Gaussian kernel ($\sigma = 1\,\text{pixel}$), and 2D spatial gradients ($I_y, I_x$) together with inter-plane intensity differences ($I_0$) were computed using all channels intensities. Sub-pixel rigid translation vectors $(\Delta y, \Delta x)$ were solved analytically from the structure tensor and applied uniformly to all channels via bicubic spline interpolation to generate drift-stabilized 3D volumes.

### 3D puncta detection
Fluorescent puncta were identified in 3D across individual channels using a multi-scale Laplacian of Gaussian (LoG) filter coupled with 3D local maxima detection. Filter scale ($\sigma$ in $XY$ and $3\sigma$ in $Z$) and detection specificity thresholds were calibrated to channel-specific signal-to-noise ratios and physical emitter dimensions: PRRT1 ($\sigma = 0.4\,\mu\text{m}$, specificity $= 6$), GluA1 ($\sigma = 0.3\,\mu\text{m}$, specificity $= 6$), PSD-95 ($\sigma = 0.1\,\mu\text{m}$, specificity $= 8$), and RIM1/2 ($\sigma = 0.1\,\mu\text{m}$, specificity $= 6$). Centroid coordinates $(Z, Y, X)$ in physical space ($\mu\text{m}$) and background-subtracted peak amplitudes were recorded for all resolved puncta. Relative puncta location were not corrected for possible chromatic aberrations. To reflect this, clustering and pairing distance are conservative.

### Spatial clustering and synapse identification
To differentiate macromolecular synaptic scaffolds from non-specific background signals, puncta coordinates for PSD-95 and RIM 1/2 channels were independently segmented using Density-Based Spatial Clustering of Applications with Noise (DBSCAN) with an interaction radius $\varepsilon = 1.0\,\mu\text{m}$ and a minimum cluster threshold of $k = 3$ puncta. Puncta failing to meet density criteria were discarded as noise. Centroids of surviving clusters were calculated in 3D space.

Synaptic junctions were defined by the spatial apposition of pre- and post-synaptic scaffold assemblies. Pairwise Euclidean separations between post-synaptic (PSD-95) and pre-synaptic (RIM 1/2) cluster centroids were determined using $k$-d trees. Clusters were paired if inter-centroid distances satisfied:
$$d_{\text{synapse}} \le 1.0\,\mu\text{m}$$
Each validated pre-/post-synaptic pair was indexed with a distinct synapse identifier, and the trans-synaptic scaffold distance was quantified. Synapses failing quality-control criteria ($\ge 3$ puncta in both the pre- and post-synaptic channel) were flagged.

### Receptor–modulator pairing and synaptic profiling
Spatial association between PRRT1 and GluA1 puncta was determined using mutual nearest-neighbor distance queries. Puncta pairs separated by Euclidean distances:
$$d_{\text{pair}} \le 0.5\,\mu\text{m}$$
were classified as paired complexes and assigned a unique pair identifier.

To evaluate synaptic molecular composition, puncta from all channels were assigned to a synapse if their 3D coordinates fell within an inclusion radius of $1.0\,\mu\text{m}$ from either the PSD or RIM scaffold centroid. Euclidean distances from each punctum to the corresponding post-synaptic (PSD) centroid, pre-synaptic (RIM) centroid, and the trans-synaptic axis were computed. Synapse-level metrics—including puncta copy numbers per channel, PRRT1-GluA1 colocalization frequencies, and relative spatial distributions—as well as global specimen-level statistics were exported for population-wide analysis.

### Spatial randomization and colocalization significance testing
To establish whether spatial associations between molecular markers exceeded chance expectations, two complementary null models were implemented:

1. **Complete Spatial Randomness (CSR) Poisson model**: Puncta distributions were modeled as homogeneous spatial Poisson point processes within the effective tissue volume $V$, computed as the voxelized union of spheres ($2.0\,\mu\text{m}$ radius) centered on puncta coordinates to avoid overestimating volume from empty background regions. The expected number of random pairings ($\mu$) within interaction distance $r = 0.5\,\mu\text{m}$ was evaluated analytically as:
   $$\mu = \frac{n_1 n_2}{V} V_{\text{ball}}$$
   where $n_1$ and $n_2$ denote the total puncta counts in the respective channels and $V_{\text{ball}} = \frac{4}{3}\pi r^3$. The statistical significance of observing at least $n_0$ proximity pairs under CSR was determined from the discrete Poisson upper tail:
   $$p = 1 - \sum_{k=0}^{n_0 - 1} \frac{\mu^k e^{-\mu}}{k!}$$

2. **Monte Carlo spatial displacement test**: To preserve intrinsic intra-channel clustering, local emitter density variations, and tissue geometry while decorrelating cross-channel alignments, random lateral translation vectors $\mathbf{s} = [0, \Delta y, \Delta x]$ were applied to one channel. Displacements were sampled uniformly within a maximum lateral envelope ($10 \times r$) with an exclusion threshold ($\|\mathbf{s}\| \ge r$) to prevent spurious self-correlation. Across $N = 100$ independent randomized iterations, neighbor counts $n_k$ satisfying $d \le r$ were re-evaluated, and empirical one-tailed Monte Carlo $p$-values were computed as:
   $$p = \frac{\sum_{k=1}^N \mathbb{I}(n_k \ge n_0) + 1}{N + 1}$$
   where $\mathbb{I}$ is the indicator function.
