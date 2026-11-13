# Methods

### Analysis
Images were processed using custom Python code (Python 3.13) with scikit-learn, SciPy, BioIO, and Xarray. The code is available on GitHub.

### Image registration and drift correction
Four-channel three-dimensional (3D) confocal image volumes were imported with native physical voxel coordinates using BioIO. Distances are reported based on images in post-expansion scale. To correct for axial and lateral displacement between optical planes acquired sequentially along the $Z$-axis, adjacent planes were aligned using Lucas–Kanade optical flow estimation. For each pair of successive planes, the images were smoothed with a spatial Gaussian kernel ($\sigma = 1\,\text{pixel}$). The 2D spatial gradients ($I_y, I_x$) and inter-plane intensity difference ($I_0$) were calculated from the average image across all channels. Subpixel rigid translations $(\Delta y, \Delta x)$ were estimated analytically from the structure tensor and applied uniformly to all channels using bicubic spline interpolation to produce drift-corrected 3D volumes.

### 3D puncta detection
Fluorescent puncta were detected in 3D in each channel using a Laplacian of Gaussian (LoG) filter and 3D local-maximum detection. Before filtering, images were variance-stabilized using a square-root transform. The filter scale ($\sigma$ in $XY$ and $3\sigma$ in $Z$) and detection-specificity parameter ($\beta$) were manually selected for each channel based on its signal-to-noise ratio and the physical dimensions of the emitters: PRRT1 ($\sigma = 0.4\,\mu\text{m}$, $\beta= 6$), GluA1 ($\sigma = 0.3\,\mu\text{m}$, $\beta= 6$), PSD-95 ($\sigma = 0.1\,\mu\text{m}$, $\beta = 8$), and RIM 1/2 ($\sigma = 0.1\,\mu\text{m}$, $\beta= 6$). Assuming a Laplace distribution for the LoG score, $\beta$ sets the detection threshold and thus the expected false-alarm rate. Adjacent puncta were separated by watershed segmentation, and regions smaller than the volume threshold of $9$ voxels were excluded. For each detected punctum, centroid coordinates $(Z, Y, X)$ in physical space ($\mu\text{m}$) and total intensity and volumes were recorded. Punctum locations were not corrected for possible chromatic aberrations; conservative distance thresholds were therefore used for clustering and pairing below. Detection conditions were tested, optimized, and visually checked on a representative image, and consistently applied to all images.

### Spatial clustering and synapse identification
To distinguish synaptic scaffold assemblies from nonspecific background signals, PSD-95 and RIM 1/2 puncta were clustered independently using Density-Based Spatial Clustering of Applications with Noise (DBSCAN). The interaction radius was $\varepsilon = 1.0\,\mu\text{m}$, and the minimum number of puncta per cluster was $k = 3$. Puncta that did not meet the density criterion were classified as noise. The 3D centroid of each retained cluster was calculated for reference.

Synaptic junctions were defined by the spatial apposition of pre- and postsynaptic scaffold assemblies. Using $k$-d trees, we calculated the Euclidean distances between constituent puncta in each postsynaptic (PSD-95) and presynaptic (RIM 1/2) cluster pair. Clusters were considered candidates for pairing if their minimum inter-punctum distance was:
$$d_{\text{synapse}} \le 1.0\,\mu\text{m}$$
Pre- and postsynaptic cluster pairs were matched using a linear assigment algorithm and assigned a unique synapse identifier.

### Receptor–modulator pairing and synaptic profiling
Spatial associations between PRRT1 and GluA1 puncta were identified using a linear assignment with a maximum Euclidean separation of $0.5\,\mu\text{m}$.

To characterize synaptic molecular composition, puncta from all channels were assigned to a synapse when they fell within $1.0\,\mu\text{m}$ of a constituent PSD or RIM punctum in 3D.

Each punctum was uniquely assigned to the nearest synapse and the Euclidean distance from each punctum to its nearest postsynaptic (PSD-95) and presynaptic (RIM 1/2) punctum was recorded. Synapse-level measures included puncta counts per channel, PRRT1-GluA1 colocalization frequency, and relative spatial distributions. Global specimen-level statistics were also exported for population-level analyses.

### Spatial randomization and colocalization significance testing
To test whether the observed number of DBSCAN clusters ($n_0$) in the RIM 1/2 and PSD-95 channels exceeded that expected under spatial randomness, we used $N=1000$ Monte Carlo simulations. For each simulation,  random points were sampled at the estimated density within the original point-cloud support, defined as the union of spheres with a radius of $5.0\,\mu\text{m}$. We then reran DBSCAN using the same parameters and recorded the number of clusters. The one-tailed empirical Monte Carlo $p$-value was calculated as
$$p = \frac{\sum_{k=1}^N \mathbb{I}(n_k \ge n_0) + 1}{N + 1}$$
where $\mathbb{I}$ is the indicator function.

We used two complementary null models to test whether spatial associations between molecular markers exceeded chance expectations:

1. **Complete spatial randomness (CSR) Poisson model**: Puncta in each channel were modeled as homogeneous spatial Poisson point processes within the effective tissue volume $V$. This volume was estimated by voxelizing the union of spheres (radius $5.0\,\mu\text{m}$) centered on puncta coordinates and restricting it to the image domain. The expected number of random pairs ($\mu$) within an interaction distance of $r = 0.5\,\mu\text{m}$ was calculated as:
   $$\mu = \frac{n_1 n_2}{V} V_{\text{ball}}$$
   where $n_1$ and $n_2$ are the puncta counts in the two channels, and $V_{\text{ball}} = \frac{4}{3}\pi r^3$. The one-tailed probability of observing at least $n_0$ nearby pairs under CSR was calculated from the Poisson upper tail:
   $$p = 1 - \sum_{k=0}^{n_0 - 1} \frac{\mu^k e^{-\mu}}{k!}$$

2. **Monte Carlo spatial-displacement test**: To preserve within-channel clustering, local emitter-density variation, and tissue geometry while disrupting cross-channel alignment, one channel was translated laterally by random vectors $\mathbf{s} = [0, \Delta y, \Delta x]$ under toroidal periodic boundary conditions. Displacements were sampled uniformly within a maximum lateral range of $10 \times r$, subject to the exclusion criterion $\|\mathbf{s}\| \ge r$ to prevent near-zero shifts. No shifts were applied along the $Z$-axis because the image domain is a narrow slice. For each of $N = 1000$ randomized iterations, we recalculated the number of neighboring pairs ($n_k$) separated by $d \le r$. The one-tailed empirical Monte Carlo $p$-value was calculated as:
   $$p = \frac{\sum_{k=1}^N \mathbb{I}(n_k \ge n_0) + 1}{N + 1}$$
   where $\mathbb{I}$ is the indicator function.
