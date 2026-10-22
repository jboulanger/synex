## Project Overview
- **Domain**: Pre/post-synaptic puncta analysis in 3D multi-channel expansion confocal microscopy.
- **Goal**: Register Z-drift, detect puncta, cluster synaptic markers, and evaluate co-localization across 4 channels.
- **Environment**: Python >=3.13 managed with `uv`.

## Key Commands
```bash
uv sync 
uv run pytest 
uv run ruff check
```

## Architecture & Data Flow
1. **Z-Drift Correction** (`drift.py`):
   - Input: 5D `xr.DataArray` `(T, C, Z, Y, X)`.
   - Method: Lucas-Kanade gradient translation estimation across sequential Z-planes.
   - Output: `shifts` list and stabilized 5D `xr.DataArray`.
2. **Puncta Detection** (`puncta.py`):
   - Input: Multi-channel image array (`xr.DataArray`).
   - Method: 3D Laplacian of Gaussian (`_detect_puncta_log`) with local maxima filtering; optional 3D Gaussian fitting (`Gaussian3D`).
   - Output: `pandas.DataFrame` with columns `['Z', 'Y', 'X', 'C', 'intensity']`.
3. **Clustering** (`clustering.py`):
   - Input: Spot coordinates table (`pandas.DataFrame`).
   - Method: Per-channel spatial clustering (DBSCAN) to discard background noise.
   - Output: `pandas.DataFrame` with assigned `'class'` labels.
4. **Visualization** (`plot.py`):
   - 2D max projections (`plot_data`) and overlayed multi-channel spot scatter (`plot_points`).

## Coding Standards & Conventions
- Use `xr.DataArray` for gridded N-D image stacks.
- Use `pandas.DataFrame` for spot/puncta detection tables and cluster assignments.
- Coordinate order: `(Z, Y, X)` in physical units / pixel coordinates matching spatial dimensions.
- Run tests via `uv run pytest` before finalizing changes.
- Run linting via `uv run ruff` before finalizing changes.

## Language
- Use 80% ASD-STE100.
