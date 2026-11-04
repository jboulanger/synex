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
1. **Load data as Xarray and annotate channel**
   - Input: Path
   - Output: `xr.DataArray`
2. **Z-Drift Correction** (`drift.py`):
   - Input: 5D `xr.DataArray` `(T, C, Z, Y, X)`.
   - Method: Lucas-Kanade gradient translation estimation across sequential Z-planes.
   - Output: `shifts` list and stabilized 5D `xr.DataArray`.
3. **Puncta Detection** (`puncta.py`):
   - Input: Multi-channel image array (`xr.DataArray`).
   - Method: 3D Laplacian of Gaussian (`_detect_puncta_log`) with local maxima filtering; optional 3D Gaussian fitting (`Gaussian3D`).
   - Output: `pandas.DataFrame` with columns `['Z', 'Y', 'X', 'C', ...]`.
3. **Clustering** (`clustering.py`):
   - Input: Spot coordinates table (`pandas.DataFrame`).
   - Method: Per-channel spatial clustering (DBSCAN) to discard background noise in the channels `"RIM"` and `"PSD"`
   - Output: `pandas.DataFrame` with assigned `'cluster id'` labels
4. **Synapse detection** (`clustering.py`):
   - Input: `pandas.DataFrame` with columns `['Z', 'Y', 'X', 'C', 'cluster id']`.
   - Method: Clustering and proximity of cluster < 1um in the channels `"RIM"` and `"PSD"`
   - Output: `pandas.DataFrame` with columns `['synapse id','synpase distance']`.
5. **Pairing** (`clustering.py`):
   - Input: `pandas.DataFrame` with columns `['Z', 'Y', 'X', 'C',...]`.
   - Method: Clustering and proximity of points < 0.5um in the channels `"A1"` and `"PRRT"`
   - Output: `pandas.DataFrame` with columns `['pair id']`.
6. **Summarize in table**
   - Input: `pandas.DataFrame` with columns `['Z', 'Y', 'X', 'C', 'cluster id', 'synapse id', 'synpase distance', 'pair id', 'pair distance']`
   - Method: For each synapse check the proximity of puncta in other channels
   - Output: `pandas.DataFrame` with columns 
7. **CSR Randomization test**
   - Method: compute the probability of measuring a PRRT or A1 in a synapse
8. **Random shift test**
   - Method: Run the workflow on shifted channels several time.
8. **Visualization** (`plot.py`):
   - 2D max projections (`plot_data`) and overlayed multi-channel spot scatter (`plot_points`).

## Coding Standards & Conventions
- Use `xr.DataArray` for gridded N-D image stacks.
- Use `pandas.DataFrame` for spot/puncta detection tables and cluster assignments.
- Coordinate order: `(Z, Y, X)` in physical units / pixel coordinates matching spatial dimensions.
- Run tests via `uv run pytest` before finalizing changes.
- Run linting via `uv run ruff` before finalizing changes.

## Language
- Use 80% ASD-STE100.

## Guardrails
- **Always do:** Add unit tests in `tests/` matching new function logic.
- **Ask first:** Before adding new 3rd-party dependencies via `uv add`.
- **Never do:** Inspect files outside the working folder.