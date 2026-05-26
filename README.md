# Richierich

Investor-facing Dash/Plotly dashboard for screening future European holiday resort locations using climate-change indicators.

This version is a real-data dashboard preview: it reads Copernicus climate indicator NetCDF files from the sibling `../data/` folder and keeps a local ignored `.climate_cache/` of processed year/month slices for responsive interactions.

## Dashboard Contents

- Main Europe grid map for monthly mean temperature.
- Linked mini maps for mosquito season, heat risk, and dryness risk.
- Shared year/month controls.
- Weighted custom ranking chart for resort candidate scoring.
- Optimal Area Finder with investor-selected climate limits.
- Parallel coordinates-style climate profile chart.
- Scatterplot for variable-pair correlation scouting.
- Shared selection state across maps, ranking, scatterplot, and parallel chart.
- Large focus view for every chart, with focus-window selections linked back into the dashboard.

## Data

The app expects the Copernicus NetCDF files in `../data/`. The available files currently cover 1950-2100 on a regular Europe grid. If more than 20,000 valid grid cells are available for a slice, the data provider applies deterministic spatial sampling to keep Plotly responsive.

Override the sampling limit before running:

```powershell
$env:CLIMATE_MAX_CELLS = "50000"
```

## Run

From this repository folder:

### Option A: venv (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

### Option B: conda (existing)

```powershell
& "C:\Users\20260009\miniconda3\Scripts\conda.exe" run -n shadow python app.py
```

The app binds to `127.0.0.1` and starts on the first available port from `8050`.

## Verify

```powershell
& "C:\Users\20260009\miniconda3\Scripts\conda.exe" run -n shadow python -m py_compile app.py climate_data.py
& "C:\Users\20260009\miniconda3\Scripts\conda.exe" run -n shadow python -c "import app; print('ok')"
```

Or with `venv`:

```powershell
.\.venv\Scripts\python -m py_compile app.py climate_data.py
.\.venv\Scripts\python -c "import app; print('ok')"
```

## Notes

The app fails loudly if the NetCDF dependencies or expected files are missing. It does not fall back to synthetic values.
