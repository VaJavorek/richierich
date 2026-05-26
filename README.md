# Richierich

Investor-facing Dash/Plotly dashboard for screening future European holiday resort locations using climate-change indicators.

This version is a placeholder dashboard shell: it uses deterministic synthetic Europe-grid values that mirror the Copernicus climate indicator variables, while keeping the callback and data-provider structure ready for real NetCDF integration from the sibling `../data/` folder.

## Dashboard Contents

- Main Europe grid map for monthly mean temperature.
- Linked mini maps for mosquito season, heat risk, and dryness risk.
- Shared year/month controls.
- Weighted custom ranking chart for resort candidate scoring.
- Optimal Area Finder with investor-selected climate limits.
- Parallel coordinates-style climate profile chart.
- Scatterplot for variable-pair correlation scouting.
- Shared selection state across maps, ranking, scatterplot, and parallel chart.

## Run

From this repository folder:

```powershell
& "C:\Users\20260009\miniconda3\Scripts\conda.exe" run -n shadow python app.py
```

The app binds to `127.0.0.1` and starts on the first available port from `8050`.

## Verify

```powershell
& "C:\Users\20260009\miniconda3\Scripts\conda.exe" run -n shadow python -m py_compile app.py climate_data.py
& "C:\Users\20260009\miniconda3\Scripts\conda.exe" run -n shadow python -c "import app; print('ok')"
```

## Future Real-Data Integration

The source climate files currently live outside this repository in `../data/`. The next implementation step should add a preprocessing/cache layer using `xarray` plus `netCDF4` or `h5netcdf`, then replace the placeholder generator in `climate_data.py` without changing the dashboard layout or interaction model.
