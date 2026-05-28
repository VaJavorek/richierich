# Richierich

Investor-facing Dash/Plotly dashboard for screening future European holiday resort locations using climate-change indicators.

The app reads Copernicus climate indicator NetCDF files from a sibling `../data/` folder and keeps a local ignored `.climate_cache/` of processed slices for responsive interactions.

## Dashboard Contents

- Main Europe grid map for monthly mean temperature.
- Linked mini maps for mosquito season, heat risk, and dryness risk.
- Shared year/month controls.
- Weighted custom ranking chart with stacked opportunity-score components.
- Optimal Area Finder with one-month or all-year climate limits and minimum optimal-day thresholds.
- Parallel coordinates-style climate profile chart.
- Scatterplot for variable-pair correlation scouting.
- Shared selection state across maps, ranking, scatterplot, and parallel chart.
- Large focus view for every chart, with focus-window selections linked back into the dashboard.

## Data Layout

Place the repository next to the Copernicus NetCDF data folder:

```text
Data Visualization/
  data/
    01_mean_temperature-projections-monthly-...
    05_tropical_nights-projections-monthly-...
    ...
  richierich/
    app.py
    climate_data.py
    requirements.txt
```

The expected data path is `../data/` relative to this repository. The current files cover 1950-2100 on a regular Europe grid. If more than 20,000 valid grid cells are available for a slice, the data provider applies deterministic spatial sampling to keep Plotly responsive.

To change the sampling limit before running:

```powershell
$env:CLIMATE_MAX_CELLS = "50000"
```

## Install With Conda

Use any environment name you want. The examples below use `climate-dashboard`.

```powershell
conda create -n climate-dashboard python=3.11 -y
conda activate climate-dashboard
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not want to activate the environment, replace `climate-dashboard` with your environment name:

```powershell
conda run -n climate-dashboard python -m pip install -r requirements.txt
conda run -n climate-dashboard python app.py
```

## Run

From the `richierich/` repository folder:

```powershell
python app.py
```

The app binds to `127.0.0.1` and starts on the first available port from `8050`. You can choose a different port:

```powershell
python app.py --port 8060
```

Open the printed local URL in a browser, for example `http://127.0.0.1:8050/`.

## Verify

After installing dependencies:

```powershell
python -m py_compile app.py climate_data.py
python -c "import app; print('ok')"
```

With `conda run`:

```powershell
conda run -n climate-dashboard python -m py_compile app.py climate_data.py
conda run -n climate-dashboard python -c "import app; print('ok')"
```

## Notes

- The app fails loudly if the NetCDF dependencies or expected files are missing. It does not fall back to synthetic values.
- `.climate_cache/`, Python caches, and server logs are ignored by git.
- First load for a new year/month can take longer because the NetCDF slice is read and cached.
