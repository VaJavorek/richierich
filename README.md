# Richierich

Investor-facing Dash/Plotly dashboard for screening future European holiday resort locations under climate-change scenarios.

The app reads Copernicus climate indicator NetCDF files from a sibling `../data/` folder and keeps a local ignored `.climate_cache/` of processed slices for responsive interactions. Comfortable resort days are estimated from monthly aggregate files of daily minimum and maximum temperatures, keeping the map and ranking logic consistent without requiring raw daily time series.

## Dashboard Contents

- Shared target year/month controls for the full dashboard.
- Main Europe grid map for monthly mean temperature.
- Linked mini maps for mosquito season, heat risk, and dryness risk.
- Optimal Area Finder with one-month or all-year climate limits, comfort-temperature settings, minimum estimated optimal days, and minimum temperature-change filtering.
- Weighted custom ranking chart for the top 100 Finder-approved candidate areas, with stacked opportunity-score components and vertical scrolling.
- Parallel coordinates-style climate profile chart with selectable cluster size.
- Scatterplot for variable-pair correlation scouting.
- Shared selection state across maps, ranking, scatterplot, and parallel chart.
- Large focus view for every chart, with focus-window selections linked back into the dashboard.

## Finder And Ranking Logic

- The Finder can evaluate either one selected month or the full year.
- Estimated optimal days are calculated from daily minimum and maximum temperature monthly aggregates. For each month, the app estimates a daily-mean temperature range and counts the proportional overlap with the user-selected comfort window.
- Mosquito, hot, dry, and tropical-night filters cap the allowed number of period days for those risks.
- The minimum temperature-change filter keeps cells where projected warming from the baseline to the target year is high enough to support the investment thesis.
- Custom Ranking uses the same Finder year, month, comfort window, and filters. It ranks only cells already selected by the Finder; if none match, the chart prompts the user to relax the Finder limits.
- Ranking areas are grouped with the cluster size selected in the parallel coordinates chart.

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
- Cache keys are versioned; remove `.climate_cache/` if the underlying NetCDF files are replaced and you want to rebuild every processed slice.
- The optional `premium` branch carries the luxury visual theme work. `main` stays as the lean functional baseline.
