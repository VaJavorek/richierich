from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - exercised by runtime environment
    xr = None
    _XARRAY_IMPORT_ERROR = exc
else:
    _XARRAY_IMPORT_ERROR = None


YEAR_MIN = 1950
YEAR_MAX = 2100
DEFAULT_YEAR = 2070
DEFAULT_MONTH = 7
DEFAULT_MAX_CELLS = 20_000

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent / ".climate_cache"

MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

VARIABLES = {
    "mean_temp": {
        "label": "Mean temperature",
        "unit": "deg C",
        "colorscale": [
            [0.0, "#fde0dd"],
            [0.2, "#fcbba1"],
            [0.4, "#fc9272"],
            [0.6, "#fb6a4a"],
            [0.8, "#de2d26"],
            [1.0, "#a50f15"],
        ],
        "range": (-10, 43),
    },
    "mosquito_days": {
        "label": "Mosquito season",
        "unit": "days/year",
        "colorscale": "Viridis",
        "range": (0, 210),
    },
    "hot_days": {
        "label": "Hot days",
        "unit": "days/month",
        "colorscale": "YlOrRd",
        "range": (0, 31),
    },
    "dry_days": {
        "label": "Consecutive dry days",
        "unit": "days",
        "colorscale": "YlOrBr",
        "range": (0, 65),
    },
    "tropical_nights": {
        "label": "Tropical nights",
        "unit": "nights/month",
        "colorscale": "Magma",
        "range": (0, 31),
    },
    "daily_max_mean": {
        "label": "Daily maximum temp",
        "unit": "deg C",
        "colorscale": "OrRd",
        "range": (-5, 53),
    },
    "daily_min_mean": {
        "label": "Daily minimum temp",
        "unit": "deg C",
        "colorscale": "Blues_r",
        "range": (-20, 36),
    },
    "heat_risk": {
        "label": "Heat risk",
        "unit": "score",
        "colorscale": "Inferno",
        "range": (0, 100),
    },
    "dryness_risk": {
        "label": "Dryness risk",
        "unit": "score",
        "colorscale": "YlOrBr",
        "range": (0, 100),
    },
    "temp_change": {
        "label": "Temperature change",
        "unit": "deg C vs 2020",
        "colorscale": "PuOr",
        "range": (-2, 10),
    },
}

ANALYSIS_VARIABLES = [
    "mean_temp",
    "daily_max_mean",
    "daily_min_mean",
    "hot_days",
    "tropical_nights",
    "mosquito_days",
    "dry_days",
    "heat_risk",
    "dryness_risk",
    "temp_change",
]

DEFAULT_WEIGHTS = {
    "temperature": 35,
    "mosquito": 20,
    "heat": 20,
    "dryness": 15,
    "nights": 10,
}

DEFAULT_FILTERS = {
    "target_year": 2070,
    "target_month": 7,
    "temp_min": 24,
    "temp_max": 30,
    "max_mosquito": 95,
    "max_hot_days": 12,
    "max_dry_days": 34,
    "max_tropical_nights": 10,
    "max_temp_change": 2.6,
}

EUROPE_EXTENT = {
    "lat_min": 34.0,
    "lat_max": 65.0,
    "lon_min": -13.0,
    "lon_max": 38.0,
}

SOURCE_SPECS = {
    "mean_temp": {
        "filename": "01_mean_temperature-projections-monthly-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "mean_temperature",
        "frequency": "monthly",
        "temperature": True,
    },
    "tropical_nights": {
        "filename": "05_tropical_nights-projections-monthly-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "tropical_nights",
        "frequency": "monthly",
        "temperature": False,
    },
    "hot_days": {
        "filename": "06_hot_days-projections-monthly-30deg-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "hot_days",
        "frequency": "monthly",
        "temperature": False,
    },
    "dry_days": {
        "filename": "18_consecutive_dry_days-projections-monthly-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "consecutive_dry_days",
        "frequency": "monthly",
        "temperature": False,
    },
    "daily_max_mean": {
        "filename": "l1_daily_maximum_temperature-projections-monthly-mean-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "daily_maximum_temperature",
        "frequency": "monthly",
        "temperature": True,
    },
    "daily_min_mean": {
        "filename": "l2_daily_minimum_temperature-projections-monthly-mean-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "daily_minimum_temperature",
        "frequency": "monthly",
        "temperature": True,
    },
    "mosquito_days": {
        "filename": "l3_tiger_mosquito_season_length-projections-yearly-rcp_8_5-cclm4_8_17-mpi_esm_lr-r1i1p1-grid-v2.0.nc",
        "var": "tiger_mosquito_season_length",
        "frequency": "yearly",
        "temperature": False,
    },
}


def variable_options() -> list[dict[str, str]]:
    return [{"label": VARIABLES[key]["label"], "value": key} for key in ANALYSIS_VARIABLES]


def month_marks() -> dict[int, str]:
    return {month: label for month, label in MONTH_LABELS.items()}


def year_marks() -> dict[int, str]:
    return {year: str(year) for year in [1950, 1980, 2020, 2060, 2100]}


@lru_cache(maxsize=1)
def base_grid() -> pd.DataFrame:
    source = _load_slice("mean_temp", DEFAULT_YEAR, DEFAULT_MONTH)
    return _frame_from_array(source["lat"], source["lon"], np.isfinite(source["values"]))


@lru_cache(maxsize=512)
def climate_slice(year: int, month: int) -> pd.DataFrame:
    year = int(year)
    month = int(month)
    _validate_time(year, month)

    max_cells = _max_cells()
    cache_file = _cache_path(year, month, max_cells)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    mean = _load_slice("mean_temp", year, month)
    base = _load_slice("mean_temp", min(max(2020, YEAR_MIN), YEAR_MAX), month)
    valid_mask = _valid_mask(mean["lat"], mean["lon"], mean["values"])
    sampled_mask = _sample_mask(valid_mask, max_cells)
    grid = _frame_from_array(mean["lat"], mean["lon"], sampled_mask)

    for key in [
        "mean_temp",
        "daily_max_mean",
        "daily_min_mean",
        "hot_days",
        "tropical_nights",
        "dry_days",
        "mosquito_days",
    ]:
        loaded = _load_slice(key, year, month)
        grid[key] = loaded["values"][sampled_mask]

    grid["temp_change"] = mean["values"][sampled_mask] - base["values"][sampled_mask]
    grid["heat_risk"] = _heat_risk(grid)
    grid["dryness_risk"] = _dryness_risk(grid)
    grid["year"] = year
    grid["month"] = month
    grid["season_label"] = f"{MONTH_LABELS[month]} {year}"

    numeric_columns = ANALYSIS_VARIABLES + ["lat", "lon"]
    grid[numeric_columns] = grid[numeric_columns].round(2)
    grid = grid.dropna(subset=ANALYSIS_VARIABLES).reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(cache_file, index=False)
    return grid


def selected_subset(df: pd.DataFrame, selected_ids: Iterable[str] | None) -> pd.DataFrame:
    selected = set(selected_ids or [])
    if not selected:
        return df.iloc[0:0].copy()
    return df[df["cell_id"].isin(selected)].copy()


def score_candidates(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_weight = sum(max(float(value), 0) for value in weights.values()) or 1
    result = df.copy()

    temperature_score = np.clip(100 - np.abs(result["mean_temp"] - 27.0) * 11.0, 0, 100)
    mosquito_score = np.clip(100 - result["mosquito_days"] / 180 * 100, 0, 100)
    heat_score = np.clip(100 - result["heat_risk"], 0, 100)
    dryness_score = np.clip(100 - result["dryness_risk"], 0, 100)
    nights_score = np.clip(100 - result["tropical_nights"] / 24 * 100, 0, 100)

    result["comfort_score"] = temperature_score
    result["low_mosquito_score"] = mosquito_score
    result["low_heat_score"] = heat_score
    result["low_dryness_score"] = dryness_score
    result["low_night_score"] = nights_score
    result["resort_score"] = (
        temperature_score * float(weights["temperature"])
        + mosquito_score * float(weights["mosquito"])
        + heat_score * float(weights["heat"])
        + dryness_score * float(weights["dryness"])
        + nights_score * float(weights["nights"])
    ) / total_weight

    return result.sort_values("resort_score", ascending=False).reset_index(drop=True)


def optimal_candidates(df: pd.DataFrame, filters: dict[str, float] | None = None) -> pd.DataFrame:
    filters = {**DEFAULT_FILTERS, **(filters or {})}
    mask = (
        (df["mean_temp"] >= float(filters["temp_min"]))
        & (df["mean_temp"] <= float(filters["temp_max"]))
        & (df["mosquito_days"] <= float(filters["max_mosquito"]))
        & (df["hot_days"] <= float(filters["max_hot_days"]))
        & (df["dry_days"] <= float(filters["max_dry_days"]))
        & (df["tropical_nights"] <= float(filters["max_tropical_nights"]))
        & (df["temp_change"] <= float(filters["max_temp_change"]))
    )
    return df[mask].copy()


def normalize_for_parallel(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    normalized = df[["cell_id", "area_name", "lat", "lon"] + variables].copy()
    for variable in variables:
        low, high = VARIABLES[variable]["range"]
        if high == low:
            normalized[variable] = 0.5
        else:
            normalized[variable] = ((normalized[variable] - low) / (high - low)).clip(0, 1)
    return normalized


def _validate_time(year: int, month: int) -> None:
    if year < YEAR_MIN or year > YEAR_MAX:
        raise ValueError(f"Year {year} is outside available Copernicus range {YEAR_MIN}-{YEAR_MAX}.")
    if month not in MONTH_LABELS:
        raise ValueError(f"Month {month} is outside range 1-12.")


def _max_cells() -> int:
    raw = os.environ.get("CLIMATE_MAX_CELLS", str(DEFAULT_MAX_CELLS))
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError("CLIMATE_MAX_CELLS must be an integer.") from exc


def _cache_path(year: int, month: int, max_cells: int) -> Path:
    return CACHE_DIR / f"copernicus_v1_y{year}_m{month:02d}_n{max_cells}.parquet"


def _dataset_path(key: str) -> Path:
    path = DATA_DIR / SOURCE_SPECS[key]["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Missing Copernicus source file: {path}")
    return path


@lru_cache(maxsize=len(SOURCE_SPECS))
def _open_dataset(key: str):
    if xr is None:
        raise RuntimeError(
            "Real Copernicus data requires xarray plus h5netcdf/netCDF4 dependencies."
        ) from _XARRAY_IMPORT_ERROR

    path = _dataset_path(key)
    try:
        return xr.open_dataset(path, engine="h5netcdf", decode_times=True, chunks=None)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot read {path}. If this is running inside the Codex sandbox, run the app normally "
            "from the user session or make sure the OneDrive file is locally available."
        ) from exc


@lru_cache(maxsize=1024)
def _load_slice(key: str, year: int, month: int) -> dict[str, np.ndarray]:
    spec = SOURCE_SPECS[key]
    ds = _open_dataset(key)
    time_index = _time_index(key, year, month if spec["frequency"] == "monthly" else None)
    da = ds[spec["var"]].isel(time=time_index)
    values = da.values.astype(float)
    if spec["temperature"]:
        values = _to_celsius(values, da.attrs.get("units", ""))
    return {
        "values": values,
        "lat": ds["lat"].values.astype(float),
        "lon": ds["lon"].values.astype(float),
    }


@lru_cache(maxsize=512)
def _time_index(key: str, year: int, month: int | None) -> int:
    ds = _open_dataset(key)
    years = []
    months = []
    for value in ds["time"].values:
        years.append(int(getattr(value, "year", str(value)[:4])))
        months.append(int(getattr(value, "month", 1 if month is None else str(value)[5:7])))

    for index, (candidate_year, candidate_month) in enumerate(zip(years, months)):
        if candidate_year == year and (month is None or candidate_month == month):
            return index
    if month is None:
        raise ValueError(f"Year {year} is not available in {Path(ds.encoding.get('source', 'dataset')).name}.")
    raise ValueError(
        f"Year/month {year}-{month:02d} is not available in {Path(ds.encoding.get('source', 'dataset')).name}."
    )


def _to_celsius(values: np.ndarray, units: str) -> np.ndarray:
    if units.strip().lower() in {"k", "kelvin"}:
        return values - 273.15
    return values


def _valid_mask(lat: np.ndarray, lon: np.ndarray, values: np.ndarray) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return (
        np.isfinite(values)
        & (lat_grid >= EUROPE_EXTENT["lat_min"])
        & (lat_grid <= EUROPE_EXTENT["lat_max"])
        & (lon_grid >= EUROPE_EXTENT["lon_min"])
        & (lon_grid <= EUROPE_EXTENT["lon_max"])
    )


def _sample_mask(valid_mask: np.ndarray, max_cells: int) -> np.ndarray:
    valid_count = int(valid_mask.sum())
    if valid_count <= max_cells:
        return valid_mask
    stride = max(1, int(math.ceil(math.sqrt(valid_count / max_cells))))
    sampled = np.zeros_like(valid_mask, dtype=bool)
    sampled[::stride, ::stride] = True
    sampled &= valid_mask
    while int(sampled.sum()) > max_cells:
        stride += 1
        sampled = np.zeros_like(valid_mask, dtype=bool)
        sampled[::stride, ::stride] = True
        sampled &= valid_mask
    return sampled


def _frame_from_array(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    rows = pd.DataFrame(
        {
            "lat": lat_grid[mask],
            "lon": lon_grid[mask],
        }
    )
    rows["cell_id"] = [f"EU-{index:05d}" for index in range(1, len(rows) + 1)]
    rows["area_name"] = [_area_name(lat_value, lon_value) for lat_value, lon_value in zip(rows["lat"], rows["lon"])]
    return rows[["cell_id", "lat", "lon", "area_name"]]


def _area_name(lat: float, lon: float) -> str:
    if lat < 40 and lon < -2:
        return "Atlantic Iberia"
    if lat < 42 and lon < 5:
        return "Spanish Mediterranean"
    if lat < 43 and lon < 13:
        return "Western Mediterranean"
    if lat < 43:
        return "Adriatic and Aegean"
    if lat < 47 and lon < 5:
        return "Southern France"
    if lat < 48 and lon < 16:
        return "Alpine Lakes"
    if lat < 49:
        return "Danube Coast"
    if lat < 53 and lon < 2:
        return "Channel and Atlantic"
    if lat < 54 and lon < 15:
        return "Central Europe"
    if lat < 56:
        return "Baltic South"
    if lat < 60 and lon < 10:
        return "North Sea"
    if lat < 60:
        return "Baltic North"
    return "Nordic Gateway"


def _heat_risk(df: pd.DataFrame) -> np.ndarray:
    return np.clip(
        df["hot_days"] * 2.4
        + df["tropical_nights"] * 1.6
        + np.clip(df["daily_max_mean"] - 33.0, 0, None) * 4.5,
        0,
        100,
    )


def _dryness_risk(df: pd.DataFrame) -> np.ndarray:
    return np.clip(
        df["dry_days"] * 2.0 + np.clip(df["mean_temp"] - 26.0, 0, None) * 4.0,
        0,
        100,
    )
