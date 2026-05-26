from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd


YEAR_MIN = 1940
YEAR_MAX = 2100
DEFAULT_YEAR = 2070
DEFAULT_MONTH = 7

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
        "range": (-5, 36),
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
        "range": (0, 45),
    },
    "daily_min_mean": {
        "label": "Daily minimum temp",
        "unit": "deg C",
        "colorscale": "Blues_r",
        "range": (-12, 28),
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
        "range": (-2, 5),
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


def variable_options() -> list[dict[str, str]]:
    return [{"label": VARIABLES[key]["label"], "value": key} for key in ANALYSIS_VARIABLES]


def month_marks() -> dict[int, str]:
    return {month: label for month, label in MONTH_LABELS.items()}


def year_marks() -> dict[int, str]:
    return {year: str(year) for year in [1940, 1980, 2020, 2060, 2100]}


@lru_cache(maxsize=1)
def base_grid() -> pd.DataFrame:
    rows = []
    index = 1
    for lat in np.arange(35.0, 64.1, 2.0):
        for lon in np.arange(-11.0, 36.1, 2.0):
            if _outside_placeholder_europe(lat, lon):
                continue
            rows.append(
                {
                    "cell_id": f"EU-{index:03d}",
                    "lat": round(float(lat), 3),
                    "lon": round(float(lon), 3),
                    "area_name": _area_name(lat, lon),
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _outside_placeholder_europe(lat: float, lon: float) -> bool:
    if lat < 36 and lon < -5:
        return True
    if lat < 38 and lon > 24:
        return True
    if lat > 60 and lon < -6:
        return True
    if lat > 62 and lon > 25:
        return True
    if lon > 32 and lat > 55:
        return True
    return False


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


@lru_cache(maxsize=512)
def climate_slice(year: int, month: int) -> pd.DataFrame:
    year = int(year)
    month = int(month)
    grid = base_grid().copy()
    values = _compute_climate_values(grid["lat"].to_numpy(), grid["lon"].to_numpy(), year, month)
    base_values = _compute_climate_values(grid["lat"].to_numpy(), grid["lon"].to_numpy(), 2020, month)

    for key, series in values.items():
        grid[key] = series

    grid["temp_change"] = grid["mean_temp"] - base_values["mean_temp"]
    grid["year"] = year
    grid["month"] = month
    grid["season_label"] = f"{MONTH_LABELS[month]} {year}"

    numeric_columns = ANALYSIS_VARIABLES + ["lat", "lon"]
    grid[numeric_columns] = grid[numeric_columns].round(2)
    return grid


def _compute_climate_values(lat: np.ndarray, lon: np.ndarray, year: int, month: int) -> dict[str, np.ndarray]:
    seasonal = np.cos(2 * np.pi * (month - 7) / 12)
    shoulder = np.cos(2 * np.pi * (month - 8) / 12)
    future_years = max(year - 2020, 0)
    warming = 0.020 * (year - 2020) + 0.00035 * future_years**2
    mediterranean = np.clip((47 - lat) / 12, 0, 1)
    continental = np.clip((lon - 8) / 22, 0, 1)
    oceanic = np.clip((4 - np.abs(lon + 2)) / 8, 0, 1)
    terrain = 0.55 * np.sin(lat * 0.72) + 0.45 * np.cos(lon * 0.41)

    base_temp = 14.4 - 0.46 * (lat - 45) + 0.045 * lon + 0.6 * np.sin((lon + 4) * 0.23)
    mean_temp = (
        base_temp
        + 10.4 * seasonal
        + warming * (1 + 0.22 * mediterranean + 0.08 * continental)
        + 0.65 * terrain
        - 1.2 * oceanic * seasonal
    )

    daily_max_mean = mean_temp + 5.0 + 2.2 * seasonal + 1.2 * mediterranean + 0.8 * continental
    daily_min_mean = mean_temp - 5.2 + 0.9 * shoulder + 0.8 * oceanic
    hot_days = np.clip((mean_temp - 22.5) * 2.1 + (daily_max_mean - 30.0) * 1.35, 0, 31)
    tropical_nights = np.clip((daily_min_mean - 18.4) * 3.15 + mediterranean * 3.2, 0, 31)

    dry_base = 6.5 + mediterranean * 14 + continental * 5 - np.clip((lat - 54) / 12, 0, 1) * 4
    dry_days = np.clip(dry_base + (seasonal + 1) * 4.4 + warming * 1.0 + terrain * 1.4, 1, 65)

    wetland = 28 * np.exp(-(((lat - 45.0) / 6.0) ** 2 + ((lon - 14.0) / 8.0) ** 2))
    mild_winter_bonus = np.clip(daily_min_mean + 2, 0, 18) * 2.2
    mosquito_days = np.clip(
        32
        + (base_temp + warming - 9.5) * 7.1
        + wetland
        + mild_winter_bonus
        - dry_days * 0.42,
        0,
        240,
    )

    heat_risk = np.clip(hot_days * 2.5 + tropical_nights * 1.55 + np.clip(daily_max_mean - 33, 0, None) * 4.0, 0, 100)
    dryness_risk = np.clip(dry_days * 1.55 + np.clip(mean_temp - 26, 0, None) * 4.2, 0, 100)

    return {
        "mean_temp": mean_temp,
        "mosquito_days": mosquito_days,
        "hot_days": hot_days,
        "dry_days": dry_days,
        "tropical_nights": tropical_nights,
        "daily_max_mean": daily_max_mean,
        "daily_min_mean": daily_min_mean,
        "heat_risk": heat_risk,
        "dryness_risk": dryness_risk,
    }


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
