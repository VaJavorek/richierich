from __future__ import annotations

import argparse
import calendar
import socket
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from climate_data import (
    ANALYSIS_VARIABLES,
    DEFAULT_FILTERS,
    DEFAULT_MONTH,
    DEFAULT_WEIGHTS,
    DEFAULT_YEAR,
    MONTH_LABELS,
    VARIABLES,
    YEAR_MAX,
    YEAR_MIN,
    climate_slice,
    month_marks,
    score_candidates,
    variable_options,
    year_marks,
)


HIGHLIGHT = "#f4d35e"
INK = "#17212b"
MUTED = "#667782"
CARD_BG = "#fffaf1"
PAPER_BG = "rgba(0,0,0,0)"

GRAPH_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["autoScale2d", "toggleSpikelines"],
}

PARALLEL_BASE_YEAR = 2026
PARALLEL_CLUSTER_LAT_STEP = 2.0
PARALLEL_CLUSTER_LON_STEP = 3.0
PARALLEL_CLUSTER_MODES = {
    "small": "2 deg x 3 deg regional tiles",
    "big": "13 macro climate regions",
}
PARALLEL_AXIS_SPECS = [
    {"key": "temp_m0", "label": "Mean temp {year}", "unit": "deg C", "group": "temp", "band": "mean_temp", "year_index": 0},
    {"key": "temp_m1", "label": "Mean temp {year}", "unit": "deg C", "group": "temp", "band": "mean_temp", "year_index": 1},
    {"key": "temp_m2", "label": "Mean temp {year}", "unit": "deg C", "group": "temp", "band": "mean_temp", "year_index": 2},
    {"key": "temp_m3", "label": "Mean temp {year}", "unit": "deg C", "group": "temp", "band": "mean_temp", "year_index": 3},
    {"key": "daily_min_2026", "label": "Daily min 2026", "unit": "deg C", "group": "temp", "band": "daily_min"},
    {"key": "daily_min_target", "label": "Daily min target", "unit": "deg C", "group": "temp", "band": "daily_min"},
    {"key": "daily_max_2026", "label": "Daily max 2026", "unit": "deg C", "group": "temp", "band": "daily_max"},
    {"key": "daily_max_target", "label": "Daily max target", "unit": "deg C", "group": "temp", "band": "daily_max"},
    {"key": "hot_2026", "label": "Hot days 2026", "unit": "days", "group": "days", "band": "hot"},
    {"key": "hot_target", "label": "Hot days target", "unit": "days", "group": "days", "band": "hot"},
    {"key": "dry_2026", "label": "Dry days 2026", "unit": "days", "group": "days", "band": "dry"},
    {"key": "dry_target", "label": "Dry days target", "unit": "days", "group": "days", "band": "dry"},
    {"key": "mosquito_2026", "label": "Mosquito 2026", "unit": "days", "group": "days", "band": "mosquito"},
    {"key": "mosquito_target", "label": "Mosquito target", "unit": "days", "group": "days", "band": "mosquito"},
]

AREA_LINE_COLORS = [
    "#21475c",
    "#be7f42",
    "#4f8f6b",
    "#9c5c34",
    "#6c5ca8",
    "#2f8f9d",
    "#b85f6a",
    "#748047",
    "#80643e",
    "#526a84",
    "#8a6a96",
    "#5d7f52",
]

RANKING_COMPONENTS = [
    {
        "score": "optimal_temperature_score",
        "contribution": "optimal_temperature_weighted",
        "weight": "temperature",
        "label": "Optimal temperature days",
        "color": "#f4d35e",
    },
    {
        "score": "low_heat_score",
        "contribution": "low_heat_weighted",
        "weight": "heat",
        "label": "Low heat risk",
        "color": "#de2d26",
    },
    {
        "score": "low_dryness_score",
        "contribution": "low_dryness_weighted",
        "weight": "dryness",
        "label": "Low dry risk",
        "color": "#8b9299",
    },
    {
        "score": "low_mosquito_score",
        "contribution": "low_mosquito_weighted",
        "weight": "mosquito",
        "label": "Low mosquito risk",
        "color": "#2f8fce",
    },
    {
        "score": "low_tropical_night_score",
        "contribution": "low_tropical_night_weighted",
        "weight": "nights",
        "label": "Low tropical night risk",
        "color": "#815ac0",
    },
]

FOCUS_CHARTS = {
    "main-map": "Main map",
    "mosquito-map": "Mosquito season",
    "heat-risk-map": "Heat risk",
    "dry-risk-map": "Dryness risk",
    "ranking-chart": "Custom ranking",
    "parallel-chart": "Parallel coordinates",
    "optimal-map": "Optimal Area Finder",
    "scatter-chart": "Correlation scout",
}

ALL_YEAR_VALUE = "all"
MONTH_DAY_LIMIT = 31
YEAR_DAY_LIMIT = 365
OPTIMAL_DAYS_COLORSCALE = [
    [0.0, "#fff7bc"],
    [0.35, "#fec44f"],
    [0.72, "#fe9929"],
    [1.0, "#cc4c02"],
]


def finder_month_options() -> list[dict[str, int | str]]:
    return [{"label": "All year", "value": ALL_YEAR_VALUE}] + [
        {"label": label, "value": month}
        for month, label in MONTH_LABELS.items()
    ]


def card(title: str, subtitle: str, children, class_name: str = "", focus_key: str | None = None):
    header_children = [
        html.Div([html.H2(title), html.P(subtitle)], className="card-title-block"),
    ]
    if focus_key:
        header_children.append(
            html.Button(
                "Focus",
                id=f"focus-open-{focus_key}",
                className="focus-open-button",
                n_clicks=0,
                title=f"Open larger {title} view",
            )
        )
    return html.Section(
        [
            html.Div(
                header_children,
                className="card-head",
            ),
            children,
        ],
        className=f"panel-card {class_name}".strip(),
    )


def frame(title: str, children, class_name: str = "", show_title: bool = True):
    content = []
    if show_title:
        content.append(html.Div(title, className="frame-title"))
    content.append(html.Div(children, className="frame-body"))
    return html.Section(
        content,
        className=f"frame {class_name}".strip(),
    )


def slider_value(
    id_: str,
    label: str,
    value: float,
    minimum: float,
    maximum: float,
    step: float = 1,
    show_value_label: bool = True,
):
    value_class = "slider-value" if show_value_label else "slider-value is-hidden"
    return html.Div(
        [
            html.Div([html.Span(label), html.Strong(id=f"{id_}-label", className=value_class)], className="slider-label"),
            dcc.Slider(
                id=id_,
                min=minimum,
                max=maximum,
                step=step,
                value=value,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="control-row",
    )


def make_layout():
    return html.Div(
        [
            dcc.Store(id="selected-cells-store", data=[]),
            dcc.Store(id="active-time-store", data={"year": DEFAULT_YEAR, "month": DEFAULT_MONTH}),
            dcc.Store(id="ranking-weights-store", data=DEFAULT_WEIGHTS),
            dcc.Store(id="optimal-filter-store", data=DEFAULT_FILTERS),
            dcc.Store(id="focused-chart-store", data=None),
            dcc.Store(id="parallel-area-map-store", data={}),
            dcc.Store(id="ranking-area-map-store", data={}),
            html.Header(
                [
                    html.Div(
                        [
                            html.Div("Richard Vile Resorts", className="eyebrow"),
                            html.H1("European Climate Opportunity Screen"),
                        ],
                        className="hero-copy",
                    ),
                    html.Div(
                        [
                            html.Div("Target window", className="control-kicker"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Year"),
                                            dcc.Slider(
                                                id="year-slider",
                                                min=YEAR_MIN,
                                                max=YEAR_MAX,
                                                step=1,
                                                value=DEFAULT_YEAR,
                                                marks=year_marks(),
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                        ],
                                        className="hero-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Month"),
                                            dcc.Slider(
                                                id="month-slider",
                                                min=1,
                                                max=12,
                                                step=1,
                                                value=DEFAULT_MONTH,
                                                marks=month_marks(),
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                        ],
                                        className="hero-control",
                                    ),
                                ],
                                className="hero-control-grid",
                            ),
                        ],
                        className="hero-controls",
                    ),
                ],
                className="dashboard-hero",
            ),
            html.Main(
                [
                    frame(
                        "Maps",
                        html.Div(
                            [
                                card(
                                    "Main map",
                                    "Mean temperature heatmap across the Copernicus Europe grid.",
                                    dcc.Graph(id="main-map", config=GRAPH_CONFIG, className="map-graph main-map-graph"),
                                    focus_key="main-map",
                                ),
                                html.Div(
                                    [
                                        card(
                                            "Mosquito season",
                                            "Projected tiger mosquito season length.",
                                            dcc.Graph(id="mosquito-map", config=GRAPH_CONFIG, className="map-graph mini-map-graph"),
                                            "mini-map-card",
                                            focus_key="mosquito-map",
                                        ),
                                        card(
                                            "Heat risk",
                                            "Composite risk from hot days and tropical nights.",
                                            dcc.Graph(id="heat-risk-map", config=GRAPH_CONFIG, className="map-graph mini-map-graph"),
                                            "mini-map-card",
                                            focus_key="heat-risk-map",
                                        ),
                                        card(
                                            "Dryness risk",
                                            "Consecutive dry-day pressure for resort operations.",
                                            dcc.Graph(id="dry-risk-map", config=GRAPH_CONFIG, className="map-graph mini-map-graph"),
                                            "mini-map-card",
                                            focus_key="dry-risk-map",
                                        ),
                                    ],
                                    className="mini-map-stack",
                                ),
                            ],
                            className="maps-frame-grid",
                        ),
                        "maps-frame",
                    ),
                    frame(
                        "Custom ranking",
                        card(
                            "Custom ranking",
                            "Weighted annual score across climate opportunity components.",
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            slider_value(
                                                "weight-temperature",
                                                "Optimal temp days",
                                                DEFAULT_WEIGHTS["temperature"],
                                                0,
                                                50,
                                            ),
                                            slider_value("weight-heat", "Low heat risk", DEFAULT_WEIGHTS["heat"], 0, 50),
                                            slider_value("weight-dryness", "Low dry risk", DEFAULT_WEIGHTS["dryness"], 0, 50),
                                            slider_value("weight-mosquito", "Low mosquito risk", DEFAULT_WEIGHTS["mosquito"], 0, 50),
                                            slider_value("weight-nights", "Low tropical night risk", DEFAULT_WEIGHTS["nights"], 0, 50),
                                        ],
                                        className="weight-grid",
                                    ),
                                    dcc.Graph(id="ranking-chart", config=GRAPH_CONFIG, className="ranking-graph"),
                                ],
                                className="ranking-content",
                            ),
                            focus_key="ranking-chart",
                        ),
                        "ranking-frame",
                        show_title=False,
                    ),
                    frame(
                        "Parallel coordinate chart",
                        card(
                            "Parallel coordinates",
                            "Each line is one geographical area cluster across climate milestones.",
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Target year"),
                                                    dcc.Slider(
                                                        id="parallel-target-year",
                                                        min=YEAR_MIN,
                                                        max=YEAR_MAX,
                                                        step=1,
                                                        value=DEFAULT_YEAR,
                                                        marks=year_marks(),
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                    ),
                                                ],
                                                className="parallel-year-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Target month"),
                                                    dcc.Dropdown(
                                                        id="parallel-target-month",
                                                        options=[
                                                            {"label": label, "value": month}
                                                            for month, label in MONTH_LABELS.items()
                                                        ],
                                                        value=DEFAULT_MONTH,
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="small-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Cluster size"),
                                                    dcc.RadioItems(
                                                        id="parallel-cluster-mode",
                                                        options=[
                                                            {"label": "2 deg x 3 deg tiles", "value": "small"},
                                                            {"label": "13 macro regions", "value": "big"},
                                                        ],
                                                        value="small",
                                                        inline=True,
                                                        className="cluster-radio",
                                                        labelStyle={"display": "inline-flex", "alignItems": "center"},
                                                    ),
                                                ],
                                                className="cluster-mode-control",
                                            ),
                                        ],
                                        className="parallel-controls",
                                    ),
                                    dcc.Graph(id="parallel-chart", config=GRAPH_CONFIG, className="parallel-graph"),
                                ],
                                className="parallel-content",
                            ),
                            focus_key="parallel-chart",
                        ),
                        "parallel-frame",
                        show_title=False,
                    ),
                    frame(
                        "Optimal Area Finder",
                        card(
                            "Optimal Area Finder",
                            "Use custom climate limits to find European resort cells with enough comfortable days and manageable heat, drought, nights, and mosquito exposure.",
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Target year"),
                                                    dcc.Slider(
                                                        id="target-year-filter",
                                                        min=YEAR_MIN,
                                                        max=YEAR_MAX,
                                                        step=1,
                                                        value=DEFAULT_FILTERS["target_year"],
                                                        marks=year_marks(),
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                    ),
                                                ],
                                                className="wide-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Target month"),
                                                    dcc.Dropdown(
                                                        id="target-month-filter",
                                                        options=finder_month_options(),
                                                        value=DEFAULT_FILTERS["target_month"],
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="small-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Comfort temperature"),
                                                    dcc.RangeSlider(
                                                        id="temp-range-filter",
                                                        min=15,
                                                        max=35,
                                                        step=0.5,
                                                        value=[DEFAULT_FILTERS["temp_min"], DEFAULT_FILTERS["temp_max"]],
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                    ),
                                                ],
                                                className="wide-control",
                                            ),
                                            slider_value(
                                                "min-optimal-days-filter",
                                                "Min optimal days",
                                                DEFAULT_FILTERS["min_optimal_days"],
                                                1,
                                                MONTH_DAY_LIMIT,
                                                1,
                                                show_value_label=False,
                                            ),
                                            slider_value(
                                                "max-mosquito-filter",
                                                "Max mosquito days",
                                                DEFAULT_FILTERS["max_mosquito"],
                                                1,
                                                MONTH_DAY_LIMIT,
                                                1,
                                                show_value_label=False,
                                            ),
                                            slider_value(
                                                "max-hot-filter",
                                                "Max hot days",
                                                DEFAULT_FILTERS["max_hot_days"],
                                                1,
                                                MONTH_DAY_LIMIT,
                                                1,
                                                show_value_label=False,
                                            ),
                                            slider_value(
                                                "max-dry-filter",
                                                "Max dry days",
                                                DEFAULT_FILTERS["max_dry_days"],
                                                1,
                                                MONTH_DAY_LIMIT,
                                                1,
                                                show_value_label=False,
                                            ),
                                            slider_value(
                                                "max-tropical-filter",
                                                "Max tropical nights",
                                                DEFAULT_FILTERS["max_tropical_nights"],
                                                1,
                                                MONTH_DAY_LIMIT,
                                                1,
                                                show_value_label=False,
                                            ),
                                            slider_value(
                                                "max-change-filter",
                                                "Max temp change",
                                                DEFAULT_FILTERS["max_temp_change"],
                                                0,
                                                5,
                                                0.1,
                                                show_value_label=False,
                                            ),
                                        ],
                                        className="finder-controls",
                                    ),
                                    dcc.Graph(id="optimal-map", config=GRAPH_CONFIG, className="map-graph optimal-map-graph"),
                                ],
                                className="finder-content",
                            ),
                            "finder-card",
                            focus_key="optimal-map",
                        ),
                        "finder-frame",
                        show_title=False,
                    ),
                    frame(
                        "Correlation scout",
                        card(
                            "Correlation scout",
                            "Compare any two variables and brush candidate clusters.",
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("X variable"),
                                                    dcc.Dropdown(
                                                        id="scatter-x",
                                                        options=variable_options(),
                                                        value="mean_temp",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="small-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Y variable"),
                                                    dcc.Dropdown(
                                                        id="scatter-y",
                                                        options=variable_options(),
                                                        value="mosquito_days",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="small-control",
                                            ),
                                        ],
                                        className="scatter-controls",
                                    ),
                                    dcc.Graph(id="scatter-chart", config=GRAPH_CONFIG, className="scatter-graph"),
                                ],
                                className="scatter-content",
                            ),
                            "scatter-card",
                            focus_key="scatter-chart",
                        ),
                        "scatter-frame",
                        show_title=False,
                    ),
                    html.Aside(
                        [
                            html.Div("Shared selection", className="selection-kicker"),
                            html.Div(id="selection-summary", className="selection-summary"),
                            html.Button("Clear selection", id="clear-selection-button", className="clear-button", n_clicks=0),
                        ],
                        className="selection-panel",
                    ),
                ],
                className="dashboard-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2(id="focus-modal-title"),
                                    html.Button("Close", id="focus-close-button", className="focus-close-button", n_clicks=0),
                                ],
                                className="focus-modal-head",
                            ),
                            html.Div(
                                dcc.Graph(id="focus-graph", config=GRAPH_CONFIG, className="focus-graph"),
                                id="focus-modal-body",
                                className="focus-modal-body",
                            ),
                        ],
                        className="focus-modal-card",
                    ),
                ],
                id="focus-modal",
                className="focus-modal is-hidden",
            ),
        ],
        className="app-shell",
    )


def geo_layout(fig: go.Figure, height: int, compact: bool = False) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        margin={"l": 4, "r": 4, "t": 4, "b": 4},
        height=height,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        geo={
            "scope": "europe",
            "projection": {"type": "natural earth"},
            "showland": True,
            "landcolor": "#f8efe0",
            "showocean": True,
            "oceancolor": "#dfe9ed",
            "showlakes": True,
            "lakecolor": "#dfe9ed",
            "coastlinecolor": "rgba(23,33,43,0.28)",
            "countrycolor": "rgba(23,33,43,0.18)",
            "lonaxis": {"range": [-13, 38]},
            "lataxis": {"range": [34, 65]},
            "bgcolor": PAPER_BG,
        },
        showlegend=False,
        dragmode="lasso",
        uirevision="climate-map",
    )
    if compact:
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def map_figure(
    df,
    variable: str,
    selected_ids: list[str] | None,
    compact: bool = False,
    candidate_ids: set[str] | None = None,
    title_suffix: str = "",
    height: int | None = None,
    auto_range: bool = False,
    marker_opacity: float = 1.0,
) -> go.Figure:
    selected = set(selected_ids or [])
    meta = VARIABLES[variable]
    show_risk_colorbar = compact and variable in {"mosquito_days", "heat_risk", "dryness_risk"}
    map_colorbar = {
        "title": meta["unit"],
        "len": 0.62,
        "thickness": 10,
    }
    if show_risk_colorbar:
        map_colorbar = {
            "title": {"text": f"{meta['label']}<br>{meta['unit']}", "font": {"size": 10}},
            "len": 0.78,
            "thickness": 8,
            "x": 1.01,
            "y": 0.5,
            "tickfont": {"size": 9},
        }
    fig = go.Figure()
    if auto_range:
        scale_series = df[variable]
        if candidate_ids is not None:
            scale_series = df.loc[df["cell_id"].isin(candidate_ids), variable]
        cmin = float(scale_series.min())
        cmax = float(scale_series.max())
        if cmin == cmax:
            cmin -= 1.0
            cmax += 1.0
    else:
        cmin, cmax = meta["range"]

    if candidate_ids is None:
        fig.add_trace(
            go.Scattergeo(
                lon=df["lon"],
                lat=df["lat"],
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 11 if compact else 13,
                    "color": df[variable],
                    "colorscale": meta["colorscale"],
                    "cmin": cmin,
                    "cmax": cmax,
                    "line": {"width": 0.4, "color": "rgba(255,255,255,0.65)"},
                    "colorbar": map_colorbar
                    if not compact or show_risk_colorbar
                    else None,
                },
                opacity=marker_opacity,
                customdata=df[["cell_id", "area_name", variable]],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Cell %{customdata[0]}<br>"
                    f"{meta['label']}: "
                    "%{customdata[2]:.2f} "
                    f"{meta['unit']}<extra></extra>"
                ),
                name=meta["label"],
            )
        )
    else:
        candidates = df[df["cell_id"].isin(candidate_ids)]
        fig.add_trace(
            go.Scattergeo(
                lon=df["lon"],
                lat=df["lat"],
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 10,
                    "color": "rgba(102,119,130,0.24)",
                    "line": {"width": 0},
                },
                customdata=df[["cell_id", "area_name", variable]],
                hovertemplate="<b>%{customdata[1]}</b><br>Cell %{customdata[0]}<extra></extra>",
                name="All cells",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lon=candidates["lon"],
                lat=candidates["lat"],
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 14,
                    "color": candidates[variable],
                    "colorscale": meta["colorscale"],
                    "cmin": cmin,
                    "cmax": cmax,
                    "line": {"width": 1.2, "color": "#fff7d6"},
                    "colorbar": {"title": meta["unit"], "len": 0.58, "thickness": 10},
                },
                opacity=marker_opacity,
                customdata=candidates[["cell_id", "area_name", variable]],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Candidate %{customdata[0]}<br>"
                    f"{meta['label']}: "
                    "%{customdata[2]:.2f} "
                    f"{meta['unit']}<extra></extra>"
                ),
                name="Matching cells",
            )
        )

    if selected:
        selected_df = df[df["cell_id"].isin(selected)]
        fig.add_trace(
            go.Scattergeo(
                lon=selected_df["lon"],
                lat=selected_df["lat"],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "size": 18 if not compact else 15,
                    "color": HIGHLIGHT,
                    "line": {"width": 2.2, "color": INK},
                    "opacity": 0.95,
                },
                customdata=selected_df[["cell_id", "area_name", variable]],
                hovertemplate="<b>Selected %{customdata[1]}</b><br>Cell %{customdata[0]}<extra></extra>",
                name="Selected",
            )
        )

    figure_height = height if height is not None else 140 if compact else 545
    fig = geo_layout(fig, height=figure_height, compact=compact)
    if show_risk_colorbar:
        fig.update_layout(margin={"l": 0, "r": 46, "t": 0, "b": 0})
    if title_suffix:
        fig.add_annotation(
            text=title_suffix,
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font={"size": 12, "color": MUTED},
            bgcolor="rgba(255,250,241,0.85)",
            bordercolor="rgba(23,33,43,0.08)",
            borderpad=5,
        )
    return fig


def _normalize_finder_month(value) -> int | str:
    if value == ALL_YEAR_VALUE:
        return ALL_YEAR_VALUE
    return int(value)


def _finder_months(value) -> list[int]:
    month = _normalize_finder_month(value)
    if month == ALL_YEAR_VALUE:
        return list(MONTH_LABELS)
    return [int(month)]


def _finder_day_limit(value) -> int:
    return YEAR_DAY_LIMIT if _normalize_finder_month(value) == ALL_YEAR_VALUE else MONTH_DAY_LIMIT


def _finder_period_label(target_year: int, target_month) -> str:
    month = _normalize_finder_month(target_month)
    if month == ALL_YEAR_VALUE:
        return f"All year {int(target_year)}"
    return f"{MONTH_LABELS[int(month)]} {int(target_year)}"


def _clamp_day_value(value, limit: int) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = limit
    return int(np.clip(numeric, 1, limit))


@lru_cache(maxsize=128)
def finder_period_summary(
    target_year: int,
    target_month,
    temp_min_tenths: int,
    temp_max_tenths: int,
) -> pd.DataFrame:
    target_year = int(target_year)
    target_month = _normalize_finder_month(target_month)
    temp_min = temp_min_tenths / 10
    temp_max = temp_max_tenths / 10
    day_limit = _finder_day_limit(target_month)

    monthly_rows = []
    for month in _finder_months(target_month):
        month_days = calendar.monthrange(target_year, month)[1]
        month_df = climate_slice(target_year, month)[
            [
                "cell_id",
                "area_name",
                "lat",
                "lon",
                "mean_temp",
                "hot_days",
                "dry_days",
                "tropical_nights",
                "mosquito_days",
                "temp_change",
            ]
        ].copy()
        month_df["optimal_days"] = np.where(
            month_df["mean_temp"].between(temp_min, temp_max),
            month_days,
            0,
        )
        month_df["weighted_mean_temp"] = month_df["mean_temp"] * month_days
        month_df["weighted_temp_change"] = month_df["temp_change"] * month_days
        month_df["period_days"] = month_days
        monthly_rows.append(month_df)

    if len(monthly_rows) == 1:
        period = monthly_rows[0].copy()
    else:
        total_days = sum(calendar.monthrange(target_year, month)[1] for month in MONTH_LABELS)
        period = (
            pd.concat(monthly_rows, ignore_index=True)
            .groupby(["cell_id", "area_name", "lat", "lon"], as_index=False)
            .agg(
                {
                    "optimal_days": "sum",
                    "hot_days": "sum",
                    "dry_days": "sum",
                    "tropical_nights": "sum",
                    "mosquito_days": "max",
                    "weighted_mean_temp": "sum",
                    "weighted_temp_change": "sum",
                }
            )
        )
        period["mean_temp"] = period["weighted_mean_temp"] / total_days
        period["temp_change"] = period["weighted_temp_change"] / total_days
        period["period_days"] = total_days

    for column in ["optimal_days", "hot_days", "dry_days", "tropical_nights", "mosquito_days"]:
        period[column] = period[column].clip(lower=0, upper=day_limit)
    period["period_day_limit"] = day_limit
    return period[
        [
            "cell_id",
            "area_name",
            "lat",
            "lon",
            "mean_temp",
            "optimal_days",
            "hot_days",
            "dry_days",
            "tropical_nights",
            "mosquito_days",
            "temp_change",
            "period_days",
            "period_day_limit",
        ]
    ].round(
        {
            "mean_temp": 2,
            "optimal_days": 0,
            "hot_days": 0,
            "dry_days": 0,
            "tropical_nights": 0,
            "mosquito_days": 0,
            "temp_change": 2,
        }
    )


def finder_candidates(period_df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    day_limit = int(period_df["period_day_limit"].iloc[0]) if not period_df.empty else MONTH_DAY_LIMIT
    min_optimal_days = _clamp_day_value(filters.get("min_optimal_days"), day_limit)
    max_mosquito = _clamp_day_value(filters.get("max_mosquito"), day_limit)
    max_hot_days = _clamp_day_value(filters.get("max_hot_days"), day_limit)
    max_dry_days = _clamp_day_value(filters.get("max_dry_days"), day_limit)
    max_tropical_nights = _clamp_day_value(filters.get("max_tropical_nights"), day_limit)
    max_temp_change = float(filters.get("max_temp_change", DEFAULT_FILTERS["max_temp_change"]))

    mask = (
        (period_df["optimal_days"] >= min_optimal_days)
        & (period_df["mosquito_days"] <= max_mosquito)
        & (period_df["hot_days"] <= max_hot_days)
        & (period_df["dry_days"] <= max_dry_days)
        & (period_df["tropical_nights"] <= max_tropical_nights)
        & (period_df["temp_change"] <= max_temp_change)
    )
    return period_df[mask].copy()


def optimal_map_figure(
    period_df: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_ids: list[str] | None,
    title_suffix: str,
    height: int = 360,
) -> go.Figure:
    selected = set(selected_ids or [])
    day_limit = int(period_df["period_day_limit"].iloc[0]) if not period_df.empty else MONTH_DAY_LIMIT
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=period_df["lon"],
            lat=period_df["lat"],
            mode="markers",
            marker={
                "symbol": "square",
                "size": 10,
                "color": "rgba(102,119,130,0.22)",
                "line": {"width": 0},
            },
            customdata=period_df[["cell_id", "area_name", "optimal_days"]],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Cell %{customdata[0]}<br>"
                "Optimal days: %{customdata[2]:.0f}<extra></extra>"
            ),
            name="All cells",
        )
    )

    if not candidates.empty:
        fig.add_trace(
            go.Scattergeo(
                lon=candidates["lon"],
                lat=candidates["lat"],
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 14,
                    "color": candidates["optimal_days"],
                    "colorscale": OPTIMAL_DAYS_COLORSCALE,
                    "cmin": 0,
                    "cmax": day_limit,
                    "line": {"width": 1.2, "color": "#fff7d6"},
                    "colorbar": {"title": "Optimal days", "len": 0.58, "thickness": 10},
                },
                customdata=candidates[
                    [
                        "cell_id",
                        "area_name",
                        "optimal_days",
                        "mean_temp",
                        "mosquito_days",
                        "hot_days",
                        "dry_days",
                        "tropical_nights",
                        "temp_change",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Candidate %{customdata[0]}<br>"
                    "Optimal days: %{customdata[2]:.0f}<br>"
                    "Mean temp: %{customdata[3]:.2f} deg C<br>"
                    "Mosquito days: %{customdata[4]:.0f}<br>"
                    "Hot days: %{customdata[5]:.0f}<br>"
                    "Dry days: %{customdata[6]:.0f}<br>"
                    "Tropical nights: %{customdata[7]:.0f}<br>"
                    "Temp change: %{customdata[8]:.2f} deg C<extra></extra>"
                ),
                name="Matching cells",
            )
        )

    if selected:
        selected_df = period_df[period_df["cell_id"].isin(selected)]
        fig.add_trace(
            go.Scattergeo(
                lon=selected_df["lon"],
                lat=selected_df["lat"],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "size": 18,
                    "color": HIGHLIGHT,
                    "line": {"width": 2.2, "color": INK},
                    "opacity": 0.95,
                },
                customdata=selected_df[["cell_id", "area_name", "optimal_days"]],
                hovertemplate=(
                    "<b>Selected %{customdata[1]}</b><br>"
                    "Cell %{customdata[0]}<br>"
                    "Optimal days: %{customdata[2]:.0f}<extra></extra>"
                ),
                name="Selected",
            )
        )

    fig = geo_layout(fig, height=height)
    if title_suffix:
        fig.add_annotation(
            text=title_suffix,
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font={"size": 12, "color": MUTED},
            bgcolor="rgba(255,250,241,0.85)",
            bordercolor="rgba(23,33,43,0.08)",
            borderpad=5,
        )
    return fig


@lru_cache(maxsize=96)
def _annual_ranking_cell_scores(target_year: int, temp_min_tenths: int, temp_max_tenths: int) -> pd.DataFrame:
    target_year = int(target_year)
    temp_min = temp_min_tenths / 10
    temp_max = temp_max_tenths / 10
    days_in_year = sum(calendar.monthrange(target_year, month)[1] for month in MONTH_LABELS)

    monthly_rows = []
    for month in MONTH_LABELS:
        month_days = calendar.monthrange(target_year, month)[1]
        month_df = climate_slice(target_year, month)[
            [
                "cell_id",
                "area_name",
                "lat",
                "lon",
                "mean_temp",
                "hot_days",
                "dry_days",
                "tropical_nights",
                "mosquito_days",
            ]
        ].copy()
        month_df["optimal_temperature_days"] = np.where(
            month_df["mean_temp"].between(temp_min, temp_max),
            month_days,
            0,
        )
        monthly_rows.append(month_df)

    yearly = (
        pd.concat(monthly_rows, ignore_index=True)
        .groupby(["cell_id", "area_name", "lat", "lon"], as_index=False)
        .agg(
            {
                "optimal_temperature_days": "sum",
                "hot_days": "sum",
                "dry_days": "sum",
                "tropical_nights": "sum",
                "mosquito_days": "max",
            }
        )
    )
    yearly["hot_days"] = yearly["hot_days"].clip(upper=days_in_year)
    yearly["dry_days"] = yearly["dry_days"].clip(upper=days_in_year)
    yearly["tropical_nights"] = yearly["tropical_nights"].clip(upper=days_in_year)
    yearly["mosquito_days"] = yearly["mosquito_days"].clip(upper=days_in_year)

    yearly["optimal_temperature_score"] = yearly["optimal_temperature_days"] / days_in_year * 100
    yearly["low_mosquito_score"] = (days_in_year - yearly["mosquito_days"]) / days_in_year * 100
    yearly["low_dryness_score"] = (days_in_year - yearly["dry_days"]) / days_in_year * 100
    yearly["low_heat_score"] = (days_in_year - yearly["hot_days"]) / days_in_year * 100
    yearly["low_tropical_night_score"] = (days_in_year - yearly["tropical_nights"]) / days_in_year * 100
    return yearly


def ranking_scores(filters, weights):
    filters = {**DEFAULT_FILTERS, **(filters or {})}
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    target_year = int(filters["target_year"])
    temp_min_tenths = int(round(float(filters["temp_min"]) * 10))
    temp_max_tenths = int(round(float(filters["temp_max"]) * 10))
    cell_scores = _annual_ranking_cell_scores(target_year, temp_min_tenths, temp_max_tenths).copy()
    cluster_lookup = _parallel_cluster_lookup(cell_scores)
    area_map = {
        area_key: area_df["cell_id"].tolist()
        for area_key, area_df in cluster_lookup.groupby("parallel_area_key", sort=True)
    }

    clustered = cell_scores.merge(
        cluster_lookup[["lat", "lon", "parallel_area_key", "parallel_area_name"]],
        on=["lat", "lon"],
        how="inner",
    )
    score_columns = [component["score"] for component in RANKING_COMPONENTS]
    grouped = (
        clustered.groupby(["parallel_area_key", "parallel_area_name"], as_index=False)
        .agg({**{column: "mean" for column in score_columns}, "cell_id": "count"})
        .rename(
            columns={
                "parallel_area_key": "area_key",
                "parallel_area_name": "area_name",
                "cell_id": "cell_count",
            }
        )
    )

    total_weight = sum(max(float(weights[component["weight"]]), 0) for component in RANKING_COMPONENTS) or 1
    for component in RANKING_COMPONENTS:
        grouped[component["contribution"]] = (
            grouped[component["score"]] * max(float(weights[component["weight"]]), 0) / total_weight
        )
    grouped["resort_score"] = grouped[[component["contribution"] for component in RANKING_COMPONENTS]].sum(axis=1)
    return grouped.sort_values("resort_score", ascending=False).reset_index(drop=True), area_map


def ranking_figure(filters, weights, selected_ids: list[str] | None) -> tuple[go.Figure, dict[str, list[str]]]:
    ranking, area_map = ranking_scores(filters, weights)
    top = ranking.head(12).copy().sort_values("resort_score")
    selected_areas = _selected_area_keys(area_map, selected_ids)
    selected_indices = [index for index, area_key in enumerate(top["area_key"]) if area_key in selected_areas]
    fig = go.Figure()

    for component in RANKING_COMPONENTS:
        customdata = top[
            [
                "area_key",
                "area_name",
                "resort_score",
                component["score"],
                component["contribution"],
                "cell_count",
            ]
        ]
        fig.add_trace(
            go.Bar(
                x=top[component["contribution"]],
                y=top["area_name"],
                orientation="h",
                name=component["label"],
                marker={
                    "color": component["color"],
                    "line": {"color": "rgba(23,33,43,0.18)", "width": 0.8},
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Total score: %{customdata[2]:.1f}/100<br>"
                    f"{component['label']}: "
                    "%{customdata[3]:.1f}%<br>"
                    "Weighted contribution: %{customdata[4]:.1f}<br>"
                    "%{customdata[5]} grid cells<extra></extra>"
                ),
                selectedpoints=selected_indices or None,
                selected={"marker": {"opacity": 1}},
                unselected={"marker": {"opacity": 0.22}},
            )
        )

    fig.update_layout(
        height=350,
        margin={"l": 142, "r": 24, "t": 34, "b": 34},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        xaxis={"range": [0, 100], "title": "Resort opportunity score", "gridcolor": "rgba(23,33,43,0.09)"},
        yaxis={"title": "", "tickfont": {"size": 11}},
        barmode="stack",
        bargap=0.3,
        dragmode="select",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 9},
        },
    )
    return fig, area_map


def _small_parallel_cluster_lookup(df: pd.DataFrame) -> pd.DataFrame:
    clusters = df[["cell_id", "area_name", "lat", "lon"]].copy()
    lat_origin = float(np.floor(clusters["lat"].min()))
    lon_origin = float(np.floor(clusters["lon"].min()))
    clusters["lat_bin"] = np.floor((clusters["lat"] - lat_origin) / PARALLEL_CLUSTER_LAT_STEP).astype(int)
    clusters["lon_bin"] = np.floor((clusters["lon"] - lon_origin) / PARALLEL_CLUSTER_LON_STEP).astype(int)
    clusters["lat_center"] = lat_origin + (clusters["lat_bin"] + 0.5) * PARALLEL_CLUSTER_LAT_STEP
    clusters["lon_center"] = lon_origin + (clusters["lon_bin"] + 0.5) * PARALLEL_CLUSTER_LON_STEP
    area_slug = clusters["area_name"].str.replace(r"[^A-Za-z0-9]+", "-", regex=True).str.strip("-")
    clusters["parallel_area_key"] = (
        area_slug
        + "-"
        + clusters["lat_bin"].map("{:02d}".format)
        + "-"
        + clusters["lon_bin"].map("{:02d}".format)
    )
    lat_label = clusters["lat_center"].map(lambda value: f"{abs(value):.0f}{'N' if value >= 0 else 'S'}")
    lon_label = clusters["lon_center"].map(lambda value: f"{abs(value):.0f}{'E' if value >= 0 else 'W'}")
    clusters["parallel_area_name"] = clusters["area_name"] + " | " + lat_label + " " + lon_label
    return clusters


def _big_parallel_cluster_lookup(df: pd.DataFrame) -> pd.DataFrame:
    clusters = df[["cell_id", "area_name", "lat", "lon"]].copy()
    area_slug = clusters["area_name"].str.replace(r"[^A-Za-z0-9]+", "-", regex=True).str.strip("-")
    clusters["parallel_area_key"] = "macro-" + area_slug
    clusters["parallel_area_name"] = clusters["area_name"]
    return clusters


def _parallel_cluster_lookup(df: pd.DataFrame, cluster_mode: str = "small") -> pd.DataFrame:
    if cluster_mode == "big":
        return _big_parallel_cluster_lookup(df)
    return _small_parallel_cluster_lookup(df)


def _clustered_means(df: pd.DataFrame, cluster_lookup: pd.DataFrame) -> pd.DataFrame:
    clustered = df.merge(
        cluster_lookup[["lat", "lon", "parallel_area_key"]],
        on=["lat", "lon"],
        how="inner",
    )
    value_columns = [
        "mean_temp",
        "daily_min_mean",
        "daily_max_mean",
        "hot_days",
        "dry_days",
        "mosquito_days",
    ]
    return clustered.groupby("parallel_area_key", sort=True)[value_columns].mean()


def _cluster_value(grouped: pd.DataFrame, area_key: str, column: str) -> float:
    if area_key not in grouped.index:
        return np.nan
    return float(grouped.at[area_key, column])


def _parallel_mean_temp_years(target_year: int) -> list[int]:
    target_year = int(np.clip(int(target_year), YEAR_MIN, YEAR_MAX))
    return [
        int(np.clip(round(year), YEAR_MIN, YEAR_MAX))
        for year in np.linspace(PARALLEL_BASE_YEAR, target_year, 4)
    ]


def _parallel_axis_label(spec: dict[str, Any], target_year: int) -> str:
    label = spec["label"]
    if "year_index" in spec:
        year = _parallel_mean_temp_years(target_year)[spec["year_index"]]
        return label.format(year=year)
    return label.replace("target", str(int(target_year)))


def _parallel_band_ranges() -> list[tuple[str, int, int]]:
    bands: list[tuple[str, int, int]] = []
    start = 0
    current = PARALLEL_AXIS_SPECS[0]["band"]
    for index, spec in enumerate(PARALLEL_AXIS_SPECS[1:], start=1):
        if spec["band"] != current:
            bands.append((current, start, index - 1))
            current = spec["band"]
            start = index
    bands.append((current, start, len(PARALLEL_AXIS_SPECS) - 1))
    return bands


def build_parallel_profile(target_year: int, month: int, cluster_mode: str = "small"):
    target_year = int(target_year)
    month = int(month)
    cluster_mode = cluster_mode if cluster_mode in PARALLEL_CLUSTER_MODES else "small"
    mean_temp_years = _parallel_mean_temp_years(target_year)
    target_df = climate_slice(target_year, month)
    source_by_year = {
        year: climate_slice(year, month)
        for year in sorted(set(mean_temp_years + [PARALLEL_BASE_YEAR, target_year]))
    }
    cluster_lookup = _parallel_cluster_lookup(target_df, cluster_mode)
    cluster_names = (
        cluster_lookup[["parallel_area_key", "parallel_area_name", "area_name"]]
        .drop_duplicates("parallel_area_key")
        .set_index("parallel_area_key")
    )
    area_map = {
        area_key: area_df["cell_id"].tolist()
        for area_key, area_df in cluster_lookup.groupby("parallel_area_key", sort=True)
    }
    grouped_by_year = {
        year: _clustered_means(year_df, cluster_lookup)
        for year, year_df in source_by_year.items()
    }

    rows = []
    for area_key, cell_ids in area_map.items():
        cluster_meta = cluster_names.loc[area_key]
        row = {
            "area_key": area_key,
            "area_name": cluster_meta["parallel_area_name"],
            "macro_area": cluster_meta["area_name"],
            "cell_count": len(cell_ids),
        }
        for index, year in enumerate(mean_temp_years):
            row[f"temp_m{index}"] = _cluster_value(grouped_by_year[year], area_key, "mean_temp")

        base_grouped = grouped_by_year[PARALLEL_BASE_YEAR]
        target_grouped = grouped_by_year[target_year]
        row["daily_min_2026"] = _cluster_value(base_grouped, area_key, "daily_min_mean")
        row["daily_max_2026"] = _cluster_value(base_grouped, area_key, "daily_max_mean")
        row["daily_min_target"] = _cluster_value(target_grouped, area_key, "daily_min_mean")
        row["daily_max_target"] = _cluster_value(target_grouped, area_key, "daily_max_mean")
        row["hot_2026"] = _cluster_value(base_grouped, area_key, "hot_days")
        row["hot_target"] = _cluster_value(target_grouped, area_key, "hot_days")
        row["dry_2026"] = _cluster_value(base_grouped, area_key, "dry_days")
        row["dry_target"] = _cluster_value(target_grouped, area_key, "dry_days")
        row["mosquito_2026"] = _cluster_value(base_grouped, area_key, "mosquito_days")
        row["mosquito_target"] = _cluster_value(target_grouped, area_key, "mosquito_days")
        rows.append(row)

    axis_keys = [spec["key"] for spec in PARALLEL_AXIS_SPECS]
    profile = pd.DataFrame(rows).dropna(subset=axis_keys).sort_values("area_name").reset_index(drop=True)
    area_map = {area_key: area_map[area_key] for area_key in profile["area_key"]}
    return profile, area_map


def _normalize_parallel_value(value: float, value_range: tuple[float, float]) -> float:
    low, high = value_range
    if high <= low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0, 1))


def _parallel_ranges(profile) -> tuple[tuple[float, float], tuple[float, float]]:
    temp_keys = [spec["key"] for spec in PARALLEL_AXIS_SPECS if spec["group"] == "temp"]
    day_keys = [spec["key"] for spec in PARALLEL_AXIS_SPECS if spec["group"] == "days"]
    temp_values = profile[temp_keys].to_numpy(dtype=float)
    day_values = profile[day_keys].to_numpy(dtype=float)
    temp_low = float(np.nanmin(temp_values))
    temp_high = float(np.nanmax(temp_values))
    day_low = 0.0
    day_high = float(max(1, np.nanmax(day_values)))
    temp_pad = max(1.0, (temp_high - temp_low) * 0.08)
    day_pad = max(1.0, day_high * 0.06)
    return (temp_low - temp_pad, temp_high + temp_pad), (day_low, day_high + day_pad)


def _selected_area_keys(area_map: dict[str, list[str]], selected_ids: list[str] | None) -> set[str]:
    selected = set(selected_ids or [])
    if not selected:
        return set()
    return {
        area_key
        for area_key, cell_ids in area_map.items()
        if selected.intersection(cell_ids)
    }


def parallel_figure(
    profile,
    area_map: dict[str, list[str]],
    selected_ids: list[str] | None,
    target_year: int,
    month: int,
    cluster_mode: str = "small",
) -> go.Figure:
    cluster_mode = cluster_mode if cluster_mode in PARALLEL_CLUSTER_MODES else "small"
    selected_areas = _selected_area_keys(area_map, selected_ids)
    labels = [
        _parallel_axis_label(spec, target_year)
        for spec in PARALLEL_AXIS_SPECS
    ]
    temp_range, day_range = _parallel_ranges(profile)
    macro_colors = {
        macro_area: AREA_LINE_COLORS[index % len(AREA_LINE_COLORS)]
        for index, macro_area in enumerate(sorted(profile["macro_area"].unique()))
    }
    fig = go.Figure()

    for index, row in profile.iterrows():
        area_key = row["area_key"]
        is_selected = area_key in selected_areas
        line_color = HIGHLIGHT if is_selected else macro_colors[row["macro_area"]]
        y_values = []
        custom = []
        for spec in PARALLEL_AXIS_SPECS:
            raw_value = float(row[spec["key"]])
            value_range = temp_range if spec["group"] == "temp" else day_range
            unit = "deg C" if spec["group"] == "temp" else "days"
            y_values.append(_normalize_parallel_value(raw_value, value_range))
            custom.append(
                [
                    area_key,
                    row["area_name"],
                    _parallel_axis_label(spec, target_year),
                    raw_value,
                    unit,
                    int(row["cell_count"]),
                    row["macro_area"],
                ]
            )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(PARALLEL_AXIS_SPECS))),
                y=y_values,
                mode="lines+markers",
                line={
                    "color": line_color,
                    "width": 4.0 if is_selected else 1.05,
                },
                marker={
                    "size": 7 if is_selected else 4,
                    "color": line_color,
                    "line": {
                        "width": 1.2 if is_selected else 0,
                        "color": INK,
                    },
                },
                opacity=1 if is_selected else (0.42 if not selected_areas else 0.08),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[6]}<br>"
                    "%{customdata[2]}: %{customdata[3]:.2f} %{customdata[4]}<br>"
                    "%{customdata[5]} grid cells<extra></extra>"
                ),
                name=row["area_name"],
                showlegend=False,
            )
        )

    for x in range(len(PARALLEL_AXIS_SPECS)):
        fig.add_vline(x=x, line={"width": 1, "color": "rgba(23,33,43,0.16)"}, layer="below")
    band_colors = {
        "mean_temp": "rgba(244,211,94,0.24)",
        "daily_min": "rgba(90,169,230,0.18)",
        "daily_max": "rgba(240,146,74,0.18)",
        "hot": "rgba(222,45,38,0.14)",
        "dry": "rgba(102,119,130,0.17)",
        "mosquito": "rgba(129,90,192,0.18)",
    }
    for band, start, end in _parallel_band_ranges():
        fig.add_vrect(
            x0=start - 0.5,
            x1=end + 0.5,
            fillcolor=band_colors[band],
            line_width=0,
            layer="below",
        )

    tick_text = [
        f"{temp_range[0]:.0f} C / {day_range[0]:.0f} d",
        f"{(temp_range[0] + temp_range[1]) / 2:.0f} C / {(day_range[0] + day_range[1]) / 2:.0f} d",
        f"{temp_range[1]:.0f} C / {day_range[1]:.0f} d",
    ]
    fig.update_layout(
        height=430,
        margin={"l": 64, "r": 28, "t": 18, "b": 118},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        xaxis={
            "range": [-0.5, len(PARALLEL_AXIS_SPECS) - 0.5],
            "tickmode": "array",
            "tickvals": list(range(len(PARALLEL_AXIS_SPECS))),
            "ticktext": labels,
            "tickangle": -32,
            "title": "",
            "showgrid": False,
        },
        yaxis={
            "range": [0, 1],
            "title": "Shared normalized scale",
            "tickmode": "array",
            "tickvals": [0, 0.5, 1],
            "ticktext": tick_text,
            "gridcolor": "rgba(23,33,43,0.09)",
        },
        dragmode="lasso",
        hovermode="closest",
        uirevision=f"parallel-{target_year}-{month}-{cluster_mode}",
    )
    fig.add_annotation(
        text=(
            f"{MONTH_LABELS[int(month)]} profile | {len(profile)} {PARALLEL_CLUSTER_MODES[cluster_mode]} | "
            f"mean temperature spans {_parallel_mean_temp_years(target_year)[0]}-"
            f"{_parallel_mean_temp_years(target_year)[-1]} | "
            "temperature axes use C, remaining axes use days"
        ),
        x=0,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font={"size": 12, "color": MUTED},
    )
    return fig


def scatter_figure(df, x_var: str, y_var: str, selected_ids: list[str] | None, weights) -> go.Figure:
    selected = set(selected_ids or [])
    scored = score_candidates(df, weights)
    marker_line = ["#111827" if cell_id in selected else "rgba(255,255,255,0.75)" for cell_id in scored["cell_id"]]
    marker_size = [16 if cell_id in selected else 10 for cell_id in scored["cell_id"]]
    opacity = [1 if not selected or cell_id in selected else 0.24 for cell_id in scored["cell_id"]]

    fig = go.Figure(
        go.Scatter(
            x=scored[x_var],
            y=scored[y_var],
            mode="markers",
            marker={
                "size": marker_size,
                "color": scored["resort_score"],
                "colorscale": "Tealgrn",
                "cmin": 0,
                "cmax": 100,
                "opacity": opacity,
                "line": {"width": 1.5, "color": marker_line},
                "colorbar": {"title": "Score", "len": 0.72, "thickness": 10},
            },
            customdata=scored[["cell_id", "area_name", "resort_score"]],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Cell %{customdata[0]}<br>"
                f"{VARIABLES[x_var]['label']}: "
                "%{x:.2f}<br>"
                f"{VARIABLES[y_var]['label']}: "
                "%{y:.2f}<br>"
                "Score: %{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=650,
        dragmode="lasso",
        margin={"l": 62, "r": 12, "t": 8, "b": 58},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        xaxis={"title": f"{VARIABLES[x_var]['label']} ({VARIABLES[x_var]['unit']})", "gridcolor": "rgba(23,33,43,0.09)"},
        yaxis={"title": f"{VARIABLES[y_var]['label']} ({VARIABLES[y_var]['unit']})", "gridcolor": "rgba(23,33,43,0.09)"},
        hovermode="closest",
    )
    return fig


def selection_summary(df, selected_ids: list[str] | None) -> html.Div:
    selected = set(selected_ids or [])
    if not selected:
        return html.Div(
            [
                html.Strong("No grid cells selected"),
                html.Span("Click a map cell, ranking bar, scatter point, or parallel area line to link the dashboard."),
            ]
        )
    selected_df = df[df["cell_id"].isin(selected)].head(6)
    return html.Div(
        [
            html.Strong(f"{len(selected)} selected grid cell{'s' if len(selected) != 1 else ''}"),
            html.Ul(
                [
                    html.Li(
                        f"{row.cell_id} - {row.area_name}: {row.mean_temp:.1f} deg C, "
                        f"{row.mosquito_days:.0f} mosquito days"
                    )
                    for row in selected_df.itertuples()
                ]
            ),
        ]
    )


def extract_cell_ids(payload: dict[str, Any] | None, area_map: dict[str, list[str]] | None = None) -> list[str]:
    if not payload or not payload.get("points"):
        return []
    ids: list[str] = []
    area_map = area_map or {}
    for point in payload["points"]:
        custom = point.get("customdata")
        if isinstance(custom, (list, tuple)) and custom:
            candidate = custom[0]
        else:
            candidate = custom
        if isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[0]
        if isinstance(candidate, str) and candidate.startswith("EU-"):
            ids.append(candidate)
        elif isinstance(candidate, str) and candidate in area_map:
            ids.extend(area_map[candidate])
    return list(dict.fromkeys(ids))


app = Dash(
    __name__,
    title="Climate Resort Opportunity Screen",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.layout = make_layout


@app.callback(
    Output("focused-chart-store", "data"),
    Input("focus-close-button", "n_clicks"),
    Input("focus-open-main-map", "n_clicks"),
    Input("focus-open-mosquito-map", "n_clicks"),
    Input("focus-open-heat-risk-map", "n_clicks"),
    Input("focus-open-dry-risk-map", "n_clicks"),
    Input("focus-open-ranking-chart", "n_clicks"),
    Input("focus-open-parallel-chart", "n_clicks"),
    Input("focus-open-optimal-map", "n_clicks"),
    Input("focus-open-scatter-chart", "n_clicks"),
    prevent_initial_call=True,
)
def update_focused_chart(_close_clicks, *_focus_clicks):
    prop_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    trigger_id = prop_id.split(".")[0]
    if trigger_id == "focus-close-button":
        return None
    if trigger_id.startswith("focus-open-"):
        return trigger_id.replace("focus-open-", "", 1)
    return None


@app.callback(
    Output("focus-modal", "className"),
    Output("focus-modal-title", "children"),
    Output("focus-graph", "figure"),
    Input("focused-chart-store", "data"),
    Input("main-map", "figure"),
    Input("mosquito-map", "figure"),
    Input("heat-risk-map", "figure"),
    Input("dry-risk-map", "figure"),
    Input("ranking-chart", "figure"),
    Input("parallel-chart", "figure"),
    Input("optimal-map", "figure"),
    Input("scatter-chart", "figure"),
)
def render_focus_modal(
    focused_chart,
    main_map,
    mosquito_map,
    heat_risk_map,
    dry_risk_map,
    ranking_chart,
    parallel_chart,
    optimal_map,
    scatter_chart,
):
    figures = {
        "main-map": main_map,
        "mosquito-map": mosquito_map,
        "heat-risk-map": heat_risk_map,
        "dry-risk-map": dry_risk_map,
        "ranking-chart": ranking_chart,
        "parallel-chart": parallel_chart,
        "optimal-map": optimal_map,
        "scatter-chart": scatter_chart,
    }
    if not focused_chart or focused_chart not in figures or not figures[focused_chart]:
        return "focus-modal is-hidden", "", go.Figure()

    figure = go.Figure(figures[focused_chart])
    figure.update_layout(height=760, autosize=True)

    return (
        "focus-modal",
        FOCUS_CHARTS.get(focused_chart, "Focused chart"),
        figure,
    )


@app.callback(
    Output("active-time-store", "data"),
    Input("year-slider", "value"),
    Input("month-slider", "value"),
)
def update_active_time(year, month):
    return {"year": int(year), "month": int(month)}


@app.callback(
    Output("ranking-weights-store", "data"),
    Output("weight-temperature-label", "children"),
    Output("weight-mosquito-label", "children"),
    Output("weight-heat-label", "children"),
    Output("weight-dryness-label", "children"),
    Output("weight-nights-label", "children"),
    Input("weight-temperature", "value"),
    Input("weight-mosquito", "value"),
    Input("weight-heat", "value"),
    Input("weight-dryness", "value"),
    Input("weight-nights", "value"),
)
def update_weights(temperature, mosquito, heat, dryness, nights):
    weights = {
        "temperature": temperature,
        "mosquito": mosquito,
        "heat": heat,
        "dryness": dryness,
        "nights": nights,
    }
    return weights, f"{temperature:.0f}", f"{mosquito:.0f}", f"{heat:.0f}", f"{dryness:.0f}", f"{nights:.0f}"


@app.callback(
    Output("min-optimal-days-filter", "max"),
    Output("min-optimal-days-filter", "value"),
    Output("max-mosquito-filter", "max"),
    Output("max-mosquito-filter", "value"),
    Output("max-hot-filter", "max"),
    Output("max-hot-filter", "value"),
    Output("max-dry-filter", "max"),
    Output("max-dry-filter", "value"),
    Output("max-tropical-filter", "max"),
    Output("max-tropical-filter", "value"),
    Input("target-month-filter", "value"),
    State("min-optimal-days-filter", "value"),
    State("max-mosquito-filter", "value"),
    State("max-hot-filter", "value"),
    State("max-dry-filter", "value"),
    State("max-tropical-filter", "value"),
)
def update_finder_day_slider_bounds(target_month, min_optimal, max_mosquito, max_hot, max_dry, max_tropical):
    day_limit = _finder_day_limit(target_month)
    return (
        day_limit,
        _clamp_day_value(min_optimal, day_limit),
        day_limit,
        _clamp_day_value(max_mosquito, day_limit),
        day_limit,
        _clamp_day_value(max_hot, day_limit),
        day_limit,
        _clamp_day_value(max_dry, day_limit),
        day_limit,
        _clamp_day_value(max_tropical, day_limit),
    )


@app.callback(
    Output("optimal-filter-store", "data"),
    Output("min-optimal-days-filter-label", "children"),
    Output("max-mosquito-filter-label", "children"),
    Output("max-hot-filter-label", "children"),
    Output("max-dry-filter-label", "children"),
    Output("max-tropical-filter-label", "children"),
    Output("max-change-filter-label", "children"),
    Input("target-year-filter", "value"),
    Input("target-month-filter", "value"),
    Input("temp-range-filter", "value"),
    Input("min-optimal-days-filter", "value"),
    Input("max-mosquito-filter", "value"),
    Input("max-hot-filter", "value"),
    Input("max-dry-filter", "value"),
    Input("max-tropical-filter", "value"),
    Input("max-change-filter", "value"),
)
def update_optimal_filters(
    target_year,
    target_month,
    temp_range,
    min_optimal,
    max_mosquito,
    max_hot,
    max_dry,
    max_tropical,
    max_change,
):
    target_month = _normalize_finder_month(target_month)
    day_limit = _finder_day_limit(target_month)
    min_optimal = _clamp_day_value(min_optimal, day_limit)
    max_mosquito = _clamp_day_value(max_mosquito, day_limit)
    max_hot = _clamp_day_value(max_hot, day_limit)
    max_dry = _clamp_day_value(max_dry, day_limit)
    max_tropical = _clamp_day_value(max_tropical, day_limit)
    filters = {
        "target_year": int(target_year),
        "target_month": target_month,
        "temp_min": float(temp_range[0]),
        "temp_max": float(temp_range[1]),
        "min_optimal_days": min_optimal,
        "max_mosquito": float(max_mosquito),
        "max_hot_days": float(max_hot),
        "max_dry_days": float(max_dry),
        "max_tropical_nights": float(max_tropical),
        "max_temp_change": float(max_change),
    }
    return (
        filters,
        f"{min_optimal:.0f}",
        f"{max_mosquito:.0f}",
        f"{max_hot:.0f}",
        f"{max_dry:.0f}",
        f"{max_tropical:.0f}",
        f"{max_change:.1f} deg C",
    )


@app.callback(
    Output("selected-cells-store", "data"),
    Input("clear-selection-button", "n_clicks"),
    Input("main-map", "clickData"),
    Input("main-map", "selectedData"),
    Input("mosquito-map", "clickData"),
    Input("mosquito-map", "selectedData"),
    Input("heat-risk-map", "clickData"),
    Input("heat-risk-map", "selectedData"),
    Input("dry-risk-map", "clickData"),
    Input("dry-risk-map", "selectedData"),
    Input("ranking-chart", "clickData"),
    Input("ranking-chart", "selectedData"),
    Input("optimal-map", "clickData"),
    Input("optimal-map", "selectedData"),
    Input("parallel-chart", "clickData"),
    Input("parallel-chart", "selectedData"),
    Input("scatter-chart", "clickData"),
    Input("scatter-chart", "selectedData"),
    Input("focus-graph", "clickData"),
    Input("focus-graph", "selectedData"),
    State("parallel-area-map-store", "data"),
    State("ranking-area-map-store", "data"),
    State("selected-cells-store", "data"),
)
def update_selection(
    _clear_clicks,
    main_click,
    main_selected,
    mosquito_click,
    mosquito_selected,
    heat_click,
    heat_selected,
    dry_click,
    dry_selected,
    ranking_click,
    ranking_selected,
    optimal_click,
    optimal_selected,
    parallel_click,
    parallel_selected,
    scatter_click,
    scatter_selected,
    focus_click,
    focus_selected,
    parallel_area_map,
    ranking_area_map,
    current_selection,
):
    prop_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if prop_id == "clear-selection-button.n_clicks":
        return []

    payload_by_prop = {
        "main-map.clickData": main_click,
        "main-map.selectedData": main_selected,
        "mosquito-map.clickData": mosquito_click,
        "mosquito-map.selectedData": mosquito_selected,
        "heat-risk-map.clickData": heat_click,
        "heat-risk-map.selectedData": heat_selected,
        "dry-risk-map.clickData": dry_click,
        "dry-risk-map.selectedData": dry_selected,
        "ranking-chart.clickData": ranking_click,
        "ranking-chart.selectedData": ranking_selected,
        "optimal-map.clickData": optimal_click,
        "optimal-map.selectedData": optimal_selected,
        "parallel-chart.clickData": parallel_click,
        "parallel-chart.selectedData": parallel_selected,
        "scatter-chart.clickData": scatter_click,
        "scatter-chart.selectedData": scatter_selected,
        "focus-graph.clickData": focus_click,
        "focus-graph.selectedData": focus_selected,
    }
    area_map = {**(parallel_area_map or {}), **(ranking_area_map or {})}
    ids = extract_cell_ids(payload_by_prop.get(prop_id), area_map)
    if prop_id.endswith(".selectedData"):
        return ids if ids else current_selection or []
    if ids:
        if prop_id.startswith(("ranking-chart.", "parallel-chart.")) or (prop_id.startswith("focus-graph.") and len(ids) > 1):
            return ids
        return ids[:1]
    return current_selection or []


@app.callback(
    Output("main-map", "figure"),
    Output("mosquito-map", "figure"),
    Output("heat-risk-map", "figure"),
    Output("dry-risk-map", "figure"),
    Output("ranking-chart", "figure"),
    Output("ranking-area-map-store", "data"),
    Output("optimal-map", "figure"),
    Output("scatter-chart", "figure"),
    Output("selection-summary", "children"),
    Input("active-time-store", "data"),
    Input("ranking-weights-store", "data"),
    Input("optimal-filter-store", "data"),
    Input("selected-cells-store", "data"),
    Input("scatter-x", "value"),
    Input("scatter-y", "value"),
)
def render_dashboard(active_time, weights, filters, selected_ids, scatter_x, scatter_y):
    active_time = active_time or {"year": DEFAULT_YEAR, "month": DEFAULT_MONTH}
    weights = weights or DEFAULT_WEIGHTS
    filters = {**DEFAULT_FILTERS, **(filters or {})}
    selected_ids = selected_ids or []

    df = climate_slice(active_time["year"], active_time["month"])
    time_label = f"{MONTH_LABELS[int(active_time['month'])]} {int(active_time['year'])}"
    finder_month = _normalize_finder_month(filters["target_month"])
    finder_df = finder_period_summary(
        int(filters["target_year"]),
        finder_month,
        int(round(float(filters["temp_min"]) * 10)),
        int(round(float(filters["temp_max"]) * 10)),
    )
    candidates = finder_candidates(finder_df, filters)
    candidate_label = (
        f"{len(candidates):,} matching cells | "
        f"{_finder_period_label(int(filters['target_year']), finder_month)} | "
        f">= {int(filters['min_optimal_days'])} optimal days"
    )
    ranking_fig, ranking_area_map = ranking_figure(filters, weights, selected_ids)

    return (
        map_figure(df, "mean_temp", selected_ids, title_suffix=time_label, auto_range=True, marker_opacity=0.65),
        map_figure(df, "mosquito_days", selected_ids, compact=True, title_suffix=time_label, marker_opacity=0.42),
        map_figure(df, "heat_risk", selected_ids, compact=True, title_suffix=time_label, marker_opacity=0.42),
        map_figure(df, "dryness_risk", selected_ids, compact=True, title_suffix=time_label, marker_opacity=0.42),
        ranking_fig,
        ranking_area_map,
        optimal_map_figure(
            finder_df,
            candidates,
            selected_ids,
            title_suffix=candidate_label,
            height=360,
        ),
        scatter_figure(df, scatter_x, scatter_y, selected_ids, weights),
        selection_summary(df, selected_ids),
    )


@app.callback(
    Output("parallel-chart", "figure"),
    Output("parallel-area-map-store", "data"),
    Input("parallel-target-year", "value"),
    Input("parallel-target-month", "value"),
    Input("parallel-cluster-mode", "value"),
    Input("selected-cells-store", "data"),
)
def render_parallel_chart(target_year, target_month, cluster_mode, selected_ids):
    selected_ids = selected_ids or []
    cluster_mode = cluster_mode if cluster_mode in PARALLEL_CLUSTER_MODES else "small"
    profile, area_map = build_parallel_profile(int(target_year), int(target_month), cluster_mode)
    return (
        parallel_figure(profile, area_map, selected_ids, int(target_year), int(target_month), cluster_mode),
        area_map,
    )


def find_available_port(start_port: int) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def parse_args():
    parser = argparse.ArgumentParser(description="Run the climate resort dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(debug=args.debug, host=args.host, port=find_available_port(args.port))
