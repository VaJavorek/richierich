from __future__ import annotations

import argparse
import socket
from typing import Any

import numpy as np
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
    normalize_for_parallel,
    optimal_candidates,
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

PARALLEL_VARIABLES = [
    "mean_temp",
    "daily_max_mean",
    "daily_min_mean",
    "hot_days",
    "tropical_nights",
    "mosquito_days",
    "dry_days",
    "heat_risk",
    "dryness_risk",
]


def card(title: str, subtitle: str, children, class_name: str = ""):
    return html.Section(
        [
            html.Div(
                [
                    html.Div([html.H2(title), html.P(subtitle)], className="card-title-block"),
                ],
                className="card-head",
            ),
            children,
        ],
        className=f"panel-card {class_name}".strip(),
    )


def slider_value(id_: str, label: str, value: float, minimum: float, maximum: float, step: float = 1):
    return html.Div(
        [
            html.Div([html.Span(label), html.Strong(id=f"{id_}-label")], className="slider-label"),
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
                    card(
                        "Main map",
                        "Mean temperature heatmap across the placeholder Europe grid.",
                        dcc.Graph(id="main-map", config=GRAPH_CONFIG, className="map-graph main-map-graph"),
                        "main-map-card",
                    ),
                    html.Div(
                        [
                            card(
                                "Mosquito season",
                                "Projected tiger mosquito season length.",
                                dcc.Graph(id="mosquito-map", config=GRAPH_CONFIG, className="map-graph mini-map-graph"),
                                "mini-map-card",
                            ),
                            card(
                                "Heat risk",
                                "Composite risk from hot days and tropical nights.",
                                dcc.Graph(id="heat-risk-map", config=GRAPH_CONFIG, className="map-graph mini-map-graph"),
                                "mini-map-card",
                            ),
                            card(
                                "Dryness risk",
                                "Consecutive dry-day pressure for resort operations.",
                                dcc.Graph(id="dry-risk-map", config=GRAPH_CONFIG, className="map-graph mini-map-graph"),
                                "mini-map-card",
                            ),
                        ],
                        className="mini-map-stack",
                    ),
                    card(
                        "Custom ranking",
                        "Weighted shortlist of climate-driven resort candidates.",
                        html.Div(
                            [
                                html.Div(
                                    [
                                        slider_value("weight-temperature", "Temperature comfort", DEFAULT_WEIGHTS["temperature"], 0, 50),
                                        slider_value("weight-mosquito", "Low mosquito burden", DEFAULT_WEIGHTS["mosquito"], 0, 50),
                                        slider_value("weight-heat", "Low heat risk", DEFAULT_WEIGHTS["heat"], 0, 50),
                                        slider_value("weight-dryness", "Low dryness risk", DEFAULT_WEIGHTS["dryness"], 0, 50),
                                        slider_value("weight-nights", "Low tropical nights", DEFAULT_WEIGHTS["nights"], 0, 50),
                                    ],
                                    className="weight-grid",
                                ),
                                dcc.Graph(id="ranking-chart", config=GRAPH_CONFIG, className="ranking-graph"),
                            ],
                            className="ranking-content",
                        ),
                        "ranking-card",
                    ),
                    card(
                        "Parallel coordinates",
                        "Each line is one grid cell, normalized across climate variables.",
                        dcc.Graph(id="parallel-chart", config=GRAPH_CONFIG, className="parallel-graph"),
                        "parallel-card",
                    ),
                    card(
                        "Optimal Area Finder",
                        "Highlight places matching the investor's climate limits.",
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
                                                    options=[
                                                        {"label": label, "value": month}
                                                        for month, label in MONTH_LABELS.items()
                                                    ],
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
                                        slider_value("max-mosquito-filter", "Max mosquito days", DEFAULT_FILTERS["max_mosquito"], 0, 210, 5),
                                        slider_value("max-hot-filter", "Max hot days", DEFAULT_FILTERS["max_hot_days"], 0, 31, 1),
                                        slider_value("max-dry-filter", "Max dry days", DEFAULT_FILTERS["max_dry_days"], 0, 65, 1),
                                        slider_value("max-tropical-filter", "Max tropical nights", DEFAULT_FILTERS["max_tropical_nights"], 0, 31, 1),
                                        slider_value("max-change-filter", "Max temp change", DEFAULT_FILTERS["max_temp_change"], 0, 5, 0.1),
                                    ],
                                    className="finder-controls",
                                ),
                                dcc.Graph(id="optimal-map", config=GRAPH_CONFIG, className="map-graph optimal-map-graph"),
                            ],
                            className="finder-content",
                        ),
                        "finder-card",
                    ),
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
) -> go.Figure:
    selected = set(selected_ids or [])
    meta = VARIABLES[variable]
    fig = go.Figure()
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
                    "colorbar": {
                        "title": meta["unit"],
                        "len": 0.62,
                        "thickness": 10,
                    }
                    if not compact
                    else None,
                },
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


def ranking_figure(df, weights, selected_ids: list[str] | None) -> go.Figure:
    selected = set(selected_ids or [])
    top = score_candidates(df, weights).head(12).copy().sort_values("resort_score")
    labels = top["area_name"] + " | " + top["cell_id"]
    colors = [HIGHLIGHT if cell_id in selected else "#be7f42" for cell_id in top["cell_id"]]
    fig = go.Figure(
        go.Bar(
            x=top["resort_score"],
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(23,33,43,0.18)", "width": 1}},
            customdata=top[["cell_id", "area_name", "mean_temp", "mosquito_days", "heat_risk", "dryness_risk"]],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Cell %{customdata[0]}<br>"
                "Suitability: %{x:.1f}/100<br>"
                "Mean temp: %{customdata[2]:.1f} deg C<br>"
                "Mosquito season: %{customdata[3]:.0f} days<br>"
                "Heat risk: %{customdata[4]:.0f}<br>"
                "Dryness risk: %{customdata[5]:.0f}<extra></extra>"
            ),
            selected={"marker": {"color": HIGHLIGHT}},
            unselected={"marker": {"opacity": 0.35}},
        )
    )
    fig.update_layout(
        height=330,
        margin={"l": 132, "r": 24, "t": 10, "b": 34},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        xaxis={"range": [0, 100], "title": "Resort opportunity score", "gridcolor": "rgba(23,33,43,0.09)"},
        yaxis={"title": "", "tickfont": {"size": 11}},
        bargap=0.28,
        dragmode="select",
    )
    return fig


def parallel_figure(df, selected_ids: list[str] | None) -> go.Figure:
    selected = set(selected_ids or [])
    normalized = normalize_for_parallel(df, PARALLEL_VARIABLES)
    labels = [VARIABLES[key]["label"] for key in PARALLEL_VARIABLES]
    fig = go.Figure()

    for _, row in normalized.iterrows():
        cell_id = row["cell_id"]
        is_selected = cell_id in selected
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[row[key] for key in PARALLEL_VARIABLES],
                mode="lines+markers",
                line={
                    "color": HIGHLIGHT if is_selected else "rgba(92, 73, 54, 0.24)",
                    "width": 3.2 if is_selected else 1.25,
                },
                marker={
                    "size": 7 if is_selected else 5,
                    "color": HIGHLIGHT if is_selected else "rgba(92, 73, 54, 0.52)",
                    "line": {
                        "width": 1 if is_selected else 0,
                        "color": INK,
                    },
                },
                opacity=1 if is_selected or not selected else 0.18,
                customdata=[[cell_id, row["area_name"]]] * len(PARALLEL_VARIABLES),
                hovertemplate="<b>%{customdata[1]}</b><br>%{x}: %{y:.2f} normalized<br>Cell %{customdata[0]}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=380,
        margin={"l": 46, "r": 18, "t": 10, "b": 96},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": INK},
        xaxis={"tickangle": -28, "title": "", "showgrid": True, "gridcolor": "rgba(23,33,43,0.12)"},
        yaxis={"range": [0, 1], "title": "Normalized risk or value", "gridcolor": "rgba(23,33,43,0.09)"},
        dragmode="lasso",
        hovermode="closest",
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
                html.Span("Click a map cell, ranking bar, scatter point, or parallel line to link the dashboard."),
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


def extract_cell_ids(payload: dict[str, Any] | None) -> list[str]:
    if not payload or not payload.get("points"):
        return []
    ids: list[str] = []
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
    Output("optimal-filter-store", "data"),
    Output("max-mosquito-filter-label", "children"),
    Output("max-hot-filter-label", "children"),
    Output("max-dry-filter-label", "children"),
    Output("max-tropical-filter-label", "children"),
    Output("max-change-filter-label", "children"),
    Input("target-year-filter", "value"),
    Input("target-month-filter", "value"),
    Input("temp-range-filter", "value"),
    Input("max-mosquito-filter", "value"),
    Input("max-hot-filter", "value"),
    Input("max-dry-filter", "value"),
    Input("max-tropical-filter", "value"),
    Input("max-change-filter", "value"),
)
def update_optimal_filters(target_year, target_month, temp_range, max_mosquito, max_hot, max_dry, max_tropical, max_change):
    filters = {
        "target_year": int(target_year),
        "target_month": int(target_month),
        "temp_min": float(temp_range[0]),
        "temp_max": float(temp_range[1]),
        "max_mosquito": float(max_mosquito),
        "max_hot_days": float(max_hot),
        "max_dry_days": float(max_dry),
        "max_tropical_nights": float(max_tropical),
        "max_temp_change": float(max_change),
    }
    return (
        filters,
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
    }
    ids = extract_cell_ids(payload_by_prop.get(prop_id))
    if prop_id.endswith(".selectedData"):
        return ids if ids else current_selection or []
    if ids:
        return ids[:1]
    return current_selection or []


@app.callback(
    Output("main-map", "figure"),
    Output("mosquito-map", "figure"),
    Output("heat-risk-map", "figure"),
    Output("dry-risk-map", "figure"),
    Output("parallel-chart", "figure"),
    Output("ranking-chart", "figure"),
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
    filters = filters or DEFAULT_FILTERS
    selected_ids = selected_ids or []

    df = climate_slice(active_time["year"], active_time["month"])
    time_label = f"{MONTH_LABELS[int(active_time['month'])]} {int(active_time['year'])}"
    optimal_df = climate_slice(filters["target_year"], filters["target_month"])
    candidates = optimal_candidates(optimal_df, filters)
    candidate_ids = set(candidates["cell_id"])
    candidate_label = f"{len(candidates):,} matching cells | {MONTH_LABELS[int(filters['target_month'])]} {int(filters['target_year'])}"

    return (
        map_figure(df, "mean_temp", selected_ids, title_suffix=time_label),
        map_figure(df, "mosquito_days", selected_ids, compact=True, title_suffix=time_label),
        map_figure(df, "heat_risk", selected_ids, compact=True, title_suffix=time_label),
        map_figure(df, "dryness_risk", selected_ids, compact=True, title_suffix=time_label),
        parallel_figure(df, selected_ids),
        ranking_figure(df, weights, selected_ids),
        map_figure(optimal_df, "mean_temp", selected_ids, candidate_ids=candidate_ids, title_suffix=candidate_label, height=360),
        scatter_figure(df, scatter_x, scatter_y, selected_ids, weights),
        selection_summary(df, selected_ids),
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
