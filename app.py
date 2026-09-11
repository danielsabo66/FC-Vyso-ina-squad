from pathlib import Path
from io import BytesIO
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ============================================================
# PAGE
# ============================================================

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
LOGO_PATH = APP_DIR / "assets" / "jihlava_logo.png"
HERO_PATH = APP_DIR / "assets" / "jihlava_hero.png"

st.set_page_config(
    page_title="FC Vysočina Scouting",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

OWN_TEAM = "Vysočina Jihlava"


# ============================================================
# CLUB UI
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --fcv-navy: #142E63;
        --fcv-navy-deep: #081426;
        --fcv-yellow: #FFD900;
        --fcv-surface: #101B2B;
        --fcv-surface-2: #15243A;
        --fcv-border: rgba(255,255,255,0.09);
        --fcv-muted: #9DABBE;
    }

    /* Main page */
    .stApp {
        background:
            radial-gradient(circle at 75% 0%, rgba(20,46,99,.22), transparent 34rem),
            linear-gradient(180deg, #091321 0%, #0A1422 100%);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.0rem;
        padding-bottom: 4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1728 0%, #0A1320 100%);
        border-right: 1px solid rgba(255,217,0,.10);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #DCE5F0;
    }

    /* Headings */
    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750 !important;
    }

    h2, h3 {
        color: #F6F8FB !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(21,36,58,.94), rgba(15,27,44,.94));
        border: 1px solid var(--fcv-border);
        border-radius: 14px;
        padding: .9rem 1rem;
        box-shadow: 0 5px 18px rgba(0,0,0,.10);
    }

    [data-testid="stMetricLabel"] {
        color: var(--fcv-muted);
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: .4rem;
        border-bottom: 1px solid rgba(255,255,255,.08);
    }

    button[data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(255,217,0,.09);
        color: #FFFFFF;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Inputs / controls */
    div[data-baseweb="select"] > div,
    [data-baseweb="input"] {
        background-color: rgba(19,34,54,.88);
        border-color: rgba(255,255,255,.10);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,217,0,.35);
        background: rgba(255,217,0,.08);
        transition: all .16s ease;
    }

    .stButton > button:hover {
        border-color: var(--fcv-yellow);
        color: #FFFFFF;
        background: rgba(255,217,0,.14);
    }

    /* Dataframes / cards */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        overflow: hidden;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--fcv-muted) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,.07) !important;
    }

    /* Hide default Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Make top decoration club yellow */
    [data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #FFD900, #1B4285);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

POSITION_ORDER = [
    "GK",
    "LB",
    "RB",
    "CB",
    "CM/CDM",
    "CAM",
    "RW/LW",
    "ST",
]

RADAR_COLORS = [
    "#00AEEF",  # bright blue
    "#FF3B5C",  # vivid red/pink
    "#00C875",  # emerald green
    "#FFB000",  # amber
    "#8B5CF6",  # violet
]

# Only metrics where a lower raw value is clearly more favorable are reversed.
LOWER_IS_MORE_FAVORABLE = {
    "Conceded goals",
    "Conceded goals per 90",
    "Fouls per 90",
}

# These can describe role, workload or playing style. A high percentile should
# not automatically be interpreted as "better".
CONTEXT_METRICS = {
    "Shots against",
    "Shots against per 90",
    "xG against",
    "xG against per 90",
    "Back passes received as GK per 90",
    "Exits per 90",
    "Aerial duels per 90",
    "Defensive duels per 90",
    "Offensive duels per 90",
    "Passes per 90",
    "Forward passes per 90",
    "Received passes per 90",
    "Crosses per 90",
    "Dribbles per 90",
    "Shots per 90",
    "Accelerations per 90",
    "Touches in box per 90",
    "Successful defensive actions per 90",
    "Successful attacking actions per 90",
    "Progressive runs per 90",
}

NON_METRIC_COLUMNS = {
    "Name",
    "Player",
    "Team",
    "PositionGroup",
    "SourceFile",
    "Our player",
    "Minutes",
    "Minutes played",
    "Age",
    "Nationality",
    "Foot",
    "Wyscout_ID",
    "id",
}

# Suggested defaults only. Every metric from each Wyscout export remains selectable.
PREFERRED_METRICS = {
    "GK": [
        "Save rate, %",
        "Prevented goals per 90",
        "Conceded goals per 90",
        "Accurate passes, %",
        "Accurate long passes, %",
        "Exits per 90",
        "Aerial duels per 90",
        "Back passes received as GK per 90",
    ],
    "LB": [
        "Defensive duels won, %",
        "Aerial duels won, %",
        "PAdj Interceptions",
        "Accurate crosses, %",
        "Successful dribbles, %",
        "Progressive runs per 90",
        "xA per 90",
        "Passes to penalty area per 90",
    ],
    "RB": [
        "Defensive duels won, %",
        "Aerial duels won, %",
        "PAdj Interceptions",
        "Accurate crosses, %",
        "Successful dribbles, %",
        "Progressive runs per 90",
        "xA per 90",
        "Passes to penalty area per 90",
    ],
    "CB": [
        "Defensive duels won, %",
        "Aerial duels won, %",
        "PAdj Interceptions",
        "Fouls per 90",
        "Successful defensive actions per 90",
        "Progressive runs per 90",
        "Passes to final third per 90",
    ],
    "CM/CDM": [
        "Defensive duels won, %",
        "Aerial duels won, %",
        "PAdj Interceptions",
        "Assists per 90",
        "Progressive runs per 90",
        "Accurate passes, %",
        "Accurate forward passes, %",
        "Accurate passes to final third, %",
    ],
    "CAM": [
        "xA",
        "Successful attacking actions per 90",
        "Goals per 90",
        "xG per 90",
        "Shots on target, %",
        "Assists per 90",
        "Successful dribbles, %",
        "Progressive runs per 90",
        "Smart passes per 90",
        "Key passes per 90",
    ],
    "RW/LW": [
        "Successful attacking actions per 90",
        "Goals per 90",
        "xG per 90",
        "Shots on target, %",
        "Goal conversion, %",
        "Assists per 90",
        "Accurate crosses, %",
        "Successful dribbles, %",
        "Offensive duels won, %",
        "Progressive runs per 90",
        "xA per 90",
        "Shot assists per 90",
    ],
    "ST": [
        "Successful attacking actions per 90",
        "Goals per 90",
        "Non-penalty goals per 90",
        "xG per 90",
        "Shots on target, %",
        "Goal conversion, %",
        "Assists per 90",
        "Successful dribbles, %",
        "Offensive duels won, %",
        "Touches in box per 90",
        "xA per 90",
        "Shot assists per 90",
    ],
}



# ============================================================
# DATA
# ============================================================

def infer_position(filename: str) -> str:
    stem = Path(filename).stem.upper()
    normalized = re.sub(r"[^A-Z0-9]+", " ", stem)
    tokens = normalized.split()

    if "CM" in tokens and "CDM" in tokens:
        return "CM/CDM"

    if "RW" in tokens and "LW" in tokens:
        return "RW/LW"

    if "CAM" in tokens:
        return "CAM"

    for code in ["GK", "LB", "RB", "CB", "ST"]:
        if code in tokens:
            return code

    return Path(filename).stem


@st.cache_data(show_spinner=False)
def load_all_data():
    files = sorted(DATA_DIR.glob("*.xlsx"))
    frames = []
    warnings = []

    for file in files:
        try:
            frame = pd.read_excel(file)
        except Exception as exc:
            warnings.append(f"{file.name}: {exc}")
            continue

        if frame.empty:
            continue

        # Normalize Excel headers to avoid KeyErrors from minor naming differences.
        frame.columns = [str(col).strip() for col in frame.columns]

        minute_aliases = {
            "Minutes": "Minutes played",
            "Minutes Played": "Minutes played",
            "minutes played": "Minutes played",
        }
        frame = frame.rename(
            columns={
                col: minute_aliases[col]
                for col in frame.columns
                if col in minute_aliases
            }
        )

        if "Player" in frame.columns and "Name" not in frame.columns:
            frame = frame.rename(columns={"Player": "Name"})

        if "Name" not in frame.columns:
            warnings.append(f"{file.name}: missing Player/Name column")
            continue

        if "Team" not in frame.columns:
            frame["Team"] = ""

        frame["PositionGroup"] = infer_position(file.name)
        frame["SourceFile"] = file.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), warnings

    return pd.concat(frames, ignore_index=True, sort=False), warnings


def get_metrics(frame: pd.DataFrame) -> list[str]:
    metrics = []

    for col in frame.columns:
        if col in NON_METRIC_COLUMNS:
            continue

        numeric = pd.to_numeric(frame[col], errors="coerce")

        if numeric.notna().sum() >= 2:
            metrics.append(col)

    return metrics


def preferred_metrics(position: str, available: list[str], maximum: int = 8) -> list[str]:
    preferred = [
        metric
        for metric in PREFERRED_METRICS.get(position, [])
        if metric in available
    ]

    if len(preferred) < min(maximum, len(available)):
        for metric in available:
            if metric not in preferred:
                preferred.append(metric)
            if len(preferred) >= maximum:
                break

    return preferred[:maximum]


# ============================================================
# PERCENTILES / RANKS
# ============================================================

def raw_percentile(value, series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(value) or len(values) == 0:
        return np.nan

    lower = np.sum(values < float(value))
    equal = np.sum(
        np.isclose(
            values,
            float(value),
            rtol=0,
            atol=1e-12,
        )
    )

    return 100.0 * (lower + 0.5 * equal) / len(values)


def display_percentile(metric: str, value, frame: pd.DataFrame):
    percentile = raw_percentile(value, frame[metric])

    if pd.isna(percentile):
        return np.nan

    if metric in LOWER_IS_MORE_FAVORABLE:
        return 100.0 - percentile

    return percentile


def league_rank(metric: str, value, frame: pd.DataFrame):
    series = pd.to_numeric(frame[metric], errors="coerce")
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(value):
        return np.nan

    if metric in LOWER_IS_MORE_FAVORABLE:
        return int((series < value).sum() + 1)

    return int((series > value).sum() + 1)


def prepare_position_data(
    frame: pd.DataFrame,
    metrics: list[str],
    reference_frame: pd.DataFrame | None = None,
):
    output = frame.copy()
    reference = output if reference_frame is None else reference_frame

    for metric in metrics:
        output[f"PCTL__{metric}"] = output[metric].apply(
            lambda x, m=metric: display_percentile(m, x, reference)
        )

    output["Our player"] = output["Team"].astype(str).eq(OWN_TEAM)

    return output


def overall_score(row, metrics: list[str], weights: dict[str, float]):
    numer = 0.0
    denom = 0.0

    for metric in metrics:
        value = row.get(f"PCTL__{metric}", np.nan)
        weight = float(weights.get(metric, 1.0))

        if pd.notna(value) and weight > 0:
            numer += float(value) * weight
            denom += weight

    if denom == 0:
        return np.nan

    return numer / denom


def average_metric_rank(row, metrics: list[str], frame: pd.DataFrame):
    ranks = []

    for metric in metrics:
        value = pd.to_numeric(
            pd.Series([row[metric]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            ranks.append(
                league_rank(metric, value, frame)
            )

    if not ranks:
        return np.nan

    return float(np.mean(ranks))


# ============================================================
# DISPLAY HELPERS
# ============================================================

def short_team(team: str) -> str:
    team = str(team or "").strip()

    if team == OWN_TEAM:
        return "Jihlava"

    if len(team) <= 18:
        return team

    return team[:17] + "…"


def player_label(frame: pd.DataFrame, player_name: str):
    row = frame.loc[frame["Name"] == player_name].iloc[0]
    team = str(row.get("Team", "") or "").strip()
    our_tag = " • OUR" if team == OWN_TEAM else ""

    return f"{player_name} — {team or 'Unknown team'}{our_tag}"


def radar_player_label(frame: pd.DataFrame, player_name: str):
    row = frame.loc[frame["Name"] == player_name].iloc[0]
    return f"{player_name} · {short_team(row.get('Team', ''))}"


def metric_note(metric: str):
    if metric in LOWER_IS_MORE_FAVORABLE:
        return "Lower raw value = higher percentile"

    if metric in CONTEXT_METRICS:
        return "Profile / volume metric"

    return "Higher raw value = higher percentile"


def short_metric_label(metric: str):
    replacements = {
        "Successful defensive actions": "Defensive actions",
        "Successful attacking actions": "Attacking actions",
        "Defensive duels won": "Def. duels won",
        "Defensive duels": "Def. duels",
        "Offensive duels won": "Off. duels won",
        "Offensive duels": "Off. duels",
        "Aerial duels won": "Aerial won",
        "Successful dribbles": "Dribbles won",
        "Accurate crosses": "Cross accuracy",
        "Accurate forward passes": "Forward pass acc.",
        "Accurate passes to final third": "Final 3rd pass acc.",
        "Accurate passes to penalty area": "Box pass accuracy",
        "Passes to final third": "Final 3rd passes",
        "Passes to penalty area": "Passes to box",
        "Back passes received as GK": "Back passes received",
        "Non-penalty goals": "Non-penalty goals",
        "Progressive runs": "Progressive runs",
        "PAdj Interceptions": "PAdj interceptions",
        "PAdj Sliding tackles": "PAdj tackles",
    }

    label = metric

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = label.replace(" per 90", " /90")
    label = label.replace(", %", " %")

    # Wrap long labels onto two/three lines around the radar.
    return "<br>".join(textwrap.wrap(label, width=18))


def hex_to_rgba(hex_color: str, alpha: float):
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    return f"rgba({red},{green},{blue},{alpha})"


# ============================================================
# RADAR
# ============================================================

def make_radar(
    frame: pd.DataFrame,
    players: list[str],
    metrics: list[str],
    show_fill: bool = True,
    compact_legend: bool = True,
):
    """
    Modern scouting radar:
    - dark navy card
    - polygon grid
    - 50th percentile reference
    - high-contrast traces
    - raw value + percentile in hover
    """

    fig = go.Figure()

    metric_labels = [
        short_metric_label(metric)
        for metric in metrics
    ]

    # League median reference polygon.
    if metrics:
        fig.add_trace(
            go.Scatterpolar(
                r=[50] * (len(metrics) + 1),
                theta=metric_labels + [metric_labels[0]],
                mode="lines",
                line=dict(
                    color="rgba(255,255,255,0.30)",
                    width=1.5,
                    dash="dot",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fill_alpha = (
        0.20 if len(players) == 1
        else 0.12 if len(players) == 2
        else 0.055
    )

    for index, player in enumerate(players):
        row = frame.loc[
            frame["Name"] == player
        ].iloc[0]

        color = RADAR_COLORS[
            index % len(RADAR_COLORS)
        ]

        percentiles = [
            float(row[f"PCTL__{metric}"])
            for metric in metrics
        ]

        raw_values = [
            pd.to_numeric(
                pd.Series([row[metric]]),
                errors="coerce",
            ).iloc[0]
            for metric in metrics
        ]

        if not percentiles:
            continue

        trace_name = (
            radar_player_label(frame, player)
            if compact_legend
            else player_label(frame, player)
        )

        fig.add_trace(
            go.Scatterpolar(
                r=percentiles + [percentiles[0]],
                theta=metric_labels + [metric_labels[0]],
                customdata=raw_values + [raw_values[0]],
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=3.4,
                ),
                marker=dict(
                    color=color,
                    size=7,
                    line=dict(
                        color="#07111F",
                        width=1.4,
                    ),
                ),
                fill="toself" if show_fill else "none",
                fillcolor=hex_to_rgba(
                    color,
                    fill_alpha,
                ),
                name=trace_name,
                hovertemplate=(
                    "<b>%{fullData.name}</b>"
                    "<br>%{theta}"
                    "<br>Percentile: <b>%{r:.0f}</b>"
                    "<br>Raw: <b>%{customdata:.2f}</b>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#0B1626",
            gridshape="linear",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode="array",
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                tickangle=0,
                angle=90,
                gridcolor="rgba(255,255,255,0.18)",
                linecolor="rgba(255,255,255,0.22)",
                tickfont=dict(
                    size=10,
                    color="rgba(255,255,255,0.72)",
                ),
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.20)",
                tickfont=dict(
                    size=12,
                    color="#F5F7FA",
                ),
            ),
        ),
        paper_bgcolor="#08111D",
        plot_bgcolor="#08111D",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(
                size=11,
                color="#F5F7FA",
            ),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="#111C2D",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(
                color="#FFFFFF",
                size=12,
            ),
        ),
        height=640,
        margin=dict(
            l=90,
            r=90,
            t=40,
            b=100,
        ),
    )

    return fig

def make_percentile_bars(
    frame: pd.DataFrame,
    players: list[str],
    metrics: list[str],
):
    """
    Grouped horizontal bars as a clearer alternative for 3-4 player comparison.
    """
    fig = go.Figure()

    display_metrics = [
        short_metric_label(metric).replace("<br>", " ")
        for metric in metrics
    ]

    # Reverse so the first selected metric appears at the top.
    display_metrics = display_metrics[::-1]
    reversed_metrics = metrics[::-1]

    for index, player in enumerate(players):
        row = frame.loc[frame["Name"] == player].iloc[0]
        color = RADAR_COLORS[index % len(RADAR_COLORS)]

        values = [
            float(row[f"PCTL__{metric}"])
            for metric in reversed_metrics
        ]

        fig.add_trace(
            go.Bar(
                x=values,
                y=display_metrics,
                orientation="h",
                name=radar_player_label(frame, player),
                marker=dict(color=color),
                opacity=0.88,
                hovertemplate=(
                    "<b>%{fullData.name}</b>"
                    "<br>%{y}"
                    "<br>Percentile: %{x:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=50,
        line_width=1.5,
        line_dash="dot",
        line_color="rgba(170,170,170,0.6)",
    )

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            title="League percentile",
            gridcolor="rgba(145,145,145,0.12)",
        ),
        yaxis=dict(
            title="",
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.17,
            xanchor="center",
            x=0.5,
        ),
        height=max(430, 55 * len(metrics) + 180),
        margin=dict(
            l=25,
            r=30,
            t=25,
            b=100,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def show_individual_radars(
    frame: pd.DataFrame,
    players: list[str],
    metrics: list[str],
):
    columns = st.columns(2)

    for index, player in enumerate(players):
        with columns[index % 2]:
            st.markdown(f"**{player_label(frame, player)}**")

            chart = make_radar(
                frame,
                [player],
                metrics,
                show_fill=True,
                compact_legend=True,
            )

            chart.update_layout(
                height=470,
                showlegend=False,
                margin=dict(l=60, r=60, t=20, b=40),
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
                theme="streamlit",
                key=f"individual_radar_{player}_{index}",
            )



# ============================================================
# CLUB VS OPPONENT / BEST XI
# ============================================================

FORMATION_SLOTS = {
    "4-2-3-1": [
        ("GK", ["GK"]),
        ("LB", ["LB"]),
        ("LCB", ["CB"]),
        ("RCB", ["CB"]),
        ("RB", ["RB"]),
        ("LDM", ["CM/CDM"]),
        ("RDM", ["CM/CDM"]),
        ("CAM", ["CAM"]),
        ("LW", ["RW/LW"]),
        ("RW", ["RW/LW"]),
        ("ST", ["ST"]),
    ],
    "4-3-3": [
        ("GK", ["GK"]),
        ("LB", ["LB"]),
        ("LCB", ["CB"]),
        ("RCB", ["CB"]),
        ("RB", ["RB"]),
        ("DM", ["CM/CDM"]),
        ("LCM", ["CM/CDM"]),
        ("RCM", ["CM/CDM"]),
        ("LW", ["RW/LW"]),
        ("RW", ["RW/LW"]),
        ("ST", ["ST"]),
    ],
    "4-4-2": [
        ("GK", ["GK"]),
        ("LB", ["LB"]),
        ("LCB", ["CB"]),
        ("RCB", ["CB"]),
        ("RB", ["RB"]),
        ("LM", ["RW/LW"]),
        ("LCM", ["CM/CDM"]),
        ("RCM", ["CM/CDM"]),
        ("RM", ["RW/LW"]),
        ("LST", ["ST"]),
        ("RST", ["ST"]),
    ],
    "3-5-2": [
        ("GK", ["GK"]),
        ("LCB", ["CB"]),
        ("CB", ["CB"]),
        ("RCB", ["CB"]),
        ("LWB", ["LB"]),
        ("LCM", ["CM/CDM"]),
        ("CM", ["CM/CDM"]),
        ("RCM", ["CM/CDM"]),
        ("RWB", ["RB"]),
        ("LST", ["ST"]),
        ("RST", ["ST"]),
    ],
    "3-4-2-1": [
        ("GK", ["GK"]),
        ("LCB", ["CB"]),
        ("CB", ["CB"]),
        ("RCB", ["CB"]),
        ("LWB", ["LB"]),
        ("LCM", ["CM/CDM"]),
        ("RCM", ["CM/CDM"]),
        ("RWB", ["RB"]),
        ("LAM", ["CAM", "RW/LW"]),
        ("RAM", ["CAM", "RW/LW"]),
        ("ST", ["ST"]),
    ],
    "3-4-3": [
        ("GK", ["GK"]),
        ("LCB", ["CB"]),
        ("CB", ["CB"]),
        ("RCB", ["CB"]),
        ("LWB", ["LB"]),
        ("LCM", ["CM/CDM"]),
        ("RCM", ["CM/CDM"]),
        ("RWB", ["RB"]),
        ("LW", ["RW/LW"]),
        ("RW", ["RW/LW"]),
        ("ST", ["ST"]),
    ],
}

FORMATION_COORDS = {
    "4-2-3-1": {
        "GK": (50, 8),
        "LB": (15, 27), "LCB": (38, 25), "RCB": (62, 25), "RB": (85, 27),
        "LDM": (38, 45), "RDM": (62, 45),
        "CAM": (50, 63),
        "LW": (20, 76), "RW": (80, 76),
        "ST": (50, 90),
    },
    "4-3-3": {
        "GK": (50, 8),
        "LB": (15, 27), "LCB": (38, 25), "RCB": (62, 25), "RB": (85, 27),
        "DM": (50, 44), "LCM": (33, 58), "RCM": (67, 58),
        "LW": (20, 80), "RW": (80, 80), "ST": (50, 90),
    },
    "4-4-2": {
        "GK": (50, 8),
        "LB": (15, 27), "LCB": (38, 25), "RCB": (62, 25), "RB": (85, 27),
        "LM": (18, 55), "LCM": (40, 55), "RCM": (60, 55), "RM": (82, 55),
        "LST": (40, 84), "RST": (60, 84),
    },
    "3-5-2": {
        "GK": (50, 8),
        "LCB": (28, 28), "CB": (50, 24), "RCB": (72, 28),
        "LWB": (10, 55), "LCM": (35, 55), "CM": (50, 62), "RCM": (65, 55), "RWB": (90, 55),
        "LST": (40, 86), "RST": (60, 86),
    },
    "3-4-2-1": {
        "GK": (50, 8),
        "LCB": (28, 28), "CB": (50, 24), "RCB": (72, 28),
        "LWB": (10, 55), "LCM": (40, 52), "RCM": (60, 52), "RWB": (90, 55),
        "LAM": (36, 72), "RAM": (64, 72),
        "ST": (50, 89),
    },
    "3-4-3": {
        "GK": (50, 8),
        "LCB": (28, 28), "CB": (50, 24), "RCB": (72, 28),
        "LWB": (10, 55), "LCM": (40, 52), "RCM": (60, 52), "RWB": (90, 55),
        "LW": (22, 80), "RW": (78, 80), "ST": (50, 89),
    },
}


def position_reference_frame(
    full_data: pd.DataFrame,
    position: str,
    min_minutes: int,
):
    position_data = full_data[
        full_data["PositionGroup"] == position
    ].copy()

    if position_data.empty:
        return position_data

    if "Minutes played" in position_data.columns:
        minutes = pd.to_numeric(
            position_data["Minutes played"],
            errors="coerce",
        ).fillna(0)

        position_data = position_data[
            minutes >= min_minutes
        ].copy()

    return position_data


def position_scored_frame(
    full_data: pd.DataFrame,
    position: str,
    min_minutes: int,
):
    reference = position_reference_frame(
        full_data,
        position,
        min_minutes,
    )

    if reference.empty:
        return reference, []

    position_metrics = get_metrics(reference)

    if not position_metrics:
        return pd.DataFrame(), []

    scored = prepare_position_data(
        reference,
        position_metrics,
    )

    scoring_metrics = preferred_metrics(
        position,
        position_metrics,
        maximum=min(8, len(position_metrics)),
    )

    scored["Position overall"] = scored.apply(
        lambda row: overall_score(
            row,
            scoring_metrics,
            {metric: 1.0 for metric in scoring_metrics},
        ),
        axis=1,
    )

    return scored, scoring_metrics


def build_club_candidate_pool(
    full_data: pd.DataFrame,
    team: str,
    min_minutes: int,
    allow_winger_wingbacks: bool = False,
):
    pools = {}

    for position in POSITION_ORDER:
        scored, scoring_metrics = position_scored_frame(
            full_data,
            position,
            min_minutes,
        )

        if scored.empty:
            pools[position] = pd.DataFrame()
            continue

        team_rows = scored[
            scored["Team"].astype(str).eq(team)
        ].copy()

        if not team_rows.empty:
            team_rows["Score position"] = position
            team_rows["Scoring metrics"] = ", ".join(scoring_metrics)

        pools[position] = team_rows

    if allow_winger_wingbacks:
        # This does not merge the scoring models. Wingers keep their winger overall score.
        # They are simply allowed as tactical fallback candidates in WB slots.
        pass

    return pools


def slot_candidates(
    slot_name: str,
    allowed_positions: list[str],
    pools: dict,
    allow_winger_wingbacks: bool,
):
    candidate_frames = []

    positions = list(allowed_positions)

    if allow_winger_wingbacks:
        if slot_name == "LWB" and "RW/LW" not in positions:
            positions.append("RW/LW")
        if slot_name == "RWB" and "RW/LW" not in positions:
            positions.append("RW/LW")

    for position in positions:
        frame = pools.get(position, pd.DataFrame())

        if frame is None or frame.empty:
            continue

        subset = frame.copy()
        subset["Candidate source"] = position
        candidate_frames.append(subset)

    if not candidate_frames:
        return pd.DataFrame()

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
        sort=False,
    )

    candidates = candidates.sort_values(
        "Position overall",
        ascending=False,
    )

    # One player can appear in more than one positional dataset.
    candidates = candidates.drop_duplicates(
        subset=["Name", "Team"],
        keep="first",
    )

    return candidates


def build_best_xi(
    full_data: pd.DataFrame,
    team: str,
    formation: str,
    min_minutes: int,
    allow_winger_wingbacks: bool = False,
):
    pools = build_club_candidate_pool(
        full_data,
        team,
        min_minutes,
        allow_winger_wingbacks=allow_winger_wingbacks,
    )

    slot_defs = FORMATION_SLOTS[formation]

    # Build all candidate lists first.
    candidate_map = {}

    for slot_name, allowed_positions in slot_defs:
        candidate_map[slot_name] = slot_candidates(
            slot_name,
            allowed_positions,
            pools,
            allow_winger_wingbacks,
        )

    # Fill the scarcest slots first, then choose the best available player.
    # This avoids many duplicate-player conflicts across positional exports.
    slot_order = sorted(
        slot_defs,
        key=lambda item: len(candidate_map[item[0]]),
    )

    selected = {}
    used_players = set()

    for slot_name, allowed_positions in slot_order:
        candidates = candidate_map[slot_name]

        if candidates.empty:
            selected[slot_name] = None
            continue

        choice = None

        for _, row in candidates.iterrows():
            player_key = (
                str(row["Name"]),
                str(row["Team"]),
            )

            if player_key not in used_players:
                choice = row
                used_players.add(player_key)
                break

        selected[slot_name] = choice

    # Return in the formation's natural slot order.
    return [
        (
            slot_name,
            allowed_positions,
            selected.get(slot_name),
        )
        for slot_name, allowed_positions in slot_defs
    ]


def lineup_dataframe(lineup):
    rows = []

    for slot_name, allowed_positions, row in lineup:
        if row is None:
            rows.append(
                {
                    "Slot": slot_name,
                    "Player": "—",
                    "Team": "",
                    "Data position": "/".join(allowed_positions),
                    "Minutes": np.nan,
                    "Overall percentile": np.nan,
                }
            )
            continue

        minutes = pd.to_numeric(
            pd.Series([row.get("Minutes played", np.nan)]),
            errors="coerce",
        ).iloc[0]

        rows.append(
            {
                "Slot": slot_name,
                "Player": row["Name"],
                "Team": row["Team"],
                "Data position": row.get("Candidate source", row.get("Score position", "")),
                "Minutes": int(minutes) if pd.notna(minutes) else np.nan,
                "Overall percentile": round(
                    float(row["Position overall"]),
                    1,
                ),
            }
        )

    return pd.DataFrame(rows)


def make_pitch_figure(
    lineup,
    formation: str,
    title: str,
):
    fig = go.Figure()

    # Pitch background.
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        line=dict(
            color="rgba(255,255,255,0.70)",
            width=2,
        ),
        fillcolor="#176B3A",
        layer="below",
    )

    # Halfway line.
    fig.add_shape(
        type="line",
        x0=0,
        y0=50,
        x1=100,
        y1=50,
        line=dict(
            color="rgba(255,255,255,0.60)",
            width=1.5,
        ),
    )

    # Centre circle.
    fig.add_shape(
        type="circle",
        x0=40,
        y0=40,
        x1=60,
        y1=60,
        line=dict(
            color="rgba(255,255,255,0.55)",
            width=1.5,
        ),
    )

    # Penalty boxes.
    for y0, y1 in [(0, 16), (84, 100)]:
        fig.add_shape(
            type="rect",
            x0=20,
            y0=y0,
            x1=80,
            y1=y1,
            line=dict(
                color="rgba(255,255,255,0.55)",
                width=1.4,
            ),
        )

    coords = FORMATION_COORDS[formation]

    x_values = []
    y_values = []
    labels = []
    hover = []

    for slot_name, allowed_positions, row in lineup:
        x, y = coords[slot_name]
        x_values.append(x)
        y_values.append(y)

        if row is None:
            labels.append(f"<b>{slot_name}</b><br>—")
            hover.append(
                f"{slot_name}<br>No eligible player above minutes cutoff"
            )
        else:
            score = float(row["Position overall"])
            short_name = str(row["Name"])

            labels.append(
                f"<b>{slot_name}</b><br>{short_name}<br>{score:.0f}p"
            )

            minutes = pd.to_numeric(
                pd.Series([row.get("Minutes played", np.nan)]),
                errors="coerce",
            ).iloc[0]

            hover.append(
                f"<b>{slot_name} · {short_name}</b>"
                f"<br>Data position: {row.get('Candidate source', '')}"
                f"<br>Overall percentile: {score:.1f}"
                + (
                    f"<br>Minutes: {int(minutes)}"
                    if pd.notna(minutes)
                    else ""
                )
            )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=54,
                color="rgba(10,24,44,0.90)",
                line=dict(
                    color="rgba(255,255,255,0.88)",
                    width=2,
                ),
            ),
            textfont=dict(
                color="white",
                size=10,
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[-4, 104],
            visible=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-3, 103],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=720,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#176B3A",
    )

    return fig





# ============================================================
# TEAM / CLUB PROFILE
# ============================================================

TEAM_STYLE_AXES = {
    "Possession": ["possession"],
    "Directness": ["long_pass_share"],
    "Tempo": ["match_tempo"],
    "Wing play": ["crosses_per_match", "after_crosses_xg_per_match"],
    "Chance creation": ["xg_per_match", "key_passes_per_match", "shots_per_match"],
    "Pressing": ["INV__ppda", "final_third_recoveries_per_match"],
    "1v1 threat": ["final_third_dribbles_per_match"],
    "Finishing": ["goals_per_match", "shots_on_target_pct", "finishing_delta_per_match"],
}


TEAM_FULL_RADAR_METRICS = [
    ("Possession", "possession"),
    ("Pass accuracy", "pass_accuracy"),
    ("Match tempo", "match_tempo"),
    ("Long-pass share", "long_pass_share"),
    ("Pressing intensity", "ppda"),
    ("Shots / match", "shots_per_match"),
    ("xG / match", "xg_per_match"),
    ("Goals / match", "goals_per_match"),
    ("Shots on target %", "shots_on_target_pct"),
    ("Key passes / match", "key_passes_per_match"),
    ("xA / match", "xa_per_match"),
    ("Crosses / match", "crosses_per_match"),
    ("Cross accuracy", "cross_accuracy"),
    ("Final-third dribbles", "final_third_dribbles_per_match"),
    ("Final-third recoveries", "final_third_recoveries_per_match"),
    ("xG after crosses", "after_crosses_xg_per_match"),
    ("Set-piece xG", "after_setpieces_xg_per_match"),
]


@st.cache_data
def load_team_database():
    path = DATA_DIR / "team_data.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    numeric_cols = [
        col for col in df.columns
        if col not in {"Team", "Updated", "Formation_1", "Formation_2", "Formation_3"}
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return prepare_team_percentiles(df)


def team_midrank_percentile(series: pd.Series, lower_is_better: bool = False):
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)

    valid = numeric.dropna()
    if valid.empty:
        return result

    values = valid.to_numpy(dtype=float)

    for idx, value in valid.items():
        if lower_is_better:
            lower = np.sum(values > value)
        else:
            lower = np.sum(values < value)
        equal = np.sum(values == value)
        result.loc[idx] = 100.0 * (lower + 0.5 * equal) / len(values)

    return result


def prepare_team_percentiles(df: pd.DataFrame):
    if df.empty:
        return df

    out = df.copy()

    raw_metrics = set()
    for axis_metrics in TEAM_STYLE_AXES.values():
        for metric in axis_metrics:
            if metric.startswith("INV__"):
                raw_metrics.add(metric.replace("INV__", ""))
            else:
                raw_metrics.add(metric)

    for metric in raw_metrics:
        if metric not in out.columns:
            continue
        inverse = metric == "ppda"
        out[f"PCTL__{metric}"] = team_midrank_percentile(
            out[metric],
            lower_is_better=inverse,
        )

    for axis, axis_metrics in TEAM_STYLE_AXES.items():
        cols = []
        for metric in axis_metrics:
            raw_metric = metric.replace("INV__", "")
            pcol = f"PCTL__{raw_metric}"
            if pcol in out.columns:
                cols.append(pcol)
        if cols:
            out[f"STYLE__{axis}"] = out[cols].mean(axis=1, skipna=True)
        else:
            out[f"STYLE__{axis}"] = np.nan

    return out


def club_style_radar(team_db: pd.DataFrame, teams: list[str]):
    """
    Complete team comparison radar.

    Uses the full set of available team-report benchmarking metrics rather than
    only the 8 composite style axes. Each spoke is a league percentile.
    PPDA is already inverted when percentiles are prepared, so a higher
    percentile always means more intense pressing.
    """
    metric_pairs = [
        (label, metric)
        for label, metric in TEAM_FULL_RADAR_METRICS
        if f"PCTL__{metric}" in team_db.columns
    ]

    colors = ["#FFD900", "#4DA3FF", "#7ED957", "#FF9F43"]
    fig = go.Figure()

    labels = [label for label, _ in metric_pairs]

    for idx, team in enumerate(teams):
        row_df = team_db[
            team_db["Team"].astype(str).eq(str(team))
        ]

        if row_df.empty:
            continue

        row = row_df.iloc[0]
        vals = []
        hover_raw = []

        for label, metric in metric_pairs:
            pct = pd.to_numeric(
                pd.Series([row.get(f"PCTL__{metric}", np.nan)]),
                errors="coerce",
            ).iloc[0]

            raw = pd.to_numeric(
                pd.Series([row.get(metric, np.nan)]),
                errors="coerce",
            ).iloc[0]

            vals.append(0.0 if pd.isna(pct) else float(pct))
            hover_raw.append(raw)

        if not vals:
            continue

        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels + [labels[0]],
                customdata=hover_raw + [hover_raw[0]],
                mode="lines",
                line=dict(
                    color=color,
                    width=3,
                ),
                fill="toself",
                fillcolor=hex_to_rgba(
                    color,
                    0.11 if len(teams) > 1 else 0.20,
                ),
                name=team,
                hovertemplate=(
                    "<b>%{fullData.name}</b>"
                    "<br>%{theta}"
                    "<br>League percentile: <b>%{r:.0f}</b>"
                    "<br>Raw value: %{customdata:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#0B1626",
            gridshape="linear",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                tickfont=dict(
                    size=9,
                    color="rgba(255,255,255,.58)",
                ),
                gridcolor="rgba(255,255,255,.15)",
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(255,255,255,.10)",
                tickfont=dict(
                    size=10,
                    color="#F5F7FA",
                ),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=len(teams) > 1,
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.5,
            xanchor="center",
        ),
        height=760,
        margin=dict(
            l=125,
            r=125,
            t=40,
            b=90,
        ),
    )

    return fig

def team_style_description(row: pd.Series):
    scores = {axis: float(row.get(f"STYLE__{axis}", np.nan)) for axis in TEAM_STYLE_AXES}
    parts = []

    if scores.get("Possession", 50) >= 70:
        parts.append("possession-oriented")
    elif scores.get("Possession", 50) <= 30:
        parts.append("lower-possession")

    if scores.get("Directness", 50) >= 70:
        parts.append("direct")
    elif scores.get("Directness", 50) <= 30:
        parts.append("shorter-passing")

    if scores.get("Wing play", 50) >= 70:
        parts.append("wide / crossing-heavy")
    if scores.get("Pressing", 50) >= 70:
        parts.append("aggressive pressing")
    if scores.get("Tempo", 50) >= 70:
        parts.append("high-tempo")
    if scores.get("1v1 threat", 50) >= 70:
        parts.append("strong 1v1 threat")

    if not parts:
        parts.append("balanced")

    return ", ".join(parts[:4]).capitalize() + "."


def team_attack_route(row: pd.Series):
    wing = float(row.get("STYLE__Wing play", np.nan))
    dribble = float(row.get("STYLE__1v1 threat", np.nan))
    creation = float(row.get("STYLE__Chance creation", np.nan))

    if pd.notna(wing) and wing >= max(dribble if pd.notna(dribble) else 0, creation if pd.notna(creation) else 0) and wing >= 60:
        route = "Wide / crossing"
    elif pd.notna(dribble) and dribble >= 60:
        route = "1v1 / carries"
    elif pd.notna(creation) and creation >= 60:
        route = "Combination / chance creation"
    else:
        route = "Balanced"

    left = float(row.get("cross_left_share", 0) or 0)
    right = float(row.get("cross_right_share", 0) or 0)
    if max(left, right) >= 45 and abs(left-right) >= 15:
        side = "left" if left > right else "right"
        route += f" · more from the {side} side"

    return route


def team_strengths_weaknesses(row: pd.Series):
    scores = [(axis, float(row.get(f"STYLE__{axis}", np.nan))) for axis in TEAM_STYLE_AXES]
    scores = [(a,v) for a,v in scores if pd.notna(v)]
    strengths = sorted(scores, key=lambda x:x[1], reverse=True)[:3]
    weaknesses = sorted(scores, key=lambda x:x[1])[:3]
    return strengths, weaknesses


def key_team_differences(team_db: pd.DataFrame, team_a: str, team_b: str, top_n: int = 5):
    a = team_db[team_db["Team"].astype(str).eq(str(team_a))]
    b = team_db[team_db["Team"].astype(str).eq(str(team_b))]
    if a.empty or b.empty:
        return []
    ra, rb = a.iloc[0], b.iloc[0]
    diffs=[]
    for axis in TEAM_STYLE_AXES:
        va=float(ra.get(f"STYLE__{axis}", np.nan)); vb=float(rb.get(f"STYLE__{axis}", np.nan))
        if pd.notna(va) and pd.notna(vb):
            diffs.append((axis, va, vb, abs(va-vb)))
    return sorted(diffs, key=lambda x:x[3], reverse=True)[:top_n]


def formation_summary(row: pd.Series):
    parts=[]
    for i in range(1,4):
        form=row.get(f"Formation_{i}", "")
        pct=row.get(f"Formation_{i}_pct", np.nan)
        if isinstance(form, str) and form.strip():
            parts.append(f"{form} ({float(pct):.0f}%)" if pd.notna(pct) else form)
    return " · ".join(parts) if parts else "—"

# ============================================================
# MATCH OPPONENT REPORT
# ============================================================

MATCH_REPORT_CATEGORIES = {
    "Attacking threat": [
        "Goals per 90",
        "xG per 90",
        "Shots per 90",
        "Touches in box per 90",
        "Successful attacking actions per 90",
    ],
    "Chance creation": [
        "xA per 90",
        "Shot assists per 90",
        "Key passes per 90",
        "Passes to penalty area per 90",
        "Smart passes per 90",
    ],
    "Progression": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Progressive runs per 90",
        "Passes to final third per 90",
    ],
    "Duels": [
        "Defensive duels won, %",
        "Offensive duels won, %",
        "Aerial duels won, %",
    ],
    "Aerial strength": [
        "Aerial duels won, %",
        "Aerial duels per 90",
        "Head goals per 90",
    ],
}


def opponent_player_pool(
    full_data: pd.DataFrame,
    team: str,
    min_minutes: int,
):
    frames = []

    for position in POSITION_ORDER:
        scored, _ = position_scored_frame(
            full_data,
            position,
            min_minutes,
        )

        if scored.empty:
            continue

        team_rows = scored[
            scored["Team"].astype(str).eq(str(team))
        ].copy()

        if team_rows.empty:
            continue

        team_rows["Data position"] = position
        frames.append(team_rows)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def opponent_category_table(
    pool: pd.DataFrame,
    category_name: str,
    top_n: int = 5,
):
    if pool.empty:
        return pd.DataFrame()

    metrics = MATCH_REPORT_CATEGORIES.get(
        category_name,
        [],
    )

    work = pool.copy()
    available_pctl_cols = [
        f"PCTL__{metric}"
        for metric in metrics
        if f"PCTL__{metric}" in work.columns
    ]

    if not available_pctl_cols:
        return pd.DataFrame()

    for col in available_pctl_cols:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work["Category score"] = work[
        available_pctl_cols
    ].mean(
        axis=1,
        skipna=True,
    )

    work["Metric coverage"] = work[
        available_pctl_cols
    ].notna().sum(axis=1)

    work = work[
        work["Category score"].notna()
    ].copy()

    if work.empty:
        return work

    work = work.sort_values(
        [
            "Category score",
            "Metric coverage",
        ],
        ascending=[False, False],
    )

    # A player can occur in several positional exports.
    # Keep the positional row where he scores best in this category.
    work = work.drop_duplicates(
        subset=["Name", "Team"],
        keep="first",
    )

    display_cols = [
        col
        for col in [
            "Name",
            "Data position",
            "Minutes played",
            "Category score",
        ]
        if col in work.columns
    ]

    return work[
        display_cols
    ].head(top_n).reset_index(drop=True)


def opponent_team_snapshot(
    pool: pd.DataFrame,
):
    rows = []

    for category in MATCH_REPORT_CATEGORIES:
        table = opponent_category_table(
            pool,
            category,
            top_n=999,
        )

        if table.empty:
            score = np.nan
        else:
            score = pd.to_numeric(
                table["Category score"],
                errors="coerce",
            ).mean()

        rows.append(
            {
                "Area": category,
                "Team score": score,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLAYER CARD EXPORT
# ============================================================

def build_player_card_png(
    player_name: str,
    team: str,
    position: str,
    minutes_value,
    row: pd.Series,
    metrics: list[str],
):
    valid_metrics = [
        metric
        for metric in metrics
        if pd.notna(
            row.get(
                f"PCTL__{metric}",
                np.nan,
            )
        )
    ]

    if len(valid_metrics) < 3:
        return b""

    percentiles = [
        float(row[f"PCTL__{metric}"])
        for metric in valid_metrics
    ]

    labels = [
        f"{short_metric_label(metric)} {percentile:.0f}%"
        for metric, percentile in zip(
            valid_metrics,
            percentiles,
        )
    ]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(valid_metrics),
        endpoint=False,
    ).tolist()

    values = percentiles + [percentiles[0]]
    angles_closed = angles + [angles[0]]

    fig = plt.figure(
        figsize=(12, 7),
        facecolor="#081426",
    )

    # Header
    fig.text(
        0.08,
        0.90,
        player_name,
        fontsize=28,
        fontweight="bold",
        color="white",
    )

    minutes_text = (
        f"{int(minutes_value):,}".replace(",", " ")
        if pd.notna(minutes_value)
        else "—"
    )

    fig.text(
        0.08,
        0.845,
        f"{team}  •  {position}  •  {minutes_text} min",
        fontsize=13,
        color="#B9C5D6",
    )

    fig.lines.append(
        plt.Line2D(
            [0.08, 0.92],
            [0.81, 0.81],
            transform=fig.transFigure,
            color="#FFD900",
            linewidth=2.2,
        )
    )

    # Logo
    if LOGO_PATH.exists():
        try:
            logo_ax = fig.add_axes(
                [0.82, 0.82, 0.10, 0.12]
            )
            logo_ax.imshow(
                plt.imread(str(LOGO_PATH))
            )
            logo_ax.axis("off")
        except Exception:
            pass

    # Radar
    ax = fig.add_axes(
        [0.06, 0.10, 0.56, 0.64],
        polar=True,
        facecolor="#081426",
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([])
    ax.grid(
        color="white",
        alpha=0.14,
        linewidth=0.9,
    )
    ax.spines["polar"].set_color(
        (1, 1, 1, 0.20)
    )

    ax.plot(
        angles_closed,
        values,
        color="#F4F6F8",
        linewidth=2.2,
    )
    ax.fill(
        angles_closed,
        values,
        color="#F4F6F8",
        alpha=0.78,
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(
        labels,
        fontsize=9.5,
        color="#E5E9F0",
    )

    # Strongest / weakest summary
    summary = player_trait_summary(
        row,
        valid_metrics,
    )

    fig.text(
        0.67,
        0.68,
        "STRONGEST",
        fontsize=11,
        fontweight="bold",
        color="#FFD900",
    )

    y = 0.635
    for metric, percentile in summary[:3]:
        fig.text(
            0.67,
            y,
            f"{short_metric_label(metric)}",
            fontsize=11,
            color="white",
        )
        fig.text(
            0.91,
            y,
            f"{percentile:.0f}%",
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="right",
        )
        y -= 0.055

    fig.text(
        0.67,
        0.43,
        "LOWEST",
        fontsize=11,
        fontweight="bold",
        color="#FFD900",
    )

    weakest = sorted(
        summary,
        key=lambda item: item[1],
    )[:3]

    y = 0.385
    for metric, percentile in weakest:
        fig.text(
            0.67,
            y,
            f"{short_metric_label(metric)}",
            fontsize=11,
            color="white",
        )
        fig.text(
            0.91,
            y,
            f"{percentile:.0f}%",
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="right",
        )
        y -= 0.055

    fig.text(
        0.67,
        0.17,
        "Percentiles vs current league reference sample",
        fontsize=8.5,
        color="#8391A5",
    )

    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def player_card_pdf_from_png(
    png_bytes: bytes,
):
    if not png_bytes:
        return b""

    output = BytesIO()
    page_size = landscape(A4)
    pdf = canvas.Canvas(
        output,
        pagesize=page_size,
    )

    page_w, page_h = page_size

    image = ImageReader(
        BytesIO(png_bytes)
    )

    margin = 18
    pdf.drawImage(
        image,
        margin,
        margin,
        width=page_w - 2 * margin,
        height=page_h - 2 * margin,
        preserveAspectRatio=True,
        anchor="c",
    )

    pdf.showPage()
    pdf.save()

    output.seek(0)
    return output.getvalue()


# ============================================================
# PLAYER TRAITS RADAR
# ============================================================

def make_player_traits_radar(
    frame: pd.DataFrame,
    player: str,
    metrics: list[str],
):
    """
    Minimal single-player radar inspired by modern player-traits cards:
    dark background, polygon grid, light player shape and percentile
    values embedded directly into the axis labels.
    """
    player_row = frame.loc[
        frame["Name"].astype(str).eq(str(player))
    ]

    if player_row.empty or not metrics:
        return go.Figure()

    row = player_row.iloc[0]

    valid_metrics = [
        metric
        for metric in metrics
        if (
            f"PCTL__{metric}" in frame.columns
            and pd.notna(
                row.get(
                    f"PCTL__{metric}",
                    np.nan,
                )
            )
        )
    ]

    if not valid_metrics:
        return go.Figure()

    percentiles = [
        float(row[f"PCTL__{metric}"])
        for metric in valid_metrics
    ]

    raw_values = [
        pd.to_numeric(
            pd.Series([row.get(metric, np.nan)]),
            errors="coerce",
        ).iloc[0]
        for metric in valid_metrics
    ]

    axis_labels = []

    for metric, percentile in zip(
        valid_metrics,
        percentiles,
    ):
        label = short_metric_label(metric)
        axis_labels.append(
            f"{label} {percentile:.0f}%"
        )

    closed_r = percentiles + [percentiles[0]]
    closed_theta = axis_labels + [axis_labels[0]]
    closed_raw = raw_values + [raw_values[0]]

    fig = go.Figure()

    # subtle 50th percentile reference
    fig.add_trace(
        go.Scatterpolar(
            r=[50] * (len(valid_metrics) + 1),
            theta=closed_theta,
            mode="lines",
            line=dict(
                color="rgba(255,255,255,0.28)",
                width=1.3,
                dash="dot",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=closed_r,
            theta=closed_theta,
            customdata=closed_raw,
            mode="lines",
            line=dict(
                color="rgba(245,247,250,0.96)",
                width=2.2,
            ),
            fill="toself",
            fillcolor="rgba(245,247,250,0.78)",
            name=player,
            hovertemplate=(
                "<b>%{theta}</b>"
                "<br>Percentile: %{r:.0f}"
                "<br>Raw value: %{customdata:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            gridshape="linear",
            radialaxis=dict(
                visible=False,
                range=[0, 100],
                showline=False,
                gridcolor="rgba(255,255,255,0.13)",
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.18)",
                tickfont=dict(
                    size=14,
                    color="#D8DCE2",
                ),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=610,
        margin=dict(
            l=105,
            r=105,
            t=55,
            b=55,
        ),
        hoverlabel=dict(
            bgcolor="#111820",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(
                color="#FFFFFF",
                size=12,
            ),
        ),
    )

    return fig


def player_trait_summary(
    row: pd.Series,
    metrics: list[str],
):
    values = []

    for metric in metrics:
        percentile = row.get(
            f"PCTL__{metric}",
            np.nan,
        )

        if pd.notna(percentile):
            values.append(
                (
                    metric,
                    float(percentile),
                )
            )

    values.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return values


# ============================================================
# TABLES
# ============================================================

def build_player_table(
    frame: pd.DataFrame,
    player: str,
    metrics: list[str],
):
    row = frame.loc[frame["Name"] == player].iloc[0]
    records = []

    for metric in metrics:
        raw = pd.to_numeric(
            pd.Series([row[metric]]),
            errors="coerce",
        ).iloc[0]

        values = pd.to_numeric(
            frame[metric],
            errors="coerce",
        )

        percentile = row[f"PCTL__{metric}"]
        rank = league_rank(metric, raw, frame)

        records.append(
            {
                "Metric": metric,
                "Raw": round(float(raw), 3) if pd.notna(raw) else np.nan,
                "League average": round(float(values.mean()), 3),
                "League median": round(float(values.median()), 3),
                "Percentile": round(float(percentile), 1),
                "Rank": f"{int(rank)} / {values.notna().sum()}",
                "Interpretation": metric_note(metric),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# LOAD POSITION
# ============================================================

all_data, load_warnings = load_all_data()

if all_data.empty:
    st.error(
        "No usable Wyscout Excel files were found in the data/ folder."
    )
    st.stop()

available_positions = [
    position
    for position in POSITION_ORDER
    if position in all_data["PositionGroup"].unique()
]

available_positions += sorted(
    set(all_data["PositionGroup"].unique()) - set(available_positions)
)

# ============================================================
# MAIN APPLICATION MODE
# ============================================================

with st.sidebar:
    st.header("Workspace")

    app_mode = st.radio(
        "Mode",
        ["Player Analysis", "Club vs Opponent"],
        index=0,
        key="app_mode",
        help=(
            "Player Analysis = positional player benchmarking and rankings. "
            "Club vs Opponent = separate team-level lineup workspace."
        ),
    )

if HERO_PATH.exists():
    st.image(
        str(HERO_PATH),
        use_container_width=True,
    )

st.markdown(
    "<div style='height:0.35rem'></div>",
    unsafe_allow_html=True,
)

if app_mode == "Club vs Opponent":
    st.subheader("Club vs Opponent")

    all_team_names = sorted(
        team
        for team in all_data["Team"].dropna().astype(str).unique()
        if team.strip()
    )

    top_controls = st.columns([1.2, 1.2, 1])

    with top_controls[0]:
        home_team = st.selectbox(
            "Club A",
            all_team_names,
            index=(
                all_team_names.index(OWN_TEAM)
                if OWN_TEAM in all_team_names
                else 0
            ),
            key="club_mode_home_team",
        )

    with top_controls[1]:
        away_options = [
            team
            for team in all_team_names
            if team != home_team
        ]

        away_team = st.selectbox(
            "Club B / opponent",
            away_options,
            key="club_mode_away_team",
        )

    with top_controls[2]:
        all_minutes_values = pd.to_numeric(
            all_data.get(
                "Minutes played",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()

        club_max_minutes = (
            int(all_minutes_values.max())
            if not all_minutes_values.empty
            else 1
        )

        club_min_minutes = st.number_input(
            "Minimum minutes",
            min_value=1,
            max_value=max(1, club_max_minutes),
            value=min(450, max(1, club_max_minutes)),
            step=90,
            key="club_mode_min_minutes",
        )

    team_db = load_team_database()

    tab_club_profile, tab_best_xi, tab_match_report = st.tabs(
        [
            "Club Profile",
            "Best XI",
            "Opponent report",
        ]
    )

    with tab_club_profile:
        if team_db.empty:
            st.warning("Team database is missing. Upload data/team_data.csv.")
        elif away_team not in team_db["Team"].astype(str).tolist():
            st.warning(f"No team profile data for {away_team}.")
        else:
            away_row = team_db[team_db["Team"].astype(str).eq(str(away_team))].iloc[0]
            home_row_df = team_db[team_db["Team"].astype(str).eq(str(home_team))]

            st.markdown(f"### {away_team} — Club Profile")
            st.caption(
                f"Team report updated {away_row.get('Updated', '—')} · "
                f"{int(away_row.get('Matches', 0))} league matches in the sample"
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Primary formation", str(away_row.get("Formation_1", "—")))
            k2.metric("Possession", f"{away_row.get('possession', np.nan):.1f}%")
            k3.metric("xG / match", f"{away_row.get('xg_per_match', np.nan):.2f}")
            k4.metric("PPDA", f"{away_row.get('ppda', np.nan):.2f}")

            st.caption("Formation usage: " + formation_summary(away_row))

            left_col, right_col = st.columns([1.65, 1])

            with left_col:
                compare_teams = [away_team]
                if not home_row_df.empty:
                    compare_teams = [home_team, away_team]
                st.caption(
                    "Complete team radar · every spoke is a separate league percentile from the Wyscout team reports."
                )
                st.plotly_chart(
                    club_style_radar(team_db, compare_teams),
                    use_container_width=True,
                    theme=None,
                )

            with right_col:
                st.markdown("#### Style")
                st.write(team_style_description(away_row))
                st.markdown("#### Most likely attacking route")
                st.write(team_attack_route(away_row))

                strengths, weaknesses = team_strengths_weaknesses(away_row)
                st.markdown("#### Strongest team traits")
                for axis, score in strengths:
                    st.write(f"**{axis}** — {score:.0f}p")
                st.markdown("#### Lowest team traits")
                for axis, score in weaknesses:
                    st.write(f"**{axis}** — {score:.0f}p")

            st.markdown("### Key team metrics")
            metric_rows = [
                ("Possession", "possession", "%"),
                ("Pass accuracy", "pass_accuracy", "%"),
                ("Match tempo", "match_tempo", ""),
                ("Long-pass share", "long_pass_share", "%"),
                ("PPDA", "ppda", ""),
                ("Shots / match", "shots_per_match", ""),
                ("xG / match", "xg_per_match", ""),
                ("Goals / match", "goals_per_match", ""),
                ("Crosses / match", "crosses_per_match", ""),
                ("Cross accuracy", "cross_accuracy", "%"),
                ("Key passes / match", "key_passes_per_match", ""),
                ("Final-third recoveries / match", "final_third_recoveries_per_match", ""),
                ("Final-third dribbles / match", "final_third_dribbles_per_match", ""),
                ("Set-piece xG / match", "after_setpieces_xg_per_match", ""),
            ]
            metric_table=[]
            for label, metric, suffix in metric_rows:
                val=away_row.get(metric, np.nan)
                pct=away_row.get(f"PCTL__{metric}", np.nan)
                metric_table.append({
                    "Metric": label,
                    "Value": (f"{val:.1f}{suffix}" if suffix == "%" and pd.notna(val) else f"{val:.2f}" if pd.notna(val) else "—"),
                    "League percentile": pct,
                })
            st.dataframe(
                pd.DataFrame(metric_table),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "League percentile": st.column_config.ProgressColumn(
                        "League percentile", min_value=0, max_value=100, format="%.0f"
                    )
                },
            )

            if not home_row_df.empty:
                st.markdown(f"### {home_team} vs {away_team} — biggest style differences")
                for axis, home_score, away_score, diff in key_team_differences(team_db, home_team, away_team):
                    leader = home_team if home_score > away_score else away_team
                    st.write(
                        f"**{axis}:** {home_team} {home_score:.0f}p · "
                        f"{away_team} {away_score:.0f}p — **{leader} +{diff:.0f}p**"
                    )

            st.info(
                "Team percentiles are calculated from the 16 uploaded Wyscout team reports. "
                "Volume metrics are normalized per match. Formation and attacking-route text is descriptive; "
                "the left/right note is estimated from cross takers' listed positions in the report."
            )

    with tab_best_xi:
        st.caption(
            "The XI is built from the best available positional overall percentiles "
            "for each formation slot. Each player can be used only once."
        )

        formation_cols = st.columns(2)

        with formation_cols[0]:
            home_formation = st.selectbox(
                f"{home_team} formation",
                list(FORMATION_SLOTS.keys()),
                index=0,
                key="home_formation",
            )

        with formation_cols[1]:
            away_formation = st.selectbox(
                f"{away_team} formation",
                list(FORMATION_SLOTS.keys()),
                index=0,
                key="away_formation",
            )

        allow_wingers = st.toggle(
            "Allow RW/LW players as wing-back alternatives in 3-at-the-back systems",
            value=False,
            help=(
                "LWB normally selects from LB and RWB from RB. "
                "Turn this on if you also want to consider natural wingers as tactical wing-backs."
            ),
        )

        home_lineup = build_best_xi(
            all_data,
            home_team,
            home_formation,
            int(club_min_minutes),
            allow_winger_wingbacks=allow_wingers,
        )

        away_lineup = build_best_xi(
            all_data,
            away_team,
            away_formation,
            int(club_min_minutes),
            allow_winger_wingbacks=allow_wingers,
        )

        home_df = lineup_dataframe(home_lineup)
        away_df = lineup_dataframe(away_lineup)

        score_cols = st.columns(2)

        with score_cols[0]:
            valid_scores = home_df[
                "Overall percentile"
            ].dropna()

            home_score = (
                valid_scores.mean()
                if not valid_scores.empty
                else np.nan
            )

            st.metric(
                f"{home_team} XI average",
                f"{home_score:.1f}p"
                if pd.notna(home_score)
                else "—",
            )

        with score_cols[1]:
            valid_scores = away_df[
                "Overall percentile"
            ].dropna()

            away_score = (
                valid_scores.mean()
                if not valid_scores.empty
                else np.nan
            )

            st.metric(
                f"{away_team} XI average",
                f"{away_score:.1f}p"
                if pd.notna(away_score)
                else "—",
            )

        pitch_cols = st.columns(2)

        with pitch_cols[0]:
            st.plotly_chart(
                make_pitch_figure(
                    home_lineup,
                    home_formation,
                    f"{home_team} · {home_formation}",
                ),
                use_container_width=True,
                theme=None,
            )

            st.dataframe(
                home_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Overall percentile": st.column_config.ProgressColumn(
                        "Overall percentile",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    )
                },
            )

        with pitch_cols[1]:
            st.plotly_chart(
                make_pitch_figure(
                    away_lineup,
                    away_formation,
                    f"{away_team} · {away_formation}",
                ),
                use_container_width=True,
                theme=None,
            )

            st.dataframe(
                away_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Overall percentile": st.column_config.ProgressColumn(
                        "Overall percentile",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    )
                },
            )

        missing_home = home_df[
            home_df["Player"] == "—"
        ]["Slot"].tolist()

        missing_away = away_df[
            away_df["Player"] == "—"
        ]["Slot"].tolist()

        if missing_home or missing_away:
            messages = []

            if missing_home:
                messages.append(
                    f"{home_team}: missing {', '.join(missing_home)}"
                )

            if missing_away:
                messages.append(
                    f"{away_team}: missing {', '.join(missing_away)}"
                )

            st.warning(
                "Not enough eligible players for a complete XI at the current minutes cutoff. "
                + " | ".join(messages)
            )

    with tab_match_report:
        st.markdown(
            f"### {away_team} — opponent profile"
        )

        st.caption(
            "This report ranks the opponent's players by category using "
            "their positional league percentiles. It is a scouting aid, "
            "not a tactical verdict."
        )

        opponent_pool = opponent_player_pool(
            all_data,
            away_team,
            int(club_min_minutes),
        )

        if opponent_pool.empty:
            st.warning(
                "No eligible opponent players at the current minutes cutoff."
            )
        else:
            snapshot = opponent_team_snapshot(
                opponent_pool
            )

            snapshot_cols = st.columns(
                len(snapshot)
            )

            for idx, snapshot_row in snapshot.iterrows():
                score = snapshot_row["Team score"]

                with snapshot_cols[idx]:
                    st.metric(
                        snapshot_row["Area"],
                        (
                            f"{score:.0f}p"
                            if pd.notna(score)
                            else "—"
                        ),
                    )

            st.markdown("### Key players")

            category_names = list(
                MATCH_REPORT_CATEGORIES.keys()
            )

            for row_start in range(
                0,
                len(category_names),
                2,
            ):
                cols = st.columns(2)

                for offset in range(2):
                    category_index = row_start + offset

                    if category_index >= len(category_names):
                        continue

                    category = category_names[
                        category_index
                    ]

                    table = opponent_category_table(
                        opponent_pool,
                        category,
                        top_n=5,
                    )

                    with cols[offset]:
                        st.markdown(
                            f"#### {category}"
                        )

                        if table.empty:
                            st.caption(
                                "Not enough comparable metrics."
                            )
                        else:
                            st.dataframe(
                                table,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Category score": st.column_config.ProgressColumn(
                                        "Score",
                                        min_value=0,
                                        max_value=100,
                                        format="%.0f",
                                    ),
                                },
                            )

            st.info(
                "A player may appear in more than one positional Wyscout export. "
                "For each category, the app keeps the positional profile in which "
                "that player scores highest."
            )

    st.stop()


def _apply_quick_minutes(slider_key, quick_key):
    value = st.session_state.get(quick_key, "Custom")

    if value != "Custom":
        st.session_state[slider_key] = int(value)


def _sync_quick_minutes(slider_key, quick_key, quick_values):
    value = int(st.session_state.get(slider_key, 1))

    if value in quick_values:
        st.session_state[quick_key] = value
    else:
        st.session_state[quick_key] = "Custom"


with st.sidebar:
    st.header("Player Analysis")

    selected_position = st.selectbox(
        "Position",
        available_positions,
    )


# Full position dataset is kept intact for manual Comparison.
full_position_df = all_data[
    all_data["PositionGroup"] == selected_position
].copy()

metrics = get_metrics(full_position_df)

if not metrics:
    st.error("No numeric scouting metrics were detected for this position.")
    st.stop()

# ============================================================
# GLOBAL MINUTES FILTER
# Applies to Overview, Our player vs league and League ranking.
# Comparison keeps every player selectable.
# ============================================================

minimum_minutes = 1
maximum_minutes = 1

if "Minutes played" in full_position_df.columns:
    all_minutes = pd.to_numeric(
        full_position_df["Minutes played"],
        errors="coerce",
    )

    if all_minutes.notna().any():
        maximum_minutes = max(1, int(all_minutes.max()))

        safe_position_key = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            selected_position,
        )

        slider_key = f"minutes_slider_{safe_position_key}"
        quick_key = f"minutes_quick_{safe_position_key}"

        quick_values = list(
            range(
                90,
                maximum_minutes + 1,
                90,
            )
        )

        quick_options = ["Custom"] + quick_values

        if slider_key not in st.session_state:
            st.session_state[slider_key] = 1

        if st.session_state[slider_key] > maximum_minutes:
            st.session_state[slider_key] = maximum_minutes

        if quick_key not in st.session_state:
            st.session_state[quick_key] = "Custom"

        if st.session_state[quick_key] not in quick_options:
            st.session_state[quick_key] = "Custom"

        with st.sidebar:
            st.divider()
            st.subheader("Minutes filter")

            minimum_minutes = st.slider(
                "Minimum minutes played",
                min_value=1,
                max_value=maximum_minutes,
                key=slider_key,
                step=1,
                help=(
                    "Used for league benchmark, percentiles and rankings. "
                    "Move by one minute for an exact cutoff."
                ),
                on_change=_sync_quick_minutes,
                args=(
                    slider_key,
                    quick_key,
                    quick_values,
                ),
            )

            st.selectbox(
                "Quick select (90-minute steps)",
                options=quick_options,
                key=quick_key,
                format_func=lambda x: (
                    "Custom / exact slider value"
                    if x == "Custom"
                    else f"{x} min ({x // 90} × 90)"
                ),
                on_change=_apply_quick_minutes,
                args=(
                    slider_key,
                    quick_key,
                ),
            )


reference_raw_df = full_position_df.copy()

if "Minutes played" in reference_raw_df.columns:
    reference_minutes = pd.to_numeric(
        reference_raw_df["Minutes played"],
        errors="coerce",
    ).fillna(0)

    reference_raw_df = reference_raw_df[
        reference_minutes >= minimum_minutes
    ].copy()


if reference_raw_df.empty:
    st.error(
        "The selected minimum-minutes threshold leaves no players "
        "in the reference sample."
    )
    st.stop()


# position_df = filtered reference cohort used by league-facing tabs.
position_df = prepare_position_data(
    reference_raw_df,
    metrics,
)

# comparison_df = all players, but their percentiles are calculated
# against the currently filtered reference cohort.
comparison_df = prepare_position_data(
    full_position_df,
    metrics,
    reference_frame=reference_raw_df,
)

our_players = position_df.loc[
    position_df["Our player"],
    "Name",
].tolist()

comparison_our_players = comparison_df.loc[
    comparison_df["Our player"],
    "Name",
].tolist()

teams = sorted(
    team
    for team in position_df["Team"].dropna().astype(str).unique()
    if team.strip()
)

comparison_teams = sorted(
    team
    for team in comparison_df["Team"].dropna().astype(str).unique()
    if team.strip()
)

with st.sidebar:
    team_filter = st.multiselect(
        "Team filter",
        teams,
    )

    st.divider()
    st.caption("OUR TEAM")
    st.write(OWN_TEAM)

    st.caption("REFERENCE SAMPLE")
    st.write(
        f"{len(position_df)} / {len(full_position_df)} players"
    )
    st.write(
        f"Minimum: {minimum_minutes} min"
    )
    st.write(
        f"{position_df['Team'].nunique()} clubs"
    )
    st.write(
        f"{len(metrics)} Wyscout metrics"
    )

    if len(position_df) < 10:
        st.warning(
            "Small reference sample. Percentiles may be unstable."
        )


# ============================================================
# HEADER
# ============================================================

st.subheader(f"{selected_position} Analysis")

a, b, c, d = st.columns(4)
a.metric("Position", selected_position)
b.metric("Reference players", f"{len(position_df)} / {len(full_position_df)}")
c.metric("Our players", len(our_players))
d.metric("Metrics", len(metrics))

st.info(
    "Each position uses only the metrics contained in its own Wyscout export. "
    f"League tabs use players with at least {minimum_minutes} minutes. "
    "Manual Comparison keeps every player selectable, while its percentiles are "
    "still benchmarked against the filtered reference sample."
)

if load_warnings:
    with st.expander("Import warnings"):
        for warning in load_warnings:
            st.write(warning)


# ============================================================
# TABS
# ============================================================

tab_overview, tab_our, tab_compare, tab_ranking = st.tabs(
    [
        "Overview",
        "Our player vs league",
        "Comparison",
        "League ranking",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    st.subheader(f"{selected_position} — league overview")

    view = position_df.copy()

    if team_filter:
        view = view[
            view["Team"].isin(team_filter)
        ]

    quick_metrics = preferred_metrics(
        selected_position,
        metrics,
        maximum=min(8, len(metrics)),
    )

    # --------------------------------------------------------
    # PLAYER SEARCH
    # --------------------------------------------------------

    player_options = sorted(
        view["Name"].dropna().astype(str).unique()
    )

    if (
        "overview_selected_player" not in st.session_state
        or st.session_state["overview_selected_player"]
        not in player_options
    ):
        st.session_state["overview_selected_player"] = (
            player_options[0]
            if player_options
            else None
        )

    search_col, info_col = st.columns(
        [2.2, 1],
        vertical_alignment="bottom",
    )

    with search_col:
        selected_overview_player = st.selectbox(
            "Search player",
            player_options,
            index=(
                player_options.index(
                    st.session_state[
                        "overview_selected_player"
                    ]
                )
                if (
                    player_options
                    and st.session_state.get(
                        "overview_selected_player"
                    ) in player_options
                )
                else 0
            ),
            key=f"overview_player_search_{selected_position}",
            help=(
                "Start typing the player's name. "
                "You can also click a row in the table below."
            ),
        )

    if selected_overview_player:
        st.session_state[
            "overview_selected_player"
        ] = selected_overview_player

    with info_col:
        st.caption(
            f"{len(view)} players in current sample"
        )

    # --------------------------------------------------------
    # LEAGUE TABLE — CLICK TO OPEN
    # --------------------------------------------------------

    overview_base_columns = ["Name", "Team"]

    if "Minutes played" in view.columns:
        overview_base_columns.append("Minutes played")

    overview_base_columns.append("Our player")

    overview_display = view[
        overview_base_columns + quick_metrics
    ].reset_index(drop=True)

    overview_event = st.dataframe(
        overview_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Our player": st.column_config.CheckboxColumn("OUR"),
        },
        on_select="rerun",
        selection_mode="single-row",
        key=f"overview_table_select_{selected_position}",
    )

    try:
        selected_rows = overview_event.selection.rows
    except Exception:
        selected_rows = []

    if selected_rows:
        selected_index = selected_rows[0]

        if 0 <= selected_index < len(overview_display):
            clicked_player = str(
                overview_display.iloc[
                    selected_index
                ]["Name"]
            )

            if (
                clicked_player
                != st.session_state.get(
                    "overview_selected_player"
                )
            ):
                st.session_state[
                    "overview_selected_player"
                ] = clicked_player
                st.rerun()

    st.caption(
        "Type a name above or click a player row to open the player traits card."
    )

    # --------------------------------------------------------
    # PLAYER TRAITS CARD
    # --------------------------------------------------------

    selected_player = st.session_state.get(
        "overview_selected_player"
    )

    if selected_player:
        player_rows = position_df[
            position_df["Name"].astype(str).eq(
                str(selected_player)
            )
        ]

        if not player_rows.empty:
            player_row = player_rows.iloc[0]

            st.markdown("---")
            st.markdown("### Player traits")

            player_team = str(
                player_row.get(
                    "Team",
                    "",
                )
            )

            minutes_value = pd.to_numeric(
                pd.Series([
                    player_row.get(
                        "Minutes played",
                        np.nan,
                    )
                ]),
                errors="coerce",
            ).iloc[0]

            meta_1, meta_2, meta_3 = st.columns(3)

            meta_1.metric(
                "Player",
                selected_player,
            )

            meta_2.metric(
                "Team",
                player_team,
            )

            meta_3.metric(
                "Minutes",
                (
                    f"{int(minutes_value):,}".replace(",", " ")
                    if pd.notna(minutes_value)
                    else "—"
                ),
            )

            default_trait_metrics = preferred_metrics(
                selected_position,
                metrics,
                maximum=min(6, len(metrics)),
            )

            trait_metrics = st.multiselect(
                "Traits shown on radar",
                metrics,
                default=default_trait_metrics,
                max_selections=min(
                    8,
                    len(metrics),
                ),
                key=f"overview_traits_{selected_position}",
                help=(
                    "Choose up to 8 metrics. "
                    "The number next to each trait is the league percentile."
                ),
            )

            if len(trait_metrics) >= 3:
                radar_col, summary_col = st.columns(
                    [2.2, 1],
                    vertical_alignment="top",
                )

                with radar_col:
                    st.plotly_chart(
                        make_player_traits_radar(
                            position_df,
                            selected_player,
                            trait_metrics,
                        ),
                        use_container_width=True,
                        theme=None,
                    )

                with summary_col:
                    trait_values = player_trait_summary(
                        player_row,
                        trait_metrics,
                    )

                    st.markdown("#### Strongest")

                    for metric_name, percentile in trait_values[:3]:
                        st.write(
                            f"**{short_metric_label(metric_name)}**  "
                            f"{percentile:.0f}%"
                        )

                    st.markdown("#### Weakest")

                    weakest = sorted(
                        trait_values,
                        key=lambda item: item[1],
                    )[:3]

                    for metric_name, percentile in weakest:
                        st.write(
                            f"**{short_metric_label(metric_name)}**  "
                            f"{percentile:.0f}%"
                        )

                    st.caption(
                        "Percentiles are calculated against the current "
                        "league reference sample and minimum-minutes filter."
                    )

                player_card_png = build_player_card_png(
                    selected_player,
                    player_team,
                    selected_position,
                    minutes_value,
                    player_row,
                    trait_metrics,
                )

                player_card_pdf = player_card_pdf_from_png(
                    player_card_png
                )

                export_1, export_2 = st.columns(2)

                safe_player_name = re.sub(
                    r"[^A-Za-z0-9_-]+",
                    "_",
                    selected_player,
                ).strip("_")

                with export_1:
                    st.download_button(
                        "Download player card · PNG",
                        data=player_card_png,
                        file_name=(
                            f"{safe_player_name}_player_card.png"
                        ),
                        mime="image/png",
                        use_container_width=True,
                    )

                with export_2:
                    st.download_button(
                        "Download player card · PDF",
                        data=player_card_pdf,
                        file_name=(
                            f"{safe_player_name}_player_card.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.info(
                    "Select at least 3 metrics to display the radar."
                )

    st.markdown("---")
    st.markdown("### Metric explorer")

    metric = st.selectbox(
        "Select metric",
        metrics,
        key="overview_metric",
    )

    ranking = position_df[
        [
            "Name",
            "Team",
            "Our player",
            metric,
            f"PCTL__{metric}",
        ]
    ].copy()

    ranking.columns = [
        "Name",
        "Team",
        "Our player",
        "Raw",
        "Percentile",
    ]

    ranking = ranking.sort_values(
        "Percentile",
        ascending=False,
    )

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1),
    )

    st.caption(metric_note(metric))

    ranking = ranking.reset_index(drop=True)

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Our player": st.column_config.CheckboxColumn("OUR"),
            "Percentile": st.column_config.ProgressColumn(
                "Percentile",
                min_value=0,
                max_value=100,
                format="%.0f",
            ),
        },
    )


# ============================================================
# OUR PLAYER VS LEAGUE
# ============================================================

with tab_our:
    st.subheader(f"Our {selected_position} vs league")

    if not our_players:
        st.warning(
            f"No {selected_position} player with Team = {OWN_TEAM} was found."
        )
    else:
        selected_player = st.selectbox(
            "Our player",
            our_players,
            format_func=lambda x: player_label(position_df, x),
            key="our_player",
        )

        selected_metrics = st.multiselect(
            "Radar metrics",
            metrics,
            default=preferred_metrics(
                selected_position,
                metrics,
                maximum=min(8, len(metrics)),
            ),
            key="our_radar_metrics",
        )

        if selected_metrics:
            row = position_df.loc[
                position_df["Name"] == selected_player
            ].iloc[0]

            percentiles = [
                row[f"PCTL__{metric}"]
                for metric in selected_metrics
            ]

            best_metric = max(
                selected_metrics,
                key=lambda m: row[f"PCTL__{m}"],
            )

            lowest_metric = min(
                selected_metrics,
                key=lambda m: row[f"PCTL__{m}"],
            )

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Average selected percentile",
                f"{np.nanmean(percentiles):.0f}p",
            )

            k2.metric(
                "Highest percentile",
                f"{row[f'PCTL__{best_metric}']:.0f}p",
                best_metric,
            )

            k3.metric(
                "Lowest percentile",
                f"{row[f'PCTL__{lowest_metric}']:.0f}p",
                lowest_metric,
            )

            radar = make_radar(
                position_df,
                [selected_player],
                selected_metrics,
                show_fill=True,
            )

            st.plotly_chart(
                radar,
                use_container_width=True,
                theme="streamlit",
            )

            st.dataframe(
                build_player_table(
                    position_df,
                    selected_player,
                    selected_metrics,
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Percentile": st.column_config.ProgressColumn(
                        "Percentile",
                        min_value=0,
                        max_value=100,
                        format="%.0f",
                    )
                },
            )


# ============================================================
# COMPARISON
# ============================================================

with tab_compare:
    st.subheader(f"{selected_position} comparison")

    st.caption(
        f"All {len(comparison_df)} players remain selectable here regardless of minutes. "
        f"Percentiles and ranks are benchmarked against the {len(position_df)} players "
        f"with at least {minimum_minutes} minutes."
    )

    if comparison_our_players:
        compare_mode = st.radio(
            "Comparison mode",
            [
                "Our player vs opponent",
                "Manual 2–4 players",
            ],
            horizontal=True,
        )
    else:
        compare_mode = "Manual 2–4 players"

    if compare_mode == "Our player vs opponent":
        left, right = st.columns(2)

        with left:
            our_pick = st.selectbox(
                "Our player",
                comparison_our_players,
                format_func=lambda x: player_label(comparison_df, x),
                key="compare_our_player",
            )

        opponent_teams = [
            team
            for team in comparison_teams
            if team != OWN_TEAM
        ]

        with right:
            opponent_team = st.selectbox(
                "Opponent team",
                opponent_teams,
                key="opponent_team",
            )

            opponent_players = comparison_df.loc[
                comparison_df["Team"] == opponent_team,
                "Name",
            ].tolist()

            opponent_pick = st.selectbox(
                "Opponent player",
                opponent_players,
                format_func=lambda x: player_label(comparison_df, x),
                key="opponent_player",
            )

        compare_players = [
            our_pick,
            opponent_pick,
        ]

    else:
        defaults = comparison_our_players[:1]

        defaults += [
            player
            for player in comparison_df["Name"].tolist()
            if player not in defaults
        ][: max(0, 2 - len(defaults))]

        compare_players = st.multiselect(
            "Select 2–4 players",
            comparison_df["Name"].tolist(),
            default=defaults,
            max_selections=4,
            format_func=lambda x: player_label(comparison_df, x),
            key="manual_compare_players",
        )

    compare_metrics = st.multiselect(
        "Comparison metrics",
        metrics,
        default=preferred_metrics(
            selected_position,
            metrics,
            maximum=min(8, len(metrics)),
        ),
        key="comparison_metrics",
    )

    if compare_players and "Minutes played" in comparison_df.columns:
        below_cutoff = []

        for player in compare_players:
            player_row = comparison_df.loc[
                comparison_df["Name"] == player
            ].iloc[0]

            player_minutes = pd.to_numeric(
                pd.Series([player_row["Minutes played"]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(player_minutes) and player_minutes < minimum_minutes:
                below_cutoff.append(
                    f"{player} ({int(player_minutes)} min)"
                )

        if below_cutoff:
            st.warning(
                "Below the current reference cutoff: "
                + ", ".join(below_cutoff)
                + ". They are still available for manual comparison, "
                "but their sample is smaller than the benchmark threshold."
            )

    if len(compare_players) >= 2 and compare_metrics:
        st.markdown("### Comparison profile")

        visual_mode = st.radio(
            "Visualisation",
            ["Radar", "Percentile bars", "Individual radars"],
            horizontal=True,
            help=(
                "Radar is best for direct comparison. Percentile bars can be "
                "clearer for 3–4 players."
            ),
        )

        if visual_mode == "Radar":
            show_fill = st.toggle(
                "Radar fill",
                value=True,
                help=(
                    "Filled classic scouting radar. Fill becomes lighter "
                    "when more players are selected."
                ),
            )

            radar = make_radar(
                comparison_df,
                compare_players,
                compare_metrics,
                show_fill=show_fill,
            )

            st.plotly_chart(
                radar,
                use_container_width=True,
                theme="streamlit",
            )

            st.caption(
                "Hover over a point to see both league percentile "
                "and the raw Wyscout value."
            )

        elif visual_mode == "Percentile bars":
            bars = make_percentile_bars(
                comparison_df,
                compare_players,
                compare_metrics,
            )

            st.plotly_chart(
                bars,
                use_container_width=True,
                theme="streamlit",
            )

            st.caption(
                "Dotted line = 50th percentile of the filtered reference sample."
            )

        else:
            show_individual_radars(
                comparison_df,
                compare_players,
                compare_metrics,
            )

        comparison_rows = []

        for metric in compare_metrics:
            record = {
                "Metric": metric,
                "Interpretation": metric_note(metric),
            }

            for player in compare_players:
                row = comparison_df.loc[
                    comparison_df["Name"] == player
                ].iloc[0]

                raw = pd.to_numeric(
                    pd.Series([row[metric]]),
                    errors="coerce",
                ).iloc[0]

                percentile = row[f"PCTL__{metric}"]

                # Rank is measured against the filtered league reference.
                rank = league_rank(
                    metric,
                    raw,
                    position_df,
                )

                record[player] = (
                    f"{raw:.3f} | "
                    f"{percentile:.0f}p | "
                    f"#{rank}/{len(position_df)}"
                )

            comparison_rows.append(record)

        st.caption(
            "Cell format: raw value | league percentile | rank vs reference sample"
        )

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Selected-metric summary")

        summary_rows = []

        for player in compare_players:
            row = comparison_df.loc[
                comparison_df["Name"] == player
            ].iloc[0]

            selected_percentiles = [
                row[f"PCTL__{metric}"]
                for metric in compare_metrics
            ]

            summary_rows.append(
                {
                    "Player": player,
                    "Team": row["Team"],
                    "Minutes": int(row["Minutes played"])
                    if pd.notna(row.get("Minutes played", np.nan))
                    else np.nan,
                    "Average selected percentile": round(
                        float(np.nanmean(selected_percentiles)),
                        1,
                    ),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            "Average selected percentile",
            ascending=False,
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Average selected percentile": st.column_config.ProgressColumn(
                    "Average selected percentile",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                )
            },
        )

    elif len(compare_players) < 2:
        st.info("Select at least two players.")


# ============================================================
# LEAGUE RANKING
# ============================================================

with tab_ranking:
    st.subheader(f"{selected_position} league ranking")

    ranking_mode = st.radio(
        "Ranking type",
        [
            "Multi-metric overall",
            "Single metric",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # MULTI-METRIC RANKING
    # --------------------------------------------------------

    if ranking_mode == "Multi-metric overall":
        default_ranking_metrics = preferred_metrics(
            selected_position,
            metrics,
            maximum=min(8, len(metrics)),
        )

        ranking_metrics = st.multiselect(
            "Metrics included in overall ranking",
            metrics,
            default=default_ranking_metrics,
            key="overall_ranking_metrics",
        )

        if not ranking_metrics:
            st.info("Select at least one metric.")
        else:
            selected_context = [
                metric
                for metric in ranking_metrics
                if metric in CONTEXT_METRICS
            ]

            if selected_context:
                st.warning(
                    "The selected set contains profile/volume metrics. "
                    "Their high percentile is included in the overall score, "
                    "but it should not automatically be interpreted as higher quality."
                )

            weights = {
                metric: 1.0
                for metric in ranking_metrics
            }

            with st.expander("Advanced weights"):
                use_weights = st.toggle(
                    "Use custom metric weights",
                    value=False,
                    key="use_custom_weights",
                )

                if use_weights:
                    st.caption(
                        "Weights are relative and automatically normalized. "
                        "For example, 2.0 counts twice as much as 1.0."
                    )

                    weight_columns = st.columns(2)

                    for index, metric in enumerate(ranking_metrics):
                        with weight_columns[index % 2]:
                            weights[metric] = st.number_input(
                                metric,
                                min_value=0.0,
                                max_value=10.0,
                                value=1.0,
                                step=0.25,
                                key=f"weight_{selected_position}_{metric}",
                            )

            table_value_mode = st.radio(
                "Show individual metrics as",
                ["Raw values", "Percentiles"],
                horizontal=True,
                index=0,
                help=(
                    "Overall percentile is always calculated from league percentiles. "
                    "This setting changes only how the individual metric columns are displayed."
                ),
            )

            overall_rows = []

            for _, row in position_df.iterrows():
                score = overall_score(
                    row,
                    ranking_metrics,
                    weights,
                )

                avg_rank = average_metric_rank(
                    row,
                    ranking_metrics,
                    position_df,
                )

                record = {
                    "Name": row["Name"],
                    "Team": row["Team"],
                    "Minutes played": row.get("Minutes played", np.nan),
                    "Our player": row["Our player"],
                    "Overall percentile": score,
                    "Average metric rank": avg_rank,
                }

                for metric in ranking_metrics:
                    display_name = short_metric_label(metric).replace("<br>", " ")

                    if table_value_mode == "Raw values":
                        record[display_name] = pd.to_numeric(
                            pd.Series([row[metric]]),
                            errors="coerce",
                        ).iloc[0]
                    else:
                        record[display_name] = row[f"PCTL__{metric}"]

                overall_rows.append(record)

            overall_df = pd.DataFrame(overall_rows)

            overall_df = overall_df.sort_values(
                ["Overall percentile", "Name"],
                ascending=[False, True],
            ).reset_index(drop=True)

            overall_df.insert(
                0,
                "Overall rank",
                range(1, len(overall_df) + 1),
            )

            overall_df["Overall percentile"] = (
                overall_df["Overall percentile"].round(1)
            )

            overall_df["Average metric rank"] = (
                overall_df["Average metric rank"].round(1)
            )

            metric_display_columns = [
                short_metric_label(metric).replace("<br>", " ")
                for metric in ranking_metrics
            ]

            if table_value_mode == "Percentiles":
                for col in metric_display_columns:
                    overall_df[col] = overall_df[col].round(1)
            else:
                for col in metric_display_columns:
                    overall_df[col] = pd.to_numeric(
                        overall_df[col],
                        errors="coerce",
                    ).round(3)

            st.caption(
                "Overall percentile = weighted average of the selected league percentiles. "
                "Overall rank is calculated from that combined score. "
                "The individual metric columns can show either raw Wyscout values or percentiles."
            )

            column_config = {
                "Our player": st.column_config.CheckboxColumn("OUR"),
                "Overall percentile": st.column_config.ProgressColumn(
                    "Overall percentile",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                ),
            }

            if table_value_mode == "Percentiles":
                for col in metric_display_columns:
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format="%.1f",
                        help="League percentile",
                    )

            overall_display = overall_df[
                [
                    "Overall rank",
                    "Name",
                    "Team",
                    "Minutes played",
                    "Our player",
                    "Overall percentile",
                    "Average metric rank",
                ]
                + metric_display_columns
            ].reset_index(drop=True)

            st.dataframe(
                overall_display,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )

            top_n = min(10, len(overall_df))

            ranking_chart = go.Figure(
                go.Bar(
                    x=overall_df.head(top_n)["Overall percentile"][::-1],
                    y=overall_df.head(top_n)["Name"][::-1],
                    orientation="h",
                    text=overall_df.head(top_n)["Overall percentile"][::-1].map(
                        lambda x: f"{x:.1f}"
                    ),
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Overall percentile: %{x:.1f}"
                        "<extra></extra>"
                    ),
                )
            )

            ranking_chart.update_layout(
                title=f"Top {top_n} — combined percentile",
                xaxis=dict(
                    range=[0, 100],
                    title="Overall percentile",
                ),
                yaxis=dict(title=""),
                height=460,
                margin=dict(l=30, r=45, t=55, b=45),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                ranking_chart,
                use_container_width=True,
                theme="streamlit",
            )

    # --------------------------------------------------------
    # SINGLE METRIC
    # --------------------------------------------------------

    else:
        ranking_metric = st.selectbox(
            "Ranking metric",
            metrics,
            key="league_ranking_metric",
        )

        rank_data = []

        for _, row in position_df.iterrows():
            raw = pd.to_numeric(
                pd.Series([row[ranking_metric]]),
                errors="coerce",
            ).iloc[0]

            rank_data.append(
                {
                    "Name": row["Name"],
                    "Team": row["Team"],
                    "Minutes played": row.get("Minutes played", np.nan),
                    "Our player": row["Our player"],
                    "Raw": raw,
                    "Percentile": row[f"PCTL__{ranking_metric}"],
                    "Rank": league_rank(
                        ranking_metric,
                        raw,
                        position_df,
                    ),
                }
            )

        ranking_df = pd.DataFrame(rank_data).sort_values(
            ["Rank", "Name"]
        )

        st.caption(metric_note(ranking_metric))

        ranking_df = ranking_df.reset_index(drop=True)

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Our player": st.column_config.CheckboxColumn("OUR"),
                "Percentile": st.column_config.ProgressColumn(
                    "Percentile",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                ),
            },
        )

        values = pd.to_numeric(
            position_df[ranking_metric],
            errors="coerce",
        ).dropna()

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "League average",
            f"{values.mean():.3f}",
        )

        r2.metric(
            "League median",
            f"{values.median():.3f}",
        )

        r3.metric(
            "25th raw percentile",
            f"{values.quantile(.25):.3f}",
        )

        r4.metric(
            "75th raw percentile",
            f"{values.quantile(.75):.3f}",
        )
