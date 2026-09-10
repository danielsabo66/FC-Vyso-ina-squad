from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Vysočina Scouting Benchmark",
    page_icon="⚽",
    layout="wide",
)

OWN_TEAM = "Vysočina Jihlava"
DATA_DIR = Path(__file__).parent / "data"

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
    "#00B8FF",  # cyan
    "#FF5A5F",  # red
    "#00D084",  # green
    "#FFB000",  # amber
    "#B784FF",  # purple
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
        "Clean sheets",
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


def prepare_position_data(frame: pd.DataFrame, metrics: list[str]):
    output = frame.copy()

    for metric in metrics:
        output[f"PCTL__{metric}"] = output[metric].apply(
            lambda x, m=metric: display_percentile(m, x, output)
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
    fig = go.Figure()

    metric_labels = [
        short_metric_label(metric)
        for metric in metrics
    ]

    fill_alpha = 0.13 if len(players) <= 2 else 0.045

    for index, player in enumerate(players):
        row = frame.loc[frame["Name"] == player].iloc[0]
        color = RADAR_COLORS[index % len(RADAR_COLORS)]

        values = [
            float(row[f"PCTL__{metric}"])
            for metric in metrics
        ]

        if not values:
            continue

        trace_name = (
            radar_player_label(frame, player)
            if compact_legend
            else player_label(frame, player)
        )

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metric_labels + [metric_labels[0]],
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=3.2,
                ),
                marker=dict(
                    color=color,
                    size=7,
                    line=dict(width=1),
                ),
                fill="toself" if show_fill else "none",
                fillcolor=hex_to_rgba(color, fill_alpha),
                name=trace_name,
                hovertemplate=(
                    "<b>%{fullData.name}</b>"
                    "<br>%{theta}"
                    "<br>Percentile: %{r:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                angle=90,
                gridcolor="rgba(150,150,150,0.25)",
                linecolor="rgba(150,150,150,0.30)",
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(150,150,150,0.20)",
                linecolor="rgba(150,150,150,0.30)",
                tickfont=dict(size=11),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=620,
        margin=dict(
            l=90,
            r=90,
            t=35,
            b=95,
        ),
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

with st.sidebar:
    st.header("Scouting database")

    selected_position = st.selectbox(
        "Position",
        available_positions,
    )

position_df = all_data[
    all_data["PositionGroup"] == selected_position
].copy()

metrics = get_metrics(position_df)

if not metrics:
    st.error("No numeric scouting metrics were detected for this position.")
    st.stop()

if "Minutes" in position_df.columns:
    minutes = pd.to_numeric(
        position_df["Minutes"],
        errors="coerce",
    )

    if minutes.notna().any():
        with st.sidebar:
            maximum_minutes = int(minutes.max())

            minimum_minutes = st.number_input(
                "Minimum minutes",
                min_value=0,
                max_value=maximum_minutes,
                value=0,
                step=90,
            )

        position_df = position_df[
            minutes.fillna(0) >= minimum_minutes
        ].copy()

position_df = prepare_position_data(
    position_df,
    metrics,
)

our_players = position_df.loc[
    position_df["Our player"],
    "Name",
].tolist()

teams = sorted(
    team
    for team in position_df["Team"].dropna().astype(str).unique()
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

    st.caption("REFERENCE")
    st.write(f"{len(position_df)} players")
    st.write(f"{position_df['Team'].nunique()} clubs")
    st.write(f"{len(metrics)} Wyscout metrics")


# ============================================================
# HEADER
# ============================================================

st.title("⚽ Vysočina Scouting Benchmark")
st.caption("Chance National League • Wyscout positional benchmarking")

a, b, c, d = st.columns(4)
a.metric("Position", selected_position)
b.metric("League players", len(position_df))
c.metric("Our players", len(our_players))
d.metric("Metrics", len(metrics))

st.info(
    "Each position uses only the metrics contained in its own Wyscout export. "
    "Percentiles are calculated only against players in the selected position group."
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

    st.dataframe(
        view[
            ["Name", "Team", "Our player"] + quick_metrics
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Our player": st.column_config.CheckboxColumn("OUR"),
        },
    )

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

    if our_players:
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
                our_players,
                format_func=lambda x: player_label(position_df, x),
                key="compare_our_player",
            )

        opponent_teams = [
            team
            for team in teams
            if team != OWN_TEAM
        ]

        with right:
            opponent_team = st.selectbox(
                "Opponent team",
                opponent_teams,
                key="opponent_team",
            )

            opponent_players = position_df.loc[
                position_df["Team"] == opponent_team,
                "Name",
            ].tolist()

            opponent_pick = st.selectbox(
                "Opponent player",
                opponent_players,
                format_func=lambda x: player_label(position_df, x),
                key="opponent_player",
            )

        compare_players = [
            our_pick,
            opponent_pick,
        ]

    else:
        defaults = our_players[:1]

        defaults += [
            player
            for player in position_df["Name"].tolist()
            if player not in defaults
        ][: max(0, 2 - len(defaults))]

        compare_players = st.multiselect(
            "Select 2–4 players",
            position_df["Name"].tolist(),
            default=defaults,
            max_selections=4,
            format_func=lambda x: player_label(position_df, x),
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

    if len(compare_players) >= 2 and compare_metrics:
        chart_controls = st.columns([1.3, 1])

        with chart_controls[0]:
            radar_view = st.radio(
                "Radar view",
                ["Overlay", "Individual"],
                horizontal=True,
                help=(
                    "Overlay is best for direct 1v1 comparison. "
                    "Individual is clearer when comparing 3–4 players."
                ),
            )

        with chart_controls[1]:
            show_fill = st.toggle(
                "Fill radar areas",
                value=len(compare_players) <= 2,
                help=(
                    "For 3–4 players, turning fill off usually makes the radar easier to read."
                ),
            )

        st.markdown("### Percentile radar")

        if radar_view == "Overlay":
            radar = make_radar(
                position_df,
                compare_players,
                compare_metrics,
                show_fill=show_fill,
            )

            st.plotly_chart(
                radar,
                use_container_width=True,
                theme="streamlit",
            )
        else:
            show_individual_radars(
                position_df,
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
                row = position_df.loc[
                    position_df["Name"] == player
                ].iloc[0]

                raw = pd.to_numeric(
                    pd.Series([row[metric]]),
                    errors="coerce",
                ).iloc[0]

                percentile = row[f"PCTL__{metric}"]
                rank = league_rank(
                    metric,
                    raw,
                    position_df,
                )

                record[player] = (
                    f"{raw:.3f} | "
                    f"{percentile:.0f}p | "
                    f"#{rank}"
                )

            comparison_rows.append(record)

        st.caption(
            "Cell format: raw value | league percentile | league rank"
        )

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Selected-metric summary")

        summary_rows = []

        for player in compare_players:
            row = position_df.loc[
                position_df["Name"] == player
            ].iloc[0]

            selected_percentiles = [
                row[f"PCTL__{metric}"]
                for metric in compare_metrics
            ]

            summary_rows.append(
                {
                    "Player": player,
                    "Team": row["Team"],
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
                    "Our player": row["Our player"],
                    "Overall percentile": score,
                    "Average metric rank": avg_rank,
                }

                for metric in ranking_metrics:
                    record[short_metric_label(metric).replace("<br>", " ")] = (
                        row[f"PCTL__{metric}"]
                    )

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

            for col in metric_display_columns:
                overall_df[col] = overall_df[col].round(1)

            st.caption(
                "Overall percentile = weighted average of the selected league percentiles. "
                "Overall rank is then calculated from that combined score."
            )

            st.dataframe(
                overall_df[
                    [
                        "Overall rank",
                        "Name",
                        "Team",
                        "Our player",
                        "Overall percentile",
                        "Average metric rank",
                    ]
                    + metric_display_columns
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Our player": st.column_config.CheckboxColumn("OUR"),
                    "Overall percentile": st.column_config.ProgressColumn(
                        "Overall percentile",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                },
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
