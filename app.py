from pathlib import Path
import re

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


# ============================================================
# METRIC INTERPRETATION
# ============================================================

# Only clearly inverse metrics are reversed.
# All other metrics keep the natural Wyscout direction:
# higher raw value -> higher percentile.
LOWER_IS_MORE_FAVORABLE = {
    "Conceded goals",
    "Conceded goals per 90",
    "Fouls per 90",
}

# Metrics where a higher value is often contextual/profile-based rather than
# automatically "better". We still calculate the raw percentile, but label it.
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


# ============================================================
# DATA LOADING
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
            warnings.append(f"{file.name}: chybí sloupec Player/Name.")
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

        converted = pd.to_numeric(frame[col], errors="coerce")

        if converted.notna().sum() >= 2:
            metrics.append(col)

    return metrics


# ============================================================
# PERCENTILES
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
    p = raw_percentile(value, frame[metric])

    if pd.isna(p):
        return np.nan

    if metric in LOWER_IS_MORE_FAVORABLE:
        return 100.0 - p

    return p


def league_rank(metric: str, value, frame: pd.DataFrame):
    series = pd.to_numeric(frame[metric], errors="coerce")
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(value):
        return np.nan

    if metric in LOWER_IS_MORE_FAVORABLE:
        return int((series < value).sum() + 1)

    return int((series > value).sum() + 1)


def prepare_position_data(frame: pd.DataFrame, metrics: list[str]):
    out = frame.copy()

    for metric in metrics:
        out[f"PCTL__{metric}"] = out[metric].apply(
            lambda x, m=metric: display_percentile(m, x, out)
        )

    out["Our player"] = out["Team"].astype(str).eq(OWN_TEAM)
    return out


# ============================================================
# HELPERS
# ============================================================

def player_label(frame, player_name):
    row = frame.loc[frame["Name"] == player_name].iloc[0]
    team = str(row.get("Team", "") or "").strip()
    our_tag = " • OUR" if team == OWN_TEAM else ""

    return f"{player_name} — {team or 'Unknown team'}{our_tag}"


def metric_note(metric):
    if metric in LOWER_IS_MORE_FAVORABLE:
        return "Lower raw value = higher percentile"

    if metric in CONTEXT_METRICS:
        return "Profile / volume metric"

    return "Higher raw value = higher percentile"


def make_radar(frame, players, metrics):
    fig = go.Figure()

    for player in players:
        row = frame.loc[frame["Name"] == player].iloc[0]

        values = [
            float(row[f"PCTL__{metric}"])
            for metric in metrics
        ]

        if not values:
            continue

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=player_label(frame, player),
                opacity=0.48,
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
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            xanchor="center",
            x=0.5,
        ),
        height=670,
        margin=dict(
            l=70,
            r=70,
            t=40,
            b=115,
        ),
    )

    return fig


def build_player_table(frame, player, metrics):
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
# LOAD
# ============================================================

all_data, load_warnings = load_all_data()

if all_data.empty:
    st.error(
        "Ve složce data/ nebyly nalezeny žádné použitelné Wyscout Excel soubory."
    )
    st.stop()

available_positions = [
    position
    for position in POSITION_ORDER
    if position in all_data["PositionGroup"].unique()
]

other_positions = sorted(
    set(all_data["PositionGroup"].unique()) - set(available_positions)
)

available_positions += other_positions


# ============================================================
# SIDEBAR
# ============================================================

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
    st.error("Pro tuto pozici nebyly nalezeny numerické scoutingové metriky.")
    st.stop()


# Optional minutes filter if future exports contain minutes.
if "Minutes" in position_df.columns:
    minutes = pd.to_numeric(position_df["Minutes"], errors="coerce")

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
    st.write(f"{len(metrics)} selected Wyscout metrics")


# ============================================================
# HEADER
# ============================================================

st.title("⚽ Vysočina Scouting Benchmark")

st.caption(
    "Chance National League • Wyscout positional benchmarking"
)

a, b, c, d = st.columns(4)

a.metric("Position", selected_position)
b.metric("League players", len(position_df))
c.metric("Our players", len(our_players))
d.metric("Metrics", len(metrics))

st.info(
    "Každá pozice používá pouze metriky obsažené v jejím Wyscout exportu. "
    "Percentil se vždy počítá pouze proti hráčům stejné poziční skupiny v tomto datasetu. "
    "U jasně inverzních metrik (např. inkasované góly nebo fauly) je škála otočena."
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

    base_cols = [
        "Name",
        "Team",
        "Our player",
    ]

    quick_metrics = metrics[: min(8, len(metrics))]

    st.dataframe(
        view[base_cols + quick_metrics],
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
            f"V exportu pro {selected_position} nebyl nalezen žádný hráč "
            f"s Team = {OWN_TEAM}."
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
            default=metrics[: min(8, len(metrics))],
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
            )

            st.plotly_chart(
                radar,
                use_container_width=True,
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
        default=metrics[: min(8, len(metrics))],
        key="comparison_metrics",
    )

    if len(compare_players) >= 2 and compare_metrics:
        st.markdown("### Radar comparison")

        radar = make_radar(
            position_df,
            compare_players,
            compare_metrics,
        )

        st.plotly_chart(
            radar,
            use_container_width=True,
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
