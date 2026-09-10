# Vysočina Scouting Benchmark v7

## New: Club vs Opponent
- Select two clubs.
- Select a formation independently for each club.
- Supported: 4-2-3-1, 4-3-3, 4-4-2, 3-5-2, 3-4-2-1, 3-4-3.
- Best XI is selected from positional overall percentiles.
- A player cannot appear twice in the same XI.
- LWB maps to LB and RWB maps to RB.
- Optional switch allows RW/LW players to be considered as wing-backs.
- Best XI respects a minimum-minutes cutoff.
- Each club is shown on a pitch plus a lineup table.

## Existing features retained
- Position-specific datasets and metrics.
- Global minute filter for league-facing tabs.
- Manual player comparison without excluding low-minute players.
- Radar / bars / individual radar views.
- Multi-metric overall ranking.
- Custom weights.
- Raw values / percentile display switch.

To update the live Streamlit app, replace app.py and commit to GitHub.
The existing v6 data folder can remain unchanged.
