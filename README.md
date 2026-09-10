# Vysočina Scouting Benchmark v6

## New in v6
- New Wyscout exports with `Minutes played`.
- Global minute filter from 1 minute to the maximum for each position.
- Exact one-minute slider.
- Quick selector at 90-minute intervals.
- Overview, Our player vs league and League ranking use the filtered reference sample.
- Percentiles and overall ranking are recalculated after every minute-filter change.
- Comparison keeps ALL players selectable regardless of the minute cutoff.
- Comparison percentiles/ranks are still benchmarked against the filtered reference sample.
- Warning when a manually compared player is below the current minute threshold.
- GK preferred radar now includes passing accuracy and long-pass accuracy.
- Existing multi-metric overall ranking, weights, raw/percentile switch and radar views are retained.

## Updating the live Streamlit app
Replace:
- `app.py`
- the whole `data/` folder

Keep `requirements.txt` (or replace it with the included version).

Commit the changes to the GitHub branch used by Streamlit Community Cloud.
The app should redeploy automatically.
