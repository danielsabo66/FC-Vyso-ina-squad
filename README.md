# Vysočina Scouting Benchmark

This project loads position-specific Wyscout Excel exports and benchmarks players
against the same positional group in the Chance National League.

## Included position groups

- GK
- LB
- RB
- CB
- CM/CDM
- CAM
- RW/LW
- ST

Each position keeps its own metrics exactly as supplied in its Excel export.

## Project structure

app.py
requirements.txt
data/
    GK Chance National League.xlsx
    LB Chance National League.xlsx
    RB Chance National League.xlsx
    CB Chance National League.xlsx
    CM_CDM Chance National League.xlsx
    CAM Chance National League.xlsx
    RW_LW Chance National League.xlsx
    ST Chance National League.xlsx

## Local run

Install Python 3.10+.

Then:

    pip install -r requirements.txt
    streamlit run app.py

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload app.py, requirements.txt, README.md and the complete data/ folder.
3. In Streamlit Community Cloud, create a new app from the GitHub repository.
4. Set the main file path to app.py.
5. Deploy.

## Updating data

Replace the relevant Excel file in data/ with a newer Wyscout export and push the
change to GitHub. Keep a recognizable position in the filename.

Examples:
- GK ...
- LB ...
- RB ...
- CB ...
- CM_CDM ...
- CAM ...
- RW_LW ...
- ST ...

## Notes

- Players from Vysočina Jihlava are detected automatically using the Team column.
- Percentiles are calculated separately inside the selected position group.
- The radar can compare 2–4 players from the same position group.
- Clearly inverse metrics such as conceded goals and fouls are direction-adjusted.
- Volume/context metrics should be interpreted as player-profile information,
  not automatically as better/worse.
