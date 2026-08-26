# Historical FPL data and walk-forward testing

## Install

Use the repository environment, then ensure Parquet support is present:

```bash
pip install -e .
pip install pyarrow
```

## Download the last two completed seasons

```bash
python scripts/download_historical_fpl_data.py --seasons 2023-24 2024-25
```

This downloads source data from `vaastav/Fantasy-Premier-League` into `data/historical/<season>/`:

- `player_gameweeks.parquet`: player-level outcomes and gameweek statistics
- `fixtures.parquet`: fixtures and their FPL metadata, including difficulty ratings when supplied by source
- `teams.parquet`: team metadata and strengths
- `players_raw.parquet`: player reference information

The raw dataset is intentionally ignored by Git. Re-run the downloader to reproduce it locally.

## Run local walk-forward search

```bash
python scripts/run_walk_forward.py --trials 100000 --workers 0
```

`--workers 0` uses all detected local logical CPU cores. Set a lower integer if you need your machine to remain responsive, e.g. `--workers 6`.

The runner uses only features shifted by one player appearance before rolling, then evaluates each chronological weekly test fold after fitting only to earlier weeks. This prevents future gameweek information leaking into feature values or fitting.

Results are written to `data/outputs/`:

- `walk_forward_trials.parquet` and `.csv`: metric per configuration
- `walk_forward_best.json`: lowest-MSE setting

## Scaling responsibly

A full 100,000-trial run repeats equivalent parameter combinations after the finite search space is exhausted. Start with 2,000–10,000 trials to establish whether model choices matter; then use additional seeds, held-out final-season testing, or a larger feature/model search space rather than treating repeated trials as independent evidence.

For long local jobs, use `tmux`, `screen`, or a system service. Keep model runs local or on paid/controlled compute; do not use GitHub Actions as a free high-volume training farm. The included workflow refreshes data only.
