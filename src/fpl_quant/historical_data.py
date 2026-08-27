from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def source_url(season: str, relative_path: str) -> str:
    return f"{RAW_BASE_URL}/{season}/{relative_path.lstrip('/')}"


def read_source_csv(season: str, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(source_url(season, relative_path))


def fetch_season(season: str, output_dir: str | Path) -> dict[str, Path]:
    """Download player-GW, fixture and team context for one FPL season.

    The function deliberately stores source-derived files locally instead of
    committing them, so the repository remains small and reproducible.
    """
    output = Path(output_dir) / season
    output.mkdir(parents=True, exist_ok=True)

    datasets = {
        "player_gameweeks": "gws/merged_gw.csv",
        "fixtures": "fixtures.csv",
        "teams": "teams.csv",
        "players_raw": "players_raw.csv",
    }
    written: dict[str, Path] = {}
    for name, relative_path in datasets.items():
        frame = read_source_csv(season, relative_path)
        frame["season"] = season
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        written[name] = path
    return written


def fetch_history(seasons: Iterable[str], output_dir: str | Path) -> dict[str, dict[str, Path]]:
    return {season: fetch_season(season, output_dir) for season in seasons}


def load_player_gameweeks(data_dir: str | Path, seasons: Iterable[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = Path(data_dir) / season / "player_gameweeks.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run download_historical_fpl_data.py first.")
        frame = pd.read_parquet(path)
        if "GW" not in frame.columns:
            raise ValueError(f"{path} has no GW column.")
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["kickoff_time", "season", "GW", "element"])
