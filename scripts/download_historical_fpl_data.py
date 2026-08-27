#!/usr/bin/env python3
"""Download FPL historical player, fixture and team data as Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

from fpl_quant.historical_data import fetch_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25"])
    parser.add_argument("--output-dir", default="data/historical")
    args = parser.parse_args()

    result = fetch_history(args.seasons, Path(args.output_dir))
    for season, files in result.items():
        print(f"{season}:")
        for name, path in files.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
