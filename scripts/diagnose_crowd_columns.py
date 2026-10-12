"""Checks the plan's A2 and A3 (docs/plans/2026-10_beat_the_template.md) on the cached DB and the
FPL-Core-Insights git history. Prints JSON.

A3 -- are the crowd columns populated? Per season in fact_player_season_stats: rows, non-null
counts and the gameweeks covered for status, selected_by_percent, transfers_in_event,
transfers_out_event and now_cost, and the status codes seen.

A2 -- is row gw = G-1 knowable before G's deadline? The live DB is rebuilt each run from a
depth-1 clone, so its raw tables hold one snapshot and carry no history. The source repo's own
history does: for each finished gameweek G of a season, the last commit to that season's
playerstats.csv before G's deadline is what a pipeline run at that moment ingested. Its gw = G-1
rows are compared with HEAD's gw = G-1 rows, which are what the DB's fact_player_season_stats
row G-1 holds now (HEAD can be newer than the cached DB; a later edit to row G-1 can only lower
the match, so that errs on the safe side). A column "matches" for a player when both values are equal (numbers within
1e-9, both missing counts as equal). The deadline is the gameweek's earliest kickoff minus 90
minutes.

The declared rule (on 2026-27, the live season): A2 holds if at least 2 finished gameweeks have a
commit before their deadline and, in each, status, selected_by_percent, transfers_in_event and
transfers_out_event all match for >= 95% of players. Otherwise it is inconclusive and the levers
read row G-2. The historical seasons are reported the same way, for information.

Usage (from repo root, with FPL-Core-Insights cloned with history, e.g. --filter=blob:none):
    PYTHONPATH=src python scripts/diagnose_crowd_columns.py --source-repo PATH [--seasons 2026-2027,...]
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest, db  # noqa: E402

COLUMNS = ("status", "selected_by_percent", "transfers_in_event", "transfers_out_event", "now_cost")
JUDGED = ("status", "selected_by_percent", "transfers_in_event", "transfers_out_event")
MATCH_SHARE = 0.95
MIN_GAMEWEEKS = 2
LIVE_SEASON = "2026-2027"
DEADLINE_BEFORE_KICKOFF = timedelta(minutes=90)


def coverage(con) -> dict:
    """A3: per season, non-null counts and gameweeks covered for each crowd column."""
    out: dict = {}
    for (season,) in con.execute("SELECT DISTINCT season FROM fact_player_season_stats ORDER BY season").fetchall():
        n, n_gw = con.execute(
            "SELECT count(*), count(DISTINCT gw) FROM fact_player_season_stats WHERE season = ?", [season],
        ).fetchone()
        cols = {}
        for col in COLUMNS:
            non_null, gws = con.execute(
                f"SELECT count({col}), count(DISTINCT CASE WHEN {col} IS NOT NULL THEN gw END) "
                "FROM fact_player_season_stats WHERE season = ?", [season],
            ).fetchone()
            cols[col] = {"non_null": non_null, "share": round(non_null / n, 4) if n else None, "gameweeks": gws}
        codes = dict(con.execute(
            "SELECT coalesce(status, '(null)'), count(*) FROM fact_player_season_stats WHERE season = ? GROUP BY 1 ORDER BY 1",
            [season],
        ).fetchall())
        out[season] = {"rows": n, "gameweeks": n_gw, "columns": cols, "status_codes": codes}
    return out


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _playerstats_path(repo: Path, season: str) -> str | None:
    for rel in (f"data/{season}/playerstats.csv", f"data/{season}/playerstats/playerstats.csv"):
        if _git(repo, "ls-tree", "--name-only", "HEAD", rel).strip():
            return rel
    return None


def _rows_by_gw(text: str) -> dict[int, dict[str, dict]]:
    """{gw: {player id: {column: value}}} from one playerstats.csv snapshot."""
    out: dict[int, dict[str, dict]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            row_gw = int(float(row.get("gw") or "nan"))
        except ValueError:
            continue
        if row.get("id"):
            out.setdefault(row_gw, {})[row["id"]] = {col: row.get(col) for col in COLUMNS}
    return out


def _same(a: str | None, b: str | None) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except ValueError:
        return False


def compare_snapshots(before: dict[str, dict], final: dict[str, dict]) -> dict:
    """Share of players (in the final row) whose value in the pre-deadline snapshot matches."""
    players = sorted(final)
    if not players:
        return {"n_players": 0, "n_in_snapshot": 0, "match_share": {}}
    shares = {}
    for col in COLUMNS:
        shares[col] = round(sum(
            1 for p in players if p in before and _same(before[p].get(col), final[p].get(col))
        ) / len(players), 4)
    return {"n_players": len(players), "n_in_snapshot": sum(1 for p in players if p in before), "match_share": shares}


def knowability(con, repo: Path, season: str) -> dict:
    """A2 for one season (see module docstring)."""
    rel = _playerstats_path(repo, season)
    if rel is None:
        return {"error": f"no playerstats.csv for {season} at HEAD"}
    head = _rows_by_gw(_git(repo, "show", f"HEAD:{rel}"))
    finished = [gw for (gw,) in con.execute(
        "SELECT gameweek FROM fact_match WHERE season = ? AND competition = ? GROUP BY gameweek "
        "HAVING bool_and(coalesce(finished, FALSE)) ORDER BY gameweek",
        [season, backtest.PL],
    ).fetchall()]
    gameweeks = []
    for gw in finished:
        if gw < 2:
            continue
        kickoff = backtest.gameweek_deadline(con, season, gw)
        if kickoff is None:
            continue
        deadline = kickoff - DEADLINE_BEFORE_KICKOFF
        log = _git(repo, "log", "-1", "--format=%H %cI", f"--before={deadline.isoformat()}", "--", rel).split()
        entry = {"gw": gw, "deadline": deadline.isoformat()}
        if not log:
            gameweeks.append({**entry, "commit": None})
            continue
        sha, when = log
        before = _rows_by_gw(_git(repo, "show", f"{sha}:{rel}")).get(gw - 1, {})
        gameweeks.append({**entry, "commit": sha[:12], "commit_time": when,
                          **compare_snapshots(before, head.get(gw - 1, {}))})
    judged = [g for g in gameweeks if g.get("commit") and g.get("n_players")]
    holds = len(judged) >= MIN_GAMEWEEKS and all(
        all(g["match_share"].get(col, 0) >= MATCH_SHARE for col in JUDGED) for g in judged
    )
    return {
        "path": rel, "gameweeks": gameweeks, "n_judged": len(judged),
        "verdict": "holds" if holds else "inconclusive",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-repo", required=True, type=Path, help="FPL-Core-Insights clone with history")
    parser.add_argument("--seasons", default="2024-2025,2025-2026,2026-2027")
    args = parser.parse_args(argv)
    con = db.connect(read_only=True)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    out = {
        "db_cache_key": os.environ.get("DB_CACHE_KEY") or None,
        "source_head": _git(args.source_repo, "rev-parse", "HEAD").strip(),
        "a3_coverage": coverage(con),
        "a2_knowability": {season: knowability(con, args.source_repo, season) for season in seasons},
    }
    live = out["a2_knowability"].get(LIVE_SEASON) or {}
    out["a2_verdict"] = {
        "rule": f"{LIVE_SEASON}: >= {MIN_GAMEWEEKS} finished gameweeks with a pre-deadline commit, each with "
                f"{', '.join(JUDGED)} matching for >= {MATCH_SHARE:.0%} of players",
        "verdict": live.get("verdict", "inconclusive"),
        "feature_row": "G-1" if live.get("verdict") == "holds" else "G-2",
    }
    print(json.dumps(out, indent=2, default=str))
    con.close()


if __name__ == "__main__":
    main()
