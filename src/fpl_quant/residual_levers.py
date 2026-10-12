"""The residual pipeline behind the residual-fitted levers of docs/plans/2026-10_beat_the_template.md
(L1 crowd prior, L2 band bias, L3 per-player residual).

feature_rows() is the one place crowd features are read, in every harness (walk-forward, season
sim, live), so a feature means the same thing wherever a lever applies it (Req 11). Features
come from row gw = G - lag of fact_player_season_stats -- lag 1 by default, the latest completed
gameweek before G's deadline (the plan's A2 decides whether that row is knowable in time; its
fallback is lag 2).

residual_rows() is what `run_walkforward.py --export-residuals` writes as residual_features.csv
(Req 12): one row per player-gameweek a walk-forward step projected and FPL scored, with the
realized points, the projection before any promoted lever's adjustment, and the features. The
lever fits (scripts/fit_residual_lever.py) read that file. No lever is promoted yet, so the
pre-adjustment projection is ep_total itself; a promoted lever adds its own-season adjustment as
its own column here.
"""

import math

import duckdb

FEATURES = ("own_pct", "log_sbp", "xfer_pct")
PRICE_BANDS = (("<5.0", 5.0), ("5.0-7.0", 7.0), ("7.0-9.0", 9.0), ("9.0+", math.inf))
RESIDUAL_COLUMNS = (
    "season", "gw", "player_uid", "position", "price", "price_band", "realized_points", "ep_pre_adjustment",
    *FEATURES,
)


def price_band(price: float | None) -> str | None:
    if price is None:
        return None
    return next(band for band, upper in PRICE_BANDS if price < upper)


def feature_rows(con: duckdb.DuckDBPyConnection, season: str, gameweek: int, lag: int = 1) -> dict[str, dict]:
    """{player_uid: {"own_pct", "log_sbp", "xfer_pct"}} from row gw = gameweek - lag of `season`.

    - own_pct: the player's ownership percentile within his position (percent_rank, 0 to 1),
      over that row's players with an ownership figure;
    - log_sbp: log(1 + selected_by_percent);
    - xfer_pct: the percentile of transfers_in_event - transfers_out_event among every player in
      that row with both figures, or None when he lacks one.

    A player with no row at gw = gameweek - lag, or no ownership in it, is absent (no feature row
    means no adjustment). Empty for gameweek - lag < 1. Reads fact_player_season_stats by bare
    name, so inside asof_scope() for G it sees rows before G only."""
    source_gw = gameweek - lag
    if source_gw < 1:
        return {}
    rows = con.execute(
        """
        WITH r AS (
            SELECT s.player_uid, dp.position, s.selected_by_percent AS sbp,
                   s.transfers_in_event - s.transfers_out_event AS net_xfer
            FROM fact_player_season_stats s JOIN dim_player dp ON dp.player_uid = s.player_uid
            WHERE s.season = ? AND s.gw = ?
        ),
        x AS (
            -- among every player in the row with both transfer figures, owned or not
            SELECT player_uid, percent_rank() OVER (ORDER BY net_xfer) AS xfer_pct
            FROM r WHERE net_xfer IS NOT NULL
        ),
        owned AS (SELECT * FROM r WHERE sbp IS NOT NULL)
        SELECT owned.player_uid, percent_rank() OVER (PARTITION BY owned.position ORDER BY owned.sbp) AS own_pct,
               ln(1 + greatest(owned.sbp, 0)) AS log_sbp, x.xfer_pct
        FROM owned LEFT JOIN x USING (player_uid)
        """,
        [season, source_gw],
    ).fetchall()
    return {uid: {"own_pct": own, "log_sbp": log_sbp, "xfer_pct": xfer} for uid, own, log_sbp, xfer in rows}


def residual_rows(con: duckdb.DuckDBPyConnection, backtest_run_id: int, lag: int = 1) -> list[dict]:
    """Req 12's rows for one walk-forward run (RESIDUAL_COLUMNS): every player-gameweek with a
    projection at the step (ep_gameweek_outputs, so a double gameweek is summed and a blank one
    drops out) and a realized FPL score. price is the latest before the deadline, as the
    optimizer saw it; features are feature_rows() for that step. Ordered by season, gw, player."""
    steps = con.execute(
        "SELECT season, gameweek, ep_model_version FROM backtest_gameweek_steps "
        "WHERE backtest_run_id = ? AND ep_model_version IS NOT NULL ORDER BY season, gameweek",
        [backtest_run_id],
    ).fetchall()
    out = []
    for season, gw, ep_mv in steps:
        features = feature_rows(con, season, gw, lag=lag)
        rows = con.execute(
            """
            WITH price AS (
                SELECT player_uid, now_cost FROM fact_player_season_stats
                WHERE season = ? AND gw < ? AND now_cost IS NOT NULL
                QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1
            )
            SELECT o.player_uid, dp.position, price.now_cost, f.event_points, o.ep_total
            FROM ep_gameweek_outputs o
            JOIN dim_player dp ON dp.player_uid = o.player_uid
            JOIN fact_player_season_stats f ON f.player_uid = o.player_uid AND f.season = ? AND f.gw = ?
            LEFT JOIN price ON price.player_uid = o.player_uid
            WHERE o.model_version = ? AND f.event_points IS NOT NULL
            ORDER BY o.player_uid
            """,
            [season, gw, season, gw, ep_mv],
        ).fetchall()
        for uid, position, price, realized, ep_total in rows:
            feat = features.get(uid, {})
            out.append({
                "season": season, "gw": gw, "player_uid": uid, "position": position, "price": price,
                "price_band": price_band(price), "realized_points": realized, "ep_pre_adjustment": ep_total,
                **{name: feat.get(name) for name in FEATURES},
            })
    return out


def write_residual_csv(con: duckdb.DuckDBPyConnection, backtest_run_id: int, path, lag: int = 1) -> int:
    """Writes residual_rows() as CSV (empty cells for missing values); returns the row count."""
    import csv

    rows = residual_rows(con, backtest_run_id, lag=lag)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESIDUAL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return len(rows)
