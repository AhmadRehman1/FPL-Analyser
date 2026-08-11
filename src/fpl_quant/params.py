"""Generic versioned-parameter mechanism (kickoff notes item 2).

One physical table (param_versions) backs every module's versioned parameters --
risk_aversion_params, source_tier_weights, minutes_adjustment_params, model_decay_params,
and whatever future modules need -- distinguished by param_family. Rows are immutable
once written: a new tuning is a new params_version, never an edit.

Resolution is by explicit params_version only (never "latest as of date"), matching M5's
hard-error requirement: a model run pins its param versions in a config snapshot, and an
unpopulated or misrouted lookup must fail loudly, not silently fall back to a default.
"""

import json

import duckdb


class ParamNotFoundError(Exception):
    pass


def _canonical_dimensions(dimensions: dict | None) -> str:
    if not dimensions:
        return "{}"
    return json.dumps(dimensions, sort_keys=True, separators=(",", ":"))


def write_param(
    con: duckdb.DuckDBPyConnection,
    param_family: str,
    param_version: int,
    effective_date: str,
    param_key: str,
    *,
    value_numeric: float | None = None,
    value_text: str | None = None,
    dimensions: dict | None = None,
) -> None:
    """Inserts one immutable parameter row. Raises if (family, version, dimensions, key)
    already exists with a different value -- versions are never edited in place."""
    dims = _canonical_dimensions(dimensions)
    existing = con.execute(
        "SELECT value_numeric, value_text FROM param_versions "
        "WHERE param_family = ? AND param_version = ? AND dimensions = ? AND param_key = ?",
        [param_family, param_version, dims, param_key],
    ).fetchone()
    if existing is not None:
        if existing == (value_numeric, value_text):
            return  # idempotent re-write of the identical value
        raise ValueError(
            f"param_versions is immutable: ({param_family}, v{param_version}, {dims}, {param_key}) "
            f"already has value {existing}, refusing to overwrite with ({value_numeric}, {value_text})"
        )
    con.execute(
        "INSERT INTO param_versions (param_family, param_version, effective_date, dimensions, "
        "param_key, value_numeric, value_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [param_family, param_version, effective_date, dims, param_key, value_numeric, value_text],
    )


def resolve_param(
    con: duckdb.DuckDBPyConnection,
    param_family: str,
    param_key: str,
    params_version: int,
    dimensions: dict | None = None,
):
    """Returns (value_numeric, value_text) for a pinned params_version. Hard error on a
    missing lookup -- never silently returns a default (per M5's explicit requirement)."""
    dims = _canonical_dimensions(dimensions)
    row = con.execute(
        "SELECT value_numeric, value_text FROM param_versions "
        "WHERE param_family = ? AND param_version = ? AND dimensions = ? AND param_key = ?",
        [param_family, params_version, dims, param_key],
    ).fetchone()
    if row is None:
        raise ParamNotFoundError(
            f"no param_versions row for family={param_family!r} version={params_version} "
            f"dimensions={dims} key={param_key!r} -- refusing to fall back to a default"
        )
    return row
