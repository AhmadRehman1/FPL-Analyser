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


def get_or_create_version(
    con: duckdb.DuckDBPyConnection,
    param_family: str,
    param_key: str,
    effective_date: str,
    *,
    value_numeric: float | None = None,
    value_text: str | None = None,
    dimensions: dict | None = None,
) -> int:
    """Returns the params_version that already holds this exact (family, key, dimensions,
    value), minting a fresh one (family's current max + 1) and writing it via write_param()
    if no such row exists yet.

    Use this instead of a hardcoded params_version literal whenever the caller doesn't -- and
    can't -- control what version number a family's OWN organically-growing recalibration
    lineage has already reached. A hardcoded literal is only safe for the version that seeds
    a family for the very first time (v1, written once by that module's own seed_v1_params()
    before anything else ever touches the family); any later hardcoded version number is a
    real collision waiting to happen the moment recalibrate()'s own version-minting (a plain
    max(param_version)+1 over the same family) organically reaches that same integer with a
    different value. write_param()'s immutability check would then raise on the second write.

    This is exactly what happened to risk_posture.py's "attack" posture: it hardcoded
    tc_risk_aversion_params version 2 (value 0.5) months before any real recalibration had
    ever confirmed a version 2 for that family. Once data/recalibration/seeds_1.json later
    confirmed a genuine kappa_tc v1->v2 recalibration (value 0.2) and run_ingestion.py started
    loading confirmed seeds into the SAME live database the attack-posture step also writes
    into (in that same job), every attack-posture solve started raising ValueError on
    resolve_versions()'s own write_param() call -- silently, since the scheduled workflow runs
    that step under continue-on-error. See docs/reports/2026-09_model_failure_diagnosis.md
    section 9 for the incident this fixes.
    """
    dims = _canonical_dimensions(dimensions)
    existing_version = con.execute(
        "SELECT param_version FROM param_versions WHERE param_family = ? AND param_key = ? "
        "AND dimensions = ? AND value_numeric IS NOT DISTINCT FROM ? AND value_text IS NOT DISTINCT FROM ?",
        [param_family, param_key, dims, value_numeric, value_text],
    ).fetchone()
    if existing_version is not None:
        return existing_version[0]
    next_version_row = con.execute(
        "SELECT coalesce(max(param_version), 0) + 1 FROM param_versions WHERE param_family = ?",
        [param_family],
    ).fetchone()
    assert next_version_row is not None  # a bare aggregate always returns exactly one row
    next_version = next_version_row[0]
    write_param(
        con, param_family, next_version, effective_date, param_key,
        value_numeric=value_numeric, value_text=value_text, dimensions=dimensions,
    )
    return next_version


# ============================================================
# M9 adapter -- parameter transparency panel
# ============================================================

def transparency_panel(con: duckdb.DuckDBPyConnection, active_versions: dict[str, int]) -> list[dict]:
    """M9's assumptions/parameter-transparency section: "every versioned parameter active in
    this run, flagged as either backtested/recalibrated via M7 or still literature/invented
    default." active_versions is the caller's explicit statement of which params_version is
    "active" for each family -- there is no "latest" concept anywhere in this project
    (resolve_param() is explicit-version-only by design), so this panel is only ever a report
    on versions the caller actually names, never a guess.

    A family counts as "backtested_via_m7" iff it has at least one row in
    recalibration_proposals, regardless of that proposal's status (pending/confirmed/rejected)
    -- M7 having *attempted* to validate/tune it is the signal this flag reports, not whether
    a human has since accepted the result.
    """
    touched_families = {
        r[0] for r in con.execute("SELECT DISTINCT param_family FROM recalibration_proposals").fetchall()
    }

    panel = []
    for family, version in active_versions.items():
        rows = con.execute(
            "SELECT param_key, dimensions, value_numeric, value_text, effective_date "
            "FROM param_versions WHERE param_family = ? AND param_version = ?",
            [family, version],
        ).fetchall()
        for param_key, dimensions, value_numeric, value_text, effective_date in rows:
            panel.append({
                "param_family": family, "param_version": version, "param_key": param_key,
                "dimensions": json.loads(dimensions) if dimensions and dimensions != "{}" else None,
                "value": value_numeric if value_numeric is not None else value_text,
                "effective_date": effective_date, "backtested_via_m7": family in touched_families,
            })
    return panel
