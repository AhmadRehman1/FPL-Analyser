"""The one gate every recalibration proposal passes before it can be confirmed.

Used by both the automated path (backtest.evaluate_and_promote_proposal) and the human path
(scripts/review_recalibration.py --confirm). There is no bypass. See
docs/reports/2026-10_model_status_refresh.md, Finding 5, for the incidents behind each check:
k_minutes 450 -> 900 confirmed on a 0.029% in-sample gain at the edge of its grid, rho_residual
0.0 -> 0.0 confirmed twice by hand, and a hardcoded version colliding with a confirmed one.

Checks, in order (all failures are reported, not just the first):
  1. no-op: the value didn't change;
  2. held-out score required (score metrics only): the proposal must carry a score on
     gameweek-steps its own search never saw;
  3. effect size: the held-out improvement must clear min_relative_improvement (versioned);
  4. collision: the (family, version, key) must not already hold a different value, in the DB
     or in any committed seed file;
  5. grid boundary: a winner on the edge of its searched grid means the grid needs widening.
"""

import json
from pathlib import Path

import duckdb

from . import params as params_mod

# metric_name -> direction for every metric recalibrate() writes. rho_hat is not a score (its
# before/after ARE the old/new values, moment-matched), so it skips the score checks.
METRIC_DIRECTION = {
    "neg_log_likelihood": "lower_is_better",
    "log_score_minutes_mean": "higher_is_better",
    "log_score_minutes_mean_holdout": "higher_is_better",
    "realized_sharpe": "higher_is_better",
    "ep_total_calibration_mae": "lower_is_better",
}
NOT_A_SCORE_METRICS = {"rho_hat"}
# Metrics whose own before/after already come from held-out steps.
HOLDOUT_METRICS = {"log_score_minutes_mean_holdout"}

GATE_FAMILY = "recalibration_gate_params"
# Placeholder, not fitted: 1% is a round "clearly more than noise" floor for these metrics.
PLACEHOLDER_MIN_RELATIVE_IMPROVEMENT = 0.01
CV_FOLDS = 5
CV_EMBARGO_GAMEWEEKS = 2


def seed_gate_params(con: duckdb.DuckDBPyConnection) -> None:
    params_mod.write_param(
        con, GATE_FAMILY, 1, "2026-09-29", "min_relative_improvement",
        value_numeric=PLACEHOLDER_MIN_RELATIVE_IMPROVEMENT,
    )


def resolve_min_relative_improvement(con: duckdb.DuckDBPyConnection, gate_params_version: int | None) -> float:
    if gate_params_version is None:
        return PLACEHOLDER_MIN_RELATIVE_IMPROVEMENT
    value, _ = params_mod.resolve_param(con, GATE_FAMILY, "min_relative_improvement", gate_params_version)
    return value


# ------------------------------------------------------------------
# purged, embargoed K-fold over ordered walk-forward steps
# ------------------------------------------------------------------

def purged_kfold_splits(
    steps: list, k: int = CV_FOLDS, embargo: int = CV_EMBARGO_GAMEWEEKS,
) -> list[tuple[list, list]]:
    """Contiguous folds over the time-ordered steps. Each fold's training set drops the fold
    itself plus `embargo` steps either side of it, so no training step sits next to a test step
    (form and minutes carry over week to week). Returns [(train, test), ...]."""
    steps = list(steps)
    n = len(steps)
    if n == 0:
        return []
    k = max(1, min(k, n))
    bounds = [round(i * n / k) for i in range(k + 1)]
    splits = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        if lo == hi:
            continue
        test = steps[lo:hi]
        train = [s for j, s in enumerate(steps) if j < lo - embargo or j >= hi + embargo]
        splits.append((train, test))
    return splits


def cv_holdout_scores(
    per_step_scores: dict, steps: list, current_value, lower_is_better: bool,
    k: int = CV_FOLDS, embargo: int = CV_EMBARGO_GAMEWEEKS,
) -> dict | None:
    """per_step_scores: {candidate_value: {step: score}}. For each purged fold, pick the best
    candidate on the training steps only, then score it (and the current value) on the fold's
    test steps. Returns the mean test score of the fold-chosen candidates ("after") vs the
    current value ("before") -- an honest estimate of what the search itself is worth.
    None when there is nothing to score."""
    if current_value not in per_step_scores:
        return None

    def mean_over(candidate, subset):
        vals = [per_step_scores[candidate][s] for s in subset if s in per_step_scores[candidate]]
        return sum(vals) / len(vals) if vals else None

    befores, afters, chosen = [], [], []
    for train, test in purged_kfold_splits(steps, k, embargo):
        train_scores = {c: mean_over(c, train) for c in per_step_scores}
        train_scores = {c: v for c, v in train_scores.items() if v is not None}
        if not train_scores:
            continue
        pick = (min if lower_is_better else max)(train_scores, key=train_scores.get)
        before, after = mean_over(current_value, test), mean_over(pick, test)
        if before is None or after is None:
            continue
        befores.append(before)
        afters.append(after)
        chosen.append(pick)
    if not befores:
        return None
    return {
        "holdout_metric_before": sum(befores) / len(befores),
        "holdout_metric_after": sum(afters) / len(afters),
        "fold_choices": chosen,
    }


# ------------------------------------------------------------------
# the gate
# ------------------------------------------------------------------

def _relative_improvement(before: float, after: float, direction: str) -> float:
    gain = (before - after) if direction == "lower_is_better" else (after - before)
    denom = abs(before) if before != 0 else abs(after)
    if denom == 0:
        return 0.0
    return gain / denom


def _seed_file_collisions(seed_dir: Path | str | None, family: str, key: str, dims: str | None,
                          version: int, value: float) -> list[str]:
    if seed_dir is None or not Path(seed_dir).is_dir():
        return []
    out = []
    for path in sorted(Path(seed_dir).glob("seeds_*.json")):
        try:
            proposals = json.loads(path.read_text()).get("proposals") or []
        except (OSError, json.JSONDecodeError):
            continue
        for p in proposals:
            p_dims = json.dumps(p.get("dimensions"), sort_keys=True) if p.get("dimensions") else None
            if (
                p.get("status") == "confirmed" and p.get("param_family") == family and p.get("param_key") == key
                and p_dims == dims and p.get("new_params_version") == version and p.get("new_value") != value
            ):
                out.append(f"{path.name} proposal #{p.get('proposal_id')} already confirmed v{version} = {p.get('new_value')}")
    return out


def gate_reasons(
    con: duckdb.DuckDBPyConnection, proposal_id: int, seed_dir: Path | str | None = None,
    *, min_relative_improvement: float | None = None, gate_params_version: int | None = None,
) -> list[str]:
    """Every reason this proposal must not be confirmed. Empty list = it passes."""
    row = con.execute(
        "SELECT param_family, param_key, dimensions, new_params_version, old_value, new_value, "
        "metric_name, metric_before, metric_after, holdout_metric_before, holdout_metric_after, "
        "grid_min, grid_max FROM recalibration_proposals WHERE proposal_id = ?", [proposal_id],
    ).fetchone()
    if row is None:
        return [f"no recalibration_proposals row for proposal_id={proposal_id}"]
    (family, key, dims, new_version, old_value, new_value, metric_name, metric_before, metric_after,
     hold_before, hold_after, grid_min, grid_max) = row
    if min_relative_improvement is None:
        min_relative_improvement = resolve_min_relative_improvement(con, gate_params_version)
    reasons = []

    # 1. no-op
    if metric_name in NOT_A_SCORE_METRICS:
        if metric_before == metric_after:
            reasons.append(f"{metric_name}: value unchanged ({metric_before}), nothing to promote")
    elif old_value is not None and new_value is not None and old_value == new_value:
        reasons.append(f"{family}.{key}: value unchanged ({new_value}), nothing to promote")

    # 2 + 3. held-out effect size
    if metric_name not in NOT_A_SCORE_METRICS:
        direction = METRIC_DIRECTION.get(metric_name)
        if direction is None:
            reasons.append(f"unrecognized metric_name {metric_name!r} -- refusing to guess a direction")
        else:
            if metric_name in HOLDOUT_METRICS:
                before, after = metric_before, metric_after
            else:
                before, after = hold_before, hold_after
            if before is None or after is None:
                reasons.append(f"{metric_name}: no held-out score -- the in-sample gain is optimistic by construction")
            else:
                rel = _relative_improvement(before, after, direction)
                if rel <= 0:
                    reasons.append(f"{metric_name}: held-out {before} -> {after} is not an improvement ({direction})")
                elif rel < min_relative_improvement:
                    reasons.append(
                        f"{metric_name}: held-out improvement {rel:.4%} below the "
                        f"{min_relative_improvement:.2%} noise floor"
                    )

    # 4. collision
    dims_parsed = json.loads(dims) if dims else None
    existing = con.execute(
        "SELECT value_numeric FROM param_versions WHERE param_family = ? AND param_version = ? "
        "AND dimensions = ? AND param_key = ?",
        [family, new_version, params_mod._canonical_dimensions(dims_parsed), key],
    ).fetchone()
    if existing is not None and new_value is not None and existing[0] != new_value:
        reasons.append(f"collision: {family} v{new_version}.{key} already holds {existing[0]}, proposal says {new_value}")
    clash = con.execute(
        "SELECT proposal_id, new_value FROM recalibration_proposals WHERE status = 'confirmed' AND proposal_id != ? "
        "AND param_family = ? AND param_key = ? AND coalesce(dimensions, '') = coalesce(?, '') "
        "AND new_params_version = ? AND new_value IS DISTINCT FROM ?",
        [proposal_id, family, key, dims, new_version, new_value],
    ).fetchall()
    for other_id, other_value in clash:
        reasons.append(f"collision: proposal #{other_id} already confirmed {family} v{new_version} = {other_value}")
    dims_key = json.dumps(dims_parsed, sort_keys=True) if dims_parsed else None
    for msg in _seed_file_collisions(seed_dir, family, key, dims_key, new_version, new_value):
        reasons.append(f"collision: {msg}")

    # 5. grid boundary
    if grid_min is not None and grid_max is not None and grid_min < grid_max and new_value is not None:
        if new_value in (grid_min, grid_max):
            reasons.append(
                f"{family}.{key}: winner {new_value} is on the edge of the searched grid "
                f"[{grid_min}, {grid_max}] -- widen the grid and rerun"
            )
    return reasons
