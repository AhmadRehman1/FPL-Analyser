"""Shared named-exception vocabulary for the user-facing decision-layer features (F1/F3/F4/
F8/F9 and friends). Co-located per-module exceptions (e.g. squad_optimizer.
DivergenceCheckFailedError) stay where they are -- this module exists for exceptions that
cross module boundaries, so every caller matches on the same name rather than each module
inventing its own near-duplicate.
"""


class MissingModelVersionError(Exception):
    """An EP/uncertainty/Monte Carlo output is requested for an asof/gameweek whose
    model_version row doesn't exist (no fixtures, or the calibration was never run) --
    raised instead of silently falling back to a stale or point-estimate-only result."""


class MissingProvenanceError(Exception):
    """A user-facing number can't be tagged with a model_version + data_asof -- raised
    instead of showing an unprovenanced number, per this project's own "every number shown
    to the user carries its provenance" rule."""


class InvalidScenarioError(Exception):
    """A what-if scenario (Feature 4) references an unknown player_uid/team_uid, or an
    impossible perturbation (e.g. a negative price delta larger than the player's own
    price) -- raised by scenario.validate_scenario() before any re-solve is attempted."""
