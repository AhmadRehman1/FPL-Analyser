"""The two live model switches (Fix D: captain_risk_params, Fix F: minutes_bounds_params) reach
every squad builder and every minutes model run, from one source: active_recalibratable_versions().

Before this, only run_ingestion.py and run_walkforward.py passed them; every other caller fell
back to the old captain penalty and no minutes floor (docs/reports/2026-10_live_path_diagnosis.md,
finding 4). These tests fail when a new call site forgets them.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import backtest as bt  # noqa: E402
from fpl_quant import forward_season_sim as fss  # noqa: E402
from fpl_quant import minutes_model as mm  # noqa: E402
from fpl_quant import params  # noqa: E402
from fpl_quant import squad_optimizer as so  # noqa: E402

SEED_DIR = REPO_ROOT / "data" / "recalibration"
CAPTAIN = "captain_risk_params_version"
MINUTES = "minutes_bounds_params_version"
RATE = "rate_shrinkage_params_version"

# (module, function) -> keyword every call must pass explicitly (a None is allowed: it's visible).
REQUIRED = {
    ("squad_optimizer", "run"): [CAPTAIN],
    ("squad_optimizer", "solve"): ["captain_variance_multiplier"],
    ("minutes_model", "run"): [MINUTES],
    ("expected_points", "run"): [RATE],
    ("transfer_planner", "compute_horizon_ep"): [RATE],
    ("projections", "build_projections"): [RATE],
    ("transfer_planner", "run"): [CAPTAIN, RATE],
    ("decision_engine", "recommend_best_move"): [CAPTAIN, RATE],
    ("squad_grade", "grade_squad"): [CAPTAIN],
    ("backtest", "run"): [CAPTAIN, MINUTES, RATE],
    ("backtest", "run_season_simulation"): [CAPTAIN, MINUTES, RATE],
}

# Calls that pass their versions through a ** dict. Each one's dict builder is checked below, or it
# is deliberately different; a new one fails test_every_splat_call_is_reviewed until listed here.
REVIEWED_SPLATS = {
    ("scripts/explain_my_move.py", "decision_engine.recommend_best_move"),  # _param_versions()
    ("scripts/run_scenarios.py", "decision_engine.recommend_best_move"),  # _param_versions()
    ("src/fpl_quant/elite_tracking.py", "decision_engine.recommend_best_move"),  # track_elite._param_versions()
    ("src/fpl_quant/scenario.py", "decision_engine.recommend_best_move"),  # run_scenarios' base_state
    ("src/fpl_quant/decision_engine.py", "decision_engine.recommend_best_move"),  # its own run_kwargs
    ("scripts/run_backtest.py", "backtest.run"),  # run_backtest._param_versions()
    ("scripts/run_walkforward.py", "backtest.run"),  # run_backtest._param_versions()
    ("scripts/run_minutes_model_current_season_sensitivity_arm.py", "backtest.run"),  # same
    ("scripts/run_squad_optimizer_wiring_sensitivity_arm.py", "backtest.run"),  # same
    ("scripts/run_season_simulation.py", "backtest.run_season_simulation"),  # _param_versions()
    ("scripts/export_leaderboard.py", "backtest.run_season_simulation"),  # _param_versions()
    ("scripts/run_chip_timing_sensitivity_arm.py", "backtest.run_season_simulation"),  # run_season_simulation's
    ("src/fpl_quant/backtest.py", "backtest.run_season_simulation"),  # sensitivity sweep's base_versions
    # Deliberately "blind": every version at its pre-recalibration default.
    ("scripts/run_retrospective_engine_simulation.py", "backtest.run_season_simulation"),
    ("src/fpl_quant/squad_optimizer.py", "squad_optimizer.solve"),  # run()'s own solve_kwargs
    ("scripts/export_projections.py", "projections.build_projections"),  # its param_versions dict
    ("scripts/grade_squad.py", "transfer_planner.compute_horizon_ep"),  # grade_squad._param_versions()
    ("scripts/print_chip_timing_roadmap.py", "transfer_planner.compute_horizon_ep"),  # its PARAM_VERSIONS
}


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    modules = {m for m, _ in REQUIRED}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module is None or node.module.startswith("fpl_quant")):
            for alias in node.names:
                if alias.name in modules:
                    out[alias.asname or alias.name] = alias.name
    return out


def _required_calls():
    """Yields (relative path, "module.function", call node) for every call to a REQUIRED function."""
    files = sorted((REPO_ROOT / "src" / "fpl_quant").glob("*.py")) + sorted((REPO_ROOT / "scripts").glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text())
        aliases = _module_aliases(tree)
        own_module = path.stem
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in aliases:
                key = (aliases[func.value.id], func.attr)
            elif isinstance(func, ast.Name):
                key = (own_module, func.id)
            else:
                continue
            if key in REQUIRED:
                yield str(path.relative_to(REPO_ROOT)), f"{key[0]}.{key[1]}", node


def test_every_direct_call_passes_the_live_switches():
    missing = []
    for path, name, node in _required_calls():
        keywords = {k.arg for k in node.keywords}
        if None in keywords:  # a ** dict: covered by the splat tests below
            continue
        module, function = name.split(".")
        for kwarg in REQUIRED[(module, function)]:
            if kwarg not in keywords:
                missing.append(f"{path}:{node.lineno} {name}() without {kwarg}=")
    assert not missing, "\n".join(missing)


def test_every_splat_call_is_reviewed():
    unreviewed = sorted({
        (path, name) for path, name, node in _required_calls()
        if None in {k.arg for k in node.keywords} and (path, name) not in REVIEWED_SPLATS
    })
    assert not unreviewed, f"new ** calls -- check their dict carries the live switches, then list them: {unreviewed}"


@pytest.mark.parametrize("script, keys", [
    ("run_backtest", [CAPTAIN, MINUTES, RATE]),
    ("run_season_simulation", [CAPTAIN, MINUTES, RATE]),
    ("export_leaderboard", [CAPTAIN, MINUTES, RATE]),
    ("explain_my_move", [CAPTAIN, RATE]),
    ("run_scenarios", [CAPTAIN, RATE]),
    ("track_elite", [CAPTAIN, RATE]),
    ("grade_squad", [RATE]),
])
def test_script_param_dicts_carry_the_live_switches(script, keys):
    module = __import__(script)
    active = bt.active_recalibratable_versions(SEED_DIR)
    versions = module._param_versions(active)
    for key in keys:
        assert versions[key] == active[key]


def test_forward_sim_resolves_the_live_switches():
    active = bt.active_recalibratable_versions(SEED_DIR)
    versions = fss._resolve_versions(None, active)
    assert versions[CAPTAIN] == active[CAPTAIN]
    assert versions[MINUTES] == active[MINUTES]
    assert versions[RATE] == active[RATE]


def test_active_versions_resolve_to_the_live_values(con):
    """Materialize the committed seeds the way run_ingestion.py does, then resolve."""
    so.seed_v1_params(con)
    mm.seed_minutes_bounds_params(con)
    for seed in bt.load_confirmed_recalibration_seeds(SEED_DIR):
        params.write_param(
            con, seed["param_family"], seed["new_params_version"], "2026-08-12",
            seed["param_key"], value_numeric=seed["new_value"], dimensions=seed["dimensions"],
        )
    active = bt.active_recalibratable_versions(SEED_DIR)
    captain, _ = params.resolve_param(con, "captain_risk_params", "captain_variance_multiplier", active[CAPTAIN])
    floor, _ = params.resolve_param(con, "minutes_bounds_params", "p_floor", active[MINUTES])
    assert captain == so.LIVE_CAPTAIN_VARIANCE_MULTIPLIER == 0.0
    assert floor == mm.PLACEHOLDER_MINUTES_P_FLOOR


def test_live_switch_seed_file_is_confirmed():
    payload = json.loads((SEED_DIR / "seeds_live_switches_2026-09-29.json").read_text())
    (captain,) = payload["proposals"]
    assert (captain["param_family"], captain["status"], captain["new_value"]) == ("captain_risk_params", "confirmed", 0.0)
