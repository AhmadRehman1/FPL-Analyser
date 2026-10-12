"""scripts/run_season_sim_batch.py: season_sim_arms.yml's same_runner mode
(docs/plans/2026-10_beat_the_template.md, Req 7)."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_season_sim_batch as batch  # noqa: E402


def _arm(label, flags=""):
    return {"label": label, "flags": flags, "season": "2025-2026", "start": 2, "end": 38,
            "tag": f"{label}_2025-2026_gw2-38"}


def test_runs_every_arm_in_order_as_its_own_process(tmp_path):
    calls = []

    def run(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    arms = [_arm("control"), _arm("control_b"), _arm("arm", "--lambda 0.05 --chip-wait")]
    assert batch.main([json.dumps(arms)], run=run, out_dir=tmp_path) == 0
    assert [c[c.index("--label") + 1] for c in calls] == ["control", "control_b", "arm"]
    assert calls[2][-3:] == ["--lambda", "0.05", "--chip-wait"]
    assert calls[0][calls[0].index("--out") + 1] == str(tmp_path / "control_2025-2026_gw2-38.json")
    assert calls[0][calls[0].index("--start-gw") + 1] == "2" and calls[0][calls[0].index("--end-gw") + 1] == "38"


def test_a_failed_arm_does_not_stop_the_rest_but_fails_the_job(tmp_path):
    seen = []

    def run(cmd):
        label = cmd[cmd.index("--label") + 1]
        seen.append(label)
        return subprocess.CompletedProcess(cmd, 1 if label == "control_b" else 0)

    assert batch.main([json.dumps([_arm("control"), _arm("control_b"), _arm("arm")])], run=run, out_dir=tmp_path) == 1
    assert seen == ["control", "control_b", "arm"]


def test_refuses_more_than_five_arms_or_none(tmp_path):
    def run(cmd):  # pragma: no cover - never called
        raise AssertionError("should not run")

    assert batch.main([json.dumps([_arm(f"a{i}") for i in range(6)])], run=run, out_dir=tmp_path) == 2
    assert batch.main(["[]"], run=run, out_dir=tmp_path) == 2
    assert batch.main([], run=run, out_dir=tmp_path) == 2


def test_workflow_offers_the_same_runner_mode():
    wf = (REPO_ROOT / ".github" / "workflows" / "season_sim_arms.yml").read_text()
    assert "same_runner:" in wf and "run_season_sim_batch.py" in wf
    assert "data/season_sim_arms/*.json" in wf
