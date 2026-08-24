"""Source provenance manifest (Phase A5 hardening).

The lost-local-DB incident (see README) meant a real, activated calibration (fact_type_multiplier_
params v8, minutes_model_shrinkage_params v10) could no longer be traced back to anything -- the
DB that held the winning values was gone, and nothing durable recorded which upstream inputs
(FPL-Core-Insights commit, evidence-workbook contents, package versions) that run was even built
from. This script closes that specific gap: it records what a run's inputs actually WERE, written
next to the report snapshots that already get committed every scheduled run, so a future "what
changed" question always has a real answer instead of a guess.

This is a record, not a gate -- it never raises. Recording nothing because an upstream input is
genuinely missing would defeat the point (a missing input is exactly the kind of thing this
manifest exists to surface); every gap is reported via ::warning:: and still written to the
manifest as null/false so the manifest is honest about what it does and doesn't know.

Usage (from repo root):
    PYTHONPATH=src python scripts/record_provenance.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FPL_CORE_INSIGHTS_DIR = REPO_ROOT / "data" / "external" / "FPL-Core-Insights-main"
EVIDENCE_WORKBOOK_PATH = REPO_ROOT / "data" / "external" / "FPL_202627_Master_Evidence_Database.xlsx"
RESEARCH_PULL_WORKBOOK_PATH = REPO_ROOT / "data" / "external" / "FPL_Evidence_Claims_Research_Pull.xlsx"
OUTPUT_DIR = REPO_ROOT / "data" / "report_history"

# distribution (PyPI) name for each import name, matching requirements.lock exactly -- these can
# differ (e.g. the pyscipopt package installs as import name pyscipopt but distribution PySCIPOpt).
TRACKED_PACKAGES = {
    "duckdb": "duckdb",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "openpyxl": "openpyxl",
    "pytest": "pytest",
    "python-dateutil": "python-dateutil",
    "pyscipopt": "PySCIPOpt",
    "requests": "requests",
}


def _git_sha(repo_dir: Path) -> str | None:
    if not repo_dir.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _file_fingerprint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return {"sha256": sha256.hexdigest(), "size_bytes": path.stat().st_size}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for import_name, dist_name in TRACKED_PACKAGES.items():
        try:
            versions[import_name] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            versions[import_name] = None
            print(f"::warning::record_provenance: package {import_name!r} (dist {dist_name!r}) is not installed")
    return versions


def build_manifest(asof: date) -> dict:
    repo_sha = _git_sha(REPO_ROOT)
    if repo_sha is None:
        print("::warning::record_provenance: could not resolve this repo's own git commit SHA")

    core_insights_sha = _git_sha(FPL_CORE_INSIGHTS_DIR)
    if core_insights_sha is None:
        print(
            "::warning::record_provenance: FPL-Core-Insights clone not found at "
            f"{FPL_CORE_INSIGHTS_DIR} (or has no .git dir) -- this run's public-dataset "
            "provenance is unrecorded"
        )

    evidence_workbook = _file_fingerprint(EVIDENCE_WORKBOOK_PATH)
    if evidence_workbook is None:
        print(
            f"::warning::record_provenance: evidence workbook not found at {EVIDENCE_WORKBOOK_PATH} "
            "-- this run's curated-evidence provenance is unrecorded"
        )

    research_pull_workbook = _file_fingerprint(RESEARCH_PULL_WORKBOOK_PATH)
    if research_pull_workbook is None:
        print(
            f"::warning::record_provenance: research-pull workbook not found at "
            f"{RESEARCH_PULL_WORKBOOK_PATH} -- this run's research-pull provenance is unrecorded "
            "(this input is optional to run_ingestion.py itself, but still worth flagging here)"
        )

    return {
        "data_asof": asof.isoformat(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fpl_analyser_commit_sha": repo_sha,
        "fpl_core_insights_commit_sha": core_insights_sha,
        "evidence_workbook": evidence_workbook,
        "research_pull_workbook": research_pull_workbook,
        "installed_packages": _package_versions(),
    }


def main() -> None:
    asof = date.today()
    manifest = build_manifest(asof)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"provenance_{asof.isoformat()}.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[provenance] wrote {out_path}")


if __name__ == "__main__":
    main()
