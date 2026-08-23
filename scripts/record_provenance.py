"""Priority A5: per-run source provenance manifest.

Records what a scheduled pipeline run's outputs were actually built from: this repo's own
commit SHA, the FPL-Core-Insights commit SHA it cloned, the sha256+size of both evidence
workbooks, and every pinned package's installed version -- closing a real provenance gap
(README: a lost local db/fpl_quant_v2.duckdb erased which xi/rho_residual/
fact_type_multiplier_params/minutes_model_shrinkage_params values a run was actually built
from, with no record anywhere of what upstream state produced them). Written once per run to
data/report_history/provenance_<data_asof>.json.

Never blocks the pipeline and never invents a value: an absent upstream input (no
FPL-Core-Insights clone yet, no evidence workbook materialized) is reported as a named,
GitHub-Actions-visible ::warning:: and the corresponding manifest field is left null/omitted,
not filled with a fabricated stand-in.

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
EVIDENCE_WORKBOOKS = [
    REPO_ROOT / "data" / "external" / "FPL_202627_Master_Evidence_Database.xlsx",
    REPO_ROOT / "data" / "external" / "FPL_Evidence_Claims_Research_Pull.xlsx",
]
# Matches requirements.lock exactly -- the point is to record what's ACTUALLY installed for
# this run, which requirements.lock only pins the intent for.
PACKAGE_NAMES = [
    "duckdb", "pandas", "numpy", "scipy", "openpyxl", "pytest",
    "python-dateutil", "pyscipopt", "requests",
]


def _git_commit_sha(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _workbook_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return {"sha256": sha256.hexdigest(), "size_bytes": path.stat().st_size}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def record_provenance(
    data_asof: date,
    *,
    repo_root: Path = REPO_ROOT,
    fpl_core_insights_dir: Path = FPL_CORE_INSIGHTS_DIR,
    evidence_workbooks: list[Path] = EVIDENCE_WORKBOOKS,
) -> dict:
    """Builds the manifest dict. Pure aside from reading the 3 fragile upstream inputs off
    disk and hashing files -- does not write anything itself (see main() for the write), so
    tests can point repo_root/fpl_core_insights_dir/evidence_workbooks at tmp_path fixtures."""
    repo_commit = _git_commit_sha(repo_root)
    if repo_commit is None:
        print(
            "::warning::record_provenance: could not determine this repo's own commit SHA "
            f"(no .git directory, or `git rev-parse HEAD` failed, at {repo_root}) -- "
            "repo_commit will be null in the manifest."
        )

    fpl_core_insights_commit = _git_commit_sha(fpl_core_insights_dir)
    if fpl_core_insights_commit is None:
        print(
            f"::warning::record_provenance: FPL-Core-Insights clone not found or not a git "
            f"repo at {fpl_core_insights_dir} -- fpl_core_insights_commit will be null in the "
            f"manifest."
        )

    workbooks: dict[str, dict[str, object]] = {}
    for path in evidence_workbooks:
        manifest = _workbook_manifest(path)
        if manifest is None:
            print(
                f"::warning::record_provenance: evidence workbook missing at {path} -- "
                f"omitted from the manifest."
            )
        else:
            workbooks[path.name] = manifest

    return {
        "data_asof": data_asof.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": repo_commit,
        "fpl_core_insights_commit": fpl_core_insights_commit,
        "evidence_workbooks": workbooks,
        "package_versions": _package_versions(),
    }


def main() -> None:
    data_asof = date.today()
    manifest = record_provenance(data_asof)
    out_dir = REPO_ROOT / "data" / "report_history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"provenance_{data_asof.isoformat()}.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[record_provenance] wrote {out_path}")


if __name__ == "__main__":
    main()
