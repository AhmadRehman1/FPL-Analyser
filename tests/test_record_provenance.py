import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from record_provenance import (  # noqa: E402
    PACKAGE_NAMES,
    _git_commit_sha,
    _package_versions,
    _workbook_manifest,
    record_provenance,
)


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "README.md").write_text("test repo")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=path, check=True)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_git_commit_sha_returns_none_without_git_dir(tmp_path):
    assert _git_commit_sha(tmp_path) is None


def test_git_commit_sha_returns_real_sha(tmp_path):
    expected = _init_git_repo(tmp_path)
    assert _git_commit_sha(tmp_path) == expected
    assert len(expected) == 40


def test_workbook_manifest_returns_none_when_missing(tmp_path):
    assert _workbook_manifest(tmp_path / "nope.xlsx") is None


def test_workbook_manifest_returns_sha256_and_size(tmp_path):
    p = tmp_path / "evidence.xlsx"
    p.write_bytes(b"some fake workbook bytes")
    manifest = _workbook_manifest(p)
    assert manifest["sha256"] == hashlib.sha256(b"some fake workbook bytes").hexdigest()
    assert manifest["size_bytes"] == len(b"some fake workbook bytes")


def test_package_versions_covers_every_pinned_package():
    versions = _package_versions()
    assert set(versions) == set(PACKAGE_NAMES)
    # Every one of these is a real hard dependency of the installed test environment (see
    # requirements.lock) -- none should resolve to None here.
    for name, version in versions.items():
        assert version is not None, f"{name} not found in the installed environment"


def test_record_provenance_warns_when_fpl_core_insights_missing(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    workbook = tmp_path / "wb.xlsx"
    workbook.write_bytes(b"data")

    manifest = record_provenance(
        date(2026, 8, 23),
        repo_root=repo_root,
        fpl_core_insights_dir=tmp_path / "does-not-exist",
        evidence_workbooks=[workbook],
    )

    out = capsys.readouterr().out
    assert "::warning::record_provenance" in out
    assert "FPL-Core-Insights clone not found" in out
    assert manifest["fpl_core_insights_commit"] is None
    assert manifest["repo_commit"] is not None


def test_record_provenance_warns_when_workbook_missing(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)
    fci_dir = tmp_path / "fci"
    fci_dir.mkdir()
    _init_git_repo(fci_dir)

    manifest = record_provenance(
        date(2026, 8, 23),
        repo_root=repo_root,
        fpl_core_insights_dir=fci_dir,
        evidence_workbooks=[tmp_path / "missing.xlsx"],
    )

    out = capsys.readouterr().out
    assert "::warning::record_provenance" in out
    assert "evidence workbook missing" in out
    assert manifest["evidence_workbooks"] == {}


def test_record_provenance_warns_when_repo_commit_unavailable(tmp_path, capsys):
    manifest = record_provenance(
        date(2026, 8, 23),
        repo_root=tmp_path,  # no .git here
        fpl_core_insights_dir=tmp_path,
        evidence_workbooks=[],
    )
    out = capsys.readouterr().out
    assert "::warning::record_provenance: could not determine this repo's own commit SHA" in out
    assert manifest["repo_commit"] is None


def test_record_provenance_happy_path_no_warnings(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo_sha = _init_git_repo(repo_root)
    fci_dir = tmp_path / "fci"
    fci_dir.mkdir()
    fci_sha = _init_git_repo(fci_dir)
    workbook = tmp_path / "evidence.xlsx"
    workbook.write_bytes(b"real workbook bytes")

    manifest = record_provenance(
        date(2026, 8, 23),
        repo_root=repo_root,
        fpl_core_insights_dir=fci_dir,
        evidence_workbooks=[workbook],
    )

    assert capsys.readouterr().out == ""
    assert manifest["repo_commit"] == repo_sha
    assert manifest["fpl_core_insights_commit"] == fci_sha
    assert manifest["data_asof"] == "2026-08-23"
    assert manifest["evidence_workbooks"]["evidence.xlsx"]["sha256"] == hashlib.sha256(b"real workbook bytes").hexdigest()
    assert set(manifest["package_versions"]) == set(PACKAGE_NAMES)
