"""The shared static-site file list covers everything the pages load, on both hosts."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["index.html", "landing.html", "track-record.html"]


def _site_entries():
    lines = (ROOT / "scripts" / "site_files.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def test_every_listed_path_exists():
    for entry in _site_entries():
        assert (ROOT / entry).exists(), entry


def test_local_refs_in_pages_are_deployed():
    entries = _site_entries()
    for page in PAGES:
        html = (ROOT / page).read_text()
        for ref in re.findall(r'(?:src|href)="([^"#:$]+)"', html):
            top = ref.lstrip("./").split("/")[0]
            assert top in entries, f"{page} loads {ref}, not in scripts/site_files.txt"


def test_sw_precache_is_deployed():
    entries = _site_entries()
    sw = (ROOT / "sw.js").read_text()
    for ref in re.findall(r'"\./([^"]+)"', sw):
        assert ref.split("/")[0] in entries, ref


def test_pages_workflow_uses_shared_build():
    wf = (ROOT / ".github" / "workflows" / "deploy_pages.yml").read_text()
    assert "scripts/build_cloudflare_site.sh site" in wf


def test_build_script_outputs_list(tmp_path):
    out = tmp_path / "public"
    subprocess.run(["bash", "scripts/build_cloudflare_site.sh", str(out)], cwd=ROOT, check=True)
    for entry in _site_entries():
        assert (out / entry).exists(), entry
    assert (out / "_headers").exists()
    assert not (out / "data").exists()
