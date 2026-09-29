"""Validate and ingest one expert-consensus research file (Fix G).

  PYTHONPATH=src python scripts/ingest_expert_consensus.py [file.md] [--validate-only] [--deadline YYYY-MM-DDTHH:MM]

Defaults to the newest data/external/expert_consensus/expert_consensus_GW<n>_<date>.md.

1. Parses the tables and validates every player against FPL bootstrap-static (strict: unknown
   or ambiguous names are rejected, FPL's club wins). Prints every rejection and coercion and
   writes them to data/expert_consensus/validation_GW<n>_<date>.json.
2. With --deadline, refuses a file dated on or after that gameweek's deadline.
3. Unless --validate-only: the six standard evidence tables go through
   build_evidence_pull_workbook.build() -> ingest_research_pull.ingest_all() unchanged, and the
   consensus tables become numeric expert_* claims in evidence_claims.
"""

import json
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fpl_quant import db, ingest_expert_consensus as iec, ingest_research_pull  # noqa: E402

CONSENSUS_DIR = REPO_ROOT / "data" / "external" / "expert_consensus"
VALIDATION_DIR = REPO_ROOT / "data" / "expert_consensus"


def _bootstrap() -> dict:
    req = urllib.request.Request("https://fantasy.premierleague.com/api/bootstrap-static/", headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def main() -> None:
    args = sys.argv[1:]
    validate_only = "--validate-only" in args
    deadline = None
    if "--deadline" in args:
        deadline = datetime.fromisoformat(args[args.index("--deadline") + 1]).replace(tzinfo=timezone.utc)
    paths = [a for a in args if a.endswith(".md")]
    path = Path(paths[0]) if paths else max(CONSENSUS_DIR.glob("expert_consensus_GW*_*.md"), default=None)
    if path is None:
        raise SystemExit(f"no expert_consensus_GW*_*.md in {CONSENSUS_DIR}")
    meta = iec.file_meta(path)
    if meta is None:
        raise SystemExit(f"{path.name} doesn't follow expert_consensus_GW<n>_<YYYY-MM-DD>.md")
    gameweek, file_date = meta
    if deadline is not None and not iec.usable_for_gameweek(file_date, deadline):
        raise SystemExit(f"{path.name} is dated {file_date}, not before the GW{gameweek} deadline {deadline} -- not used")

    tables = iec.parse_markdown(path.read_text(encoding="utf-8"))
    clean, log = iec.validate(tables, iec.FplIndex(_bootstrap()))
    for e in log:
        print(f"[{e['action']}] {e['where']}: {e['detail']}")
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "file": path.name, "gameweek": gameweek, "file_date": file_date.isoformat(),
        "rows_in": {t: len(r) for t, r in tables.items()}, "rows_kept": {t: len(r) for t, r in clean.items()},
        "log": log,
    }
    (VALIDATION_DIR / f"validation_GW{gameweek}_{file_date}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}, indent=2))
    if validate_only:
        return

    import build_evidence_pull_workbook as bepw

    con = db.connect()
    now = datetime.now(timezone.utc)
    md = iec.standard_tables_markdown(clean)
    if md.strip():
        with tempfile.TemporaryDirectory() as tmp:
            md_path, xlsx_path = Path(tmp) / "standard.md", Path(tmp) / "standard.xlsx"
            md_path.write_text(md, encoding="utf-8")
            bepw.build(md_path, xlsx_path, resolve=False)  # names are already FPL-canonical
            print("[standard tables]", ingest_research_pull.ingest_all(con, str(xlsx_path), source_tier_params_version=1))
    print("[consensus claims]", iec.ingest_consensus_claims(con, clean, file_date, now))
    con.close()


if __name__ == "__main__":
    main()
