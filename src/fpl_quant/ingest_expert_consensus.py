"""Expert-consensus research file -> validated rows -> evidence claims with real numbers.

Reads data/external/expert_consensus/expert_consensus_GW<n>_<YYYY-MM-DD>.md (the Cowork research
output). Four consensus tables become numeric evidence claims:

  CaptainPicks    -> expert_captain_share   (share of sources captaining him, 0-1)
  TransferTargets -> expert_transfer_net    (net buy score / sources in the file, -1..1)
  Differentials   -> expert_differential    (sources recommending / sources in the file)
  Avoid           -> expert_avoid           (-sources warning off / sources in the file)

All four go in under one source ("Expert consensus (Cowork)") whose confidence is the mean of
the Sources table's weight_1_10 / 10. They then decay by observed_date like every other claim.
They are a check (consensus_check), not a model input, until the disagreement log shows they help.

The six standard evidence tables in the same file are cleaned here and handed to the existing
build_evidence_pull_workbook -> ingest_research_pull path unchanged (see the script).

Validation is strict because the research tool gets names and clubs wrong:
  * every player must match FPL's bootstrap-static list; a two-token name needs the first
    name to match too ("Alessandro Donnarumma" is rejected, not guessed to Gianluigi);
  * a row naming two players ("Robert Sanchez / Martinez") is rejected;
  * when the file's club disagrees with FPL's, FPL wins (logged);
  * messy values are coerced, never the whole row dropped ("95+" -> 95, "72-80" -> 76,
    "n/a" -> blank, "Yes, if fit" -> No), and every coercion is logged.
"""

import json
import re
from datetime import date, datetime

import duckdb

from . import entity_resolution as er
from . import ingest_workbook as iw

CONSENSUS_TABLES = ("CaptainPicks", "TransferTargets", "Differentials", "Avoid", "Sources")
STANDARD_TABLES = ("Injuries", "PredictedXI", "Rotation", "RoleChange", "SetPieces", "PriceWatch")
CONSENSUS_SOURCE_NAME = "Expert consensus (Cowork)"

_FILE_RE = re.compile(r"expert_consensus_GW(\d+)_(\d{4}-\d{2}-\d{2})\.md$")
_PLAYER_COLS = {
    "CaptainPicks": ["player"], "TransferTargets": ["player"], "Differentials": ["player"], "Avoid": ["player"],
    "Injuries": ["player"], "PredictedXI": ["player"], "Rotation": ["player"], "RoleChange": ["player"],
    "PriceWatch": ["player"], "SetPieces": ["primary_taker", "secondary_taker", "deputy_if_primary_absent"],
}
_NUMERIC_COLS = {
    "CaptainPicks": ["n_sources_captaining", "n_sources_total", "share"],
    "TransferTargets": ["price", "n_sources_buy", "n_sources_sell", "net_score"],
    "Differentials": ["ownership_pct", "n_sources"],
    "Avoid": ["n_sources"],
    "Sources": ["weight_1_10"],
    "PredictedXI": ["start_confidence_pct", "confidence_1_10"],
    "Injuries": ["confidence_1_10"],
}


# ------------------------------------------------------------------ file + parsing

def file_meta(path) -> tuple[int, date] | None:
    """(gameweek, file_date) from the filename, or None if it doesn't follow the convention."""
    m = _FILE_RE.search(str(path))
    if not m:
        return None
    return int(m.group(1)), datetime.strptime(m.group(2), "%Y-%m-%d").date()


def usable_for_gameweek(file_date: date, deadline: datetime) -> bool:
    """Point-in-time rule: a file dated on or after the deadline's date is never used for that
    gameweek (we can't tell whether it was written before or after the deadline that day)."""
    return file_date < deadline.date()


def _canon_table_name(head: str) -> str | None:
    key = re.sub(r"^table\s*\d*\s*[-—–:]+\s*", "", head.strip(), flags=re.IGNORECASE)
    key = re.sub(r"[^a-z]", "", key.lower())
    for name in CONSENSUS_TABLES + STANDARD_TABLES:
        if key == name.lower():
            return name
    return None


def parse_markdown(text: str) -> dict[str, list[dict]]:
    """{table_name: [row, ...]} for every '## <Table>' section with a markdown table under it."""
    out: dict[str, list[dict]] = {}
    for sec in re.split(r"\n#{2,3}\s+", "\n" + text):
        head, _, body = sec.partition("\n")
        name = _canon_table_name(head)
        if not name:
            continue
        lines = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("|")]
        if len(lines) < 2:
            continue
        header = [re.sub(r"[\s\-]+", "_", c.strip().lower()) for c in lines[0].strip("|").split("|")]
        for ln in lines[1:]:
            if re.fullmatch(r"[\s|:\-]+", ln):
                continue
            cells = [c.strip().replace("**", "") for c in ln.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            out.setdefault(name, []).append(dict(zip(header, cells)))
    return out


# ------------------------------------------------------------------ coercion

def coerce_number(raw) -> tuple[float | None, str | None]:
    """(value, note). note is set whenever the raw text wasn't a plain number."""
    if raw is None:
        return None, None
    s = str(raw).strip()
    if s == "":
        return None, None
    if s.lower() in ("n/a", "na", "-", "—", "–", "unknown", "?"):
        return None, f"{s!r} -> blank"
    try:
        return float(s), None
    except ValueError:
        pass
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)%?", s)
    if m:
        v = (float(m.group(1)) + float(m.group(2))) / 2
        return v, f"{s!r} -> {v:g} (range midpoint)"
    m = re.fullmatch(r"[<>~≈]?\s*(\d+(?:\.\d+)?)\s*(\+|%)?", s)
    if m:
        v = float(m.group(1))
        return v, f"{s!r} -> {v:g}"
    return None, f"{s!r} -> blank (not a number)"


def coerce_predicted_starter(raw) -> tuple[str | None, str | None]:
    s = (raw or "").strip()
    if s.lower() in ("yes", "no"):
        return s.capitalize(), None
    if not s:
        return None, None
    return "No", f"predicted_starter {s!r} -> No"


def coerce_expected_return(raw) -> tuple[str, str | None]:
    s = (raw or "").strip()
    if not s or s.lower() == "unknown":
        return "unknown", None
    if re.fullmatch(r"\d{4}-\d{2}-(\d{2}|xx)", s):
        return s, None
    return "unknown", f"expected_return {s!r} -> unknown"


# ------------------------------------------------------------------ FPL matching

class FplIndex:
    """Name / club lookup over FPL bootstrap-static."""

    def __init__(self, bootstrap: dict):
        self.teams = {t["id"]: t for t in bootstrap["teams"]}
        self.players = bootstrap["elements"]
        self._club_keys: dict[str, int] = {}
        for t in bootstrap["teams"]:
            for k in (t.get("name"), t.get("short_name")):
                if k:
                    self._club_keys[er.normalize_name(k)] = t["id"]

    def club_id(self, club: str | None) -> int | None:
        if not club:
            return None
        return self._club_keys.get(er.normalize_name(club))

    def match(self, name: str) -> tuple[dict | None, str | None]:
        """(player, reject_reason). Strict: never guesses between players."""
        if not name:
            return None, "empty name"
        if re.search(r"/|&|\band\b", name):
            return None, f"{name!r}: more than one player in one row"
        tokens = er.normalize_name(name).split()
        if not tokens:
            return None, f"{name!r}: empty after normalising"
        hits = []
        for p in self.players:
            first = er.normalize_name(p.get("first_name")).split()
            second = er.normalize_name(p.get("second_name")).split()
            web = er.normalize_name(p.get("web_name"))
            if len(tokens) == 1:
                if tokens[0] == web.replace(" ", "") or (second and tokens[0] == second[-1]) or tokens[0] == web:
                    hits.append(p)
            elif tokens[0] in first and set(tokens[1:]) <= set(first + second):
                hits.append(p)
            elif " ".join(tokens) == web:
                hits.append(p)
        if len(hits) == 1:
            return hits[0], None
        if not hits:
            return None, f"{name!r}: no FPL player matches"
        return None, f"{name!r}: matches {len(hits)} FPL players"

    def full_name(self, p: dict) -> str:
        return f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()

    def club_name(self, p: dict) -> str:
        return self.teams[p["team"]]["name"]


# ------------------------------------------------------------------ validation

def validate(tables: dict[str, list[dict]], fpl: FplIndex) -> tuple[dict[str, list[dict]], list[dict]]:
    """(clean_tables, log). Rows with an unmatched player are dropped (and logged); everything
    else is coerced in place and logged. Clean rows gain `fpl_id` and canonical names/clubs."""
    log: list[dict] = []
    clean: dict[str, list[dict]] = {}
    for table, rows in tables.items():
        for i, row in enumerate(rows, start=1):
            row = dict(row)
            where = f"{table} row {i}"
            ok = True
            for col in _PLAYER_COLS.get(table, []):
                raw = row.get(col, "")
                if not raw:
                    continue
                p, reason = fpl.match(raw)
                if p is None:
                    if col == "player":
                        log.append({"where": where, "action": "rejected", "detail": reason})
                        ok = False
                        break
                    log.append({"where": where, "action": "cleared", "detail": f"{col}: {reason}"})
                    row[col] = ""
                    continue
                row[col] = fpl.full_name(p)
                if col == "player":
                    row["fpl_id"] = p["id"]
                    fpl_club = fpl.club_name(p)
                    file_club = row.get("club")
                    if file_club and fpl.club_id(file_club) != p["team"]:
                        log.append({"where": where, "action": "club_fixed",
                                    "detail": f"{raw}: file says {file_club!r}, FPL says {fpl_club!r} (FPL wins)"})
                    if "club" in row:
                        row["club"] = fpl_club
            if not ok:
                continue
            for col in _NUMERIC_COLS.get(table, []):
                if col in row:
                    val, note = coerce_number(row[col])
                    row[col] = val
                    if note:
                        log.append({"where": where, "action": "coerced", "detail": f"{col}: {note}"})
            if table == "PredictedXI" and "predicted_starter" in row:
                val, note = coerce_predicted_starter(row["predicted_starter"])
                if note:
                    row["notes"] = f"{row.get('notes', '')} [{row['predicted_starter']}]".strip()
                    log.append({"where": where, "action": "coerced", "detail": note})
                row["predicted_starter"] = val
            if table == "Injuries" and "expected_return" in row:
                val, note = coerce_expected_return(row["expected_return"])
                if note:
                    row["notes"] = f"{row.get('notes', '')} [expected_return: {row['expected_return']}]".strip()
                    log.append({"where": where, "action": "coerced", "detail": note})
                row["expected_return"] = val
            if table == "CaptainPicks" and row.get("share") is None:
                n_cap, n_tot = row.get("n_sources_captaining"), row.get("n_sources_total")
                if n_cap is not None and n_tot:
                    row["share"] = n_cap / n_tot
                    log.append({"where": where, "action": "coerced", "detail": f"share derived as {n_cap:g}/{n_tot:g}"})
            clean.setdefault(table, []).append(row)
    return clean, log


def standard_tables_markdown(clean: dict[str, list[dict]]) -> str:
    """The six standard tables, cleaned, in the '## Table N - Name' markdown
    build_evidence_pull_workbook.py already parses."""
    parts = []
    for n, table in enumerate(STANDARD_TABLES, start=1):
        rows = clean.get(table) or []
        if not rows:
            continue
        cols = [c for c in rows[0] if c != "fpl_id"]
        parts.append(f"## Table {n} - {table}\n")
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("|" + "---|" * len(cols))
        for r in rows:
            parts.append("| " + " | ".join("" if r.get(c) is None else f"{r.get(c):g}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols) + " |")
        parts.append("")
    return "\n".join(parts)


# ------------------------------------------------------------------ claims

def consensus_confidence(clean: dict[str, list[dict]]) -> float | None:
    weights = [r["weight_1_10"] for r in clean.get("Sources", []) if r.get("weight_1_10") is not None]
    return sum(weights) / len(weights) / 10.0 if weights else None


def consensus_claims(clean: dict[str, list[dict]]) -> list[dict]:
    """One numeric claim per consensus row: {fpl_id, player, claim_type, numeric, payload, table, row}."""
    n_sources = len(clean.get("Sources", [])) or None
    out = []

    def add(table, i, row, claim_type, numeric, keys):
        if numeric is None:
            return
        out.append({
            "fpl_id": row["fpl_id"], "player": row["player"], "claim_type": claim_type,
            "numeric": float(numeric), "payload": {k: row.get(k) for k in keys}, "table": table, "row": i,
        })

    for i, r in enumerate(clean.get("CaptainPicks", []), start=1):
        add("CaptainPicks", i, r, "expert_captain_share", r.get("share"),
            ["n_sources_captaining", "n_sources_total", "notes"])
    for i, r in enumerate(clean.get("TransferTargets", []), start=1):
        net = r.get("net_score")
        if net is None and r.get("n_sources_buy") is not None and r.get("n_sources_sell") is not None:
            net = r["n_sources_buy"] - r["n_sources_sell"]
        add("TransferTargets", i, r, "expert_transfer_net", None if net is None or not n_sources else net / n_sources,
            ["n_sources_buy", "n_sources_sell", "net_score", "reason_tags", "notes"])
    for i, r in enumerate(clean.get("Differentials", []), start=1):
        n = r.get("n_sources")
        add("Differentials", i, r, "expert_differential", None if n is None or not n_sources else n / n_sources,
            ["ownership_pct", "n_sources", "reason_tags", "notes"])
    for i, r in enumerate(clean.get("Avoid", []), start=1):
        n = r.get("n_sources")
        add("Avoid", i, r, "expert_avoid", None if n is None or not n_sources else -n / n_sources,
            ["n_sources", "reason", "notes"])
    return out


def ensure_consensus_source(con: duckdb.DuckDBPyConnection, reliability: float) -> str:
    source_id = "src_expert_consensus_cowork"
    con.execute(
        "INSERT INTO sources (source_id, source_name, source_type, base_reliability_score, citation_count, "
        "source_notes, last_reviewed_date) VALUES (?, ?, 'specialist', ?, 1, ?, NULL) "
        "ON CONFLICT (source_name) DO NOTHING",
        [source_id, CONSENSUS_SOURCE_NAME, reliability,
         "Aggregated expert consensus from the Cowork research job; confidence = mean source weight"],
    )
    return iw._source_id_for(con, CONSENSUS_SOURCE_NAME)


def ingest_consensus_claims(
    con: duckdb.DuckDBPyConnection, clean: dict[str, list[dict]], file_date: date, ingested_date: datetime,
    *, source_reliability: float = 0.5,
) -> dict:
    """Writes consensus_claims() into evidence_claims. A row whose player isn't in this DB's
    player_alias is skipped (counted), same as every other ingest here."""
    source_id = ensure_consensus_source(con, source_reliability)
    confidence = consensus_confidence(clean)
    ins = skip = 0
    for c in consensus_claims(clean):
        uid = iw._resolve_player(con, c["player"])
        if not uid:
            skip += 1
            continue
        ok = iw._insert_claim(
            con, subject_entity_type="player", subject_entity_id=uid, claim_type=c["claim_type"],
            claim_value=c["payload"], claim_value_numeric=c["numeric"], information_type="OPINION",
            source_id=source_id, source_reliability_score=iw._reliability_for(con, source_id),
            confidence=confidence, observed_date=file_date, ingested_date=ingested_date,
            tab_origin=f"expert_consensus:{c['table']}", row_origin=c["row"],
            raw_text=json.dumps({"player": c["player"], "fpl_id": c["fpl_id"]}),
        )
        ins += 1 if ok else 0
        skip += 0 if ok else 1
    return {"inserted": ins, "skipped": skip}
