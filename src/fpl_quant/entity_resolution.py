"""Name normalization and deterministic UID derivation for the (normalized_name, team_code,
season) join key the M0 spec prescribes for player/team identity across seasons.
"""

import re
import unicodedata


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def team_uid_for(canonical_name: str) -> str:
    return "team_" + normalize_name(canonical_name).replace(" ", "_")


def player_uid_for(canonical_name: str) -> str:
    # Known v1 limitation: two distinct real players sharing an identical normalized full
    # name would collide onto the same player_uid. Not resolved automatically -- per M1b's
    # own precedent, disambiguation is a manual-curation problem (a distinguishing alias
    # row), not something to guess at with heuristics.
    return "player_" + normalize_name(canonical_name).replace(" ", "_")
