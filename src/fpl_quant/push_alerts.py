"""App gap 1: deciding what is worth a phone push, and building the payload -- pure logic, all
data passed in, so it is fully unit-testable without a live FPL fetch or a push service.

The existing scripts/check_deadline_alerts.py path (a newly-doubtful starter in the MODEL's
own from-scratch squad -> a GitHub Issue) stays as-is. This module adds the user-facing,
held-player alerts the app-feature-gaps prompt asks for, computed off the same
data/dashboard/*.json the PWA already reads:

  * captain_doubtful  -- the account's captain or vice-captain is flagged doubtful/injured
  * price_change      -- a player in the account's squad is a projected riser/faller tonight
  * pending_transfer  -- within `lead_hours` of the deadline, the model still recommends a
                         transfer the app can't confirm the user has made
  * pending_chip      -- same, for a recommended chip

`build_push_payload()` collapses a list of alerts to the single notification actually sent
(highest priority first), so a deadline evening never fires four separate buzzes.
"""

from datetime import datetime

APP_URL = "https://ahmadrehman1.github.io/FPL-Analyser/"

# FPL's own player status codes (bootstrap-static elements[].status).
_DOUBTFUL_STATUSES = {"d", "i", "s", "u", "n"}
_STATUS_WORD = {
    "d": "a doubt", "i": "injured", "s": "suspended", "u": "unavailable", "n": "not in the squad",
}
_CHIP_LABELS = {
    "wildcard": "Wildcard", "free_hit": "Free Hit",
    "triple_captain": "Triple Captain", "bench_boost": "Bench Boost",
}
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

DEFAULT_LEAD_HOURS = 3.0
# chance_of_playing_next_round at or below this (when present) counts as doubtful even if the
# status code hasn't flipped yet.
_CHANCE_DOUBT_THRESHOLD = 75


def _is_doubtful(player: dict | None) -> bool:
    if not player:
        return False
    if player.get("status") in _DOUBTFUL_STATUSES:
        return True
    chance = player.get("chance_of_playing_next_round")
    return chance is not None and chance <= _CHANCE_DOUBT_THRESHOLD


def _status_word(player: dict) -> str:
    return _STATUS_WORD.get(player.get("status"), "a doubt")


def compute_alerts(
    *,
    real_squad: dict | None,
    team: dict | None,
    players_by_id: dict[int, dict],
    price_watch: dict | None,
    next_deadline_utc: datetime | None,
    now_utc: datetime,
    lead_hours: float = DEFAULT_LEAD_HOURS,
) -> list[dict]:
    """Returns a list of alert dicts (possibly empty). Each dict: kind, priority, title, body,
    plus entry_id for routing. Deterministic and side-effect free."""
    alerts: list[dict] = []
    squad = (team or {}).get("squad") or []
    entry_id = (team or {}).get("entry_id")

    hours_to_deadline = None
    if next_deadline_utc is not None:
        hours_to_deadline = (next_deadline_utc - now_utc).total_seconds() / 3600.0
    within_lead = hours_to_deadline is not None and 0.0 <= hours_to_deadline <= lead_hours

    # 1. Captain / vice-captain flagged doubtful
    for p in squad:
        if not (p.get("is_captain") or p.get("is_vice_captain")):
            continue
        pdata = players_by_id.get(p.get("player_id")) or {}
        if _is_doubtful(pdata):
            role = "Captain" if p.get("is_captain") else "Vice-captain"
            alerts.append({
                "kind": "captain_doubtful", "priority": "high", "entry_id": entry_id,
                "title": f"{role} doubt: {p.get('web_name')}",
                "body": pdata.get("news") or f"{p.get('web_name')} is flagged {_status_word(pdata)} before the deadline.",
            })

    # 2. Price change on a held player
    held_ids = {p.get("player_id") for p in squad}
    for direction, rows in (
        ("rise", (price_watch or {}).get("risers") or []),
        ("fall", (price_watch or {}).get("fallers") or []),
    ):
        for r in rows:
            if r.get("id") in held_ids:
                verb = "buy before it rises" if direction == "rise" else "sell before it drops"
                alerts.append({
                    "kind": "price_change", "priority": "medium", "entry_id": entry_id,
                    "title": f"Price {direction} tonight: {r.get('web_name')}",
                    "body": f"{r.get('web_name')} ({r.get('team')}) in your squad is projected to {direction} &mdash; "
                            f"{verb} if you were planning a move around them.".replace("&mdash;", "—"),
                })

    # 3. Unconfirmed pending model recommendation, close to the deadline
    if within_lead:
        h = round(hours_to_deadline)
        hold = (real_squad or {}).get("hold_vs_transfer_now") or {}
        # hold_recommendations.recommended_action is one of 'hold' | 'transfer_now' |
        # 'no_action_available' (schema CHECK + transfer_planner.evaluate_hold_vs_transfer);
        # 'transfer_now' is the only one that means "the model wants a transfer this week".
        if hold.get("recommended_action") == "transfer_now":
            recs = (real_squad or {}).get("transfer_recommendations") or []
            top = recs[0] if recs else None
            body = (
                f"{top['player_out']} → {top['player_in']} (+{top['net']:.1f}). Make it on FPL, or decide to skip it."
                if top else "The model still recommends using a transfer this week."
            )
            alerts.append({
                "kind": "pending_transfer", "priority": "high", "entry_id": entry_id,
                "title": f"Deadline in ~{h}h — transfer still recommended", "body": body,
            })
        rec_chip = next((c for c in (real_squad or {}).get("chip_evaluations") or [] if c.get("recommended")), None)
        if rec_chip:
            label = _CHIP_LABELS.get(rec_chip.get("chip_type"), rec_chip.get("chip_type"))
            alerts.append({
                "kind": "pending_chip", "priority": "high", "entry_id": entry_id,
                "title": f"Deadline in ~{h}h — {label} recommended",
                "body": f"The model recommends playing {label} this gameweek. Play it on FPL or hold it deliberately.",
            })

    return alerts


def build_push_payload(alerts: list[dict], *, app_url: str = APP_URL) -> dict | None:
    """Collapse a list of alerts to the ONE notification actually delivered. None when there is
    nothing to send (so the caller can no-op cleanly)."""
    if not alerts:
        return None
    ordered = sorted(alerts, key=lambda a: _PRIORITY_ORDER.get(a.get("priority"), 9))
    lead = ordered[0]
    extra = len(ordered) - 1
    return {
        "title": lead["title"],
        "body": lead["body"] + (f"  (+{extra} more alert{'s' if extra > 1 else ''})" if extra else ""),
        "url": app_url,
        "tag": "fpl-quant-deadline",
        "alert_kinds": [a["kind"] for a in ordered],
    }
