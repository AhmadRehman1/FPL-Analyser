"""Track E, Phase E-1: draw a real, uniform-random sample of ~2,000 real FPL entries' actual
2025-26 season-total points. See docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md
for the full design and src/fpl_quant/retrospective_validation.py for why this is uniform-random
sampling over the real ID space rather than ingest_fpl_entry_picks.fetch_top_entries() (which
would compare against the world's elite managers, not real managers generally).

Usage (from repo root):
    .venv/Scripts/python scripts/run_retrospective_sample.py [target_n] [--force]

R6: a real live run takes tens of minutes and makes thousands of real requests to FPL's own
API -- this script will not silently redo that work. If a cache file already exists, it prints
a summary of what's cached and exits without any network access; pass --force to re-sample.

R4/R15 (this plan's own privacy requirements): this script's own stdout/log lines and the cache
file it writes contain only aggregate point totals and counts -- no entry_id or name, at any
point, in any output. That guarantee lives in retrospective_validation.sample_real_season_totals's
own return value (never includes entry_id, and a real fetch failure is reported as a sanitized
n_errors count rather than a raw, URL-bearing exception -- see that module's own docstring for
the Critique Engine finding this fixed), not something this script has to separately enforce.
"""

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fpl_quant import retrospective_validation as rv  # noqa: E402

SEASON_NAME = "2025/26"  # the real API's own slash format -- see module docstring
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
FORCE = "--force" in sys.argv[1:]
TARGET_N = int(_args[0]) if _args else 2000
CACHE_DIR = REPO_ROOT / "data" / "retrospective"
CACHE_FILE = CACHE_DIR / "2025-2026_real_season_totals.json"

# A rough sanity backstop, not a precise statistical test (per the plan's [A12]: uniform-random
# sampling still risks a dormancy/signup-date skew that this alone cannot fully rule out).
# Tightened from an earlier (300, 3500) after a Critique Engine pass on this phase noted the
# original band was wide enough to let a meaningfully elite-skewed sample (mean ~2800) pass
# silently -- 3000 is still a generous upper bound (comfortably above all but the most
# exceptional individual full-season totals), but no longer that permissive for a *sample mean*.
PLAUSIBLE_SEASON_TOTAL_RANGE = (500, 3000)


def _print_cached_summary(cached: dict) -> None:
    print(f"[run_retrospective_sample] a cached sample already exists at {CACHE_FILE} "
          f"(sampled_at_utc={cached.get('sampled_at_utc')}): "
          f"n_sampled={cached.get('n_sampled')} mean={cached.get('mean')} median={cached.get('median')}")
    print("[run_retrospective_sample] not re-fetching (R6) -- pass --force to re-sample.")


def main() -> None:
    if CACHE_FILE.exists() and not FORCE:
        _print_cached_summary(json.loads(CACHE_FILE.read_text()))
        return

    print(f"[run_retrospective_sample] sampling {TARGET_N} real {SEASON_NAME} season totals "
          f"via uniform-random entry_id draws over [1, {rv.DEFAULT_ID_SPACE_MAX}]...")
    t0 = time.time()
    result = rv.sample_real_season_totals(SEASON_NAME, TARGET_N)
    elapsed = time.time() - t0

    totals = result["totals"]
    n_sampled, n_rejected = result["n_sampled"], result["n_rejected"]
    n_errors, n_attempted = result["n_errors"], result["n_attempted"]
    rejection_rate = n_rejected / n_attempted if n_attempted else None

    print(f"[run_retrospective_sample] done in {elapsed:.1f}s: "
          f"n_sampled={n_sampled} n_rejected={n_rejected} n_errors={n_errors} "
          f"n_attempted={n_attempted} rejection_rate={rejection_rate}")
    if n_errors:
        print(f"[run_retrospective_sample] NOTE: {n_errors} draw(s) hit a real fetch error "
              f"(not just 'entry doesn't exist') -- if this is a large share of n_attempted, "
              f"investigate before trusting the sample (could indicate real API throttling).")

    if n_sampled < TARGET_N:
        print(f"[run_retrospective_sample] WARNING: only reached {n_sampled}/{TARGET_N} -- "
              f"attempt budget exhausted. Per the plan's Open Items fallback, the correct next "
              f"step is drawing more (never switching to a rank-ordered source).")

    if totals:
        mean_total = statistics.mean(totals)
        median_total = statistics.median(totals)
        lo, hi = PLAUSIBLE_SEASON_TOTAL_RANGE
        plausible = lo <= mean_total <= hi and lo <= median_total <= hi
        print(f"[run_retrospective_sample] mean={mean_total:.1f} median={median_total:.1f} "
              f"plausible_general_population_range={plausible}")
        if not plausible:
            print(f"[run_retrospective_sample] WARNING: sample mean/median falls outside "
                  f"{PLAUSIBLE_SEASON_TOTAL_RANGE} -- possible sampling skew (see plan [A12]), "
                  f"investigate before using this sample for the Phase E-3 report.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "season_name": SEASON_NAME,
        "target_n": TARGET_N,
        "n_sampled": n_sampled,
        "n_rejected": n_rejected,
        "n_errors": n_errors,
        "n_attempted": n_attempted,
        "rejection_rate": rejection_rate,
        "wall_clock_seconds": result["wall_clock_seconds"],
        "id_space_max": rv.DEFAULT_ID_SPACE_MAX,
        "sampled_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totals": totals,
        "mean": statistics.mean(totals) if totals else None,
        "median": statistics.median(totals) if totals else None,
    }
    CACHE_FILE.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    print(f"[run_retrospective_sample] cached to {CACHE_FILE}")


if __name__ == "__main__":
    main()
