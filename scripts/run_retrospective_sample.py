"""Track E, Phase E-1: draw a real, uniform-random sample of ~2,000 real FPL entries' actual
2025-26 season-total points. See docs/plans/2026-08_retrospective_validation_and_ml_decision_plan.md
for the full design and src/fpl_quant/retrospective_validation.py for why this is uniform-random
sampling over the real ID space rather than ingest_fpl_entry_picks.fetch_top_entries() (which
would compare against the world's elite managers, not real managers generally).

Usage (from repo root):
    .venv/Scripts/python scripts/run_retrospective_sample.py [target_n]

R4/R15 (this plan's own privacy requirements): this script's own stdout/log lines and the cache
file it writes contain only aggregate point totals and counts -- no entry_id or name, at any
point, in any output. That guarantee lives in retrospective_validation.sample_real_season_totals's
own return value (never includes entry_id), not something this script has to separately enforce.
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
TARGET_N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CACHE_DIR = REPO_ROOT / "data" / "retrospective"
CACHE_FILE = CACHE_DIR / "2025-2026_real_season_totals.json"

# Real general-population FPL score references (public knowledge, not derived from any single
# entry we fetched): the median/average FPL manager score for a full season has historically
# landed in a wide but bounded band. This isn't a precise external benchmark -- it exists only
# as a sanity backstop per the plan's [A12] (uniform-random sampling still risks a
# dormancy/signup-date skew) -- a sample whose mean/median falls far outside any plausible real
# range is a signal to investigate before trusting the retrospective comparison, not proof the
# sample is correct.
PLAUSIBLE_SEASON_TOTAL_RANGE = (300, 3500)


def main() -> None:
    print(f"[run_retrospective_sample] sampling {TARGET_N} real {SEASON_NAME} season totals "
          f"via uniform-random entry_id draws over [1, {rv.DEFAULT_ID_SPACE_MAX}]...")
    t0 = time.time()
    result = rv.sample_real_season_totals(SEASON_NAME, TARGET_N)
    elapsed = time.time() - t0

    totals = result["totals"]
    n_sampled, n_rejected, n_attempted = result["n_sampled"], result["n_rejected"], result["n_attempted"]
    rejection_rate = n_rejected / n_attempted if n_attempted else None

    print(f"[run_retrospective_sample] done in {elapsed:.1f}s: "
          f"n_sampled={n_sampled} n_rejected={n_rejected} n_attempted={n_attempted} "
          f"rejection_rate={rejection_rate:.3f}" if rejection_rate is not None else
          f"[run_retrospective_sample] done in {elapsed:.1f}s: n_sampled={n_sampled}")

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
        "n_attempted": n_attempted,
        "rejection_rate": rejection_rate,
        "wall_clock_seconds": result["wall_clock_seconds"],
        "id_space_max": rv.DEFAULT_ID_SPACE_MAX,
        "sampled_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totals": totals,
        "mean": statistics.mean(totals) if totals else None,
        "median": statistics.median(totals) if totals else None,
    }
    CACHE_FILE.write_text(json.dumps(cache_payload, indent=2))
    print(f"[run_retrospective_sample] cached to {CACHE_FILE}")


if __name__ == "__main__":
    main()
