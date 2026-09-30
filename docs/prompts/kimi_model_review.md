# Prompt: independent model review (for Kimi)

Paste everything below the line into Kimi. It assumes Kimi can read the repository
(`AhmadRehman1/FPL-Analyser`, branch `master`), either through repo access or by the files
listed under "Read first" being attached.

---

You are reviewing an existing Fantasy Premier League prediction and decision model as an
independent, skeptical quantitative reviewer. Your job is to find the changes that would most
improve **season-long FPL points**, not to tidy code or restate what the docs already say.

## What the system is

A Python + DuckDB pipeline in `src/fpl_quant/`, built as modules M0-M9:

- **M1 `team_strength.py`**: Dixon-Coles bivariate Poisson with time decay (`xi`) and an Elo prior.
- **M2 `minutes_model.py`**: three-state minutes distribution (0 / 1-59 / 60+) per player,
  shrunk to position averages, adjusted by evidence claims (injury news etc).
- **M3 `expected_points.py`**: per-fixture `ep_total` as a sum of category sub-models (appearance,
  goals, assists, clean sheet, goals conceded, DefCon, bonus via Plackett-Luce, saves). Per-90
  rates are shrunk toward position averages (`rate_shrinkage_params.k_minutes`).
- **M4 `uncertainty.py`** / **M6 `monte_carlo.py`**: variance, cross-player covariance, simulation.
- **M5 `squad_optimizer.py`**: MIQP (SCIP) squad pick, `linear_EP - lambda * risk`, captain variance
  weighted 3x.
- **M7 `backtest.py`**: walk-forward over 2024-25 and 2025-26 with `data_asof` leakage protection,
  plus `recalibrate()` grid searches that propose parameter changes.
- **M8 `transfer_planner.py`**: transfers, hits and chips over a 5-GW horizon.
- **`research/ml/`**: a LightGBM residual model on top of `ep_total`, currently shadow-only.

Parameters are versioned in `params.py`; most were invented at spec time and never fitted.

## Where it stands (as of 2026-09-29)

- Walk-forward headline: **-0.71 pts/GW vs the average manager** (70 GWs).
- The model's own live 2026-27 team: **-69 pts vs the field average after GW5**.
- Retrospective 2025-26 validation with blind v1 params: 6.9th percentile of real managers.
- Residual by price band (positive = model under-predicts): `<5.0` -0.24, `5.0-7.0` +0.31,
  `7.0-9.0` +0.78, `9.0+` +0.93. Cheap defenders/GKs are over-rated and premium attackers are
  under-rated, so the optimizer builds defensive squads and has captained defenders.
- Minutes log score -1.256, **worse than a uniform 3-state baseline (-1.099)**. Brier 0.371.
- The ML residual model beats the quant model on every weekly run (MAE gain ~0.20, CI excludes 0;
  4,222 vs 4,134 simulated manager points). Its biggest correction is the 0-2 predicted bucket,
  where quant predicts 1.14 against a realised 0.53.
- `lambda_value = 0.15` (risk aversion) was never fitted. The recalibration gate optimises an
  unweighted EP MAE, and the walk-forward does not pass the live `rate_shrinkage_params` version,
  so live and backtest run different `k_minutes` (900 vs 450).

## Read first

1. `README.md` (module status and design notes; long, skim the Status section)
2. `docs/reports/2026-10_model_status_refresh.md` (current open items, most important file)
3. `docs/reports/2026-09_model_failure_diagnosis.md`
4. `docs/plans/2026-09_ep_attacker_defender_imbalance.md`
5. `research/ml/REPORT.md` sections 4, 6, 7 and 9, and `research/ml/EXISTING_MODEL_AUDIT.md`
6. Source: `minutes_model.py`, `expected_points.py`, `squad_optimizer.py`, `backtest.py`
   (`run_gameweek_step`, `score_gameweek`, `recalibrate`, `_ep_calibration_mae_for_step`),
   `transfer_planner.py` (chip evaluation), `params.py`.

## What I want from you

Work through these areas. For each one, read the code before forming a view, and say where the
docs above already identified the issue so you don't just repeat them.

1. **Minutes model.** Why is it worse than uniform? Look at the hard `avail == 0 => p_0min = 1`
   rule, how regular starters end up with too little P(60+), shrinkage strength, recency
   weighting, and what signals it ignores (rotation after European fixtures, manager tendencies,
   return-from-injury ramp, FPL `chance_of_playing`). Propose a concrete replacement or fix.
2. **Premium under-prediction.** Break down the +0.93 at 9.0+ (appearance, goals, assists, bonus).
   Is the per-90 shrinkage toward position averages the main cause? Should shrinkage target a
   price- or role-conditioned prior instead? Is FPL's assist definition being under-counted by an
   xA-based rate? Check penalty and set-piece handling.
3. **Cheap defender / GK over-prediction.** Check clean-sheet probability, DefCon thresholds and
   saves against the realised data. Is the 0-2 bucket over-prediction coming from appearance
   points given to players who don't play?
4. **Optimizer and captaincy.** Is the risk term (`lambda`, 3x captain variance) helping or
   hurting points? Should the captain be chosen by EV (or EV with effective ownership) rather
   than a risk-penalised score? Are there multiple captain code paths that disagree?
5. **Evaluation and recalibration.** Is the walk-forward measuring the right thing (XI only, no
   auto-subs; unweighted MAE target; no held-out split; grid ceilings)? What objective should
   recalibration optimise so improvements show up as points, not just MAE?
6. **The ML residual.** Its features and its wins tell you where quant is wrong. Which of its
   corrections could be pulled back into the structural model, and is there a case for
   promoting it out of shadow mode?
7. **Transfers and chips.** Is there "wait for a better week" logic for chips? Is the hit
   threshold and horizon sensible?
8. **Anything else** you find that is costing points, including leakage risks or bugs.

## Rules

- Cite `file:line` for every claim about code. If you are guessing, say so.
- Don't propose anything that would use information not available at the prediction deadline.
  The `data_asof` / `asof_scope()` discipline must hold.
- Prefer changes that can be tested in the existing walk-forward (`scripts/run_backtest.py`) and
  say which metric should move and by roughly how much.
- Don't spend space on code style, typing or refactors unless they hide a correctness problem.

## Output format

1. **Top 5 changes, ranked by expected points gained per unit of effort.** For each: the problem,
   the evidence (numbers or `file:line`), the proposed change, how to test it, the expected
   effect, and the risk.
2. **Bugs found**, with a failing input or scenario for each.
3. **Everything else**, one line each, grouped by area.
4. **Where you disagree with the existing diagnosis docs**, and why.
