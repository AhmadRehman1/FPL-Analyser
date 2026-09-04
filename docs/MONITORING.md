# Scheduled-workflow monitoring (dead-man's-switch)

GitHub's `schedule:` trigger is **best-effort**: it delays or drops runs under load, and a
public repo's scheduled workflows **auto-disable after 60 days** with no commit to the default
branch. Both `live_data.yml` (Tier A, ~2h) and `scheduled_pipeline.yml` (Tier B, 06:00 / 18:00
UTC) can therefore just *stop running* with nothing in the Actions tab to notice — no failed
run, because there's no run at all.

Each of those workflows ends with a **dead-man's-switch** step: on a fully green run it pings a
[Healthchecks.io](https://healthchecks.io) URL. Healthchecks.io alerts you when a ping **doesn't
arrive** within the period + grace you configure — so it catches both a **missed** run and a
**failed** run (a failed run never reaches the ping step).

All of this is **$0** and consistent with the project's free-tier design. Until it's set up the
step is a clean no-op (`HEALTHCHECK_URL_* not set -- skipping`) and never fails a run.

---

## Setup (one-time, repo owner)

### 1. Create the checks

- Sign up at healthchecks.io (free tier: 20 checks, ample here).
- Create **two** checks:

  | check name | schedule | period | grace |
  |---|---|---|---|
  | `fpl-live-data` | `17 */2 * * *` (+ `40 1,2 * * *`) | 2 hours | 1 hour |
  | `fpl-model-pipeline` | `0 6,18 * * *` | 12 hours | 3 hours |

  Use "Cron" schedule mode and paste the same cron the workflow uses. The generous grace
  absorbs GitHub's scheduling jitter and the pipeline's own multi-hour runtime, so you're only
  alerted on a *real* miss, not a 20-minute slip.

- Copy each check's **ping URL** (`https://hc-ping.com/<uuid>`).

### 2. Add the repo secrets

Settings → Secrets and variables → Actions:

| secret | value |
|---|---|
| `HEALTHCHECK_URL_LIVE` | the `fpl-live-data` ping URL |
| `HEALTHCHECK_URL_MODEL` | the `fpl-model-pipeline` ping URL |

### 3. Choose how you're notified

In Healthchecks.io → Integrations, wire the checks to email / Slack / Telegram / a webhook —
whatever you'll actually see. Test it with the check's "Send a Test Notification" button.

That's it. The next successful run of each workflow sends the first ping; miss the window after
that and you get alerted.

---

## Notes

- The ping is the **last step** and runs only if every prior step succeeded — a broken pipeline
  stops pinging and trips the same alert as a dropped schedule.
- `scheduled_pipeline.yml`'s separate `scenarios` job (a ~2h secondary what-if panel) is **not**
  monitored — it's not on the critical path, and folding it in would make a slow-scenarios run
  look like a pipeline outage.
- The step pings `hc-ping.com` only; no payload, no repo data leaves the runner.
- To monitor a **manual** `workflow_dispatch` too, nothing extra is needed — the step fires on
  any successful run. It just won't *penalise* you for not dispatching one.
