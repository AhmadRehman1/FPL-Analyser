#!/usr/bin/env bash
# Commit + push generated artifacts from a scheduled CI job, resilient to another scheduled
# job pushing to master in the same window.
#
# The real failure that motivated this (nightly_backtest run 33661854852): the plain
# `git fetch; git rebase origin/master` retry loop hit a content conflict in
# data/dashboard/app_track_record.json -- both the nightly and the twice-daily pipeline write
# it -- and left a half-finished rebase the loop couldn't get out of. scheduled_pipeline.yml
# already solved this the right way: `git merge -X ours` instead of rebase, so a content
# conflict resolves in this job's favour and there is never a partial state to recover from.
# This is that logic, factored out so every scheduled committer shares it.
#
# "Our output wins" is safe here: every file passed is a deterministic re-compute from
# committed inputs, so the version that loses a race is regenerated identically next run.
#
# Usage: ci_commit_generated.sh [--branch <name>] "<commit message>" <path> [<path> ...]
#   --branch <name>  the ref to reconcile against on a push race (default: master). Pass the
#                    workflow's own GITHUB_REF_NAME for a workflow that can be dispatched on a
#                    non-master branch and still commits (chip_timing_analysis, ml_experiment).

branch="master"
if [ "$1" = "--branch" ]; then
  branch="$2"; shift 2
fi
msg="$1"; shift

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add -- "$@" 2>/dev/null || true
if git diff --cached --quiet; then
  echo "ci_commit_generated: no staged change in $*"
  exit 0
fi
git commit -m "$msg" || { echo "::error::ci_commit_generated: commit failed"; exit 1; }

for attempt in 1 2 3 4 5 6; do
  if git push; then
    exit 0
  fi
  echo "ci_commit_generated: push rejected (attempt ${attempt}/6) -- merging origin/${branch}, our output wins"
  git fetch origin "${branch}"
  if ! git merge -X ours "origin/${branch}" -m "Merge origin/${branch} into scheduled-job commit (generated output wins)"; then
    git merge --abort 2>/dev/null || true
    echo "::error::ci_commit_generated: could not merge origin/${branch} automatically"
    exit 1
  fi
  sleep $((attempt * 3))
done
echo "::error::ci_commit_generated: push kept failing after 6 attempts"
exit 1
