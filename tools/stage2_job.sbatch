#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:50:00
#SBATCH -o runs/slurm-%x-%j.out
# Stage 2: one job per (model, cell). It runs the frozen code of runs/stage2-code and nothing else.
#
#   sbatch -A <account> -J s2-<model>-<cell> \
#     --export=ALL,MODEL=<alias or id>,CELL=<pilot|primary|default>,OUT=<dir>,CAP=<usd> \
#     tools/stage2_job.sbatch
#
# Submit from the repository root (runs/ must exist: the SLURM log goes there). Required: MODEL,
# CELL, OUT (a fresh directory per model and cell), CAP (the most USD this job may spend, over all
# its runs together). Optional, with their defaults:
#   WORKERS=8 CHUNK=1 (seeds per chunk) MAX_TRIAL_USD=0.75 EFFORT=medium MAX_TURNS=30
#   AA=1 (primary cell: also the bare replicate on P) EXCLUDE="" (pilot: families to leave out)
#   SETS=docs/preregistration/stage2-sets.json  FREEZE=docs/preregistration/freeze-stage2.txt
#   SNAP=runs/stage2-code  LOCKS=envs/locks  NEED=$CAP (USD the OpenRouter balance must cover)
# Switches that skip a check and say so: FREEZE=none, ENV_CHECK=0, SCAN=0. Passed on to the runner
# as they are: TBM_NEUTRAL_ROOT (where the sandboxed work directories are made; the default is
# /tmp/tbm-<job id> on the node, and the runner refuses a path that names a trial or a mount without exec)
# and TBM_BREAKER_LIMIT (consecutive API errors that stop the job, default 10).
#
# cell  what it runs
# pilot    arm bare, seeds 0-4, every family (minus EXCLUDE), primary-cell settings; no sets file
# primary  bare + block-nonote on the gated families, warn-nonote on the bite families, each on
#          its registered seeds; reporting bare, tier implicit, route forced
# default  bare + block-nonote on the gated families and F4, the first three seeds of each block
# The seeds of a cell are cut into chunks of CHUNK seeds. Every chunk runs each arm as its own
# runner call, and the order of the arms is reversed in odd chunks (A B C, C B A, A B C ...), so
# a drift in provider speed or load does not line up with one arm. A call that ends with status 3
# (a trial absent, or an infra_error with tries left) is run again, up to 3 times in all: finished
# trials are skipped. Status 2 (refused to start), 4 (credit, API outage, promotion table) and a
# spent CAP stop the job. The job succeeds only if `preflight.py verify` finds a final record for
# every trial the frozen plan has for this model and cell.
set -euo pipefail

: "${MODEL:?alias or id of the model}" "${CELL:?pilot, primary or default}"
: "${OUT:?a fresh output directory for this model and cell}" "${CAP:?USD this job may spend}"
cd "${SLURM_SUBMIT_DIR:?run this script with sbatch, from the repository root}"
case $CELL in
  pilot | primary | default) ;;
  *) echo "CELL must be pilot, primary or default, not $CELL" >&2; exit 2 ;;
esac

PKG=science_harness
PY=${PYTHON:-$PWD/.venv/bin/python}
SNAP=${SNAP:-runs/stage2-code}
SETS=${SETS:-docs/preregistration/stage2-sets.json}
FREEZE=${FREEZE:-docs/preregistration/freeze-stage2.txt}
LOCKS=${LOCKS:-envs/locks}
WORKERS=${WORKERS:-8}
CHUNK=${CHUNK:-1}
MAX_TRIAL_USD=${MAX_TRIAL_USD:-0.75}
EFFORT=${EFFORT:-medium}
MAX_TURNS=${MAX_TURNS:-30}
AA=${AA:-1}
EXCLUDE=${EXCLUDE:-}
ENV_CHECK=${ENV_CHECK:-1}
SCAN=${SCAN:-1}
NEED=${NEED:-$CAP}
JOB="slurm-${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"  # the runner's job id

SNAP_ABS=$(realpath "$SNAP")
export SCIENCE_HARNESS_BWRAP=${SCIENCE_HARNESS_BWRAP:-$PWD/.local/bin/bwrap}
export PYTHONPATH="$SNAP_ABS/src"  # plain python: -I would ignore it
export PYTHONUNBUFFERED=1  # progress lines reach the log as they happen

# the sandboxed work directories get random names under this root (nothing in them names the
# trial); it is the node's temp directory by default, one per job so a killed job leaves only its own
export TBM_NEUTRAL_ROOT=${TBM_NEUTRAL_ROOT:-/tmp/tbm-${SLURM_JOB_ID}}
mkdir -p "$TBM_NEUTRAL_ROOT"
trap 'rm -rf "$TBM_NEUTRAL_ROOT"' EXIT

mkdir -p "$OUT"
LOG="$OUT/job-${SLURM_JOB_ID}.log"
log() { printf '%s\n' "$*" | tee -a "$LOG"; }
# run CMD...: output to the SLURM log and to $LOG; returns the status of CMD, not of tee
run() { local rc=0; "$@" 2>&1 | tee -a "$LOG" || rc=${PIPESTATUS[0]}; return "$rc"; }
fail() { log "ABORT: $*"; exit 1; }

log "=== $(date -u +%FT%TZ) $JOB host=$(hostname) model=$MODEL cell=$CELL out=$OUT cap=$CAP"

# -- before the first trial ---------------------------------------------------------------------
# the package must come from the snapshot (a stray editable install would be run silently)
run "$PY" - "$SNAP_ABS/src" "$PKG" <<'PYEOF' || fail "the package is not imported from $SNAP"
import importlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
found = Path(importlib.import_module(sys.argv[2]).__file__).resolve()
if root not in found.parents:
    sys.exit(f"{sys.argv[2]} is imported from {found}, not from {root}")
print(f"code: {found}")
PYEOF

if [[ -n $(find "$SNAP_ABS" -type f -writable -print -quit 2>/dev/null) ]]; then
  log "WARNING: the snapshot can be written to; freeze it with: chmod -R a-w $SNAP"
fi

if [[ $FREEZE == none ]]; then
  log "WARNING: the snapshot is not checked against a freeze file (FREEZE=none)"
else
  run "$PY" docs/preregistration/freeze_hashes.py --stage2 --code-root "$SNAP" --repo-root . \
    --verify "$FREEZE" || fail "the code, the job template or the env locks differ from $FREEZE"
fi

if [[ $ENV_CHECK == 0 ]]; then
  log "WARNING: the conda environments are not compared with $LOCKS (ENV_CHECK=0)"
else
  run "$PY" tools/env_lock.py --check --out "$LOCKS" || fail "an environment differs from $LOCKS"
fi

SETS_ARGS=()
if [[ $CELL != pilot ]]; then
  [[ -f $SETS ]] || fail "the sets file $SETS is missing"
  if [[ $FREEZE != none ]]; then
    # the code-only freeze (the pilot's) verifies without the sets: a tampered sets file would run
    grep -qF -- "$SETS" "$FREEZE" || fail "$FREEZE does not list $SETS: use the second freeze"
  fi
  SETS_ARGS=(--sets "$SETS")
  if [[ $SCAN != 0 ]]; then
    # no registered (family, seed) may exist in any other run: only this job's own directory (and
    # the directories of the same campaign, kept in one parent) may hold them
    run "$PY" -m "$PKG.bench.stage2" scan --sets "$SETS" --runs runs/* --own "$OUT" \
      || fail "a registered (family, seed) has been run before"
  else
    log "WARNING: the seed registry is not compared with other runs (SCAN=0)"
  fi
fi

run "$PY" tools/preflight.py credits --snapshot "$SNAP" --model "$MODEL" --need "$NEED" \
  || fail "not enough credit for a job of $NEED USD"

# -- the arms and their order ---------------------------------------------------------------------
case $CELL in
  pilot) EVEN=(bare); ODD=(bare) ;;
  primary) EVEN=(bare block-nonote warn-nonote); ODD=(warn-nonote block-nonote bare) ;;
  default) EVEN=(bare block-nonote); ODD=(block-nonote bare) ;;
esac
AA_ARGS=()
if [[ $CELL == primary && $AA == 1 ]]; then
  AA_ARGS=(--aa)
fi

NCHUNKS=$("$PY" tools/preflight.py chunks --snapshot "$SNAP" --cell "$CELL" --size "$CHUNK" \
  ${SETS_ARGS[@]+"${SETS_ARGS[@]}"})
log "chunks: $NCHUNKS of $CHUNK seed(s)"
if (( NCHUNKS < 1 )); then fail "the plan has no seed chunk: is $SETS the frozen sets file?"; fi

# what the runs of this job cost so far: the usd of its ledger lines in jobs.jsonl
spent() {
  "$PY" - "$OUT/jobs.jsonl" "$JOB" <<'PYEOF'
import json
import sys

total = 0.0
try:
    with open(sys.argv[1]) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("job_id") == sys.argv[2]:
                    total += float(row.get("usd") or 0)
            except (ValueError, TypeError, AttributeError):
                sys.exit(f"{sys.argv[1]} line {number} cannot be read: the spend is unknown")
except FileNotFoundError:
    pass
print(f"{total:.4f}")
PYEOF
}

# invoke ARM CONTRACT TIER ROUTE REPS SEEDS FAMILIES: one runner call, repeated on status 3
SPENT=0
invoke() {
  local arm=$1 contract=$2 tier=$3 route=$4 reps=$5 seeds=$6 fams=$7
  local families=() tries=0 rc left
  if [[ $fams != - ]]; then families=(--families $fams); fi  # $fams: ids separated by spaces
  if [[ -n $EXCLUDE ]]; then families+=(--exclude-families $EXCLUDE); fi
  while :; do
    tries=$((tries + 1))
    left=$(awk -v cap="$CAP" -v spent="$SPENT" 'BEGIN { printf "%.4f", cap - spent }')
    if awk -v left="$left" 'BEGIN { exit !(left <= 0) }'; then
      log "STOP: the cap of $CAP USD is spent"
      return 5
    fi
    log "--- $arm reps=$reps seeds=$seeds ($contract/$tier/$route) try $tries, \$$left left"
    rc=0
    run "$PY" -m "$PKG.bench.runner.cli" run --out "$OUT" --models "$MODEL" \
      ${families[@]+"${families[@]}"} --seeds "$seeds" --reps "$reps" --arm "$arm" \
      --contract "$contract" --tier "$tier" --route "$route" --effort "$EFFORT" \
      --max-turns "$MAX_TURNS" --max-trial-usd "$MAX_TRIAL_USD" --max-usd "$left" \
      --workers "$WORKERS" --snapshot "$SNAP" --env-locks "$LOCKS" || rc=$?
    SPENT=$(spent) || { log "ABORT: the spend of this job cannot be read, so the cap cannot hold"; return 6; }
    case $rc in
      0) return 0 ;;
      3) if (( tries >= 3 )); then return 3; fi ;;  # run it again: finished trials are skipped
      *) return "$rc" ;;
    esac
  done
}

# -- the loop: seed chunks outside, arms inside (reversed in odd chunks) -----------------------
FAILED=()
ABORT=0
for ((chunk = 0; chunk < NCHUNKS; chunk++)); do
  if (( chunk % 2 )); then ORDER=("${ODD[@]}"); else ORDER=("${EVEN[@]}"); fi
  if (( ${#AA_ARGS[@]} )); then ORDER+=(aa); fi  # after the others, in every chunk
  for arm in "${ORDER[@]}"; do
    BLOCKS=$("$PY" tools/preflight.py blocks --snapshot "$SNAP" --cell "$CELL" --arm "$arm" \
      --chunk "$chunk" --size "$CHUNK" ${SETS_ARGS[@]+"${SETS_ARGS[@]}"})
    [[ -n $BLOCKS ]] || continue  # this arm has no families in this cell or chunk
    while IFS=$'\t' read -r contract tier route runner_arm reps seeds fams; do
      rc=0
      invoke "$runner_arm" "$contract" "$tier" "$route" "$reps" "$seeds" "$fams" </dev/null || rc=$?
      if (( rc != 0 )); then
        FAILED+=("chunk $chunk $arm seeds $seeds: status $rc")
        log "FAILED chunk $chunk $arm seeds $seeds: status $rc"
        # 3: some trials are still open, the other chunks may go on; anything else stops the job
        if (( rc != 3 )); then ABORT=1; break 3; fi
      fi
    done <<< "$BLOCKS"
  done
done

# -- the verdict is the data: every planned trial has a final record --------------------------------
VERDICT=0
run "$PY" tools/preflight.py verify --snapshot "$SNAP" --out "$OUT" --model "$MODEL" \
  --cell "$CELL" ${SETS_ARGS[@]+"${SETS_ARGS[@]}"} ${AA_ARGS[@]+"${AA_ARGS[@]}"} \
  ${EXCLUDE:+--exclude $EXCLUDE} || VERDICT=$?
log "spent \$$(spent) of \$$CAP; ${#FAILED[@]} failed call(s); verify status $VERDICT; abort=$ABORT"
if (( ${#FAILED[@]} || VERDICT || ABORT )); then
  log "=== $MODEL $CELL INCOMPLETE: submit the same command again to finish it"
  exit 1
fi
log "=== $MODEL $CELL DONE (stage2)"
