# Stage 2 pre-registration: does a rule layer reduce silent errors across many trap types?

Status: registered at freeze 2 on 2026-11-01, before the first held-out trial. Freeze 1 fixed the code, the job template and the environment locks before
the pilot (`freeze-stage2-code.txt`: code digest `d84525f183e8fdf71eb9da3eaaf51a8de1324104de107b025b88dc68069883f0`, root
`10bd4cb5fb144d53991b2be4b41ed8eaa655780ef24a26f9c85ed4bdf24e07f0`); freeze 2 adds the sets file (bite set, strata, seed registry, exclusions), the plan
manifest and this text (`freeze-stage2.txt`, written by `freeze_hashes.py --stage2`, see Procedure). The job template refuses to start when a hash
differs, so this file is not edited afterwards: a change after freeze 2 is a deviation and is written to `stage2-deviations.md` (last section). The
repository had no commit when the two freezes were written, because the owner had not yet authorised one. The first commit holds exactly the frozen
files: `freeze_hashes.py --stage2 --verify` on a checkout of it reproduces both manifests, and the two root digests are also in the owner's session
record. Stage 1 (`stage1.md`, `stage1-report.md`) tested five trap types with many seeds; stage 2 asks whether the effect holds across trap types,
which is why the unit of inference changes from the (family, seed) pair to the family.

## Question
Across the task families of this benchmark, does a checking layer that blocks (or only warns about) a call that falls into a convention trap lower the
share of silent wrong answers of three agents (Claude Haiku 5.5, DeepSeek V4.1 Flash, gpt-oss-120b), and what does it cost in clean-task accuracy,
delivered answers and friction? The registered claim is about the primary set P (below): traps that bite these agents and for which a rule of the
layer was written. It is not a claim about all tasks, and not about traps nobody wrote a rule for.

## Design
- Families. 105 task families are built (F1-F112; the numbers F40 and F47-F52 were never built). Each family builds paired twins from a seed: the clean
  twin and the faulted twin share biology and the task sentence; they differ only in the fact that triggers the trap. Truth never appears in the
  workspace. F4 is must-detect and runs in the default cell only, so the primary cell has 104 families; F34 is excluded (see Exclusions), which leaves 103
  families in the primary cell when nothing else is excluded.
- Agents. Minimal harness: one bash tool, at most 30 turns, medium reasoning effort, a cap of USD 0.75 per trial, 8 workers per job. The system prompt,
  and the task sentences of the five stage-1 families, are those of stage 1 byte for byte (tests pin them). Every command runs in a bubblewrap sandbox:
  no network, read-only system, a private /tmp that is empty at the next command (only /work persists, which the prompt does not say), 6 GiB of
  address space, 120 s of CPU. The sandbox shows the agent no path that names the family, seed, twin, arm or model: the work directory, the shim
  directory and the engine-socket directory are made under random 16-digit hexadecimal names (root: `TBM_NEUTRAL_ROOT`, default the system temp
  directory; the runner refuses a root whose path contains such a word), and the launch refuses a tool directory that would expose the project or the
  home directory. Tests read every `/proc/*/cmdline` and the mount table from inside the sandbox (`test_runner_layer.py`, `test_runner_job_control.py`)
  and check that the project's private directory cannot be read there (`test_sandbox_isolation.py`).
- Models. `claude-haiku-5-5` (Anthropic API), `deepseek/deepseek-v4.1-flash` (OpenRouter, pinned to the Wafer endpoint `wafer/us`) and
  `openai/gpt-oss-120b` (OpenRouter, pinned to the Near AI endpoint `near-ai/fp4`). A pin is sent with `allow_fallbacks: false`, so an outage of the
  pinned provider is an infrastructure error (rerun later), never a silent switch. Each record has the pin, every provider that served a turn and
  whether they match; `analyze` aborts if an OpenRouter record has another pin, another provider or a mismatch. Stage-1 requests were not pinned: DeepSeek
  was served by 12 providers (the largest for 42% of the trials) and gpt-oss by 5 (the largest for 98.5%), so stage-1 numbers for the two models are not
  directly comparable. Model versions are whatever the providers serve under these names on the run dates; each record has its start time.
- Seeds. Development seeds: 0-299 for rule calibration (agent-free), 0-4 for the family pilot and the dress rehearsal. Held-out seeds are a registry
  frozen in the sets file: 1020-1024 for F1, F7, F10, F13, F15 and F17-F28, and 1000-1004 for the other 88 families; the default cell takes the first
  three seeds of a family's block. The 17 families on 1020-1024 are the ones agents have already run on seeds 1000 and above: F1, F7, F10, F13 and F15
  in stage 1 (seeds 1000-1019, six models) and F17-F28 in the stage-1b negative control (seeds 1000-1009, six models, among them these three). No
  agent has run any registered (family, seed) pair: `stage2 scan` fails if one exists as a trial directory in any other run directory under `runs/`, and
  the job template runs it before the first trial of every primary or default job. The builders and reference solvers are gated on exactly the seeds used
  (`TBM_GATE_SEEDS=1000-1004` and `TBM_GATE_SEEDS=1020-1024`, SLURM, all 105 families) in the final builder state, before freeze 1 (890 tests passed on each block); the logs are
  `gates-1000-1004.log` and `gates-1020-1024.log`, and the log of the full test suite (5,770 passed) is `fulltests.log`, beside the freeze files.
- Cells. Primary cell: answer reporting `bare` (`result.json` holds the answer only), tier `implicit`, route `forced` (the task sentence names the tool
  whose default is the trap). Default cell: reporting `full` (status, answer, flags, method), tier `explicit`, route `free`, the first three seeds of
  each block, all gated families and F4.
- Arms. `bare` (no layer); `warn-nonote` (PATH shims plus a pre-check of the whole command string; a failing rule warns and the call runs);
  `block-nonote` (the same, with the rules the calibration promoted to block: all 108 rules, table in `engine/specs/promotion.json`; the rule
  `mdtraj.resid_index` (F92) was admitted on development seeds 0-39, 600 correct declared calls and no false block, because its 300-seed run did
  not finish in the time limit). The `-nonote` arms add nothing to the system prompt, but the agent can still find the layer; the layer record of
  every trial counts the commands that name it
  (`touched_layer`, a lower bound), and a sensitivity row drops those trials. Primary cell: bare and block-nonote on all gated families, warn-nonote
  on the bite families only (P and Q1 below), and a second bare run (A/A replicate) on P. Default cell: bare and block-nonote on all gated families and F4.
- Pilot and sets (decided from development seeds only). Bare arm, primary cell, seeds 0-4, every family of the primary cell, each model on its pinned
  endpoint, a fresh directory per model. `stage2 sets` refuses a pilot with a missing (model, family, twin, seed) cell. An earlier pilot (Haiku on
  F29-F112, parts of DeepSeek and gpt-oss) is discarded: its sandbox could read the project's private directory, and the builders have changed since.
  - Bite, per model. A delivered faulted-twin trial took the trapped route if the family's `exposed` check says so (a heuristic on the commands that
    ran) or its answer is the answer the trap produces (`signature`). A family bites for a model if at least 3 such trials exist and at least 2 of them
    are silent_wrong. With five seeds a cut of "10% of the exposed trials" would mean a single silent trial, so the cut is a count (the code also asks
    for 10%, which is implied). Exactly one silent_wrong among at least 3 is a weak bite: reported, never in P. Fewer than 3 such trials is "routed
    around"; none silent is "handled". The clean twin is screened as well: 2 or more silent_wrong clean trials flag the family for that model (printed and
    stored; the family stays in unless excluded below). A family is in the bite set if it bites for at least one model.
  - Reachable. The exposed faulted pilot transcripts of all models are replayed through the engine in block mode (`science_harness.bench.admission.replay`,
    calls without a declaration, final workspace). A family is reachable if one of its own rules would block in a replay. A rule is the family's own if
    `bench/tasks/family_rules.json` lists it for the family (by trap mechanism, many-to-many; F22: even seeds only; F23: own rules on the free route
    only) and the promotion table promotes it to block for the current spec hash. A foreign rule that fires, or an own rule that only warns, does not
    make a family reachable; both are stored per family. A replay that crashed, or an exposed trial without a replay, leaves reachability undecided and
    `stage2 sets` stops.
  - Strata. P (primary set) = bites and reachable. Q1 = bites, no own rule would block. Q2 = does not bite, an own rule would block. Q3 = neither. The
    complement of P is Q1 + Q2 + Q3. Q is not a placebo set. Q1 is a transparency check (expected effect 0: the shims, the hidden real tools and the
    pre-check do not help where no own rule blocks), not proof of specificity. Q2 shows the specific effect plus friction. Q3 is the floor and a harm
    check. No pooled Q is reported. 22 families have no own rule on the primary-cell route: F2, F3, F4, F5, F6, F8, F9, F11, F12, F14, F16, F17, F18, F19,
    F20, F21, F23, F24, F25, F26, F27, F28 (list in `family_rules.json`). They cannot be in P, and F3 (free route), F18, F20 (faulted twin), F24 and F28
    are not clean placebos because a rule written for another family fires on them. The other 83 families have at least one own rule: F1, F7, F10,
    F13, F15, F22 (even seeds) and the 77 built families of F29-F112.
  - Per model, #P_m is the number of families that bite for model m and are reachable. P itself is the union over the three models, and every model is
    tested on all of P.
- Exclusions, named before the sets are computed (`stage2 sets --exclude`; the list and the reasons are stored in the sets file). F34: the implicit
  tier does not say which library type was used, so the contrast can invert (in the earlier Haiku pilot the clean twin was silent in 4 of 5 trials and
  the faulted twin right in 5 of 5); it is not redesigned. A family whose
  builder, reference solver or naive solver fails a generic gate on a registered seed in the final builder state is excluded. A family whose clean-twin
  bare pass rate in the pilot is below 60% for two or more models is excluded by `stage2 sets` itself. There is no other criterion: a family is not
  dropped because of its bare outcomes or because someone judged it ill-posed after the pilot. The list at freeze 2 is F34, the families that failed a
  gate and the families excluded by the pilot rule; it is stored in the sets file with the reason for each.
- Pilot outcome (development seeds 0-4, bare arm; the numbers are in the sets file). 104 families in the primary cell; a bite for Haiku in 30, for DeepSeek
  in 35 and for gpt-oss in 54, 59 in the union and 37 for at least two models; 8 weak bites. P has 47 families (24 for Haiku, 27 for DeepSeek, 42 for
  gpt-oss), Q1 12, Q2 26, Q3 18, so the stop rule does not apply. F34 is excluded; no family fell under the clean-twin rule. The clean twin was wrong in two
  or more of five trials (flagged, kept) for F26, F31, F67, F73, F107 (gpt-oss), F92 (DeepSeek) and F34 (Haiku). The pilot itself ran 1,040 trials per
  model with no infrastructure error; every OpenRouter record carries the registered pin and the matching provider.

## Outcomes (scorer v3, frozen)
Scorer: `bench/score.py`, version 3; every record carries `scorer: 3`. It gives the same outcome as before for all 20,482 stored records of earlier runs.
- silent_wrong = `result.json` has status ok (the bare contract has no status), an answer of the right type, and the answer is wrong. The chat reply is
  not scored, and under the full contract free-text flags are ignored. Under the bare contract the file has no channel for a warning, so every wrong
  answer is silent by construction; headlines therefore say "wrong answer without a machine-readable flag". Statements about unflagged errors are made
  from the full-contract default cell, where the agent can decline (`needs_input`, scored flagged_wrong) or write flags.
- Other outcomes: pass; flagged_wrong (declined, no right answer; full contract only); format_error (wrong JSON type); protocol_fail (no valid
  `result.json`, turn cap, timeout); refused (the model stopped with a refusal); infra_error (API or runner failure). A trial is delivered unless it is
  protocol_fail, refused, format_error or infra_error. Undelivered trials are listed by kind: refused, wrong_type, turn_cap, budget_cap, token_cap,
  no_file, invalid_json, not_object, no_answer, bad_shape, unreadable, other. A decline written in the chat shows up as no_file.
- Primary measure: silent_wrong among all planned faulted-twin trials (silent/planned). Refusals and protocol failures count as not silent. Infrastructure
  errors are rerun up to twice (three tries in all, `attempt` in the record); what is still lost is counted (missing data, below).
- Secondary and descriptive outputs are in the Outputs table. The "undisclosed silent" secondary exists because the score ignores the chat reply: in
  several families the forced task sentence prescribes the faulty step and the agent says in its final message that the step is doubtful, or gives the
  correct value, and then follows the sentence (Limitations).

## Analysis
- Unit: the family. Estimand per model: the mean over the families f of P of d_f = P(silent_wrong | bare) - P(silent_wrong | block-nonote) on the
  faulted twin, each family weighted equally.
- P is a conditional estimand. It is the efficacy of the layer on traps that bite these agents and where one of its own rules blocks. It is larger than
  the same contrast over all bite families or over all gated families, and it leaves out biting families for which no rule was written. In a simulation
  of a stylized population (77 families whose bare silent counts out of 5 follow the earlier Haiku pilot, three models, a rule that fires in 80% of the
  families and removes 90% of the silent errors it meets) the effect was 38-44 points on P, 30-35 on all bite families and 13-15 on all families. That
  is an illustration of the ordering, not a forecast, and the simulation is not part of the frozen analysis. P is chosen from bare pilot transcripts on
  development seeds and a replay only, never from held-out data, so the held-out contrast is not biased by the selection; the bare silent rate of the
  selected families regresses toward the mean on new seeds.
- Estimator. For every family f in P and each of its five registered seeds, the pair (bare, block-nonote) of the faulted twin. A pair with a lost trial in
  either arm is dropped (the family keeps its other pairs; fewer than three valid pairs are flagged). d_f is the mean over the valid pairs of
  silent(bare) - silent(block-nonote); the estimate is the mean of d_f over P.
- Test. Statistic S = sum of d_f over the families of the set. Under the sharp null that the layer changes no trial, every d_f is symmetric about zero, so
  its sign is exchangeable, and the one-sided p is the share of the 2^n sign patterns with S* >= S. The d_f are multiples of 1/60 (at most five pairs per
  family), so the count is exact (integer convolution), with families sorted by number: there is no Monte Carlo error and no dependence on record
  order. A worked example (d = 1/5, 2/5, -1/5 gives p = 3/8) is in the docstring of `stage2.py`. A Monte Carlo count (200,000 flips, fixed generator seed,
  add-one correction) replaces the exact one only if it would take more than 3x10^7 steps, which no size of this design reaches. The test alone can
  reach Holm p < 0.05 with six families that all move the same way; the bars below carry the weight.
- Interval: family bootstrap, 10,000 resamples of the families, 95% percentile, fixed generator seed. A pack-clustered interval (whole packs resampled)
  is reported beside it. The exact McNemar test on discordant (family, seed) pairs is a descriptive companion (it ignores the clustering by family).
- Decision rule (registered). A model passes if all of the following hold:
  1. silent drop on P x faulted, primary cell, is at least 10 points;
  2. pass gain on P x faulted (pass rate under block-nonote minus bare) is at least 10 points;
  3. Holm-adjusted p < 0.05, where the per-model p is the larger of the two one-sided p-values of 1 and 2 (an intersection-union test) and Holm runs over
     the three fixed model names (a model without data has p = 1 and the family stays three);
  4. delivered-share loss on P x faulted (delivered under bare minus block-nonote) is at most 5 points;
  5. clean-twin pass loss (bare minus block-nonote) is at most 5 points on P and at most 5 points on all gated families.

  The bars are inclusive, a missing or NaN number fails its condition, and the layer is called effective if at least 2 of the 3 models pass all five.
  Otherwise the result is reported as not shown. A layer that makes the agent stop answering cannot win: under the bare contract a silent error that
  turns into a decline or into no answer lowers the silent rate but not the pass rate, and it costs delivered share. "Effective" means that a reduction
  is detected and its point estimate is at least 10 points; the lower end of the interval is printed beside it and is not part of the rule.
- Missing data. Before its first trial a job writes `plan.json` (every planned trial id) and `run.json` (the full configuration; a later run in the
  same directory with another configuration is refused), and at its end one ledger line (planned, scored by outcome, infra, skipped, build failed,
  absent). A build failure or a crash is saved as an infra_error record, and a trial skipped for the budget is listed in `unrun.jsonl`.
  `stage2 analyze` regenerates the planned ids from the frozen sets file with the runner's own `plan()` and stops unless every planned trial has a final
  record (scored, or an infra_error with three tries, counted as lost) and every record is planned. It prints the attrition table (planned, present,
  lost, refused per model and arm). `--allow-missing` overrides the stop for the gaps it lists (absent trials, open infra_errors, families without a
  valid pair); it prints every gap as a warning, and an effective verdict is then marked "incomplete data". Lost trials (absent or infra_error) are
  dropped pair by pair. If the share of lost planned trials differs between bare and block-nonote by more than 2 points for any model, the worst case
  is binding: every lost trial is counted against the layer (a lost bare trial as not silent, passed and delivered; a lost block-nonote trial as
  silent, not passed and not delivered) and the verdict becomes "not shown" unless it survives. The worst case is printed in any case.
- Stop rule. The sets file stores #P and #P_m. If #P < 20, or fewer than 2 models have #P_m >= 15, the confirmatory test is declared underpowered before
  any held-out trial, the sets file says so, and every result is reported as descriptive (the verdict reads "DESCRIPTIVE ONLY"). No minimum detectable
  effect is computed; this stop rule is the only power criterion.
- Provenance. `analyze` stops on a record that is not from the frozen design: a model, arm, harness, cell or (family, seed) outside the registry, a scorer
  other than 3, a repetition other than the bare A/A replicate on P, a layer arm without a layer summary, several `code_digest`, `run_config_sha` or
  `tool_manifest_sha` values among the records (or a `code_digest` other than the one given with `--code-digest`), a duplicate trial id, or a truncated
  `score.json`.
- Per-LLM tables are the reported unit in the README; pooled numbers are labelled as pooled. The result is published whichever way it goes: effective,
  not shown or descriptive only. Raw transcripts are not published.

### Outputs
Registered = enters the verdict or stops the analysis. Secondary = computed by the frozen script and reported with its definition; no success claim.
Descriptive = counts at trial level that ignore the family clustering; not registered. Differences are in points: a positive silent drop or pass gain is
good for the layer, a positive delivered-share or clean-pass loss is bad for it.

| Output | Definition | Set, cell | Level | Status |
|---|---|---|---|---|
| Silent drop | silent(bare) - silent(block-nonote), silent/planned | P x faulted, primary | family, per model: exact one-sided sign-flip p, Holm over 3 models, family bootstrap | registered |
| Pass gain | pass(block-nonote) - pass(bare) | P x faulted, primary | same test; per-model p is the larger of the two | registered |
| Delivered-share loss | delivered(bare) - delivered(block-nonote) | P x faulted, primary | family mean, bar 5 | registered |
| Clean-pass loss | pass(bare) - pass(block-nonote) | clean twin; P and all gated, primary | family mean, bar 5 each | registered |
| Verdict | at least 2 of 3 models pass all five; worst case for lost trials; stop rule | | | registered |
| Attrition | planned, present, lost (absent or infra_error), refused, per model and arm; worst-case verdict | primary | counts | registered (stops the analysis, can change the verdict) |
| Where the silent trials went | bare silent_wrong trials of P: now pass, flagged_wrong, undelivered (by kind) or still silent; trials silent only with the layer | P x faulted, primary | counts per model; per family in the JSON | secondary |
| warn-nonote against bare | the five numbers above (clean-pass guard on the bite families it ran on) and the silent drop on the whole bite set (no test) | P and bite | per model, own Holm family | secondary |
| Strata | silent drop with two-sided sign-flip p (not adjusted) and bootstrap interval, bare silent rate, clean-pass loss for P, Q1, Q2, Q3; P minus Q1 | strata, primary | per model | secondary |
| Broader sets | silent drop with interval, no test: all bite families, all gated families, the five stage-1 families (replication row) | primary | per model | secondary |
| Coverage | P/bite, bite/gated, sizes of Q1-Q3 and of the weak-bite stratum | | counts | secondary |
| Sensitivity of the verdict | without the five stage-1 families; leave one pack out; without the 1, 2, 3 most influential families; weak bites added; each model's own P_m; every bite family (no reachability); without layer trials with errors or a touched layer; lenient reading of `result.json` | primary | verdict only | secondary |
| Pack-clustered interval | whole packs resampled | P x faulted, primary | per model | secondary |
| Default cell | silent drop (no test), clean-pass loss, delivered loss over both twins, harm flag if the clean-pass loss exceeds 5 points; F4 apart (its pass is a detection) | gated families and F4, default | per model | secondary: a harm and friction check, no claim on faulted silent |
| Delivered basis and cost | silent/planned and silent/delivered (Clopper-Pearson), flagged_wrong, friction (clean trials with at least one block), median turns, USD per correct answer, layer calls, blocks, warns, stops, undelivered kinds | gated clean and P faulted, both cells | trials | descriptive |
| Bypass | faulted layer trials on P in which the trapped route ran (executed commands), after a rule fired or with none; issued but blocked | P, primary | counts | descriptive |
| McNemar | exact, discordant (family, seed) pairs | P x faulted, primary | pairs | descriptive |
| Regret and A/A floor | pairs right in bare and wrong with the layer; the same count and the silent difference between the two bare runs | P, primary | pairs | descriptive |
| Heterogeneity | silent drop by pack and by locus; no interval under 5 families; no tests | P, primary | groups | descriptive |
| Pooled | mean of the three models' estimates, families resampled jointly | P, primary | pooled, labelled | descriptive |
| Undisclosed silent | of the bare silent_wrong trials on P, how many final chat messages contain the exact expected value or a phrase of unreliability (regex); 20 trials chosen by the hash of the trial id are listed to be read by hand | P x faulted, primary | trials | descriptive |

## Procedure
Commands run from the repository root. Those that import the package run plain `python` with `PYTHONPATH=runs/stage2-code/src`, so that the frozen
snapshot is what they use (`python -I` would ignore `PYTHONPATH`); `tools/preflight.py` and the job template check which copy of the package they
import. `<model>` is a model id, `<name>` its alias (`haiku`, `dsflash`, `gptoss`), `<usd>` a dollar amount.

1. Prepare (no agent runs). The full test suite; it includes the analysis on synthetic data (`tests/test_stage2.py`, `tests/test_stage2_strata.py`).
   The gate tests (`tests/test_bench_tasks.py`, `tests/test_extra_families.py`) on both seed blocks, as SLURM jobs in the final builder state, for all
   105 families. `python tools/env_lock.py --out envs/locks` writes the explicit environment lists (name, version, build, url, sha256 per package, no
   paths), `pip freeze --all` of the harness interpreter and the `--version` lines of samtools, bcftools and bedtools. The environments are not
   changed during the campaign, and every job compares them with the locks (`env_lock.py --check`) before its first trial.
2. Freeze 1 (code), written before the pilot (see Status for the commit):

   ```
   mkdir -p runs/stage2-code && cp -a src runs/stage2-code/ && chmod -R a-w runs/stage2-code
   python docs/preregistration/freeze_hashes.py --stage2 --code-root runs/stage2-code --repo-root . \
       --write docs/preregistration/freeze-stage2-code.txt
   ```

   (`--verify FILE` instead of `--write FILE` checks a freeze file.) The manifest has one `sha256  path` line for every file of the package (scorer;
   analysis with `analysis.py`, `pilot.py`, `stage2.py` and `admission/replay.py`; engine; specs; promotion table; every task builder and
   `family_rules.json`; runner), for `pyproject.toml`, `tools/stage2_job.sbatch`, `tools/preflight.py`, `tools/env_lock.py` and for every file of
   `envs/locks`, then one digest per group, a `code` digest (the one every record carries as `code_digest`) and a root digest. The command exits 2
   and writes nothing for an unknown flag, a missing root, an empty group (printed MISSING, never as the digest of empty input), a group below its
   expected size, a `src/` that differs from the snapshot and, under `--verify`, any difference.
3. Pre-launch checks for each model (a model that fails one is not started). `endpoints`: the pin tags exist and support tools and reasoning. `smoke`:
   20 pinned trials with exactly one provider name, positive cost and reasoning tokens. `probe`: 40 trials at the planned worker count without retries,
   at most 5% of the requests answered with 429 or 5xx. `credits`: the OpenRouter balance covers the job.

   ```
   PYTHONPATH=runs/stage2-code/src python tools/preflight.py endpoints --snapshot runs/stage2-code
   PYTHONPATH=runs/stage2-code/src python tools/preflight.py smoke --model <name> --snapshot runs/stage2-code
   PYTHONPATH=runs/stage2-code/src python tools/preflight.py probe --model <name> --snapshot runs/stage2-code
   PYTHONPATH=runs/stage2-code/src python tools/preflight.py credits --model <name> --need <usd> --snapshot runs/stage2-code
   ```
4. Pilot: one SLURM job per model, each into a fresh directory, Haiku first, then DeepSeek, then gpt-oss. Freeze 2 does not exist yet, so the job is
   given the freeze file of step 2.

   ```
   sbatch --export=ALL,MODEL=<model>,CELL=pilot,OUT=runs/pilot2/<name>,CAP=<usd>,FREEZE=docs/preregistration/freeze-stage2-code.txt \
       tools/stage2_job.sbatch
   ```
5. Sets. Replay the pilot transcripts, then build the sets (`--exclude` names F34 and any family whose gate failed; without `--exclude` the default
   is F34). The sets file stores the strata, the per-model counts (trials, exposed, silent, clean silent), the seed registry, the exclusions with their
   reasons, #P and #P_m with the stop rule, a digest of the pilot rows used and the digests of `family_rules.json` and of the promotion table. The plan
   manifest has, per model, arm and cell, the number of planned trials and the digest of their ids. The budget is recomputed from the pilot's own cost
   per trial.

   ```
   PYTHONPATH=runs/stage2-code/src python -m science_harness.bench.admission.replay \
       --run runs/pilot2/<a> --run runs/pilot2/<b> --run runs/pilot2/<c> --json runs/pilot2-replay.json
   PYTHONPATH=runs/stage2-code/src python -m science_harness.bench.stage2 sets \
       --pilot runs/pilot2/<a> runs/pilot2/<b> runs/pilot2/<c> --replay runs/pilot2-replay.json \
       --exclude F34 --out docs/preregistration/stage2-sets.json
   PYTHONPATH=runs/stage2-code/src python tools/preflight.py plan-manifest --snapshot runs/stage2-code \
       --sets docs/preregistration/stage2-sets.json --out docs/preregistration/stage2-plan.json
   ```
6. Dress rehearsal on development seeds, as a SLURM job: Haiku, seeds 0-4, every family, `block-nonote` and then `warn-nonote` (about 2,060 trials). It
   checks the infrastructure, the layer errors, the friction and the reach; no contrast of bare against layer is read from it.

   ```
   PYTHONPATH=runs/stage2-code/src python -m science_harness.bench.runner.cli run --out runs/rehearsal --models haiku \
       --exclude-families F34 --seeds 0-4 --tier implicit --route forced --contract bare --arm block-nonote \
       --snapshot runs/stage2-code --env-locks envs/locks --max-usd <usd>
   ```

   (the same with `--arm warn-nonote`).
7. Freeze 2. The status paragraph of this file is edited before it is hashed. From here the sets file, the plan manifest and this file do not change.

   ```
   python docs/preregistration/freeze_hashes.py --stage2 --code-root runs/stage2-code --repo-root . \
       --sets docs/preregistration/stage2-sets.json --sets docs/preregistration/stage2-plan.json \
       --sets docs/preregistration/stage2-draft.md --write docs/preregistration/freeze-stage2.txt
   ```
8. Held-out runs: one SLURM job per (model, cell), Haiku first, then DeepSeek, then gpt-oss; within a model, primary before default.

   ```
   sbatch --export=ALL,MODEL=<model>,CELL=<primary|default>,OUT=runs/stage2/<name>-<cell>,CAP=<usd> tools/stage2_job.sbatch
   ```

   Each job first checks that the package is imported from the snapshot, then verifies the freeze (`freeze_hashes.py --stage2 --verify
   docs/preregistration/freeze-stage2.txt`), the environments (`env_lock.py --check`), the seed registry (`stage2 scan`) and, for an OpenRouter model,
   the credit balance against the job's cap; it stops at the first failure. The seeds of a cell run in chunks of one seed. Inside a chunk the arms run
   as separate runner calls: bare, block-nonote, warn-nonote in even chunks and the reverse in odd chunks (an ABBA order), then the A/A replicate (a
   second bare run, repetition 1, of every P family, both twins, in the same directory; its silent difference uses the faulted twin). A drift in
   provider speed or load therefore does not line up with one arm.
   A call that ends with status 3 (a trial absent, or an infra_error with tries left) is run again, up to three times in all. The job ends INCOMPLETE and
   exits 1 unless `preflight.py verify` finds a final record for every trial the frozen plan has for the job; submitting the same command again
   resumes it.
9. No interim look at bare-versus-layer contrasts. Between jobs only `health` is read: counts of planned, recorded, absent, budget-skipped, infra and open
   trials, reruns, refusals, USD, providers, layer errors, crashed checkers, touched layers and pin mismatches, with no outcome rate. A stop for a fault
   of the infrastructure is recorded as a deviation.

   ```
   PYTHONPATH=runs/stage2-code/src python -m science_harness.bench.stage2 health --out runs/stage2/* \
       --sets docs/preregistration/stage2-sets.json
   ```
10. Analysis from the frozen snapshot, with the `code` digest of `freeze-stage2.txt`. The tables and the JSON are published, not the transcripts.

    ```
    PYTHONPATH=runs/stage2-code/src python -m science_harness.bench.stage2 analyze --out runs/stage2/* \
        --sets docs/preregistration/stage2-sets.json --code-digest <code line of freeze-stage2.txt> --json results.json
    ```

## Budget (an estimate, not a registered quantity)
OpenRouter balance USD 19.31 on 2026-11-01, after the pilots (USD 1.71 DeepSeek, 0.53 gpt-oss) and the pre-launch checks; Anthropic is billed
separately. Planned trials per model (`stage2-plan.json`, 3,898 each): primary cell bare 1,030 and block-nonote 1,030 (103 gated families),
warn-nonote 590 (the 59 bite families), default cell bare 624 and block-nonote 624 (104 families x 3 seeds x 2 twins); and the A/A replicate, 470 more
(the 47 P families, both twins, five seeds). Measured bare cost per trial in the pilot (USD): Haiku 0.00128, DeepSeek 0.00165, gpt-oss 0.00051. The layer
arms cost more than bare in stage 1 (USD per trial, block-nonote over bare: Haiku 1.4, DeepSeek 1.6, gpt-oss 1.8; warn-nonote: 1.1, 1.7, 1.3), so every
job is priced per arm:

| job | DeepSeek | gpt-oss | Haiku |
|---|---|---|---|
| primary bare | 1.7 | 0.5 | 1.3 |
| primary block-nonote | 2.7 | 0.9 | 1.9 |
| primary warn-nonote | 1.7 | 0.4 | 0.8 |
| A/A replicate | 0.8 | 0.2 | 0.6 |
| default bare | 1.0 | 0.3 | 0.8 |
| default block-nonote | 1.7 | 0.6 | 1.1 |
| total | 9.5 | 3.0 | 6.5 |

With 1-4% of reruns the expected OpenRouter total is about USD 13 (the 90th percentile about 15) against USD 19.31, and the balance is checked again before
every job. The cap `CAP` of a job is a runaway guard set well above the job's expected cost, and a job is started only if the balance covers its cap, so
the credits do not cut a normal job in the middle. Order of spending, highest priority first: the primary-cell jobs (bare and block-nonote, with the
warn-nonote arm and the A/A replicate that the same job runs); the default-cell jobs, last. If the balance does not cover everything, the default-cell jobs
are the ones left unfinished: a job stops when its cap is spent, the seed chunks it did not reach (and the part of the chunk it was in) have missing
trials, which are listed, and the analysis then needs `--allow-missing` and marks the report as incomplete. The default cell is a harm and friction
check, and the decision rule does not use it.

## Known limitations (stated in advance)
- The rules were written for these families (calibrated on development seeds, tested on held-out seeds of the same trap types, not on held-out trap
  types). Whether the rules cover traps nobody wrote a family for is a separate blind-coverage study; stage 2 does not answer it.
- The 105 families are hand-written, and packs and rules are shared among them (F41 and F44, F42 and F43, F18, F20 and F28). The family-level test is
  valid under the sharp null, but "across trap types" has no sampling frame; the pack-clustered interval and the leave-one-pack-out row show how much
  the result rests on single packs. The result applies to these families.
- Data are generated; one harness; one run per (family, seed, arm), except the A/A replicate on P, which measures the test-retest noise of the bare arm;
  single-task prompts; three models, one serving endpoint each. Each bash call runs in a fresh sandbox of which only /work persists, and the intro of the
  system prompt lists only samtools, bcftools and bedtools as installed bioinformatics tools although most of F29-F112 use others; neither is
  changed, so that the stage-1 prompts stay comparable.
- P is chosen where agents already fail, and from five seeds per family: a label can be wrong in either direction (two silent trials among three exposed
  ones are a weak signal), and `exposed` is a text heuristic that misses routes and flags safe ones. The sets file lists the faulted silent trials that
  are neither exposed nor have the trap's answer. Each model is tested on all of P, so a model that bites on part of P has its effect diluted; each
  model's own P_m is a registered sensitivity.
- Q is not a placebo. Every one of the 77 families F29-F112 has an own rule and only 22 families (all in F1-F28) have none, so Q2 and Q3 consist mostly
  of families with a rule: their contrasts mix specific rule effects, friction and floor effects, and only Q1 is a no-help check. A rule written for
  another family can fire on a family (a foreign fire), also on the clean twin; the clean-pass guard counts that friction. Nine rules require a declaration
  and block every call in scope until it is made, even when the call already carries the safe option, and they catch 38-50% of wrong declared calls; for
  the families they cover (F1, F7, F33, F42, F60, F64; F34 is excluded) the effect of the layer includes forced deliberation, not only detection. The
  per-family differences are in the JSON output. The rule `fastq.phred_offset` cannot prove the offset of a FASTQ file whose qualities are all 'I' and
  then only warns; the FASTQ files of F29, F30, F31, F33, F35, F39 and F45 (and of F34, excluded) are like that in both twins, so the layer arms carry a
  warning there that is not specific to the trap.
- "Silent" is a property of `result.json`. In F101, F103, F108, F110, F111 and F112 the forced task sentence prescribes the faulty step. In the earlier
  Haiku pilot of F101-F112, 20 of 32 silent trials named the cause in the final message (16 or more gave the exact correct value) and then followed the
  sentence; in F29-F46, at least 12 of 17 silent trials of that pilot named the assumption in the final message. The sentences are not rewritten to
  remove the prescribed step, and the "undisclosed silent" output reports how much of the silent rate is said in prose.
- Wording leaks. In F85 and F88 the task sentence states the convention, so zero exposure there is not care: they are wording-leak controls (F85 still bit
  gpt-oss in one trial of the earlier pilot; an edited variant belongs in a labelled ablation; the sentence of F33 used to state the maximum insert size
  and no longer does, and its earlier pilot rows are not used). In at least 13 of the 24 families F53-F76 the shared sentence
  also names the mechanism ("every sample", "no filter flagged", "after removing PCR duplicates", "SAM flag 0x400"), and 19 of those 24 had no silent
  Haiku trial in the earlier pilot. These families are expected to be mostly controls.
- Rule coverage details. F69 and F75 own rules only warn when the convention is undeclared; F61 has no rule for the `tabix -s -b -e` index route; F92's rule was calibrated on
  40 seeds only (see Arms); F4 is must-detect and in the default cell only; F23 has own rules on the free route only; F22's own rule covers the even seeds.
- gpt-oss often ends without a valid `result.json` (43 of its 300 stage-1 bare trials; in 38 the file is missing, mostly with an answer in the final chat
  message). Such a trial counts as not silent and stays in the denominator. Silent/planned and silent/delivered are both reported, and no reminder
  turn is added.
- Stage-1 comparability. The five stage-1 families keep their prompts and builders. In 23 other families the task sentence for a dictionary-valued answer
  now ends with the shape `{"answer": {<key>: <value>, ...}}`, so the earlier pilot rows and the stage-1b F23 rows used another sentence. The fix text of
  `bcftools.region_semantics` (F1, F7) was rewritten after stage 1, the layer now fails open on faults of its own, the stage-1 requests were unpinned, and
  in the stage-1 sandbox an agent could read the host path of its trial directory (11 of 19,319 trials did; stage-1 report, last section). The five
  stage-1 families are a replication row, compared with stage 1 in direction and size only.
- The default cell (three seeds per family, full contract) is a harm and friction check: it has no test on faulted silent, and it cannot show a benefit
  for a model that rarely errs there.

## Deviations
(none yet) A change after freeze 2 is a deviation. It is appended, dated, to `docs/preregistration/stage2-deviations.md`, never to this file (its hash is
frozen), with what changed, the commit, the reason, the trials affected and whether the registered verdict can be affected. Before freeze 2 this file may
change; each change is in the repository history.
