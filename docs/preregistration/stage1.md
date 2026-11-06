# Stage 1 pre-registration: does a rule layer reduce silent errors of cheap open-weight models?

Written before any stage-1 trial was run. Anything that differs in the final report is listed under "Deviations".

## Question
When an agent calls samtools, bcftools, bedtools or short scripts on bioinformatics files, a call can exit 0 and return a valid-looking but
wrong result because a convention differs from the agent's assumption (a silent error). Does a checking layer that blocks suspicious calls
with a fix, or only warns, reduce silent errors of cheap open-weight models, and what does it cost in friction, undelivered answers and steps?

## Design
- Task families (real tools, paired twins): F1 (bcftools reads a regions file by extension), F7 (`-R` against `-T`), F10 (indel anchor base
  when turning VCF records into intervals), F13 (Phred+33 against Phred+64), F15 (log-scale expression matrix). Twins share biology and task
  sentence; the faulted twin differs only in the fact that triggers the trap. Truth never appears in the workspace.
- Models (minimal harness, one bash tool, 30 turns, medium reasoning effort): deepseek/deepseek-v4.1-flash, z-ai/glm-5.3-flash,
  qwen/qwen3.8-flash, xiaomi/mimo-v2.6-flash, openai/gpt-oss-120b via OpenRouter; claude-haiku-5-5 as a reference model.
- Seeds: 1000-1019 for every family (held out: rules were developed and calibrated on seeds 0-299; earlier exploratory runs used seeds 0-14).
- Primary cell: answer reporting `bare`, tier `implicit`, route `forced` (the task sentence dictates the tool whose default is the trap).
  Safety cell: reporting `full`, tier `explicit`, route `free`, seeds 1000-1009, arms bare / warn / block only.
- Arms (what the agent works with): `bare` no layer; `warn` shims plus command pre-check, every failing rule warns and the call runs;
  `block` the same, rules the calibration promoted block (all five did); `block-bypass` like `block` but the real tools stay on PATH;
  `block-gateway` no shims, an extra `run_tool` function checked by the engine next to an unchecked bash. Non-bare arms add one fixed sentence to
  the system prompt (constant across arms). Optional ablation `block-shim` (no command pre-check) if budget remains.
- Rules were calibrated before this study (`python -m science_harness.bench.admission`, seeds 0-299): zero false blocks on thousands of correct
  declared calls per rule; the table is `engine/specs/promotion.json`.

## Outcomes (scorer v3, frozen)
- Primary: `silent_wrong` (status ok, right type, wrong answer) among ALL planned faulted-twin trials of a model and arm (silent/planned),
  paired with the same (family, seed) in `bare`.
- Secondary: silent_wrong among delivered answers; delivered share (protocol failures, refusals separate); `flagged_wrong` share; clean-twin pass rate
  and the share of clean trials with at least one block (friction); median turns; USD per correct answer; for layer arms, the layer's own record
  (calls, blocks, warns, stops); bypass (trapped-route commands that ran although a rule would have fired).
- Exposure counts only commands that actually ran (refused commands are excluded).

## Analysis
- Per model and arm against `bare`: exact McNemar on discordant (family, seed) pairs of faulted-twin silent_wrong; Clopper-Pearson intervals for rates;
  Holm correction over the four non-bare contrasts per model. Pooled over models: sum of paired differences with a family-by-seed cluster bootstrap
  (exploratory).
- Success criterion (decided in advance): the layer is called effective if `block` versus `bare` lowers faulted silent/planned by at least 10 percentage
  points for at least 3 of 5 open-weight models with Holm-adjusted p < 0.05, AND pooled clean-twin pass rate falls by at most 5 points, AND pooled delivered
  share falls by at most 5 points. Otherwise the result is reported as not shown.
- Claims are limited to these tools, these families, this harness and these models. Claude results are a reference column and not part of the criterion.

## Stopping and exclusions
- Infrastructure errors are rerun up to twice; remaining ones are counted and excluded from rates. Refusals are counted separately.
- Per-job budget cap $6, per-trial cap $1. No interim peeking at faulted-vs-bare contrasts; run health (infra errors, spend) may be inspected.

## Frozen artefacts
SHA-256 digests (`python docs/preregistration/freeze_hashes.py`) are recorded in `freeze.txt` at launch.

## Deviations
(none yet). Launch: code frozen as a copy of src/ (digests in freeze.txt), six SLURM jobs (one per model), output runs/stage1.

1. Infrastructure rerun (2026-10-29, before any result was inspected). Run health showed 20 qwen trials that never completed (a ledger line
   that the splitter cut at a Unicode line separator made the trial's summary fail) and 6 qwen trials lost to provider rate limits (HTTP 429).
   These 26 trials were rerun, as the stopping rule allows, with a second code snapshot (`runs/stage1-code2`, digests in `freeze-rerun.txt`).
   It differs from the first only in: the ledger splits records on "\n" only and scrubs lone surrogates; one engine-wide lock; the layer summary
   tolerates a bad ledger line; the binomial interval used for display is computed in log space (it overflowed on large samples and only
   crashed the end-of-run printout). Rules, promotion table, task builders and scoring decisions are unchanged (digest groups specs, promotion,
   tasks, analysis identical). Three GLM trials had a few calls unchecked because a lone surrogate broke the ledger write (`layer.errors`); they
   are kept, and a sensitivity analysis excludes them.
