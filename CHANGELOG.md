# Changelog

All notable changes to this project are recorded here. The format follows Keep a Changelog 1.1.0. The project is research
software at version 0.0.x: rule ids, messages and the command line may change between versions.

## [Unreleased]

Nothing yet.

## [0.0.1] - Unreleased

The first public state. The version is the one in `pyproject.toml`; the date is set when the first tag is made.

Until 2026-11-04 the project was called `trustworthy-bio-MCP`: its Python package was `science_harness`, its commands
`tbm-guard` and `tbm-mcp`, its environment variables `TBM_*`, `SCIENCE_HARNESS_*` and `HARNESS_*_API_KEY`, and the sandbox
mounted the shims at `/opt/tbm`. They are now `biowarden`, `biowarden` and `biowarden-mcp`, `BIOWARDEN_*` and
`/opt/biowarden`. The registered documents and the logs in `docs/preregistration/` keep the old names, which
[its README](docs/preregistration/README.md) maps; `ORDER_SEED` keeps its old spelling because the registered plan depends on
it. The first commit holds the frozen files as they were frozen. The commit after it changes them by the new names, the
project description in `pyproject.toml`, the order of the imports that the new name moves and the line breaks and wording of a
few comments, and by nothing else.

### Added

- Checking layer: 108 rules (79 call rules, 29 fact rules) in 51 YAML specs, with their checkers in
  `src/biowarden/engine/rules/`. The engine turns rule verdicts into allow, warn or block and never rewrites a call.
  Tiers T0 and T1 are in use; T2 is defined and no rule has it.
- Calibration: `bench.admission` measures every rule on generated instances and writes `engine/specs/promotion.json`; all 108
  rules are promoted to block. `bench.admission.replay` replays finished trials through the engine, and `bench.pilot` says
  whether a new family bites.
- Front ends: PATH shims, a whole-command pre-check, the MCP server `biowarden-mcp`, an in-loop gateway tool (the benchmark arm
  `block-gateway`) and `biowarden -- <command>`.
- Ledger: an append-only, hash-chained record of every decision.
- Benchmark: 105 task families (F1 to F112; F40 and F47 to F52 were not built), each a clean and a faulted twin; a runner that
  executes the agent's commands in bubblewrap, with 13 arms and the commands `run`, `report`, `rescore` and `analyze`; a scorer
  (version 3) that separates silent wrong answers from flagged, undelivered and refused outcomes.
- Two pre-registered studies with their registrations, reports and deviations: stage 1 (6 models, 5 families, 20 held-out
  seeds) and stage 2 (3 models, 104 of the 105 families after one exclusion, 13,104 trials), and
  `docs/results/stage2-results.json` with every number of the stage-2 analysis.
- Two checks after the freeze of stage 2, not pre-registered, without a model call: the regression suites of bedtools, samtools
  and bcftools run with and without the layer (`tools/real_code_scan.py`, `docs/real-code-scan.md`), and silent-error cases
  written without sight of the rules (`tools/blind_coverage.py`, `docs/blind-coverage.md`).
- Reproducibility: environment lock files in `envs/locks/` and `tools/env_lock.py`, the freeze tool
  `docs/preregistration/freeze_hashes.py`, the job template `tools/stage2_job.sbatch` and `tools/preflight.py`.
- Documentation and project files: the README; pages in `docs/` for the architecture, the rule catalogue, the task families,
  the supported tools, the detailed results and how to reproduce them; `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, two
  issue forms and a pull request template; a CI workflow that runs the quick tests.
