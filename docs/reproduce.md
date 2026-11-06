# Run and reproduce

The commands the [README](../README.md) leaves out: the install extras, the tests, the benchmark runner, rule calibration, how
the two studies are run, and what can and cannot be regenerated from the files in this repository. Commands run from the
repository root. A Python module is started with `python -I -m ...`: `-I` ignores `PYTHONPATH`, the user site and the current
directory, so no stray file shadows the installed package. The frozen stage-2 jobs are the exception ([Stage 2](#stage-2)).

The project was called `trustworthy-bio-MCP` until 2026-11-04. The registered documents in
[`preregistration/`](preregistration/) and the logs keep the old names of the package, the commands and the variables;
[its README](preregistration/README.md) maps them to the current ones.

[Install](#install) | [API keys](#api-keys) | [MCP client](#mcp-client-configuration) | [Tests](#tests) |
[Benchmark runner](#benchmark-runner) | [Calibration](#calibration) | [Stage 1](#stage-1) | [Stage 2](#stage-2) |
[Regenerating](#what-can-and-cannot-be-regenerated)

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate     # Python 3.10 or newer
pip install -e .                                  # the checking layer: biowarden, biowarden-mcp
pip install -e ".[bench]"                         # add an extra, or several: ".[bench,figures]"
```

| Extra | Adds | Needed for |
|---|---|---|
| none | `pydantic`, `pyyaml`, `gemmi`, `fastmcp`, `h5py` | the rules, `biowarden`, `biowarden-mcp` |
| `bench` | `anthropic>=2.3,<3`, `openai>=3.40,<4` | the benchmark runner; `bench.stage2` imports the runner's `plan()` |
| `dev` | `pytest`, `pytest-xdist`, `ruff` and the two SDKs of `bench` | the tests, which import the runner |
| `figures` | `matplotlib` | `tools/make_figures.py`, `tools/make_diagrams.py` |

**Programs.** The checking layer needs only the programs you call through it. The runner and the calibration need samtools,
bcftools and bedtools; the runner also needs bubblewrap, and the calibration of a rule needs the tool the rule is about. A test
that needs a missing program skips. The three are looked up in this order: the JSON file that `BIOWARDEN_TOOLS` names,
`.local/tools.json` of the current directory (for `biowarden-mcp`, of its `--cwd` directory), then `PATH`. The file has `bin_dirs` (the
directories of the three programs) and optionally `lib_dirs` (shared libraries) and `tail_dirs` (more tool directories,
searched after `/usr/bin`, for example a conda environment's `bin`); `.local/` is git-ignored.
[`supported-tools.md`](supported-tools.md) lists the versions of the other programs the studies used, and
[`envs/locks/`](../envs/locks/) pins them.

**Tool environments.** The slow tests and the calibration of the tool-pack rules (alignment, quantification, structure,
single-cell and the rest) need programs and Python environments that this repository does not install. [`envs/locks/`](../envs/locks/)
has one `<environment>.tsv` for each of four conda environments (`bio-a`, `bio-py`, `bio-sc`, `bio-struct`): name, version, build,
URL and sha256 of every package, without paths. An equivalent environment can be built from the URLs (checked here only with
`conda create --dry-run`; packages that were pip-installed inside an environment are not in the lock, and an old archive may have
left the channel):

```bash
for env in bio-a bio-py bio-sc bio-struct; do
  (echo @EXPLICIT; tail -n +2 envs/locks/$env.tsv | cut -f4) > /tmp/$env.explicit.txt
  conda create -n $env --file /tmp/$env.explicit.txt
done
```

Then name the `bin` directory of each environment in `tail_dirs` of `.local/tools.json`. The rules and tests also name three
programs, `python-bio`, `python-sc` and `python-struct`: symlinks, in a directory listed first in `tail_dirs`, to the Python of
`bio-py`, `bio-sc` and `bio-struct`. None of this is needed for the quick tests, the layer or the README demo.

**Bubblewrap.** The runner refuses to start without it (exit status 2). It is looked up in `BIOWARDEN_BWRAP`, the system
`bwrap`, `.local/bin/bwrap` of the current directory, then `PATH`, and it needs unprivileged user namespaces: the CI job installs
the distribution's package and sets `kernel.apparmor_restrict_unprivileged_userns=0` on Ubuntu 24.04
([`ci.yml`](../.github/workflows/ci.yml)).

## API keys

Only the benchmark runner and `tools/preflight.py` use keys. Put them in a `.env` file at the repository root; `.env` is
git-ignored and [`.env.example`](../.env.example) is the template. Two names matter: `BIOWARDEN_ANTHROPIC_API_KEY` for the Claude
models and `BIOWARDEN_OPENROUTER_API_KEY` for every model id that contains a `/` (the DeepSeek, gpt-oss and other open-weight
models of the studies, served through OpenRouter). The example file also lists `BIOWARDEN_DEEPSEEK_API_KEY` and
`BIOWARDEN_ZAI_API_KEY`, which `biowarden.keys` knows but no runner path reads.

A key is looked up in the process environment, then in the file `BIOWARDEN_ENV` names, `.env` of the project root and
`~/.config/biowarden/keys.env`, and is never exported to the environment. Use the prefixed names: the bare
`ANTHROPIC_API_KEY` is also read, but when it is exported Claude Code bills the API instead of your plan.
`python -I -m biowarden.keys` prints names, sources and file modes, never values, and flags a key file that is open to
group or others, a key file that git does not ignore, and a bare `ANTHROPIC_API_KEY`.

## MCP client configuration

`biowarden-mcp` serves one tool, `run_tool(tool, args, declared)`, over MCP on stdio. In Claude Code:

```bash
claude mcp add biowarden -- "$PWD/.venv/bin/biowarden-mcp" --cwd /path/to/data
```

For a client that reads an `mcpServers` JSON file:

```json
{
  "mcpServers": {
    "biowarden": {
      "command": "/path/to/repo/.venv/bin/biowarden-mcp",
      "args": ["--cwd", "/path/to/data"]
    }
  }
}
```

- The command is an absolute path because a client does not start the server with the virtual environment activated. The
  programs must be on the `PATH` the client gives the server, or named in a tools file ([Install](#install)).
- `--cwd` is required and is where the tools run. Use a directory you trust: `biowarden-mcp` reads `.local/tools.json` there unless
  `BIOWARDEN_TOOLS` is set ([`SECURITY.md`](../SECURITY.md#what-the-guard-is-not)).
- The server opens no port. Started in a terminal it logs one INFO line to stderr and waits for a client on stdin.
- A convention goes in the `declared` object, for example `{"region_semantics": "overlap"}`. Block messages word a declaration
  as `--declare key=value`; through `run_tool` give the same key and value in `declared`, because a `--declare` marker in
  `args` is rejected. A blocked call is an MCP error whose first text block is the fix.
- Options: `--mode {off,warn,block,promoted}` (default `promoted`), `--ledger FILE`, `--promotion FILE`, `--specs DIR`,
  `--no-packaged`, `--watch PROGRAM`, `--trust-python`.

## Tests

```bash
python -m pytest -q -m "not slow"             # the quick tests
python -m pytest -q -n 4                      # the full suite; -n needs pytest-xdist (extra dev)
python -m pytest -q tests/test_rules_var_tabix.py
```

`tests/conftest.py` marks `slow` every test file whose name starts with `test_family_`, `test_rules_`, `test_extra_families` or
`test_admission`: the tests that build real workspaces and run the real tools (family gates, rule calibration). Of about 5,900 tests,
about 4,600 are marked and 1,300 are not. The quick subset takes one to three minutes in one process, depending on the load of the
machine; without matplotlib, the two tests of `tools/make_figures.py` and two of `tools/make_diagrams.py` skip. CI runs it with `-n auto` on Python 3.10 and 3.12,
and fails if a test skips because samtools, bcftools, bedtools or the sandbox is missing, or if more than 250 tests skip.

The last recorded run of the full suite, on the tree before the first commit, is [`fulltests.log`](fulltests.log): 5,907 passed,
10 skipped and 1 expected failure in 2,246.83 s (37 min 26 s), with `-n 16 --dist loadfile` on a 16-core cluster node. An earlier log, taken
before the code freeze of stage 2 and before the project was renamed, is [`preregistration/fulltests.log`](preregistration/fulltests.log):
5,770 passed, 6 skipped and 1 expected failure in 3,403.74 s (56 min 43 s). The suite needs the tool environments: a test whose
program is missing skips. Two
environment variables change the seeds: `BIOWARDEN_GATE_SEEDS=1000-1004` runs the family gates of `tests/test_bench_tasks.py` and
`tests/test_extra_families.py` on a held-out seed block of stage 2, and `BIOWARDEN_ADMISSION_SEEDS` (default 40) sets how many seeds
the rule tests calibrate on.

## Benchmark runner

`run` plans one trial for every model, family, seed, repetition and twin (clean or faulted), builds each workspace, runs the
model with one `bash` tool inside bubblewrap, and scores `result.json` outside the sandbox. The outcomes are `pass`,
`silent_wrong`, `flagged_wrong`, `format_error`, `protocol_fail` and `refused`, and the runner adds `infra_error` and
`skipped_budget` ([`bench/score.py`](../src/biowarden/bench/score.py)). It needs the programs and bubblewrap
([Install](#install)) and a key ([API keys](#api-keys)), calls a model API and spends money. Two trials, the clean and the
faulted twin of F1 for seed 0, with the block arm:

```bash
python -I -m biowarden.bench.runner.cli run --out runs/x --models haiku --families F1 --seeds 0 --arm block
python -I -m biowarden.bench.runner.cli report --out runs/x
```

The registered hardest cell for one model and one arm, as in stage 2 (103 families, 2 twins, 5 seeds: 1,030 trials):

```bash
python -I -m biowarden.bench.runner.cli run --out runs/hard --models haiku --exclude-families F34 --seeds 0-4 \
    --contract bare --tier implicit --route forced --arm block-nonote --env-locks envs/locks
```

| Flag | Meaning | Default |
|---|---|---|
| `--out DIR` | run directory (required); one run at a time per directory | |
| `--models` | aliases `haiku`, `sonnet`, `opus`, `dsflash`, `glmflash`, `qwenflash`, `mimo`, `gptoss` and others, or full model ids; an id with a `/` goes to OpenRouter, the others to the Anthropic API | `haiku` |
| `--families`, `--exclude-families` | family ids `F1` to `F112` (105 exist) | all |
| `--seeds`, `--reps` | `0-9` or `0,3,5-7`; repetitions of each trial | `0-1`, 1 |
| `--contract`, `--tier`, `--route` | the cell, see below | `full`, `explicit`, `free` |
| `--arm` | the checking layer, see below | `bare` |
| `--max-turns`, `--effort` | turns per trial; reasoning effort | 30, `medium` |
| `--max-usd`, `--max-trial-usd` | stop launching trials past this total; cap on one trial | 5.0, 0.75 |
| `--workers` | parallel trials; lowered to 2 outside a SLURM job and to what the memory limit allows | 4 |
| `--limit N` | only the first N trials of the plan | all |
| `--promotion FILE` | promotion table for the block arms | the shipped table |
| `--no-pin` | let OpenRouter pick the provider; by default `dsflash` and `gptoss` are pinned to one endpoint, without fallback | pinned |
| `--env-locks DIR` | lock files hashed into `run.json` | `.local/env-locks` |
| `--snapshot DIR` | refuse to start unless the package is imported from this frozen copy | none |
| `--harness` | `minimal` (one bash tool, used in both studies) or `opencode` (OpenRouter models, arms `bare` and `block-shim`, SLURM job or `--allow-login-node`) | `minimal` |

**Cells.** A cell is the combination of three settings. Reporting (`--contract`): `full` asks for `status`, `answer`, `flags`
and `method` in `result.json`, `bare` for `answer` only, so every wrong answer is silent by construction. Evidence (`--tier`):
`explicit` states the convention outright in the workspace README, `implicit` gives at most a term of art, or nothing when the
data themselves carry the evidence. Tool (`--route`): `free` leaves the tool open in the task sentence, `forced` names the tool
whose default is the trap. The hardest cell, the primary cell of both studies, is `bare`, `implicit`, `forced`; the easy cell,
the runner's default and the default cell of stage 2, is `full`, `explicit`, `free`. F4 asks the agent to detect a problem,
needs the `full` contract, and is planned only in the easy cell.

**Arms.** The settings are `ARMS` in [`bench/runner/layer.py`](../src/biowarden/bench/runner/layer.py); the shims and
the whole-command pre-check are described in [`architecture.md`](architecture.md#front-ends). The last column says which report
uses the arm.

| Arm | What the agent works with | Note in the system prompt | Reported in |
|---|---|---|---|
| `bare` | no layer | none | both studies, the baseline |
| `note` | no layer | the layer's note | stage 1, post hoc |
| `careful` | no layer | a note asking to check formats and biological sense | stage 1, post hoc |
| `warn` | shims and pre-check; failing rules warn and the call runs; real tools hidden | the layer's note | stage 1 |
| `block` | the same; rules the promotion table promotes block | the layer's note | stage 1 |
| `warn-nonote`, `block-nonote` | as `warn` and `block` | none | stage 1, post hoc; stage 2 |
| `warn-generic`, `block-generic` | detection only: the message says only to check formats and biological meaning | none | stage 1, post hoc |
| `block-bypass` | as `block`, with the real tools also on `PATH`, so a call can bypass the layer | the layer's note | stage 1 |
| `block-gateway` | no shims; a `run_tool` function checked by the engine next to an unchecked bash | the layer's note | stage 1 |
| `block-shim` | as `block` without the pre-check | the layer's note | not reported |
| `blockall` | every failing rule blocks, whatever the promotion table says (development) | the layer's note | not reported |

**Run directory.** Beside `trials/<model>/<trial>/` the runner writes `plan.json` (every planned trial), `run.json` (the
configuration; a later run in the directory must match it), `unrun.jsonl` (trials the budget skipped) and `jobs.jsonl` (one
line per run). A trial directory holds `score.json`, `task.json`, `transcript.json`, `work/` and, for a layer arm, `biowarden/` with
the ledger. Exit status: 0 every planned trial has a final record; 2 refused to start; 3 a trial is absent or an `infra_error`
has tries left (run the same command again: finished trials are skipped, an `infra_error` is tried up to three times); 4 the run
stopped early (credit exhausted, an API outage, or a promotion table that no longer matches the specs). `report --out DIR`
summarises a directory by cell, twin and model; `analyze --out DIR [DIR ...]` compares clean and faulted twins per cell and
model; `rescore --out DIR` scores finished trials again with the current rules.

**Cost.** The runner stops launching trials once `--max-usd` is spent and caps one trial at `--max-trial-usd`. For scale, the
stage-2 held-out jobs spent USD 6.32 on 4,368 Haiku trials, 7.47 on 4,368 DeepSeek V4.1 Flash trials and 2.31 on 4,368
gpt-oss-120b trials ([`stage2-report.md`](preregistration/stage2-report.md)), about USD 0.0014, 0.0017 and 0.0005 per trial.
The 4,368 trials of a model are both cells, three arms and the replicate run.

## Calibration

A rule blocks only after it has been measured. For every rule, `bench.admission` builds workspaces for each seed, runs the real
tool to label each candidate call right or wrong, lets the engine judge it in block mode, and writes the counts to the
promotion table. A rule is admitted when it fails at least one wrong call and no correct declared call. It is promoted to `block`
when, in addition, the exact 95% upper bound of its false-block rate is at most 2% over at least 200 correct declared calls. A
rule that should stay silent on correct data without a declaration (a fact rule, or a call rule without a required
declaration: 99 of the 108) must pass the same test on correct calls without one
([details](architecture.md#calibration-and-promotion)).

```bash
python -I -m biowarden.bench.admission --rules tabix.regions_file_base --seeds 0-1 --out runs/promotion-test.json
python -I -m biowarden.bench.admission --seeds 0-299      # every rule; writes the shipped table
```

The first command prints one line per rule and leaves it at `warn`, because 20 correct declared calls are fewer than 200:

```console
tabix.regions_file_base            warn   admitted=True false-block 0/20 (upper 0.1684) recall 0.6667 [1s]
```

Without `--out` the command writes `src/biowarden/engine/specs/promotion.json`, merging into the entries already there.
The shipped table holds all 108 rules, promoted for their current spec hashes, so use `--out` unless you mean to replace it.

- `python -I -m biowarden.bench.admission.replay --run runs/x [--json FILE]` replays the commands of finished trials
  through the engine in block mode and counts, by family, twin and outcome, the trials in which a block would have happened.
  The calls carry no declaration and the workspace is the final one, so read the caveats in its module docstring.
- `python -I -m biowarden.bench.pilot --out runs/pilot-new --seeds 0-4` reads the finished `bare` runs of one model in the
  hardest cell and says for each new family whether the trap bites (at least 2 silent wrong answers among at least 3 trials
  that took the trapped route), was routed around, or was handled.

## Stage 1

Registered in [`stage1.md`](preregistration/stage1.md), reported in [`stage1-report.md`](preregistration/stage1-report.md): six
models (`haiku`, `dsflash`, `glmflash`, `qwenflash`, `mimo`, `gptoss`) and the families F1, F7, F10, F13 and F15, in two cells.

| Cell | Flags | Seeds | Arms |
|---|---|---|---|
| Primary | `--contract bare --tier implicit --route forced` | 1000 to 1019 | `bare`, `warn`, `block`, `block-bypass`, `block-gateway` |
| Safety | `--contract full --tier explicit --route free` | 1000 to 1009 | `bare`, `warn`, `block` |

Each combination of model, cell and arm is one `run` call into `runs/stage1`. Stage-1 requests to OpenRouter were not pinned
to an endpoint, so every call into the directory takes `--no-pin` (`run.json` records the setting). For example:

```bash
python -I -m biowarden.bench.runner.cli run --out runs/stage1 --models dsflash --families F1 F7 F10 F13 F15 \
    --seeds 1000-1019 --contract bare --tier implicit --route forced --arm block --no-pin
python -I -m biowarden.bench.stage1 --out runs/stage1
```

`bench.stage1` prints, per model and arm, the share of faulted trials that ended silent over all planned trials, exact McNemar
tests against `bare` with Holm correction, the side effects (clean pass, friction, delivered, flagged), the pooled cluster
bootstrap and the registered criterion. The job scripts of the registered runs are not part of the repository; the stage-2
template is.

## Stage 2

[`stage2-draft.md`](preregistration/stage2-draft.md) is the registration, despite its name: the freeze lists it by path. Its
Procedure section has every command; in five lines:

1. Prepare: the full test suite, the family gates on both seed blocks (`BIOWARDEN_GATE_SEEDS=1000-1004` and `1020-1024`), and
   `python tools/env_lock.py --out envs/locks`.
2. Freeze the code in `runs/stage2-code` with `docs/preregistration/freeze_hashes.py --stage2`, and check each provider with
   `tools/preflight.py` (`endpoints`, `smoke`, `probe`, `credits`).
3. Run the pilot (arm `bare`, seeds 0-4, `CELL=pilot`), replay it (`bench.admission.replay`), build the sets (`bench.stage2
   sets`) and the plan (`tools/preflight.py plan-manifest`), run a dress rehearsal on development seeds, and freeze again with
   the sets, the plan and the registration text.
4. Run one job per model and cell (`CELL=primary` or `default`). Each job verifies the snapshot, the freeze, the environments,
   the seed registry (`bench.stage2 scan`) and, for an OpenRouter model, the credit before its first trial; between jobs only
   `bench.stage2 health` is read.
5. Analyse once, from the frozen snapshot: `bench.stage2 analyze --out runs/stage2/* --sets
   docs/preregistration/stage2-sets.json --code-digest <digest> --json results.json`.

A job is submitted from the repository root, where `runs/` must exist for the SLURM log. `OUT` is a fresh directory, `CAP` the
most USD the job may spend, and `MODEL` an alias or id:

```bash
sbatch --export=ALL,MODEL=dsflash,CELL=primary,OUT=runs/stage2/dsflash-primary,CAP=<usd> tools/stage2_job.sbatch
```

Add your account and partition on the `sbatch` command line, which takes precedence over the `#SBATCH` lines. Do not edit the
template: [`tools/stage2_job.sbatch`](../tools/stage2_job.sbatch) is part of the freeze. The job runs plain `python` with
`PYTHONPATH` set to the frozen copy and `--snapshot`, because `-I` would ignore `PYTHONPATH`, and ends `INCOMPLETE` until every
planned trial has a final record; submit the same command again to resume.

To check that the code and the registration files are the frozen ones, copy `src/` and verify against the freeze manifest:

```bash
mkdir -p runs/stage2-code && cp -a src runs/stage2-code/
python docs/preregistration/freeze_hashes.py --stage2 --code-root runs/stage2-code --repo-root . \
    --verify docs/preregistration/freeze-stage2.txt
```

It prints `verified 347 files, root 026780b7...` (the root digest in [`stage2-report.md`](preregistration/stage2-report.md))
and exits 0. Run it on the first commit of the repository: a later commit that changes `src/`, `pyproject.toml`,
`tools/stage2_job.sbatch`, `tools/preflight.py`, `tools/env_lock.py`, `envs/locks/` or the three registration files no longer
matches it.

## Two later checks

Both ran after the freeze of stage 2, against the frozen rules, with no model call, and neither was pre-registered. Their results
are in [`real-code-scan.md`](real-code-scan.md) and [`blind-coverage.md`](blind-coverage.md).

**The test suites of three tools, with and without the layer.** It needs bedtools 2.31.1, samtools 1.24 and bcftools 1.24 on `PATH`
(or a `tools.json` of the benchmark), `perl`, and `bwrap` for the sandbox (`--no-sandbox` runs without it):

```bash
python tools/real_code_scan.py fetch sources        # the three tagged releases; each unpacked test tree is checked against a pinned digest
python tools/real_code_scan.py run sources --out scan.json --compact docs/results/real-code-scan.json
python tools/real_code_scan.py summary scan.json
```

A suite runs three times in a scratch directory (twice as it is, once under `biowarden` in its default mode); it takes about
40 minutes in all on a cluster node.

**Cases written without sight of the rules.** [`blind-cases/`](blind-cases/) holds one JSON file per case; `verify` runs both
commands of each without the layer, `replay` runs them under `biowarden`:

```bash
python tools/blind_coverage.py verify docs/blind-cases --out verified.json
python tools/blind_coverage.py replay docs/blind-cases --verified verified.json --out replay.json
python tools/blind_coverage.py summary replay.json --accepted docs/blind-cases/accepted.json
```

The tools the cases call must be on `PATH`, including the Python environments the structure and single-cell cases use; a case
whose tool is missing fails `verify` and drops out.

## What can and cannot be regenerated

Raw trial logs (transcripts, commands, per-trial scores and the generated workspaces) are not published: `runs/` is git-ignored.

| What | From the published files |
|---|---|
| The registration, the sets, the plan, the freeze manifests | Published. `freeze_hashes.py --verify` checks the code and these files against the manifest (above). |
| The numbers of the stage-2 analysis | Published in [`results/stage2-results.json`](results/stage2-results.json): counts per model, arm and set, the per-family differences, intervals, p-values and the sensitivity rows. `bench.stage2 analyze` cannot be re-run without the trial records: it reads `score.json` of every trial, `plan.json` and `unrun.jsonl`. |
| The numbers of stage 1 | In [`stage1-report.md`](preregistration/stage1-report.md) and [`results/README.md`](results/README.md); there is no JSON. `bench.stage1` reads the trial records. |
| The diagrams | The two README diagrams, [`layers`](figures/layers.svg) and [`decision`](figures/decision.svg) (each with a `-dark` twin), are drawn by `python tools/make_diagrams.py` with no trial data; the numbers in them are read from the code. |
| The figures | Published in [`figures/`](figures/). `tools/make_figures.py` reads `trials/*/*/score.json` under a run directory (the argument may be a quoted glob such as `'runs/stage2/*'`), not the JSON, so a figure cannot be redrawn from the published files alone. |
| The tasks | Rebuilt from family, seed, twin, tier and route; a build of F5 or later is byte-reproducible in one environment (generic gate 5 of `tests/test_extra_families.py`). |
| The rule, tool and family pages | Regenerated from the code: `tools/make_rules_doc.py` and `tools/make_supported_tools_doc.py` (both with `--check`) and `tools/make_families_doc.py`; `tests/test_generated_docs.py` compares them with the committed pages. |
| The calibration | Re-run `bench.admission` with your tool versions; each table entry records the `--version` lines of samtools, bcftools and bedtools it was measured with, and `envs/locks/` pins the other tools. |
| The scan of the test suites | The aggregate is published in [`results/real-code-scan.json`](results/real-code-scan.json). `real_code_scan.py run` reproduces it (the counts of calls, blocks and disturbed tests; the times depend on the machine). |
| The blind cases | The cases, the reviewers' decisions and the aggregate are published ([`blind-cases/`](blind-cases/), [`results/blind-coverage.json`](results/blind-coverage.json)); `blind_coverage.py replay` reproduces the aggregate from the cases. The authors' and reviewers' transcripts are not published. |
| The agents' trials | Not reproducible: a new run gives new trials, and a provider may change what it serves under a model id. |

The figures are drawn with `python tools/make_figures.py RUN STEM` for stage 1 (add `--panel per-model` for the per-model
chart) and `python tools/make_figures.py 'runs/stage2/*' STEM --sets docs/preregistration/stage2-sets.json --stage2` for stage 2.
A bar is the mean over families of each family's share of silent wrong answers on faulted trials of the hardest cell; a
whisker is the 95% percentile interval when the families are resampled. `--stratum` picks the families of the sets file
(`primary`, `q1`, `q2`, `q3`, `bite` or `gated`), and `--theme` and `--modes` the colours. The diagrams use the same theme,
by role: warm browns for what lies outside the layer, the blue ramp for the layer.
