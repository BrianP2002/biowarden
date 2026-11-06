# Registered documents

The files in this directory are quoted as they were registered or logged (stage 2 was registered on 2026-11-01). They use the
names of the time. The project has been renamed since, and this table maps the old names, in these documents and in the first
commit, to the current ones.

| Old name | Now |
|---|---|
| `trustworthy-bio-mcp` (project and distribution name, name of the MCP server) | `biowarden` |
| `science_harness`, `src/science_harness/`, `python -m science_harness...` | `biowarden`, `src/biowarden/`, `python -m biowarden...` |
| `tbm-guard` | `biowarden` |
| `tbm-mcp` | `biowarden-mcp` |
| `science-harness-gateway`, `science-harness-keys`, `science-harness-bench` | `biowarden-gateway`, `biowarden-keys`, `biowarden-bench` |
| `science-harness` as the `X-Title` header and the key directory `~/.config/science-harness/` | `biowarden`, `~/.config/biowarden/` |
| `science-harness` as the name of the gateway's MCP server | `biowarden-gateway` |
| `TBM_*`, `SCIENCE_HARNESS_*`, `HARNESS_ANTHROPIC_API_KEY` and the other `HARNESS_*_API_KEY` | `BIOWARDEN_*` |
| `/opt/tbm`, `tbm-shim`, the message tag `[tbm]`, temporary directories named `tbm-...`, a trial's state directory `tbm/`, `/.tbm-outside-the-trial`, the engine thread `tbm-engine` | `/opt/biowarden`, `biowarden-shim`, `[biowarden]`, `biowarden-...`, `biowarden/`, `/.biowarden-outside-the-trial`, `biowarden-engine` |
| `_tbm_breaker`, `tbm_user_rules_` (internal names, renamed in the third commit) | `_biowarden_breaker`, `biowarden_user_rules_` |

One string keeps its old spelling in the code on purpose: `ORDER_SEED = "tbm-family-order"` in the runner. It seeds the order in
which the runner executes the planned trials, so keeping it lets a replication run them in the order of the stage-2 jobs. The
registered plan ([`stage2-plan.json`](stage2-plan.json)) holds counts and digests of the sorted trial ids and does not depend on
it. In these documents and in two places in the code (`bench/pilot.py`, `bench/stage2.py`), "the owner" is the author.

## Checking the freeze

The two stage-2 manifests ([`freeze-stage2.txt`](freeze-stage2.txt), [`freeze-stage2-code.txt`](freeze-stage2-code.txt)) list the
files at their old paths, `src/science_harness/...`. They verify against the first commit of the repository, which holds the code
as it was frozen and carries the tag `stage2-freeze`:

```bash
git switch --detach stage2-freeze
```

then run the commands of [Run and reproduce](../reproduce.md#stage-2). On `main` the tool stops and says so.

- The tagged tree is for checking, not for installing: it has no README, `LICENSE` or `NOTICE`, although its `pyproject.toml`
  declares Apache-2.0 (the license text is in `LICENSE` on `main`).
- The manifests show that the files are those whose digests were written at freeze time. They do not show when that was: the date
  2026-11-01 is the author's word. The first commit and the tag are from 2026-11-05, after the stage-2 runs, and the tag is not
  signed. `stage2-draft.md` says the two root digests "are also in the owner's session record"; that record is the author's
  working log with an AI assistant, which is not published and is not an independent anchor.
- A run of the code on `main` replicates the study under new names that the sandbox shows the agent (`/opt/biowarden`,
  `BIOWARDEN_*`, `biowarden-shim`, `[biowarden]`); it is not the registered condition. [`stage2-report.md`](stage2-report.md)
  counts the trials in which an agent named the old path `/opt/tbm`.
- On Windows, clone with `git -c core.autocrlf=false clone ...`: the manifests hash exact bytes, so line-end conversion fails
  every file.

## What changed in the frozen files afterwards

Two of the later commits change files of the frozen set (`src/`, `pyproject.toml`, `tools/stage2_job.sbatch`,
`tools/preflight.py`, `tools/env_lock.py`, `envs/locks/` and the three registration files); the fourth changes none of them. The
rule specs under `src/biowarden/engine/specs/`, `envs/locks/` and the registration files have the same content in all four
commits.

| Commit | What it changes in the frozen files |
|---|---|
| Second (the rename) | The names in the table above. The project description in `pyproject.toml`. The line breaks of comments and docstrings in `agent.py`, `sandbox.py`, `trial.py`, `precheck.py` and `shim_client.py`, and the wording of one docstring in `spec.py` ("separated" to "joined"). One expression in `spec.py` split over three lines. Nothing else: once the names are mapped, the syntax trees of all the Python files, docstrings aside, are the same. |
| Third | `engine/guard.py` works when `$TMPDIR` is too long for a unix socket path (it stopped with a raw `OSError`). `engine/decide.py` loses a line that declared the field `acknowledged` twice. The last two `tbm` spellings (`agent.py`, `engine/tools.py`) and the gateway's MCP name (`biowarden-gateway`) are renamed. Two docstrings no longer point to unpublished notes (`mockfold.py`, `structure_chain_order.py`), and a comment no longer calls public sequences made up (`cases_structure_chain_order.py`). `pyproject.toml` gains the author and the project URLs. None of it changes what a rule decides. |
| Fourth | None. |

The message of the first commit says that the order of the imports changes as well, and so did the changelog of the second
commit. It does not: no import line moved in any file. The message of the second commit says so.

## Other notes

- Only the stage-2 code is in the repository. The six stage-1 freeze files (`freeze.txt`, `freeze-ablation.txt`,
  `freeze-generic.txt`, `freeze-negctl.txt`, `freeze-nonote.txt`, `freeze-rerun.txt`) hold group digests only and are records: the
  snapshot they were taken from is not published. `stage1.md` gives no registration date; the only date in it is a rerun on
  2026-10-29.
- `stage2-report.md` says that a separate coverage study "has not been run". That was true on 2026-11-01; it was run afterwards
  ([cases written without sight of the rules](../blind-coverage.md)).
- `stage2-report.md` says the effect is near zero where no rule applies (Q1). For Haiku and DeepSeek the estimates are, but the Q1
  intervals reach +28.3 and +13.3 points ([results](../results/README.md#strata)).
- `stage2-deviations.md` says that no commit or tag existed for either freeze. That was true when it was written; the first commit
  and the tag `stage2-freeze` were made on 2026-11-05.
- The rules and families cite internal incident ids (`GEN-20`, `SCD-06`, `STR-19`, `T01`), and the version note of the rule spec
  `quant_kallisto.yaml` cites `docs/research/`, which is not published. Both refer to the author's incident notes. The spec stays
  as frozen, because its text is part of its hash.
