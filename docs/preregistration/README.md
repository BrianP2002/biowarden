# Registered documents

The files in this directory are quoted as they were registered or logged (stage 2 was registered on 2026-11-01). They use the
names of the time. The project has been renamed since, and this table maps the old names to the current ones.

| In these documents | Now |
|---|---|
| `science_harness`, `src/science_harness/`, `python -m science_harness...` | `biowarden`, `src/biowarden/`, `python -m biowarden...` |
| `tbm-guard` | `biowarden` |
| `tbm-mcp` | `biowarden-mcp` |
| `science-harness-gateway`, `science-harness-keys` | `biowarden-gateway`, `biowarden-keys` |
| `TBM_*`, `SCIENCE_HARNESS_*`, `HARNESS_ANTHROPIC_API_KEY` and the other `HARNESS_*_API_KEY` | `BIOWARDEN_*` |
| `/opt/tbm`, `tbm-shim` | `/opt/biowarden`, `biowarden-shim` |

One string keeps its old spelling in the code on purpose: `ORDER_SEED = "tbm-family-order"` in the runner, because the registered
plan ([`stage2-plan.json`](stage2-plan.json)) lists the trials in the order that seed gives.

The freeze manifests ([`freeze-stage2.txt`](freeze-stage2.txt), [`freeze-stage2-code.txt`](freeze-stage2-code.txt) and the
others) list the files at their old paths, `src/science_harness/...`. `freeze_hashes.py --verify` checks them against the first
commit of the repository, which holds the code as it was frozen; see [Run and reproduce](../reproduce.md#stage-2). The commit
after it renames the project. In the frozen set it changes the names above, the project description in
`pyproject.toml`, the order of the imports that the new name moves and the line breaks and wording of a few comments, and nothing else.

`stage2-report.md` says that a separate coverage study "has not been run". That was true on 2026-11-01; it was run afterwards
([cases written without sight of the rules](../blind-coverage.md)).
