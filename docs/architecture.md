# Architecture

A developer's map of the checking layer and its benchmark: how a tool call becomes allow, warn or block, how a rule earns
the right to block, which front end sees what, and what each package of `src/biowarden/` is for. Paths are relative
to `src/biowarden/` unless they begin with `docs/`, `tests/` or `tools/` or are `pyproject.toml`. What the layer
achieved is in [Results in detail](results/README.md); the 105 task families are listed in [families.md](families.md).

Terms. A *spec* is a YAML file of rules. A *rule* is one entry of a spec: when it applies, what it asks, what it says. A
*checker* is the Python function that decides a rule. A *slot* is the convention a rule is about (`index_base`,
`interval_semantics`, `units`). A *family* is a benchmark task, built as a clean and a faulted twin. An *arm* is one
experimental condition of the benchmark.

## How a call is decided

A front end turns a tool call into a `Call` ([`engine/call.py`](../src/biowarden/engine/call.py): the program's
basename, its arguments without declaration markers, the working directory and the declarations) and asks
`Engine.evaluate` ([`engine/decide.py`](../src/biowarden/engine/decide.py)). The engine then:

1. goes through the rules of every loaded spec, in the order of the specs and of the rules within each, and keeps those that
   apply to the call (`_match`);
2. checks the rule's declaration, if it has one, and runs its checker, which returns a verdict, a message and a fix (`_run`);
3. turns each verdict into none, warn or block from the verdict, the rule's tier, the engine mode and the promotion table
   (`_effective`);
4. takes the most severe effect as the action (allow, warn or block), marks the call STOP if the same call was already
   blocked twice, and, if a ledger is configured, appends one record to it.

The engine never rewrites a call. The fix is text for the agent and comes first in a block message (after the STOP line,
when there is one). The text is built in [`engine/render.py`](../src/biowarden/engine/render.py).

## Rule kinds, tiers and where they are enforced

A rule is an entry of a spec in [`engine/specs/`](../src/biowarden/engine/specs/) (schema:
[`engine/spec.py`](../src/biowarden/engine/spec.py)) plus a checker in
[`engine/rules/`](../src/biowarden/engine/rules/), registered with `@checker(name)` and named by the rule's `check`
field. The engine refuses to start on a duplicate rule id or a spec that names an unknown checker. The fields of a rule:

| Field | Meaning |
|---|---|
| `id`, `slot` | the rule's unique id (for example `bcftools.region_semantics`) and the convention it is about; with the tier they head the block message as `id [slot/tier]` |
| `kind`, `tier` | `call` or `fact`; `T0`, `T1` or `T2` |
| `applies` | when the rule is relevant: `tools`, `subcommands`, `options_any` (call rules); `file_patterns`, `tools` (fact rules) |
| `declare` | optional: `key`, allowed `values`, `required`, `description` |
| `check`, `params` | the name of the checker and the free-form arguments it receives |
| `message` | one sentence: what goes wrong without this rule. The agent reads it in the block message for a missing declaration and in the declarations that `biowarden-mcp` and the gateway tool advertise; every other message is written by the checker |
| `incidents` | optional reference ids (`GEN-20`, `SCD-06`, `T01`, ...) into the author's incident notes, which are not published; the engine does not read them |

**Two kinds of rule.** A *call rule* reads one tool's arguments (at tier T1 also the files those arguments name). It
applies when the program's basename (a versioned interpreter such as `python3.10` also matches `python3`), an optional
subcommand (a glob; `+*` matches every bcftools plugin) and optional options match. A *fact rule* reads the files a
command touches, whichever program names them, and fires only when the data are in a risky state (a Phred+64 FASTQ, a
log-scaled matrix); on other data it passes. It applies when a regex of `file_patterns`, searched in every argument
including inside code strings such as `python3 -c` and in the words of earlier commands of the same command string,
names an existing regular file (a relative name is resolved against the working directory; at most 200,000 characters
of a call are searched). `applies.tools` limits the programs a fact rule watches; empty means any program.

**Three tiers.** Of the 108 packaged rules, 79 are call rules and 29 are fact rules; 12 are T0, 96 are T1 and none is T2.

| Tier | What the checker may do | Enforced in |
|---|---|---|
| T0 | decide from the call's arguments alone: no file reads, no dry runs. A T0 rule must carry a declaration, and a fact rule cannot be T0 | `Ctx._need_files` raises `TierViolation` for every tier but T1 ([`engine/checks.py`](../src/biowarden/engine/checks.py)); `Rule._shape` rejects a T0 rule without a declaration and a T0 fact rule (`engine/spec.py`) |
| T1 | read files and run the real tool for a dry run | `Ctx.lines`, `Ctx.read_bytes` and `Ctx.run_real` (`engine/checks.py`) |
| T2 | warn only: whatever the verdict and the mode, the effect is at most a warning. It has no file access either | `Engine._effective` (`engine/decide.py`) |

`Ctx` bounds what a T1 checker can do: regular files only, at most 4,000,000 bytes per read (decoded text for `Ctx.lines`,
which reads plain or gzip files). When a sandbox sits between the agent and the host, `Call.root` is the trial's work
directory, and the engine's file matching and `Ctx.resolve` treat a name that leads out of it, by `..` or a symlink, as a
file that does not exist. The shim server and the pre-check set the root as the benchmark runner wires them; `biowarden`,
`biowarden-mcp` and the in-loop gateway tool set none. A checker that opens a file without `Ctx` is not held to the root:
`bcftools.regions_file_base` does so for the tabix or CSI index next to an input. A dry run gets no stdin, 6 GiB of
address space and at most 30 s, and never more than what is left of the 20 s allowed for the whole evaluation; the tool
and everything it started are killed on a timeout. A dry run is not otherwise confined: the checker chooses its
arguments. A checker that uses a capability its tier does not grant raises `TierViolation`, which the engine treats like
any other crash (below). Two built-in T0 checkers, `option_implies_declared` and `option_value_implies_declared`, let a
spec author write a T0 rule with `params` only; three packaged rules use the first and none uses the second.

Both command-line tools accept more specs: `--specs DIR` (repeatable) and `$BIOWARDEN_SPECS` add directories of YAML specs,
`--no-packaged` drops the shipped ones, `--watch PROGRAM` adds a program to watch, and `--trust-python` imports `*.py`
checkers from the `--specs` directories, which runs their code. Rules loaded this way have no promotion entry, so in
`promoted` mode they only warn, unless `biowarden-mcp --promotion FILE` names a table that has one.

## Verdicts and modes

A checker returns a `CheckResult` with a verdict, a code, a message, a fix and a note.

- **pass**: nothing wrong, or the data are not in a risky state.
- **fail**: the call or the data carry the risk the rule is about.
- **unknown**: the checker cannot decide (a file it cannot inspect, an ambiguous call, a value the rule does not know).
  An `unknown` verdict only ever warns, in every mode, and the warning starts with `UNVERIFIED`, so an undecidable call is
  never blocked.

Four modes (`Engine(mode=...)`; default `promoted`; `biowarden` offers `warn`, `block` and `promoted`, `biowarden-mcp` also `off`):

| Mode | A failing rule |
|---|---|
| `off` | nothing: the engine is a no-op and allows everything |
| `warn` | only warns; the call runs |
| `block` | blocks |
| `promoted` | blocks if the promotion table says `block` for this rule and the spec's current hash, otherwise warns |

Exceptions, all in `Engine` (`_effective`, `_run`, `_judge` and `_evaluate`): an `unknown` verdict and a T2 rule only
warn. A checker that raises fails closed: the verdict is `fail` with code `checker_crashed`, which blocks in `block` and
`promoted` modes (the message tells the agent to run the step another way). A fault of the layer itself is not the
call's fault: a host path the engine may not stat, a dry run that runs out of time, or an evaluation that takes longer
than its 20 s turn the rule into `unknown` with code `layer_error`, which only warns, and the agent is told nothing (the
ledger has the cause). The layer also fails open outside the engine: if a shim cannot reach the engine, the call runs
unchecked and says so on stderr, and if the engine server hits an exception, the shim gets a warning and the call runs.

## Declarations

A declaration states which convention the caller intends: a key and one of the rule's allowed values, for example
`region_semantics=overlap`. It reaches the engine in three ways:

- `--declare key=value` on the command line of a watched program. The shim reads it and removes it before the tool runs.
- the `BIOWARDEN_DECLARE` environment variable, `key=value` pairs separated by `;`. A marker on the command line overrides it.
- the `declared` object of `run_tool` in the MCP server and the gateway tool. A `--declare` inside `args` is rejected there.

A `--declare` that is not of the form `key=value` blocks the call. A value outside the rule's `values` fails with
`declaration_invalid` for any rule that has a `declare` block. The two kinds of declaration differ in `declare.required`
(default true):

- **Required** (call rules only): omission is an error, there is no default. The call fails with `declaration_missing` before
  the checker runs, and the fix names the `--declare` to use (a `@suggester` can tailor it to the call's own options). Nine
  packaged call rules are of this kind, all T0, for example `bcftools.region_semantics` and `featurecounts.read_pairs`.
- **Optional**: the checker decides from the call and the data, and compares a declaration, if there is one, with what the
  call implies. A call in a safe state passes without a declaration. A fact rule inspects the data first and asks for the
  declaration only when they are in a risky state. 89 packaged rules are of this kind.

98 of the 108 packaged rules have a `declare` block; the other 10, all T1 call rules, have none. A watched program that is
called by its path never meets the shim, so a `--declare` on it would reach the tool; the pre-check warns or blocks then
([`engine/precheck.py`](../src/biowarden/engine/precheck.py)).

## Calibration and promotion

In the default `promoted` mode a rule starts as a warning. It may block only after
`python -I -m biowarden.bench.admission` has measured it and written an entry to
[`engine/specs/promotion.json`](../src/biowarden/engine/specs/promotion.json)
([`bench/admission/`](../src/biowarden/bench/admission/)).

1. Every rule has a *case source* in `bench/admission/cases_*.py`, registered with `@case_source(rule_id)`. For one seed it
   builds benchmark workspaces and returns calls labelled right or wrong by running the real tool, each with or without the
   declaration: four cells, right declared, right undeclared, wrong declared and wrong undeclared.
2. The engine, in `block` mode with no retry cap, judges every call. The default seeds are 0-299, and `--rules` measures
   only the rules it names.
3. A rule is *admitted* if it fails at least one wrong call and none of the right declared calls. Every rule except the nine
   with a required declaration is *conditional* (it should stay silent on correct data without a declaration) and must also
   fail none of the right undeclared calls.
4. A rule is *promoted* to `block` if it is admitted and the upper end of the exact (Clopper-Pearson) 95% interval of its
   false-block rate on right declared calls is at most 2% over at least 200 such calls; for a conditional rule the same
   holds for right undeclared calls. With no false block, 200 calls give an upper end of 1.8%. Otherwise the entry says
   `warn`.
5. The table is keyed by rule id, and each entry carries the hash of the spec it was measured on, so the evidence is valid
   for that spec hash only (an entry with no hash is accepted for any spec). `Spec.content_hash()` hashes the parsed spec
   file, so a change to any rule in it makes `Promotion.mode_of` answer `warn` for every rule of that file until the file
   is measured again. The benchmark runner refuses to start unless every packaged rule has a current entry
   (`check_promotion` in `bench/runner/layer.py`).

An entry holds `mode`, `spec_hash`, `seeds`, the tool versions it saw (`tools`), `right_declared` and `right_undeclared` as
[false blocks, calls], `false_block_upper`, `recall` (the share of wrong calls the rule fails), `killed` (wrong calls
failed), `admitted` and the four cells with their counts of calls, failed, unknown and crashed.

In the shipped table all 108 rules are promoted for their current spec hashes, with no false block among 600 to 19,200
right declared calls per rule (upper end of the exact 95% interval 0.02% to 0.61%) and a recall of 0.15 to 1.00. One rule,
`mdtraj.resid_index`, was admitted on seeds 0-39 (600 calls); the others on seeds 0-299. The cases come from the
benchmark's own generator, so these rates describe the generator and not real data.
[`bench/admission/replay.py`](../src/biowarden/bench/admission/replay.py) replays the commands of finished trials
through the engine and counts, by family, twin and outcome, the trials in which a block would have happened; the held-out
seeds of the two studies are the other independent check.

## STOP after repeated blocks

The engine counts how often each call was blocked (`BlockState`: a JSON file with a file lock for `biowarden` and the
benchmark, memory for `biowarden-mcp`). A call is the program, its arguments, its declarations and the working directory, hashed
(`Call.call_hash`); the same call with another declaration is another call. The first two blocks of a call are normal
block messages. When the same call is blocked a third time, because it was already blocked `retry_cap` = 2 times, the
message starts with a STOP line and the call stays blocked:

```text
STOP: this exact call was already blocked 2 times. Do not retry it. Report the problem to the user with status needs_input and explain which convention could not be confirmed.
```

The benchmark's bare reporting has no status, so there the line says to stop and write `{"answer": null}`
(`contract="bare"`). `retry_cap` is a constructor argument; no command-line flag sets it. In stage 2 the layer issued 4, 61
and 16 STOP messages in the 1,030 gated trials of Haiku, DeepSeek and gpt-oss under `block-nonote` (`layer_stops` in the
[results JSON](results/stage2-results.json)). Under the `generic` message style, used only by two post-hoc arms of stage 1,
the text names no rule, and a second identical call is let through instead of reaching STOP.

## Ledger

[`ledger/ledger.py`](../src/biowarden/ledger/ledger.py) keeps an append-only JSONL file with a hash chain. Each record
has `seq`, a UTC timestamp `ts`, its fields, `prev_hash` and `hash` (SHA-256 of the canonical JSON of the record without
`hash`). A sidecar `<ledger>.head` holds the latest `seq` and `hash`, so cutting records off the end is detectable. It is an
integrity aid, not a security boundary: whoever can edit the ledger can edit the sidecar. `append` verifies the whole file
first and refuses to add to one that fails (`LedgerCorruptError`).

With a ledger configured (`biowarden --ledger FILE`, `biowarden-mcp --ledger FILE`; the benchmark writes one per trial), the engine
appends an `engine_decision` record for every call it evaluates: the source (`shim`, `precheck`, `mcp` or `gateway`), tool,
arguments, declarations, call hash, mode, action, `stop`, the hash of every loaded spec, and for each rule that applied its
verdict, code, effective action and note. To check a ledger:

```bash
python -I -c "from biowarden.ledger import verify_ledger; print(verify_ledger('ledger.jsonl'))"
```

## Front ends

Every front end builds `Call` objects and hands them to one `Engine`. They differ in what they can see.

| Front end | Use when | Command | What it sees |
|---|---|---|---|
| `biowarden` ([`engine/guard.py`](../src/biowarden/engine/guard.py)) | any shell, script or agent that runs the tools as processes, and you do not want to change it | `biowarden -- bash`; `biowarden --mode warn -- python my_agent.py` | the engine runs in the `biowarden` process and a directory of shims goes first on PATH for the command and its children. Every call by name of a watched program is checked: the programs named by call rules, and the interpreters of `engine/tools.py` (awk, python3, perl, Rscript and a few more) and other programs that fact rules name. No sandbox. A blocked call exits with status 97 |
| PATH shim ([`engine/install.py`](../src/biowarden/engine/install.py), [`shim_client.py`](../src/biowarden/engine/shim_client.py), [`server.py`](../src/biowarden/engine/server.py)) | you build your own runner and can put a directory first on PATH inside it | library: `install_shims`, `EngineServer`, as wired in [`bench/runner/layer.py`](../src/biowarden/bench/runner/layer.py) | one process at a time: program, arguments, working directory, `--declare` markers, `BIOWARDEN_DECLARE`, and the files named in the arguments. A stdlib-only client asks the engine over a Unix socket. It does not see a pipeline stage that reads stdin and names no file, a program called by its path, or a file opened inside a script. If the engine cannot be reached the call runs unchecked and stderr says so |
| Whole-command pre-check ([`engine/precheck.py`](../src/biowarden/engine/precheck.py), [`shellsplit.py`](../src/biowarden/engine/shellsplit.py)) | your runner receives the whole bash string before it runs | library: `precheck(command, cwd, engine)`, called by the benchmark's sandbox | the command string, split into simple commands (quotes, pipes, `&&`, subshells, here-documents, `cd` tracking, `for` over literal words, wrappers such as `env`, `timeout` and `xargs` unwrapped). Each command is judged with the words of the earlier ones as context, so a fact rule can find the file of a pipeline stage that reads stdin. A block stops the whole string before anything runs. Globs, brace expansion, variables not set by a literal assignment, functions, aliases and sourced files stay unexpanded or invisible. Only the first 200 simple commands are checked, and a warning says so |
| `biowarden-mcp` ([`engine/mcp_server.py`](../src/biowarden/engine/mcp_server.py)) | the host speaks MCP and the model can be given a typed tool | `biowarden-mcp --cwd DIR` | one tool, `run_tool(tool, args, declared)`, for the programs named by call rules (and any `--watch` program) that it finds on PATH or in the tool directories that `ToolEnv.discover` names (`$BIOWARDEN_TOOLS` is one of its sources). Only calls made through that tool; the host's other tools are not seen. A refused call is an MCP error whose first text block is the fix. No sandbox |
| In-loop gateway tool ([`bench/runner/gateway_tool.py`](../src/biowarden/bench/runner/gateway_tool.py)) | you write the agent loop and can add a typed tool next to bash | library: `make_gateway`, used by the benchmark's `block-gateway` arm | the same `run_tool`, for samtools, bcftools and bedtools; an allowed call runs in the sandbox as bash would. Bash stays available |

A typed tool next to an unchecked `bash` leaves a way round, and stage 1 measured it. In the `block-gateway` arm the five
open-weight models used `run_tool` in 386 of 1,000 trials, and 42% of their faulted tasks still ended in a silent wrong
answer, against 79% with no layer and 3% with shims and pre-check
([details](results/README.md#stage-1-five-trap-types-six-agents)). A typed tool protects the calls that go through it; the
shim and the pre-check protect the calls that go through bash.

The benchmark's arms are engine settings. The table is `ARMS` in
[`bench/runner/layer.py`](../src/biowarden/bench/runner/layer.py).

| Arm (`--arm`) | Engine mode | What the agent works with | System prompt |
|---|---|---|---|
| `bare` | none | no layer | no note about the layer |
| `warn`, `warn-nonote` | `warn` | shims and pre-check; the real tools are hidden, so only the shims reach them | the layer's note; none |
| `block`, `block-nonote` | `promoted` | the same | the layer's note; none |
| `block-shim` | `promoted` | shims without the pre-check; the real tools are hidden (never run in the two studies) | the layer's note |
| `block-bypass` | `promoted` | shims and pre-check, with the real tools also on PATH, so a call can bypass the layer | the layer's note |
| `block-gateway` | `promoted` | `run_tool` next to an unchecked bash; no shims, no pre-check | the layer's note |
| `blockall` | `block` | a development arm: shims and pre-check, and every failing rule blocks whatever the promotion table says | the layer's note |
| `warn-generic`, `block-generic` | `warn`, `promoted` | as `warn-nonote` and `block-nonote`, with the generic message style | none |
| `careful`, `note` | none | no layer | a sentence asking to check formats; the layer's note only |

The layer's note is `LAYER_NOTE` in [`bench/runner/agent.py`](../src/biowarden/bench/runner/agent.py): three
sentences that say commands may be checked, to read a refusal or warning and follow its FIX, and that some commands take
`--declare key=value`. The `careful` sentence is `CAREFUL_NOTE` in the same file.

`block` is the engine's `promoted` mode. Because every packaged rule is promoted for its current spec hash, `promoted` and
`block` decide alike on the packaged rules.

## Families by locus

The 105 families are grouped by *locus*, the layer where the error lives
([`bench/tasks/locus.py`](../src/biowarden/bench/tasks/locus.py); the per-family list is [families.md](families.md)).

| Locus | Families | The error lives in | Example |
|---|---|---|---|
| tool default | 65 | the tool's documented but obscure default or option semantics | F1: bcftools reads a regions file by its extension |
| data | 25 | the files themselves (encoding, ID form, coordinate base) | F13: FASTQ qualities written as Phred+33 or Phred+64 |
| documentation | 7 | a README or `--help` text that no longer matches the installed behaviour | F25: the notes swap the two quality letters of `samtools coverage` |
| task statement | 4 | the user's wording, which disagrees with the data or a convention | F21: the sentence says how CDS coordinates are written, and the faulted file uses the other base |
| wrapper | 4 | a local wrapper script that changes what the tool does or hides what it says | F17: the lab's counting wrapper adds a mapping-quality filter nobody asked for |

The code also defines a sixth locus, `service`, which no family uses. None of the four wrapper families has a rule of its
own; the 47 families of the stage-2 primary set are 30 tool default, 15 data, 1 documentation and 1 task statement.

## Directory map

Every top-level package of `src/biowarden/`:

| Path | What it is for | Does the rule engine use it? |
|---|---|---|
| [`engine/`](../src/biowarden/engine/) | the rule engine, the specs, the checkers and the front ends | it is the engine |
| [`ledger/`](../src/biowarden/ledger/) | the append-only hash-chained decision record | yes: `Engine._log` appends one record per decision |
| [`bench/`](../src/biowarden/bench/) | the benchmark: task families, sandboxed runner, scorer, the analyses of the two studies, rule calibration | no. It uses the engine. One exception: `biowarden-mcp` calls `bench.tasks.tools.ToolEnv` at start-up to find the real tool directories |
| [`contracts/`](../src/biowarden/contracts/) | tool contracts for the mock structure predictor `mockfold`: a pydantic schema, two YAML contracts in `data/`, the request payload and the hash helpers `canonical_json` and `sha256_hex` (`request.py`) | only in part: `engine/spec.py` and `ledger/` use the two hash helpers, and the rule `structure.chain_order` reads the contract data through `contracts/loader.py` and `contracts/schema.py` |
| [`validators/`](../src/biowarden/validators/) | pre-call validation of a request against a contract; a repairable request gets a proposed fix that is never applied silently | no; `gateway/` uses it |
| [`oracles/`](../src/biowarden/oracles/) | post-call invariants on a tool's output, metamorphic probes, and a gemmi-based output parser | no; `gateway/` and one admission case source (`bench/admission/cases_structure_chain_order.py`) use it |
| [`gateway/`](../src/biowarden/gateway/) | an MCP server with one tool, `predict_complex`, in front of the mock predictor: pre-call validation, the tool run, post-call oracles and a ledger record per call (console script `biowarden-gateway`) | no. It is a separate, contract-based path. The typed `run_tool` gateway tool of the benchmark is not here but in `bench/runner/gateway_tool.py` |
| [`keys.py`](../src/biowarden/keys.py) | API-key lookup for the benchmark runner (`get_key`) and a hygiene audit that prints names and sources, never values (`biowarden-keys`, or `python -I -m biowarden.keys`) | no |

Inside `engine/`:

| Path | What it is for |
|---|---|
| [`spec.py`](../src/biowarden/engine/spec.py) | the spec schema (`Rule`, `Spec`, `Declare`, `Applies`), the loaders (packaged, `--specs` directories, `$BIOWARDEN_SPECS`) and `Spec.content_hash()` |
| [`specs/`](../src/biowarden/engine/specs/) | 51 YAML specs with 108 rules, and `promotion.json` |
| [`rules/`](../src/biowarden/engine/rules/) | 56 modules with the checkers and their shared helpers; `rules/__init__.py` imports them all, which registers the checkers |
| [`call.py`](../src/biowarden/engine/call.py), [`argv.py`](../src/biowarden/engine/argv.py) | `Call`, the `--declare` and `BIOWARDEN_DECLARE` parsing and the call hash; option readers shared by the checkers |
| [`checks.py`](../src/biowarden/engine/checks.py) | `CheckResult`, `Match`, the checker and suggester registries, `Ctx` (the tier-limited access to files and dry runs) |
| [`decide.py`](../src/biowarden/engine/decide.py), [`render.py`](../src/biowarden/engine/render.py) | `Engine`, `Promotion`, `BlockState`, `Decision`; the block, warning and STOP text |
| [`tools.py`](../src/biowarden/engine/tools.py) | which programs the layer watches, where the real binaries are, `--trust-python` imports |
| [`install.py`](../src/biowarden/engine/install.py), [`shim_client.py`](../src/biowarden/engine/shim_client.py), [`server.py`](../src/biowarden/engine/server.py) | the shim directory, the stdlib-only shim client, the engine's Unix-socket server and the sandbox-to-host path map |
| [`guard.py`](../src/biowarden/engine/guard.py), [`mcp_server.py`](../src/biowarden/engine/mcp_server.py) | the `biowarden` and `biowarden-mcp` commands |
| [`precheck.py`](../src/biowarden/engine/precheck.py), [`shellsplit.py`](../src/biowarden/engine/shellsplit.py) | the whole-command pre-check and the bash splitter it uses |

Inside `bench/`:

| Path | What it is for |
|---|---|
| [`tasks/`](../src/biowarden/bench/tasks/) | the families: `families.py` (F1-F4 and the registry), `extra/` (the other 101, one module each), `family_rules.json` (the rules written for each family), `locus.py`, `reference.py` (a reference and a naive solver per family), `tools.py` (`ToolEnv`), and the shared `base.py` (the `Task` record), `common.py` and `instance.py` (workspace helpers and the random instance). Each extra family must pass the generic gates of `tests/test_extra_families.py` (F1-F4 are gated in `tests/test_bench_tasks.py`): the reference solver passes, the naive solver passes the clean twin and falls into the trap on the faulted one, the truth is not in the workspace |
| [`runner/`](../src/biowarden/bench/runner/) | `cli.py` (`run`, `report`, `rescore`, `analyze`), `trial.py` (one trial: build the workspace, run the agent, score), `agent.py` (the minimal tool loop on the Anthropic API), `agent_openai.py` (the same loop on OpenRouter, with provider pins), `opencode.py` with `proxy.py` and `fwd.py` (the OpenCode harness and the route of its API calls out of the sandbox), `sandbox.py` (bubblewrap), `layer.py` (the arms), `gateway_tool.py`, `exposure.py` (whether the trapped route ran), `pricing.py` (model prices for the cost guard), `resources.py` (sizes the worker pool) |
| [`score.py`](../src/biowarden/bench/score.py), [`analysis.py`](../src/biowarden/bench/analysis.py) | the scorer (pass, silent wrong, flagged wrong and the undelivered outcomes) and clean-against-faulted statistics |
| [`stage1.py`](../src/biowarden/bench/stage1.py), [`stage2.py`](../src/biowarden/bench/stage2.py), [`pilot.py`](../src/biowarden/bench/pilot.py) | the analyses of the two studies and the family admission pilot |
| [`admission/`](../src/biowarden/bench/admission/) | rule calibration (`__main__.py`, `__init__.py`), the case sources `cases_*.py` (54 modules) and `replay.py` |
| [`fixtures/mockfold.py`](../src/biowarden/bench/fixtures/mockfold.py) | `mockfold`, a CPU-only stand-in for structure predictors that has the chain-order trap |

Console scripts (`pyproject.toml`): `biowarden` (`engine/guard.py`), `biowarden-mcp` (`engine/mcp_server.py`),
`biowarden-gateway` (`gateway/server.py`), `biowarden-keys` (`keys.py`) and `mockfold`
(`bench/fixtures/mockfold.py`). Modules run with `python -I -m biowarden.<module>`: `bench.runner.cli` (`run`,
`report`, `rescore`, `analyze`), `bench.admission` and `bench.admission.replay`, `bench.stage1`, `bench.stage2`
(`sets`, `scan`, `health`, `analyze`), `bench.pilot` and `keys`.

## Tests

Tests are in [`tests/`](../tests/). `test_engine_core.py` covers verdicts, modes, tiers, declarations, STOP, the ledger
record and the promotion rule (47 tests, about 2 s). `test_engine_guard.py`, `test_engine_shim.py`, `test_engine_mcp.py`,
`test_engine_shellsplit.py` and `test_engine_hardening.py` cover the front ends and the way the layer fails;
`test_ledger.py` the hash chain; `test_rules_*.py` one rule group each; `test_bench_tasks.py`, `test_extra_families.py`
and `test_family_*.py` the family builders and gates; `test_gateway_mcp.py`, `test_precall.py`, `test_oracles.py` and
`test_probes.py` the contract-based path. The runner and the sandbox have `test_runner_*.py`, `test_bench_runner.py` and
`test_sandbox_isolation.py`, and the two analyses `test_stage1.py`, `test_stage2.py` and `test_stage2_strata.py`. Tests
marked `slow` build real workspaces and run the real tools (`pyproject.toml`).
