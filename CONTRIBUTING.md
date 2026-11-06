# Contributing

If the layer let a silent error through, or blocked a call that was right, open an issue with the
[missed silent error](https://github.com/BrianP2002/biowarden/issues/new?template=missed-error.yml) or the
[wrong block](https://github.com/BrianP2002/biowarden/issues/new?template=wrong-block.yml) form: the exact command, the smallest
input that shows it, the tool versions, and how you know the right answer. Vulnerabilities go to [`SECURITY.md`](SECURITY.md),
not to an issue.

## Development setup

```bash
python3 -m venv .venv && . .venv/bin/activate     # Python 3.10 or newer
pip install -e ".[dev]"                           # pytest, pytest-xdist, ruff and the two model SDKs
python -m pytest -q -m "not slow"                 # the quick tests: about 1,300 of 5,900, one to three minutes
```

The tests need samtools, bcftools, bedtools and bubblewrap on Linux ([`docs/reproduce.md`](docs/reproduce.md#install)); a test
whose program is missing skips. The slow tests (`test_rules_*`, `test_family_*`, `test_extra_families.py`, `test_admission*`)
build real workspaces and need the tool environments of [`docs/supported-tools.md`](docs/supported-tools.md): run those for the
files you touch, for example `python -m pytest -q tests/test_rules_var_tabix.py`. CI runs the quick tests on Python 3.10 and
3.12, verifies the stage-2 freeze on the tag `stage2-freeze`, and does not run ruff.

Style is set in `pyproject.toml`: line length 100, Python 3.10, ruff rules E, F, I, UP and B (`ruff check src tests tools`,
`ruff format src tests tools`). New modules start with `from __future__ import annotations`, and function signatures carry
type hints. A module docstring says what the module does and, for a rule or a family, which tool version and commands showed
the behaviour.

## Add a rule

A rule is a spec entry, a checker, tests, admission cases and a promotion entry
([how a call is decided](docs/architecture.md#how-a-call-is-decided)).

1. **Spec.** Add the rule to a YAML file in `src/biowarden/engine/specs/` (the schema, `engine/spec.py`, rejects unknown
   keys). A `call` rule reads one tool's arguments, a `fact` rule the files a command touches. Tier `T0` decides from the
   arguments alone, `T1` may read files and run the tool read-only, `T2` can only warn. `message` is one sentence on what goes
   wrong without the rule; the file's `version_note` names the tool versions its rules were checked on.
2. **Checker.** A function `(call, match, ctx) -> CheckResult` registered with `@checker("name")` in
   `src/biowarden/engine/rules/` and named by the rule's `check` field; the package imports every module in it. Read
   files only through `ctx` (`ctx.lines`, `ctx.read_bytes`, `ctx.run_real`), which applies the tier, the trial-directory
   confinement and the size and time limits. Return `fail` with `fix` lines, `pass`, or `unknown` (which only warns).
3. **Tests.** `tests/test_rules_<pack>.py`: unit tests on small files, evidence tests that run the installed tool, and an
   admission test with `measure(...)` over `ADMISSION_SEEDS` seeds (40 by default). `test_rules_*` files are marked `slow`.
4. **Admission cases.** A function registered with `@case_source("rule.id")` in
   `src/biowarden/bench/admission/cases_<pack>.py`. For one seed it builds benchmark workspaces and returns calls
   labelled right or wrong by running the real tool, never by the rule, each with and without the declaration.
5. **Promotion.** `python -I -m biowarden.bench.admission --rules <rule ids>` measures the rules over seeds 0 to 299 and
   writes their entries to `src/biowarden/engine/specs/promotion.json`. A rule blocks only if it is admitted (it fails a
   wrong call and no correct declared call) and the exact 95% upper bound of its false-block rate is at most 2% over at least
   200 correct declared calls; otherwise it warns. A fact rule, or a call rule without a required declaration, must pass the
   same test on correct calls without one. An entry is valid for one hash of the spec file, so after any edit to a spec file
   measure every rule in it again. A layer arm of the runner refuses to start unless every packaged rule has a current entry
   ([details](docs/architecture.md#calibration-and-promotion)).
6. **Family map and pages.** If the rule is written for a family, list it under that family in
   `src/biowarden/bench/tasks/family_rules.json`. Regenerate the pages that quote rule counts:
   `python tools/make_rules_doc.py > docs/rules.md` and `python tools/make_supported_tools_doc.py > docs/supported-tools.md`.

## Add a family

A family builds a clean and a faulted workspace from a seed; the [families page](docs/families.md) lists the 105 there are.

1. **Module.** `src/biowarden/bench/tasks/extra/fNN_<name>.py` exports `SPEC`, a `FamilySpec` with `id`, `slot`, `source`
   (the documentation or issue the trap comes from), `build`, a `reference` solver that handles the trap, a `naive` solver
   that falls into it, an `exposed` check on the agent's commands, `twin_files` and `locus`. The file is found by its name.
   The number sets the domain heading in `docs/families.md` (`DOMAINS` in `tools/make_families_doc.py`; F40 and F47 to F52 are
   unused); a number outside every range is not listed.
2. **Twins.** The clean and the faulted twin share the task sentence and the expected answer and differ only in the fact that
   triggers the trap, in the files named in `twin_files`. The truth never appears in the workspace.
3. **Gates.** `tests/test_extra_families.py` runs on the module without registration: the reference solver passes every twin
   and tier, the naive solver passes the clean twin and gives the recorded trap answer on the faulted one, the twins differ
   only in `twin_files`, the forced route changes only the prompt, the truth is not in the workspace, builds are
   byte-reproducible, the exposure check flags no unrelated command, and `locus` is one of the loci of
   `bench/tasks/locus.py`. Put family-specific tests, such as the exposure check on concrete command lines, in the
   `tests/test_family_*.py` file of the pack.
4. **Map and pages.** Add the family to `src/biowarden/bench/tasks/family_rules.json` (an empty `rules` list if no rule
   is written for it) and regenerate the pages: `python tools/make_families_doc.py > docs/families.md` and the two commands of
   step 6 above. `tests/test_generated_docs.py` fails when one is out of date.

The two studies ran on frozen code (stage 2: 105 families and 108 rules), so what you add is not part of them. A change to
`src/` also makes the tree differ from the stage-2 freeze manifest, which only the first commit of the repository (the tag `stage2-freeze`) matches
([how to check](docs/reproduce.md#stage-2)).

## Pull request checklist

The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) covers the same points.

- [ ] A rule or a family; the twins differ only in the trigger fact, and a test shows the rule blocking the faulted call and
  passing the clean one.
- [ ] Admission cases, and the promotion table updated for every rule of an edited spec file.
- [ ] `python -m pytest -q -m "not slow"` passes, and so do the slow tests of the files you touched.
- [ ] `ruff check` and `ruff format --check` report nothing for the files you changed, the generated pages are regenerated and
  the README counts still hold.
- [ ] `CHANGELOG.md` has a line under Unreleased; no absolute path, key, personal data or raw trial log is in the change.
