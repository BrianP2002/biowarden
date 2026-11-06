## What this changes

<!-- One or two sentences: the rule or family, the tool and version, and the behaviour it is about. -->

## Checklist

`CONTRIBUTING.md` describes each step. Delete the items that do not apply.

- [ ] **Rule or family.** The rule has a spec entry and a checker; the family has a module `fNN_*.py` that passes the generic gates in `tests/test_extra_families.py`.
- [ ] **Twin family and test.** A clean and a faulted twin that differ only in the trigger fact, with the truth outside the workspace, and a test that the rule blocks the faulted call and lets the clean one through.
- [ ] **Admission and promotion.** The rule has a case source in `src/biowarden/bench/admission/`, `python -I -m biowarden.bench.admission --rules <rule ids>` was run for every rule of the edited spec file, and `engine/specs/promotion.json` is part of the change.
- [ ] **Quick tests.** `python -m pytest -q -m "not slow"` passes, and so do the slow tests of the files you touched.
- [ ] **Style.** `ruff check` and `ruff format --check` report nothing for the files you changed.
- [ ] **Docs.** The generated pages (`docs/rules.md`, `docs/families.md`, `docs/supported-tools.md`) are regenerated, counts in the README still hold, and `CHANGELOG.md` has a line under Unreleased.
- [ ] No absolute paths, API keys, personal data or raw trial logs in the files or the description.
