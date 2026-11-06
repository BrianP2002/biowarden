"""Rule sc.mudata_obs_union: the spec, the checker on F106 workspaces and real-looking code."""

from __future__ import annotations

import re
import time

import pytest
from test_rules_sc_common import engine_for, tools, verdict  # noqa: F401  (fixture)

from biowarden.bench.tasks import build_task
from biowarden.engine import known_checkers
from biowarden.engine.rules import sc_multimodal as mm
from biowarden.engine.spec import load_packaged

SPEC = load_packaged("sc_multimodal")
RULE = "sc.mudata_obs_union"
ENGINE = engine_for("sc_multimodal")
PY = "python-sc"
HEAD = "import json, mudata as md\nm = md.read_h5mu('cells.h5mu')\n"


def test_spec_is_valid():
    (rule,) = SPEC.rules
    assert rule.id == RULE and (rule.kind, rule.tier) == ("fact", "T1")
    assert rule.declare.key == "mudata_cells" and rule.declare.required is False
    assert rule.declare.values == ["union", "intersection"] and rule.check in known_checkers()
    assert {"python-sc", "python3", "python"} <= set(rule.applies.tools)


@pytest.fixture(scope="module")
def worlds(tools, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("sc-mu")
    out = {}
    for twin in ("clean", "faulted"):
        work = root / twin
        out[twin] = (work, build_task("F106", 4, twin, work, tools, "implicit", "free"))
    return out


def run(worlds, twin, body, declared=None, head=HEAD):
    work, _ = worlds[twin]
    return verdict(ENGINE, RULE, PY, ["-c", head + body + "\n"], work, declared)


def codes(result):
    return None if result is None else result.code


def test_modality_cells_of_the_two_twins(worlds):
    clean = mm.modality_cells(worlds["clean"][0] / "cells.h5mu")
    faulted = mm.modality_cells(worlds["faulted"][0] / "cells.h5mu")
    assert set(clean) == set(faulted) == {"rna", "atac"}
    assert set(clean["rna"]) == set(clean["atac"])
    assert set(faulted["rna"]) != set(faulted["atac"])
    both = set(faulted["rna"]) & set(faulted["atac"])
    assert len(both) == worlds["faulted"][1].truth["expected"] == len(clean["rna"])
    assert mm.modality_cells(worlds["faulted"][0] / "README.md") is None


@pytest.mark.parametrize(
    "body",
    [
        "print(m.n_obs)",
        "print(len(m.obs_names))",
        "print(m.shape[0])",
        "print(m.obs.shape[0])",
        "print(len(m.obs))",
        "n = m.n_obs",
        "print(len(m))",
    ],
)
def test_counting_the_mudata_itself_fails_on_unequal_modalities(worlds, body):
    result = run(worlds, "faulted", body)
    assert codes(result) == "mudata_n_obs_is_the_union", body
    assert "union" in result.message and "rna:" in result.message and "atac:" in result.message
    assert "mudata_cells" in " ".join(result.fix)


def test_a_chained_read_and_other_aliases_are_seen(worlds):
    chained = "import mudata as md\nprint(md.read_h5mu('cells.h5mu').n_obs)\n"
    assert codes(run(worlds, "faulted", "", head=chained)) == "mudata_n_obs_is_the_union"
    alias = "import mudata as mu\nmdata = mu.read_h5mu('cells.h5mu')\n"
    assert (
        codes(run(worlds, "faulted", "print(mdata.n_obs)", head=alias))
        == "mudata_n_obs_is_the_union"
    )
    other = "import mudata as md\nmu = md.read('cells.h5mu')\n"
    assert (
        codes(run(worlds, "faulted", "print(mu.n_obs)", head=other)) == "mudata_n_obs_is_the_union"
    )


@pytest.mark.parametrize(
    "body",
    [
        "print(len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names)))",
        "print((m.obsm['rna'] & m.obsm['atac']).sum())",
        "print(m['rna'].obs_names.isin(m['atac'].obs_names).sum())",
        "print(len(m.mod['rna'].obs_names.intersection(m.mod['atac'].obs_names)))",
        "print(m.mod['rna'].n_obs)",
        "print(m['atac'].n_obs)",
        "print(m.mod)",
        "print(m)",
    ],
)
def test_intersections_per_modality_counts_and_inspection_pass(worlds, body):
    assert codes(run(worlds, "faulted", body)) == "ok"


@pytest.mark.parametrize("body", ["print(m.n_obs)", "print(len(m.obs_names))"])
def test_identical_modalities_cost_nothing(worlds, body):
    assert codes(run(worlds, "clean", body)) == "ok"


def test_declarations(worlds):
    count = "print(m.n_obs)"
    assert codes(run(worlds, "faulted", count, {"mudata_cells": "union"})) == "ok"
    assert (
        codes(run(worlds, "faulted", count, {"mudata_cells": "intersection"}))
        == "declared_intersection_but_union_count"
    )
    inter = "print(len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names)))"
    assert codes(run(worlds, "faulted", inter, {"mudata_cells": "intersection"})) == "ok"
    assert codes(run(worlds, "clean", count, {"mudata_cells": "intersection"})) == "ok"
    assert codes(run(worlds, "faulted", count, {"mudata_cells": "both"})) == "declaration_invalid"


def test_script_heredoc_and_unrelated_commands(worlds):
    work, _ = worlds["faulted"]
    (work / "count.py").write_text(HEAD + "print(m.n_obs)\n")
    assert codes(verdict(ENGINE, RULE, PY, ["count.py"], work)) == "mudata_n_obs_is_the_union"
    heredoc = verdict(ENGINE, RULE, PY, ["-"], work, context=(HEAD + "print(m.n_obs)\n",))
    assert codes(heredoc) == "mudata_n_obs_is_the_union"
    assert verdict(ENGINE, RULE, PY, ["-c", "print(1)"], work) is None
    assert verdict(ENGINE, RULE, "python3", ["-c", HEAD + "print(m.n_obs)"], work).verdict == "fail"
    assert verdict(ENGINE, RULE, PY, ["-c", "md.read_h5mu('nofile.h5mu').n_obs"], work) is None
    (work / "bad.h5mu").write_bytes(b"not hdf5")
    bad = "import mudata as md\nm = md.read_h5mu('bad.h5mu')\nprint(m.n_obs)\n"
    assert codes(verdict(ENGINE, RULE, PY, ["-c", bad], work)) == "ok"


def test_without_h5py_it_only_warns(worlds, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "h5py", None)
    result = run(worlds, "faulted", "print(m.n_obs)")
    assert result.verdict == "unknown" and result.code == "h5py_missing"


# ---- the regexes stay linear (E-7) -------------------------------------------------------------

# the forms before E-7: unanchored and unbounded
UNBOUNDED = re.compile(r"(\w+)\s*=\s*[\w.]*\b(?:read_h5mu|MuData|read)\s*\(")
OLD_INTERSECTS = re.compile(
    r"intersect|&|\bobsm\b|\.isin\(|\.mod\s*\[|\bmod\s*\[[^\]]+\]"
    r"|\[\s*['\"][\w-]+['\"]\s*\]\s*\.\w*obs"
)
OLD_CHAINED = re.compile(r"read_h5mu\s*\([^)]*\)\s*\.(?:n_obs|obs_names|shape|obs)\b[^\n;]*")

CODE = [
    "m = md.read_h5mu('cells.h5mu')",
    "mdata = mu.read_h5mu('cells.h5mu')",
    "mu = md.read('cells.h5mu')",
    "x=MuData({'rna': a, 'atac': b})",
    "self.m  =  mu.MuData(",
    "m = read_h5mu('c.h5mu')",
    "a1b2 = some.module.read_h5mu(",
    "data = pd.read_csv('x.csv')\nm = md.read_h5mu('c.h5mu')\nprint(m.n_obs)",
    "a = b = md.read_h5mu('c.h5mu')",
    "m = mu.read_h5mu ('c.h5mu')",
    "ids = 'ACGTACGT'\nm = md.read_h5mu('c.h5mu')\nk = 1\nn = np.read(",
    "no assignment here: read_h5mu('c.h5mu')",
    "print(md.read_h5mu('cells.h5mu', backed='r').n_obs)",
    "print(len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names)))",
    "print(m['rna'].obs_names.isin(m['atac'].obs_names).sum())",
    "print(m.obsm['rna'], mod['atac'].n_obs, a['x'].obs)",
    "print(m.mod)",
]


@pytest.mark.parametrize("text", CODE)
def test_the_bounded_regexes_find_what_the_unbounded_ones_found(text):
    assert mm.MAKES_MUDATA.findall(text) == UNBOUNDED.findall(text)
    assert bool(mm.INTERSECTS.search(text)) == bool(OLD_INTERSECTS.search(text))
    new, old = mm.CHAINED.search(text), OLD_CHAINED.search(text)
    assert (new and new.group(0)) == (old and old.group(0))


ADVERSARIAL = {
    "a line of letters (more than 12 s before E-7)": "a" * 40_000,
    "an assignment of a line of letters": "x = " + "a" * 40_000,
    "assignments in a chain": "a=" * 20_000,
    "a run of blanks": "a" + " " * 40_000 + "b",
    "a dotted path": "x = " + "a." * 20_000,
    "many names": "".join(f"v{i}=" for i in range(8_000)),
    "many reads of one name": "m=read(" * 5_000 + "\nm.n_obs",
    "many distinct handles": "".join(f"v{i}=read(\n" for i in range(20_000)),
    "calls that never close": "read_h5mu(" * 20_000,
    "indexes that never close": "mod[" * 50_000,
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_long_lines_cost_a_scan_not_a_quadratic_search(name):
    text = ADVERSARIAL[name]
    started = time.perf_counter()
    mm.counts_mudata_cells(text)
    mm.INTERSECTS.search(text)
    assert time.perf_counter() - started < 2.0, name  # each of these took 3 s to 30 s before


def test_only_the_first_handles_are_followed_up_and_they_are_found_in_order():
    names = [f"h{i}" for i in range(mm.MAX_HANDLES)]
    reads = "".join(f"{n} = md.read_h5mu('{n}.h5mu')\n" for n in names)
    assert mm.counts_mudata_cells(reads + "print(h0.n_obs)") == "h0.n_obs)"
    assert mm.counts_mudata_cells(reads + f"print({names[-1]}.n_obs)") == f"{names[-1]}.n_obs)"
    beyond = reads + "late = md.read_h5mu('late.h5mu')\nprint(late.n_obs)"
    assert mm.counts_mudata_cells(beyond) is None  # past the cap: not followed up, never a block


def test_a_script_with_a_40_kb_line_still_gets_its_verdict_quickly(worlds):
    work, _ = worlds["faulted"]
    (work / "long.py").write_text(HEAD + "ids = '" + "a" * 40_000 + "'\nprint(m.n_obs)\n")
    started = time.perf_counter()
    result = verdict(ENGINE, RULE, PY, ["long.py"], work)
    assert time.perf_counter() - started < 2.0
    assert codes(result) == "mudata_n_obs_is_the_union"
