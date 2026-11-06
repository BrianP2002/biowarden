"""Rule struct.bfactor_scale: the checker on real argv and files, the library facts it relies on
(run with the real tools), and admission over calls labelled by running them."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f89_chain_ids import (
    Chain,
    make_residues,
    mmcif_text,
    pdb_text,
    random_sequence,
)
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.precheck import precheck
from biowarden.engine.render import render_block
from biowarden.engine.rules import struct_bfactor as sb
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "struct.bfactor_scale"
KEY = "bfactor_scale"
PY = "python-struct"
AWK70 = '$1=="ATOM" && $3=="CA" && substr($0,61,6)+0>70 {n++} END{print n+0}'
AWK07 = '$1=="ATOM" && $3=="CA" && substr($0,61,6)+0>0.7 {n++} END{print n+0}'
PY70 = "print(sum(float(l[60:66]) > 70 for l in open('{f}') if l[12:16] == ' CA '))"
PY07 = "print(sum(float(l[60:66]) > 0.7 for l in open('{f}') if l[12:16] == ' CA '))"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "python-struct" not in env.real_bins(["python-struct"]):
        pytest.skip("python-struct is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("struct_bfactor")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def write_scaled(path: Path, scale: float, flag: bool = False, cif: bool = False) -> None:
    rng = random.Random(2)
    res = make_residues(rng, random_sequence(rng, 20), 1)
    for i, r in enumerate(res):
        for a in r.atoms:
            a.bfac = float(i % 2) if flag else round((40 + 2.5 * i) * scale, 2)
    chain = Chain("A", res)
    path.write_text(mmcif_text([chain]) if cif else pdb_text([chain]))


def outcome(eng: Engine, tool: str, argv, cwd: Path, declared=None):
    decision = eng.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_valid_and_the_checker_is_registered(spec):
    (rule,) = spec.rules
    assert rule.id == RULE and rule.kind == "fact" and rule.tier == "T1"
    assert {"awk", "python-struct", "python3"} <= set(rule.applies.tools)
    assert rule.check in known_checkers() and rule.declare.key == KEY
    assert rule.declare.values == ["0_1", "0_100", "experimental_b"] and not rule.declare.required


def test_checker_cannot_read_files_below_tier_t1(spec, tmp_path):
    write_scaled(tmp_path / "p.pdb", 0.01)
    call = Call("awk", (AWK70, "p.pdb"), tmp_path)
    with pytest.raises(TierViolation):
        sb.struct_bfactor_scale(
            call, Match(spec.rules[0], (tmp_path / "p.pdb",)), Ctx(tmp_path, "T0")
        )


@pytest.mark.parametrize("fmt", ["pdb", "cif"])
def test_a_0_100_cutoff_on_a_0_1_file_fails_until_declared(spec, tmp_path, fmt):
    name = f"p.{fmt}"
    write_scaled(tmp_path / name, 0.01, cif=fmt == "cif")
    eng = engine_for(spec)
    calls = [("awk", [AWK70.replace("p.pdb", name), name])]
    if fmt == "pdb":
        calls.append((PY, ["-c", PY70.format(f=name)]))
    for tool, argv in calls:
        decision, result = outcome(eng, tool, argv, tmp_path)
        assert decision.action == "block" and result.code == "bfactor_scale_undeclared"
        text = render_block(decision)
        assert text.startswith("FIX: multiply the B-factors by 100")
        assert "selects nothing" in text and "0 to" not in text.split("BLOCKED")[0] and name in text
        assert f"--declare {KEY}=0_1" in text
        # declared 0_1 with a cutoff of 70 contradicts itself; 0_100 contradicts the data
        assert outcome(eng, tool, argv, tmp_path, {KEY: "0_1"})[1].code == (
            "cutoff_contradicts_declaration"
        )
        assert outcome(eng, tool, argv, tmp_path, {KEY: "0_100"})[1].code == "declared_mismatch"
        assert outcome(eng, tool, argv, tmp_path, {KEY: "experimental_b"})[0].action == "allow"


def test_the_opposite_mistake_and_correct_calls_stay_silent(spec, tmp_path):
    write_scaled(tmp_path / "p.pdb", 0.01)
    write_scaled(tmp_path / "q.pdb", 1.0)
    eng = engine_for(spec)
    # a 0.7 cutoff on a 0-100 file selects everything
    decision, result = outcome(eng, PY, ["-c", PY07.format(f="q.pdb")], tmp_path)
    assert result.code == "bfactor_scale_undeclared" and "selects everything" in render_block(
        decision
    )
    silent = [
        ("awk", [AWK07.replace("0.7", "0.7"), "p.pdb"], None),  # 0-1 cutoff on a 0-1 file
        ("awk", [AWK70, "q.pdb"], None),  # 0-100 cutoff on a 0-100 file
        (PY, ["-c", PY07.format(f="p.pdb")], {KEY: "0_1"}),
        (PY, ["-c", PY70.format(f="q.pdb")], {KEY: "0_100"}),
        ("awk", ['$3=="CA" {print substr($0,61,6)+0}', "p.pdb"], None),  # no cutoff
        (PY, ["-c", "b = [float(l[60:66]) for l in open('p.pdb') if l[12:16] == ' CA ']; "
              "print(sum(x * (100 if max(b) <= 1 else 1) > 70 for x in b))"], None),  # rescales
        ("awk", ['$3=="CA" && substr($0,61,6)*100>70 {n++} END{print n+0}', "p.pdb"], None),
        ("head", ["-3", "p.pdb"], None),
    ]  # fmt: skip
    for tool, argv, declared in silent:
        decision, result = outcome(eng, tool, argv, tmp_path, declared)
        assert decision.action == "allow", (tool, argv)


def test_design_flags_and_ambiguous_ranges(spec, tmp_path):
    write_scaled(tmp_path / "flag.pdb", 1.0, flag=True)
    decision, result = outcome(engine_for(spec), "awk", [AWK70, "flag.pdb"], tmp_path)
    assert result.code == "bfactor_scale_undeclared" and "design flag" in render_block(decision)
    # B-factors between 1 and 1.5 are neither scale: nothing to say
    rng = random.Random(1)
    res = make_residues(rng, random_sequence(rng, 8), 1)
    for r in res:
        for a in r.atoms:
            a.bfac = 1.2
    (tmp_path / "mid.pdb").write_text(pdb_text([Chain("A", res)]))
    assert outcome(engine_for(spec), "awk", [AWK70, "mid.pdb"], tmp_path)[0].action == "allow"
    (tmp_path / "bad.pdb").write_text("nothing\n")
    decision, result = outcome(engine_for(spec), "awk", [AWK70, "bad.pdb"], tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_the_precheck_sees_awk_stages(spec, tmp_path):
    write_scaled(tmp_path / "p.pdb", 0.01)
    result = precheck(f"awk '{AWK70}' p.pdb", tmp_path, engine_for(spec))
    assert result.action == "block" and "bfactor_scale_undeclared" not in result.text


def test_the_same_cutoff_selects_nothing_on_the_0_1_file(tools, tmp_path):
    write_scaled(tmp_path / "hi.pdb", 1.0)
    write_scaled(tmp_path / "lo.pdb", 0.01)
    counts = {}
    for name in ("hi", "lo"):
        proc = tools.run(["awk", AWK70, f"{name}.pdb"], tmp_path)
        assert proc.returncode == 0 and proc.stderr == ""
        counts[name] = int(proc.stdout)
    code = (
        "import json, prody, gemmi; "
        "b = lambda f: prody.parsePDB(f).select('calpha').getBetas(); "
        "print(json.dumps({'prody_hi': int((b('hi.pdb') > 70).sum()), "
        "'prody_lo': int((b('lo.pdb') > 70).sum()), "
        "'gemmi_lo_max': max(a.b_iso for r in gemmi.read_structure('lo.pdb')[0]['A'] for a in r)}))"
    )
    proc = tools.run([PY, "-s", "-c", code], tmp_path)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert counts["hi"] > 5 and counts["lo"] == 0
    assert out["prody_hi"] == counts["hi"] and out["prody_lo"] == 0 and out["gemmi_lo_max"] <= 1.0


def test_admission_on_calls_labelled_by_running_the_tools(spec, tools, tmp_path):
    stats = measure(RULE, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    print(
        "ADMISSION",
        stats.rule_id,
        "seeds",
        stats.seeds,
        {
            f"{k[0]}_{'declared' if k[1] else 'undeclared'}": [c.n, c.failed, c.unknown, c.crashed]
            for k, c in stats.cells.items()
        },
        "recall",
        round(stats.recall() or 0, 3),
        "killed",
        stats.killed(),
    )
    assert stats.admitted(), stats.cells
    assert stats.right_declared.n >= MIN_RIGHT_CALLS
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and (stats.recall() or 0) > 0.5
    assert all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
