"""Rule structure.chain_order: the checker on real `mockfold predict` argv, the tool facts it relies
on, the engine / precheck / shim path, and admission.

The unit tests write small request files, run the real mockfold fixture (the same module the
console script wraps) and compare the rule's verdict with what the tool really wrote. The
admission test measures the rule on calls labelled by running the tool
(bench/admission/cases_structure_chain_order.py).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import known_sources, measure
from biowarden.bench.tasks import ToolEnv
from biowarden.contracts import Entity, Request
from biowarden.contracts import load_packaged as load_v0_contract
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.install import install_shims
from biowarden.engine.precheck import precheck
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.rules import structure_chain_order as sco
from biowarden.engine.server import EngineServer, PathMap, serve_in_thread
from biowarden.engine.spec import load_packaged as load_spec
from biowarden.oracles.structure import parse_output
from biowarden.validators import validate_request
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, ROLES, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE_ID = "structure.chain_order"
KEY = "chain_pairing"
MODULE = "biowarden.bench.fixtures.mockfold"
SPEC = load_spec("structure_chain_order")


def engine_for(spec: Spec = SPEC) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def rows(roles: str, labels: bool = True) -> list[dict[str, str]]:
    return [
        {"role": r, **({"label": ROLES[r][0]} if labels else {}), "sequence": ROLES[r][1]}
        for r in roles
    ]


def write_request(path: Path, roles: str, labels: bool = True) -> Path:
    path.write_text(json.dumps({"entities": rows(roles, labels)}))
    return path


def predict(
    request: str = "req.json", backend: str = "tfoldlike", out: str = "out", *extra: str
) -> list[str]:
    return ["predict", "--backend", backend, "--request", request, "--out-dir", out, *extra]


def run_real(cwd: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-I", "-m", MODULE, *argv]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def written_chains(out_dir: Path) -> str:
    return "".join(chain.chain_id for chain in parse_output(out_dir / "complex.pdb").chains)


def outcome(engine: Engine, argv: list[str], cwd: Path, declared=None, tool: str = "mockfold"):
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == RULE_ID]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def verdict_of(argv: list[str], cwd: Path, declared=None) -> str:
    result = outcome(engine_for(), argv, cwd, declared)[1]
    assert result is not None
    return result.verdict


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_declares_an_optional_two_valued_key():
    (rule,) = SPEC.rules
    assert SPEC.name == "structure_chain_order" and rule.id == RULE_ID
    assert (rule.slot, rule.kind, rule.tier) == ("chain_order", "call", "T1")
    assert rule.incidents == ["T01"]
    assert rule.applies.tools == ["mockfold"] and rule.applies.subcommands == ["predict"]
    assert rule.declare is not None
    assert (rule.declare.key, rule.declare.values) == (KEY, ["by_request_position", "by_role"])
    assert rule.declare.required is False  # conditional: only an order that differs asks for it
    engine_for()  # building the engine imports the rule modules and checks the checker name
    assert rule.check in known_checkers()
    schema = SPEC.declaration_schema()
    assert schema["properties"][KEY]["enum"] == ["by_request_position", "by_role"]
    assert schema["required"] == []


def test_spec_rejects_a_wrong_shape():
    data = SPEC.model_dump(mode="json")
    data["rules"][0]["applies"]["tools"] = []
    with pytest.raises(ValidationError, match="needs applies.tools"):
        Spec.model_validate(data)
    data = SPEC.model_dump(mode="json")
    data["rules"][0]["unexpected"] = 1
    with pytest.raises(ValidationError):
        Spec.model_validate(data)


def test_the_order_comes_from_the_v0_contract_data_not_from_v0_logic():
    tfold = sco.backend_slot("tfoldlike")
    assert tfold is not None and tfold.tool_policy == "declared_by_tool" and tfold.key == "role"
    assert tfold.order == ["B", "A", "D", "G", "M", "N", "P"]
    boltz = sco.backend_slot("boltzlike")
    assert boltz is not None and boltz.tool_policy == "input_order"
    assert sco.backend_slot("nonsense") is None
    source = Path(sco.__file__).read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+(biowarden\.\S+)", source, re.M)
    assert imports and not any(
        name.startswith(("biowarden.validators", "biowarden.oracles")) for name in imports
    )
    assert "biowarden.contracts.loader" in imports  # the plain data loader


# ---- the tool facts the rule relies on (the module docstring quotes these) ---------------------


def test_tfoldlike_assembles_in_role_order_and_boltzlike_keeps_the_request_order(tmp_path):
    for roles, tfold, boltz in (("ABMNP", "BAMNP", "ABCDE"), ("PNMGD", "DGMNP", "ABCDE")):
        write_request(tmp_path / "req.json", roles)
        for backend, expected in (("tfoldlike", tfold), ("boltzlike", boltz)):
            proc = run_real(tmp_path, predict("req.json", backend, f"out-{backend}"))
            assert proc.returncode == 0 and proc.stderr == "", proc.stderr
            assert written_chains(tmp_path / f"out-{backend}") == expected


STRUCTURAL = {  # role sets the tool refuses with exit 2 (the rule leaves them to it)
    "duplicate_role": "BBAMNP",
    "missing_n": "BAMP",
    "missing_m": "BANP",
    "missing_p": "BAMN",
    "b_without_a": "BMNP",
    "d_without_g": "DMNP",
    "both_pairs": "BADGMNP",
    "no_tcr": "MNP",
}


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"entities": []}', "[]", '{"entities": ["B", "A"]}', ""],
)
def test_malformed_request_files_are_refused_loudly_by_the_tool(tmp_path, content):
    (tmp_path / "req.json").write_text(content)
    proc = run_real(tmp_path, predict())
    # exit 2 with an error line; a JSON list (not an object) even ends in a traceback and exit 1
    assert proc.returncode in (1, 2) and proc.stderr.strip() and proc.stdout == ""
    assert not (tmp_path / "out" / "complex.pdb").exists()


def test_the_two_faults_let_requests_through_that_the_tool_otherwise_refuses(tmp_path):
    write_request(tmp_path / "no_n.json", "BAMP")
    write_request(tmp_path / "both.json", "BADGMNP")
    assert run_real(tmp_path, predict("no_n.json", "tfoldlike", "o1")).returncode == 2
    proc = run_real(
        tmp_path, predict("no_n.json", "tfoldlike", "o1", "--fault", "silent_missing_n")
    )
    assert proc.returncode == 0 and written_chains(tmp_path / "o1") == "BAMP"
    assert run_real(tmp_path, predict("both.json", "tfoldlike", "o2")).returncode == 2
    proc = run_real(
        tmp_path, predict("both.json", "tfoldlike", "o2", "--fault", "dg_overwrites_ba")
    )
    assert proc.returncode == 0 and written_chains(tmp_path / "o2") == "DGMNP"


# ---- verdicts ----------------------------------------------------------------------------------


def test_a_request_in_the_backend_order_passes_silently(tmp_path):
    for roles in ("BAMNP", "DGMNP"):
        write_request(tmp_path / "req.json", roles)
        decision, result = outcome(engine_for(), predict(), tmp_path)
        assert decision.action == "allow" and result is not None and result.verdict == "pass"


def test_swapped_tcr_chains_fail_with_the_reordered_entity_list_first(tmp_path):
    write_request(tmp_path / "req.json", "ABMNP")
    decision, result = outcome(engine_for(), predict(), tmp_path)

    assert decision.action == "block"
    assert result is not None and result.verdict == "fail" and result.code == "order_mismatch"
    assert result.expected == list("BAMNP") and result.got == list("ABMNP")
    assert result.fix[0] == (
        "reorder the entities of req.json to B (tcr_beta), A (tcr_alpha), M (mhc), N (b2m), "
        "P (peptide); or"
    )
    assert "--declare chain_pairing=by_role" in result.fix[1]
    for fact in ("tfoldlike", "B,A,M,N,P", "req.json", "A,B,M,N,P", "2 of 5"):
        assert fact in result.message
    text = render_block(decision)
    assert text.startswith("FIX: reorder the entities of req.json to B (tcr_beta), A (tcr_alpha)")
    assert "structure.chain_order [chain_order/T1]" in text


def test_the_fix_names_roles_alone_when_the_entities_carry_no_label(tmp_path):
    write_request(tmp_path / "req.json", "PNMGD", labels=False)
    result = outcome(engine_for(), predict(), tmp_path)[1]
    assert result is not None and result.fix[0].startswith("reorder the entities of req.json to ")
    assert "D, G, M, N, P; or" in result.fix[0] and "(" not in result.fix[0]


def test_by_role_is_safe_and_by_request_position_is_not(tmp_path):
    write_request(tmp_path / "req.json", "ABMNP")
    engine = engine_for()
    _, by_role = outcome(engine, predict(), tmp_path, {KEY: "by_role"})
    assert by_role is not None and by_role.verdict == "pass"
    decision, by_position = outcome(engine, predict(), tmp_path, {KEY: "by_request_position"})
    assert by_position is not None and by_position.verdict == "fail"
    assert by_position.code == "declared_mismatch" and decision.action == "block"
    assert "you declared chain_pairing=by_request_position" in by_position.message
    assert by_position.fix[0].startswith("reorder the entities of req.json to B (tcr_beta)")


def test_a_declaration_changes_nothing_when_the_order_already_matches(tmp_path):
    write_request(tmp_path / "req.json", "BAMNP")
    for declared in ({}, {KEY: "by_role"}, {KEY: "by_request_position"}):
        assert verdict_of(predict(), tmp_path, declared) == "pass"


def test_an_invalid_declaration_is_refused_by_the_engine(tmp_path):
    write_request(tmp_path / "req.json", "BAMNP")
    _, result = outcome(engine_for(), predict(), tmp_path, {KEY: "by_position"})
    assert result is not None and result.code == "declaration_invalid"


def test_the_rule_agrees_with_the_tool_on_every_ordering_of_each_panel(tmp_path):
    """For every permutation: pass exactly when output chain k holds request entity k."""
    jobs = [
        (panel, "".join(p)) for panel in ("BAMNP", "DGMNP") for p in itertools.permutations(panel)
    ]

    def run_one(job):
        panel, roles = job
        work = tmp_path / f"{panel}-{roles}"
        work.mkdir()
        write_request(work / "req.json", roles)
        proc = run_real(work, predict())
        return panel, roles, work, proc.returncode

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(run_one, jobs))
    engine = engine_for()
    assert len(results) == 240
    for panel, roles, work, code in results:
        assert code == 0
        chains = written_chains(work / "out")
        result = outcome(engine, predict(), work)[1]
        assert result is not None
        assert (result.verdict == "pass") == (chains == roles), roles
        assert (result.verdict == "pass") == (roles == panel), roles
        if result.verdict == "fail":
            assert "".join(result.expected) == chains  # what the rule expects is what was written


def test_the_assembled_order_and_the_fix_equal_the_v0_proposed_fix(tmp_path):
    contract = load_v0_contract("mockfold_tfoldlike")
    for panel in ("BAMNP", "DGMNP"):
        for permutation in itertools.permutations(panel):
            roles = "".join(permutation)
            write_request(tmp_path / "req.json", roles)
            result = outcome(engine_for(), predict(), tmp_path)[1]
            request = Request(
                entities=[Entity(role=r, label=ROLES[r][0], sequence=ROLES[r][1]) for r in roles]
            )
            v0 = validate_request(contract, request)
            assert (result.verdict == "pass") == v0.ok
            if not v0.ok:
                assert v0.proposed_fix is not None
                assert result.expected == [e.role for e in v0.proposed_fix.entities]


# ---- option spellings --------------------------------------------------------------------------

PREFIX_PAIRS = list(zip(sco.spellings("--backend"), sco.spellings("--request"), strict=True))


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize(("backend_opt", "request_opt"), PREFIX_PAIRS)
def test_every_abbreviation_of_the_options_is_read_like_the_tool_reads_it(
    tmp_path, backend_opt, request_opt, attached
):
    write_request(tmp_path / "bad.json", "ABMNP")
    write_request(tmp_path / "good.json", "BAMNP")

    def argv(request: str) -> list[str]:
        if attached:
            return ["predict", f"{backend_opt}=tfoldlike", f"{request_opt}={request}", "--o=out"]
        return ["predict", request_opt, request, backend_opt, "tfoldlike", "--out", "out"]

    proc = run_real(tmp_path, argv("bad.json"))
    assert proc.returncode == 0 and written_chains(tmp_path / "out") == "BAMNP"  # tool accepted it
    assert verdict_of(argv("bad.json"), tmp_path) == "fail"
    assert verdict_of(argv("good.json"), tmp_path) == "pass"


def test_path_forms_resolve_against_the_call_directory(tmp_path, monkeypatch):
    work = tmp_path / "ws"
    work.mkdir()
    write_request(work / "req.json", "ABMNP")
    monkeypatch.chdir(tmp_path)  # not the call's directory
    for request in ("req.json", "./req.json", str(work / "req.json")):
        assert verdict_of(predict(request), work) == "fail"
    assert verdict_of(predict("../ws/req.json"), work) == "fail"


def test_a_repeated_option_means_its_last_value_as_it_does_for_the_tool(tmp_path):
    write_request(tmp_path / "bad.json", "ABMNP")
    write_request(tmp_path / "good.json", "BAMNP")
    cases = [
        (["--request", "good.json", "--request", "bad.json"], "fail"),
        (["--request", "bad.json", "--request", "good.json"], "pass"),
        (["--request=bad.json", "--req", "good.json"], "pass"),
    ]
    for tail, expected in cases:
        argv = ["predict", "--backend", "tfoldlike", *tail, "--out-dir", "out"]
        assert verdict_of(argv, tmp_path) == expected, tail
        proc = run_real(tmp_path, argv)
        assert proc.returncode == 0
    both = ["predict", "--request", "bad.json", "--out-dir", "out"]
    assert (
        verdict_of([*both, "--backend", "boltzlike", "--backend", "tfoldlike"], tmp_path) == "fail"
    )
    assert (
        verdict_of([*both, "--backend", "tfoldlike", "--backend", "boltzlike"], tmp_path) == "pass"
    )
    # the tool agrees: the last --backend decides how the chains are named
    proc = run_real(tmp_path, [*both, "--backend", "boltzlike", "--backend", "tfoldlike"])
    assert proc.returncode == 0 and written_chains(tmp_path / "out") == "BAMNP"


def test_calls_the_rule_does_not_concern_are_not_matched_or_pass(tmp_path):
    write_request(tmp_path / "req.json", "ABMNP")
    engine = engine_for()
    assert outcome(engine, ["--version"], tmp_path)[1] is None
    assert outcome(engine, ["other", "--backend", "tfoldlike"], tmp_path)[1] is None
    assert outcome(engine, predict(), tmp_path, tool="python3")[1] is None
    assert outcome(engine, predict(), tmp_path, tool="boltz")[1] is None
    for argv in (
        ["predict", "-h"],
        ["predict", "--help"],
        ["predict", "--he"],
        ["predict", "--backend", "tfoldlike", "--out-dir", "out"],  # no --request: usage error
        ["predict", "--request", "req.json", "--out-dir", "out"],  # no --backend: usage error
    ):
        result = outcome(engine, argv, tmp_path)[1]
        assert result is not None and result.verdict == "pass", argv
    # the tool really does stop on those with no prediction
    assert (
        run_real(tmp_path, ["predict", "--request", "req.json", "--out-dir", "o"]).returncode == 2
    )
    assert not (tmp_path / "o").exists()


# ---- an order-preserving backend, declarations that settle it, tier honesty --------------------


def test_boltzlike_keeps_any_order_and_the_file_is_not_even_read(tmp_path):
    match = Match(SPEC.rules[0])
    check = known_checkers()["structure_chain_order"]
    t0 = Ctx(tmp_path, "T0")  # a T0 context raises if the checker touches a file
    for argv in (predict("req.json", "boltzlike"), predict("missing.json", "boltzlike")):
        for declared in ({}, {KEY: "by_request_position"}):
            result = check(Call("mockfold", tuple(argv), tmp_path, declared), match, t0)
            assert result.verdict == "pass" and "keeps the request order" in result.note
    write_request(tmp_path / "req.json", "PNMGD")
    assert verdict_of(predict("req.json", "boltzlike"), tmp_path) == "pass"


def test_declaring_by_role_settles_it_without_reading_the_file(tmp_path):
    check = known_checkers()["structure_chain_order"]
    call = Call("mockfold", tuple(predict("not-written-yet.json")), tmp_path, {KEY: "by_role"})
    assert check(call, Match(SPEC.rules[0]), Ctx(tmp_path, "T0")).verdict == "pass"


def test_the_checker_reads_a_file_so_it_is_tier_t1_and_a_t0_context_refuses(tmp_path):
    write_request(tmp_path / "req.json", "ABMNP")
    check = known_checkers()["structure_chain_order"]
    call = Call("mockfold", tuple(predict()), tmp_path)
    match = Match(SPEC.rules[0])
    with pytest.raises(TierViolation):
        check(call, match, Ctx(tmp_path, "T0"))
    assert check(call, match, Ctx(tmp_path, "T1")).verdict == "fail"
    assert SPEC.rules[0].tier == "T1"


def test_the_checker_changes_neither_the_files_nor_the_call(tmp_path):
    request = write_request(tmp_path / "req.json", "ABMNP")
    before = request.read_bytes()
    call = Call("mockfold", tuple(predict()), tmp_path, {KEY: "by_request_position"})
    snapshot = (call.tool, call.argv, dict(call.declared))
    engine_for().evaluate(call)
    assert request.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == [
        "req.json"
    ]
    assert (call.tool, call.argv, dict(call.declared)) == snapshot


# ---- a request the rule cannot read ------------------------------------------------------------


def test_a_missing_request_is_unknown_with_the_order_to_follow_and_never_blocks(tmp_path):
    decision, result = outcome(engine_for(), predict("later.json"), tmp_path)
    assert decision.action == "warn"
    assert (
        result is not None and result.verdict == "unknown" and result.code == "request_unreadable"
    )
    assert "later.json" in result.message and "No such file" in result.message
    assert "B,A,D,G,M,N,P" in result.fix[0] and "--declare chain_pairing=by_role" in result.fix[1]
    assert render_warning(decision).startswith("UNVERIFIED structure.chain_order")
    # the same call with the declaration that settles it passes
    assert verdict_of(predict("later.json"), tmp_path, {KEY: "by_role"}) == "pass"


def test_unreadable_requests_of_every_kind_are_unknown_not_crashes(tmp_path):
    (tmp_path / "notjson.json").write_text("{not json")
    (tmp_path / "empty.json").write_text("")
    (tmp_path / "list.json").write_text("[1, 2]")
    (tmp_path / "noentities.json").write_text('{"name": "x"}')
    (tmp_path / "emptyentities.json").write_text('{"entities": []}')
    (tmp_path / "isdir.json").mkdir()
    (tmp_path / "bad.gz").write_bytes(b"\x1f\x8b" + b"junk" * 8)
    (tmp_path / "deep.json").write_text("[" * 100_000)
    for name in (
        "notjson.json",
        "empty.json",
        "list.json",
        "noentities.json",
        "emptyentities.json",
        "isdir.json",
        "bad.gz",
        "deep.json",
    ):
        decision, result = outcome(engine_for(), predict(name), tmp_path)
        assert result is not None, name
        assert (result.verdict, result.code) == ("unknown", "request_unreadable"), name
        assert decision.action == "warn", name


def test_a_request_larger_than_the_read_budget_is_unknown(tmp_path):
    write_request(tmp_path / "req.json", "ABMNP")
    check = known_checkers()["structure_chain_order"]
    call = Call("mockfold", tuple(predict()), tmp_path)
    result = check(call, Match(SPEC.rules[0]), Ctx(tmp_path, "T1", max_bytes=40))
    assert result.verdict == "unknown"


def test_an_unknown_backend_is_unknown_not_a_pass(tmp_path):
    write_request(tmp_path / "req.json", "ABMNP")
    result = outcome(engine_for(), predict("req.json", "newfold"), tmp_path)[1]
    assert result is not None and (result.verdict, result.code) == ("unknown", "backend_unknown")
    assert (
        run_real(tmp_path, predict("req.json", "newfold")).returncode == 2
    )  # today the tool stops


# ---- requests the tool refuses itself ----------------------------------------------------------


@pytest.mark.parametrize(
    "entities",
    [
        [{"role": "A", "sequence": "AC"}, {"role": "X", "sequence": "AC"}],
        [{"role": "a", "sequence": "AC"}, {"role": "B", "sequence": "AC"}],
        [{"role": "", "sequence": "AC"}, {"role": "B", "sequence": "AC"}],
        [{"sequence": "AC"}, {"role": "B", "sequence": "AC"}],
        [{"role": 5, "sequence": "AC"}],
        [{"role": ["B"], "sequence": "AC"}],
        ["A", "B"],
    ],
)
def test_unknown_or_malformed_role_codes_are_left_to_the_tool(tmp_path, entities):
    (tmp_path / "req.json").write_text(json.dumps({"entities": entities}))
    result = outcome(engine_for(), predict(), tmp_path)[1]
    assert result is not None and result.verdict == "pass" and "refuses" in result.note
    assert run_real(tmp_path, predict()).returncode == 2


@pytest.mark.parametrize("name", sorted(STRUCTURAL))
def test_requests_missing_roles_or_holding_both_pairs_are_left_to_the_tool(tmp_path, name):
    scrambled = STRUCTURAL[name][::-1]  # out of order too: the tool refuses first, nothing to fix
    write_request(tmp_path / "req.json", scrambled)
    result = outcome(engine_for(), predict(), tmp_path)[1]
    assert result is not None and result.verdict == "pass" and "refuses" in result.note
    assert run_real(tmp_path, predict()).returncode == 2


# ---- the two documented faults -------------------------------------------------------------------


def test_silent_missing_n_makes_a_request_without_n_subject_to_the_order(tmp_path):
    fault = ["--fault", "silent_missing_n"]
    for roles, expected in (("BAMP", "pass"), ("ABMP", "fail"), ("MPBA", "fail"), ("DGMP", "pass")):
        write_request(tmp_path / "req.json", roles)
        assert verdict_of(predict("req.json", "tfoldlike", "o", *fault), tmp_path) == expected
        proc = run_real(tmp_path, predict("req.json", "tfoldlike", "o", *fault))
        assert proc.returncode == 0
        assert (written_chains(tmp_path / "o") == roles) == (expected == "pass")
    # without the fault the tool refuses a request without N, in any order
    write_request(tmp_path / "req.json", "ABMP")
    assert verdict_of(predict(), tmp_path) == "pass"
    assert run_real(tmp_path, predict()).returncode == 2
    assert verdict_of(
        predict("req.json", "tfoldlike", "o", "--fau=silent_missing_n"), tmp_path
    ) == ("fail")


def test_dg_overwrites_ba_drops_a_pair_which_is_not_an_order_problem(tmp_path):
    fault = ["--fault", "dg_overwrites_ba"]
    write_request(tmp_path / "both.json", "ABGDMNP")
    result = outcome(engine_for(), predict("both.json", "tfoldlike", "o", *fault), tmp_path)[1]
    assert result is not None and (result.verdict, result.code) == ("unknown", "pair_dropped")
    # the tool really writes five chains for seven entities
    assert run_real(tmp_path, predict("both.json", "tfoldlike", "o", *fault)).returncode == 0
    assert written_chains(tmp_path / "o") == "DGMNP"
    # without the fault the tool refuses both pairs; with one pair the fault changes nothing
    assert verdict_of(predict("both.json"), tmp_path) == "pass"
    write_request(tmp_path / "one.json", "ABMNP")
    assert verdict_of(predict("one.json", "tfoldlike", "o", *fault), tmp_path) == "fail"


# ---- the engine front ends ---------------------------------------------------------------------


def test_the_command_precheck_blocks_the_whole_command_and_warns_for_a_file_made_on_the_way(
    tmp_path,
):
    write_request(tmp_path / "req.json", "ABMNP")
    engine = engine_for()
    command = "mockfold predict --backend tfoldlike --request req.json --out-dir out"
    blocked = precheck(command, tmp_path, engine)
    assert blocked.action == "block" and blocked.text.startswith("FIX: reorder the entities")
    declared = precheck(f"BIOWARDEN_DECLARE='{KEY}=by_role' {command}", tmp_path, engine)
    assert declared.action == "allow"
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    made = precheck(
        "printf '{}' > new.json && mockfold predict --backend tfoldlike --request new.json "
        "--out-dir out",
        fresh,
        engine,
    )
    assert made.action == "warn" and "UNVERIFIED structure.chain_order" in made.text


@pytest.fixture
def shim_rig(tmp_path):
    """A shim named mockfold in front of the real console script, behind an engine server."""
    candidates = [Path(sys.executable).parent / "mockfold", Path(shutil.which("mockfold") or "")]
    real = next((c for c in candidates if c.is_file()), None)
    if real is None or not Path("/usr/bin/python3").exists():
        pytest.skip("the mockfold console script or /usr/bin/python3 is not available")
    shims = tmp_path / "shims"
    install_shims(shims, [real.parent], names=("mockfold",))
    work = tmp_path / "ws"
    work.mkdir()
    sock = Path(f"/tmp/biowarden-chain-order-{os.getpid()}-{tmp_path.name[-6:]}.sock")
    server = EngineServer(sock, engine_for(), PathMap(work, "/sandbox-ws"))
    serve_in_thread(server)

    def run(*args: str):
        env = {
            "PATH": "/usr/bin:/bin",
            "BIOWARDEN_SOCK": str(sock),
            "BIOWARDEN_REAL_PATH": str(real.parent),
        }
        return subprocess.run(
            [str(shims / "mockfold"), *args], cwd=work, env=env, capture_output=True, text=True
        )

    yield work, run
    server.shutdown()
    server.server_close()
    sock.unlink(missing_ok=True)


def test_the_shim_blocks_a_mispairing_call_and_lets_a_declared_or_ordered_one_run(shim_rig):
    work, run = shim_rig
    write_request(work / "bad.json", "ABMNP")
    write_request(work / "good.json", "BAMNP")

    blocked = run(*predict("bad.json", "tfoldlike", "o-bad"))
    assert blocked.returncode == 97 and not (work / "o-bad").exists()
    assert blocked.stderr.startswith("FIX: reorder the entities of bad.json to B (tcr_beta)")

    ordered = run(*predict("good.json", "tfoldlike", "o-good"))
    assert ordered.returncode == 0 and ordered.stderr == ""
    assert written_chains(work / "o-good") == "BAMNP"

    # the marker is stripped before the real tool sees the arguments (its parser would refuse it)
    declared = run(*predict("bad.json", "tfoldlike", "o-role"), "--declare", f"{KEY}=by_role")
    assert declared.returncode == 0 and declared.stderr == ""
    assert written_chains(work / "o-role") == "BAMNP"

    positional = run(
        *predict("bad.json", "tfoldlike", "o-pos"), f"--declare={KEY}=by_request_position"
    )
    assert positional.returncode == 97 and not (work / "o-pos").exists()


# ---- the cases module --------------------------------------------------------------------------


def test_the_case_source_is_registered_and_its_labels_come_from_the_written_pdb(tmp_path):
    source = known_sources()[RULE_ID]
    tools = ToolEnv.discover(ROOT) or ToolEnv(())
    checked = 0
    for seed in range(3):
        cases = source(seed, tmp_path / f"s{seed}", tools)
        assert {(c.label, c.declared) for c in cases} == {
            ("right", True),
            ("right", False),
            ("wrong", True),
            ("wrong", False),
        }
        for case in cases:
            if case.declared:  # the label of by_request_position equals the undeclared one
                continue
            proc = run_real(case.call.cwd, list(case.call.argv))
            if proc.returncode != 0:
                assert case.label == "right", case.note  # loud failure: nothing silently wrong
                continue
            request = case.call.cwd / str(sco.last_value(case.call.argv, "--request"))
            entities = json.loads(request.read_text())["entities"]
            written = written_chains(
                case.call.cwd / str(sco.last_value(case.call.argv, "--out-dir"))
            )
            if sco.last_value(case.call.argv, "--backend") == "tfoldlike":
                paired = written == "".join(e["role"] for e in entities)
            else:
                paired = written == "ABCDEFGH"[: len(entities)]
            assert (case.label == "right") == paired, case.note
            checked += 1
    assert checked > 30


# ---- admission ---------------------------------------------------------------------------------


def test_admission(tmp_path):
    tools = ToolEnv.discover(ROOT) or ToolEnv(())  # the cases run mockfold; the tools are unused
    stats = measure(RULE_ID, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)

    assert stats.conditional and stats.admitted() and stats.killed() > 0
    assert all(cell.n > 0 for cell in stats.cells.values())  # the four cells are populated
    right_declared, right_undeclared = stats.right_declared, stats.cells[("right", False)]
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    # every wrong call is a request in the wrong order, which the checker sees
    assert stats.recall() == 1.0
    # unknown: only the request file that does not exist yet, once per seed in each right cell
    assert right_declared.unknown == right_undeclared.unknown == ADMISSION_SEEDS
    assert stats.cells[("wrong", True)].unknown == stats.cells[("wrong", False)].unknown == 0
    assert promotion_is_consistent(stats)
