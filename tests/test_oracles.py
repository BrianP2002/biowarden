"""T3: post-call oracles catch what the pre-call check would have blocked."""

from __future__ import annotations

from biowarden.contracts import Request, load_packaged
from biowarden.gateway import Gateway
from biowarden.oracles import run_invariants
from helpers import SWAPPED, TOOL_ORDER, entities, run_mockfold, variant


def test_t3_oracle_flags_mislabelled_output_when_validator_is_bypassed(tmp_path):
    contract = load_packaged("mockfold_tfoldlike")
    request = Request(entities=entities(SWAPPED))
    proc, out = run_mockfold(tmp_path, "tfoldlike", request.entities)
    assert proc.returncode == 0  # nothing about the run itself looks wrong

    results = {
        r.invariant_id: r
        for r in run_invariants(contract, request, out / "complex.pdb", out / "confidence.json")
    }

    assert not results["I1"].ok
    assert {f.code for f in results["I1"].findings} == {"chain_id_mismatch", "sequence_mismatch"}
    first = next(f for f in results["I1"].findings if f.code == "sequence_mismatch")
    assert first.evidence["position"] == 0
    assert first.evidence["expected_entity"] == "tcr_alpha"
    assert first.evidence["actual_matches_entity"] == "tcr_beta"  # the swap, named by hash
    assert first.evidence["expected_sha256"] != first.evidence["actual_sha256"]
    assert results["I2"].ok and results["I3"].ok


def test_t3_gateway_with_precall_ablated_returns_flagged_not_ok(tmp_path):
    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl", precall=False)

    result = gateway.predict_complex(entities(SWAPPED), "tfoldlike")

    assert result.status == "flagged" and result.is_error
    assert result.entity_map is None  # no chain map is offered for output that failed its checks
    assert result.artifacts is not None  # the files are kept for inspection
    assert [r.invariant_id for r in result.oracle_results if not r.ok] == ["I1"]
    assert result.text.startswith("FLAGGED")


def test_t3_scale_oracle_flags_plddt_outside_the_declared_scale(tmp_path):
    def declare_unit_scale(data):
        data["slots"]["metric_scale"]["plddt"] = {"min": 0.0, "max": 1.0}

    contract = variant("boltzlike", declare_unit_scale)  # boltzlike actually writes 0-100
    request = Request(entities=entities(TOOL_ORDER))
    _, out = run_mockfold(tmp_path, "boltzlike", request.entities)

    results = {
        r.invariant_id: r
        for r in run_invariants(contract, request, out / "complex.pdb", out / "confidence.json")
    }

    assert results["I1"].ok and results["I2"].ok  # structure and counts are fine
    assert not results["I3"].ok
    [finding] = results["I3"].findings
    assert finding.code == "metric_out_of_scale"
    assert finding.evidence["declared"] == [0.0, 1.0]
    assert finding.evidence["observed"]["structure_max"] > 50


def test_t3_scale_oracle_suspects_a_unit_scale_file_declared_as_percent(tmp_path):
    def declare_percent_scale(data):
        data["slots"]["metric_scale"]["plddt"] = {"min": 0.0, "max": 100.0}

    contract = variant("tfoldlike", declare_percent_scale)  # tfoldlike actually writes 0-1
    request = Request(entities=entities(TOOL_ORDER))
    _, out = run_mockfold(tmp_path, "tfoldlike", request.entities)

    results = {
        r.invariant_id: r
        for r in run_invariants(contract, request, out / "complex.pdb", out / "confidence.json")
    }

    assert results["I1"].ok and results["I2"].ok
    [finding] = results["I3"].findings
    assert finding.code == "metric_scale_suspect"  # in range, so only the suspect check can see it
    assert finding.evidence["observed"]["structure_max"] <= 1.0


def test_t3_unreadable_output_fails_every_invariant(tmp_path):
    contract = load_packaged("mockfold_tfoldlike")
    request = Request(entities=entities(TOOL_ORDER))

    results = run_invariants(contract, request, tmp_path / "missing.pdb")

    assert len(results) == len(contract.invariants)
    assert all(not r.ok and r.findings[0].code == "unreadable_output" for r in results)


def test_t3_swapped_sequences_under_correct_chain_ids_are_still_flagged(tmp_path):
    contract = load_packaged("mockfold_boltzlike")
    produced = Request(entities=entities("BAMNP"))
    claimed = Request(entities=entities("ABMNP"))  # same chain IDs A..E, two sequences swapped
    _, out = run_mockfold(tmp_path, "boltzlike", produced.entities)

    [i1, _, _] = run_invariants(contract, claimed, out / "complex.pdb", out / "confidence.json")

    assert not i1.ok
    assert {f.code for f in i1.findings} == {"sequence_mismatch"}
    assert sorted(f.evidence["position"] for f in i1.findings) == [0, 1]
