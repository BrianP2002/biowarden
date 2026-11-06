"""T7: the D:G overwrite and the silent missing-beta2m behaviour are caught by the contract."""

from __future__ import annotations

from pathlib import Path

from biowarden.contracts import Request, load_packaged
from biowarden.gateway import Gateway
from biowarden.oracles.structure import parse_output
from biowarden.validators import validate_request
from helpers import entities, run_mockfold

BOTH_PAIRS = "BADGMNP"  # alpha/beta and gamma/delta together, already in the contract's order
NO_B2M = "BAMP"


def codes(verdict):
    return {v.code for v in verdict.violations}


def test_t7_precall_rejects_alpha_beta_and_gamma_delta_together():
    verdict = validate_request(
        load_packaged("mockfold_tfoldlike"), Request(entities=entities(BOTH_PAIRS))
    )

    assert not verdict.ok and "group_conflict" in codes(verdict)
    assert verdict.proposed_fix is None  # there is no safe repair


def test_t7_precall_rejects_a_complex_without_beta2m():
    verdict = validate_request(
        load_packaged("mockfold_tfoldlike"), Request(entities=entities(NO_B2M))
    )

    assert not verdict.ok and codes(verdict) == {"required_missing"}
    assert verdict.proposed_fix is None


def test_t7_precall_rejects_incomplete_pairs_unknown_and_repeated_roles():
    contract = load_packaged("mockfold_tfoldlike")

    lone_beta = validate_request(contract, Request(entities=entities("BMNP")))
    unknown = validate_request(
        contract,
        Request(
            entities=[*entities("BAMNP")[:4], entities("P")[0].model_copy(update={"role": "X"})]
        ),
    )
    repeated = validate_request(
        contract,
        Request(entities=[*entities("BAMNP"), entities("P")[0].model_copy(update={"label": "p2"})]),
    )

    assert "group_incomplete" in codes(lone_beta)
    assert "unknown_key" in codes(unknown)
    assert "duplicate_key" in codes(repeated)


def test_t7_dg_overwrite_fault_is_flagged_by_record_count_and_roundtrip_oracles(tmp_path):
    gateway = Gateway(
        tmp_path / "work",
        tmp_path / "ledger.jsonl",
        precall=False,
        faults={"tfoldlike": ["dg_overwrites_ba"]},
    )

    result = gateway.predict_complex(entities(BOTH_PAIRS), "tfoldlike")

    assert result.status == "flagged"
    assert parse_output(Path(result.artifacts["pdb"])).chains[0].chain_id == "D"  # D:G won
    failed = {r.oracle: r for r in result.oracle_results if not r.ok}
    assert set(failed) == {"chain_sequence_roundtrip", "record_count"}
    [count] = [f for f in failed["record_count"].findings if f.code == "chain_count_mismatch"]
    assert count.evidence == {"expected": 7, "got": 5}


def test_t7_silent_missing_beta2m_fault_is_flagged_by_the_required_chain_check(tmp_path):
    gateway = Gateway(
        tmp_path / "work",
        tmp_path / "ledger.jsonl",
        precall=False,
        faults={"tfoldlike": ["silent_missing_n"]},
    )

    result = gateway.predict_complex(entities(NO_B2M), "tfoldlike")

    assert result.status == "flagged"
    failed = {r.oracle: r for r in result.oracle_results if not r.ok}
    assert set(failed) == {"record_count"}  # the chains that exist are all correctly paired
    [finding] = failed["record_count"].findings
    assert finding.code == "required_chain_missing" and finding.evidence["missing"] == ["N"]


def test_t7_both_faults_pass_silently_without_the_contract(tmp_path):
    for order, fault in ((BOTH_PAIRS, "dg_overwrites_ba"), (NO_B2M, "silent_missing_n")):
        proc, out = run_mockfold(tmp_path / fault, "tfoldlike", entities(order), faults=(fault,))
        assert proc.returncode == 0 and proc.stderr == ""
        assert (out / "complex.pdb").exists()


def test_t7_without_the_fault_flags_the_same_inputs_fail_loudly(tmp_path):
    for order in (BOTH_PAIRS, NO_B2M):
        proc, _ = run_mockfold(tmp_path / order, "tfoldlike", entities(order))
        assert proc.returncode == 2 and "error" in proc.stderr

    gateway = Gateway(tmp_path / "work", tmp_path / "ledger.jsonl", precall=False)
    result = gateway.predict_complex(entities(NO_B2M), "tfoldlike")
    assert result.status == "tool_error" and result.is_error
