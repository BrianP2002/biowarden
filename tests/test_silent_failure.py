"""T1: the tFold-class failure, reproduced without any validation in the way."""

from __future__ import annotations

from biowarden.oracles.structure import parse_output
from helpers import SWAPPED, TOOL_ORDER, entities, run_mockfold


def test_t1_permuted_request_exits_zero_with_valid_pdb_but_positional_pairing_is_wrong(tmp_path):
    request = entities(SWAPPED)
    proc, out = run_mockfold(tmp_path, "tfoldlike", request)

    assert proc.returncode == 0
    assert proc.stderr == ""
    parsed = parse_output(out / "complex.pdb", out / "confidence.json")
    assert parsed.n_atoms > 0 and parsed.coords_finite
    assert [c.chain_id for c in parsed.chains] == list(TOOL_ORDER)

    # A consumer that pairs output chain k with request entity k, as it would for any other
    # predictor, silently pairs alpha with the beta chain and vice versa.
    mispaired = [
        e.label for e, c in zip(request, parsed.chains, strict=True) if e.sequence != c.sequence
    ]
    assert mispaired == ["tcr_alpha", "tcr_beta"]


def test_t1_control_boltzlike_keeps_request_order_for_the_same_input(tmp_path):
    request = entities(SWAPPED)
    proc, out = run_mockfold(tmp_path, "boltzlike", request)

    assert proc.returncode == 0
    parsed = parse_output(out / "complex.pdb", out / "confidence.json")
    assert [c.chain_id for c in parsed.chains] == list("ABCDE")
    assert [c.sequence for c in parsed.chains] == [e.sequence for e in request]


def test_t1_tfoldlike_in_tool_order_pairs_correctly(tmp_path):
    request = entities(TOOL_ORDER)
    proc, out = run_mockfold(tmp_path, "tfoldlike", request)

    parsed = parse_output(out / "complex.pdb")
    assert proc.returncode == 0
    assert [c.sequence for c in parsed.chains] == [e.sequence for e in request]


def test_t1_metric_scale_differs_between_backends(tmp_path):
    _, tfold = run_mockfold(tmp_path, "tfoldlike", entities(TOOL_ORDER))
    _, boltz = run_mockfold(tmp_path, "boltzlike", entities(TOOL_ORDER))

    assert max(parse_output(tfold / "complex.pdb").plddt) <= 1.0
    assert max(parse_output(boltz / "complex.pdb").plddt) > 50
