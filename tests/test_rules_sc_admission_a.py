"""Admission of sc rules (group A): each rule measured over ADMISSION_SEEDS seeds of its family
with calls labelled by running python-sc code in the generated workspaces (cases_sc_*.py).

Gates: no failure on correct calls whether they carry the declaration or not (these rules are
conditional: zero friction on correct data), at least one wrong call caught, recall of at least
0.4 over a deliberately mixed set of wrong calls (some are wrong for reasons the rule cannot see),
no crash, and the promotion table agrees with the numbers. The full 300-seed calibration is
`python -I -m biowarden.bench.admission`.
"""

from __future__ import annotations

import pytest
from test_rules_sc_common import check_admission, tools  # noqa: F401  (fixture)

CASES = [
    ("sc.de_raw_default", "sc_matrix_state"),
    ("sc.small_int_qc", "sc_matrix_state"),
    ("sc.double_transform", "sc_matrix_state"),
]


@pytest.mark.parametrize(("rule_id", "spec_name"), CASES)
def test_admission(rule_id, spec_name, tools, tmp_path):  # noqa: F811
    check_admission(rule_id, spec_name, tools, tmp_path)
