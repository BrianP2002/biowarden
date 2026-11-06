"""Every family whose answer is a mapping shows where the mapping goes in result.json (FA-7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.common import MAPPING_SHAPE
from biowarden.bench.tasks.families import all_families

ROOT = Path(__file__).resolve().parents[1]
MAPPING_KINDS = {"dict_num", "contexts"}  # the truth kinds whose answer is a JSON object
IDS = sorted(all_families(), key=lambda family: int(family[1:]))


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


@pytest.mark.parametrize("family", IDS)
def test_a_mapping_answer_shows_where_it_goes_and_no_other_answer_does(tmp_path, tools, family):
    task = build_task(family, 1000, "faulted", tmp_path / family, tools)

    assert (MAPPING_SHAPE in task.prompt) == (task.truth["kind"] in MAPPING_KINDS), task.prompt
