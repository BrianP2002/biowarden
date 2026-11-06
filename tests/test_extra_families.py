"""Generic validity gates for every registered extra task family (F5 and up)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.reference import solve
from helpers import gate_seeds

SEEDS = gate_seeds("0-4")
ROOT = Path(__file__).resolve().parents[1]
SPECS = load()


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def make(tmp_path, spec, seed, twin, tools, tier="explicit", route="free"):
    work = tmp_path / f"{spec.id}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(spec.id, seed, twin, work, tools, tier, route)


pytestmark = pytest.mark.skipif(not SPECS, reason="no extra families registered yet")
ids = sorted(SPECS, key=lambda i: int(i[1:]))


@pytest.mark.parametrize("family", ids)
@pytest.mark.parametrize("twin", ["clean", "faulted"])
def test_gate1_reference_passes_every_twin_and_tier(tmp_path, tools, family, twin):
    spec = SPECS[family]
    for tier in spec.tiers:
        for seed in SEEDS:
            work, task = make(tmp_path, spec, seed, twin, tools, tier)
            score = score_result(task.truth, solve(task, work, tools, reference=True))
            assert score.outcome == "pass", (task.task_id, score)


@pytest.mark.parametrize("family", ids)
def test_gate2_naive_passes_clean_and_falls_into_the_recorded_trap(tmp_path, tools, family):
    spec = SPECS[family]
    for tier in spec.tiers:
        for seed in SEEDS:
            work, task = make(tmp_path, spec, seed, "clean", tools, tier)
            assert (
                score_result(task.truth, solve(task, work, tools, reference=False)).outcome
                == "pass"
            )
            work, task = make(tmp_path, spec, seed, "faulted", tools, tier)
            naive = solve(task, work, tools, reference=False)
            score = score_result(task.truth, naive)
            assert score.outcome == "silent_wrong", (task.task_id, score)
            recorded = {**task.truth, "expected": task.truth["naive"]}
            assert answer_matches(recorded, naive["answer"]), task.task_id  # truth["naive"] is real
            assert score.signature is True


@pytest.mark.parametrize("family", ids)
def test_gate3_twins_share_prompt_truth_and_differ_only_in_declared_files(tmp_path, tools, family):
    spec = SPECS[family]
    for tier in spec.tiers:
        work_c, clean = make(tmp_path, spec, 2, "clean", tools, tier)
        work_f, faulted = make(tmp_path, spec, 2, "faulted", tools, tier)
        assert clean.prompt == faulted.prompt and clean.slot == faulted.slot == spec.slot
        assert clean.truth["expected"] == faulted.truth["expected"]
        assert faulted.evidence and clean.evidence != faulted.evidence
        files_c = {p.name for p in work_c.iterdir()}
        files_f = {p.name for p in work_f.iterdir()}
        changed = {
            n for n in files_c & files_f if (work_c / n).read_bytes() != (work_f / n).read_bytes()
        }
        assert (changed | (files_c ^ files_f)) <= spec.twin_files, (family, tier)


@pytest.mark.parametrize("family", ids)
def test_gate4_forced_route_changes_only_the_prompt_and_names_a_tool(tmp_path, tools, family):
    spec = SPECS[family]
    if "forced" not in spec.routes:
        pytest.skip("no forced route for this family")
    _, free = make(tmp_path, spec, 1, "faulted", tools, "explicit", "free")
    _, forced = make(tmp_path, spec, 1, "faulted", tools, "explicit", "forced")
    assert forced.prompt != free.prompt and forced.truth == free.truth
    assert forced.evidence == free.evidence and forced.task_id.endswith("-forced")


@pytest.mark.parametrize("family", ids)
def test_gate5_truth_stays_out_of_the_workspace_and_builds_are_reproducible(
    tmp_path, tools, family
):
    spec = SPECS[family]
    work, task = make(tmp_path, spec, 3, "faulted", tools)
    text = "\n".join(
        p.read_text(errors="ignore")
        for p in work.iterdir()
        if p.suffix not in {".gz", ".csi", ".bam", ".bai", ".fai"}
    )
    assert "expected" not in text and "truth" not in text.lower()
    assert not (work / "result.json").exists()
    again, _ = make(tmp_path / "again", spec, 3, "faulted", tools)
    for p in work.iterdir():
        assert p.read_bytes() == (again / p.name).read_bytes(), p.name


@pytest.mark.parametrize("family", ids)
def test_gate6_exposure_check_sees_the_naive_route_and_not_an_unrelated_command(
    tmp_path, tools, family
):
    spec = SPECS[family]
    work, task = make(tmp_path, spec, 1, "faulted", tools)
    assert spec.exposed(["ls -la; cat README.md"], work) is False
    assert spec.exposed([], work) is False


@pytest.mark.parametrize("family", ids)
def test_gate_locus_every_family_names_where_its_error_lives(family):
    from biowarden.bench.tasks.locus import LOCI, locus_of

    assert locus_of(family) in LOCI
