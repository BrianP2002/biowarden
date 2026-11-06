"""Validity gates for the Phase-0 task families, on the real samtools / bcftools / bedtools."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import (
    FAMILIES,
    ROUTES,
    TIERS,
    TWINS,
    ToolEnv,
    build_task,
    make_instance,
)
from biowarden.bench.tasks.common import prompt_for
from biowarden.bench.tasks.instance import EDGE_KINDS, FLANK
from biowarden.bench.tasks.reference import solve
from helpers import gate_seeds

SEEDS = gate_seeds("0-5")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def workspace(tmp_path, family, seed, twin, tools, tier="explicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def test_instances_are_deterministic_and_enriched_at_region_edges():
    a, b = make_instance(3), make_instance(3)

    assert a == b and a != make_instance(4)
    assert len(a.variants) == 40
    edges = {(v.chrom, v.pos) for v in a.variants}
    for r in a.regions:
        assert r.end0 - r.start0 >= 20
    for kind in EDGE_KINDS:
        assert sum((r.chrom, kind(r)) in edges for r in a.regions) >= 3
    assert all(FLANK < v.pos <= a.contigs[v.chrom] - FLANK for v in a.variants)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("tier", TIERS)
def test_gate1_the_reference_solution_passes_every_twin(tmp_path, tools, family, twin, tier):
    for seed in SEEDS:
        work, task = workspace(tmp_path, family, seed, twin, tools, tier)

        score = score_result(task.truth, solve(task, work, tools, reference=True))

        assert score.outcome == "pass", (task.task_id, score)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("tier", TIERS)
def test_gate2_the_naive_solution_passes_clean_and_falls_into_every_faulted_twin(
    tmp_path, tools, family, tier
):
    for seed in SEEDS:
        work, task = workspace(tmp_path, family, seed, "clean", tools, tier)
        assert score_result(
            task.truth, solve(task, work, tools, reference=False)
        ).outcome == "pass" or (
            family == "F4"  # the clean F4 twin has no problem to find; the naive answer is right
        )
        work, task = workspace(tmp_path, family, seed, "faulted", tools, tier)
        score = score_result(task.truth, solve(task, work, tools, reference=False))
        assert score.outcome == "silent_wrong", (task.task_id, score)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("tier", TIERS)
def test_gate3_twins_share_biology_prompt_and_differ_only_in_the_trap(
    tmp_path, tools, family, tier
):
    work_c, clean = workspace(tmp_path, family, 2, "clean", tools, tier)
    work_f, faulted = workspace(tmp_path, family, 2, "faulted", tools, tier)

    assert clean.prompt == faulted.prompt
    assert clean.slot == faulted.slot and clean.seed == faulted.seed
    assert clean.truth["expected"] == faulted.truth["expected"]
    assert faulted.evidence and clean.evidence != faulted.evidence
    names_c = {p.name for p in work_c.iterdir()}
    names_f = {p.name for p in work_f.iterdir()}
    changed = {
        n
        for n in names_c & names_f
        if (work_c / n).read_bytes() != (work_f / n).read_bytes()
        and not n.endswith((".gz", ".csi"))
    }
    allowed = {
        "F1": {"README.md"},
        "F2": {"README.md", "regions.txt"},
        "F3": {"README.md", "regions.bed"},
        "F4": {"ref.fa", "ref.fa.fai"},
    }[family]
    assert changed | (names_c ^ names_f) <= allowed | {"regions.bed", "regions.tsv"}
    assert changed <= allowed


@pytest.mark.parametrize("family", FAMILIES)
def test_the_truth_never_appears_in_the_workspace(tmp_path, tools, family):
    work, task = workspace(tmp_path, family, 1, "faulted", tools)
    text = "\n".join(
        p.read_text(errors="ignore") for p in work.iterdir() if p.suffix not in {".gz", ".csi"}
    )

    assert "expected" not in text and "truth" not in text.lower()
    if family == "F4":
        contexts = task.truth["expected"].values()
        assert not any(c in text for c in contexts)
    assert not (work / "result.json").exists()


def test_implicit_tier_readmes_state_less_than_explicit_ones(tmp_path, tools):
    cache: dict[tuple, str] = {}

    def readme(family, twin, tier):
        if (family, twin, tier) not in cache:
            work, _ = workspace(tmp_path, family, 1, twin, tools, tier)
            cache[(family, twin, tier)] = (work / "README.md").read_text()
        return cache[(family, twin, tier)]

    assert "0-based" in readme("F1", "faulted", "explicit")
    assert "0-based" not in readme("F1", "faulted", "implicit")
    assert "BED-style" in readme("F1", "faulted", "implicit")
    assert "1-based" in readme("F2", "clean", "explicit")
    assert "1-based" not in readme("F2", "clean", "implicit")
    assert "Ensembl" in readme("F3", "faulted", "explicit")
    assert "Ensembl" not in readme("F3", "faulted", "implicit")
    assert readme("F3", "clean", "implicit") == readme("F3", "faulted", "implicit")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_recorded_trap_answer_is_what_the_real_naive_call_returns(tmp_path, tools, family):
    for seed in SEEDS:
        work, task = workspace(tmp_path, family, seed, "faulted", tools)

        naive = solve(task, work, tools, reference=False)["answer"]

        assert answer_matches({**task.truth, "expected": task.truth["naive"]}, naive), task.task_id
        assert not answer_matches(task.truth, naive)


@pytest.mark.parametrize(
    ("family", "tool"),
    [("F1", "bcftools view -R"), ("F2", "samtools faidx"), ("F3", "bedtools intersect")],
)
def test_the_forced_route_names_the_trapped_tool_and_changes_nothing_else(
    tmp_path, tools, family, tool
):
    _, free = workspace(tmp_path, family, 1, "faulted", tools, route="free")
    _, forced = workspace(tmp_path, family, 1, "faulted", tools, route="forced")

    assert tool in forced.prompt and tool not in free.prompt
    assert forced.truth == free.truth and forced.evidence == free.evidence
    assert forced.task_id.endswith("-forced") and "forced" not in free.task_id


def test_f3_instances_keep_every_variant_clear_of_every_region_edge():
    margin = 6
    for seed in range(25):
        inst = make_instance(seed, profile="no_edge")
        for v in inst.variants:
            for r in inst.regions:
                if r.chrom == v.chrom:
                    assert not r.start0 - margin <= v.pos - 1 < r.start0 + margin, (seed, v, r)
                    assert not r.end0 - margin <= v.pos - 1 < r.end0 + margin, (seed, v, r)
        assert 15 <= len(inst.in_regions()) <= 25 and len(inst.variants) == 40


def test_instance_invariants_hold_for_every_seed_the_pilots_use():
    for seed in range(30):
        inst = make_instance(seed)
        spans = sorted((r.chrom, r.start0, r.end0) for r in inst.regions)
        assert all(e - s >= 20 for _, s, e in spans)
        by_chrom = {}
        for chrom, start, end in spans:
            by_chrom.setdefault(chrom, []).append((start, end))
        for items in by_chrom.values():
            assert all(b[0] - a[1] >= 8 for a, b in zip(items, items[1:], strict=False))
        assert all(6 <= v.pos <= inst.contigs[v.chrom] - 6 for v in inst.variants)
        at_start = [
            v
            for v in inst.variants
            if any(v.chrom == r.chrom and v.pos == r.start0 for r in inst.regions)
        ]
        assert len(at_start) >= 3
        assert all(v not in inst.in_regions() for v in [x.id for x in at_start])


def test_twin_workspaces_are_byte_identical_except_for_the_planted_difference(tmp_path, tools):
    work_a, _ = workspace(tmp_path, "F1", 3, "clean", tools)
    work_b, _ = workspace(tmp_path / "again", "F1", 3, "clean", tools)
    work_c, _ = workspace(tmp_path, "F1", 3, "faulted", tools)

    assert (work_a / "variants.vcf.gz").read_bytes() == (work_b / "variants.vcf.gz").read_bytes()
    assert (work_a / "variants.vcf.gz").read_bytes() == (work_c / "variants.vcf.gz").read_bytes()
    assert (work_a / "regions.bed").read_bytes() == (work_c / "regions.tsv").read_bytes()


def test_explicit_f1_and_f2_readmes_differ_only_in_the_planted_fact(tmp_path, tools):
    clean, _ = workspace(tmp_path, "F1", 2, "clean", tools)
    faulted, _ = workspace(tmp_path, "F1", 2, "faulted", tools)

    assert (clean / "README.md").read_text().replace("regions.bed", "regions.tsv") == (
        faulted / "README.md"
    ).read_text()
    clean2, _ = workspace(tmp_path, "F2", 2, "clean", tools)
    faulted2, _ = workspace(tmp_path, "F2", 2, "faulted", tools)
    first = lambda w: (w / "README.md").read_text().splitlines()[2:]  # noqa: E731
    assert first(clean2)[0] == first(faulted2)[0]  # the ref.fa line is shared
    assert first(clean2)[1].startswith(
        "- regions.txt: target regions, one per line as chrom:start-end."
    )
    assert first(faulted2)[1].startswith(
        "- regions.txt: target regions, one per line as chrom:start-end."
    )


def test_f4_detection_tokens_are_instance_facts(tmp_path, tools):
    work, task = workspace(tmp_path, "F4", 4, "faulted", tools)
    rows = [line.split("\t") for line in (work / "ref.fa.fai").read_text().splitlines()]
    lengths = {row[0]: row[1] for row in rows}

    tokens = task.truth["detect_tokens"]

    assert lengths["chr1"] in tokens and len(tokens) >= 5
    assert task.truth["must_detect"] is True


def test_every_family_has_a_valid_locus():
    from biowarden.bench.tasks.families import all_families
    from biowarden.bench.tasks.locus import LOCI, locus_of

    for family in all_families():
        assert locus_of(family) in LOCI, family


def test_f2_error_lives_in_the_data_because_its_sentence_says_nothing_about_coordinates():
    from biowarden.bench.tasks.locus import locus_of

    assert locus_of("F2") == "data"


# --- the answer sentence: where a mapping answer goes in result.json --------------------------

# sha256 of the prompts of the five stage-1 families as recorded in the stage-1 trials (seeds
# 1000-1019, both twins): one prompt per route. Their wording must not change (comparability).
STAGE1_PROMPTS = {
    ("F1", "free"): "a85d428ef6defd7dc4257723ad3afb84cff933d98639a44f80ef159bf3d50565",
    ("F1", "forced"): "087204063285df5ddaf1459f2dcc467d68f93d0d2d3d2229d10550a150de755a",
    ("F7", "free"): "0515e26c8141950cb13280aee0acee447159b1b2e0d2947d938a8f383e3ca120",
    ("F7", "forced"): "fc7795ee951dc2dba3fd38c30e23e2afef611b17a343a488cd014810f51e4540",
    ("F10", "free"): "5ba4dbe249c828df9c452a1eadf9473d1ea44800a3ab68318bd78a787005fb7d",
    ("F10", "forced"): "7e5850a2e94a3f1c22fb91fd43bb96bdbf17df2820724a1d408879329b934898",
    ("F13", "free"): "3b4942740c49a992799b46186358a215a2ecdb8069a180cf38f69a2e4e5942e1",
    ("F13", "forced"): "bf2903ab1ab9f6ad7d5784c4da5d2c23b02b206392a527e1b27957471f98826e",
    ("F15", "free"): "99489bc4a99ef4ab1a1bc0aae86e4df2cc697883f163ec15e8030196c330038d",
    ("F15", "forced"): "35e0af2b12c667c246ee9b3ca3dbef4f34605a1c4158eb1bed7bdc6ea12eff35",
}
WRAPPER = ', as in {"answer": {<key>: <value>, ...}}.'


@pytest.mark.parametrize(("family", "route"), sorted(STAGE1_PROMPTS))
def test_the_stage1_families_keep_their_stage1_prompts_byte_for_byte(
    tmp_path, tools, family, route
):
    tier = "explicit" if route == "free" else "implicit"  # the two stage-1 cells
    _, task = workspace(tmp_path, family, 1020, "faulted", tools, tier, route)

    digest = hashlib.sha256(task.prompt.encode()).hexdigest()

    assert digest == STAGE1_PROMPTS[(family, route)], task.prompt


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "answer",
    [
        "a JSON integer",
        "a JSON number (the mean coverage per bin)",
        "a JSON list of variant IDs",
        "a JSON list of integers, one per region, in the order of the regions file",
    ],
)
def test_prompt_for_keeps_the_stage1_sentence_for_every_answer_that_is_not_a_mapping(route, answer):
    prompt = prompt_for(route, "the free sentence", "the forced sentence", answer)

    assert prompt == (
        f"Task: the {route} sentence (see README.md for the files). "
        f"In result.json, `answer` must be {answer}."
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "answer",
    [
        "a JSON object mapping each contig name to its read count",
        "a JSON object mapping each gene ID in genes.bed to its read count (an integer)",
        "a JSON object mapping each gene ID to '+' or '-'",
    ],
)
def test_prompt_for_shows_where_a_mapping_answer_goes_and_changes_nothing_else(route, answer):
    stage1 = (
        f"Task: the {route} sentence (see README.md for the files). "
        f"In result.json, `answer` must be {answer}."
    )

    prompt = prompt_for(route, "the free sentence", "the forced sentence", answer)

    assert prompt == stage1[:-1] + WRAPPER
    assert prompt.endswith(f"{answer}{WRAPPER}")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("family", ["F8", "F23"])
def test_the_mapping_families_show_where_the_mapping_goes(tmp_path, tools, family, route):
    _, task = workspace(tmp_path, family, 1000, "faulted", tools, route=route)

    assert task.truth["kind"] == "dict_num"
    assert task.prompt.endswith(WRAPPER) and task.prompt.count("<key>") == 1


@pytest.mark.parametrize(
    ("family", "answer"),
    [("F2", "a JSON list of uppercase sequence strings, one per region"), ("F3", "a JSON integer")],
)
def test_the_list_and_scalar_original_families_keep_their_answer_sentence(
    tmp_path, tools, family, answer
):
    _, task = workspace(tmp_path, family, 1000, "faulted", tools)

    assert task.prompt.endswith(f"In result.json, `answer` must be {answer}.")
