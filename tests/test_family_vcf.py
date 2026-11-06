"""Family-specific tests for F7 (regions_vs_targets), F9 (multiallelic) and F10 (indel_bed).

The generic gates live in test_extra_families.py. Here: the exposure functions on concrete command
lines, the trap on every seed 0-29 from the plans alone (pure Python, no tools), and, where the
real tools are available, that the tools behave as the evidence claims on the installed versions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f07_regions_vs_targets as f7
from biowarden.bench.tasks.extra import f09_multiallelic as f9
from biowarden.bench.tasks.extra import f10_indel_bed as f10

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
WORK = Path("/nonexistent-workspace")  # the exposure checks read only the command text


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def built(tmp_path, family, seed, twin, tools, tier="explicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def vcf_rows(tools, work):
    proc = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work)
    return [line.split("\t") for line in proc.stdout.splitlines() if line]


# ---- exposure ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "bcftools view -H -T regions.bed variants.vcf.gz | cut -f3",
        "bcftools query -f '%ID\\n' -T regions.bed variants.vcf.gz",
        "bcftools view -H --targets-file regions.bed variants.vcf.gz",
        "bcftools view -H -t chr1:50-100 variants.vcf.gz",
        "cd /w && bcftools view -T regions.bed variants.vcf.gz 2>&1 | head",
        "bcftools view -H -T regions.bed --targets-overlap 0 variants.vcf.gz",
        "bcftools view -H -R regions.bed --regions-overlap 0 variants.vcf.gz",
    ],
)
def test_f7_exposed_on_start_only_selection(command):
    assert f7.exposed_f7([command], WORK) is True


@pytest.mark.parametrize(
    "command",
    [
        "bcftools view -H -R regions.bed variants.vcf.gz",
        "bcftools view -H -R regions.bed --regions-overlap 1 variants.vcf.gz",
        "bcftools view -H -T regions.bed --targets-overlap 1 variants.vcf.gz",
        "bcftools view -H -T regions.bed --targets-overlap record variants.vcf.gz",
        "bcftools view -H -T regions.bed --targets-overlap=2 variants.vcf.gz",
        "bedtools intersect -a variants.vcf.gz -b regions.bed -u",
        "bcftools view -H -r chr1:1-100 variants.vcf.gz",
        "bcftools view -H variants.vcf.gz | sort -T /tmp | head",
        "bcftools view -h variants.vcf.gz",
        "ls -la; cat README.md",
    ],
)
def test_f7_not_exposed_on_overlap_aware_or_unrelated_commands(command):
    assert f7.exposed_f7([command], WORK) is False


def test_f7_exposure_looks_at_every_command_and_ignores_none():
    safe = "bcftools view -H -R regions.bed variants.vcf.gz"
    trapped = "bcftools view -H -T regions.bed variants.vcf.gz"
    assert f7.exposed_f7([safe, trapped], WORK) is True
    assert f7.exposed_f7([safe], WORK) is False
    assert f7.exposed_f7([], WORK) is False


@pytest.mark.parametrize(
    "command",
    [
        "bcftools view -H variants.vcf.gz | wc -l",
        "bcftools query -f '%ALT\\n' variants.vcf.gz | wc -l",
        "zcat variants.vcf.gz | grep -vc '^#'",
        "bcftools index -n variants.vcf.gz",
        "bcftools view -H variants.vcf.gz | awk 'END{print NR}'",
    ],
)
def test_f9_exposed_on_row_counts(command):
    assert f9.exposed_f9([command], WORK) is True


@pytest.mark.parametrize(
    "command",
    [
        "bcftools view -H variants.vcf.gz | cut -f5 | tr ',' '\\n' | wc -l",
        "bcftools query -f '%ALT\\n' variants.vcf.gz | awk -F, '{n+=NF} END{print n}'",
        "bcftools view -H variants.vcf.gz | awk '{n+=split($5,a,\",\")} END{print n}'",
        "bcftools norm -m -any variants.vcf.gz | grep -vc '^#'",
        "bcftools view -H variants.vcf.gz | head",
        "ls | wc -l",
        "ls -la; cat README.md",
    ],
)
def test_f9_not_exposed_on_allele_aware_or_unrelated_commands(command):
    assert f9.exposed_f9([command], WORK) is False


@pytest.mark.parametrize(
    "command",
    [
        "bcftools query -f '%CHROM\\t%POS0\\t%END\\n' variants.vcf.gz",
        "bcftools query -f '%CHROM\\t%POS\\t%REF\\n' variants.vcf.gz | "
        'awk \'{print $1"\\t"$2-1"\\t"$2-1+length($3)}\'',
        "bcftools view -H variants.vcf.gz | awk -v OFS='\\t' '{print $1,$2-1,$2-1+length($4)}'",
        "zcat variants.vcf.gz | grep -v '^#' | awk '{print $1, $2-1, $2-1+length($4)}'",
        # seen in the open-weight runs: ALT is printed but only REF is measured
        "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' variants.vcf.gz | "
        "awk -F'\\t' '{print $1\"\\t\"$2-1\"\\t\"$2-1+length($3)}'",
    ],
)
def test_f10_exposed_on_ref_span_intervals(command):
    assert f10.exposed_f10([command], WORK) is True


@pytest.mark.parametrize(
    "command",
    [
        "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' variants.vcf.gz | "
        "awk '{s=$2-1; if (length($3)>1 && length($4)==1) s++; print $1\"\\t\"s}'",
        "bcftools view -H variants.vcf.gz | "
        "awk '{s=$2-1; if (length($5)==1 && length($4)>1) s++; print $1,s,$2-1+length($4)}'",
        "bcftools view -H variants.vcf.gz | head -5",
        "bcftools view -h variants.vcf.gz",
        "awk '{print $1, $2-1, $3}' regions.bed",
        "ls -la; cat README.md",
    ],
)
def test_f10_not_exposed_when_alt_is_read_or_command_is_unrelated(command):
    assert f10.exposed_f10([command], WORK) is False


@pytest.mark.parametrize("exposed", [f7.exposed_f7, f9.exposed_f9, f10.exposed_f10])
def test_exposure_is_false_without_commands(exposed):
    assert exposed([], WORK) is False


# ---- the trap is there on every seed, by construction (no tools) ------------------------------


def test_f7_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f7.make_plan(seed)
        c_exp, c_naive = f7.truth_answers(plan, "clean")
        f_exp, f_naive = f7.truth_answers(plan, "faulted")
        assert c_exp == f_exp and c_naive == c_exp, seed
        assert set(f_naive) < set(f_exp), seed  # start-only never invents an ID
        missed = set(f_exp) - set(f_naive)
        assert missed == {"g0", "g1", "g2", "g3"}, (seed, missed)
        recs = plan.records("faulted")
        assert len(recs) <= 80 and len({r.id for r in recs}) == len(recs), seed
        assert len(plan.contigs) <= 3 and sum(plan.contigs.values()) <= 3000, seed
        # the wide block really spans two regions; decoys overlap none and are the same in both
        g0 = next(r for r in recs if r.id == "g0")
        assert sum(f7.overlaps(g0, [r]) for r in plan.regions) == 2, seed
        for twin_recs in (plan.records("clean"), recs):
            decoys = [r for r in twin_recs if r.id in {"g4", "g5", "g6"}]
            assert len(decoys) == 3 and not any(f7.overlaps(d, plan.regions) for d in decoys)
        assert [r for r in plan.records("clean") if r.id in {"g4", "g5", "g6"}] == [
            r for r in recs if r.id in {"g4", "g5", "g6"}
        ], seed


def test_f7_clean_blocks_start_inside_their_region_and_faulted_ones_before_it():
    for seed in SEEDS:
        plan = f7.make_plan(seed)
        for rec in plan.records("clean"):
            if rec.id in {"g0", "g1", "g2", "g3"}:
                assert f7.starts_in(rec, plan.regions), (seed, rec)
        for rec in plan.records("faulted"):
            if rec.id in {"g0", "g1", "g2", "g3"}:
                assert not f7.starts_in(rec, plan.regions), (seed, rec)
                assert f7.overlaps(rec, plan.regions), (seed, rec)


def test_f9_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f9.make_plan(seed)
        c_exp, c_naive = f9.truth_counts(plan, "clean")
        f_exp, f_naive = f9.truth_counts(plan, "faulted")
        assert c_exp == f_exp and c_naive == c_exp, seed
        assert f_exp - f_naive >= 6, (seed, f_exp, f_naive)
        assert all(len(s.alts) == 1 for s in plan.clean), seed
        assert sum(len(s.alts) > 1 for s in plan.faulted) >= 6, seed
        for site in plan.faulted:
            assert site.ref not in site.alts and len(set(site.alts)) == len(site.alts)
        assert len(plan.clean) <= 80 and sum(plan.contigs.values()) <= 3000, seed


def test_f10_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f10.make_plan(seed)
        c_exp, c_naive = f10.truth_answers(plan, "clean")
        f_exp, f_naive = f10.truth_answers(plan, "faulted")
        assert c_exp == f_exp and c_naive == c_exp, seed
        wrong = [i for i, (a, b) in enumerate(zip(f_exp, f_naive, strict=True)) if a != b]
        assert len(wrong) >= 6, (seed, wrong)
        assert wrong == [i for i, s in enumerate(plan.sites) if s.deletion], seed
        assert len(plan.sites) <= 80 and sum(plan.contigs.values()) <= 3000, seed
        # a REF longer than one base appears in both twins, so it is no marker of the fault
        for twin in ("clean", "faulted"):
            assert any(len(plan.record(s, twin)[2]) > 1 for s in plan.sites if not s.deletion)
        for s in plan.sites:
            _, pos, ref, alt = plan.record(s, "faulted")
            if s.deletion:
                assert alt == ref[0] and len(ref) == s.k + 1 and pos == s.start0
            else:
                assert len(alt) == len(ref) == s.k and all(
                    a != r for a, r in zip(alt, ref, strict=True)
                )


def test_f10_deletions_are_left_aligned_and_sites_do_not_touch():
    for seed in SEEDS:
        plan = f10.make_plan(seed)
        previous_end: dict[str, int] = {}
        for s in plan.sites:
            _, _, ref, alt = plan.record(s, "faulted")
            if s.deletion:
                assert ref[-1] != alt, (seed, ref, alt)
            assert s.start0 - previous_end.get(s.chrom, -99) >= f10.MIN_GAP, (seed, s)
            previous_end[s.chrom] = s.start0 + s.k


# ---- with the real tools ---------------------------------------------------------------------


def test_installed_bcftools_help_documents_the_two_defaults(tools, tmp_path):
    proc = tools.run(["bcftools", "view", "--help"], tmp_path)
    text = proc.stdout + proc.stderr
    modes = "Include if POS in the region (0), record overlaps (1), variant overlaps (2)"
    assert f"--targets-overlap 0|1|2       {modes} [0]" in text
    assert f"--regions-overlap 0|1|2       {modes} [1]" in text


def test_f7_bcftools_r_is_the_truth_and_t_is_the_trap_on_every_seed(tmp_path, tools):
    for seed in SEEDS:
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F7", seed, twin, tools)
            for option, key in (("-R", "expected"), ("-T", "naive")):
                argv = ["bcftools", "view", "-H", option, "regions.bed", "variants.vcf.gz"]
                proc = tools.run(argv, work)
                ids = [line.split("\t")[2] for line in proc.stdout.splitlines()]
                assert proc.returncode == 0 and proc.stderr == "", (seed, twin, option)
                assert len(ids) == len(set(ids)), (seed, twin, option)  # no duplicate records
                assert sorted(ids) == sorted(task.truth[key]), (seed, twin, option)
            argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "regions.bed", "-u"]
            ids = [ln.split("\t")[2] for ln in tools.run(argv, work).stdout.splitlines()]
            assert sorted(ids) == sorted(task.truth["expected"]), (seed, twin, "bedtools")


def test_f7_start_only_overlap_mode_fixes_t_and_is_visible_in_the_files(tmp_path, tools):
    work, task = built(tmp_path, "F7", 4, "faulted", tools)
    argv = ["bcftools", "view", "-H", "-T", "regions.bed", "--targets-overlap", "1"]
    out = tools.run([*argv, "variants.vcf.gz"], work).stdout
    assert sorted(line.split("\t")[2] for line in out.splitlines()) == task.truth["expected"]
    header = tools.run(["bcftools", "view", "-h", "variants.vcf.gz"], work).stdout
    assert "##INFO=<ID=END" in header and "##ALT=<ID=NON_REF" in header
    assert "END=" in tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work).stdout


def test_f9_row_count_and_allele_count_on_every_seed(tmp_path, tools):
    for seed in SEEDS:
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F9", seed, twin, tools)
            rows = vcf_rows(tools, work)
            assert len(rows) == task.truth["naive"], (seed, twin)
            assert sum(len(r[4].split(",")) for r in rows) == task.truth["expected"], (seed, twin)
            assert len(rows) <= 80
            argv = ["bcftools", "norm", "-m", "-any", "-Ov", "variants.vcf.gz"]
            split = tools.run(argv, work)
            n_split = sum(1 for line in split.stdout.splitlines() if not line.startswith("#"))
            assert split.returncode == 0 and n_split == task.truth["expected"], (seed, twin)
            idx = tools.run(["bcftools", "index", "-n", "variants.vcf.gz"], work)
            assert int(idx.stdout) == task.truth["naive"], (seed, twin)
            assert (twin == "faulted") == any("," in r[4] for r in rows), (seed, twin)


def test_f10_ref_span_is_right_for_substitutions_and_early_for_deletions(tmp_path, tools):
    for seed in SEEDS:
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F10", seed, twin, tools)
            argv = ["bcftools", "query", "-f", "%CHROM\t%POS0\t%END\n", "variants.vcf.gz"]
            proc = tools.run(argv, work)
            assert proc.returncode == 0 and proc.stderr == "", (seed, twin)
            assert proc.stdout.splitlines() == task.truth["naive"], (seed, twin)
            rows = vcf_rows(tools, work)
            assert len(rows) == len(task.truth["expected"]) <= 80
            has_deletion = any(len(r[3]) > 1 and len(r[4]) == 1 for r in rows)
            assert has_deletion == (twin == "faulted"), (seed, twin)
            assert all(r[3][0] != r[4][0] for r in rows if len(r[3]) == len(r[4])), (seed, twin)


# ---- tiers and routes: the evidence is where the family says it is ----------------------------


@pytest.mark.parametrize(
    ("family", "needle"), [("F7", "run into"), ("F9", "multi-allelic"), ("F10", "anchor")]
)
def test_explicit_readme_states_the_convention_and_implicit_does_not(
    tmp_path, tools, family, needle
):
    work_e, explicit = built(tmp_path, family, 2, "faulted", tools, "explicit")
    work_i, implicit = built(tmp_path, family, 2, "faulted", tools, "implicit")
    assert needle in (work_e / "README.md").read_text()
    assert needle not in (work_i / "README.md").read_text()
    assert explicit.evidence != implicit.evidence and implicit.evidence
    assert explicit.truth == implicit.truth


@pytest.mark.parametrize("family", ["F7", "F9", "F10"])
def test_clean_readme_does_not_describe_the_fault(tmp_path, tools, family):
    work_c, _ = built(tmp_path, family, 2, "clean", tools, "explicit")
    work_f, _ = built(tmp_path, family, 2, "faulted", tools, "explicit")
    text_c, text_f = ((w / "README.md").read_text() for w in (work_c, work_f))
    assert text_c != text_f and text_c.count("\n") == text_f.count("\n")


@pytest.mark.parametrize(
    ("family", "dictated"), [("F7", "bcftools view -T"), ("F9", "bcftools view -H"), ("F10", "awk")]
)
def test_forced_prompt_names_the_dictated_tool_and_free_prompt_does_not(
    tmp_path, tools, family, dictated
):
    _, free = built(tmp_path, family, 1, "faulted", tools, "explicit", "free")
    _, forced = built(tmp_path, family, 1, "faulted", tools, "explicit", "forced")
    assert dictated in forced.prompt and dictated not in free.prompt
    assert free.slot == forced.slot and free.truth == forced.truth
