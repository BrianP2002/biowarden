"""Family-specific checks for F25-F28, the stale-documentation families (locus documentation).

The generic gates live in test_extra_families.py. Here: the trap size on seeds 0-29 from the plans
alone (pure Python), the twins and tiers (the faulted twin differs from the clean one by one line
of NOTES.md, the README differs only by the "check the notes" sentence), the reference and naive
solvers, the exposure functions on concrete command lines, the stale sentences against the
installed tools (help text and real runs: exit 0, empty stderr), and byte reproducibility across
processes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.score import score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f25_doc_flag as f25
from biowarden.bench.tasks.extra import f26_doc_default as f26
from biowarden.bench.tasks.extra import f27_doc_output as f27
from biowarden.bench.tasks.extra import f28_doc_format as f28
from biowarden.bench.tasks.reference import solve

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
FAMILIES = ("F25", "F26", "F27", "F28")
WORK = Path("/nonexistent-workspace")  # the exposure checks read only the command text
CHECK_MARK = "may be out of date"
FORCED_TOOL = {
    "F25": "samtools coverage",
    "F26": "bcftools mpileup",
    "F27": "samtools flagstat",
    "F28": "bcftools view -R",
}


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def built(tmp_path, family, seed, twin, tools, tier="explicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def run(tools, work, *argv):
    proc = tools.run(list(argv), work)
    assert proc.returncode == 0, (argv, proc.stderr)
    return proc


# ---- the trap is there on every seed, by construction (no tools) ------------------------------


def test_f25_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f25.make_plan(seed)
        expected, naive = f25.truth_answers(plan)
        assert len(expected) == len(naive) == len(plan.contigs) == 2
        assert all(n - e >= 9 for e, n in zip(expected, naive, strict=True)), seed
        assert len(plan.reads) <= 100 and len({r.name for r in plan.reads}) == len(plan.reads)
        for chrom in plan.contigs:
            mapqs = {r.mapq for r in plan.on(chrom)}
            assert {29, 30} <= mapqs, (seed, chrom)  # both sides of the cutoff, at the boundary
            assert sum(r.mapq >= 30 for r in plan.on(chrom)) >= 14, (seed, chrom)
        for r in plan.reads:
            assert r.qual[: f25.GOOD_HEAD] == "I" * f25.GOOD_HEAD and len(r.seq) == f25.READ_LEN
            assert r.pos + f25.READ_LEN - 1 <= plan.contigs[r.chrom]
        # some bases are below Q30, so a base-quality filter visibly changes covbases and depth
        assert any(set(r.qual) & {"5", "+"} for r in plan.reads), seed


def test_f26_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f26.make_plan(seed)
        expected, naive = f26.truth_answers(plan)
        assert naive - expected >= 50 and expected >= 40, (seed, expected, naive)
        assert 3 <= plan.threshold <= 13, seed
        quals = {q for r in plan.reads for q in r.quals}
        assert {12, 13, 40} <= quals and min(quals) >= 2, seed  # both sides of Q13, no Q0/Q1
        for r in plan.reads:
            assert r.seq == plan.genome[r.chrom][r.pos - 1 :][: len(r.seq)], seed
            assert len(r.seq) == len(r.quals) and r.pos + len(r.seq) - 1 <= plan.contigs[r.chrom]
        assert len(plan.reads) <= 100 and sum(plan.contigs.values()) <= 700, seed


def test_f27_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f27.make_plan(seed)
        expected, naive = f27.truth_answers(plan)
        assert len(expected) == len(f27.SAMPLES) == 3
        assert all(e - n >= 25 and n >= 12 for e, n in zip(expected, naive, strict=True)), seed
        for i, sample in enumerate(plan.samples):
            flags = [int(line.split("\t")[1]) for line in sample.sam_body]
            assert sum(not f & 4 for f in flags) == sample.mapped == expected[i], seed
            assert sum(bool(f & 1024) for f in flags) == sample.duplicates == naive[i], seed
            assert not any(f & (256 | 2048 | 512 | 1) for f in flags), seed
            assert len(flags) == sample.mapped + sample.unmapped <= 100


def test_f28_trap_is_observable_on_every_seed():
    for seed in SEEDS:
        plan = f28.make_plan(seed)
        expected, naive = f28.truth_answers(plan)
        assert set(naive) < set(expected) and len(expected) - len(naive) >= 14, seed
        assert len(naive) >= 6, seed  # the first panel is not empty
        text = plan.bed_text()
        assert text.count("\n\n") == f28.PANELS - 1 and not text.startswith("\n"), seed
        assert text.endswith("\n") and not text.endswith("\n\n"), seed
        assert len(plan.regions) == f28.PANELS * f28.REGIONS_PER_PANEL, seed
        by_panel = [[r for r in plan.regions if r.panel == p] for p in range(f28.PANELS)]
        assert all(len(group) == f28.REGIONS_PER_PANEL for group in by_panel), seed
        order = {c: i for i, c in enumerate(plan.contigs)}
        keys = [(order[r.chrom], r.start0) for r in plan.regions]
        assert keys == sorted(keys), seed  # sorted as a whole
        assert len(plan.variants) <= 60 and len({v.id for v in plan.variants}) == len(plan.variants)
        for v in plan.variants:  # nothing sits on an edge: BED and 1-based readings agree
            near = [
                r
                for r in plan.regions
                if r.chrom == v.chrom and r.start0 - 3 <= v.pos <= r.end0 + 3
            ]
            assert all(r.start0 + 2 <= v.pos - 1 < r.end0 - 1 for r in near), (seed, v)


# ---- twins and tiers --------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_faulted_notes_differ_from_clean_notes_by_exactly_one_line(tmp_path, tools, family):
    for tier in ("explicit", "implicit"):
        work_c, clean = built(tmp_path, family, 3, "clean", tools, tier)
        work_f, faulted = built(tmp_path, family, 3, "faulted", tools, tier)
        lines_c = (work_c / "NOTES.md").read_text().splitlines()
        lines_f = (work_f / "NOTES.md").read_text().splitlines()
        assert len(lines_c) == len(lines_f)
        differing = [i for i, (a, b) in enumerate(zip(lines_c, lines_f, strict=True)) if a != b]
        assert len(differing) == 1, (family, tier, differing)
        # everything else in the workspace is the same bytes
        for p in work_c.iterdir():
            if p.name != "NOTES.md":
                assert p.read_bytes() == (work_f / p.name).read_bytes(), (family, p.name)
        assert clean.prompt == faulted.prompt and "NOTES.md" in clean.prompt
        assert clean.truth["expected"] == faulted.truth["expected"]
        assert clean.truth["naive"] == clean.truth["expected"] != faulted.truth["naive"]
        assert clean.evidence != faulted.evidence and faulted.evidence


@pytest.mark.parametrize("family", FAMILIES)
def test_readme_adds_the_check_sentence_only_in_the_explicit_tier(tmp_path, tools, family):
    work_e, explicit = built(tmp_path, family, 2, "faulted", tools, "explicit")
    work_i, implicit = built(tmp_path, family, 2, "faulted", tools, "implicit")
    text_e = (work_e / "README.md").read_text()
    text_i = (work_i / "README.md").read_text()
    assert CHECK_MARK in text_e and CHECK_MARK not in text_i
    assert "--help" in text_e and "installed" in text_e
    assert "NOTES.md" in text_i and "stale" not in text_i.lower()
    assert [ln for ln in text_e.splitlines() if CHECK_MARK not in ln] == text_i.splitlines()
    assert explicit.truth == implicit.truth and explicit.prompt == implicit.prompt
    # the same README in both twins of a tier
    work_c, _ = built(tmp_path, family, 2, "clean", tools, "explicit")
    assert (work_c / "README.md").read_bytes() == (work_e / "README.md").read_bytes()


@pytest.mark.parametrize("family", FAMILIES)
def test_forced_route_names_the_tool_and_the_free_route_does_not(tmp_path, tools, family):
    _, free = built(tmp_path, family, 1, "faulted", tools, "explicit", "free")
    _, forced = built(tmp_path, family, 1, "faulted", tools, "explicit", "forced")
    assert FORCED_TOOL[family] in forced.prompt and FORCED_TOOL[family] not in free.prompt
    assert not re.search(r"samtools|bcftools|bedtools", free.prompt)
    assert "NOTES.md" in free.prompt and "NOTES.md" in forced.prompt
    assert forced.truth == free.truth and forced.evidence == free.evidence


# ---- the scripted solvers ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_reference_passes_and_naive_falls_into_the_trap_on_every_seed(tmp_path, tools, family):
    for seed in SEEDS:
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, family, seed, twin, tools)
            ref = solve(task, work, tools, reference=True)
            assert score_result(task.truth, ref).outcome == "pass", (family, seed, twin)
            naive = solve(task, work, tools, reference=False)
            outcome = score_result(task.truth, naive)
            assert outcome.outcome == ("pass" if twin == "clean" else "silent_wrong"), (
                family,
                seed,
                twin,
            )
            assert naive["answer"] == task.truth["naive"], (family, seed, twin)


# ---- the stale sentence against the installed tools -------------------------------------------


# The release each family's notes are wrong about (the module docstrings give the NEWS entries).
# On an older tool the "stale" sentence is the right one and the trap is gone, and the generic
# gates would then report a broken builder instead of an old tool.
MIN_VERSION = {
    "F25": ("samtools", (1, 12)),  # coverage -q is the mapping quality, as in mpileup, from 1.12
    "F26": ("bcftools", (1, 13)),  # mpileup skips bases below Q1, not Q13, by default from 1.13
    "F27": ("samtools", (1, 13)),  # flagstat prints 16 lines, mapped on line 7, from 1.13
    "F28": ("samtools", (1, 10)),  # region files may hold blank lines from 1.10
}


def version_of(tools, tool, cwd):
    proc = tools.run([tool, "--version"], cwd)
    found = re.match(rf"{tool} (\d+)\.(\d+)", proc.stdout)
    assert found, (tool, proc.stdout, proc.stderr)
    return int(found[1]), int(found[2])


@pytest.mark.parametrize("family", FAMILIES)
def test_installed_tool_is_new_enough_for_the_trap(tmp_path, tools, family):
    tool, minimum = MIN_VERSION[family]
    version = version_of(tools, tool, tmp_path)
    needed = ".".join(map(str, minimum))
    assert version >= minimum, f"{family} needs {tool} >= {needed}, found {version[0]}.{version[1]}"


def test_f25_help_text_and_real_runs_contradict_the_stale_notes(tmp_path, tools):
    coverage = run(tools, tmp_path, "samtools", "coverage", "--help")
    text = coverage.stdout + coverage.stderr
    assert re.search(r"-q, --min-MQ INT\s+mapping quality threshold", text)
    assert re.search(r"-Q, --min-BQ INT\s+base quality threshold", text)
    depth = tools.run(["samtools", "depth"], tmp_path)  # the usage text, with a non-zero exit
    text = depth.stdout + depth.stderr
    assert re.search(r"-q, --min-BQ INT\s+Filter bases with base quality", text)  # opposite letters
    assert re.search(r"-Q, --min-MQ INT\s+Filter alignments with mapping quality", text)
    for seed in SEEDS:
        work, task = built(tmp_path, "F25", seed, "faulted", tools)
        for option, key in (("-q", "expected"), ("-Q", "naive")):
            proc = tools.run(["samtools", "coverage", option, "30", "reads.bam"], work)
            rows = [ln.split("\t") for ln in proc.stdout.splitlines() if not ln.startswith("#")]
            assert proc.returncode == 0 and proc.stderr == "", (seed, option)
            assert [int(r[3]) for r in rows] == task.truth[key], (seed, option)
        plain = tools.run(["samtools", "coverage", "reads.bam"], work).stdout.splitlines()
        base = tools.run(["samtools", "coverage", "-Q", "30", "reads.bam"], work).stdout
        assert [ln.split("\t")[3] for ln in plain[1:]] == [
            ln.split("\t")[3] for ln in base.splitlines()[1:]
        ]
        assert plain[1:] != base.splitlines()[1:]  # covbases or depth move, so it looks filtered


def test_f26_help_default_and_real_runs_contradict_the_stale_notes(tmp_path, tools):
    usage = tools.run(["bcftools", "mpileup"], tmp_path)  # the usage text, with a non-zero exit
    help_text = usage.stdout + usage.stderr
    line = next(ln for ln in help_text.splitlines() if "--min-BQ" in ln)
    assert line.rstrip().endswith("[1]") and "[13]" not in line
    assert re.search(r"--config|-X, --config", help_text)
    usage = tools.run(["samtools", "mpileup"], tmp_path)
    samtools = usage.stdout + usage.stderr
    assert re.search(r"-Q, --min-BQ INT .*\[13\]", samtools)  # samtools still uses 13
    for seed in range(6):
        work, task = built(tmp_path, "F26", seed, "faulted", tools)
        plan = f26.make_plan(seed)
        for options, cutoff in (([], f26.DEFAULT_Q), (["-Q", "13"], 13), (["-X", "1.12"], 13)):
            argv = ["bcftools", "mpileup", "-f", "ref.fa", "-B", *options, "-a", "FORMAT/DP"]
            proc = tools.run([*argv, "reads.bam"], work)
            assert proc.returncode == 0, (seed, options)
            assert not re.search(r"warn|error|fail", proc.stderr, re.IGNORECASE), proc.stderr
            depth = {}
            for ln in proc.stdout.splitlines():
                if ln.startswith("#"):
                    continue
                f = ln.split("\t")
                depth[(f[0], int(f[1]))] = int(f[9].split(":")[f[8].split(":").index("DP")])
            want = {k: d for k, d in f26.depths(plan.reads, cutoff).items()}
            assert {k: d for k, d in depth.items() if d} == want, (seed, options)
        assert task.truth["naive"] > task.truth["expected"]


def test_f27_flagstat_layout_contradicts_the_stale_notes(tmp_path, tools):
    for seed in SEEDS:
        work, task = built(tmp_path, "F27", seed, "faulted", tools)
        for i, name in enumerate(f27.SAMPLES):
            proc = tools.run(["samtools", "flagstat", name], work)
            lines = proc.stdout.splitlines()
            assert proc.returncode == 0 and proc.stderr == "", (seed, name)
            assert len(lines) == 16, (seed, name)
            assert re.fullmatch(r"\d+ \+ 0 duplicates", lines[4]), lines[4]  # line 5 now
            assert re.match(r"\d+ \+ 0 mapped \(", lines[6]), lines[6]  # line 7 now
            assert int(lines[4].split()[0]) == task.truth["naive"][i], (seed, name)
            assert int(lines[6].split()[0]) == task.truth["expected"][i], (seed, name)
            view = run(tools, work, "samtools", "view", "-c", "-F", "4", name)
            assert int(view.stdout) == task.truth["expected"][i]


def test_f28_bcftools_stops_at_the_first_blank_line_and_the_other_tools_do_not(tmp_path, tools):
    for seed in SEEDS:
        work, task = built(tmp_path, "F28", seed, "faulted", tools)
        for option in ("-R", "-T"):
            argv = ["bcftools", "view", "-H", option, "regions.bed", "variants.vcf.gz"]
            proc = tools.run(argv, work)
            ids = sorted(ln.split("\t")[2] for ln in proc.stdout.splitlines())
            assert proc.returncode == 0 and proc.stderr == "", (seed, option)
            assert ids == sorted(task.truth["naive"]), (seed, option)
        argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "regions.bed", "-u"]
        proc = tools.run(argv, work)
        ids = sorted(ln.split("\t")[2] for ln in proc.stdout.splitlines())
        assert ids == sorted(task.truth["expected"]), (seed, "bedtools")
        text = (work / "regions.bed").read_text()
        (work / "gapless.bed").write_text(text.replace("\n\n", "\n"))
        argv = ["bcftools", "view", "-H", "-R", "gapless.bed", "variants.vcf.gz"]
        ids = sorted(ln.split("\t")[2] for ln in tools.run(argv, work).stdout.splitlines())
        assert ids == sorted(task.truth["expected"]), (seed, "gapless")


# ---- exposure ---------------------------------------------------------------------------------

EXPOSED = {
    "F25": [
        "samtools coverage -Q 30 reads.bam",
        "samtools coverage -Q 30 reads.bam | cut -f1,4",
        "cd /w && samtools coverage -Q30 reads.bam",
        "for f in reads.bam; do samtools coverage -Q 30 $f; done",
        "samtools coverage -Q 30 -H reads.bam | awk '{print $4}'",
        'python3 -c "import subprocess; '
        "subprocess.run(['samtools','coverage','-Q','30','reads.bam'])\"",
        [
            "samtools coverage -Q 30 reads.bam",
            "samtools coverage -q 30 reads.bam",
        ],  # corrected later
    ],
    "F26": [
        "bcftools mpileup -f ref.fa -B -a FORMAT/DP reads.bam",
        "bcftools mpileup -f ref.fa -B -a FORMAT/DP reads.bam | bcftools query -f '%POS[\\t%DP]'",
        "bcftools mpileup -Ou -f ref.fa reads.bam | bcftools view -H | wc -l",
        "bcftools mpileup -Q 1 -f ref.fa -a FORMAT/DP reads.bam",
        "bcftools mpileup -a FORMAT/DP -f ref.fa -o pile.vcf reads.bam",
        'python3 -c "import subprocess; '
        "subprocess.run(['bcftools','mpileup','-f','ref.fa','reads.bam'])\"",
        [
            "bcftools mpileup -f ref.fa reads.bam | head",
            "bcftools mpileup -Q 13 -f ref.fa reads.bam",
        ],
    ],
    "F27": [
        "samtools flagstat sample1.bam | sed -n 5p | cut -d' ' -f1",
        "samtools flagstat sample1.bam | sed -n '5p'",
        "for f in sample1.bam sample2.bam sample3.bam; do "
        "samtools flagstat $f | sed -n 5p | awk '{print $1}'; done",
        "for f in *.bam; do samtools flagstat $f | awk 'NR==5{print $1}'; done",
        "samtools flagstat sample1.bam | awk 'FNR == 5 {print $1}'",
        "samtools flagstat sample1.bam | head -5 | tail -1",
        "samtools flagstat sample1.bam | head -n 5 | tail -n 1 | cut -d' ' -f 1",
        'python3 -c "import subprocess; '
        "print(subprocess.run(['samtools','flagstat','sample1.bam'], capture_output=True, "
        'text=True).stdout.splitlines()[4])"',
        ["samtools flagstat sample1.bam | sed -n 5p", "samtools flagstat sample1.bam"],
    ],
    "F28": [
        "bcftools view -H -R regions.bed variants.vcf.gz | cut -f3",
        "bcftools view -H -T regions.bed variants.vcf.gz",
        "bcftools query -R regions.bed -f '%ID\\n' variants.vcf.gz",
        "bcftools view -H --regions-file regions.bed variants.vcf.gz",
        "bcftools view -H --targets-file=regions.bed variants.vcf.gz",
        "bcftools view -H -R ./regions.bed variants.vcf.gz",
        "cd /w && bcftools view -H -R /w/regions.bed variants.vcf.gz | wc -l",
        [
            "bcftools view -H -R regions.bed variants.vcf.gz | wc -l",
            "grep -v '^$' regions.bed > c.bed",
        ],
    ],
}
NOT_EXPOSED = {
    "F25": [
        "samtools coverage -q 30 reads.bam",
        "samtools coverage --min-MQ 30 reads.bam",
        "samtools coverage -q 30 -Q 30 reads.bam",
        "samtools coverage -Q 30 --min-MQ 30 reads.bam",
        "samtools coverage --help",
        "samtools coverage -h",
        "samtools coverage reads.bam",
        "samtools coverage --min-BQ 30 reads.bam",
        "samtools depth -Q 30 reads.bam | wc -l",
        "samtools view -c -q 30 reads.bam chr1",
        "samtools view -q 30 -b reads.bam | samtools coverage -",
        "ls -la; cat NOTES.md",
    ],
    "F26": [
        "bcftools mpileup -f ref.fa -B -Q 13 -a FORMAT/DP reads.bam",
        "bcftools mpileup -Q13 -f ref.fa reads.bam",
        "bcftools mpileup -Q 20 -f ref.fa reads.bam",
        "bcftools mpileup --min-BQ 13 -f ref.fa reads.bam",
        "bcftools mpileup -X 1.12 -f ref.fa reads.bam",
        "bcftools mpileup --config 1.12 -f ref.fa reads.bam",
        "bcftools mpileup --help",
        "bcftools mpileup 2>&1 | head -50",
        "samtools mpileup -f ref.fa reads.bam | wc -l",
        "samtools depth -q 13 reads.bam | awk '$3>=6' | wc -l",
        'python3 -c "import subprocess; '
        "subprocess.run(['bcftools','mpileup','-Q','13','-f','ref.fa','reads.bam'])\"",
        "ls -la; cat README.md NOTES.md",
    ],
    "F27": [
        "samtools flagstat sample1.bam",
        "samtools flagstat sample1.bam | grep mapped",
        "samtools flagstat sample1.bam | grep ' mapped (' | cut -d' ' -f1",
        "samtools flagstat sample1.bam | sed -n 7p | cut -d' ' -f1",
        "samtools flagstat sample1.bam | awk '/ mapped \\(/{print $1}'",
        "samtools flagstat sample1.bam | awk 'NR==5{print} / mapped /{print}'",
        "samtools flagstat sample1.bam | head -3",
        "samtools flagstat sample1.bam | head -50 | tail -10",
        "samtools flagstat sample1.bam | sed -n 5,7p",
        "samtools view -c -F 4 sample1.bam",
        "head -5 NOTES.md | tail -1",
        "ls *.bam; wc -l NOTES.md",
    ],
    "F28": [
        "bcftools view -H -R <(grep -v '^$' regions.bed) variants.vcf.gz",
        "grep -v '^$' regions.bed > clean.bed; bcftools view -H -R clean.bed variants.vcf.gz",
        "bcftools view -H -R clean_regions.bed variants.vcf.gz",
        "bcftools view -H -R regions_nogaps.bed variants.vcf.gz",
        "sed '/^$/d' regions.bed > /tmp/r.bed; bcftools view -H -R /tmp/r.bed variants.vcf.gz",
        "bcftools view -H -R <(sed '/^\\s*$/d' regions.bed) variants.vcf.gz",
        "awk 'NF' regions.bed > r.bed && bcftools view -H -T r.bed variants.vcf.gz",
        "bcftools view -H -R <(bedtools sort -i regions.bed) variants.vcf.gz",
        "bedtools intersect -a variants.vcf.gz -b regions.bed -u",
        "bcftools view -H -r chr1:50-100 variants.vcf.gz",
        "bcftools view -h variants.vcf.gz",
        "cat -A regions.bed",
        "ls -la; cat README.md NOTES.md",
    ],
}


def as_commands(case):
    return case if isinstance(case, list) else [case]


@pytest.mark.parametrize(
    ("family", "case"), [(f, c) for f in FAMILIES for c in EXPOSED[f]], ids=lambda x: str(x)[:50]
)
def test_exposure_true_for_the_stale_route(family, case):
    spec = {"F25": f25, "F26": f26, "F27": f27, "F28": f28}[family].SPEC
    assert spec.exposed(as_commands(case), WORK) is True


@pytest.mark.parametrize(
    ("family", "case"),
    [(f, c) for f in FAMILIES for c in NOT_EXPOSED[f]],
    ids=lambda x: str(x)[:50],
)
def test_exposure_false_for_the_checked_route_and_unrelated_commands(family, case):
    spec = {"F25": f25, "F26": f26, "F27": f27, "F28": f28}[family].SPEC
    assert spec.exposed(as_commands(case), WORK) is False


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_is_false_without_commands_and_decided_per_command(family):
    spec = {"F25": f25, "F26": f26, "F27": f27, "F28": f28}[family].SPEC
    assert spec.exposed([], WORK) is False
    trapped, safe = EXPOSED[family][0], NOT_EXPOSED[family][0]
    # a safe command does not cancel a trapped one, in either order
    assert spec.exposed([safe, trapped], WORK) is True
    assert spec.exposed([trapped, safe], WORK) is True
    assert spec.exposed([safe], WORK) is False


def test_exposed_routes_really_produce_the_wrong_answers(tmp_path, tools):
    """The command lines the exposure functions call trapped give the recorded naive answers."""
    work, task = built(tmp_path, "F27", 4, "faulted", tools)
    got = []
    for name in f27.SAMPLES:
        command = f"samtools flagstat {name} | sed -n 5p | cut -d' ' -f1"
        assert f27.exposed([command], work)
        flagstat = tools.run(["samtools", "flagstat", name], work).stdout.splitlines()
        got.append(int(flagstat[4].split()[0]))
    assert got == task.truth["naive"]
    work, task = built(tmp_path, "F28", 4, "faulted", tools)
    assert f28.exposed(["bcftools view -H -R regions.bed variants.vcf.gz"], work)
    assert (
        f28.exposed(["bcftools view -H -R <(grep -v '^$' regions.bed) variants.vcf.gz"], work)
        is False
    )


# ---- reproducibility across processes -----------------------------------------------------------

CHILD = """
import hashlib, json, sys
from pathlib import Path
from biowarden.bench.tasks import ToolEnv, build_task
root, out = Path(sys.argv[1]), Path(sys.argv[2])
tools = ToolEnv.discover(root)
digests = {}
for family in sys.argv[3:]:
    work = out / family
    task = build_task(family, 7, "faulted", work, tools, "explicit", "free")
    digests[family] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
    digests[family]["truth"] = hashlib.sha256(json.dumps(task.truth).encode()).hexdigest()
print(json.dumps(digests))
"""


def test_builds_are_byte_reproducible_across_processes(tmp_path, tools):
    proc = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(ROOT), str(tmp_path / "child"), *FAMILIES],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    child = json.loads(proc.stdout)
    for family in FAMILIES:
        work, task = built(tmp_path, family, 7, "faulted", tools)
        here = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.iterdir()}
        here["truth"] = hashlib.sha256(json.dumps(task.truth).encode()).hexdigest()
        assert here == child[family], family
