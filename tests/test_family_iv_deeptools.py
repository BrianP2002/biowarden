"""Family-specific checks for F73 (matrix_missing) and F76 (ignore_dups), the deepTools families.

The generic gates live in test_extra_families.py. Here, on seeds 0-29 and both tiers: the trap
changes the answer on every seed (pure Python from the plans), the twins hold the same biology and
differ only in the declared files, the real deepTools 3.5.6 reproduces the recorded trap answer and
stays silent about it, the exposure functions see realistic trapped commands and not the safe ones
(on a faulted workspace, and not on a clean one), builds are byte-reproducible across processes,
and the F76 rules (and the bamCoverage units rule) are measured on calls labelled by running the
tools (bench/admission/cases_iv_deeptools.py, 20 seeds, not the 300-seed calibration).
"""

from __future__ import annotations

import ast
import gzip
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import pytest
from family_forms import typed_forms

from biowarden.bench.admission import measure
from biowarden.bench.score import score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f73_matrix_missing as f73
from biowarden.bench.tasks.extra import f76_ignore_dups as f76
from biowarden.bench.tasks.extra import load
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
FAMILIES = ("F73", "F76")
SPECS = load()
ADMISSION = range(min(ADMISSION_SEEDS, 20))
MIN_RIGHT = max(1, 200 * len(ADMISSION) // 300)
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F73 and F76 are not registered"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    if len(env.real_bins(["computeMatrix", "multiBamSummary", "bamCoverage"])) < 3:
        pytest.skip("deepTools is not installed")
    return env


def with_bio(tools, family="F73"):
    """F73 writes its bigWig with pyBigWig through python-bio; without it only F73 skips."""
    if family == "F73" and not tools.real_bins(["python-bio"]):
        pytest.skip("python-bio is not installed")
    return tools


@pytest.fixture(scope="module")
def bio_tools(tools):
    return with_bio(tools)


def built(tmp_path, family, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


@pytest.mark.parametrize(
    ("family", "slot", "locus"),
    [("F73", "missing_value", "tool_default"), ("F76", "flag_semantics", "tool_default")],
)
def test_locus_slot_and_source_are_declared(family, slot, locus):
    spec = SPECS[family]
    assert (spec.locus, spec.slot) == (locus, slot) and spec.source
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")
    assert "README.md" in spec.twin_files and len(spec.twin_files) <= 3


# ---- the trap, family by family, on seeds 0-29 (pure Python from the plans) ---------------------


def test_f73_the_nan_skipping_mean_is_far_above_the_truth_on_every_seed():
    for seed in SEEDS:
        plan = f73.make_plan(seed)
        expected, naive = f73.truth_values(plan)
        assert naive - expected >= 3.0 and naive >= 1.6 * expected, seed
        assert 14 <= len(plan.genes) <= 16
        silent = sum(1 for g in plan.genes if not any(g.bins))
        assert silent in (4, 5), seed
        assert all(sum(1 for b in g.bins if b) >= 8 for g in plan.genes if any(g.bins))
        # genes are 1000 bp on the 50-bp grid, do not overlap and sit inside the contigs
        spans = sorted((g.chrom, g.start, g.start + f73.GENE_LEN) for g in plan.genes)
        assert all(a[0] != b[0] or a[2] <= b[1] for a, b in zip(spans, spans[1:], strict=False)), (
            seed
        )
        assert all(s % f73.BIN == 0 and e <= f73.CHROMS[c] for c, s, e in spans)
        assert any(plan.flanks), seed  # sparse tracks have entries outside the genes too


def test_f73_clean_and_faulted_hold_the_same_signal(tmp_path, bio_tools):
    for seed in (0, 7):
        clean_work, clean = built(tmp_path, "F73", seed, "clean", bio_tools)
        faulted_work, faulted = built(tmp_path, "F73", seed, "faulted", bio_tools)
        assert clean.truth["expected"] == faulted.truth["expected"]
        assert clean.truth["naive"] == clean.truth["expected"] != faulted.truth["naive"]
        assert (clean_work / "genes.bed").read_bytes() == (faulted_work / "genes.bed").read_bytes()
        assert f73.is_sparse(faulted_work) and not f73.is_sparse(clean_work)


def test_f76_ignore_duplicates_overcounts_every_region_set_and_the_clean_twin_is_untouched():
    for seed in SEEDS:
        faulted = f76.make_plan(seed, "faulted")
        clean = f76.make_plan(seed, "clean")
        expected = f76.truth_counts(faulted)
        assert expected == f76.truth_counts(clean) and len(expected) == 10
        assert f76.ignore_duplicates_counts(clean) == expected, seed
        extra = {k: v - expected[k] for k, v in f76.ignore_duplicates_counts(faulted).items()}
        assert all(v >= 0 for v in extra.values()) and sum(extra.values()) >= 40, seed
        assert sum(1 for v in extra.values() if v >= 3) >= 6, seed
        assert f76.surviving_reverse_duplicates(faulted) >= 30, seed
        assert f76.surviving_reverse_duplicates(clean) == 0
        assert [r for r in faulted.reads if not r.dup] == [r for r in clean.reads if not r.dup]
        assert sum(r.dup for r in faulted.reads) == sum(r.dup for r in clean.reads)
        # unflagged reads never share a start and strand, so --ignoreDuplicates drops only dups
        starts = [(r.chrom, r.start, r.reverse) for r in faulted.reads if not r.dup]
        assert len(starts) == len(set(starts)), seed


def test_f76_forward_duplicates_copy_their_original_and_reverse_ones_keep_the_five_prime_end():
    faulted = f76.make_plan(3, "faulted")
    originals = {(r.chrom, r.start, r.reverse): r for r in faulted.reads if not r.dup}
    trimmed = 0
    for dup in (r for r in faulted.reads if r.dup):
        if not dup.reverse:
            assert (dup.chrom, dup.start, dup.reverse) in originals
        elif (dup.chrom, dup.start, True) not in originals:
            trimmed += 1
            assert any(
                o.reverse and o.chrom == dup.chrom and o.end == dup.end for o in originals.values()
            )
    assert trimmed > 20


# ---- twins and tiers ----------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("tier", ["explicit", "implicit"])
def test_twins_share_prompt_and_differ_only_in_declared_files(tmp_path, tools, family, tier):
    with_bio(tools, family)
    for seed in (1, 12):
        work_c, clean = built(tmp_path, family, seed, "clean", tools, tier)
        work_f, faulted = built(tmp_path, family, seed, "faulted", tools, tier)
        assert (
            clean.prompt == faulted.prompt and clean.truth["expected"] == faulted.truth["expected"]
        )
        assert {p.name for p in work_c.iterdir()} == {p.name for p in work_f.iterdir()}
        changed = {
            p.name for p in work_c.iterdir() if p.read_bytes() != (work_f / p.name).read_bytes()
        }
        assert changed <= SPECS[family].twin_files
        assert ("README.md" in changed) == (tier == "explicit")
        assert changed >= ({"signal.bw"} if family == "F73" else {"reads.bam"})
        assert not (work_c / "result.json").exists()
        readme = (work_f / "README.md").read_text()
        trap_word = "absent" if family == "F73" else "trimmed"
        assert (trap_word in readme) == (tier == "explicit")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_forced_route_names_the_deeptools_program(tmp_path, tools, family):
    with_bio(tools, family)
    _, free = built(tmp_path, family, 2, "faulted", tools, "implicit", "free")
    _, forced = built(tmp_path, family, 2, "faulted", tools, "implicit", "forced")
    program = "computeMatrix" if family == "F73" else "multiBamSummary"
    assert program in forced.prompt and program not in free.prompt
    assert forced.truth == free.truth


# ---- the real tools reproduce the recorded answers and say nothing about the trap ---------------


@pytest.mark.parametrize("seed", [0, 5, 17])
def test_f73_real_computematrix_matches_the_recorded_naive_answer_silently(
    tmp_path, bio_tools, seed
):
    tools = bio_tools
    for twin in ("clean", "faulted"):
        work, task = built(tmp_path, "F73", seed, twin, tools)
        with_flag = f73.matrix_cells(work, tools, "--missingDataAsZero")
        assert sum(with_flag) / len(with_flag) == pytest.approx(task.truth["expected"], abs=1e-3)
        cells = f73.matrix_cells(work, tools)
        assert len(cells) == len(with_flag) == f73.GENE_BINS * len(f73.make_plan(seed).genes)
        assert f73.mean_skipping_nan(cells) == pytest.approx(task.truth["naive"], abs=1e-3)
        assert (twin == "faulted") == any(c != c for c in cells)
        # the reference route, the naive route and the scorer
        assert score_result(task.truth, f73.reference(task, work, tools)).outcome == "pass"
        verdict = score_result(task.truth, f73.naive(task, work, tools))
        assert verdict.outcome == ("silent_wrong" if twin == "faulted" else "pass")
    argv = [
        "computeMatrix",
        "scale-regions",
        "-S",
        "signal.bw",
        "-R",
        "genes.bed",
        "-b",
        "0",
        "-a",
        "0",
        "--regionBodyLength",
        "1000",
        "--binSize",
        "50",
        "-p",
        "1",
        "-o",
        "x.gz",
    ]
    proc = tools.run(argv, work)
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""


@pytest.mark.parametrize("seed", [0, 5, 17])
def test_f76_real_multibamsummary_matches_the_recorded_naive_answer_silently(tmp_path, tools, seed):
    for twin in ("clean", "faulted"):
        work, task = built(tmp_path, "F76", seed, twin, tools)
        assert f76.summary_counts(work, tools, "--samFlagExclude", "1024") == task.truth["expected"]
        assert f76.summary_counts(work, tools, "--ignoreDuplicates") == task.truth["naive"]
        assert f76.reference(task, work, tools)["answer"] == task.truth["expected"]
        verdict = score_result(task.truth, f76.naive(task, work, tools))
        assert verdict.outcome == ("silent_wrong" if twin == "faulted" else "pass")
        assert verdict.signature is (True if twin == "faulted" else None)
    argv = [
        "multiBamSummary",
        "BED-file",
        "--BED",
        "regions.bed",
        "-b",
        "reads.bam",
        "-p",
        "1",
        "--ignoreDuplicates",
        "-o",
        "x.npz",
    ]
    proc = tools.run(argv, work)
    assert proc.returncode == 0 and proc.stdout == ""
    assert "dup" not in proc.stderr.lower() and "warn" not in proc.stderr.lower()


def test_f76_the_python_overlap_model_agrees_with_samtools_and_the_recorded_answer(tmp_path, tools):
    work, task = built(tmp_path, "F76", 9, "faulted", tools)
    for (chrom, start, end), name in f76.region_names(work).items():
        count = tools.run(
            ["samtools", "view", "-c", "-F", "1024", "reads.bam", f"{chrom}:{start + 1}-{end}"],
            work,
        )
        assert int(count.stdout) == task.truth["expected"][name]
        everything = tools.run(
            ["samtools", "view", "-c", "reads.bam", f"{chrom}:{start + 1}-{end}"], work
        )
        assert int(everything.stdout) > task.truth["expected"][name]  # flagged reads are counted


def test_f76_bamcoverage_default_counts_flagged_duplicates_and_ignoreduplicates_keeps_the_rest(
    tmp_path, tools
):
    work, _ = built(tmp_path, "F76", 4, "faulted", tools)

    def total(*extra):
        argv = [
            "bamCoverage",
            "-b",
            "reads.bam",
            "-o",
            "o.bg",
            "-of",
            "bedgraph",
            "-p",
            "1",
            "--binSize",
            "1",
            *extra,
        ]
        assert tools.run(argv, work).returncode == 0
        return sum(
            (int(f[2]) - int(f[1])) * float(f[3])
            for f in (line.split("\t") for line in (work / "o.bg").read_text().splitlines())
        )

    plan = f76.make_plan(4, "faulted")
    every = sum(r.length for r in plan.reads)
    unflagged = sum(r.length for r in plan.reads if not r.dup)
    assert total() == every
    assert total("--samFlagExclude", "1024") == unflagged
    assert unflagged < total("--ignoreDuplicates") < every  # the trimmed reverse duplicates stay


# ---- the exposure checks read the files themselves, and F73 skips without python-bio -------------

PYBIGWIG_COVERED = (
    "import pyBigWig; b = pyBigWig.open('signal.bw'); print(b.header()['nBasesCovered'], "
    "sum(e - s for c in b.chroms() for s, e, _ in b.intervals(c)))"
)


def test_f73_bases_covered_is_what_pybigwig_reports_and_tells_the_twins_apart(tmp_path, bio_tools):
    for seed in (0, 7, 12):
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F73", seed, twin, bio_tools)
            proc = bio_tools.run(["python-bio", "-c", PYBIGWIG_COVERED], work)
            header, summed = map(int, proc.stdout.split())
            assert f73.bases_covered(work / "signal.bw") == header == summed, (seed, twin)
            assert f73.is_sparse(work) is (twin == "faulted"), (seed, twin)
    junk = tmp_path / "junk.bw"
    junk.write_bytes(b"not a bigwig" * 8)
    assert f73.bases_covered(junk) is None and f73.bases_covered(tmp_path / "absent.bw") is None
    assert f73.is_sparse(tmp_path) is False  # no signal.bw: nothing to be sparse


def test_f76_bam_records_are_what_samtools_view_prints_and_tell_the_twins_apart(
    tmp_path, tools, monkeypatch
):
    for seed in (0, 5, 12):
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F76", seed, twin, tools)
            out = tools.run(["samtools", "view", "reads.bam"], work).stdout.splitlines()
            rows = [line.split("\t") for line in out]
            expected = [(list(f76.CHROMS).index(r[2]), int(r[3]) - 1, int(r[1])) for r in rows]
            assert f76.bam_records(work / "reads.bam") == expected, (seed, twin)
            assert f76.has_unmatched_duplicates(work) is (twin == "faulted"), (seed, twin)
    raw = (work / "reads.bam").read_bytes()
    (tmp_path / "cut.bam").write_bytes(raw[:60])
    (tmp_path / "text.bam").write_bytes(gzip.compress(b"this is not a BAM"))
    for name in ("cut.bam", "text.bam", "absent.bam"):
        assert f76.bam_records(tmp_path / name) is None, name
    monkeypatch.setattr(f76, "BAM_LIMIT", 200)  # a BAM that unpacks to more is not read
    assert f76.bam_records(work / "reads.bam") is None
    assert f76.has_unmatched_duplicates(tmp_path) is False  # no reads.bam


@pytest.mark.parametrize("module", [f73, f76])
def test_the_exposure_readers_do_not_import_engine_code(module):
    """The label of an exposed trial must not depend on the code that the layer arms test."""
    tree = ast.parse(inspect.getsource(module))
    found = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    found |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert found and not any(name.startswith("biowarden.engine") for name in found)
    assert not hasattr(module, "decode_bigwig") and not hasattr(module, "aln_bam")


def test_f73_skips_cleanly_when_python_bio_is_absent(tmp_path):
    bare = ToolEnv((tmp_path / "no-bin",))  # no python-bio in any tool directory
    with pytest.raises(unittest.SkipTest, match="python-bio"):
        build_task("F73", 0, "faulted", tmp_path / "work", bare, "implicit")
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "from pathlib import Path\n"
        "from biowarden.bench.tasks import ToolEnv, build_task\n"
        "def test_build(tmp_path):\n"
        "    env = ToolEnv((tmp_path / 'no-bin',))\n"
        "    build_task('F73', 0, 'clean', tmp_path / 'w', env, 'implicit')\n"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert "1 skipped" in run.stdout and "failed" not in run.stdout, run.stdout


# ---- exposure -----------------------------------------------------------------------------------


F73_NAIVE = [
    "computeMatrix scale-regions -S signal.bw -R genes.bed -b 0 -a 0 -o m.gz",
    "computeMatrix reference-point -S signal.bw -R genes.bed -a 1000 "
    "--outFileNameMatrix m.tab -o m.gz",
    "computeMatrix scale-regions -S signal.bw -R genes.bed --skipZeros -o m.gz",
    "python-bio -c \"import pyBigWig; b = pyBigWig.open('signal.bw'); "
    "print(b.stats('chr1', 0, 1000))\"",
    'python-bio -c "import numpy as np, pyBigWig; '
    "v = pyBigWig.open('signal.bw').values('chr1', 0, 9); print(np.nanmean(v))\"",
    "bigWigSummary signal.bw chr1 0 1000 1",
]
F73_SAFE = [
    "ls -la; cat README.md; head genes.bed",
    "computeMatrix scale-regions -S signal.bw -R genes.bed -o m.gz --missingDataAsZero",
    'python-bio -c "import numpy as np, pyBigWig; '
    "v = np.nan_to_num(pyBigWig.open('signal.bw').values('chr1', 0, 9)); print(v.mean())\"",
    "python-bio -c \"import pyBigWig; print(pyBigWig.open('signal.bw').chroms())\" || true; "
    "python3 -c 'print(1)'",
    "awk '{print $4}' genes.bed",
]


def test_f73_exposure_sees_the_nan_skipping_route_only_on_the_sparse_track(tmp_path, bio_tools):
    faulted, _ = built(tmp_path, "F73", 2, "faulted", bio_tools)
    clean, _ = built(tmp_path, "F73", 2, "clean", bio_tools)
    spec = SPECS["F73"]
    for command in F73_NAIVE:
        assert spec.exposed([command], faulted) is True, command
        assert spec.exposed([command], clean) is False, command
    for command in F73_SAFE:
        assert spec.exposed([command], faulted) is False, command
    # the model that tried the default and then added the flag still ran the trapped route
    assert spec.exposed([F73_NAIVE[0], F73_SAFE[1]], faulted) is True
    assert spec.exposed([], faulted) is False


F76_NAIVE = [
    "multiBamSummary BED-file --BED regions.bed -b reads.bam --ignoreDuplicates -o o.npz",
    "multiBamSummary BED-file --BED regions.bed -b reads.bam -o o.npz --outRawCounts o.tab "
    "-p 4 --ignoreDuplicates",
    "bamCoverage -b reads.bam -o o.bw --ignoreDuplicates --binSize 1",
    "bamCoverage -b reads.bam -o o.bw --samFlagExclude 4 --ignoreDuplicates",
    "multiBamSummary BED-file --BED regions.bed -b reads.bam --ignoreDuplicates -o o.npz; "
    "multiBamSummary BED-file --BED regions.bed -b reads.bam --samFlagExclude 1024 -o p.npz",
]
F76_SAFE = [
    "ls -la; samtools view -c -F 1024 reads.bam chr1:1-100",
    "multiBamSummary BED-file --BED regions.bed -b reads.bam --samFlagExclude 1024 -o o.npz",
    "multiBamSummary BED-file --BED regions.bed -b reads.bam --ignoreDuplicates "
    "--samFlagExclude 1024 -o o.npz",
    "multiBamSummary BED-file --BED regions.bed -b reads.bam --samFlagExclude 0x400 -o o.npz",
    "multiBamSummary BED-file --BED regions.bed -b reads.bam -o o.npz",
    "samtools markdup -r reads.bam out.bam",
    "grep -c ignoreDuplicates README.md",
]


def test_f76_exposure_sees_ignoreduplicates_without_the_flag_filter_on_the_faulted_bam(
    tmp_path, tools
):
    faulted, _ = built(tmp_path, "F76", 2, "faulted", tools)
    clean, _ = built(tmp_path, "F76", 2, "clean", tools)
    spec = SPECS["F76"]
    for command in F76_NAIVE:
        assert spec.exposed([command], faulted) is True, command
        assert spec.exposed([command], clean) is False, command
    for command in F76_SAFE:
        assert spec.exposed([command], faulted) is False, command
    assert spec.exposed([], faulted) is False


def test_f73_exposure_reads_a_command_the_way_the_shell_does(tmp_path, bio_tools):
    faulted, _ = built(tmp_path, "F73", 2, "faulted", bio_tools)
    clean, _ = built(tmp_path, "F73", 2, "clean", bio_tools)
    spec = SPECS["F73"]
    for command in F73_NAIVE:
        for form in typed_forms(command):
            assert spec.exposed([form], faulted) is True, form
            assert spec.exposed([form], clean) is False, form
    for command in F73_SAFE:
        for form in typed_forms(command):
            assert spec.exposed([form], faulted) is False, form
    # a fill in another statement of the command does not undo the call that ran without it
    later = "python3 -c 'import numpy as np; print(np.nan_to_num(np.loadtxt(\"m.tab\")).mean())'"
    call = "computeMatrix scale-regions -S signal.bw -R genes.bed -o m.gz --outFileNameMatrix m.tab"
    assert spec.exposed([f"{call}; {later}"], faulted) is True
    assert spec.exposed([f"{call} --missingDataAsZero; {later}"], faulted) is False
    assert spec.exposed(["cd /work && " + call.replace("signal.bw", "/work/signal.bw")], faulted)


def test_f76_exposure_reads_a_command_the_way_the_shell_does(tmp_path, tools):
    faulted, _ = built(tmp_path, "F76", 2, "faulted", tools)
    clean, _ = built(tmp_path, "F76", 2, "clean", tools)
    spec = SPECS["F76"]
    for command in F76_NAIVE:
        for form in typed_forms(command):
            assert spec.exposed([form], faulted) is True, form
            assert spec.exposed([form], clean) is False, form
    for command in F76_SAFE:
        for form in typed_forms(command):
            assert spec.exposed([form], faulted) is False, form
    # the flag filter sits in another statement of the command: the trapped call ran without it
    other = "multiBamSummary BED-file --BED regions.bed -b reads.bam --samFlagExclude 1024 -o p.npz"
    trapped = "multiBamSummary BED-file --BED regions.bed -b reads.bam --ignoreDuplicates -o o.npz"
    assert spec.exposed([f"{other}; {trapped}"], faulted) is True
    again = f"{trapped} --samFlagExclude 1024 \\\n  --outRawCounts c.tab"
    assert spec.exposed([again], faulted) is False


# ---- reproducibility ----------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_reproducible_across_processes(tmp_path, tools, family):
    with_bio(tools, family)
    here, _ = built(tmp_path, family, 7, "faulted", tools)
    other = tmp_path / "subprocess"
    code = (
        "import sys; from pathlib import Path\n"
        "from biowarden.bench.tasks import ToolEnv, build_task\n"
        "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
        "build_task(sys.argv[2], 7, 'faulted', Path(sys.argv[3]), tools, 'implicit', 'free')\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), family, str(other)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert {p.name for p in here.iterdir()} == {p.name for p in other.iterdir()}
    for p in here.iterdir():
        assert p.read_bytes() == (other / p.name).read_bytes(), (family, p.name)


# ---- admission of the rules that belong to F76 (and the units rule on its BAM) ------------------


@pytest.mark.parametrize(
    "rule_id",
    [
        "deeptools.duplicates_semantics",
        "deeptools.multibamsummary_regions",
        "deeptools.bamcoverage_track_units",
    ],
)
def test_admission_of_the_bam_rules(tools, tmp_path, rule_id):
    spec = load_packaged("iv_deeptools")
    stats = measure(rule_id, [spec], ADMISSION, tools, tmp_path)
    right_declared, right_undeclared = stats.cells[("right", True)], stats.cells[("right", False)]
    assert stats.admitted() and stats.killed() > 0
    assert right_declared.n >= MIN_RIGHT and right_declared.failed == 0
    assert all(cell.crashed == 0 and cell.unknown == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed > 0
    if rule_id.endswith("track_units"):
        assert not stats.conditional  # a required declaration: every undeclared call is refused
        assert right_undeclared.failed == right_undeclared.n > 0
    else:
        assert stats.conditional and right_undeclared.n >= MIN_RIGHT
        assert right_undeclared.failed == 0
    assert promotion_is_consistent(stats)  # fewer than 200 correct calls: the rule stays warn
