"""Family-specific checks for F72 (iv pack: MACS3 reads a paired-end BAM as single-end tags).

The generic gates live in test_extra_families.py. Here: the plan over seeds 0-29 (trap size, the
twins share loci and fragments and differ only in the distance between the two sites of each
locus), the README tiers, the real MACS3 on a few seeds (BAMPE gives the planted peaks, the
default format gives the recorded naive count, exit 0 and no warning line), exposure on realistic
commands, byte reproducibility across processes and the scorer on the two scripted solvers.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from family_forms import typed_forms

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f72_macs3_bampe as f72
from biowarden.bench.tasks.extra import load

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
SPEC = load().get("F72")
pytestmark = pytest.mark.skipif(SPEC is None, reason="F72 is not registered")


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "macs3" not in env.real_bins(["macs3"]):
        pytest.skip("macs3 (or samtools/bcftools/bedtools) is not available")
    return env


def built(tmp_path, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"F72-{seed}-{twin}-{tier}-{route}"
    return work, build_task("F72", seed, twin, work, tools, tier, route)


def test_locus_slot_and_source_are_declared():
    assert (SPEC.locus, SPEC.slot) == ("tool_default", "read_layout")
    assert "GEN-26" in SPEC.source and "MACS#486" in SPEC.source
    assert SPEC.tiers == ("explicit", "implicit") and SPEC.routes == ("free", "forced")
    assert SPEC.twin_files == frozenset({"reads.bam", "reads.bam.bai"})


# ---- the plan, seeds 0-29 (pure Python) --------------------------------------------------------


def test_trap_size_and_the_clean_twin_over_seeds_0_29():
    sizes = []
    for seed in SEEDS:
        clean, faulted = f72.make_plan(seed, "clean"), f72.make_plan(seed, "faulted")
        assert clean.expected == faulted.expected == 2 * len(clean.loci) == 2 * (3 + seed % 3)
        assert faulted.merged == len(faulted.loci) < faulted.expected
        assert clean.loci == faulted.loci and clean.contigs == faulted.contigs
        assert f72.GAP_CLEAN[0] <= clean.gap <= f72.GAP_CLEAN[1]
        assert f72.GAP_FAULTED[0] <= faulted.gap <= f72.GAP_FAULTED[1]
        sizes.append(faulted.expected - faulted.merged)
    assert min(sizes) == 3 and max(sizes) == 5  # peaks missing from the naive answer


def test_twins_hold_the_same_fragments_and_differ_only_in_the_gap():
    for seed in SEEDS:
        clean, faulted = f72.make_plan(seed, "clean"), f72.make_plan(seed, "faulted")
        assert Counter(p[2] for p in clean.pairs) == Counter(p[2] for p in faulted.pairs)
        assert Counter((p[0], p[3]) for p in clean.pairs) == Counter(
            (p[0], p[3]) for p in faulted.pairs
        )
        # sites sit symmetrically around the same centres: the mean start per contig differs
        # little, the spread of the site reads is the gap
        assert len(clean.pairs) == len(faulted.pairs)
        assert 60 <= min(p[2] for p in faulted.pairs) and max(p[2] for p in faulted.pairs) <= 109


def test_background_is_clear_of_the_loci_and_fragments_are_short():
    for seed in (0, 7, 19):
        plan = f72.make_plan(seed, "faulted")
        near = [
            p
            for p in plan.pairs
            if any(c == p[0] and abs(p[1] + p[2] // 2 - x) < 400 for c, x in plan.loci)
        ]
        assert len(near) >= 2 * 55 * len(plan.loci) - 10  # the site fragments
        assert len(plan.pairs) - len(near) >= 90  # and a spread-out background


# ---- the workspace -----------------------------------------------------------------------------


def test_workspace_is_flat_and_the_tiers_differ_in_the_readme_only(tmp_path, tools):
    work_e, explicit = built(tmp_path, 2, "faulted", tools, "explicit")
    work_i, implicit = built(tmp_path, 2, "faulted", tools, "implicit")
    assert sorted(p.name for p in work_e.iterdir()) == [
        "README.md",
        "genome.sizes",
        "reads.bam",
        "reads.bam.bai",
    ]
    assert all(p.is_file() for p in work_e.iterdir())
    assert (work_e / "reads.bam").read_bytes() == (work_i / "reads.bam").read_bytes()
    readme_e, readme_i = (work_e / "README.md").read_text(), (work_i / "README.md").read_text()
    assert "Paired-end" in readme_e and "Paired-end" not in readme_i
    assert "paired" not in readme_i.lower() and "mate" not in readme_i.lower()
    for text in (readme_e, readme_i):
        assert "genome.sizes" in text and "effective genome size" in text
    assert implicit.prompt == explicit.prompt
    assert (
        "macs3 callpeak"
        in build_task("F72", 2, "faulted", tmp_path / "forced", tools, "implicit", "forced").prompt
    )


def test_the_flags_carry_the_layout_that_the_implicit_readme_hides(tmp_path, tools):
    work, _ = built(tmp_path, 4, "faulted", tools)
    flags = Counter(
        int(line.split("\t")[1])
        for line in tools.run(["samtools", "view", "reads.bam"], work).stdout.splitlines()
    )
    assert set(flags) == {99, 147, 163, 83}
    assert flags[99] + flags[163] == flags[147] + flags[83]  # one first and one second mate each
    assert (work / "genome.sizes").read_text().splitlines()[0].split("\t")[0] == "chr1"


# ---- the real MACS3 on seeds 0-3 ---------------------------------------------------------------


def macs3(tools, work, tmp_path, *options, name="run"):
    out = tmp_path / f"out-{work.name}-{name}"
    out.mkdir(parents=True, exist_ok=True)
    argv = ["macs3", "callpeak", "-t", "reads.bam", "-g", str(f72.genome_size(work)), "-q", "0.01"]
    proc = tools.run([*argv, "-n", name, "--outdir", str(out), *options], work)
    peaks = out / f"{name}_peaks.narrowPeak"
    count = len(peaks.read_text().splitlines()) if peaks.exists() else None
    return proc, count, out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_macs3_bampe_finds_the_planted_peaks_and_the_default_format_merges_them(
    tmp_path, tools, seed
):
    for twin in ("clean", "faulted"):
        work, task = built(tmp_path, seed, twin, tools)
        proc, count, _ = macs3(tools, work, tmp_path, "-f", "BAMPE", name="pe")
        assert proc.returncode == 0 and count == task.truth["expected"], (seed, twin)
        proc, count, _ = macs3(tools, work, tmp_path, "--nomodel", "--extsize", "200", name="se")
        assert proc.returncode == 0 and count == task.truth["naive"], (seed, twin)
        assert (count < task.truth["expected"]) == (twin == "faulted")


def test_the_trap_is_silent_exit_0_and_no_warning_line(tmp_path, tools):
    work, task = built(tmp_path, 1, "faulted", tools)
    proc, count, out = macs3(tools, work, tmp_path, "--nomodel", "--extsize", "200", name="se")
    assert proc.returncode == 0 and count == task.truth["naive"]
    assert not any(word in proc.stderr for word in ("WARNING", "ERROR", "Error", "Traceback"))
    header = (out / "se_peaks.xls").read_text()
    assert "# Paired-End mode is off" in header  # stated, never flagged
    pairs = len(f72.make_plan(1, "faulted").pairs)
    assert f"# total tags in treatment: {pairs}" in header  # one tag per pair: mate 2 is ignored
    proc, _, out = macs3(tools, work, tmp_path, "-f", "BAMPE", name="pe")
    assert f"# total fragments in treatment: {pairs}" in (out / "pe_peaks.xls").read_text()
    assert "# Paired-End mode is on" in (out / "pe_peaks.xls").read_text()


def test_a_model_run_on_this_small_data_stops_loudly(tmp_path, tools):
    work, _ = built(tmp_path, 1, "faulted", tools)
    proc, count, _ = macs3(tools, work, tmp_path, "-f", "BAM", name="model")
    assert proc.returncode != 0 and count is None
    assert "needs at least 100 paired peaks" in proc.stderr + proc.stdout


# ---- exposure ----------------------------------------------------------------------------------

TRAPPED = [
    "macs3 callpeak -t reads.bam -g 50000 -q 0.01 -n x --nomodel --extsize 200",
    "macs3 callpeak -t reads.bam -f BAM -g 50000 -q 0.01 -n x --nomodel --extsize 147",
    "macs3 callpeak -t reads.bam --format=AUTO -g 50000 -n x --nomodel",
    "macs3 callpeak -t reads.bam -fBAM -g 50000 -n x --nomodel --extsize 100 && wc -l x_peaks.xls",
    "ls; macs3 callpeak -t reads.bam -g 50000 -n x --nomodel --extsize 200 | tail -2",
    "cd . && macs3 callpeak -t reads.bam -g 50000 -q 0.01 -n x --nomodel --extsize 200",
    "python3 -m MACS3 callpeak -t reads.bam -g 50000 -q 0.01 -n x",  # the same program
    "macs3 callpeak \\\n  -t reads.bam \\\n  -g 50000 -q 0.01 -n x",
    "macs3 callpeak -t /work/reads.bam -g 50000 -f 'BAM' -n x",
    "macs3 callpeak -t reads.bam -g 50000 -q 0.01 -n x; macs3 callpeak -t reads.bam -fBAMPE -n y",
]
SAFE = [
    "macs3 callpeak -t reads.bam -f BAMPE -g 50000 -q 0.01 -n x",
    "macs3 callpeak -t reads.bam --format=BAMPE -g 50000 -q 0.01 -n x",
    "macs3 callpeak -t reads.bam --format BAMPE -g 50000 -n x --nomodel --extsize 200",
    "macs3 callpeak -t reads.bam -fBAMPE -g 50000 -q 0.01 -n x",  # the value attached to -f
    "macs3 callpeak -t reads.bam -f 'BAMPE' -g 50000 -n x",
    "python3 -m MACS3 callpeak -t reads.bam -f BAMPE -g 50000 -n x",
    "macs3 callpeak \\\n  -t /work/reads.bam \\\n  -f BAMPE -n x",
    "samtools view -c -f 1 reads.bam; samtools flagstat reads.bam",
    "ls -la; cat README.md genome.sizes",
    "macs3 --version",
]


@pytest.mark.parametrize("command", TRAPPED)
def test_exposure_sees_the_single_end_route(tmp_path, tools, command):
    work, _ = built(tmp_path, 1, "faulted", tools)
    assert SPEC.exposed([command], work) is True


@pytest.mark.parametrize("command", SAFE)
def test_exposure_ignores_bampe_and_unrelated_commands(tmp_path, tools, command):
    work, _ = built(tmp_path, 1, "faulted", tools)
    assert SPEC.exposed([command], work) is False


def test_a_corrected_rerun_does_not_hide_the_first_trapped_run(tmp_path, tools):
    work, _ = built(tmp_path, 1, "faulted", tools)
    session = [TRAPPED[0], SAFE[0]]
    assert SPEC.exposed(session, work) is True
    other = "samtools view -c -f 1 reads.bam; samtools flagstat reads.bam"
    assert other in SAFE and SPEC.exposed([SAFE[0], other], work) is False


@pytest.mark.parametrize("command", [*TRAPPED, *SAFE])
def test_exposure_reads_a_command_the_way_the_shell_does(tmp_path, tools, command):
    work, _ = built(tmp_path, 1, "faulted", tools)
    wanted = command in TRAPPED
    for form in typed_forms(command):
        assert SPEC.exposed([form], work) is wanted, form


# ---- gates the generic file does not cover -----------------------------------------------------


def test_the_scorer_marks_the_naive_answer_silent_wrong_with_the_trap_signature(tmp_path, tools):
    for seed in (0, 1):
        work, task = built(tmp_path, seed, "faulted", tools)
        score = score_result(task.truth, SPEC.naive(task, work, tools))
        assert score.outcome == "silent_wrong" and score.signature is True
        assert score_result(task.truth, SPEC.reference(task, work, tools)).outcome == "pass"
        clean_work, clean = built(tmp_path, seed, "clean", tools)
        assert score_result(clean.truth, SPEC.naive(clean, clean_work, tools)).outcome == "pass"
        recorded = {**task.truth, "expected": task.truth["naive"]}
        assert answer_matches(recorded, SPEC.naive(task, work, tools)["answer"])


def test_builds_are_byte_identical_across_processes(tmp_path, tools):
    here, _ = built(tmp_path, 7, "faulted", tools)
    other = tmp_path / "subprocess"
    code = (
        "import sys; from pathlib import Path\n"
        "from biowarden.bench.tasks import ToolEnv, build_task\n"
        "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
        "build_task('F72', 7, 'faulted', Path(sys.argv[2]), tools, 'implicit', 'free')\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), str(other)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert {p.name for p in here.iterdir()} == {p.name for p in other.iterdir()}
    for p in here.iterdir():
        assert p.read_bytes() == (other / p.name).read_bytes(), p.name


def test_the_solvers_leave_the_workspace_untouched(tmp_path, tools):
    work, task = built(tmp_path, 2, "faulted", tools)
    before = {p.name: p.read_bytes() for p in work.iterdir()}
    SPEC.reference(task, work, tools)
    SPEC.naive(task, work, tools)
    assert {p.name: p.read_bytes() for p in work.iterdir()} == before
