"""Family-specific checks for F74 (liftover_summit), pack iv.

The generic gates live in test_extra_families.py. Here, on seeds 0-29 and both tiers: the stale
summit offset is wrong on every peak of the faulted twin and right on every peak of the clean one,
the dropped peaks are the large deletions, the twins hold the same summits and the same chain and
differ only in the peak file, the real liftOver takes the trapped route silently, the exposure
function sees realistic trapped commands and not the safe ones, and builds are byte-reproducible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from family_forms import typed_forms

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f74_liftover_summit as f74
from biowarden.bench.tasks.extra import load

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
SPECS = load()
pytestmark = pytest.mark.skipif("F74" not in SPECS, reason="F74 is not registered")


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    if "liftOver" not in env.real_bins(["liftOver"]):
        pytest.skip("liftOver is not installed")
    return env


def built(tmp_path, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"F74-{seed}-{twin}-{tier}-{route}"
    return work, build_task("F74", seed, twin, work, tools, tier, route)


def test_locus_slot_and_source_are_declared():
    spec = SPECS["F74"]
    assert (spec.locus, spec.slot) == ("tool_default", "derived_column") and spec.source
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")
    assert spec.twin_files == frozenset({"peaks_old.narrowPeak"})


# ---- the plan, seed by seed ----------------------------------------------------------------------


def test_the_plan_puts_every_summit_in_an_aligned_block_after_an_indel():
    for seed in SEEDS:
        peaks, chain = f74.make_plan(seed)
        kinds = [p.kind for p in peaks]
        assert sorted(kinds) == sorted(["ins"] * 5 + ["del_small"] * 2 + ["del_large"] * 2), seed
        assert chain.count("chain ") == 2 and chain.endswith("\n")
        for p in peaks:
            assert p.start_clean >= p.block_t + 20 and p.start_clean < p.summit_old < p.end, p.name
            assert p.start_faulted < p.block_t - (p.k if p.kind != "ins" else 0) < p.summit_old
            assert 110 <= p.summit_old - p.block_t <= 160
            width = p.end - p.start_faulted
            if p.kind == "del_large":
                assert p.k / width > 0.06, (seed, p.name)  # more than 5 percent: dropped
            if p.kind == "del_small":
                assert p.k / width < 0.04, (seed, p.name)  # lifted, shrunk
            assert p.summit_new == p.summit_old + p.shift
        assert len({p.name for p in peaks}) == len(peaks)


def test_the_faulted_naive_answer_is_wrong_by_the_indel_size_and_misses_the_large_deletions(
    tmp_path, tools
):
    for seed in SEEDS:
        peaks, _ = f74.make_plan(seed)
        _, faulted = built(tmp_path, seed, "faulted", tools)
        expected, naive = faulted.truth["expected"], faulted.truth["naive"]
        lost = {p.name for p in peaks if p.kind == "del_large"}
        assert set(expected) - set(naive) == lost and len(lost) == 2, seed
        for p in peaks:
            if p.name in lost:
                continue
            assert abs(naive[p.name] - expected[p.name]) == p.k, (seed, p.name)
            assert (naive[p.name] - expected[p.name] < 0) == (p.kind == "ins"), (seed, p.name)
        assert not answer_matches(faulted.truth, naive)
        _, clean = built(tmp_path, seed, "clean", tools)
        assert clean.truth["naive"] == clean.truth["expected"] == expected, seed


def test_the_clean_naive_route_is_right_through_the_real_tool(tmp_path, tools):
    spec = SPECS["F74"]
    for seed in range(8):
        work, task = built(tmp_path, seed, "clean", tools)
        assert answer_matches(task.truth, spec.naive(task, work, tools)["answer"]), seed


# ---- the twins -------------------------------------------------------------------------------


def test_twins_share_summits_chain_and_names_and_differ_only_in_the_peak_file(tmp_path, tools):
    for seed in (0, 5, 11):
        w_clean, clean = built(tmp_path, seed, "clean", tools)
        w_fault, faulted = built(tmp_path, seed, "faulted", tools)
        assert clean.truth["expected"] == faulted.truth["expected"], seed
        assert (w_clean / "old_to_new.chain").read_bytes() == (
            w_fault / "old_to_new.chain"
        ).read_bytes()
        assert (w_clean / "README.md").read_bytes() == (w_fault / "README.md").read_bytes()
        changed = {
            p.name for p in w_clean.iterdir() if (w_fault / p.name).read_bytes() != p.read_bytes()
        }
        assert changed == {"peaks_old.narrowPeak"}
        rows_c = [
            r.split("\t") for r in (w_clean / "peaks_old.narrowPeak").read_text().splitlines()
        ]
        rows_f = [
            r.split("\t") for r in (w_fault / "peaks_old.narrowPeak").read_text().splitlines()
        ]
        assert [r[3] for r in rows_c] == [r[3] for r in rows_f]
        for c, f in zip(rows_c, rows_f, strict=True):  # the same summit base, another start
            assert int(c[1]) + int(c[9]) == int(f[1]) + int(f[9]) and c[2] == f[2]
            assert int(f[1]) < int(c[1]) and len(c) == len(f) == 10
        assert clean.evidence != faulted.evidence and faulted.evidence


def test_the_explicit_tier_states_the_summit_column_and_the_implicit_one_does_not(tmp_path, tools):
    w_exp, task_exp = built(tmp_path, 2, "faulted", tools, "explicit")
    w_imp, task_imp = built(tmp_path, 2, "faulted", tools, "implicit")
    assert "Column 10 is the summit" in (w_exp / "README.md").read_text()
    assert "Column 10" not in (w_imp / "README.md").read_text()
    assert task_exp.prompt == task_imp.prompt
    assert len(task_exp.evidence) == len(task_imp.evidence) + 1


def test_the_forced_route_dictates_liftover(tmp_path, tools):
    _, free = built(tmp_path, 1, "faulted", tools, "explicit", "free")
    _, forced = built(tmp_path, 1, "faulted", tools, "explicit", "forced")
    assert "liftOver" in forced.prompt and "liftOver" not in free.prompt


# ---- the real tool takes the trapped route silently ------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_liftover_exits_0_with_progress_lines_only_and_drops_the_large_deletions(
    tmp_path, tools, seed
):
    work, task = built(tmp_path, seed, "faulted", tools)
    proc = tools.run(
        ["liftOver", "-bedPlus=6", "peaks_old.narrowPeak", "old_to_new.chain", "o.bed", "u.bed"],
        work,
    )
    assert proc.returncode == 0
    assert proc.stderr.splitlines() == ["Reading liftover chains", "Mapping coordinates"]
    unmapped = (work / "u.bed").read_text().splitlines()
    assert unmapped[0::2] == ["#Partially deleted in new"] * 2
    lifted = [r.split("\t") for r in (work / "o.bed").read_text().splitlines()]
    got = {r[3]: int(r[1]) + int(r[9]) for r in lifted}
    assert got == task.truth["naive"]
    assert len(lifted) == len(task.truth["expected"]) - 2
    # the summit bases, lifted on their own, give the truth
    assert answer_matches(task.truth, f74.summit_positions(work, tools))
    # a ten-column file without -bedPlus stops liftOver: only the -bedPlus route is silent
    bad = tools.run(
        ["liftOver", "peaks_old.narrowPeak", "old_to_new.chain", "o2.bed", "u2.bed"], work
    )
    assert bad.returncode != 0 and "Expecting integer field 7" in bad.stderr


def test_the_lifted_peak_keeps_its_offset_and_grows_over_an_insertion(tmp_path, tools):
    work, task = built(tmp_path, 3, "faulted", tools)
    peaks, _ = f74.make_plan(3)
    inserted = next(p for p in peaks if p.kind == "ins")
    row = next(
        r.split("\t")
        for r in (work / "peaks_old.narrowPeak").read_text().splitlines()
        if r.split("\t")[3] == inserted.name
    )
    lifted = f74.lift(work, tools, ["-bedPlus=6"], "\t".join(row) + "\n")[inserted.name]
    assert lifted[9] == row[9]  # column 10 copied unchanged
    assert int(lifted[2]) - int(lifted[1]) == int(row[2]) - int(row[1]) + inserted.k


# ---- exposure --------------------------------------------------------------------------------


LIFT_PEAKS = "liftOver -bedPlus=6 peaks_old.narrowPeak old_to_new.chain lifted.bed unmapped.bed"
TRAPPED = (
    LIFT_PEAKS,
    "liftOver -minMatch=0.5 -bedPlus=6 ./peaks_old.narrowPeak old_to_new.chain l.bed /dev/null",
    "cd . && " + LIFT_PEAKS + " && awk '{print $2+$10}' lifted.bed",
    "liftOver -bedPlus=6 peaks_old.narrowPeak old_to_new.chain l.bed u.bed 2>&1 | tail -2",
    "liftOver summits.bed old_to_new.chain s.bed u.bed; " + LIFT_PEAKS,
    # the workspace is /work in the sandbox, a line may be continued, a name may be quoted
    "liftOver -bedPlus=6 /work/peaks_old.narrowPeak old_to_new.chain out.bed unmapped.bed",
    "cd /work && liftOver -bedPlus=6 /work/peaks_old.narrowPeak /work/old_to_new.chain o.bed u.bed",
    "liftOver -bedPlus=6 \\\n  peaks_old.narrowPeak \\\n  old_to_new.chain out.bed unmapped.bed",
    "liftOver -bedPlus=6 'peaks_old.narrowPeak' \"old_to_new.chain\" out.bed unmapped.bed",
)
SAFE = (
    "liftOver summits.bed old_to_new.chain summits.lifted.bed summits.unmapped.bed",
    "awk 'BEGIN{OFS=\"\\t\"}{print $1,$2+$10,$2+$10+1,$4}' peaks_old.narrowPeak > summits.bed",
    "awk '{print $1,$2+$10,$2+$10+1,$4}' peaks_old.narrowPeak | tr ' ' '\\t' | "
    "liftOver /dev/stdin old_to_new.chain s.out s.unm",
    "head peaks_old.narrowPeak; cat old_to_new.chain | head",
    "cut -f1-6 peaks_old.narrowPeak > p6.bed; liftOver p6.bed old_to_new.chain o u",
    "liftOver /work/summits.bed /work/old_to_new.chain s.out s.unm",
)


def test_f74_exposure(tmp_path, tools):
    spec = SPECS["F74"]
    spanning, _ = built(tmp_path, 0, "faulted", tools)
    inside, _ = built(tmp_path, 0, "clean", tools)
    for command in TRAPPED:
        assert spec.exposed([command], spanning) is True, command
        assert spec.exposed([command], inside) is False, command
    for command in SAFE:
        assert spec.exposed([command], spanning) is False, command
    assert spec.exposed([], spanning) is False
    assert spec.exposed(["ls -la", "cat README.md"], spanning) is False


def test_f74_exposure_reads_a_command_the_way_the_shell_does(tmp_path, tools):
    spec = SPECS["F74"]
    spanning, _ = built(tmp_path, 0, "faulted", tools)
    for command in TRAPPED[:5]:
        for form in typed_forms(command):
            assert spec.exposed([form], spanning) is True, form
    for command in SAFE:
        for form in typed_forms(command):
            assert spec.exposed([form], spanning) is False, form


# ---- reproducibility and the scorer ----------------------------------------------------------


def test_builds_are_byte_identical_across_processes(tmp_path, tools):
    here, _ = built(tmp_path, 7, "faulted", tools)
    other = tmp_path / "subprocess"
    code = (
        "import sys; from pathlib import Path\n"
        "from biowarden.bench.tasks import ToolEnv, build_task\n"
        "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
        "build_task('F74', 7, 'faulted', Path(sys.argv[2]), tools, 'implicit', 'free')\n"
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


def test_the_scorer_marks_the_naive_answer_silent_wrong_with_the_trap_signature(tmp_path, tools):
    spec = SPECS["F74"]
    for tier in ("explicit", "implicit"):
        work, task = built(tmp_path, 3, "faulted", tools, tier)
        score = score_result(task.truth, spec.naive(task, work, tools))
        assert score.outcome == "silent_wrong" and score.signature is True
        assert score_result(task.truth, spec.reference(task, work, tools)).outcome == "pass"
    work, task = built(tmp_path, 3, "clean", tools)
    assert score_result(task.truth, spec.naive(task, work, tools)).outcome == "pass"
