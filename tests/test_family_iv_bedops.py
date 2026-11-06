"""Family-specific checks for F70 (bedops_order) and F71 (bedmap_score), pack iv.

The generic gates live in test_extra_families.py. Here, on seeds 0-29 and both tiers: the trap
changes the answer on every seed, the twins hold the same biology and differ only in the declared
files, the real BEDOPS programs take the trapped route silently (exit 0, empty stderr), the
exposure functions see realistic trapped commands and not the safe ones, and builds are
byte-reproducible across processes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from family_forms import typed_forms

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f70_bedops_order as f70
from biowarden.bench.tasks.extra import f71_bedmap_score as f71
from biowarden.bench.tasks.extra import load

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
FAMILIES = ("F70", "F71")
SPECS = load()
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F70 and F71 are not registered"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    if len(env.real_bins(["bedmap", "bedops", "sort-bed"])) < 3:
        pytest.skip("BEDOPS is not installed")
    return env


def built(tmp_path, family, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


@pytest.mark.parametrize(
    ("family", "slot", "locus"),
    [("F70", "sort_order", "data"), ("F71", "score_column", "tool_default")],
)
def test_locus_slot_and_source_are_declared(family, slot, locus):
    spec = SPECS[family]
    assert (spec.locus, spec.slot) == (locus, slot) and spec.source
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")
    assert "README.md" in spec.twin_files and len(spec.twin_files) <= 3


# ---- the trap, seed by seed ----------------------------------------------------------------------


def test_f70_the_instance_is_valid_and_the_counts_are_what_an_overlap_test_says():
    for seed in SEEDS:
        regions, peaks = f70.make_instance(seed)
        assert {r[0] for r in regions} == set(f70.CHROMS) == {p[0] for p in peaks}
        counts = f70.overlap_counts(regions, peaks)
        assert len(counts) == len(regions) >= 18 and max(counts.values()) >= 3, seed
        assert sum(c == 0 for c in counts.values()) <= 4, seed
        assert sum(counts.values()) <= len(peaks)  # no peak touches two regions
        assert sorted(regions, key=f70.genome_key) != sorted(regions, key=f70.sort_bed_key), seed


def test_f70_genome_order_misses_regions_on_every_seed_and_the_clean_twin_is_untouched(
    tmp_path, tools
):
    for seed in SEEDS:
        _, faulted = built(tmp_path, "F70", seed, "faulted", tools)
        expected, naive = faulted.truth["expected"], faulted.truth["naive"]
        wrong = [r for r in expected if expected[r] != naive[r]]
        assert len(wrong) >= 5, (seed, wrong)
        assert not answer_matches(faulted.truth, naive)
        _, clean = built(tmp_path, "F70", seed, "clean", tools)
        assert clean.truth["naive"] == clean.truth["expected"] == expected, seed


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f70_the_real_tool_gives_the_recorded_answers(tmp_path, tools, seed):
    work, task = built(tmp_path, "F70", seed, "faulted", tools)
    proc = tools.run(["bedmap", "--echo", "--count", "regions.bed", "peaks.bed"], work)
    assert proc.returncode == 0 and proc.stderr == ""  # the trap is silent
    assert f70.parse_counts(proc.stdout) == task.truth["naive"]
    loud = tools.run(["bedmap", "--ec", "--echo", "--count", "regions.bed", "peaks.bed"], work)
    assert loud.returncode == 1 and "not properly sorted" in loud.stderr
    rows = {
        name: tools.run(["sort-bed", name], work).stdout for name in ("regions.bed", "peaks.bed")
    }
    for name, text in rows.items():
        (work / f"s_{name}").write_text(text)
    proc = tools.run(["bedmap", "--echo", "--count", "s_regions.bed", "s_peaks.bed"], work)
    assert answer_matches(task.truth, f70.parse_counts(proc.stdout))
    # bedtools does not care about the order: the free route has a safe tool
    proc = tools.run(["bedtools", "intersect", "-c", "-a", "regions.bed", "-b", "peaks.bed"], work)
    counts = {line.split("\t")[3]: int(line.split("\t")[4]) for line in proc.stdout.splitlines()}
    assert answer_matches(task.truth, counts)


def test_f71_the_scaled_score_is_far_from_the_fold_enrichment_on_every_seed(tmp_path, tools):
    for seed in SEEDS:
        regions, peaks = f71.make_instance(seed)
        assert all(2.0 <= p[4] <= 14.0 and 40 <= p[5] <= 380 for p in peaks), seed
        _, faulted = built(tmp_path, "F71", seed, "faulted", tools)
        gap = min(
            abs(faulted.truth["expected"][r] - faulted.truth["naive"][r])
            for r in regions_names(regions)
        )
        assert gap > 20, (seed, gap)
        _, clean = built(tmp_path, "F71", seed, "clean", tools)
        assert clean.truth["naive"] == clean.truth["expected"] == faulted.truth["expected"], seed


def regions_names(regions):
    return [r[3] for r in regions]


def test_f71_every_peak_lies_inside_one_region_so_only_the_column_matters():
    for seed in SEEDS:
        regions, peaks = f71.make_instance(seed)
        for peak in peaks:
            inside = [r for r in regions if r[0] == peak[0] and r[1] <= peak[1] and peak[2] <= r[2]]
            overlapping = [
                r for r in regions if r[0] == peak[0] and r[1] < peak[2] and peak[1] < r[2]
            ]
            assert len(inside) == len(overlapping) == 1, (seed, peak)
        per_region = {r[3]: 0 for r in regions}
        for peak in peaks:
            per_region[f71.region_of(regions, peak)] += 1
        assert min(per_region.values()) >= 2 and max(per_region.values()) <= 5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f71_the_real_tool_averages_column_5_silently(tmp_path, tools, seed):
    work, task = built(tmp_path, "F71", seed, "faulted", tools)
    proc = tools.run(["bedmap", "--echo", "--mean", "regions.bed", "peaks.narrowPeak"], work)
    assert proc.returncode == 0 and proc.stderr == ""
    got = f71.parse_means(proc.stdout)
    assert answer_matches({**task.truth, "expected": task.truth["naive"]}, got)
    assert not answer_matches(task.truth, got)
    # a bedGraph of the same signal: NAN and 0, silently
    rows = [line.split("\t") for line in (work / "peaks.narrowPeak").read_text().splitlines()]
    (work / "sig.bedgraph").write_text("".join(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[6]}\n" for r in rows))
    proc = tools.run(["bedmap", "--echo", "--mean", "regions.bed", "sig.bedgraph"], work)
    assert proc.returncode == 0 and proc.stderr == ""
    values = [line.rsplit("|", 1)[1] for line in proc.stdout.splitlines()]
    assert set(values) <= {"NAN", "0.000000"} and "NAN" in values
    reference = SPECS["F71"].reference(task, work, tools)
    assert answer_matches(task.truth, reference["answer"])


# ---- the twins -------------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_twins_share_the_biology_and_differ_only_in_declared_files(tmp_path, tools, family):
    spec = SPECS[family]
    for seed in (0, 5, 11):
        w_clean, clean = built(tmp_path, family, seed, "clean", tools)
        w_fault, faulted = built(tmp_path, family, seed, "faulted", tools)
        assert clean.truth["expected"] == faulted.truth["expected"], seed
        names_c = {p.name for p in w_clean.iterdir()}
        names_f = {p.name for p in w_fault.iterdir()}
        changed = {
            n for n in names_c & names_f if (w_clean / n).read_bytes() != (w_fault / n).read_bytes()
        }
        assert (changed | (names_c ^ names_f)) <= spec.twin_files, (family, seed, changed)
        assert clean.evidence != faulted.evidence and faulted.evidence


def test_f70_the_twins_hold_the_same_intervals_in_another_order(tmp_path, tools):
    w_clean, _ = built(tmp_path, "F70", 4, "clean", tools)
    w_fault, _ = built(tmp_path, "F70", 4, "faulted", tools)
    for name in ("regions.bed", "peaks.bed"):
        clean_lines = (w_clean / name).read_text().splitlines()
        fault_lines = (w_fault / name).read_text().splitlines()
        assert clean_lines != fault_lines and sorted(clean_lines) == sorted(fault_lines)
        assert clean_lines == sorted(
            clean_lines,
            key=lambda x: (x.split("\t")[0], int(x.split("\t")[1]), int(x.split("\t")[2])),
        )
    assert f70.workspace_is_sort_bed_order(w_clean) and not f70.workspace_is_sort_bed_order(w_fault)


def test_f71_the_twins_hold_the_same_peaks_in_another_layout(tmp_path, tools):
    w_clean, _ = built(tmp_path, "F71", 4, "clean", tools)
    w_fault, _ = built(tmp_path, "F71", 4, "faulted", tools)
    five = [line.split("\t") for line in (w_clean / "peaks.bed").read_text().splitlines()]
    ten = [line.split("\t") for line in (w_fault / "peaks.narrowPeak").read_text().splitlines()]
    assert len(five) == len(ten) and {len(r) for r in five} == {5} and {len(r) for r in ten} == {10}
    assert [r[:4] for r in five] == [r[:4] for r in ten]
    assert [r[4] for r in five] == [r[6] for r in ten]  # the fold enrichment: column 5 or column 7
    assert [r[4] for r in ten] != [r[4] for r in five]
    assert not (w_clean / "peaks.narrowPeak").exists() and not (w_fault / "peaks.bed").exists()


@pytest.mark.parametrize(("family", "phrase"), [("F70", "Sorted"), ("F71", "Columns:")])
def test_the_explicit_tier_states_the_fact_and_the_implicit_one_does_not(
    tmp_path, tools, family, phrase
):
    w_exp, task_exp = built(tmp_path, family, 2, "faulted", tools, "explicit")
    w_imp, task_imp = built(tmp_path, family, 2, "faulted", tools, "implicit")
    assert phrase in (w_exp / "README.md").read_text()
    assert phrase not in (w_imp / "README.md").read_text()
    assert task_exp.prompt == task_imp.prompt
    assert len(task_exp.evidence) == len(task_imp.evidence) + 1


def test_the_forced_route_dictates_bedmap(tmp_path, tools):
    for family in FAMILIES:
        _, free = built(tmp_path, family, 1, "faulted", tools, "explicit", "free")
        _, forced = built(tmp_path, family, 1, "faulted", tools, "explicit", "forced")
        assert "bedmap" in forced.prompt and "bedmap" not in free.prompt, family


# ---- exposure --------------------------------------------------------------------------------


F70_RAW = "bedmap --echo --count regions.bed peaks.bed"
F70_TRAPPED = (
    F70_RAW,
    "bedmap --count --echo ./regions.bed ./peaks.bed > counts.txt",
    "bedops -e 1 regions.bed peaks.bed",
    "sort-bed regions.bed | bedmap --echo --count - peaks.bed",
    "cat regions.bed | bedmap --count - peaks.bed",
    "closest-features regions.bed peaks.bed | head",
    "bedmap --count regions.bed peaks.bed; sort-bed regions.bed > r.bed",
    "bedmap --ec --count r.sorted.bed p.sorted.bed || bedmap --count regions.bed peaks.bed",
    # the workspace is /work in the sandbox, a line may be continued, a name may be quoted
    "bedmap --echo --count /work/regions.bed /work/peaks.bed",
    "cd /work && bedmap --echo --count ./regions.bed ./peaks.bed",
    "bedmap --echo \\\n  --count \\\n  regions.bed peaks.bed",
    "bedmap --echo --count 'regions.bed' \"peaks.bed\"",
)
F70_SAFE = (
    "bedmap --ec --echo --count regions.bed peaks.bed",
    "bedmap --ec --echo --count /work/regions.bed /work/peaks.bed",
    "sort-bed regions.bed > r.bed && sort-bed peaks.bed > p.bed && "
    "bedmap --echo --count r.bed p.bed",
    "bedmap --echo --count <(sort-bed regions.bed) <(sort-bed peaks.bed)",
    "bedmap --echo --count regions.sorted.bed peaks.sorted.bed",
    "bedtools intersect -c -a regions.bed -b peaks.bed",
    "head -3 regions.bed peaks.bed; cat README.md",
    "sort -k1,1V regions.bed | head",
)


def test_f70_exposure(tmp_path, tools):
    spec = SPECS["F70"]
    karyotypic, _ = built(tmp_path, "F70", 0, "faulted", tools)
    lexicographic, _ = built(tmp_path, "F70", 0, "clean", tools)
    for command in F70_TRAPPED:
        assert spec.exposed([command], karyotypic) is True, command
        assert spec.exposed([command], lexicographic) is False, command
    for command in F70_SAFE:
        assert spec.exposed([command], karyotypic) is False, command
    assert spec.exposed([], karyotypic) is False
    assert spec.exposed([F70_RAW], tmp_path / "absent") is False  # an unreadable workspace


def test_f70_exposure_reads_a_command_the_way_the_shell_does(tmp_path, tools):
    spec = SPECS["F70"]
    karyotypic, _ = built(tmp_path, "F70", 0, "faulted", tools)
    for command in F70_TRAPPED[:8]:
        for form in typed_forms(command):
            assert spec.exposed([form], karyotypic) is True, form
    for command in F70_SAFE:
        for form in typed_forms(command):
            assert spec.exposed([form], karyotypic) is False, form


F71_TRAPPED = (
    "bedmap --echo --mean regions.bed peaks.narrowPeak",
    "bedmap --mean --echo regions.bed ./peaks.narrowPeak > means.txt",
    "bedmap --delim '\\t' --echo --median regions.bed peaks.narrowPeak",
    "bedmap --echo --wmean --prec 3 regions.bed peaks.narrowPeak | cut -f4,5",
    "bedmap --echo --echo-map-score regions.bed peaks.narrowPeak",
    "cat regions.bed > r.bed; bedmap --echo --sum r.bed peaks.narrowPeak",
    "bedmap --echo --mean regions.bed /work/peaks.narrowPeak",
    "bedmap --echo \\\n  --mean \\\n  regions.bed 'peaks.narrowPeak'",
)
F71_SAFE = (
    "bedmap --echo --count regions.bed peaks.narrowPeak",
    "bedmap --echo --echo-map-id regions.bed peaks.narrowPeak",
    "awk 'BEGIN{OFS=\"\\t\"}{$5=$7; print}' peaks.narrowPeak > p5.bed; "
    "bedmap --echo --mean regions.bed p5.bed",
    "bedmap --echo --mean regions.bed <(cut -f1-4,7 peaks.narrowPeak)",
    "cut -f1-4,7 peaks.narrowPeak | bedmap --echo --mean regions.bed -",
    "awk '{print $1,$2,$3,$4,$7}' peaks.narrowPeak | tr ' ' '\\t' | "
    "bedmap --echo --mean regions.bed -",
    "bedtools map -a regions.bed -b peaks.narrowPeak -c 7 -o mean",
    "head peaks.narrowPeak",
)


def test_f71_exposure(tmp_path, tools):
    spec = SPECS["F71"]
    narrow, _ = built(tmp_path, "F71", 0, "faulted", tools)
    plain, _ = built(tmp_path, "F71", 0, "clean", tools)
    for command in F71_TRAPPED:
        assert spec.exposed([command], narrow) is True, command
    for command in F71_SAFE:
        assert spec.exposed([command], narrow) is False, command
    assert spec.exposed(["bedmap --echo --mean regions.bed peaks.bed"], plain) is False
    assert spec.exposed(["bedmap --echo --mean regions.bed peaks.narrowPeak"], plain) is False
    assert spec.exposed([], narrow) is False


def test_f71_exposure_reads_a_command_the_way_the_shell_does(tmp_path, tools):
    spec = SPECS["F71"]
    narrow, _ = built(tmp_path, "F71", 0, "faulted", tools)
    for command in F71_TRAPPED[:6]:
        for form in typed_forms(command):
            assert spec.exposed([form], narrow) is True, form
    for command in F71_SAFE:
        for form in typed_forms(command):
            assert spec.exposed([form], narrow) is False, form


# ---- reproducibility and the scorer ----------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_identical_across_processes(tmp_path, tools, family):
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


@pytest.mark.parametrize("family", FAMILIES)
def test_the_scorer_marks_the_naive_answer_silent_wrong_with_the_trap_signature(
    tmp_path, tools, family
):
    spec = SPECS[family]
    for tier in ("explicit", "implicit"):
        work, task = built(tmp_path, family, 3, "faulted", tools, tier)
        score = score_result(task.truth, spec.naive(task, work, tools))
        assert score.outcome == "silent_wrong" and score.signature is True
        assert score_result(task.truth, spec.reference(task, work, tools)).outcome == "pass"
    work, task = built(tmp_path, family, 3, "clean", tools)
    assert score_result(task.truth, spec.naive(task, work, tools)).outcome == "pass"
