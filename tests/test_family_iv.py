"""Family-specific tests for the bedtools part of pack iv: F65-F69 and F75.

The generic gates live in test_extra_families.py. Here, per family: locus, slot and twin files; the
size of the trap over seeds 0-29 (read off the built truth; clean twin without a trap, faulted with
one); twins that differ only in the declared files; explicit and implicit READMEs; exposure on
concrete command lines; builds that are byte-reproducible across processes; and checks that the
installed bedtools does what the module docstrings say with exit 0 and an empty stderr.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from family_forms import typed_forms

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f65_split_reciprocal as f65
from biowarden.bench.tasks.extra import f66_map_weighted as f66
from biowarden.bench.tasks.extra import f67_spliced_coverage as f67
from biowarden.bench.tasks.extra import f68_subtract_trim as f68
from biowarden.bench.tasks.extra import f69_window_gap as f69
from biowarden.bench.tasks.extra import f75_multicov_dups as f75
from biowarden.bench.tasks.extra import load

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
WORK = Path("/nonexistent-workspace")
MODULES = {"F65": f65, "F66": f66, "F67": f67, "F68": f68, "F69": f69, "F75": f75}
FAMILIES = tuple(MODULES)
SPECS = load()
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F65-F69 and F75 are not registered"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def built(tmp_path, family, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


@pytest.mark.parametrize(
    ("family", "slot", "locus", "files"),
    [
        ("F65", "tool_logic", "tool_default", {"tx_b.bed"}),
        ("F66", "weighting", "tool_default", {"coverage.bedgraph"}),
        ("F67", "split", "tool_default", {"reads.bam", "reads.bam.bai"}),
        ("F68", "overlap_policy", "tool_default", {"blacklist.bed"}),
        ("F69", "distance_unit", "tool_default", {"peaks.bed"}),
        ("F75", "default_param", "tool_default", {"sample.bam", "sample.bam.bai"}),
    ],
)
def test_locus_slot_and_twin_files_are_declared(family, slot, locus, files):
    spec = SPECS[family]
    assert (spec.slot, spec.locus) == (slot, locus) and spec.source
    assert set(spec.twin_files) == files
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")


# ---- the trap over seeds 0-29 ------------------------------------------------------------------


def test_trap_sizes_over_seeds_0_29_and_a_clean_twin_without_a_trap(tmp_path, tools):
    sizes: dict[str, list[int]] = {f: [] for f in FAMILIES}
    for seed in SEEDS:
        for family in FAMILIES:
            _, clean = built(tmp_path, family, seed, "clean", tools)
            _, faulted = built(tmp_path, family, seed, "faulted", tools)
            assert clean.truth["naive"] == clean.truth["expected"], (family, seed)
            assert clean.truth["expected"] == faulted.truth["expected"], (family, seed)
            recorded = {**faulted.truth, "expected": faulted.truth["naive"]}
            assert not answer_matches(faulted.truth, recorded["expected"]), (family, seed)
            kind, exp, nv = faulted.truth["kind"], faulted.truth["expected"], faulted.truth["naive"]
            if kind == "id_list":
                sizes[family].append(len(set(exp) ^ set(nv)))
            elif kind == "int":
                sizes[family].append(abs(exp - nv))
            else:
                sizes[family].append(sum(abs(exp[k] - nv[k]) > 0.01 for k in exp))
    # people-sized traps: names lost, lines extra, bases, peaks, windows and regions that move
    floors = {"F65": 4, "F66": 5, "F67": 1000, "F68": 8, "F69": 3, "F75": 3}
    for family, found in sizes.items():
        assert min(found) >= floors[family], (family, min(found))


def test_f67_expected_is_not_trivially_zero(tmp_path, tools):
    for seed in SEEDS:
        _, task = built(tmp_path, "F67", seed, "faulted", tools)
        assert task.truth["expected"] >= 40, seed


def test_f65_the_exon_oracle_equals_one_intersect_per_record(tmp_path, tools):
    for seed in range(8):
        work, task = built(tmp_path, "F65", seed, "faulted", tools)
        assert f65.per_record(tools, work) == task.truth["expected"], seed
        # decoys never touch an exon of an a record
        plan = f65.make_plan(seed)
        for decoy in plan.decoys_faulted + plan.decoys_clean:
            for tx in plan.a:
                if tx.chrom == decoy.chrom:
                    assert f65.shared_bases(decoy.exons, tx.exons) == 0, (seed, decoy.name)


def test_f66_the_two_twins_hold_the_same_depth_and_the_weighted_mean_matches_intersect_wo(
    tmp_path, tools
):
    for seed in range(8):
        plan = f66.make_plan(seed)
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F66", seed, twin, tools)
            assert answer_matches(task.truth, f66.weighted_means(tools, work)), (seed, twin)
            rows = [x.split("\t") for x in (work / "coverage.bedgraph").read_text().splitlines()]
            per_base = {}
            for chrom, start, end, value in rows:
                for x in range(int(start), int(end)):
                    per_base[(chrom, x)] = float(value)
            for chrom, segs in plan["segments"].items():
                for a, b, v in segs:
                    assert all(per_base[(chrom, x)] == v for x in range(a, b)), (seed, twin)
        clean = (tmp_path / f"F66-{seed}-clean-implicit-free/coverage.bedgraph").read_text()
        faulted = (tmp_path / f"F66-{seed}-faulted-implicit-free/coverage.bedgraph").read_text()
        assert len(clean.splitlines()) > len(faulted.splitlines())
        assert {len(x.split("\t")) for x in clean.splitlines()} == {4}


def test_f68_the_reference_equals_intersect_v_and_the_clean_naive_counts_the_same(tmp_path, tools):
    for seed in range(10):
        work, task = built(tmp_path, "F68", seed, "clean", tools)
        v = tools.run(
            ["bedtools", "intersect", "-a", "peaks.bed", "-b", "blacklist.bed", "-v"], work
        )
        assert len(v.stdout.splitlines()) == task.truth["expected"]
        assert len(f68.subtract_lines(tools, work)) == task.truth["expected"]


def test_f69_boundary_peaks_sit_at_the_gap_and_closest_d_prints_gap_plus_one(tmp_path, tools):
    for seed in range(8):
        work, task = built(tmp_path, "F69", seed, "faulted", tools)
        out = tools.run(
            ["bedtools", "closest", "-a", "peaks.bed", "-b", "genes.bed", "-d"], work
        ).stdout.splitlines()
        distances = sorted(int(x.split("\t")[-1]) for x in out)
        assert distances.count(1001) >= 4, seed  # gap of exactly 1000 bases
        assert sum(1 for d in distances if d <= 1001) == task.truth["expected"], seed
        assert distances.count(1002) == 2, seed  # gap of 1001: not within, in both twins


# ---- one name pattern per file: a name never marks a trap row ----------------------------------


def bam_names(tools, work, name):
    out = tools.run(["samtools", "view", name], work).stdout.splitlines()
    return [line.split("\t")[0] for line in out]


def test_f65_tx_b_names_are_one_series_in_position_order_partners_and_extras_alike(tmp_path, tools):
    for seed in range(10):
        counts = {}
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F65", seed, twin, tools)
            text = (work / "tx_b.bed").read_text()
            rows = [line.split("\t") for line in text.splitlines()]
            assert [r[3] for r in rows] == [f"tx_b_{i + 1:03d}" for i in range(len(rows))], seed
            assert "decoy" not in text
            keys = [(r[0], int(r[1])) for r in rows]
            assert keys == sorted(keys), seed
            counts[twin] = len(rows)
        assert counts["clean"] == counts["faulted"], seed  # the extras are as many in both twins
        a_names = [x.split("\t")[3] for x in (work / "tx_a.bed").read_text().splitlines()]
        assert all(n.startswith("tx_a_") for n in a_names)  # the names in the answer are unchanged


def test_f65_builder_fails_when_the_one_call_route_returns_the_expected_names(
    tmp_path, tools, monkeypatch
):
    expected = f65.expected_names(f65.make_plan(3))
    monkeypatch.setattr(f65, "single_call", lambda tools, work: list(expected))
    with pytest.raises(RuntimeError, match=r"F65 seed 3: .* keeps every pair"):
        built(tmp_path, "F65", 3, "faulted", tools)
    work, task = built(tmp_path, "F65", 3, "clean", tools)  # the clean twin holds no trap
    assert task.truth["naive"] == task.truth["expected"] == expected
    monkeypatch.setattr(f65, "single_call", lambda tools, work: expected[:-1])
    with pytest.raises(RuntimeError, match=r"F65 seed 3: .* drops pairs of the clean twin"):
        built(tmp_path, "F65", 3, "clean", tools)
    assert built(tmp_path, "F65", 3, "faulted", tools)[1].truth["naive"] == expected[:-1]


def test_f65_the_one_call_route_drops_pairs_on_every_seed_here(tmp_path, tools):
    for seed in SEEDS:
        _, task = built(tmp_path, "F65", seed, "faulted", tools)  # no seed trips the guard
        assert task.truth["naive"] != task.truth["expected"], seed


def test_f67_every_read_has_a_name_of_one_pattern_junction_reads_included(tmp_path, tools):
    for seed in (0, 3, 11):
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F67", seed, twin, tools)
            names = sorted(bam_names(tools, work, "reads.bam"))
            assert names == [f"r{i:04d}" for i in range(len(names))], (seed, twin)
            assert not any(n[0] in "eij" for n in names)
    plan = f67.make_plan(0)
    assert len(f67.sam_text(plan, "faulted", 0).splitlines()) > len(
        f67.sam_text(plan, "clean", 0).splitlines()
    )  # the junction reads are the difference


def test_f75_every_read_has_a_name_of_one_pattern_duplicates_included(tmp_path, tools):
    for seed in (0, 3, 11):
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F75", seed, twin, tools)
            names = sorted(bam_names(tools, work, "sample.bam"))
            assert names == [f"r{i:04d}" for i in range(len(names))], (seed, twin)
        flagged = tools.run(["samtools", "view", "-f", "1024", "sample.bam"], work).stdout
        assert flagged  # the faulted twin does hold duplicates, and the flag is what marks them
        assert not any(line.startswith("d") for line in flagged.splitlines())


# ---- READMEs, twins, tiers ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_explicit_readme_states_the_convention_and_implicit_does_not(tmp_path, tools, family):
    _, _ = built(tmp_path, family, 1, "faulted", tools, "explicit")
    explicit = (tmp_path / f"{family}-1-faulted-explicit-free/README.md").read_text()
    implicit = (
        built(tmp_path, family, 1, "faulted", tools, "implicit")[0] / "README.md"
    ).read_text()
    assert len(explicit) > len(implicit) and explicit.startswith(implicit.rstrip("\n"))
    # the implicit README never names the option or the default that is the trap
    for word in ("-split", "-A", "-w 1000", "-D", "multicov", "weighted", "trim"):
        assert word not in implicit, (family, word)


@pytest.mark.parametrize("family", FAMILIES)
def test_twins_differ_only_in_the_declared_files_on_every_seed(tmp_path, tools, family):
    spec = SPECS[family]
    for seed in range(6):
        wc, _ = built(tmp_path, family, seed, "clean", tools)
        wf, _ = built(tmp_path, family, seed, "faulted", tools)
        names = {p.name for p in wc.iterdir()} | {p.name for p in wf.iterdir()}
        changed = {
            n
            for n in names
            if not (wc / n).exists()
            or not (wf / n).exists()
            or (wc / n).read_bytes() != (wf / n).read_bytes()
        }
        assert changed and changed <= spec.twin_files, (family, seed, changed)


@pytest.mark.parametrize("family", FAMILIES)
def test_forced_route_names_the_tool_and_reference_passes_every_cell(tmp_path, tools, family):
    spec = SPECS[family]
    for route in ("free", "forced"):
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, family, 2, twin, tools, "implicit", route)
            assert score_result(task.truth, spec.reference(task, work, tools)).outcome == "pass"
    _, forced = built(tmp_path, family, 2, "faulted", tools, "implicit", "forced")
    assert "bedtools" in forced.prompt


# ---- exposure ------------------------------------------------------------------------------


EXPOSED = {
    "F65": (
        [
            "bedtools intersect -a tx_a.bed -b tx_b.bed -split -r -f 0.9 -wa -u",
            "bedtools intersect -a tx_a.bed -b tx_b.bed -split -F 0.9 -f 0.9 -wa -wb | cut -f4",
            "cd /w && bedtools intersect -wa -wb -split -r -f 0.9 -a tx_a.bed -b tx_b.bed",
            # a loop in another statement of the command is not the call's loop
            "bedtools intersect -a tx_a.bed -b tx_b.bed -split -r -f 0.9 -wa -wb > o.txt; "
            "for x in 1 2; do echo $x; done",
            "bedtools intersect -a tx_a.bed -b tx_b.bed -split -r -f 0.9 -wa -u; python3 -c 1",
        ],
        [
            "cut -f1 list.txt | xargs -I{} bedtools intersect -a tx_a.bed -b {} -split -r -f 0.9",
            "ls; head tx_b.bed",
            "bedtools intersect -a tx_a.bed -b tx_b.bed -f 0.9 -r -wa -u",
            "bedtools intersect -a tx_a.bed -b tx_b.bed -split -wa -wb",
            "while read l; do echo $l > one.bed; bedtools intersect -a tx_a.bed -b one.bed "
            "-split -r -f 0.9 -u; done < tx_b.bed",
            "for i in 1 2; do bedtools intersect -a tx_a.bed -b $i.bed -split -r -f 0.9; done",
        ],
    ),
    "F66": (
        [
            "bedtools map -a windows.bed -b coverage.bedgraph -c 4 -o mean",
            "bedtools map -a windows.bed -b coverage.bedgraph -c 4 -o median -null 0",
            "bedtools map -o sum -c 4 -a windows.bed -b coverage.bedgraph",
        ],
        [
            "bedtools intersect -a windows.bed -b coverage.bedgraph -wo | awk '{s[$4]+=$8*$9}'",
            "bedtools map -a windows.bed -b coverage.bedgraph -c 4 -o max",
            "bedtools map -a windows.bed -b coverage.bedgraph -c 4 -o count",
            "head coverage.bedgraph",
        ],
    ),
    "F67": (
        [
            "bedtools coverage -a introns.bed -b reads.bam",
            "bedtools genomecov -ibam reads.bam -bg | bedtools intersect -a introns.bed -b -",
            "bedtools intersect -a introns.bed -b reads.bam -c",
            "bedtools coverage -a introns.bed -b reads.bam -d",
            # a BED made from the BAM without -split spans the introns of its spliced reads
            "bedtools bamtobed -i reads.bam > reads.bed; "
            "bedtools coverage -a introns.bed -b reads.bed",
            "bedtools bamtobed -i reads.bam | bedtools coverage -a introns.bed -b stdin",
        ],
        [
            "bedtools bamtobed -split -i reads.bam > reads.bed; "
            "bedtools coverage -a introns.bed -b reads.bed",
            "bedtools bamtobed -i reads.bam | head",
            "bedtools coverage -a introns.bed -b reads.bam -split",
            "bedtools genomecov -ibam reads.bam -bg -split",
            "samtools depth -a -b introns.bed reads.bam",
            "bedtools coverage -a introns.bed -b other.bed",
            "samtools view -c reads.bam",
        ],
    ),
    "F68": (
        [
            "bedtools subtract -a peaks.bed -b blacklist.bed | wc -l",
            "bedtools subtract -a peaks.bed -b blacklist.bed -f 0.5 | cut -f4 | sort -u | wc -l",
        ],
        [
            "bedtools subtract -a peaks.bed -b blacklist.bed -A | wc -l",
            "bedtools subtract -N -a peaks.bed -b blacklist.bed | wc -l",
            "bedtools intersect -a peaks.bed -b blacklist.bed -v | wc -l",
            "cat blacklist.bed",
        ],
    ),
    "F69": (
        [
            "bedtools window -a peaks.bed -b genes.bed -w 1000 -u | wc -l",
            "bedtools window -a peaks.bed -b genes.bed -l 1000 -r 1000 -c",
            "bedtools closest -a peaks.bed -b genes.bed -d | awk '$NF<=1000' | wc -l",
            "bedtools closest -a peaks.bed -b genes.bed -d | awk '$NF<1001' | wc -l",
            "bedtools closest -a peaks.bed -b genes.bed -d > d.txt; awk '$NF<=1000' d.txt | wc -l",
        ],
        [
            # the shift of the distance is in the test: a gap of 1000 is distance 1001
            "bedtools closest -a peaks.bed -b genes.bed -d | awk '$NF-1<=1000' | wc -l",
            "bedtools closest -a peaks.bed -b genes.bed -d | awk '($NF - 1) <= 1000' | wc -l",
            "bedtools window -a peaks.bed -b genes.bed -w 1001 -u | wc -l",
            "bedtools closest -a peaks.bed -b genes.bed -d | awk '$NF<=1001' | wc -l",
            "bedtools window -a peaks.bed -b genes.bed -w 500 -u | wc -l",
            "head peaks.bed",
        ],
    ),
    "F75": (
        [
            "bedtools coverage -a regions.bed -b sample.bam -counts",
            "bedtools intersect -a regions.bed -b sample.bam -c",
            "bedtools genomecov -ibam sample.bam -bg",
            # the filter is another statement: it does not cover the call on the raw BAM
            "samtools view -b -F 1024 sample.bam -o nodup.bam; "
            "bedtools coverage -a regions.bed -b sample.bam -counts",
        ],
        [
            "samtools view -bF 1024 sample.bam | bedtools coverage -a regions.bed -b - -counts",
            "bedtools multicov -bams sample.bam -bed regions.bed",
            "samtools view -b -F 1024 sample.bam | bedtools coverage -a regions.bed -b - -counts",
            "bedtools coverage -a regions.bed -b nodup.bam -counts",
            "samtools view -c -F 1024 sample.bam chr1:1-1000",
        ],
    ),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_is_true_for_the_trapped_route_only(family):
    spec = SPECS[family]
    trapped, safe = EXPOSED[family]
    for command in trapped:
        assert spec.exposed([command], WORK) is True, command
    for command in safe:
        assert spec.exposed([command], WORK) is False, command
    assert (
        spec.exposed([], WORK) is False and spec.exposed(["ls -la; cat README.md"], WORK) is False
    )
    # a corrected command after the trapped one does not undo the exposure
    assert spec.exposed([trapped[0], safe[0]], WORK) is True


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_reads_a_command_the_way_the_shell_does(family):
    """One option per line, files under /work, quoted file names and a Python argument list change
    no verdict."""
    spec = SPECS[family]
    trapped, safe = EXPOSED[family]
    for command in trapped:
        for form in typed_forms(command):
            assert spec.exposed([form], WORK) is True, form
    for command in safe:
        for form in typed_forms(command):
            assert spec.exposed([form], WORK) is False, form


# ---- byte reproducibility ----------------------------------------------------------------------

CHILD = (
    "import sys, hashlib; from pathlib import Path\n"
    "from biowarden.bench.tasks import ToolEnv, build_task\n"
    "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
    "work = Path(sys.argv[2]); build_task(sys.argv[3], 5, 'faulted', work, tools, 'explicit')\n"
    "h = hashlib.sha256()\n"
    "for p in sorted(work.iterdir()): h.update(p.name.encode()); h.update(p.read_bytes())\n"
    "print(h.hexdigest())\n"
)


def digest(work: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(work.iterdir()):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_identical_across_processes_and_carry_no_path_or_time(
    tmp_path, tools, family
):
    mine = tmp_path / "mine"
    build_task(family, 5, "faulted", mine, tools, "explicit")
    proc = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(ROOT), str(tmp_path / "child"), family],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == digest(mine)
    for p in mine.iterdir():
        data = p.read_bytes()
        assert str(tmp_path).encode() not in data, p.name
        assert b"/projects/" not in data and b"/gpfs/" not in data, p.name


# ---- the installed bedtools behaves as the docstrings say, silently -----------------------------


def silent(proc) -> bool:
    return proc.returncode == 0 and proc.stderr.strip() == ""


def test_the_naive_routes_exit_0_with_an_empty_stderr(tmp_path, tools):
    calls = {
        "F65": [
            "intersect",
            "-a",
            "tx_a.bed",
            "-b",
            "tx_b.bed",
            "-split",
            "-r",
            "-f",
            "0.9",
            "-wa",
            "-u",
        ],
        "F66": ["map", "-a", "windows.bed", "-b", "coverage.bedgraph", "-c", "4", "-o", "mean"],
        "F67": ["coverage", "-a", "introns.bed", "-b", "reads.bam"],
        "F68": ["subtract", "-a", "peaks.bed", "-b", "blacklist.bed"],
        "F69": ["window", "-a", "peaks.bed", "-b", "genes.bed", "-w", "1000", "-u"],
        "F75": ["coverage", "-a", "regions.bed", "-b", "sample.bam", "-counts"],
    }
    for family, argv in calls.items():
        work, _ = built(tmp_path, family, 3, "faulted", tools)
        proc = tools.run(["bedtools", *argv], work)
        assert silent(proc), (family, proc.stderr)
        assert proc.stdout.strip(), family


def test_f67_samtools_depth_does_not_count_the_skipped_bases_but_bedtools_does(tmp_path, tools):
    work, task = built(tmp_path, "F67", 4, "faulted", tools)
    proc = tools.run(["samtools", "depth", "-a", "-b", "introns.bed", "reads.bam"], work)
    assert proc.returncode == 0
    covered = sum(1 for line in proc.stdout.splitlines() if int(line.split("\t")[2]) > 0)
    assert covered == task.truth["expected"] != task.truth["naive"]


def test_f75_multicov_and_coverage_disagree_on_the_same_bam(tmp_path, tools):
    work, task = built(tmp_path, "F75", 4, "faulted", tools)
    assert answer_matches(task.truth, f75.multicov_counts(tools, work))
    assert not answer_matches(task.truth, f75.coverage_counts(tools, work))
    proc = tools.run(
        ["bedtools", "multicov", "-D", "-bams", "sample.bam", "-bed", "regions.bed"], work
    )
    assert {x.split("\t")[3]: int(x.split("\t")[4]) for x in proc.stdout.splitlines()} == (
        f75.coverage_counts(tools, work)
    )


def test_f66_map_median_and_sum_are_also_per_line(tmp_path, tools):
    work, task = built(tmp_path, "F66", 4, "faulted", tools)
    assert not answer_matches(task.truth, f66.map_means(tools, work, "median"))
    sums = f66.map_means(tools, work, "sum")
    assert not answer_matches(task.truth, sums)
    # the unweighted mean equals the sum divided by the number of lines
    counts = f66.map_means(tools, work, "count")
    means = f66.map_means(tools, work, "mean")
    assert all(abs(sums[k] / counts[k] - means[k]) < 1e-3 for k in means)


def test_f68_subtract_a_and_intersect_v_agree_and_plain_subtract_splits_a_peak(tmp_path, tools):
    for seed in range(10):
        work, task = built(tmp_path, "F68", seed, "faulted", tools)
        lines = f68.subtract_lines(tools, work)
        names = [x.split("\t")[3] for x in lines]
        assert len(names) > len(set(names)) or len(set(names)) > task.truth["expected"]
