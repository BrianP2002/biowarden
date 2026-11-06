"""Family-specific checks for F5 (depth_coords), F6 (dup_depth) and F8 (reads_vs_alignments)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f05_depth_coords as f05
from biowarden.bench.tasks.extra import f06_dup_depth as f06
from biowarden.bench.tasks.extra import f08_reads_vs_alignments as f08

SEEDS = range(30)
ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/nonexistent")  # the three exposure checks read only the command lines


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


# exposure -----------------------------------------------------------------------------------------

F5_TRAPPED = [
    "samtools depth -b regions.bed reads.bam",
    "samtools depth -r chr1:100-200 reads.bam | awk '$3>=3 {print $2; exit}'",
    "cd /work && samtools mpileup -r chr1:5-50 reads.bam",
    "head regions.bed; samtools depth -a -b regions.bed reads.bam > d.txt",
]
F5_SAFE = [
    "samtools view reads.bam | awk '{print $4}'",
    "bedtools genomecov -ibam reads.bam -bg",
    "python3 -c 'print(1)'",
    "cat regions.bed; cat README.md",
    "samtools index reads.bam",
]
F6_TRAPPED = [
    "samtools depth -a -r chr1:60-60 reads.bam",
    "samtools depth -b positions.bed reads.bam",
    "samtools mpileup -r chr1:60-60 reads.bam",
    "bcftools mpileup -f ref.fa -r chr1:60-60 reads.bam",
]
F6_SAFE = [
    "samtools depth -g 0x400 -a -r chr1:60-60 reads.bam",
    "samtools depth -g0x400 reads.bam",
    "samtools mpileup --ff 0 -r chr1:60-60 reads.bam",
    "samtools mpileup --ff=0 reads.bam",
    "samtools view -c reads.bam chr1:60-60",
    "samtools flagstat reads.bam",
    "cat positions.txt",
]
F8_TRAPPED = [
    "samtools idxstats reads.bam",
    "samtools idxstats reads.bam | awk '{print $1, $3}'",
    "samtools view -c reads.bam chr1",
    "samtools view -c -F 4 reads.bam chr2",
    "samtools view -F 4 -c reads.bam chr3",
    "samtools view -cF 4 reads.bam chr1",
    "samtools view -c -F 256 reads.bam chr1",
    "samtools view reads.bam chr1 | wc -l",
    "samtools view reads.bam | cut -f3 | sort | uniq -c",
    "samtools view -c -F 0x904 reads.bam chr1; samtools view -c reads.bam chr2",
    "samtools view -c -F 0x904 reads.bam chr1 && samtools idxstats reads.bam",
    # idxstats cut down to its counts, or a loop that counts records of the contigs it lists
    "samtools idxstats reads.bam | cut -f1,3",
    "samtools idxstats reads.bam | cut -f3",
    "samtools idxstats reads.bam | cut -f1-3 | head -3",
    "samtools idxstats reads.bam | awk '{print $1\"\\t\"$3}'",
    "samtools idxstats reads.bam | cut -f1 > contigs.txt; samtools idxstats reads.bam | head -2",
    "for c in $(samtools idxstats reads.bam | cut -f1); do samtools view -c reads.bam $c; done",
    "samtools idxstats reads.bam | cut -f1 | xargs -I{} samtools view -c reads.bam {}",
]
F8_SAFE = [
    "samtools view -c -F 0x904 reads.bam chr1",
    "samtools view -c -F 2308 reads.bam chr1",
    "samtools view -cF0x900 reads.bam chr1",
    "samtools view -F 4 reads.bam | cut -f1,3 | sort -u | cut -f2 | uniq -c",
    "samtools view -F 4 reads.bam | awk '!seen[$1,$3]++ {n[$3]++}'",
    "samtools view -F 0x900 reads.bam | wc -l",
    "samtools flagstat reads.bam",
    "samtools view -H reads.bam",
    "ls -la; cat README.md",
    # idxstats used only to list the contigs: its first column, no count in sight
    "samtools idxstats reads.bam | cut -f1",
    "samtools idxstats reads.bam | cut -f 1",
    "samtools idxstats reads.bam | cut -d$'\\t' -f1 | grep -v '^[*]'",
    "samtools idxstats reads.bam 2>/dev/null | cut -f1",
    "samtools idxstats reads.bam 2>&1 | cut -f1",
    "samtools idxstats reads.bam | awk '{print $1}'",
    "samtools idxstats reads.bam | awk -F'\\t' '$1 != \"*\" {print $1}'",
    "samtools idxstats reads.bam | grep -v '^[*]' | cut -f1 > contigs.txt",
    "samtools idxstats reads.bam | head -n 3 | cut -f1",
    "for c in $(samtools idxstats reads.bam | cut -f1 | grep -v '[*]'); do echo $c; done",
]


@pytest.mark.parametrize(
    ("spec", "trapped", "safe"),
    [
        (f05.SPEC, F5_TRAPPED, F5_SAFE),
        (f06.SPEC, F6_TRAPPED, F6_SAFE),
        (f08.SPEC, F8_TRAPPED, F8_SAFE),
    ],
    ids=["F5", "F6", "F8"],
)
def test_exposure_is_true_for_the_trapped_route_only(spec, trapped, safe):
    for command in trapped:
        assert spec.exposed([command], WORK) is True, command
    for command in safe:
        assert spec.exposed([command], WORK) is False, command
    assert spec.exposed(["ls -la; cat README.md", *safe], WORK) is False
    assert spec.exposed(["ls", trapped[0]], WORK) is True
    assert spec.exposed([], WORK) is False


# the trap is observable on every seed, by construction (no tools needed) ------------------------


def test_f5_trap_shifts_every_item_by_one_and_depth_first_reaches_3_inside_each_region():
    for seed in SEEDS:
        p = f05.plan(seed)
        assert f05.plan(seed) == p
        clean, raw_clean = f05.answers(p, "clean")
        faulted, raw_faulted = f05.answers(p, "faulted")
        assert clean == faulted == [c.answer for c in p.clusters]  # same expected in both twins
        assert raw_clean == clean  # clean: depth's position is the answer as printed
        assert all(r == e + 1 for r, e in zip(raw_faulted, faulted, strict=True))  # BED: one less
        for twin, raw in (("clean", raw_clean), ("faulted", raw_faulted)):
            shift = 1 if twin == "faulted" else 0
            for c, first in zip(p.clusters, raw, strict=True):
                assert c.start + shift < first < c.end  # strictly inside the 1-based region
        assert len(f05.read_starts(p, "clean")) <= 150 and sum(p.contigs.values()) <= 3000


def test_f5_depth_stays_below_3_before_the_answer_even_across_the_whole_contig():
    p = f05.plan(7)
    starts = f05.read_starts(p, "faulted")
    for c, first in zip(p.clusters, f05.answers(p, "faulted")[1], strict=True):
        before = [
            sum(chrom == c.chrom and s <= pos < s + f05.READ_LEN for chrom, s in starts)
            for pos in range(c.start + 1, first)
        ]
        assert before and max(before) < f05.THRESHOLD


def test_f6_every_listed_position_loses_a_record_to_the_duplicate_filter():
    for seed in SEEDS:
        p = f06.plan(seed)
        assert f06.plan(seed) == p
        everything, kept = f06.counts(p, "faulted")
        clean_all, clean_kept = f06.counts(p, "clean")
        assert everything == clean_all == clean_kept  # same records; the clean twin flags none
        assert all(k < e for k, e in zip(kept, everything, strict=True)), seed
        assert all(k >= 1 for k in kept)  # something is left for depth to report
        assert len(p.records) <= 150 and sum(p.contigs.values()) <= 3000
        assert len(p.positions) == 8 and len(set(p.positions)) == 8


def test_f8_every_contig_has_more_records_than_reads_in_the_faulted_twin_only():
    for seed in SEEDS:
        p = f08.plan(seed)
        assert f08.plan(seed) == p
        reads_f, rows_f = f08.counts(p, "faulted")
        reads_c, rows_c = f08.counts(p, "clean")
        assert reads_f == reads_c == rows_c  # one record per read when only primaries are kept
        assert all(rows_f[c] > reads_f[c] for c in p.contigs), seed
        assert all(r.chrom == q.chrom for r in p.extra for q in p.primary if q.name == r.name)
        assert len(f08.records(p, "faulted")) <= 150 and sum(p.contigs.values()) <= 3000
        assert len(p.contigs) <= 3
        # a read with several extra records exists, so "once however many records" is exercised
        names = [r.name for r in p.extra]
        assert len(names) > len(set(names))


def test_seeds_give_different_instances():
    assert f05.plan(1) != f05.plan(2)
    assert f06.plan(1) != f06.plan(2)
    assert f08.plan(1) != f08.plan(2)


# the built tasks carry the same facts, and the trap is silent on the installed tools --------------


@pytest.mark.parametrize("module", [f05, f06, f08], ids=["F5", "F6", "F8"])
def test_built_truth_has_a_trap_on_every_seed_and_tier_and_none_on_the_clean_twin(
    tmp_path, tools, module
):
    spec = module.SPEC
    for seed in SEEDS:
        for tier in spec.tiers:
            clean = build_task(spec.id, seed, "clean", tmp_path / f"c{seed}{tier}", tools, tier)
            faulted = build_task(spec.id, seed, "faulted", tmp_path / f"f{seed}{tier}", tools, tier)
            assert clean.truth["naive"] == clean.truth["expected"]
            e, n = faulted.truth["expected"], faulted.truth["naive"]
            if spec.id == "F8":
                assert all(e[c] != n[c] for c in e)
            else:
                assert len(e) == len(n) and all(a != b for a, b in zip(e, n, strict=True))


def silent(proc) -> bool:
    return proc.returncode == 0 and proc.stderr == ""


def test_f5_depth_exits_0_without_a_warning_and_prints_1_based_positions(tmp_path, tools):
    task = build_task("F5", 4, "faulted", tmp_path, tools)
    proc = tools.run(["samtools", "depth", "-b", "regions.bed", "reads.bam"], tmp_path)
    assert silent(proc)
    rows = f05.depth_rows(proc.stdout)
    regions, is_bed = f05.regions_of(tmp_path)
    raw = [f05.first_at_least(rows, c, lo, hi) for c, lo, hi in regions]
    assert is_bed and raw == task.truth["naive"] and raw != task.truth["expected"]


def test_f6_depth_exits_0_without_a_warning_and_drops_the_flagged_reads(tmp_path, tools):
    task = build_task("F6", 4, "faulted", tmp_path, tools)
    depths = []
    for chrom, pos in f06.positions_of(tmp_path):
        proc = tools.run(
            ["samtools", "depth", "-a", "-r", f"{chrom}:{pos}-{pos}", "reads.bam"], tmp_path
        )
        assert silent(proc)
        depths.append(int(proc.stdout.split("\t")[2]))
    assert depths == task.truth["naive"] != task.truth["expected"]
    flagstat = tools.run(["samtools", "flagstat", "reads.bam"], tmp_path).stdout
    assert " duplicates" in flagstat and not flagstat.startswith("0 + 0 duplicates")
    helptext = tools.run(["samtools", "depth"], tmp_path)  # usage goes to stderr on bare calls
    assert "DUP" in helptext.stdout + helptext.stderr
    for chrom, pos in f06.positions_of(tmp_path)[:2]:  # the documented override restores them
        argv = ["samtools", "depth", "-g", "0x400", "-a", "-r", f"{chrom}:{pos}-{pos}", "reads.bam"]
        assert int(tools.run(argv, tmp_path).stdout.split("\t")[2]) >= 1


def test_f8_idxstats_and_view_count_records_silently(tmp_path, tools):
    task = build_task("F8", 4, "faulted", tmp_path, tools)
    idx = tools.run(["samtools", "idxstats", "reads.bam"], tmp_path)
    assert silent(idx)
    per_contig = {line.split("\t")[0]: int(line.split("\t")[2]) for line in idx.stdout.splitlines()}
    per_contig.pop("*")
    assert per_contig == task.truth["naive"] != task.truth["expected"]
    for contig, rows in task.truth["naive"].items():
        view = tools.run(["samtools", "view", "-c", "-F", "4", "reads.bam", contig], tmp_path)
        assert silent(view) and int(view.stdout) == rows
        primary = tools.run(
            ["samtools", "view", "-c", "-F", "0x904", "reads.bam", contig], tmp_path
        )
        assert int(primary.stdout) == task.truth["expected"][contig]


def test_f8_idxstats_cut_to_its_first_column_shows_no_count(tmp_path, tools):
    """The commands the exposure check lets pass print contig names and nothing else."""
    task = build_task("F8", 4, "faulted", tmp_path, tools)
    names = {"*", *task.truth["expected"]}
    for command in F8_SAFE:
        if "idxstats" not in command or "contigs.txt" in command or "for c in" in command:
            continue
        if "2>&1" in command:  # the dynamic loader's notice would be mixed into the output
            continue
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=tools.process_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (command, proc.stderr)
        assert set(proc.stdout.split()) <= names, command
    # while one that keeps the counts column prints the record counts the trap lives in
    proc = subprocess.run(
        ["bash", "-c", "samtools idxstats reads.bam | cut -f1,3"],
        cwd=tmp_path,
        env=tools.process_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    shown = {line.split("\t")[0]: int(line.split("\t")[1]) for line in proc.stdout.splitlines()}
    assert {c: n for c, n in shown.items() if c != "*"} == task.truth["naive"]
