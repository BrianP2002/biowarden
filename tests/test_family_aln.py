"""Family-specific tests for the alignment pack, F29-F38.

The generic gates live in test_extra_families.py. Here, per family: the exposure functions on
concrete command lines, the size of the trap over seeds 0-29 on both tiers (read off the built
truth), twins that differ in the one declared fact, the explicit and implicit READMEs, builds that
are byte-reproducible across processes, and checks that the naive route runs the installed tool
without a warning (exit 0) and that the tool behaves as the module docstring says.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task, common
from biowarden.bench.tasks.extra import (
    f29_mapq_unique as f29,
)
from biowarden.bench.tasks.extra import (
    f30_bwa_short as f30,
)
from biowarden.bench.tasks.extra import (
    f31_star_genecounts as f31,
)
from biowarden.bench.tasks.extra import (
    f32_stats_nm as f32,
)
from biowarden.bench.tasks.extra import (
    f33_bt2_maxins as f33,
)
from biowarden.bench.tasks.extra import (
    f34_hisat2_strand as f34,
)
from biowarden.bench.tasks.extra import (
    f35_mm2_short as f35,
)
from biowarden.bench.tasks.extra import (
    f36_fixmate_order as f36,
)
from biowarden.bench.tasks.extra import (
    f37_merge_rg as f37,
)
from biowarden.bench.tasks.extra import (
    f38_idxstats_unmapped as f38,
)
from biowarden.bench.tasks.extra import (
    f39_bwa_split as f39,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
WORK = Path("/nonexistent-workspace")
MODULES = {
    "F29": f29, "F30": f30, "F31": f31, "F32": f32, "F33": f33,
    "F34": f34, "F35": f35, "F36": f36, "F37": f37, "F38": f38, "F39": f39,
}  # fmt: skip
FAMILIES = sorted(MODULES, key=lambda i: int(i[1:]))


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    needed = ["STAR", "bwa", "bowtie2", "hisat2", "minimap2", "stringtie"]
    if env is None or set(needed) - set(env.real_bins(needed)):
        pytest.skip("samtools and the aligners are not available")
    return env


# ---- exposure -------------------------------------------------------------------------------

TRAPPED = {
    "F29": [
        "STAR --genomeDir . --readFilesIn reads.fastq && samtools view -c -q 1 Aligned.out.sam",
        "samtools view -c -q 1 -F 0x100 Aligned.out.bam",
        "samtools view -cq2 Aligned.out.sam",
        "samtools view -q 3 -c Aligned.out.sam",
        "samtools view --min-MQ 1 Aligned.out.sam | cut -f1 | sort -u | wc -l",
        "samtools view Aligned.out.sam | awk '$5>0' | wc -l",
        "samtools view Aligned.out.sam | awk '$5 >= 1 {print $1}' | sort -u | wc -l",
    ],
    "F30": [
        "bwa mem genome.fa reads.fastq > a.sam",
        "bwa mem -t 4 genome.fa reads.fastq | samtools view -b -o a.bam -",
        "bwa mem -T 30 genome.fa reads.fastq > a.sam",
        "bwa mem -T 25 genome.fa reads.fastq > a.sam",
        "bwa mem -M -R '@RG\\tID:a\\tSM:b' genome.fa reads.fastq > a.sam",
    ],
    "F31": [],  # needs a ReadsPerGene.out.tab in the workspace: see test_f31_exposure
    "F32": [
        "samtools stats reads.bam | grep ^SN",
        "samtools stats -r genome.fa reads.bam | grep mismatches",
        "samtools stats -@ 2 reads.bam > s.txt",
    ],
    "F33": [
        "bowtie2 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq -S o.sam",
        "bowtie2 -p 4 --very-sensitive -x bt2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "bowtie2 -X 500 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "bowtie2 --maxins 600 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "bowtie2 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq | samtools view -c -f 2",
    ],
    "F34": [
        "hisat2 --rna-strandness FR -x hs2idx -1 reads_1.fastq -2 reads_2.fastq -S o.sam",
        "hisat2 -p 2 --rna-strandness=FR -x hs2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "hisat2 --rna-strandness F -x hs2idx -U reads_1.fastq -S o.sam",
        "hisat2 --rna-strandness R -x hs2idx -1 reads_1.fastq -2 reads_2.fastq -S o.sam",
    ],
    "F35": [
        "minimap2 -a genome.fa reads.fastq > o.sam",
        "minimap2 -ax map-ont genome.fa reads.fastq > o.sam",
        "minimap2 -a -t 4 genome.fa reads.fastq | samtools sort -o o.bam -",
        "minimap2 -a -x splice genome.fa reads.fastq > o.sam",
    ],
    "F36": [
        "samtools fixmate -m reads.bam fx.bam",
        "samtools fixmate -m reads.bam - | samtools sort - | samtools markdup - out.bam",
        "samtools sort -o c.bam reads.bam && samtools fixmate -m c.bam fx.bam",
        "cd /w && samtools fixmate -r -m reads.bam fx.bam; samtools sort fx.bam -o s.bam",
        "samtools sort -n -o other.bam reads.bam; samtools fixmate -m reads.bam fx.bam",
        "samtools sort -n -o n.bam reads.bam && samtools sort -o c.bam n.bam && "
        "samtools fixmate -m c.bam fx.bam",
        "samtools sort reads.bam | samtools fixmate -m - fx.bam",
        "samtools fixmate -m -O bam reads.bam fx.bam",
    ],
    "F37": [
        "samtools merge -c -f merged.bam lane1.bam lane2.bam",
        "samtools merge -cf merged.bam lane1.bam lane2.bam",
        "samtools merge -f -c merged.bam lane1.bam lane2.bam",
        "samtools merge -@ 4 -c merged.bam lane1.bam lane2.bam && samtools index merged.bam",
    ],
    "F39": [],  # needs the README of the workspace: see test_f39_exposure
    "F38": [
        "samtools idxstats reads.bam | grep '^\\*'",
        "samtools idxstats reads.bam | tail -n 1",
        "samtools idxstats reads.bam | tail -1 | cut -f4",
        "samtools idxstats reads.bam | awk '$1==\"*\" {print $4}'",
    ],
}
SAFE = {
    "F39": [],
    "F29": [
        "samtools view -c -q 4 -F 0x100 Aligned.out.sam",
        "samtools view -c -q 10 Aligned.out.sam",
        "samtools view -c -q 255 Aligned.out.bam",
        "samtools view -c -d NH:1 Aligned.out.sam",
        "grep -c NH:i:1 Aligned.out.sam",
        "samtools view -q 1 -d NH:1 -c Aligned.out.sam",
        "samtools view Aligned.out.sam | awk '$5==255' | wc -l",
        "cat Log.final.out; samtools flagstat Aligned.out.bam",
        "STAR --genomeDir . --readFilesIn reads.fastq",
    ],
    "F30": [
        "bwa mem -T 15 genome.fa reads.fastq > a.sam",
        "bwa mem -T 20 -t 2 genome.fa reads.fastq > a.sam",
        "bwa mem -T 1 genome.fa reads.fastq > a.sam",
        "bwa aln -t 1 genome.fa reads.fastq > a.sai",
        "bowtie2 -x bt2idx -U reads.fastq -S o.sam",
        "awk 'NR%4==2 {print length($0)}' reads.fastq | sort -n | head",
    ],
    "F31": [],
    "F32": [
        "samtools calmd -b reads.bam genome.fa | samtools stats -",
        "samtools calmd reads.bam genome.fa > c.sam; samtools stats c.sam",
        "samtools view reads.bam | grep -o 'nM:i:[0-9]*' | cut -d: -f3 | paste -sd+ | bc",
        "samtools stats fixed.bam | grep ^SN",
        "samtools flagstat reads.bam",
        "samtools view -c reads.bam",
    ],
    "F33": [
        "bowtie2 -X 1000 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "bowtie2 -p 2 -X 800 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "bowtie2 --maxins=2000 -x bt2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "bowtie2 -x bt2idx -U reads_1.fastq -S o.sam",
        "bowtie2-build genome.fa bt2idx",
        "head -4 reads_1.fastq",
    ],
    "F34": [
        "hisat2 --rna-strandness RF -x hs2idx -1 reads_1.fastq -2 reads_2.fastq -S o.sam",
        "hisat2 --rna-strandness R -x hs2idx -U reads_1.fastq -S o.sam",
        "hisat2 -x hs2idx -1 reads_1.fastq -2 reads_2.fastq -S o.sam",
        "hisat2 --dta -p 4 -x hs2idx -1 reads_1.fastq -2 reads_2.fastq > o.sam",
        "stringtie --rf o.bam -o o.gtf",
    ],
    "F35": [
        "minimap2 -ax sr genome.fa reads.fastq > o.sam",
        "minimap2 -a -x sr genome.fa reads.fastq > o.sam",
        "minimap2 -xsr -a -t 4 genome.fa reads.fastq > o.sam",
        "minimap2 -a --preset sr genome.fa reads.fastq > o.sam",
        "samtools faidx genome.fa",
    ],
    "F36": [
        "samtools sort -n -o n.bam reads.bam && samtools fixmate -m n.bam fx.bam",
        "samtools collate -o c.bam reads.bam; samtools fixmate -m c.bam fx.bam",
        "samtools sort -n reads.bam | samtools fixmate -m - fx.bam",
        "samtools sort -@ 4 -n -o sorted.bam reads.bam && samtools fixmate -m sorted.bam fx.bam",
        "samtools sort -o sorted.bam -n reads.bam && samtools fixmate -m sorted.bam fx.bam",
        "samtools sort -@4 -n reads.bam > n.bam; samtools fixmate -m n.bam fx.bam",
        "samtools collate -O reads.bam | samtools fixmate -@ 2 -m - fx.bam",
        "samtools sort -n -o /work/n.bam reads.bam && samtools fixmate -m ./n.bam fx.bam",
        "samtools view -H reads.bam",
        "samtools flagstat reads.bam",
    ],
    "F37": [
        "samtools merge -f merged.bam lane1.bam lane2.bam",
        "samtools merge -r -f merged.bam lane1.bam lane2.bam",
        "samtools merge -@ 4 -u -f merged.bam lane1.bam lane2.bam",
        "samtools view -H lane1.bam | grep ^@RG",
        "samtools index merged.bam",
    ],
    "F38": [
        "samtools idxstats reads.bam | awk '{u += $4} END {print u}'",
        "samtools view -c -f 4 reads.bam",
        "samtools flagstat reads.bam",
        "samtools idxstats reads.bam",
        "samtools idxstats reads.bam | cut -f1,3",
    ],
}


@pytest.mark.parametrize("family", [f for f in FAMILIES if f not in ("F31", "F39")])
def test_exposure_is_true_for_the_trapped_route_only(family):
    spec = MODULES[family].SPEC
    for command in TRAPPED[family]:
        assert spec.exposed([command], WORK) is True, command
    for command in SAFE[family]:
        assert spec.exposed([command], WORK) is False, command
    assert spec.exposed(["ls -la; cat README.md", *SAFE[family]], WORK) is False
    assert spec.exposed(["ls", TRAPPED[family][0]], WORK) is True
    assert spec.exposed([], WORK) is False


def f31_workspace(root: Path, seed: int, library: str) -> Path:
    """The files of an F31 workspace that carry the library, as the builder writes them."""
    plan = f31.make_plan(seed)
    work = root / f"f31-{seed}-{library}"
    work.mkdir(parents=True)
    (work / "genome.fa").write_text(common.wrap_fasta(plan.genome))
    (work / "genes.gtf").write_text(f31.gtf_text(plan.genes))
    first, second = f31.read_pairs(plan, library)
    (work / "reads_1.fastq").write_text(f29.fastq_text(first))
    (work / "reads_2.fastq").write_text(f29.fastq_text(second))
    return work


def test_f31_exposure_reads_the_needed_column_off_the_twin_not_off_a_table(tmp_path):
    forward = f31_workspace(tmp_path, 3, f31.FORWARD)  # column 3 is needed
    reverse = f31_workspace(tmp_path, 3, f31.REVERSE)  # column 4
    t = "star_out/ReadsPerGene.out.tab"  # no table exists anywhere: the agent's lives in /tmp
    spec = f31.SPEC
    assert spec.exposed([f"cut -f1,3 {t}"], forward) is False
    assert spec.exposed([f"cut -f1,4 {t}"], forward) is True
    assert spec.exposed([f"cut -f1,2 {t}"], forward) is True
    assert spec.exposed([f"cut -f1,3 {t}"], reverse) is True
    assert spec.exposed([f"awk '{{print $1,$4}}' {t}"], reverse) is False
    assert spec.exposed([f"awk 'NR>4 {{print $1, $2}}' {t}"], reverse) is True
    assert spec.exposed(["tail -n +5 /tmp/out/ReadsPerGene.out.tab | cut -f 1,3"], reverse) is True
    assert spec.exposed([f"awk '{{a += $3; b += $4}} END {{print a, b}}' {t}"], reverse) is False
    assert spec.exposed([f"head {t}", f"wc -l {t}", "ls"], reverse) is False
    assert spec.exposed(["STAR --genomeDir . --quantMode GeneCounts"], reverse) is False
    assert spec.exposed([f"cut -f1,3 {t}"], tmp_path / "nowhere") is False  # no data, no verdict
    # a table the agent left in the workspace says nothing about the twin: the reads decide
    (forward / "star_out").mkdir()
    (forward / "star_out" / "ReadsPerGene.out.tab").write_text("g1\t60\t2\t58\ng2\t40\t1\t39\n")
    assert spec.exposed([f"cut -f1,3 {t}"], forward) is False
    assert spec.exposed([f"cut -f1,4 {t}"], forward) is True


def test_f31_the_library_is_read_off_the_reads_on_every_seed(tmp_path):
    for seed in (*SEEDS, 1000, 1001, 1002, 1003, 1004, 1020, 1021, 1022, 1023, 1024):
        for library, column in ((f31.FORWARD, 3), (f31.REVERSE, 4)):
            work = f31_workspace(tmp_path, seed, library)
            assert f31._library(work) == library, (seed, library)
            assert f31._needed_column(work) == column, (seed, library)


def test_f39_exposure_needs_bwa_mem_m_and_a_flag_test_for_0x800(tmp_path):
    with_m, without_m = tmp_path / "m", tmp_path / "plain"
    for work, flag in ((with_m, " -M"), (without_m, "")):
        work.mkdir()
        (work / "README.md").write_text(
            f"- Our command: `bwa mem{flag} -R 'x' genome.fa reads.fastq`\n"
        )
    spec = f39.SPEC
    for command in (
        "samtools view -c -f 0x800 aln.bam",
        "samtools view -c -f 2048 aln.bam",
        "samtools view -cf 0x800 aln.sam",
        "samtools view -c -F 0x800 aln.bam",
        "bwa mem -M genome.fa reads.fastq | samtools view -c -f 0x800",
        "samtools view -c --require-flags 0x800 aln.bam",
        "samtools view -c --incl-flags=2048 aln.bam",
        "samtools view -h aln.bam | samtools view -c -f 0x800 -",
    ):
        assert spec.exposed([command], with_m) is True, command
    # a positive -f 0x800 is trapped whatever -F (or an SA test around it) says: it finds nothing
    for command in (
        "samtools view -c -f 0x800 -F 0x100 aln.bam | grep SA",
        "samtools view -c -F 0x100 -f 0x800 aln.bam",
        "samtools view -c -f 0x800 -F 0x904 aln.bam",
        "samtools view -c -F 0x904 -f 0x800 aln.bam",
        "samtools view -c -f 0x800 -e 'exists([SA])' aln.bam",
        "samtools view -c -f 0x801 aln.bam",
    ):
        assert spec.exposed([command], with_m) is True, command
    for command in (
        "samtools view -c -F 0x904 -e 'exists([SA])' aln.bam",
        "samtools view -c -f 0x100 aln.bam",
        "samtools view -c -F 0x900 aln.bam",
        "samtools view -c -F 0x800 -f 0x100 aln.bam",
        "samtools view -c --exclude-flags 0x904 aln.bam",
        "samtools view -F 0x800 aln.bam | grep -c SA:Z",
        "samtools view -c -f 0 aln.bam",
        "samtools flagstat aln.bam",
        "bwa mem genome.fa reads.fastq > aln.sam",
    ):
        assert spec.exposed([command], with_m) is False, command
    assert spec.exposed(["samtools view -c -f 0x800 aln.bam"], without_m) is False
    assert spec.exposed(
        ["bwa mem -M g.fa r.fq > a.sam", "samtools view -c -f 0x800 a.sam"], without_m
    )
    assert spec.exposed([], with_m) is False


def test_f36_exposure_reads_the_options_of_the_sort_in_any_order():
    """`-@ N` and `-o FILE` before `-n`, bundles, long options and redirects: all name sorts."""
    for sort in (
        "samtools sort -@ 4 -n -o out.bam reads.bam",
        "samtools sort -o out.bam -n reads.bam",
        "samtools sort -@4 -n reads.bam -o out.bam",
        "samtools sort -no out.bam reads.bam",
        "samtools sort -m 2G -N -T tmp/x -o out.bam reads.bam",
        "samtools sort --threads 4 --output-fmt BAM -n -o out.bam reads.bam",
        "samtools sort -n -t RG -o out.bam reads.bam",
        "samtools sort -n reads.bam > out.bam",
        "samtools collate -u -@ 2 -o out.bam reads.bam",
        "samtools collate reads.bam out",  # writes out.bam
    ):
        assert f36.SPEC.exposed([sort, "samtools fixmate -m out.bam fx.bam"], WORK) is False, sort
    for sort in (  # not a name sort, or a value taken for the flag
        "samtools sort -@ 4 -o out.bam reads.bam",
        "samtools sort -o out.bam reads.bam",
        "samtools sort -m 2G -T n -o out.bam reads.bam",  # the `n` is the value of -T
        "samtools sort -t RG -o out.bam reads.bam",
    ):
        assert f36.SPEC.exposed([sort, "samtools fixmate -m out.bam fx.bam"], WORK) is True, sort
    fixmate = "samtools sort -n -o out.bam reads.bam && samtools fixmate"
    assert f36.SPEC.exposed([f"{fixmate} -m -@ 2 -O bam out.bam fx.bam"], WORK) is False
    assert f36.SPEC.exposed([f"{fixmate} -m -O bam reads.bam fx.bam"], WORK) is True


def test_f36_exposure_ties_the_name_sort_to_the_input_of_fixmate():
    def trapped(*commands: str) -> bool:
        return f36.SPEC.exposed(list(commands), WORK)

    by_name, fixmate = "samtools sort -n -o n.bam", "samtools fixmate -m"
    # a name sort of some other file protects nothing
    assert trapped(f"{by_name} other.bam", f"{fixmate} reads.bam f.bam")
    assert trapped(f"{by_name} reads.bam; {fixmate} m.bam f.bam")
    # a later sort without -n takes the grouping back
    assert trapped(f"{by_name} reads.bam", "samtools sort -o n.bam n.bam", f"{fixmate} n.bam f")
    # the same file under another spelling is the same file
    assert not trapped("samtools sort -n -o /work/n.bam reads.bam", f"{fixmate} ./n.bam f.bam")
    # a pipe: only a name sort straight upstream feeds the stdin of fixmate
    assert not trapped(f"samtools sort -n reads.bam | {fixmate} - f.bam")
    assert not trapped(f"samtools sort -n reads.bam | tee n.bam | {fixmate} - f.bam")
    assert not trapped(f"samtools collate -O reads.bam | {fixmate} - f.bam")
    assert trapped(f"samtools sort reads.bam | {fixmate} - f.bam")
    assert trapped(f"cat reads.bam | {fixmate} - f.bam")
    assert trapped(f"{by_name} reads.bam", f"samtools view reads.bam | {fixmate} - f")
    # inside bash -c, and with set -e and one call per line
    assert not trapped(f"bash -c '{by_name} reads.bam && {fixmate} n.bam f.bam'")
    assert not trapped(f"set -e\n{by_name} reads.bam\n{fixmate} n.bam f.bam\necho done")
    assert trapped(f"{fixmate} reads.bam /tmp/f.bam 2>&1 | head")


def test_f36_exposure_remembers_a_name_sort_from_an_earlier_command():
    assert (
        f36.SPEC.exposed(
            ["samtools sort -n -o n.bam reads.bam", "samtools fixmate -m n.bam f.bam"], WORK
        )
        is False
    )
    assert (
        f36.SPEC.exposed(
            ["samtools fixmate -m reads.bam f.bam", "samtools sort -n reads.bam -o n.bam"], WORK
        )
        is True
    )


# ---- the trap is observable on every seed, tier and route ------------------------------------


def numeric_size(expected, naive) -> int:
    if isinstance(expected, dict):
        return sum(abs(expected[k] - naive[k]) for k in expected)
    return abs(expected - naive)


MIN_TRAP = {
    "F29": 10, "F30": 5, "F31": 20, "F32": 50, "F33": 3,
    "F35": 5, "F36": 3, "F38": 2, "F39": 10,
}  # fmt: skip


@pytest.mark.parametrize("family", FAMILIES)
def test_trap_size_over_seeds_0_29_and_no_trap_on_the_clean_twin(tmp_path, tools, family):
    smallest = None
    for seed in SEEDS:
        clean = build_task(family, seed, "clean", tmp_path / f"c{seed}", tools, "explicit")
        faulted = build_task(family, seed, "faulted", tmp_path / f"f{seed}", tools, "implicit")
        assert clean.truth["naive"] == clean.truth["expected"], (family, seed)
        assert faulted.truth["naive"] != faulted.truth["expected"], (family, seed)
        assert clean.truth["expected"] == faulted.truth["expected"]
        if family in MIN_TRAP:
            size = numeric_size(faulted.truth["expected"], faulted.truth["naive"])
            smallest = size if smallest is None else min(smallest, size)
    if family in MIN_TRAP:
        assert smallest >= MIN_TRAP[family], (family, smallest)
    if family == "F34":  # strands: every gene flips
        task = build_task(family, 0, "faulted", tmp_path / "s", tools, "implicit")
        assert all(
            task.truth["expected"][g] != task.truth["naive"][g] for g in task.truth["expected"]
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_explicit_readme_states_the_convention_and_implicit_does_not(tmp_path, tools, family):
    texts = {}
    for tier in ("explicit", "implicit"):
        work = tmp_path / tier
        build_task(family, 4, "faulted", work, tools, tier)
        texts[tier] = (work / "README.md").read_text()
    assert texts["implicit"].startswith("# Project files")
    task = build_task(family, 4, "faulted", tmp_path / "t", tools, "explicit", "forced")
    free = build_task(family, 4, "faulted", tmp_path / "u", tools, "explicit", "free")
    assert task.prompt != free.prompt and task.truth == free.truth


def test_f31_f34_f36_f37_explicit_readmes_name_the_protocol_or_the_order(tmp_path, tools):
    for family, phrase in (("F31", "dUTP"), ("F34", "dUTP"), ("F36", "no particular order"),
                           ("F38", "one mapped and one unmapped mate"), ("F29", "repeat families"),
                           ("F33", "size-selected"), ("F39", "shorter split hits")):  # fmt: skip
        e, i = tmp_path / f"e{family}", tmp_path / f"i{family}"
        build_task(family, 2, "faulted", e, tools, "explicit")
        build_task(family, 2, "faulted", i, tools, "implicit")
        assert phrase in (e / "README.md").read_text(), family
        assert phrase not in (i / "README.md").read_text(), family
    e, i = tmp_path / "e37", tmp_path / "i37"
    build_task("F37", 2, "faulted", e, tools, "explicit")
    build_task("F37", 2, "faulted", i, tools, "implicit")
    assert "combine @RG headers with colliding IDs" in (e / "README.md").read_text()
    assert "colliding" not in (i / "README.md").read_text()
    assert "`samtools merge -c" in (i / "README.md").read_text()  # the documented command stays


def test_f39_twins_differ_only_in_the_readme_command(tmp_path, tools):
    c, f = tmp_path / "c", tmp_path / "f"
    build_task("F39", 3, "clean", c, tools, "implicit")
    build_task("F39", 3, "faulted", f, tools, "implicit")
    changed = {p.name for p in c.iterdir() if (f / p.name).read_bytes() != p.read_bytes()}
    assert changed == {"README.md"}
    assert "bwa mem -M" in (f / "README.md").read_text()
    assert "bwa mem -M" not in (c / "README.md").read_text()


def test_f39_forced_sentence_names_the_tool_not_the_flag_and_both_instructions_can_be_obeyed(
    tmp_path, tools
):
    """The faulted README says bwa mem -M, the sentence only `samtools view -c` with a flag filter:
    counting the secondary parts (flag 0x100) obeys both and gives the truth."""
    for seed in (1, 2, 3, 4):
        work = tmp_path / f"s{seed}"
        task = build_task("F39", seed, "faulted", work, tools, "implicit", "forced")
        assert "`samtools view -c` with a flag filter" in task.prompt
        assert "0x800" not in task.prompt and "2048" not in task.prompt
        assert "bwa mem -M" in (work / "README.md").read_text()
        f39.align(work, tools, "n.sam")
        counted = tools.run(["samtools", "view", "-c", "-f", "0x100", "n.sam"], work)
        assert int(counted.stdout) == task.truth["expected"] != task.truth["naive"]


def test_f33_prompt_states_no_fragment_length_and_only_the_explicit_readme_has_the_range(
    tmp_path, tools
):
    for route in ("free", "forced"):
        task = build_task("F33", 2, "faulted", tmp_path / route, tools, "explicit", route)
        assert "both mates map to the same contig facing each other" in task.prompt
        assert not any(n in task.prompt for n in ("1000", "800", "600", "500", "450", "250"))
        assert "-X" not in task.prompt and "maxins" not in task.prompt.lower()
    explicit, implicit = tmp_path / "e", tmp_path / "i"
    build_task("F33", 2, "faulted", explicit, tools, "explicit")
    build_task("F33", 2, "faulted", implicit, tools, "implicit")
    assert "fragments of 250-800 bases" in (explicit / "README.md").read_text()
    assert not any(n in (implicit / "README.md").read_text() for n in ("250", "800", "size"))


def test_f37_twins_differ_only_in_lane2(tmp_path, tools):
    c, f = tmp_path / "c", tmp_path / "f"
    build_task("F37", 3, "clean", c, tools, "implicit")
    build_task("F37", 3, "faulted", f, tools, "implicit")
    changed = {p.name for p in c.iterdir() if (f / p.name).read_bytes() != p.read_bytes()}
    assert changed == {"lane2.bam", "lane2.bam.bai"} or changed == {"lane2.bam"}


# ---- byte reproducibility across processes ----------------------------------------------------


def digest(work: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(work.iterdir()):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


CHILD = (
    "import sys, hashlib; from pathlib import Path\n"
    "from biowarden.bench.tasks import ToolEnv, build_task\n"
    "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
    "work = Path(sys.argv[2]); build_task(sys.argv[3], 5, 'faulted', work, tools, 'explicit')\n"
    "h = hashlib.sha256()\n"
    "for p in sorted(work.iterdir()): h.update(p.name.encode()); h.update(p.read_bytes())\n"
    "print(h.hexdigest())\n"
)


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_identical_across_processes_and_carry_no_path_or_time(
    tmp_path, tools, family
):
    mine = tmp_path / "mine"
    build_task(family, 5, "faulted", mine, tools, "explicit")
    proc = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(ROOT), str(tmp_path / "child"), family],
        capture_output=True, text=True, check=False, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == digest(mine)
    for p in mine.iterdir():
        data = p.read_bytes()
        assert str(tmp_path).encode() not in data, p.name
        assert (
            b"/projects/" not in data
            and b"/gpfs/" not in data
            and str(Path.home()).encode() not in data
        ), p.name


# ---- the tools behave as the docstrings say, and the naive route is silent -------------------


def silent(proc) -> bool:
    noise = ("warn", "error", "fail", "skip", "drop", "discard", "unmapped", "short")
    return proc.returncode == 0 and not any(w in proc.stderr.lower() for w in noise)


def test_f29_star_gives_multimappers_mapq_3_and_1_and_unique_255(tmp_path, tools):
    task = build_task("F29", 2, "faulted", tmp_path, tools, "explicit")
    proc = f29.star_map(tmp_path, tools, ["reads.fastq"], "x_")
    assert silent(proc)
    records = f29.sam_records((tmp_path / "x_Aligned.out.sam").read_text())
    by_nh = {}
    for r in records:
        by_nh.setdefault(f29.tag_value(r, "NH"), set()).add(int(r[4]))
    assert by_nh["1"] == {255} and by_nh["2"] == {3} and by_nh["3"] == {1}
    kept = tools.run(
        ["samtools", "view", "-c", "-q", "1", "-F", "0x100", "x_Aligned.out.sam"], tmp_path
    )
    assert int(kept.stdout) == task.truth["naive"] > task.truth["expected"]
    safe = tools.run(
        ["samtools", "view", "-c", "-q", "4", "-F", "0x100", "x_Aligned.out.sam"], tmp_path
    )
    assert int(safe.stdout) == task.truth["expected"]


def test_f30_bwa_mem_writes_short_reads_as_unmapped_without_a_message(tmp_path, tools):
    task = build_task("F30", 6, "faulted", tmp_path, tools, "explicit")
    proc = tools.run(["bwa", "mem", "-t", "1", "genome.fa", "reads.fastq"], tmp_path)
    assert proc.returncode == 0 and "warn" not in proc.stderr.lower()
    contigs = list(task.truth["expected"])
    assert f30.mapped_per_contig(proc.stdout, contigs) == task.truth["naive"]
    unmapped = [f for f in f30.sam_records(proc.stdout) if int(f[1]) & 4]
    assert unmapped and all(len(f[9]) < 30 for f in unmapped)


def test_f31_star_genecounts_naive_is_the_leakage(tmp_path, tools):
    task = build_task("F31", 6, "faulted", tmp_path, tools, "explicit")
    counts = f31.run_counts(tmp_path, tools, "x_")
    assert f31.column(counts, 1) == task.truth["naive"] != task.truth["expected"]
    assert f31.column(counts, 2) == task.truth["expected"]
    assert max(task.truth["naive"][g] for g in ("g1", "g2", "g3")) <= 3  # the leakage per gene
    assert counts["N_ambiguous"][0] > 0  # column 2 loses the pairs of the shared exon


def test_f32_samtools_stats_prints_zero_mismatches_for_star_default_tags(tmp_path, tools):
    task = build_task("F32", 6, "faulted", tmp_path, tools, "explicit")
    proc = tools.run(["samtools", "stats", "reads.bam"], tmp_path)
    assert silent(proc) and f32.stats_mismatches(proc.stdout) == 0 < task.truth["expected"]
    tags = tools.run(["samtools", "view", "reads.bam"], tmp_path).stdout.splitlines()[0]
    assert "NH:i:" in tags and "nM:i:" in tags and "NM:i:" not in tags


def test_f33_bowtie2_default_leaves_long_fragments_discordant(tmp_path, tools):
    task = build_task("F33", 6, "faulted", tmp_path, tools, "explicit")
    assert f33.concordant_pairs(tmp_path, tools) == task.truth["naive"]
    assert f33.concordant_pairs(tmp_path, tools, "-X", "1000") == task.truth["expected"]


def test_f35_minimap2_default_loses_short_reads(tmp_path, tools):
    task = build_task("F35", 6, "faulted", tmp_path, tools, "explicit")
    plain = f35._minimap2(tmp_path, tools)
    assert plain == task.truth["naive"] and sum(plain.values()) < sum(
        task.truth["expected"].values()
    )
    assert f35._minimap2(tmp_path, tools, "-x", "sr") == task.truth["expected"]


def test_f36_fixmate_on_scattered_mates_exits_0_and_markdup_overcounts(tmp_path, tools):
    task = build_task("F36", 6, "faulted", tmp_path, tools, "explicit")
    fixed = tools.run(["samtools", "fixmate", "-m", "reads.bam", "fx.bam"], tmp_path)
    assert fixed.returncode == 0 and fixed.stderr == ""
    assert f36.duplicates(tmp_path, tools, "n", "reads.bam") == task.truth["naive"]
    names = tools.run(["samtools", "sort", "-n", "-o", "n.bam", "reads.bam"], tmp_path)
    assert names.returncode == 0
    assert f36.duplicates(tmp_path, tools, "r", "n.bam") == task.truth["expected"]


def test_f37_merge_c_exits_0_and_relabels_the_second_sample(tmp_path, tools):
    task = build_task("F37", 6, "faulted", tmp_path, tools, "explicit")
    merged = tools.run(
        ["samtools", "merge", "-c", "-f", "m.bam", "lane1.bam", "lane2.bam"], tmp_path
    )
    assert merged.returncode == 0 and merged.stderr == ""
    assert f37.reads_per_sample(tmp_path, tools, "m.bam") == task.truth["naive"]
    assert task.truth["naive"]["sampleB"] == 0


def test_f38_idxstats_star_row_misses_the_placed_mates(tmp_path, tools):
    task = build_task("F38", 6, "faulted", tmp_path, tools, "explicit")
    rows = f38.idxstats_rows(tmp_path, tools)
    star = next(r for r in rows if r[0] == "*")
    assert (
        int(star[3]) == task.truth["naive"] < sum(int(r[3]) for r in rows) == task.truth["expected"]
    )


def test_f34_hisat2_wrong_strandness_exits_0_and_flips_every_strand(tmp_path, tools):
    task = build_task("F34", 6, "faulted", tmp_path, tools, "explicit")
    f34._align(tmp_path, tools, "n", "--rna-strandness", "FR")
    assert f34.assembled_strands(tmp_path, tools, "n") == task.truth["naive"]
    f34._align(tmp_path, tools, "r", "--rna-strandness", "RF")
    assert f34.assembled_strands(tmp_path, tools, "r") == task.truth["expected"]


def test_f39_bwa_mem_m_makes_the_supplementary_count_zero_without_a_message(tmp_path, tools):
    task = build_task("F39", 6, "faulted", tmp_path, tools, "explicit")
    f39.align(tmp_path, tools, "n.sam")
    proc = tools.run(["samtools", "view", "-c", "-f", "0x800", "n.sam"], tmp_path)
    assert silent(proc) and int(proc.stdout) == task.truth["naive"] == 0
    primaries = tools.run(
        ["samtools", "view", "-c", "-F", "0x904", "-e", "exists([SA])", "n.sam"], tmp_path
    )
    assert int(primaries.stdout) == task.truth["expected"] > 0
