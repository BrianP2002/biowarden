"""quant_common: the stdlib BAM/SAM/GTF readers and the strand profile behind the quant rules.

The readers are checked against `samtools view` on BAMs written by the benchmark simulator
(F41 / F43 / F42 workspaces), plus hand-written SAM text, a gzip GTF, option parsing, the verdict
function on synthetic profiles and tier honesty (a T0 context refuses to read). Also holds the
small helpers the other test_rules_quant_* files import.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, TierViolation
from biowarden.engine.rules import quant_common as qc
from biowarden.engine.spec import load_packaged

ROOT = Path(__file__).resolve().parents[1]
ALL_SPECS = (
    "quant_htseq",
    "quant_featurecounts",
    "quant_salmon",
    "quant_kallisto",
    "quant_rsem",
    "quant_stringtie",
)


def quant_tools() -> ToolEnv:
    """The tool environment, or a skip when samtools or the quantifiers are missing."""
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    names = ["htseq-count", "featureCounts", "salmon", "kallisto", "rsem-calculate-expression"]
    if len(env.real_bins([*names, "stringtie"])) < len(names) + 1:
        pytest.skip("the quantification tools are not installed")
    return env


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


def engine_for(*names: str) -> Engine:
    return Engine([load_packaged(n) for n in names], mode="block", retry_cap=10**9)


def outcome(engine: Engine, rule_id: str, tool: str, argv, cwd: Path, declared=None):
    """(decision, the result of `rule_id` or None when the rule did not apply)."""
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def workspace(tools, tmp_path: Path, family: str, seed: int, twin: str, tier: str = "implicit"):
    work = tmp_path / f"{family}-{seed}-{twin}"
    task = build_task(family, seed, twin, work, tools, tier)
    return work, task


# ---- the readers ---------------------------------------------------------------------------------


def samtools_rows(tools, work: Path, name: str = "reads.bam") -> list[list[str]]:
    out = tools.run(["samtools", "view", name], work).stdout
    return [line.split("\t") for line in out.splitlines()]


@pytest.mark.parametrize(("family", "seed", "twin"), [("F41", 0, "faulted"), ("F43", 1, "faulted")])
def test_bam_reader_agrees_with_samtools(tools, tmp_path, family, seed, twin):
    work, _ = workspace(tools, tmp_path, family, seed, twin)
    ctx = Ctx(work, "T1")
    mine = list(qc.iter_alignments(ctx, work / "reads.bam", limit=10_000))
    rows = samtools_rows(tools, work)
    assert len(mine) == len(rows) > 100
    for aln, row in zip(mine, rows, strict=True):
        assert (aln.name, aln.flag, aln.chrom, aln.start0 + 1, aln.mapq) == (
            row[0],
            int(row[1]),
            row[2],
            int(row[3]),
            int(row[4]),
        )
        spans = [int(n) for n, op in re.findall(r"(\d+)([MND])", row[5]) if op != "N"]
        assert sum(b - a for a, b in aln.blocks) == sum(spans)
        assert aln.nh == 1
    spliced = [a for a in mine if len(a.blocks) > 1]
    assert spliced, "the simulated reads include exon-spanning ones"


def test_bam_reader_honours_the_limit_and_skips_secondary_and_unmapped(tmp_path):
    sam = tmp_path / "x.sam"
    seq, qual = "A" * 10, "I" * 10
    rows = [
        f"r1\t0\tc\t5\t60\t10M\t*\t0\t0\t{seq}\t{qual}\tNH:i:1",
        f"r2\t256\tc\t5\t3\t10M\t*\t0\t0\t{seq}\t{qual}\tNH:i:2",
        f"r3\t4\t*\t0\t0\t*\t*\t0\t0\t{seq}\t{qual}",
        f"r4\t2048\tc\t5\t60\t10M\t*\t0\t0\t{seq}\t{qual}",
        f"r5\t16\tc\t8\t7\t4M2N6M\t*\t0\t0\t{seq}\t{qual}\tXX:Z:a\tNH:i:3",
    ]
    sam.write_text("@HD\tVN:1.6\n@SQ\tSN:c\tLN:100\n" + "\n".join(rows) + "\n")
    got = list(qc.iter_alignments(Ctx(tmp_path, "T1"), sam))
    assert [a.name for a in got] == ["r1", "r5"]
    assert got[1].blocks == ((7, 11), (13, 19)) and got[1].nh == 3 and got[1].minus
    assert len(list(qc.iter_alignments(Ctx(tmp_path, "T1"), sam, limit=1))) == 1


def test_a_t0_context_refuses_to_read_alignments_or_annotations(tmp_path):
    (tmp_path / "x.sam").write_text("@HD\tVN:1.6\n")
    (tmp_path / "g.gtf").write_text("")
    with pytest.raises(TierViolation):
        list(qc.iter_alignments(Ctx(tmp_path, "T0"), tmp_path / "x.sam"))
    with pytest.raises(TierViolation):
        qc.read_exons(Ctx(tmp_path, "T0"), tmp_path / "g.gtf")


def test_read_exons_takes_feature_and_attribute_and_reads_gzip(tmp_path):
    text = (
        "# comment\n"
        'c\ts\tgene\t1\t100\t.\t+\t.\tgene_id "G1";\n'
        'c\ts\texon\t11\t50\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
        'c\ts\texon\t61\t90\t.\t-\t.\tgene_id "G2"; transcript_id "T2";\n'
        'c\ts\tCDS\t61\t90\t.\t-\t.\tgene_id "G2"; transcript_id "T2";\n'
        'c\ts\texon\t61\t90\t.\t.\t.\tgene_id "G3";\n'
    )
    (tmp_path / "a.gtf").write_text(text)
    with gzip.open(tmp_path / "a.gtf.gz", "wt") as handle:
        handle.write(text)
    ctx = Ctx(tmp_path, "T1")
    for name in ("a.gtf", "a.gtf.gz"):
        exons = qc.read_exons(ctx, tmp_path / name)
        assert [(e.gene, e.start0, e.end0, e.strand) for e in exons] == [
            ("G1", 10, 50, "+"),
            ("G2", 60, 90, "-"),
        ]
    by_transcript = qc.read_exons(ctx, tmp_path / "a.gtf", attribute="transcript_id")
    assert [e.gene for e in by_transcript] == ["T1", "T2"]
    assert [e.gene for e in qc.read_exons(ctx, tmp_path / "a.gtf", feature="CDS")] == ["G2"]


def test_exon_index_finds_overlaps_including_a_long_early_exon():
    exons = [
        qc.Exon("c", 0, 1000, "+", "long"),
        qc.Exon("c", 100, 120, "-", "a"),
        qc.Exon("c", 300, 320, "-", "b"),
    ]
    index = qc.ExonIndex(exons)
    names = lambda blocks: sorted(e.gene for e in index.overlapping("c", blocks))  # noqa: E731
    assert names([(110, 115)]) == ["a", "long"]
    assert names([(500, 510)]) == ["long"]  # the prefix maximum keeps the long exon in reach
    assert names([(120, 300)]) == ["long"]
    assert names([(119, 301)]) == ["a", "b", "long"]
    assert names([(1000, 1010)]) == []
    assert index.overlapping("other", [(0, 5)]) == []


@pytest.mark.parametrize(
    ("same", "opposite", "expected"),
    [
        (96, 4, "forward"),
        (85, 15, "forward"),
        (84, 16, "unknown"),
        (4, 96, "reverse"),
        (15, 85, "reverse"),
        (16, 84, "unknown"),
        (50, 50, "unstranded"),
        (35, 65, "unstranded"),
        (34, 66, "unknown"),
        (10, 9, "unknown"),  # fewer than 20 informative reads
        (0, 0, "unknown"),
    ],
)
def test_library_classification_boundaries(same, opposite, expected):
    assert qc.classify(same, opposite) == expected


@pytest.mark.parametrize(
    ("family", "seed", "twin", "library", "antisense"),
    [
        ("F41", 0, "clean", "forward", 0),
        ("F41", 0, "faulted", "reverse", 0),
        ("F41", 1, "faulted", "unstranded", 0),
        ("F43", 0, "clean", "reverse", 0),
        ("F43", 0, "faulted", "reverse", None),
        ("F42", 0, "faulted", "unstranded", 0),
    ],
)
def test_strand_profile_of_the_family_workspaces(
    tools, tmp_path, family, seed, twin, library, antisense
):
    work, _ = workspace(tools, tmp_path, family, seed, twin)
    ctx = Ctx(work, "T1")
    profile = qc.strand_profile(ctx, work / "reads.bam", qc.read_exons(ctx, work / "genes.gtf"))
    assert profile.library == library
    assert profile.antisense == 0 if antisense == 0 else profile.antisense >= qc.MIN_AFFECTED


def test_paired_bams_are_judged_by_read_1_only(tools, tmp_path):
    work, _ = workspace(tools, tmp_path, "F42", 0, "faulted")
    ctx = Ctx(work, "T1")
    exons = qc.read_exons(ctx, work / "genes.gtf")
    paired = qc.strand_profile(ctx, work / "reads.bam", exons)
    both = sum(1 for a in qc.iter_alignments(ctx, work / "reads.bam", 10_000))
    assert paired.reads == both // 2  # read 2 is skipped, not counted as the opposite strand
    assert paired.library == "unstranded"


def test_antisense_pairs_and_transcriptome_profile(tools, tmp_path):
    exons = [
        qc.Exon("c", 0, 100, "+", "a"),
        qc.Exon("c", 90, 200, "-", "b"),
        qc.Exon("c", 150, 400, "-", "c"),
        qc.Exon("d", 0, 100, "+", "x"),
        qc.Exon("d", 50, 80, "+", "y"),
    ]
    assert qc.antisense_gene_pairs(exons) == [("a", "b")]
    work, _ = workspace(tools, tmp_path, "F41", 2, "faulted")  # a genome BAM: strand of read 1
    profile = qc.transcriptome_profile(Ctx(work, "T1"), work / "reads.bam")
    assert profile.same + profile.opposite == profile.reads > 100


def test_split_args_handles_attached_equals_and_positional_forms():
    values = frozenset({"-s", "--stranded", "-a"})
    opts, pos, flags = qc.split_args(
        ["-q", "-syes", "--stranded=no", "-a", "5", "x.bam", "--", "-weird"], values
    )
    assert opts == {"-s": ["yes"], "--stranded": ["no"], "-a": ["5"]}
    assert pos == ["x.bam", "-weird"] and flags == {"-q"}
    assert qc.last_value(opts, "-s", "--stranded") == "no"
    assert qc.last_value({}, "-s") is None
    assert qc.split_args(["--stranded", "reverse", "-s=yes"], values)[0] == {
        "--stranded": ["reverse"],
        "-s": ["yes"],
    }


# ---- the shared verdict --------------------------------------------------------------------------

FIX = {"forward": "-s yes", "reverse": "-s reverse", "unstranded": "-s no"}


def profile(library, same=3, opposite=97, antisense=0):
    return qc.StrandProfile(library, same, opposite, antisense, same + opposite + antisense)


def verdict(call_lib, prof, declared=None, hurts=True):
    return qc.strand_verdict(
        tool="t",
        key="library_strandedness",
        call_lib=call_lib,
        option="-s X",
        profile=prof,
        declared=declared,
        unstranded_hurts=hurts,
        fix_for=FIX,
        data="reads.bam",
    )


@pytest.mark.parametrize(
    ("call_lib", "prof", "declared", "hurts", "expected"),
    [
        ("reverse", profile("reverse"), None, True, "pass"),
        ("reverse", profile("reverse"), "reverse", True, "pass"),
        ("forward", profile("reverse"), None, True, "fail"),
        ("unstranded", profile("unstranded", 50, 50), None, True, "pass"),
        ("forward", profile("unstranded", 50, 50), None, True, "fail"),
        ("reverse", profile("unstranded", 50, 50), None, False, "fail"),
        ("unstranded", profile("reverse"), None, True, "pass"),  # nothing overlaps: no loss
        ("unstranded", profile("reverse", antisense=2), None, True, "pass"),
        ("unstranded", profile("reverse", antisense=3), None, True, "fail"),
        ("unstranded", profile("reverse", antisense=30), None, False, "pass"),
        ("forward", profile("reverse"), "reverse", True, "fail"),  # declaration vs options
        ("reverse", profile("reverse"), "forward", True, "fail"),
        ("reverse", profile("unstranded", 50, 50), "reverse", True, "fail"),  # vs the data
        ("unstranded", profile("reverse"), "unstranded", True, "fail"),
        ("forward", None, None, True, "pass"),
        ("forward", None, "reverse", True, "fail"),
        ("forward", profile("unknown", 60, 40), None, True, "unknown"),
        ("unstranded", profile("unknown", 60, 40), None, True, "pass"),
        ("forward", profile("unknown", 60, 40), "forward", True, "pass"),
    ],
)
def test_strand_verdict_table(call_lib, prof, declared, hurts, expected):
    got = verdict(call_lib, prof, declared, hurts)
    assert got.verdict == expected, got
    if got.verdict == "fail":
        assert got.fix and got.message
