"""samtools rules of the alignment pack: unit tests, evidence against the tools, admission.

Rules: samtools.mapq_unique_cutoff (F29), samtools.stats_nm_missing (F32), samtools.fixmate_grouping
(F36), samtools.merge_combine_rg (F37), samtools.idxstats_unmapped (F38),
samtools.split_flag_after_bwa_m (F39), and the stdlib BAM/SAM
reader they share (engine/rules/aln_bam.py). The workspaces are the family builders' own; the
evidence tests run samtools 1.24 and STAR/bwa/bowtie2/hisat2/minimap2 where the claims of the module
docstrings are tied to the tools. The admission tests measure each rule over ADMISSION_SEEDS seeds
with calls labelled by running the real tools (bench/admission/cases_aln_samtools.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f29_mapq_unique import star_map
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import aln_samtools as rules
from biowarden.engine.rules.aln_bam import Unreadable, read_alignment
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("aln_samtools")
RULES = {r.id: r for r in SPEC.rules}
MAPQ, STATS, FIXMATE, MERGE, IDXSTATS, SPLIT = (
    "samtools.mapq_unique_cutoff",
    "samtools.stats_nm_missing",
    "samtools.fixmate_grouping",
    "samtools.merge_combine_rg",
    "samtools.idxstats_unmapped",
    "samtools.split_flag_after_bwa_m",
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not env.real_bins(
        ["STAR", "bwa", "bowtie2", "hisat2", "minimap2"]
    ).keys() >= {"STAR"}:
        pytest.skip("samtools and the aligners are not available")
    return env


def only(rule_id: str) -> Spec:
    data = SPEC.model_dump(mode="json")
    data["rules"] = [r for r in data["rules"] if r["id"] == rule_id]
    return Spec.model_validate(data)


def outcome(rule_id: str, argv, cwd, declared=None, tools=None):
    bins = tools.real_bins() if tools else {}
    env = tools.process_env() if tools else None
    eng = Engine([only(rule_id)], mode="block", real_bins=bins, real_env=env, retry_cap=10**9)
    decision = eng.evaluate(Call("samtools", tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def verdict(*args, **kwargs):
    _, result = outcome(*args, **kwargs)
    return None if result is None else result.verdict


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_every_checker_is_registered():
    assert SPEC.name == "aln_samtools" and set(RULES) == {
        MAPQ,
        STATS,
        FIXMATE,
        MERGE,
        IDXSTATS,
        SPLIT,
    }
    registered = known_checkers()
    for rule in SPEC.rules:
        assert rule.check in registered, rule.id
        assert rule.kind == "call" and rule.tier == "T1" and rule.applies.tools == ["samtools"]
        assert rule.applies.subcommands, rule.id
    keys = {r.id: r.declare.key if r.declare else None for r in SPEC.rules}
    assert keys == {
        MAPQ: "mapq_filter_goal",
        STATS: "stats_mismatches",
        FIXMATE: None,
        MERGE: None,
        IDXSTATS: "idxstats_unmapped",
        SPLIT: None,
    }
    assert all(r.declare is None or not r.declare.required for r in SPEC.rules)
    schema = SPEC.declaration_schema()
    assert schema["required"] == []


def test_every_rule_reads_files_so_a_t0_context_refuses(tmp_path):
    (tmp_path / "a.bam").write_bytes(b"")
    cases = {
        MAPQ: ("samtools_mapq_unique_cutoff", ["view", "-q", "1", "a.bam"]),
        STATS: ("samtools_stats_nm_missing", ["stats", "a.bam"]),
        FIXMATE: ("samtools_fixmate_grouping", ["fixmate", "-m", "a.bam", "o.bam"]),
        MERGE: ("samtools_merge_combine_rg", ["merge", "-c", "o.bam", "a.bam", "b.bam"]),
        IDXSTATS: ("samtools_idxstats_unmapped", ["idxstats", "a.bam"]),
        SPLIT: ("samtools_split_flag_bwa_m", ["view", "-c", "-f", "0x800", "a.bam"]),
    }
    for rule_id, (checker_name, argv) in cases.items():
        call = Call("samtools", tuple(argv), tmp_path)
        with pytest.raises(TierViolation):
            known_checkers()[checker_name](call, Match(RULES[rule_id]), Ctx(tmp_path, "T0"))


# ---- option parsing ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "q"),
    [
        (["-q", "1", "a.bam"], ["1"]),
        (["-q1", "a.bam"], ["1"]),
        (["-cq3", "a.bam"], ["3"]),
        (["-c", "-q", "10", "-q", "4", "a.bam"], ["10", "4"]),
        (["--min-MQ", "5", "a.bam"], ["5"]),
        (["--min-MQ=7", "a.bam"], ["7"]),
        (["-bh", "-q", "20", "a.bam"], ["20"]),
        (["-F", "0x100", "-c", "a.bam"], []),
        (["-o", "q.bam", "-c", "a.bam"], []),
    ],
)
def test_parse_options_finds_the_mapq_cutoff_in_every_spelling(argv, q):
    opts, positionals = rules.parse_options(argv, rules.VIEW_ARG_SHORT, rules.VIEW_LONG)
    assert opts.get("q", []) == q
    assert positionals == ["a.bam"] or "q.bam" not in positionals


def test_parse_options_keeps_positionals_after_double_dash_and_attached_values():
    opts, pos = rules.parse_options(
        ["-@4", "-o", "o.bam", "--", "-q", "x.bam"], rules.VIEW_ARG_SHORT, rules.VIEW_LONG
    )
    assert opts["@"] == ["4"] and opts["o"] == ["o.bam"] and pos == ["-q", "x.bam"]


TABLES = {
    "view": (rules.VIEW_ARG_SHORT, rules.VIEW_LONG),
    "stats": (rules.STATS_ARG_SHORT, rules.STATS_LONG),
    "fixmate": (rules.FIXMATE_ARG_SHORT, rules.FIXMATE_LONG),
    "merge": (rules.MERGE_ARG_SHORT, rules.MERGE_LONG),
    "idxstats": (rules.IDXSTATS_ARG_SHORT, rules.IDXSTATS_LONG),
}


@pytest.mark.parametrize("subcommand", sorted(TABLES))
@pytest.mark.parametrize("option", sorted(rules.GLOBAL_LONG))
def test_the_global_options_with_a_value_are_skipped_with_it_in_every_table(subcommand, option):
    arg_short, long_map = TABLES[subcommand]
    for argv in ([option, "genome.fa", "reads.bam"], [f"{option}=genome.fa", "reads.bam"]):
        opts, positionals = rules.parse_options(argv, arg_short, long_map)
        assert positionals == ["reads.bam"], (subcommand, argv)
        # a letter of its own: never read as -r (a read group in view) or -R
        assert not {"r", "R"} & set(opts), (subcommand, argv)
        assert len(opts) == 1 and next(iter(opts.values())) == ["genome.fa"]


def test_the_global_keys_are_private_and_the_threads_option_keeps_its_letter():
    assert len(rules.GLOBAL_LONG) == 7 and rules.GLOBAL_LONG["--threads"] == "@"
    private = [key for name, key in rules.GLOBAL_LONG.items() if name != "--threads"]
    assert len(set(private)) == 6 and not any(key.isascii() for key in private)
    for arg_short, long_map in TABLES.values():
        assert all(
            long_map[name] == key
            for name, key in rules.GLOBAL_LONG.items()
            if name in long_map and name != "--output-fmt"
        )
        opts, positionals = rules.parse_options(["--threads", "4", "a.bam"], arg_short, long_map)
        assert opts == {"@": ["4"]} and positionals == ["a.bam"]
        opts, positionals = rules.parse_options(["-@", "4", "a.bam"], arg_short, long_map)
        assert opts == {"@": ["4"]} and positionals == ["a.bam"]


@pytest.mark.parametrize(
    ("text", "value"), [("0x100", 256), ("256", 256), ("0400", 400), ("1x", None), ("", None)]
)
def test_flag_value(text, value):
    assert rules.flag_value(text) == value


# ---- the stdlib reader against samtools ---------------------------------------------------------

SAM = (
    "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:c1\tLN:500\n@SQ\tSN:c2\tLN:300\n"
    "@RG\tID:g1\tSM:s1\n@PG\tID:bwa\tPN:bwa\tVN:0.7.19\n"
    "r1\t99\tc1\t10\t60\t5S20M1I4M\t=\t80\t120\tACGTACGTACGTACGTACGTACGTACGTAC\t"
    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tNM:i:2\tMD:Z:10A5T7\tAS:i:-3\tXS:i:-9\tRG:Z:g1\t"
    "XA:A:x\tZf:f:1.5\tZB:B:c,1,2,3\tHH:H:1AE3\n"
    "r1\t147\tc1\t80\t60\t30M\t=\t10\t-120\tACGTACGTACGTACGTACGTACGTACGTAC\t"
    "IIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tNH:i:1\tnM:i:0\n"
    "r2\t73\tc2\t5\t40\t10M2D8M\t=\t5\t0\tACGTACGTACGTACGTAC\tIIIIIIIIIIIIIIIIII\tNM:i:2\n"
    "r2\t133\tc2\t5\t0\t*\t=\t5\t0\tACGTACGTAC\tIIIIIIIIII\n"
    "r3\t77\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n"
    "r3\t141\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n"
    "r4\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\tUU:i:300\tUS:i:-70000\n"
)


def write_alignment_files(tmp_path: Path, tools: ToolEnv) -> tuple[Path, Path]:
    sam = tmp_path / "t.sam"
    sam.write_text(SAM)
    bam = tmp_path / "t.bam"
    proc = tools.run(["samtools", "view", "--no-PG", "-b", "-o", "t.bam", "t.sam"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    return sam, bam


def test_reader_decodes_bam_and_sam_like_samtools(tools, tmp_path):
    sam, bam = write_alignment_files(tmp_path, tools)
    ctx = Ctx(tmp_path, "T1")
    from_bam = read_alignment(ctx, bam)
    from_sam = read_alignment(ctx, sam)
    assert from_bam.kind == "bam" and from_sam.kind == "sam"
    assert from_bam.refs == from_sam.refs == ["c1", "c2"]
    assert from_bam.complete and from_sam.complete
    lines = [line.split("\t") for line in SAM.splitlines() if not line.startswith("@")]
    assert len(from_bam.records) == len(from_sam.records) == len(lines)
    for rec_b, rec_s, f in zip(from_bam.records, from_sam.records, lines, strict=True):
        assert (rec_b.name, rec_b.flag, rec_b.ref, rec_b.pos) == (f[0], int(f[1]), f[2], int(f[3]))
        assert (rec_b.mapq, rec_b.cigar, rec_b.tlen) == (int(f[4]), f[5], int(f[8]))
        mate = f[2] if f[6] == "=" else f[6]
        assert (rec_b.mate_ref, rec_b.mate_pos) == (mate, int(f[7]))
        assert (rec_s.name, rec_s.flag, rec_s.ref, rec_s.pos, rec_s.mapq, rec_s.cigar) == (
            rec_b.name,
            rec_b.flag,
            rec_b.ref,
            rec_b.pos,
            rec_b.mapq,
            rec_b.cigar,
        )
    first = from_bam.records[0].tags
    assert first["NM"] == 2 and first["MD"] == "10A5T7" and first["XS"] == -9
    assert first["RG"] == "g1" and first["XA"] == "x" and first["Zf"] == 1.5
    assert from_bam.records[-1].tags == {"UU": 300, "US": -70000}
    assert from_bam.hd()["SO"] == "coordinate"
    assert from_bam.read_groups() == [{"ID": "g1", "SM": "s1"}]
    assert from_bam.program_lines()[0]["PN"] == "bwa"


def test_reader_limit_truncation_cram_and_garbage(tools, tmp_path):
    _, bam = write_alignment_files(tmp_path, tools)
    ctx = Ctx(tmp_path, "T1")
    short = read_alignment(ctx, bam, limit=3)
    assert len(short.records) == 3 and not short.complete
    data = bam.read_bytes()
    (tmp_path / "cut.bam").write_bytes(data[: len(data) // 2])
    cut = read_alignment(ctx, tmp_path / "cut.bam")
    assert not cut.complete  # truncated: what could be decoded is kept
    (tmp_path / "x.cram").write_bytes(b"CRAM\x03\x00")
    with pytest.raises(Unreadable):
        read_alignment(ctx, tmp_path / "x.cram")
    (tmp_path / "junk.bam").write_bytes(b"not a bam file at all")
    with pytest.raises(Unreadable):
        read_alignment(ctx, tmp_path / "junk.bam")
    with pytest.raises(TierViolation):
        read_alignment(Ctx(tmp_path, "T0"), bam)


# ---- samtools.mapq_unique_cutoff ----------------------------------------------------------------


@pytest.fixture(scope="module")
def f29(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f29")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        task = build_task("F29", 3, twin, work, tools, "implicit")
        star_map(work, tools, ["reads.fastq"], "aln_", ("--outSAMtype", "BAM", "Unsorted"))
        star_map(work, tools, ["reads.fastq"], "aln_", ("--outSAMtype", "SAM"))
        out[twin] = (task, work)
    return out


BAM29 = "aln_Aligned.out.bam"


@pytest.mark.parametrize("q", ["1", "2", "3"])
def test_a_cutoff_of_1_to_3_keeps_star_multimappers_and_fails(f29, tools, q):
    _, work = f29["faulted"]
    decision, result = outcome(MAPQ, ["view", "-c", "-q", q, "-F", "0x100", BAM29], work)
    assert result.verdict == "fail" and result.code == "multimappers_pass_cutoff"
    assert decision.action == "block"
    assert "NH:1" in result.fix[0] and "-q 4" in result.fix[0]
    assert "quality_only" in result.fix[1]
    text = render_block(decision)
    assert text.startswith("FIX:")


@pytest.mark.parametrize("q", ["4", "10", "30", "255"])
def test_a_cutoff_above_every_multimapper_mapq_passes(f29, q):
    _, work = f29["faulted"]
    assert verdict(MAPQ, ["view", "-c", "-q", q, BAM29], work) == "pass"


def test_clean_twin_has_no_multimapper_so_every_cutoff_is_silent(f29):
    _, work = f29["clean"]
    for q in ("1", "2", "3", "30"):
        assert verdict(MAPQ, ["view", "-c", "-q", q, BAM29], work) == "pass"


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-cq1", BAM29],
        ["view", "--min-MQ", "1", "-c", BAM29],
        ["view", "--min-MQ=2", "-c", BAM29],
        ["view", "-q1", "-c", "aln_Aligned.out.sam"],
        ["view", "-b", "-q", "1", "-o", "out.bam", BAM29],
        ["view", "-h", "-q", "1", BAM29, "chr1"],
    ],
)
def test_every_spelling_and_the_sam_file_are_covered(f29, argv):
    _, work = f29["faulted"]
    assert verdict(MAPQ, argv, work) == "fail"


def test_quality_only_declaration_silences_but_unique_only_does_not(f29):
    _, work = f29["faulted"]
    argv = ["view", "-c", "-q", "1", BAM29]
    assert verdict(MAPQ, argv, work, {"mapq_filter_goal": "quality_only"}) == "pass"
    assert verdict(MAPQ, argv, work, {"mapq_filter_goal": "unique_only"}) == "fail"


def test_unique_only_with_a_cutoff_above_the_unique_mapq_fails_as_dropping_unique_reads(f29):
    _, work = f29["faulted"]
    decision, result = outcome(
        MAPQ, ["view", "-c", "-q", "256", BAM29], work, {"mapq_filter_goal": "unique_only"}
    )
    assert result.verdict == "fail" and result.code == "unique_reads_dropped"
    assert verdict(MAPQ, ["view", "-c", "-q", "256", BAM29], work) == "pass"  # undeclared: silent


def test_a_call_that_also_filters_on_nh_is_left_alone(f29):
    _, work = f29["faulted"]
    for argv in (
        ["view", "-c", "-q", "1", "-d", "NH:1", BAM29],
        ["view", "-c", "-q", "1", "-e", "[NH]==1", BAM29],
    ):
        assert verdict(MAPQ, argv, work) == "pass"


def test_no_cutoff_no_file_stdin_and_other_subcommands_pass(f29, tmp_path):
    _, work = f29["faulted"]
    assert verdict(MAPQ, ["view", "-c", BAM29], work) == "pass"
    assert verdict(MAPQ, ["view", "-c", "-q", "0", BAM29], work) == "pass"
    assert verdict(MAPQ, ["view", "-c", "-q", "1", "absent.bam"], work) == "pass"
    assert verdict(MAPQ, ["view", "-c", "-q", "1", "-"], work) == "pass"
    assert verdict(MAPQ, ["sort", "-q", "1", BAM29], work) is None  # not a view call
    assert verdict(MAPQ, ["view", "-q", "x", BAM29], work) == "pass"


def test_an_unreadable_file_is_unknown_not_a_crash(tmp_path):
    (tmp_path / "x.cram").write_bytes(b"CRAM\x03\x00")
    (tmp_path / "junk.bam").write_bytes(b"junk")
    assert verdict(MAPQ, ["view", "-q", "1", "x.cram"], tmp_path) == "unknown"
    assert verdict(MAPQ, ["view", "-q", "1", "junk.bam"], tmp_path) == "unknown"


def sam_with(records: list[str]) -> str:
    return "@HD\tVN:1.6\n@SQ\tSN:c\tLN:1000\n" + "".join(r + "\n" for r in records)


def rec(name, flag, mapq, *tags, pos=10):
    return f"{name}\t{flag}\tc\t{pos}\t{mapq}\t10M\t*\t0\t0\tACGTACGTAC\tIIIIIIIIII\t" + "\t".join(
        tags
    )


@pytest.mark.parametrize(
    ("records", "q", "expected"),
    [
        # bwa: multi-mapped reads get MAPQ 0, so -q 1 is a uniqueness filter
        ([rec("a", 0, 60, "NM:i:0"), rec("b", 0, 0, "XA:Z:c,+20,10M,0;")], 1, "pass"),
        # bowtie2: a tie (XS == AS) has MAPQ 1; a unique read with a weak second site stays unique
        ([rec("a", 0, 42, "AS:i:0"), rec("b", 0, 1, "AS:i:0", "XS:i:0")], 1, "fail"),
        ([rec("a", 0, 42, "AS:i:0"), rec("b", 0, 40, "AS:i:0", "XS:i:-30")], 1, "pass"),
        # hisat2: NH:i:2 with MAPQ 1, the extra alignment flagged 256
        (
            [rec("a", 0, 60, "NH:i:1"), rec("b", 0, 1, "NH:i:2"), rec("b", 256, 1, "NH:i:2")],
            1,
            "fail",
        ),
        # STAR: MAPQ 3 for 2 loci
        ([rec("a", 0, 255, "NH:i:1"), rec("b", 0, 3, "NH:i:2")], 3, "fail"),
        ([rec("a", 0, 255, "NH:i:1"), rec("b", 0, 3, "NH:i:2")], 4, "pass"),
        # bowtie2 -k 1: MAPQ 255 for all, ties included
        ([rec("a", 0, 255, "AS:i:0", "XS:i:0")], 30, "fail"),
        # supplementary and unmapped records are not looked at
        ([rec("a", 2048, 1, "NH:i:2"), rec("b", 4, 0)], 1, "pass"),
    ],
)
def test_the_multimapper_evidence_follows_the_tags_of_each_aligner(tmp_path, records, q, expected):
    (tmp_path / "x.sam").write_text(sam_with(records))
    assert verdict(MAPQ, ["view", "-c", "-q", str(q), "x.sam"], tmp_path) == expected


# ---- samtools.stats_nm_missing -------------------------------------------------------------------


@pytest.fixture(scope="module")
def f32(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f32")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        task = build_task("F32", 4, twin, work, tools, "implicit")
        proc = tools.run(["bash", "-c", "samtools calmd -b reads.bam genome.fa > calmd.bam"], work)
        assert proc.returncode == 0, proc.stderr
        out[twin] = (task, work)
    return out


def test_stats_on_a_bam_without_nm_and_with_nm_tag_mismatches_fails(f32):
    _, work = f32["faulted"]
    decision, result = outcome(STATS, ["stats", "reads.bam"], work)
    assert result.verdict == "fail" and result.code == "nm_missing_mismatches_present"
    assert "calmd" in result.fix[0] and "NM MD" in result.fix[1]
    assert "-r reference.fa" in result.message
    assert decision.action == "block"


@pytest.mark.parametrize(
    "argv",
    [
        ["stats", "-r", "genome.fa", "reads.bam"],
        ["stats", "--ref-seq", "genome.fa", "reads.bam"],
        ["stats", "-@", "2", "-F", "0x100", "reads.bam"],
        ["stats", "-c", "1,100,1", "reads.bam", "chr1"],
    ],
)
def test_stats_variants_on_the_faulted_bam_fail(f32, argv):
    _, work = f32["faulted"]
    assert verdict(STATS, argv, work) == "fail"


@pytest.mark.parametrize(
    "argv",
    [
        # the first call of 4 of 5 faulted Haiku trials of F32: --reference is not -r, and its
        # value used to be taken for the input file
        ["stats", "--reference", "genome.fa", "reads.bam"],
        ["stats", "--reference=genome.fa", "reads.bam"],
        ["stats", "genome.fa", "reads.bam"],  # the first positional that is a BAM is the input
        ["stats", "--input-fmt", "bam", "--verbosity", "1", "reads.bam"],
        ["stats", "-@", "2", "--reference", "genome.fa", "-F", "0x100", "reads.bam", "chr1"],
        ["stats", "--input-fmt-option", "required_fields=0x1", "-r", "genome.fa", "reads.bam"],
    ],
)
def test_stats_with_a_global_option_or_a_reference_before_the_bam_is_judged_on_the_bam(f32, argv):
    _, work = f32["faulted"]
    result = outcome(STATS, argv, work)[1]
    assert result.verdict == "fail" and result.code == "nm_missing_mismatches_present"
    assert "reads.bam has no NM tag" in result.message and "genome.fa" not in result.message
    _, clean = f32["clean"]
    assert verdict(STATS, argv, clean) == "pass"


def test_stats_with_no_alignment_file_among_the_positionals_still_passes(f32):
    _, work = f32["faulted"]
    for argv in (["stats", "--reference", "genome.fa"], ["stats", "genome.fa"], ["stats", "-"]):
        assert verdict(STATS, argv, work) == "pass", argv
    assert verdict(STATS, ["stats", "--reference", "genome.fa", "absent.bam"], work) == "pass"


def test_a_bam_with_nm_or_after_calmd_or_declared_not_used_passes(f32):
    _, clean = f32["clean"]
    _, work = f32["faulted"]
    assert verdict(STATS, ["stats", "reads.bam"], clean) == "pass"
    assert verdict(STATS, ["stats", "calmd.bam"], work) == "pass"
    assert verdict(STATS, ["stats", "reads.bam"], work, {"stats_mismatches": "not_used"}) == "pass"
    assert verdict(STATS, ["stats", "-"], work) == "pass"
    assert verdict(STATS, ["stats", "absent.bam"], work) == "pass"
    assert verdict(STATS, ["flagstat", "reads.bam"], work) is None


def test_no_nm_and_no_sign_of_mismatches_only_warns(tmp_path):
    (tmp_path / "x.sam").write_text(
        sam_with([rec(f"r{i}", 0, 60, "NH:i:1", pos=10 + i) for i in range(6)])
    )
    decision, result = outcome(STATS, ["stats", "x.sam"], tmp_path)
    assert result.verdict == "unknown" and result.code == "nm_missing"
    assert decision.action == "warn"
    (tmp_path / "md.sam").write_text(sam_with([rec(f"r{i}", 0, 60, "MD:Z:3A6") for i in range(3)]))
    assert verdict(STATS, ["stats", "md.sam"], tmp_path) == "fail"  # MD shows a mismatch, NM absent
    (tmp_path / "ok.sam").write_text(sam_with([rec(f"r{i}", 0, 60, "NM:i:0") for i in range(3)]))
    assert verdict(STATS, ["stats", "ok.sam"], tmp_path) == "pass"
    (tmp_path / "none.sam").write_text(sam_with([rec("u", 4, 0)]))
    assert verdict(STATS, ["stats", "none.sam"], tmp_path) == "pass"  # nothing mapped


def test_stats_prints_zero_mismatches_for_the_faulted_bam_and_the_truth_after_calmd(f32, tools):
    task, work = f32["faulted"]
    for name, expected in (("reads.bam", 0), ("calmd.bam", task.truth["expected"])):
        proc = tools.run(["samtools", "stats", name], work)
        assert proc.returncode == 0 and proc.stderr == ""
        line = next(x for x in proc.stdout.splitlines() if x.startswith("SN\tmismatches:"))
        assert int(line.split("\t")[2]) == expected
    with_ref = tools.run(["samtools", "stats", "-r", "genome.fa", "reads.bam"], work)
    assert "SN\tmismatches:\t0\t" in with_ref.stdout and task.truth["expected"] > 0


def test_stats_with_the_reference_option_prints_zero_mismatches_too(f32, tools):
    task, work = f32["faulted"]
    for argv in (["--reference", "genome.fa", "reads.bam"], ["--reference=genome.fa", "reads.bam"]):
        proc = tools.run(["samtools", "stats", *argv], work)
        assert proc.returncode == 0 and proc.stderr == ""
        assert "SN\tmismatches:\t0\t" in proc.stdout and task.truth["expected"] > 0
    # a FASTA before the BAM is not a silent trap: samtools stops (the rule still names the BAM)
    assert tools.run(["samtools", "stats", "genome.fa", "reads.bam"], work).returncode != 0


# ---- samtools.fixmate_grouping -------------------------------------------------------------------


@pytest.fixture(scope="module")
def f36(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f36")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        task = build_task("F36", 5, twin, work, tools, "implicit")
        for argv in (
            ["samtools", "sort", "-n", "-o", "names.bam", "reads.bam"],
            ["samtools", "sort", "-o", "coord.bam", "reads.bam"],
        ):
            assert tools.run(argv, work).returncode == 0
        out[twin] = (task, work)
    return out


def test_scattered_mates_in_an_unsorted_bam_fail_with_the_name_sort_fix(f36):
    _, work = f36["faulted"]
    decision, result = outcome(FIXMATE, ["fixmate", "-m", "reads.bam", "out.bam"], work)
    assert result.verdict == "fail" and result.code == "mates_not_adjacent"
    assert result.fix[0].startswith("samtools sort -n -o names.bam reads.bam && samtools fixmate")
    assert "names.bam out.bam" in result.fix[0]
    assert "SO:unsorted" in result.message and decision.action == "block"


def test_grouped_name_sorted_collated_and_coordinate_sorted_inputs_pass(f36):
    _, clean = f36["clean"]
    _, work = f36["faulted"]
    assert verdict(FIXMATE, ["fixmate", "-m", "reads.bam", "o.bam"], clean) == "pass"
    assert verdict(FIXMATE, ["fixmate", "-m", "names.bam", "o.bam"], work) == "pass"
    # samtools refuses a coordinate-sorted header by itself: the rule leaves it to the tool
    assert verdict(FIXMATE, ["fixmate", "-m", "coord.bam", "o.bam"], work) == "pass"
    assert verdict(FIXMATE, ["fixmate", "-m", "-", "o.bam"], work) == "pass"
    assert verdict(FIXMATE, ["fixmate", "-m", "absent.bam", "o.bam"], work) == "pass"
    assert verdict(FIXMATE, ["markdup", "reads.bam", "o.bam"], work) is None


def test_options_before_the_input_do_not_hide_it(f36):
    _, work = f36["faulted"]
    for argv in (
        ["fixmate", "-mr", "-@", "2", "reads.bam", "o.bam"],
        ["fixmate", "-O", "bam", "-m", "reads.bam", "o.bam"],
        ["fixmate", "--output-fmt", "bam", "-m", "reads.bam", "o.bam"],
        ["fixmate", "-z", "all", "-m", "reads.bam", "o.bam"],
        ["fixmate", "--reference", "genome.fa", "-m", "reads.bam", "o.bam"],
        ["fixmate", "--input-fmt", "bam", "--output-fmt-option", "level=1", "reads.bam", "o.bam"],
    ):
        assert verdict(FIXMATE, argv, work) == "fail", argv


def test_few_paired_records_and_single_end_data_are_not_judged(tmp_path):
    (tmp_path / "se.sam").write_text(sam_with([rec(f"r{i}", 0, 60, pos=10 + i) for i in range(10)]))
    assert verdict(FIXMATE, ["fixmate", "se.sam", "o.bam"], tmp_path) == "pass"
    tiny = [rec("a", 65, 60), rec("x", 65, 60), rec("a", 129, 60)]
    (tmp_path / "tiny.sam").write_text(sam_with(tiny))
    assert verdict(FIXMATE, ["fixmate", "tiny.sam", "o.bam"], tmp_path) == "pass"  # 3 paired < 4


def test_fixmate_on_scattered_mates_is_silent_and_clears_the_pairing(f36, tools):
    _, work = f36["faulted"]
    before = tools.run(["samtools", "view", "-c", "-f", "2", "reads.bam"], work)
    run = tools.run(["samtools", "fixmate", "-m", "reads.bam", "ev_fx.bam"], work)
    assert run.returncode == 0 and run.stderr == ""
    after = tools.run(["samtools", "view", "-c", "-f", "2", "ev_fx.bam"], work)
    assert int(before.stdout) > 100 and int(after.stdout) <= 2
    coord = tools.run(["samtools", "fixmate", "-m", "coord.bam", "ev_c.bam"], work)
    assert coord.returncode != 0 and "Coordinate sorted" in coord.stderr


# ---- samtools.merge_combine_rg -------------------------------------------------------------------


@pytest.fixture(scope="module")
def f37(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f37")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F37", 6, twin, work, tools, "implicit"), work)
    return out


def test_combining_read_groups_of_different_samples_fails_with_the_command_without_c(f37):
    _, work = f37["faulted"]
    decision, result = outcome(
        MERGE, ["merge", "-c", "-f", "m.bam", "lane1.bam", "lane2.bam"], work
    )
    assert result.verdict == "fail" and result.code == "samples_share_read_group_id"
    assert result.fix[0] == "samtools merge -f m.bam lane1.bam lane2.bam"
    assert "sampleA" in result.message and "sampleB" in result.message
    assert decision.action == "block"


@pytest.mark.parametrize(
    "argv",
    [
        ["merge", "-cf", "m.bam", "lane1.bam", "lane2.bam"],
        ["merge", "-c", "-@", "2", "-o", "m.bam", "lane1.bam", "lane2.bam"],
        ["merge", "-f", "-c", "-p", "m.bam", "lane1.bam", "lane2.bam"],
        ["merge", "-c", "-f", "--no-PG", "m.bam", "lane1.bam", "lane2.bam"],
    ],
)
def test_spellings_of_merge_c_are_covered(f37, argv):
    _, work = f37["faulted"]
    assert verdict(MERGE, argv, work) == "fail"


def test_merge_without_c_distinct_ids_identical_groups_and_other_cases_pass(f37, tmp_path):
    _, clean = f37["clean"]
    _, work = f37["faulted"]
    assert verdict(MERGE, ["merge", "-f", "m.bam", "lane1.bam", "lane2.bam"], work) == "pass"
    assert verdict(MERGE, ["merge", "-r", "-f", "m.bam", "lane1.bam", "lane2.bam"], work) == "pass"
    assert verdict(MERGE, ["merge", "-c", "-f", "m.bam", "lane1.bam", "lane2.bam"], clean) == "pass"
    assert verdict(MERGE, ["merge", "-c", "m.bam", "lane1.bam", "absent.bam"], work) == "pass"
    assert verdict(MERGE, ["merge", "-c", "m.bam", "lane1.bam"], work) == "pass"
    # one sample sequenced on two lanes with an identical @RG line: -c is harmless
    for name in ("a.sam", "b.sam"):
        (tmp_path / name).write_text(
            "@HD\tVN:1.6\n@SQ\tSN:c\tLN:100\n@RG\tID:1\tSM:s\tLB:l\n"
            "r\t0\tc\t5\t60\t4M\t*\t0\t0\tACGT\tIIII\tRG:Z:1\n"
        )
    assert verdict(MERGE, ["merge", "-c", "m.bam", "a.sam", "b.sam"], tmp_path) == "pass"
    # same sample and ID, other fields differ: a warning, not a block
    text = (tmp_path / "b.sam").read_text().replace("LB:l", "LB:other")
    (tmp_path / "b.sam").write_text(text)
    decision, result = outcome(MERGE, ["merge", "-c", "m.bam", "a.sam", "b.sam"], tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"


def test_a_global_option_before_the_output_name_does_not_make_the_output_an_input(f37, tmp_path):
    _, work = f37["faulted"]
    for name in ("lane1.bam", "lane2.bam"):
        shutil.copy(work / name, tmp_path / name)
    shutil.copy(work / "lane1.bam", tmp_path / "m.bam")  # the output of an earlier run
    argv = ["merge", "-c", "-f", "--reference", "genome.fa", "m.bam", "lane1.bam", "lane2.bam"]
    result = outcome(MERGE, argv, tmp_path)[1]
    assert result.verdict == "fail" and "m.bam" not in result.message


def test_inputs_given_by_a_list_file_are_read(f37, tmp_path):
    _, work = f37["faulted"]
    (work / "list.txt").write_text("lane1.bam\nlane2.bam\n")
    assert verdict(MERGE, ["merge", "-c", "-f", "-b", "list.txt", "m.bam"], work) == "fail"


def test_merge_c_collapses_the_samples_and_plain_merge_keeps_them(f37, tools):
    task, work = f37["faulted"]
    from biowarden.bench.tasks.extra.f37_merge_rg import reads_per_sample

    for options, collapsed in (([], False), (["-c"], True)):
        out = "ev_c.bam" if collapsed else "ev_plain.bam"
        argv = ["samtools", "merge", "-f", *options, out, "lane1.bam", "lane2.bam"]
        run = tools.run(argv, work)
        assert run.returncode == 0 and run.stderr == ""
        counts = reads_per_sample(work, tools, out)
        if collapsed:
            assert counts == task.truth["naive"] and counts["sampleB"] == 0
        else:
            assert counts == task.truth["expected"]


# ---- samtools.idxstats_unmapped ------------------------------------------------------------------


@pytest.fixture(scope="module")
def f38(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f38")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F38", 7, twin, work, tools, "implicit"), work)
    return out


def test_placed_unmapped_mates_make_the_star_row_a_partial_count(f38, tools):
    decision, result = outcome(IDXSTATS, ["idxstats", "reads.bam"], f38["faulted"][1], tools=tools)
    assert result.verdict == "unknown" and result.code == "unmapped_mates_on_contig_rows"
    assert decision.action == "warn" and "view -c -f 4" in result.fix[1]


def test_declared_star_row_only_fails_and_contig_rows_included_passes(f38, tools):
    _, work = f38["faulted"]
    _, result = outcome(
        IDXSTATS, ["idxstats", "reads.bam"], work, {"idxstats_unmapped": "star_row_only"}, tools
    )
    assert result.verdict == "fail" and result.code == "star_row_misses_placed_mates"
    task = f38["faulted"][0]
    assert str(task.truth["expected"]) in result.message
    ok = {"idxstats_unmapped": "contig_rows_included"}
    assert verdict(IDXSTATS, ["idxstats", "reads.bam"], work, ok, tools) == "pass"


def test_the_clean_twin_and_missing_pieces_pass(f38, tools, tmp_path):
    _, clean = f38["clean"]
    decl = {"idxstats_unmapped": "star_row_only"}
    assert verdict(IDXSTATS, ["idxstats", "reads.bam"], clean, decl, tools) == "pass"
    _, work = f38["faulted"]
    assert verdict(IDXSTATS, ["idxstats", "absent.bam"], work, None, tools) == "pass"
    assert verdict(IDXSTATS, ["idxstats", "-"], work, None, tools) == "pass"
    (tmp_path / "unindexed.bam").write_bytes((work / "reads.bam").read_bytes())
    # without an index idxstats still works ("reverting to slow method"), so the rule still looks
    assert verdict(IDXSTATS, ["idxstats", "unindexed.bam"], tmp_path, None, tools) == "unknown"
    assert verdict(IDXSTATS, ["idxstats", "-@", "2", "reads.bam"], work, None, tools) == "unknown"
    # without a samtools the engine can run, the rule says so instead of guessing
    assert verdict(IDXSTATS, ["idxstats", "reads.bam"], work, None, None) == "unknown"


def test_idxstats_puts_placed_unmapped_mates_on_the_contig_rows(f38, tools):
    task, work = f38["faulted"]
    proc = tools.run(["samtools", "idxstats", "reads.bam"], work)
    rows = [line.split("\t") for line in proc.stdout.splitlines()]
    star = next(r for r in rows if r[0] == "*")
    assert sum(int(r[3]) for r in rows) == task.truth["expected"]
    assert int(star[3]) == task.truth["naive"] < task.truth["expected"]
    count = tools.run(["samtools", "view", "-c", "-f", "4", "reads.bam"], work)
    assert int(count.stdout) == task.truth["expected"]


# ---- samtools.split_flag_after_bwa_m -------------------------------------------------------------


@pytest.fixture(scope="module")
def f39(tools, tmp_path_factory):
    from biowarden.bench.tasks.extra.f39_bwa_split import align

    base = tmp_path_factory.mktemp("f39")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        task = build_task("F39", 3, twin, work, tools, "implicit")
        align(work, tools, "aln.sam")
        assert (
            tools.run(["samtools", "view", "-b", "-o", "aln.bam", "aln.sam"], work).returncode == 0
        )
        out[twin] = (task, work)
    return out


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-c", "-f", "0x800", "aln.bam"],
        ["view", "-c", "-f", "2048", "aln.bam"],
        ["view", "-cf0x800", "aln.bam"],
        ["view", "-c", "--require-flags", "0x800", "aln.bam"],
        ["view", "-c", "-F", "0x800", "aln.bam"],
        ["view", "-c", "-f", "0x800", "aln.sam"],
    ],
)
def test_a_0x800_test_on_a_file_made_with_bwa_mem_m_fails(f39, argv):
    _, work = f39["faulted"]
    decision, result = outcome(SPLIT, argv, work)
    assert result.verdict == "fail" and result.code == "split_hits_are_secondary"
    assert "exists([SA])" in result.fix[0] and decision.action == "block"
    assert "-M" in result.message


def test_without_m_other_flag_tests_and_other_inputs_the_rule_is_silent(f39, tmp_path):
    _, clean = f39["clean"]
    _, work = f39["faulted"]
    assert verdict(SPLIT, ["view", "-c", "-f", "0x800", "aln.bam"], clean) == "pass"
    for argv in (
        ["view", "-c", "-F", "0x904", "-e", "exists([SA])", "aln.bam"],
        ["view", "-c", "-f", "0x100", "aln.bam"],
        ["view", "-c", "-F", "0x900", "aln.bam"],  # tests 0x100 as well
        ["view", "-c", "-F", "0x904", "aln.bam"],
        ["view", "-c", "aln.bam"],
        ["view", "-c", "-f", "0x800", "absent.bam"],
        ["view", "-c", "-f", "0x800", "-"],
    ):
        assert verdict(SPLIT, argv, work) == "pass", argv
    assert verdict(SPLIT, ["sort", "-f", "0x800", "aln.bam"], work) is None


def test_the_pg_line_decides_not_the_file_name(tmp_path):
    def sam(cl: str) -> str:
        return (
            "@HD\tVN:1.6\n@SQ\tSN:c\tLN:100\n"
            f"@PG\tID:bwa\tPN:bwa\tVN:0.7.19\tCL:{cl}\n"
            "r\t0\tc\t5\t60\t4M\t*\t0\t0\tACGT\tIIII\n"
        )

    for cl, expected in (
        ("bwa mem -M ref.fa r.fq", "fail"),
        ("/opt/bin/bwa mem -t 4 -M -R @RG ref.fa r.fq", "fail"),
        ("bwa mem -5SPM ref.fa r.fq", "fail"),
        ("bwa mem -t 4 ref.fa r.fq", "pass"),
        ("bwa mem -Y -R @RG ref.fa r.fq", "pass"),
        ("bwa aln -M ref.fa r.fq", "pass"),
        ("bowtie2 -M 5 -x idx r.fq", "pass"),
    ):
        (tmp_path / "x.sam").write_text(sam(cl))
        assert verdict(SPLIT, ["view", "-c", "-f", "0x800", "x.sam"], tmp_path) == expected, cl


def test_bwa_mem_m_writes_the_split_parts_as_secondary_and_the_sa_tag_stays(f39, tools):
    for twin, flag in (("clean", 0x800), ("faulted", 0x100)):
        task, work = f39[twin]

        def count(*args: str, work=work) -> int:
            return int(tools.run(["samtools", "view", "-c", *args, "aln.bam"], work).stdout)

        assert count("-f", hex(flag)) == task.truth["expected"]
        assert count("-F", "0x904", "-e", "exists([SA])") == task.truth["expected"]
    task, work = f39["faulted"]
    assert int(tools.run(["samtools", "view", "-c", "-f", "0x800", "aln.bam"], work).stdout) == 0
    flagstat = tools.run(["samtools", "flagstat", "aln.bam"], work).stdout
    assert "0 + 0 supplementary" in flagstat


# ---- admission -----------------------------------------------------------------------------------


def run_admission(rule_id: str, tools, tmp_path, seeds: int = ADMISSION_SEEDS):
    stats = measure(rule_id, [SPEC], range(seeds), tools, tmp_path)
    right_declared = stats.cells[("right", True)]
    assert stats.admitted()
    assert right_declared.n >= MIN_RIGHT_CALLS * seeds // ADMISSION_SEEDS
    assert right_declared.failed == 0
    assert stats.cells[("right", False)].failed == 0  # conditional: no friction on correct calls
    assert stats.killed() > 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert promotion_is_consistent(stats)
    return stats


HEAVY = max(10, ADMISSION_SEEDS // 2)  # these two build STAR runs or four samtools steps per call


def test_admission_mapq_unique_cutoff(tools, tmp_path):
    stats = run_admission(MAPQ, tools, tmp_path, HEAVY)
    # every undeclared cutoff of 1-3 on the faulted twin is caught: 7 templates per seed
    assert stats.cells[("wrong", False)].failed >= HEAVY * 5


def test_admission_stats_nm_missing(tools, tmp_path):
    stats = run_admission(STATS, tools, tmp_path)
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 8


def test_admission_fixmate_grouping(tools, tmp_path):
    stats = run_admission(FIXMATE, tools, tmp_path, HEAVY)
    assert stats.recall() == 1.0  # every scattered-input call is caught


def test_admission_merge_combine_rg(tools, tmp_path):
    stats = run_admission(MERGE, tools, tmp_path)
    assert stats.recall() == 1.0


def test_admission_idxstats_unmapped(tools, tmp_path):
    stats = run_admission(IDXSTATS, tools, tmp_path)
    assert stats.cells[("wrong", True)].failed > 0


def test_admission_split_flag_after_bwa_m(tools, tmp_path):
    stats = run_admission(SPLIT, tools, tmp_path)
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 5


def test_dependencies_are_stdlib_only():
    import ast

    for name in ("aln_bam", "aln_samtools"):
        tree = ast.parse(Path(rules.__file__).with_name(f"{name}.py").read_text())
        top = {
            n.names[0].name.split(".")[0]
            if isinstance(n, ast.Import)
            else (n.module or "").split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        }
        assert top <= {
            "__future__",
            "collections",
            "dataclasses",
            "gzip",
            "pathlib",
            "re",
            "shlex",
            "struct",
            "subprocess",
            "typing",
            "zlib",
            "biowarden",
        }, top
