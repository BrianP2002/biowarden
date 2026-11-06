"""Rules htseq.strandedness and htseq.min_mapq: spec, checkers on real argv and BAMs, admission.

Workspaces are the benchmark's F41 (single-end, forward / dUTP / unstranded libraries) and F44
(unstranded, MAPQ 0-3 reads in the faulted twin) built with the real tools. The admission tests
use 12 seeds (not 40): every case needs real htseq-count runs (0.4 s each).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_rules_quant_common import engine_for, outcome, quant_tools, workspace

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_quant_htseq import (
    MAPQ_TEMPLATES,
    STRAND_TEMPLATES,
    true_library,
)
from biowarden.engine import Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import quant_htseq as qh
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, promotion_is_consistent

STRAND, MAPQ = "htseq.strandedness", "htseq.min_mapq"
SEEDS = min(ADMISSION_SEEDS, 12)
MIN_RIGHT = max(1, 200 * SEEDS // 300)
FILES = ["reads.bam", "genes.gtf"]


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("quant_htseq")


@pytest.fixture(scope="module")
def eng():
    return engine_for("quant_htseq")


# ---- the spec ------------------------------------------------------------------------------------


def test_spec_declares_two_conditional_t1_call_rules(spec):
    rules = {r.id: r for r in spec.rules}
    assert set(rules) == {STRAND, MAPQ}
    for rule in rules.values():
        assert (rule.kind, rule.tier, rule.applies.tools) == ("call", "T1", ["htseq-count"])
        assert rule.declare is not None and rule.declare.required is False
        assert rule.check in known_checkers() or engine_for("quant_htseq")
    assert rules[STRAND].declare.key == "library_strandedness"
    assert rules[STRAND].declare.values == ["unstranded", "forward", "reverse"]
    assert (rules[STRAND].slot, rules[STRAND].incidents) == ("strand", ["GEN-22", "GEN-20"])
    assert rules[MAPQ].declare.key == "low_mapq_reads" and rules[MAPQ].declare.values == [
        "keep",
        "drop",
    ]
    assert rules[MAPQ].slot == "default_param"
    schema = spec.declaration_schema()
    assert schema["required"] == [] and set(schema["properties"]) == {
        "library_strandedness",
        "low_mapq_reads",
    }


def test_spec_rejects_a_t0_tier_for_a_rule_without_a_declaration(spec):
    data = spec.model_dump(mode="json")
    data["rules"][0]["tier"] = "T0"
    data["rules"][0]["declare"] = None
    with pytest.raises(ValidationError, match="needs a declaration"):
        Spec.model_validate(data)


# ---- htseq.strandedness --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "twin", "argv", "verdict"),
    [
        (0, "clean", FILES, "pass"),  # forward library, default -s yes
        (0, "clean", ["-s", "yes", *FILES], "pass"),
        (0, "clean", ["--stranded=yes", *FILES], "pass"),
        (
            0,
            "clean",
            ["-s", "no", *FILES],
            "pass",
        ),  # unstranded call on a stranded library: no overlaps
        (0, "clean", ["-s", "reverse", *FILES], "fail"),
        (0, "faulted", FILES, "fail"),  # dUTP library, the default counts 4 % of the reads
        (0, "faulted", ["-q", *FILES], "fail"),
        (0, "faulted", ["-s", "yes", *FILES], "fail"),
        (0, "faulted", ["--stranded", "yes", *FILES], "fail"),
        (0, "faulted", ["-s", "reverse", *FILES], "pass"),
        (0, "faulted", ["--stranded=reverse", "-q", *FILES], "pass"),
        (0, "faulted", ["-s", "no", *FILES], "pass"),
        (1, "faulted", FILES, "fail"),  # unstranded library: half of the reads
        (1, "faulted", ["-s", "reverse", *FILES], "fail"),
        (1, "faulted", ["-s", "no", *FILES], "pass"),
        (1, "faulted", ["-s", "no", "-r", "pos", "-t", "exon", "-i", "gene_id", *FILES], "pass"),
    ],
)
def test_strandedness_verdict_on_real_workspaces(tools, tmp_path, eng, seed, twin, argv, verdict):
    work, _ = workspace(tools, tmp_path, "F41", seed, twin)
    decision, result = outcome(eng, STRAND, "htseq-count", argv, work)
    assert result is not None and result.verdict == verdict, result
    assert (decision.action == "block") == (verdict == "fail")


def test_the_block_message_starts_with_the_fix_and_names_the_numbers(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F41", 0, "faulted")
    decision, result = outcome(eng, STRAND, "htseq-count", FILES, work)
    assert result.fix[0].startswith("re-run with -s reverse")
    text = render_block(decision, eng.retry_cap, eng.style)
    assert (
        text.startswith("FIX: ")
        and "reverse-stranded (dUTP)" in text
        and "-s yes (the default)" in text
    )
    assert "reads.bam" in result.message and "reads sit on their gene's strand" in result.message


@pytest.mark.parametrize(
    ("argv", "declared", "verdict", "code"),
    [
        (FILES, {"library_strandedness": "reverse"}, "fail", "declaration_contradicts_options"),
        (["-s", "reverse", *FILES], {"library_strandedness": "reverse"}, "pass", None),
        (["-s", "reverse", *FILES], {"library_strandedness": "forward"}, "fail", None),
        (["-s", "reverse", *FILES], {"library_strandedness": "unstranded"}, "fail", None),
        (
            ["-s", "no", *FILES],
            {"library_strandedness": "unstranded"},
            "fail",
            "declaration_contradicts_data",
        ),
        (FILES, {"library_strandedness": "forward"}, "fail", "declaration_contradicts_data"),
    ],
)
def test_declarations_are_checked_against_the_options_and_the_reads(
    tools, tmp_path, eng, argv, declared, verdict, code
):
    work, _ = workspace(tools, tmp_path, "F41", 0, "faulted")  # a dUTP library
    _, result = outcome(eng, STRAND, "htseq-count", argv, work, declared)
    assert result.verdict == verdict, result
    if code:
        assert result.code == code


def test_a_ambiguous_library_is_unknown_for_a_stranded_call_and_silent_otherwise(
    tools, tmp_path, eng
):
    work, _ = workspace(tools, tmp_path, "F41", 0, "clean")
    sam = tools.run(["samtools", "view", "-h", "reads.bam"], work).stdout.splitlines()
    mixed = []
    for i, line in enumerate(sam):
        f = line.split("\t")
        if not line.startswith("@") and i % 4 == 0:
            f[1] = str(int(f[1]) ^ 16)  # flip a quarter: about 71 % on the gene's strand
        mixed.append("\t".join(f))
    (work / "mixed.sam").write_text("\n".join(mixed) + "\n")
    _, stranded = outcome(eng, STRAND, "htseq-count", ["mixed.sam", "genes.gtf"], work)
    assert stranded.verdict == "unknown" and stranded.code == "library_not_provable"
    _, unstranded = outcome(
        eng, STRAND, "htseq-count", ["-s", "no", "mixed.sam", "genes.gtf"], work
    )
    assert unstranded.verdict == "pass"
    _, declared = outcome(
        eng,
        STRAND,
        "htseq-count",
        ["mixed.sam", "genes.gtf"],
        work,
        {"library_strandedness": "forward"},
    )
    assert declared.verdict == "pass"


def test_missing_files_stdin_and_too_few_arguments_are_not_judged(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F41", 0, "faulted")
    for argv in (
        ["missing.bam", "genes.gtf"],
        ["reads.bam", "missing.gtf"],
        ["-", "genes.gtf"],
        ["reads.bam"],
        [],
        ["--help"],
    ):
        _, result = outcome(eng, STRAND, "htseq-count", argv, work)
        assert result is not None and result.verdict == "pass", argv


def test_feature_type_and_id_attribute_options_reach_the_gtf_reader(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F41", 0, "faulted")
    _, by_tx = outcome(eng, STRAND, "htseq-count", ["-i", "transcript_id", *FILES], work)
    assert by_tx.verdict == "fail"  # the exons carry transcript_id too
    _, no_such = outcome(eng, STRAND, "htseq-count", ["-t", "CDS", *FILES], work)
    assert no_such.verdict == "pass"  # no CDS rows: nothing to judge, no crash


def test_the_checker_reads_files_so_it_is_t1_and_a_t0_context_refuses(tools, tmp_path, spec):
    work, _ = workspace(tools, tmp_path, "F41", 0, "faulted")
    from biowarden.engine import Call

    rule = next(r for r in spec.rules if r.id == STRAND)
    check = known_checkers()["htseq_strandedness"]
    call = Call("htseq-count", tuple(FILES), work)
    with pytest.raises(TierViolation):
        check(call, Match(rule), Ctx(work, "T0"))
    assert check(call, Match(rule), Ctx(work, "T1")).verdict == "fail"


# ---- htseq.min_mapq ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("twin", "argv", "declared", "verdict"),
    [
        ("faulted", ["-s", "no", *FILES], None, "fail"),
        ("faulted", ["-s", "no", "-a", "10", *FILES], None, "fail"),
        ("faulted", ["-s", "no", "--minaqual=10", *FILES], None, "fail"),
        ("faulted", ["-s", "no", "-a", "1", *FILES], None, "fail"),  # still drops MAPQ 0
        ("faulted", ["-s", "no", "-a", "0", *FILES], None, "pass"),
        ("faulted", ["-s", "no", "--minaqual", "0", *FILES], None, "pass"),
        ("faulted", ["-s", "no", *FILES], {"low_mapq_reads": "drop"}, "pass"),
        ("faulted", ["-s", "no", *FILES], {"low_mapq_reads": "keep"}, "fail"),
        ("faulted", ["-s", "no", "-a", "0", *FILES], {"low_mapq_reads": "keep"}, "pass"),
        ("faulted", ["-s", "no", "-a", "0", *FILES], {"low_mapq_reads": "drop"}, "pass"),
        ("clean", ["-s", "no", *FILES], None, "pass"),  # every MAPQ is 30 or more
        ("clean", ["-s", "no", *FILES], {"low_mapq_reads": "keep"}, "pass"),
        ("clean", ["-s", "no", "-a", "31", *FILES], None, "fail"),  # a high floor drops real reads
    ],
)
def test_min_mapq_verdict_on_real_workspaces(tools, tmp_path, eng, twin, argv, declared, verdict):
    work, _ = workspace(tools, tmp_path, "F44", 0, twin)
    _, result = outcome(eng, MAPQ, "htseq-count", argv, work, declared)
    assert result.verdict == verdict, result


def test_multimappers_below_the_floor_are_left_to_nonunique(tools, tmp_path, eng):
    """A STAR BAM gives MAPQ 3 / 1 / 0 to multi-mappers (NH > 1): htseq-count drops those."""
    work, _ = workspace(tools, tmp_path, "F44", 0, "clean")
    header = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:5000\n"
    seq, qual = "A" * 50, "I" * 50
    rows = [
        f"u{i}\t0\tchr1\t{100 + i}\t255\t50M\t*\t0\t0\t{seq}\t{qual}\tNH:i:1" for i in range(40)
    ]
    rows += [f"m{i}\t0\tchr1\t{100 + i}\t3\t50M\t*\t0\t0\t{seq}\t{qual}\tNH:i:2" for i in range(40)]
    (work / "star.sam").write_text(header + "\n".join(rows) + "\n")
    _, result = outcome(eng, MAPQ, "htseq-count", ["-s", "no", "star.sam", "genes.gtf"], work)
    assert result.verdict == "pass"
    rows += [
        f"b{i}\t0\tchr1\t{100 + i}\t2\t50M\t*\t0\t0\t{seq}\t{qual}" for i in range(5)
    ]  # no NH tag
    (work / "bwa.sam").write_text(header + "\n".join(rows) + "\n")
    _, result = outcome(eng, MAPQ, "htseq-count", ["-s", "no", "bwa.sam", "genes.gtf"], work)
    assert result.verdict == "fail" and "5 of" in result.message


def test_a_few_low_reads_do_not_trigger(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F44", 0, "clean")
    header = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:5000\n"
    seq, qual = "A" * 50, "I" * 50
    rows = [f"u{i}\t0\tchr1\t{100 + i}\t60\t50M\t*\t0\t0\t{seq}\t{qual}" for i in range(500)]
    rows += [f"l{i}\t0\tchr1\t{100 + i}\t1\t50M\t*\t0\t0\t{seq}\t{qual}" for i in range(4)]  # 0.8 %
    (work / "few.sam").write_text(header + "\n".join(rows) + "\n")
    _, result = outcome(eng, MAPQ, "htseq-count", ["-s", "no", "few.sam", "genes.gtf"], work)
    assert result.verdict == "pass"


def test_a_bad_minaqual_value_is_left_to_htseq(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F44", 0, "faulted")
    _, result = outcome(eng, MAPQ, "htseq-count", ["-a", "ten", *FILES], work)
    assert result.verdict == "pass"


# ---- the spellings of the admission templates, run for real ----------------------------------


def test_every_spelling_gives_the_numbers_of_its_key(tools, tmp_path):
    """The admission labels are computed once per key: check that the spellings agree with them."""
    for family, templates in (("F41", STRAND_TEMPLATES), ("F44", MAPQ_TEMPLATES)):
        for twin in ("clean", "faulted"):
            work, _ = workspace(tools, tmp_path, family, 0, twin)
            reference = {}
            for t in templates:
                proc = tools.run(
                    [t.tool, "-q", *t.argv] if "-q" not in t.argv else [t.tool, *t.argv], work
                )
                assert proc.returncode == 0 and proc.stderr == "", (t.name, proc.stderr)
                rows = (line.split("\t") for line in proc.stdout.splitlines())
                counts = {r[0]: int(r[1]) for r in rows if not r[0].startswith("__")}
                assert counts, t.name
                assert reference.setdefault(t.key, counts) == counts, (family, twin, t.name)


# ---- admission -----------------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", [STRAND, MAPQ])
def test_admission(spec, tools, tmp_path, rule_id):
    stats = measure(rule_id, [spec], range(SEEDS), tools, tmp_path)
    right_declared, right_undeclared = stats.cells[("right", True)], stats.cells[("right", False)]
    assert stats.conditional and stats.admitted() and stats.killed() > 0
    assert right_declared.n >= MIN_RIGHT and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT and right_undeclared.failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert (
        stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    )  # every wrong call
    assert stats.cells[("wrong", True)].failed > 0
    assert promotion_is_consistent(stats)


def test_true_library_matches_the_family_plan():
    assert [true_library(s, "faulted") for s in range(4)] == ["reverse", "unstranded"] * 2
    assert true_library(3, "clean") == "forward"


def test_the_module_docstring_evidence_is_current():
    assert "htseq-count" in (qh.__doc__ or "") and qh.DEFAULT_MINAQUAL == 10
