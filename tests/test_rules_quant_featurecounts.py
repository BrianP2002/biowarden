"""Rules featurecounts.read_pairs, featurecounts.strandedness and featurecounts.counts_unit.

Workspaces are the benchmark's F42 (single-end clean twin, paired faulted twin) and F43 (dUTP
library, gene pair overlapping on opposite strands in the faulted twin), built with the real tools;
the counts tables of the fact rule are written by real featureCounts runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_rules_quant_common import engine_for, outcome, quant_tools, workspace

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_quant_featurecounts import (
    PAIR_KEYS,
    PAIR_TEMPLATES,
    STRAND_TEMPLATES,
    counts_or_none,
)
from biowarden.engine import Call, Spec
from biowarden.engine.checks import Ctx, Match, known_checkers, suggestion_for
from biowarden.engine.render import render_block
from biowarden.engine.rules import quant_featurecounts as qf
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

PAIRS, STRAND, UNIT = (
    "featurecounts.read_pairs",
    "featurecounts.strandedness",
    "featurecounts.counts_unit",
)
OUT = ["-a", "genes.gtf", "-o", "counts.txt"]


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("quant_featurecounts")


@pytest.fixture(scope="module")
def eng():
    return engine_for("quant_featurecounts")


# ---- the spec ------------------------------------------------------------------------------------


def test_spec_shapes_and_declarations(spec):
    rules = {r.id: r for r in spec.rules}
    assert set(rules) == {PAIRS, STRAND, UNIT}
    pairs, strand, unit = rules[PAIRS], rules[STRAND], rules[UNIT]
    assert (pairs.kind, pairs.tier, pairs.applies.tools, pairs.applies.options_any) == (
        "call",
        "T0",
        ["featureCounts"],
        ["-p"],
    )
    assert (pairs.declare.key, pairs.declare.values, pairs.declare.required) == (
        "count_unit",
        ["read", "fragment"],
        True,
    )
    assert (strand.kind, strand.tier, strand.declare.required) == ("call", "T1", False)
    assert strand.declare.values == ["unstranded", "forward", "reverse"]
    assert (unit.kind, unit.tier, unit.declare.required, unit.applies.tools) == (
        "fact",
        "T1",
        False,
        [],
    )
    assert pairs.incidents == ["GEN-24"] and unit.slot == "units" and pairs.slot == "record_count"
    for rule in rules.values():
        assert rule.check in known_checkers()
    assert suggestion_for("featurecounts_read_pairs") is not None
    assert spec.declaration_schema()["required"] == ["count_unit"]


def test_spec_rejects_a_fact_rule_without_file_patterns_or_at_tier_t0(spec):
    data = spec.model_dump(mode="json")
    data["rules"][2]["applies"]["file_patterns"] = []
    with pytest.raises(ValidationError, match="needs applies.file_patterns"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["rules"][2]["tier"] = "T0"
    with pytest.raises(ValidationError, match="cannot be tier T0"):
        Spec.model_validate(data)


@pytest.mark.parametrize(
    ("name", "should_match"),
    [
        ("counts.txt", True),
        ("gene_counts.txt", True),
        ("featureCounts_output.tsv", True),
        ("run/sample1.counts.txt", True),
        ("counts.txt.summary", False),
        ("counts.csv", False),
        ("mycount.txt.bak", False),
        ("results.txt", False),
    ],
)
def test_counts_file_pattern(spec, name, should_match):
    (pattern,) = next(r for r in spec.rules if r.id == UNIT).applies.file_patterns
    hits = [m.group(0) for m in re.finditer(pattern, f"awk '{{print}}' {name} | head")]
    assert bool(hits) is should_match
    if should_match:
        assert hits == [name]


# ---- featurecounts.read_pairs (T0: argv only) ----------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "declared", "verdict", "code"),
    [
        (["-p", *OUT, "reads.bam"], None, "fail", "declaration_missing"),
        (["-p", *OUT, "reads.bam"], {"count_unit": "read"}, "pass", None),
        (["-p", *OUT, "reads.bam"], {"count_unit": "fragment"}, "fail", "declared_mismatch"),
        (["-p", "--countReadPairs", *OUT, "reads.bam"], {"count_unit": "fragment"}, "pass", None),
        (
            ["-p", "--countReadPairs", *OUT, "reads.bam"],
            {"count_unit": "read"},
            "fail",
            "declared_mismatch",
        ),
        (
            ["--countReadPairs", "-T", "1", "-p", *OUT, "reads.bam"],
            {"count_unit": "fragment"},
            "pass",
            None,
        ),
        (["-T", "4", "-p", "-s", "2", *OUT, "reads.bam"], None, "fail", "declaration_missing"),
        (["-p", "--countReadPairs", *OUT, "reads.bam"], None, "fail", "declaration_missing"),
        (
            ["-p", "-B", "-C", "--countReadPairs", *OUT, "reads.bam"],
            {"count_unit": "fragment"},
            "pass",
            None,
        ),
    ],
)
def test_read_pairs_verdicts(tmp_path, eng, argv, declared, verdict, code):
    _, result = outcome(eng, PAIRS, "featureCounts", argv, tmp_path, declared)
    assert result is not None and result.verdict == verdict, result
    if code:
        assert result.code == code


@pytest.mark.parametrize(
    "argv",
    [
        [*OUT, "reads.bam"],  # no -p: featureCounts itself stops on a paired BAM
        ["--primary", *OUT, "reads.bam"],  # --primary is not -p
        ["-P", "-d", "50", *OUT, "reads.bam"],  # -P is another option
        ["-s", "2", *OUT, "reads.bam"],
        ["--countReadPairs", *OUT, "reads.bam"],  # ignored without -p
    ],
)
def test_calls_without_dash_p_are_not_matched(tmp_path, eng, argv):
    _, result = outcome(eng, PAIRS, "featureCounts", argv, tmp_path)
    assert result is None


def test_read_pairs_fix_is_concrete_and_comes_first(tmp_path, eng):
    decision, result = outcome(eng, PAIRS, "featureCounts", ["-p", *OUT, "reads.bam"], tmp_path)
    assert result.fix[0].startswith(
        "for fragments add --countReadPairs and --declare count_unit=fragment"
    )
    text = render_block(decision, eng.retry_cap, eng.style)
    assert text.startswith("FIX: ") and "count_unit" in text
    _, result = outcome(
        eng, PAIRS, "featureCounts", ["-p", "--countReadPairs", *OUT, "x.bam"], tmp_path
    )
    assert result.fix[0].startswith("re-run with --declare count_unit=fragment")


def test_the_pairs_checker_reads_nothing_so_a_t0_context_is_enough(spec, tmp_path):
    rule = next(r for r in spec.rules if r.id == PAIRS)
    call = Call("featureCounts", ("-p", *OUT, "x.bam"), tmp_path, {"count_unit": "read"})
    assert (
        known_checkers()["featurecounts_read_pairs"](call, Match(rule), Ctx(tmp_path, "T0")).verdict
        == "pass"
    )


# ---- featurecounts.strandedness ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("twin", "extra", "verdict", "code"),
    [
        ("clean", [], "pass", None),  # dUTP library, genes clear of each other: -s 0 loses nothing
        ("clean", ["-s", "2"], "pass", None),
        ("clean", ["-s", "1"], "fail", "stranded_option_mismatch"),
        ("clean", ["-s1"], "fail", "stranded_option_mismatch"),
        ("faulted", [], "fail", "unstranded_on_stranded_overlap"),
        ("faulted", ["-s", "0"], "fail", "unstranded_on_stranded_overlap"),
        ("faulted", ["-s", "1"], "fail", "stranded_option_mismatch"),
        ("faulted", ["-s", "2"], "pass", None),
        ("faulted", ["-s2", "-T", "1", "-t", "exon", "-g", "gene_id"], "pass", None),
        ("faulted", ["-s", "0,0"], "fail", "unstranded_on_stranded_overlap"),
        ("faulted", ["-s", "0,2"], "pass", None),  # per-file values that differ are not judged
    ],
)
def test_strandedness_verdicts_on_the_antisense_workspace(
    tools, tmp_path, eng, twin, extra, verdict, code
):
    work, _ = workspace(tools, tmp_path, "F43", 0, twin)
    _, result = outcome(eng, STRAND, "featureCounts", [*extra, *OUT, "reads.bam"], work)
    assert result.verdict == verdict, result
    if code:
        assert result.code == code
    if verdict == "fail":
        assert result.fix[0].startswith("re-run with -s 2")


def test_strandedness_declarations_and_unjudged_calls(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F43", 0, "faulted")
    for extra, declared, verdict in (
        (["-s", "2"], "reverse", "pass"),
        (["-s", "2"], "unstranded", "fail"),
        (["-s", "0"], "reverse", "fail"),
        (["-s", "2"], "forward", "fail"),
    ):
        _, result = outcome(
            eng,
            STRAND,
            "featureCounts",
            [*extra, *OUT, "reads.bam"],
            work,
            {"library_strandedness": declared},
        )
        assert result.verdict == verdict, (extra, declared, result)
    # SAF annotation, a missing BAM or a missing GTF: nothing to judge
    for argv in (
        ["-F", "SAF", *OUT, "reads.bam"],
        [*OUT, "missing.bam"],
        ["-a", "missing.gtf", "-o", "c.txt", "reads.bam"],
        ["-o", "c.txt", "reads.bam"],
    ):
        _, result = outcome(eng, STRAND, "featureCounts", argv, work)
        assert result.verdict == "pass", argv


def test_strandedness_message_names_the_pair_and_the_reads(tools, tmp_path, eng):
    work, task = workspace(tools, tmp_path, "F43", 0, "faulted")
    decision, result = outcome(eng, STRAND, "featureCounts", [*OUT, "reads.bam"], work)
    assert re.search(r"64 of its 299 reads touch genes on both strands", result.message)
    assert "-s 0 (the default)" in result.message and "reverse-stranded (dUTP)" in result.message
    assert any("overlap on opposite strands" in e for e in task.evidence)
    text = render_block(decision, eng.retry_cap, eng.style)
    assert text.startswith("FIX: re-run with -s 2")


# ---- featurecounts.counts_unit (fact) ------------------------------------------------------------


def run_counts(tools, work, name, *args):
    argv = ["featureCounts", "-a", "genes.gtf", "-o", name, *args, "reads.bam"]
    assert tools.run(argv, work).returncode == 0


@pytest.fixture()
def tables(tools, tmp_path):
    """The F42 paired workspace with a read and a fragment table, and the single-end one."""
    work, _ = workspace(tools, tmp_path, "F42", 0, "faulted")
    run_counts(tools, work, "counts.txt", "-p")
    run_counts(tools, work, "fragment_counts.txt", "-p", "--countReadPairs")
    se, _ = workspace(tools, tmp_path, "F42", 0, "clean")
    run_counts(tools, se, "counts.txt")
    return work, se


SUM = "awk 'NR>2{s+=$7} END{print s}' {f}"


@pytest.mark.parametrize(
    ("tool", "argv", "declared", "verdict"),
    [
        ("awk", ["NR>2{s+=$7} END{print s}", "counts.txt"], None, "fail"),
        ("awk", ["-F", "\t", "NR>2{print $1, $NF}", "counts.txt"], None, "fail"),
        (
            "python3",
            [
                "-c",
                "print(sum(int(l.split()[-1]) for l in open('counts.txt') if l[0] not in '#G'))",
            ],
            None,
            "fail",
        ),
        ("perl", ["-lane", "$s+=$F[-1]; END{print $s}", "counts.txt"], None, "fail"),
        (
            "Rscript",
            ["-e", "x <- read.table('counts.txt', header=TRUE); sum(x[[7]])"],
            None,
            "fail",
        ),
        ("bash", ["-c", "grep -v '^#' counts.txt | awk '{s+=$7} END{print s}'"], None, "fail"),
        ("python3", ["-c", "print(open('counts.txt').read())"], None, "fail"),
        ("awk", ["NR>2{s+=$7} END{print s}", "counts.txt"], {"count_unit": "fragment"}, "fail"),
        ("awk", ["NR>2{s+=$7} END{print s}", "counts.txt"], {"count_unit": "read"}, "pass"),
        ("awk", ["NR>2{s+=$7/2} END{print s}", "counts.txt"], None, "pass"),  # halves the reads
        ("awk", ["NR>2{print $1, int($7 / 2)}", "counts.txt"], None, "pass"),
        (
            "python3",
            ["-c", "# reads are mates; half is fragments\nprint(open('counts.txt').read())"],
            None,
            "pass",
        ),
        (
            "awk",
            ["NR>2{s+=$7} END{print s}", "fragment_counts.txt"],
            None,
            "pass",
        ),  # --countReadPairs
        ("head", ["-4", "counts.txt"], None, "pass"),
        ("cat", ["counts.txt"], None, "pass"),
        ("wc", ["-l", "counts.txt"], None, "pass"),
        ("less", ["counts.txt"], None, "pass"),
        ("cut", ["-f1,7", "counts.txt"], None, "pass"),
        ("awk", ["{print}", "counts.txt.summary"], None, "pass"),  # not a counts table name
    ],
)
def test_counts_unit_verdicts(tables, eng, tool, argv, declared, verdict):
    work, _ = tables
    _, result = outcome(eng, UNIT, tool, argv, work, declared)
    if verdict == "pass":
        assert result is None or result.verdict == "pass", result
    else:
        assert result is not None and result.verdict == "fail", result


def test_a_single_end_table_and_foreign_tables_cost_nothing(tables, eng):
    work, se = tables
    _, result = outcome(eng, UNIT, "awk", ["NR>2{s+=$7} END{print s}", "counts.txt"], se)
    assert result.verdict == "pass"
    (work / "other_counts.txt").write_text("gene\tn\ng1\t4\n")
    _, result = outcome(eng, UNIT, "awk", ["{s+=$2} END{print s}", "other_counts.txt"], work)
    assert result.verdict == "pass"
    (work / "empty_counts.txt").write_text("")
    _, result = outcome(eng, UNIT, "awk", ["{s+=$2}", "empty_counts.txt"], work)
    assert result.verdict == "pass"


def test_counts_unit_message_and_fix(tables, eng):
    work, _ = tables
    decision, result = outcome(eng, UNIT, "awk", ["NR>2{s+=$7} END{print s}", "counts.txt"], work)
    assert result.code == "counts_are_reads"
    assert "counts.txt" in result.message and "twice the number of fragments" in result.message
    assert result.fix[0].startswith("re-run featureCounts with -p --countReadPairs")
    assert render_block(decision, eng.retry_cap, eng.style).startswith("FIX: ")
    _, contradiction = outcome(
        eng, UNIT, "awk", ["{s+=$7}", "counts.txt"], work, {"count_unit": "fragment"}
    )
    assert contradiction.code == "declared_fragments_but_reads"


def test_the_fact_checker_decides_a_viewer_without_reading_the_file(tables, spec):
    work, _ = tables
    rule = next(r for r in spec.rules if r.id == UNIT)
    check = known_checkers()["featurecounts_counts_unit"]
    match = Match(rule, (work / "counts.txt",))
    viewer = Call("head", ("-4", "counts.txt"), work)
    assert (
        check(viewer, match, Ctx(work, "T0")).verdict == "pass"
    )  # would raise if it read the file
    with pytest.raises(Exception, match="may not read files"):
        check(Call("awk", ("{s+=$7}", "counts.txt"), work), match, Ctx(work, "T0"))


# ---- the spellings of the admission templates, run for real ----------------------------------


def test_every_template_spelling_gives_the_numbers_of_its_key(tools, tmp_path):
    for family, templates, key_args in (
        ("F42", PAIR_TEMPLATES, PAIR_KEYS),
        ("F43", STRAND_TEMPLATES, {"0": ("-s", "0"), "1": ("-s", "1"), "2": ("-s", "2")}),
    ):
        for twin in ("clean", "faulted"):
            work, _ = workspace(tools, tmp_path, family, 1, twin)
            expected = {k: counts_or_none(work, tools, *a) for k, a in key_args.items()}
            for t in templates:
                proc = tools.run([t.tool, *t.argv], work)
                got = None
                if proc.returncode == 0:
                    rows = [r.split("\t") for r in (work / "counts.txt").read_text().splitlines()]
                    got = {r[0]: int(r[-1]) for r in rows if not r[0].startswith(("#", "Geneid"))}
                assert got == expected[t.key], (family, twin, t.name)


# ---- admission -----------------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", [PAIRS, STRAND, UNIT])
def test_admission(spec, tools, tmp_path, rule_id):
    stats = measure(rule_id, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared = stats.cells[("right", True)]
    assert stats.admitted() and stats.killed() > 0
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.conditional == (rule_id != PAIRS)
    if stats.conditional:
        right_undeclared = stats.cells[("right", False)]
        assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert stats.cells[("wrong", False)].failed > 0 and stats.cells[("wrong", True)].failed > 0
    assert promotion_is_consistent(stats)


def test_module_constants_match_the_tool():
    assert qf.STRAND_OF == {"0": "unstranded", "1": "forward", "2": "reverse"}
    assert Path(qf.__file__).name == "quant_featurecounts.py"
