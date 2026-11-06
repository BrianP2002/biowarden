"""Rules of the variant pack for bcftools: unit tests, evidence against bcftools, admission.

bcftools.format_filter_scope: a FORMAT test in -i/-e is true when ANY sample satisfies it.
The unit tests write small VCFs; the family tests use F54 workspaces; the evidence tests run the
installed bcftools 1.24; the admission tests measure the rules over ADMISSION_SEEDS seeds with
calls labelled by running bcftools (bench/admission/cases_var_bcftools.py).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_var_bcftools import SCOPE_TEMPLATES
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.rules import var_bcftools  # noqa: F401  (registers the checkers)
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("var_bcftools")
RULES = {rule.id: rule for rule in SPEC.rules}


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def verdict(rule_id: str, argv, cwd, declared=None, tool="bcftools"):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def write_vcf(path: Path, samples, rows, extra_header="") -> Path:
    head = (
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="d">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="q">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="a">\n' + extra_header
    )
    cols = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
    path.write_text(head + cols + "".join(rows))
    return path


def row(pos, *cells, fmt="GT:GQ"):
    return f"chr1\t{pos}\tr{pos}\tA\tC\t50\tPASS\t.\t{fmt}\t" + "\t".join(cells) + "\n"


SAMPLES = ["s1", "s2", "s3"]
MIXED = [row(100, "0/1:50", "0/0:50", "1/1:50"), row(200, "0/1:50", "0/0:12", "1/1:50")]
UNIFORM = [row(100, "0/1:50", "0/0:50", "1/1:50"), row(200, "0/1:5", "0/0:12", "1/1:9")]


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_declares_the_rules_with_the_expected_shape():
    rule = RULES["bcftools.format_filter_scope"]
    assert (rule.kind, rule.tier, rule.slot) == ("call", "T1", "filter_semantics")
    assert rule.applies.tools == ["bcftools"]
    assert rule.applies.subcommands == ["view", "filter", "query"]
    assert rule.declare is not None and not rule.declare.required
    assert (rule.declare.key, rule.declare.values) == (
        "format_filter_scope",
        ["any_sample", "all_samples"],
    )
    assert rule.check in known_checkers()


# ---- bcftools.format_filter_scope --------------------------------------------------------------

SCOPE = "bcftools.format_filter_scope"
KEY = "format_filter_scope"


def scope(tmp_path, rows, argv, declared=None, samples=SAMPLES, name="c.vcf"):
    write_vcf(tmp_path / name, samples, rows)
    return verdict(SCOPE, [*argv, name], tmp_path, declared)


def test_include_on_mixed_data_fails_without_a_declaration_and_names_the_sites(tmp_path):
    decision, result = scope(tmp_path, MIXED, ["view", "-H", "-i", "FMT/GQ>=20"])
    assert decision.action == "block" and result.code == "scope_undeclared"
    text = render_block(decision)
    assert "ANY of the 3 samples" in text and "chr1:200" in text and "1 of 2 records" in text
    assert "-e 'FMT/GQ<20'" in text and "MIN(FMT/GQ)>=20" in text
    assert text.startswith("FIX:")


def test_include_with_the_matching_declaration_passes_and_the_other_one_fails(tmp_path):
    argv = ["view", "-H", "-i", "FMT/GQ>=20"]
    assert scope(tmp_path, MIXED, argv, {KEY: "any_sample"})[1].verdict == "pass"
    decision, result = scope(tmp_path, MIXED, argv, {KEY: "all_samples"})
    assert result.code == "declared_scope_mismatch" and decision.action == "block"
    text = render_block(decision)
    assert "you declared format_filter_scope=all_samples" in text
    assert "-e 'FMT/GQ<20'" in text


def test_exclude_means_all_samples_and_needs_no_declaration(tmp_path):
    argv = ["view", "-H", "-e", "FMT/GQ<20"]
    assert scope(tmp_path, MIXED, argv)[1].verdict == "pass"
    assert scope(tmp_path, MIXED, argv, {KEY: "all_samples"})[1].verdict == "pass"
    decision, result = scope(tmp_path, MIXED, argv, {KEY: "any_sample"})
    assert result.code == "declared_scope_mismatch"
    assert "-i 'FMT/GQ>=20'" in render_block(decision)


def test_uniform_data_is_silent_whatever_is_declared(tmp_path):
    for argv in (["view", "-H", "-i", "FMT/GQ>=20"], ["view", "-H", "-e", "FMT/GQ<20"]):
        for declared in (None, {KEY: "any_sample"}, {KEY: "all_samples"}):
            assert scope(tmp_path, UNIFORM, argv, declared)[1].verdict == "pass"


@pytest.mark.parametrize(
    "expression",
    [
        "MIN(FMT/GQ)>=20",
        "MAX(FMT/GQ)>=20",
        "F_PASS(FMT/GQ>=20)==1",
        "N_PASS(GQ>=20)==N_SAMPLES",
        "FMT/GQ[0]>=20",
        "AVG(FMT/GQ)>20",
        "QUAL>30",
        "INFO/DP>10",
        "POS>100",
    ],
)
def test_aggregates_indexes_and_non_format_tests_are_silent(tmp_path, expression):
    assert scope(tmp_path, MIXED, ["view", "-H", "-i", expression])[1].verdict == "pass"


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-i", "GQ>=20"],
        ["view", "-H", "-i", "FORMAT/GQ>=20"],
        ["view", "-H", "--include", "FMT/GQ>=20"],
        ["view", "-H", "--include=FMT/GQ>=20"],
        ["view", "-H", "-iFMT/GQ>=20"],
        ["filter", "-i", "GQ>=20"],
        ["query", "-f", "%POS\\n", "-i", "FMT/GQ>=20"],
        ["view", "-H", "-i", "FMT/GQ>=20 && QUAL>10"],
    ],
)
def test_spellings_of_an_unaggregated_include_are_seen(tmp_path, argv):
    assert scope(tmp_path, MIXED, argv)[1].code == "scope_undeclared"


def test_genotype_classes_and_string_genotypes(tmp_path):
    rows = [row(100, "0/1:50", "0/1:50", "0/1:50"), row(200, "0/1:50", "0/0:50", "1/1:50")]
    assert scope(tmp_path, rows, ["view", "-H", "-i", 'GT="het"'])[1].verdict == "fail"
    assert scope(tmp_path, rows, ["view", "-H", "-i", 'GT="alt"'])[1].verdict == "fail"
    assert scope(tmp_path, rows, ["view", "-H", "-i", 'GT="0/1"'])[1].verdict == "fail"
    uniform = [row(100, "0/1:50", "0/1:50", "0/1:50")]
    assert scope(tmp_path, uniform, ["view", "-H", "-i", 'GT="het"'])[1].verdict == "pass"


def test_single_sample_files_and_query_subsets_are_silent_but_view_subsets_are_not(tmp_path):
    rows = [row(100, "0/1:50"), row(200, "0/1:5")]
    argv = ["view", "-H", "-i", "FMT/GQ>=20"]
    assert scope(tmp_path, rows, argv, samples=["s1"])[1].verdict == "pass"
    # bcftools query subsets before it tests the expression: the rows of the subset decide
    q = ["query", "-f", "%POS\\n"]
    assert scope(tmp_path, MIXED, [*q, "-s", "s1", "-i", "FMT/GQ>=20"])[1].verdict == "pass"
    (tmp_path / "keep.txt").write_text("s1\ns3\n")
    argv = [*q, "-S", "keep.txt", "-i", "FMT/GQ>=20"]
    assert scope(tmp_path, MIXED, argv)[1].verdict == "pass"  # s1 and s3 agree on every row
    argv = [*q, "-s", "^s1", "-i", "FMT/GQ>=20"]
    assert scope(tmp_path, MIXED, argv)[1].verdict == "fail"  # s2 and s3 disagree at 200
    # bcftools view tests all samples of the record and subsets afterwards: -s does not narrow it
    for subset in (["-s", "s1"], ["-S", "keep.txt"], ["-s", "^s2"]):
        decision, result = scope(tmp_path, MIXED, ["view", "-H", *subset, "-i", "FMT/GQ>=20"])
        assert result.code == "scope_undeclared", subset


def test_filter_with_set_gts_and_unreadable_inputs_are_silent(tmp_path):
    write_vcf(tmp_path / "c.vcf", SAMPLES, MIXED)
    assert (
        verdict(SCOPE, ["filter", "-S", ".", "-i", "GQ>=20", "c.vcf"], tmp_path)[1].verdict
        == "pass"
    )
    assert verdict(SCOPE, ["view", "-i", "GQ>=20", "absent.vcf"], tmp_path)[1].verdict == "pass"
    assert verdict(SCOPE, ["view", "-i", "GQ>=20", "-"], tmp_path)[1].verdict == "pass"
    assert (
        verdict(SCOPE, ["view", "-H", "c.vcf"], tmp_path)[1] is None
    )  # no expression: not matched
    assert verdict(SCOPE, ["stats", "-i", "GQ>=20", "c.vcf"], tmp_path)[1] is None


def test_an_expression_it_cannot_evaluate_gives_unknown(tmp_path):
    rows = [row(100, "0/1:50", "0/0:50", "1/1:50", fmt="GT:GQ")]
    write_vcf(tmp_path / "c.vcf", SAMPLES, rows)
    argv = ["view", "-H", "-i", 'FMT/AD>5 || FMT/GQ="x"', "c.vcf"]
    decision, result = verdict(SCOPE, argv, tmp_path)
    assert result.verdict in ("unknown", "pass") and decision.action in ("warn", "allow")


def test_gzip_vcf_and_the_checker_tier(tmp_path):
    import gzip

    write_vcf(tmp_path / "c.vcf", SAMPLES, MIXED)
    with gzip.open(tmp_path / "c.vcf.gz", "wb") as handle:
        handle.write((tmp_path / "c.vcf").read_bytes())
    argv = ["view", "-H", "-i", "FMT/GQ>=20", "c.vcf.gz"]
    assert verdict(SCOPE, argv, tmp_path)[1].code == "scope_undeclared"
    call = Call("bcftools", tuple(argv), tmp_path)
    check = known_checkers()["bcftools_format_filter_scope"]
    with pytest.raises(TierViolation):
        check(call, Match(RULES[SCOPE]), Ctx(tmp_path, "T0"))


def test_both_include_and_exclude_is_unknown(tmp_path):
    argv = ["view", "-H", "-i", "FMT/GQ>=20", "-e", "QUAL<10"]
    decision, result = scope(tmp_path, MIXED, argv)
    assert result.verdict == "unknown" and decision.action == "warn"
    assert "scope not checked" in render_warning(decision)


# ---- evidence against bcftools 1.24 and the F54 workspaces -------------------------------------


def test_bcftools_tests_a_format_field_as_any_sample(tmp_path, tools):
    task = build_task("F54", 3, "faulted", tmp_path, tools, "implicit")

    def count(*options):
        proc = tools.run(["bcftools", "view", "-H", *options, "cohort.vcf.gz"], tmp_path)
        assert proc.returncode == 0 and proc.stderr == ""
        return len(proc.stdout.splitlines())

    assert count("-i", "FMT/GQ>=20") == task.truth["naive"]
    assert count("-i", "GQ>=20") == count("-i", "FORMAT/GQ>=20") == task.truth["naive"]
    assert count("-e", "FMT/GQ<20") == count("-i", "MIN(FMT/GQ)>=20") == task.truth["expected"]
    assert count("-i", "F_PASS(FMT/GQ>=20)==1") == task.truth["expected"]
    assert count("-i", "N_PASS(FMT/GQ>=20)==N_SAMPLES") == task.truth["expected"]


def test_the_rule_fails_the_faulted_workspace_and_passes_the_clean_one(tmp_path, tools):
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F54", 6, twin, work, tools, "implicit")
        argv = ["view", "-H", "-i", "FMT/GQ>=20", "cohort.vcf.gz"]
        assert verdict(SCOPE, argv, work)[1].verdict == expected
        argv = ["view", "-H", "-e", "FMT/GQ<20", "cohort.vcf.gz"]
        assert verdict(SCOPE, argv, work)[1].verdict == "pass"


def test_every_template_is_decided_as_its_options_imply(tmp_path, tools):
    work = tmp_path / "f"
    build_task("F54", 7, "faulted", work, tools, "implicit")
    for template in SCOPE_TEMPLATES:
        result = verdict(SCOPE, list(template.argv), work, {KEY: "all_samples"})[1]
        failed = result is not None and result.verdict == "fail"
        # every -i call that tests the field for any sample contradicts all_samples, except the
        # query call that keeps a single sample before it tests (nothing to disagree about)
        expected = template.meaning == "any_sample" and template.name != "query_one_sample"
        assert failed == expected, template.name


# ---- admission ---------------------------------------------------------------------------------


def test_admission_format_filter_scope(tools, tmp_path):
    stats = measure(SCOPE, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 8
    assert stats.cells[("wrong", True)].failed > 0
    assert promotion_is_consistent(stats)


# ---- bcftools.filter_pass_dot ------------------------------------------------------------------

DOT = "bcftools.filter_pass_dot"
DOT_KEY = "unfiltered_records"


def filter_rows(values):
    return [
        f"chr1\t{100 + 10 * i}\tf{i}\tA\tC\t50\t{value}\t.\tGT:GQ\t0/1:50\t0/0:50\n"
        for i, value in enumerate(values)
    ]


def dot_call(tmp_path, values, argv, declared=None, name="d.vcf"):
    write_vcf(
        tmp_path / name,
        ["s1", "s2"],
        filter_rows(values),
        '##FILTER=<ID=LowQual,Description="l">\n',
    )
    return verdict(DOT, [*argv, name], tmp_path, declared)


WITH_DOT = ["PASS", ".", "LowQual", "PASS"]


def test_apply_filters_pass_fails_when_records_have_a_dot_filter(tmp_path):
    decision, result = dot_call(tmp_path, WITH_DOT, ["view", "-H", "-f", "PASS"])
    assert decision.action == "block" and result.code == "dot_filter_dropped"
    text = render_block(decision)
    assert text.startswith("FIX: use -f .,PASS") and "1 of 4 records" in text
    assert "f1" in text and "no filter was applied" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "--apply-filters", "PASS"],
        ["view", "-H", "-fPASS"],
        ["view", "-H", "-f", "PASS", "-c1"],
        ["isec", "-n=1", "-f", "PASS"],
        ["merge", "-f", "PASS"],
        ["view", "-H", "-i", 'FILTER="PASS"'],
        ["view", "-H", "-i", 'FILTER=="PASS"'],
        ["view", "-H", "-e", 'FILTER!="PASS"'],
        ["filter", "-e", 'FILTER!="PASS"'],
    ],
)
def test_every_pass_only_selection_is_seen(tmp_path, argv):
    assert dot_call(tmp_path, WITH_DOT, argv)[1].code == "dot_filter_dropped"


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-f", ".,PASS"],
        ["view", "-H", "-f", "PASS,."],
        ["view", "-H", "--apply-filters", ".,PASS"],
        ["view", "-H", "-i", 'FILTER="PASS" || FILTER="."'],
        ["view", "-H", "-e", 'FILTER!="PASS" && FILTER!="."'],
        ["view", "-H", "-e", 'FILTER="LowQual"'],
        ["view", "-H", "-i", 'FILTER!="PASS"'],
        ["view", "-H", "-f", "LowQual"],
        ["view", "-H"],
        ["query", "-f", "%ID\\n"],
    ],
)
def test_selections_that_keep_dot_or_do_not_select_pass_are_silent(tmp_path, argv):
    result = dot_call(tmp_path, WITH_DOT, argv)[1]
    assert result is None or result.verdict == "pass"


def test_without_dot_records_everything_passes(tmp_path):
    for declared in (None, {DOT_KEY: "keep"}, {DOT_KEY: "drop"}):
        values = ["PASS", "LowQual", "PASS"]
        assert (
            dot_call(tmp_path, values, ["view", "-H", "-f", "PASS"], declared)[1].verdict == "pass"
        )


def test_declarations_are_checked_against_the_call(tmp_path):
    drop = ["view", "-H", "-f", "PASS"]
    assert dot_call(tmp_path, WITH_DOT, drop, {DOT_KEY: "drop"})[1].verdict == "pass"
    decision, result = dot_call(tmp_path, WITH_DOT, drop, {DOT_KEY: "keep"})
    assert result.code == "declared_keep_but_dropped" and "you declared" in render_block(decision)
    keep = ["view", "-H", "-f", ".,PASS"]
    assert dot_call(tmp_path, WITH_DOT, keep, {DOT_KEY: "keep"})[1].verdict == "pass"
    assert dot_call(tmp_path, WITH_DOT, keep, {DOT_KEY: "drop"})[1].code == "declared_drop_but_kept"


def test_filter_dot_fix_matches_the_form_of_the_call(tmp_path):
    for argv, needle in (
        (["view", "-H", "-i", 'FILTER="PASS"'], '-i \'FILTER="PASS" || FILTER="."\''),
        (["view", "-H", "-e", 'FILTER!="PASS"'], '-e \'FILTER!="PASS" && FILTER!="."\''),
        (["view", "-H", "-f", "PASS"], "-f .,PASS"),
    ):
        assert needle in render_block(dot_call(tmp_path, WITH_DOT, argv)[0])


def test_query_f_is_a_format_not_a_filter_list(tmp_path):
    result = dot_call(tmp_path, WITH_DOT, ["query", "-f", "PASS"])[1]
    assert result is None or result.verdict == "pass"


def test_the_filter_dot_rule_on_the_f55_workspaces(tmp_path, tools):
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F55", 4, twin, work, tools, "implicit")
        argv = ["view", "-H", "-f", "PASS", "variants.vcf.gz"]
        assert verdict(DOT, argv, work)[1].verdict == expected
        argv = ["view", "-H", "-f", ".,PASS", "variants.vcf.gz"]
        assert verdict(DOT, argv, work)[1].verdict == "pass"


def test_admission_filter_pass_dot(tools, tmp_path):
    stats = measure(DOT, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 9
    assert stats.cells[("wrong", True)].failed > 0
    assert promotion_is_consistent(stats)


# ---- bcftools.missing_value_filter -------------------------------------------------------------

MISS = "bcftools.missing_value_filter"
MISS_KEY = "missing_annotation"


def qd_rows(values, quals=None):
    rows = []
    for i, qd in enumerate(values):
        info = "." if qd is None else f"QD={qd}"
        qual = "50" if quals is None else quals[i]
        rows.append(
            f"chr1\t{100 + 10 * i}\tq{i}\tA\tC\t{qual}\tPASS\t{info}\tGT:GQ\t0/1:50\t0/0:50\n"
        )
    return rows


def miss_call(tmp_path, values, argv, declared=None, quals=None):
    header = '##INFO=<ID=QD,Number=1,Type=Float,Description="q">\n'
    write_vcf(tmp_path / "m.vcf", ["s1", "s2"], qd_rows(values, quals), header)
    return verdict(MISS, [*argv, "m.vcf"], tmp_path, declared)


QD_PARTIAL = [5.0, None, 1.0, 9.0]


def test_exclude_on_a_partly_missing_tag_fails_and_the_fix_is_include(tmp_path):
    decision, result = miss_call(tmp_path, QD_PARTIAL, ["view", "-H", "-e", "INFO/QD<2"])
    assert decision.action == "block" and result.code == "missing_values_kept"
    text = render_block(decision)
    assert text.startswith("FIX: use -i 'INFO/QD>=2'") and "1 of 4 records" in text
    assert "false for a missing value" in text and "q1" in text


def test_include_is_silent_undeclared_and_checked_when_declared(tmp_path):
    argv = ["view", "-H", "-i", "INFO/QD>=2"]
    assert miss_call(tmp_path, QD_PARTIAL, argv)[1].verdict == "pass"
    assert miss_call(tmp_path, QD_PARTIAL, argv, {MISS_KEY: "record_removed"})[1].verdict == "pass"
    decision, result = miss_call(tmp_path, QD_PARTIAL, argv, {MISS_KEY: "record_kept"})
    assert result.code == "declared_missing_mismatch" and decision.action == "block"
    assert "-e 'INFO/QD<2'" in render_block(decision)


def test_exclude_declared_record_kept_passes_and_record_removed_fails(tmp_path):
    argv = ["view", "-H", "-e", "INFO/QD<2"]
    assert miss_call(tmp_path, QD_PARTIAL, argv, {MISS_KEY: "record_kept"})[1].verdict == "pass"
    result = miss_call(tmp_path, QD_PARTIAL, argv, {MISS_KEY: "record_removed"})[1]
    assert result.code == "declared_missing_mismatch"


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-e", "QD<2"],
        ["view", "-H", "--exclude", "INFO/QD<=1.5"],
        ["view", "-H", "--exclude=INFO/QD<2"],
        ["filter", "-e", "INFO/QD<2"],
        ["filter", "-e", "INFO/QD<2", "-s", "LowQD"],
        ["query", "-f", "%ID\\n", "-e", "INFO/QD<2"],
    ],
)
def test_spellings_of_the_exclusion_are_seen(tmp_path, argv):
    assert miss_call(tmp_path, QD_PARTIAL, argv)[1].code == "missing_values_kept"


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-e", 'INFO/QD<2 || INFO/QD="."'],
        ["view", "-H", "-e", "INFO/QD<2 && POS>0"],
        ["view", "-H", "-e", "QUAL<10"],
        ["view", "-H", "-e", "INFO/DP<2"],
        ["view", "-H", "-e", "FMT/GQ<20"],
    ],
)
def test_other_expressions_are_not_decided(tmp_path, argv):
    result = miss_call(tmp_path, QD_PARTIAL, argv)[1]
    assert result is None or result.verdict == "pass"


def test_nothing_missing_or_everything_missing_is_silent(tmp_path):
    argv = ["view", "-H", "-e", "INFO/QD<2"]
    assert miss_call(tmp_path, [5.0, 1.0, 9.0], argv)[1].verdict == "pass"
    assert miss_call(tmp_path, [None, None, None], argv)[1].verdict == "pass"


def test_qual_missing_values(tmp_path):
    quals = ["50", ".", "10", "50"]
    argv = ["view", "-H", "-e", "QUAL<30"]
    result = miss_call(tmp_path, [5.0, 5.0, 5.0, 5.0], argv, quals=quals)[1]
    assert result.code == "missing_values_kept" and "no QUAL value" in result.message
    argv = ["view", "-H", "-i", "QUAL>=30"]
    assert miss_call(tmp_path, [5.0] * 4, argv, quals=quals)[1].verdict == "pass"
    assert miss_call(tmp_path, [5.0] * 4, argv, {MISS_KEY: "record_kept"}, quals)[1].verdict == (
        "fail"
    )


def test_bcftools_missing_value_semantics(tmp_path, tools):
    task = build_task("F56", 3, "faulted", tmp_path, tools, "implicit")

    def ids(*options):
        proc = tools.run(["bcftools", "view", "-H", *options, "variants.vcf.gz"], tmp_path)
        assert proc.returncode == 0 and proc.stderr == ""
        return sorted(line.split("\t")[2] for line in proc.stdout.splitlines())

    assert ids("-e", "INFO/QD<2") == task.truth["naive"]
    assert ids("-i", "INFO/QD>=2") == task.truth["expected"]
    assert ids("-e", 'INFO/QD<2 || INFO/QD="."') == task.truth["expected"]


def test_the_missing_value_rule_on_the_f56_workspaces(tmp_path, tools):
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F56", 4, twin, work, tools, "implicit")
        argv = ["view", "-H", "-e", "INFO/QD<2", "variants.vcf.gz"]
        assert verdict(MISS, argv, work)[1].verdict == expected
        argv = ["view", "-H", "-i", "INFO/QD>=2", "variants.vcf.gz"]
        assert verdict(MISS, argv, work)[1].verdict == "pass"


def test_admission_missing_value_filter(tools, tmp_path):
    stats = measure(MISS, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 7
    assert promotion_is_consistent(stats)


# ---- bcftools.subset_monomorphic ---------------------------------------------------------------

SUBSET = "bcftools.subset_monomorphic"
SUBSET_KEY = "subset_sites"
SUB_ROWS = [
    "chr1\t100\tm0\tA\tC\t50\tPASS\tAC=3;AN=8\tGT\t0/1\t0/0\t1/1\t0/0\n",
    "chr1\t200\tm1\tA\tC\t50\tPASS\tAC=1;AN=8\tGT\t0/0\t0/0\t0/1\t0/0\n",
    "chr1\t300\tm2\tA\tC\t50\tPASS\tAC=1;AN=8\tGT\t0/1\t0/0\t0/0\t0/0\n",
]


def sub_call(tmp_path, argv, declared=None, rows=SUB_ROWS):
    header = (
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="a">\n'
        '##INFO=<ID=AN,Number=1,Type=Integer,Description="n">\n'
    )
    write_vcf(tmp_path / "s.vcf", ["s1", "s2", "s3", "s4"], rows, header)
    (tmp_path / "keep.txt").write_text("s1\ns2\n")
    return verdict(SUBSET, [*argv, "s.vcf"], tmp_path, declared)


def test_subset_without_a_filter_fails_when_a_site_is_monomorphic_in_the_subset(tmp_path):
    decision, result = sub_call(tmp_path, ["view", "-H", "-S", "keep.txt"])
    assert decision.action == "block" and result.code == "monomorphic_sites_kept"
    text = render_block(decision)
    assert text.startswith("FIX: add -c 1") and "1 of 3 sites" in text and "m1" in text
    assert "--declare subset_sites=keep_all_sites" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-s", "s1,s2"],
        ["view", "-H", "--samples", "s1,s2"],
        ["view", "-H", "--samples-file", "keep.txt"],
        ["view", "-H", "-S", "keep.txt", "-v", "snps"],
        ["view", "-H", "-s", "^s3,s4"],
        ["view", "-H", "--force-samples", "-s", "s1,s2,zz"],
    ],
)
def test_every_spelling_of_the_subset_is_seen(tmp_path, argv):
    assert sub_call(tmp_path, argv)[1].code == "monomorphic_sites_kept"


@pytest.mark.parametrize(
    "argv",
    [
        ["view", "-H", "-S", "keep.txt", "-c", "1"],
        ["view", "-H", "-S", "keep.txt", "-c1"],
        ["view", "-H", "-S", "keep.txt", "--min-ac", "1:minor"],
        ["view", "-H"],
        ["view", "-H", "-s", "s1,s2,s3,s4"],
    ],
)
def test_filters_and_no_subset_are_silent(tmp_path, argv):
    result = sub_call(tmp_path, argv)[1]
    assert result is not None and result.verdict == "pass" or result is None


def test_min_ac_zero_and_expressions_are_not_filters_after_the_subset(tmp_path):
    # -i/-e run on the full cohort before the samples are selected: AC==0 does not drop site m1
    for extra in (["-c", "0"], ["-e", "AC==0"], ["-i", "AC>0"], ["-i", 'GT="alt"']):
        argv = ["view", "-H", "-S", "keep.txt", *extra]
        assert sub_call(tmp_path, argv)[1].code == "monomorphic_sites_kept", extra


def test_subset_declarations(tmp_path):
    argv = ["view", "-H", "-S", "keep.txt"]
    for declared in ("keep_all_sites", "filtered_downstream"):
        assert sub_call(tmp_path, argv, {SUBSET_KEY: declared})[1].verdict == "pass"
    decision, result = sub_call(tmp_path, argv, {SUBSET_KEY: "variant_sites_only"})
    assert result.code == "declared_variant_sites_only" and decision.action == "block"
    assert "you declared subset_sites=variant_sites_only" in render_block(decision)
    assert sub_call(tmp_path, [*argv, "-c", "1"], {SUBSET_KEY: "variant_sites_only"})[
        1
    ].verdict == ("pass")


def test_subset_with_every_site_variable_is_silent(tmp_path):
    rows = [SUB_ROWS[0], "chr1\t400\tm3\tA\tC\t50\tPASS\tAC=2;AN=8\tGT\t0/1\t0/0\t0/1\t0/0\n"]
    for declared in (None, {SUBSET_KEY: "variant_sites_only"}, {SUBSET_KEY: "keep_all_sites"}):
        assert sub_call(tmp_path, ["view", "-H", "-S", "keep.txt"], declared, rows)[1].verdict == (
            "pass"
        )


def test_bcftools_view_subset_keeps_the_monomorphic_sites_and_a_stale_af(tmp_path, tools):
    task = build_task("F57", 3, "faulted", tmp_path, tools, "implicit")

    def count(*options):
        argv = ["bcftools", "view", "-H", "-S", "group_b.txt", *options, "cohort.vcf.gz"]
        proc = tools.run(argv, tmp_path)
        assert proc.returncode == 0 and proc.stderr == ""
        return len(proc.stdout.splitlines())

    assert count() == task.truth["naive"] > task.truth["expected"]
    assert count("-c", "1") == task.truth["expected"]
    # the test runs on the full cohort before the subset: nothing is dropped
    assert count("-e", "AC==0") == count("-i", 'GT="alt"') == task.truth["naive"]
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F57", 5, twin, work, tools, "implicit")
        argv = ["view", "-H", "-S", "group_b.txt", "cohort.vcf.gz"]
        assert verdict(SUBSET, argv, work)[1].verdict == expected


def test_admission_subset_monomorphic(tools, tmp_path):
    stats = measure(SUBSET, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 8
    assert promotion_is_consistent(stats)


# ---- bcftools.isec_representation --------------------------------------------------------------

ISEC = "bcftools.isec_representation"
ISEC_KEY = "isec_matching"
REF = "ACGTACGTTTTTTGACGTACGTACGTAAAAAACCCGGGATATATCGCGCGGGGATCTAGCTAGCATCGATCGAT"


def isec_workspace(tmp_path, tools, left_b="chr1\t7\t.\tGT\tG", right_b="chr1\t12\t.\tTT\tT"):
    (tmp_path / "ref.fa").write_text(f">chr1\n{REF}\n")
    assert tools.run(["samtools", "faidx", "ref.fa"], tmp_path).returncode == 0
    head = "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=74>\n"
    cols = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    snv = "chr1\t20\t.\tC\tA\t50\tPASS\t.\n"
    for name, line in (("a", left_b), ("b", right_b), ("c", left_b)):
        (tmp_path / f"{name}.vcf").write_text(head + cols + f"{line}\t50\tPASS\t.\n" + snv)
        for argv in (
            ["bgzip", "-f", f"{name}.vcf"],
            ["bcftools", "index", "-f", "-t", f"{name}.vcf.gz"],
        ):
            assert tools.run(argv, tmp_path).returncode == 0
    return tmp_path


def isec_verdict(tools, argv, cwd, declared=None):
    eng = Engine(
        [SPEC],
        mode="block",
        retry_cap=10**9,
        real_bins=tools.real_bins(["bcftools"]),
        real_env=tools.process_env(),
    )
    decision = eng.evaluate(Call("bcftools", tuple(argv), Path(cwd), declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == ISEC]
    return decision, (own[0].result if own else None)


def test_isec_of_a_left_and_a_right_aligned_deletion_finds_only_the_snv(tmp_path, tools):
    work = isec_workspace(tmp_path, tools)

    def shared(a, b, *options):
        proc = tools.run(["bcftools", "isec", "-n=2", "-w1", *options, a, b], work)
        assert proc.returncode == 0
        return [line.split("\t")[1] for line in proc.stdout.splitlines() if line[0] != "#"]

    for c in ("none", "some", "indels", "all"):
        assert shared("a.vcf.gz", "b.vcf.gz", "-c", c) == ["20"]
    assert shared("a.vcf.gz", "c.vcf.gz") == ["7", "20"]
    for name in ("a", "b"):
        argv = [
            "bcftools",
            "norm",
            "-f",
            "ref.fa",
            "-Oz",
            "-o",
            f"{name}.n.vcf.gz",
            f"{name}.vcf.gz",
        ]
        assert tools.run(argv, work).returncode == 0
        assert (
            tools.run(["bcftools", "index", "-f", "-t", f"{name}.n.vcf.gz"], work).returncode == 0
        )
    assert shared("a.n.vcf.gz", "b.n.vcf.gz") == ["7", "20"]


def test_isec_rule_fails_on_unnormalised_inputs_and_names_the_fix(tmp_path, tools):
    work = isec_workspace(tmp_path, tools)
    decision, result = isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "b.vcf.gz"], work)
    assert decision.action == "block" and result.code == "isec_unnormalised_inputs"
    text = render_block(decision)
    assert text.startswith("FIX: normalise every input first: bcftools norm -f ref.fa")
    assert "would realign 1 of 2 records of b.vcf.gz" in text
    assert isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "c.vcf.gz"], work)[1].verdict == "pass"


def test_isec_rule_declarations_and_normalised_inputs(tmp_path, tools):
    work = isec_workspace(tmp_path, tools)
    argv = ["isec", "-n=2", "a.vcf.gz", "b.vcf.gz"]
    assert isec_verdict(tools, argv, work, {ISEC_KEY: "as_written"})[1].verdict == "pass"
    result = isec_verdict(tools, argv, work, {ISEC_KEY: "normalized"})[1]
    assert result.code == "declared_normalized_but_not"
    for name in ("a", "b"):
        norm = [
            "bcftools",
            "norm",
            "-f",
            "ref.fa",
            "-Oz",
            "-o",
            f"{name}.n.vcf.gz",
            f"{name}.vcf.gz",
        ]
        assert tools.run(norm, work).returncode == 0
        assert (
            tools.run(["bcftools", "index", "-f", "-t", f"{name}.n.vcf.gz"], work).returncode == 0
        )
    argv = ["isec", "-n=2", "a.n.vcf.gz", "b.n.vcf.gz"]
    assert isec_verdict(tools, argv, work)[1].verdict == "pass"
    assert isec_verdict(tools, argv, work, {ISEC_KEY: "normalized"})[1].verdict == "pass"


def test_isec_without_a_reference_is_unknown_and_other_subcommands_are_not_matched(tmp_path, tools):
    work = isec_workspace(tmp_path, tools)
    (work / "ref.fa").unlink()
    decision, result = isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "b.vcf.gz"], work)
    assert result.verdict == "unknown" and decision.action == "warn"
    assert isec_verdict(tools, ["view", "a.vcf.gz"], work)[1] is None
    assert isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz"], work)[1].verdict == "pass"


def tree(root: Path) -> dict[str, tuple[int, str]]:
    """Every file under `root`: relative name -> (size, sha256), plus an entry per directory."""
    out = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        out[name] = (0, "dir") if path.is_dir() else (path.stat().st_size, sha256_of(path))
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_isec_dry_run_does_not_leave_an_index_in_the_workspace(tmp_path, tools):
    # E-8: `bcftools norm -f ref.fa` writes ref.fa.fai beside a FASTA that has no index
    work = isec_workspace(tmp_path, tools)
    (work / "ref.fa.fai").unlink()
    before = tree(work)
    for declared in (None, {ISEC_KEY: "normalized"}, {ISEC_KEY: "as_written"}):
        result = isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "b.vcf.gz"], work, declared)[1]
        assert result.verdict in ("fail", "pass")
    clean = isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "c.vcf.gz"], work)[1]
    assert clean.verdict == "pass"  # the dry run still reaches the realigned count
    assert tree(work) == before and not (work / "ref.fa.fai").exists()


def test_the_isec_dry_run_reads_an_existing_index_and_changes_nothing(tmp_path, tools):
    work = isec_workspace(tmp_path, tools)  # samtools faidx built ref.fa.fai
    before = tree(work)
    result = isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "b.vcf.gz"], work)[1]
    assert result.code == "isec_unnormalised_inputs" and "would realign 1 of 2" in result.message
    assert tree(work) == before


def test_a_dry_run_that_times_out_is_unknown_not_a_crash(tmp_path, tools, monkeypatch):
    work = isec_workspace(tmp_path, tools)

    def too_slow(self, tool, argv):
        raise subprocess.TimeoutExpired([tool, *argv], 0.1)

    monkeypatch.setattr(Ctx, "run_real", too_slow)
    decision, result = isec_verdict(tools, ["isec", "-n=2", "a.vcf.gz", "b.vcf.gz"], work)
    assert (result.verdict, result.code) == ("unknown", "normalisation_not_checked")
    assert decision.action == "warn"


def test_the_isec_rule_on_the_f62_workspaces(tmp_path, tools):
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F62", 4, twin, work, tools, "implicit")
        argv = ["isec", "-n=2", "caller_a.vcf.gz", "caller_b.vcf.gz"]
        assert isec_verdict(tools, argv, work)[1].verdict == expected


def test_admission_isec_representation(tools, tmp_path):
    stats = measure(ISEC, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 6
    assert promotion_is_consistent(stats)


# ---- bcftools.merge_absent_sites ---------------------------------------------------------------

MERGE = "bcftools.merge_absent_sites"
ABSENT_KEY = "absent_sites"


def merge_workspace(tmp_path, tools, listed=((100, 200), (200,), (200, 300)), end=False):
    head = "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
    head += '##INFO=<ID=END,Number=1,Type=Integer,Description="e">\n' if end else ""
    head += '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
    for i, positions in enumerate(listed, start=1):
        rows = "".join(
            f"chr1\t{p}\tm{p}\tA\tC\t50\tPASS\t{'END=' + str(p + 5) if end else '.'}\tGT\t0/1\n"
            for p in positions
        )
        cols = f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsmp{i}\n"
        (tmp_path / f"s{i}.vcf").write_text(head + cols + rows)
        for argv in (
            ["bgzip", "-f", f"s{i}.vcf"],
            ["bcftools", "index", "-f", "-t", f"s{i}.vcf.gz"],
        ):
            assert tools.run(argv, tmp_path).returncode == 0
    return [f"s{i}.vcf.gz" for i in range(1, len(listed) + 1)]


def merge_verdict(argv, cwd, declared=None):
    return verdict(MERGE, argv, cwd, declared)


def test_merge_writes_missing_for_unlisted_sites_and_minus_zero_writes_ref(tmp_path, tools):
    files = merge_workspace(tmp_path, tools)

    def genotypes(*options):
        proc = tools.run(["bcftools", "merge", *options, *files], tmp_path)
        assert proc.returncode == 0
        return [line.split("\t")[9:] for line in proc.stdout.splitlines() if line[0] != "#"]

    assert genotypes() == [["0/1", "./.", "./."], ["0/1", "0/1", "0/1"], ["./.", "./.", "0/1"]]
    assert genotypes("-0")[0] == ["0/1", "0/0", "0/0"]


def test_merge_of_variants_only_files_fails_and_the_fix_is_minus_zero(tmp_path, tools):
    files = merge_workspace(tmp_path, tools)
    decision, result = merge_verdict(["merge", *files], tmp_path)
    assert decision.action == "block" and result.code == "absent_sites_missing"
    text = render_block(decision)
    assert text.startswith("FIX: add -0 (--missing-to-ref)") and "3 of the 3 inputs" in text
    assert merge_verdict(["merge", "-0", *files], tmp_path)[1].verdict == "pass"
    assert merge_verdict(["merge", "--missing-to-ref", *files], tmp_path)[1].verdict == "pass"
    (tmp_path / "list.txt").write_text("".join(f"{f}\n" for f in files))
    assert merge_verdict(["merge", "-l", "list.txt"], tmp_path)[1].verdict == "fail"
    assert merge_verdict(["merge", "-0", "-l", "list.txt"], tmp_path)[1].verdict == "pass"


def test_merge_declarations(tmp_path, tools):
    files = merge_workspace(tmp_path, tools)
    assert merge_verdict(["merge", *files], tmp_path, {ABSENT_KEY: "missing"})[1].verdict == "pass"
    result = merge_verdict(["merge", *files], tmp_path, {ABSENT_KEY: "reference"})[1]
    assert result.code == "declared_absent_mismatch"
    assert merge_verdict(["merge", "-0", *files], tmp_path, {ABSENT_KEY: "reference"})[
        1
    ].verdict == ("pass")
    result = merge_verdict(["merge", "-0", *files], tmp_path, {ABSENT_KEY: "missing"})[1]
    assert result.code == "declared_absent_mismatch"


def test_merge_is_silent_when_every_input_lists_every_site_or_is_a_gvcf(tmp_path, tools):
    same = merge_workspace(tmp_path, tools, listed=((100, 200), (100, 200)))
    assert merge_verdict(["merge", *same], tmp_path)[1].verdict == "pass"
    (tmp_path / "g").mkdir()
    gvcf = merge_workspace(tmp_path / "g", tools, listed=((100, 200), (200,)), end=True)
    assert merge_verdict(["merge", *gvcf], tmp_path / "g")[1].verdict == "pass"
    assert merge_verdict(["merge", same[0]], tmp_path)[1].verdict == "pass"
    assert merge_verdict(["view", *same], tmp_path)[1] is None


def test_the_merge_rule_on_the_f63_workspaces(tmp_path, tools):
    files = [f"sample{i}.vcf.gz" for i in range(1, 7)]
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F63", 4, twin, work, tools, "implicit")
        assert merge_verdict(["merge", *files], work)[1].verdict == expected
        assert merge_verdict(["merge", "-0", *files], work)[1].verdict == "pass"


def test_admission_merge_absent_sites(tools, tmp_path):
    stats = measure(MERGE, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 4
    assert promotion_is_consistent(stats)
