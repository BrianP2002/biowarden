"""Rules of the variant pack for vcftools: unit tests, evidence against vcftools, admission.

vcftools.bed_header: the first line of a BED file given to --bed / --exclude-bed is skipped.
The unit tests write small files; the family tests use F53 workspaces; the evidence test runs the
installed vcftools 0.1.17; the admission test measures the rule over ADMISSION_SEEDS seeds with
calls labelled by running vcftools (bench/admission/cases_var_vcftools.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import var_vcftools  # noqa: F401  (registers the checkers)
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("var_vcftools")
RULES = {rule.id: rule for rule in SPEC.rules}


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "vcftools" not in env.real_bins(["vcftools"]):
        pytest.skip("samtools, bcftools and vcftools are not available")
    return env


def verdict(rule_id: str, tool: str, argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_declares_the_rules_with_the_expected_shape():
    rule = RULES["vcftools.bed_header"]
    assert (rule.kind, rule.tier, rule.slot) == ("call", "T1", "input_format")
    assert rule.applies.tools == ["vcftools"]
    assert rule.applies.options_any == ["--bed", "--exclude-bed"]
    assert rule.declare is None
    assert rule.check in known_checkers()


# ---- vcftools.bed_header -----------------------------------------------------------------------

HEADERLESS = "chr1\t99\t120\nchr2\t49\t70\n"


def bed_call(tmp_path, text, option="--bed", name="t.bed"):
    (tmp_path / name).write_text(text)
    argv = ["--gzvcf", "v.vcf.gz", option, name, "--recode", "--stdout"]
    return verdict("vcftools.bed_header", "vcftools", argv, tmp_path)


def test_a_headerless_bed_fails_with_the_lost_interval_and_a_fix(tmp_path):
    decision, result = bed_call(tmp_path, HEADERLESS)
    assert decision.action == "block" and result.code == "bed_first_line_skipped"
    text = render_block(decision)
    assert text.startswith("FIX:") and "t.bed" in text and "bcftools view -R" in text
    assert "chr1 99 120" in text and "silently ignored" in text


@pytest.mark.parametrize(
    "text",
    [
        "#chrom\tstart\tend\n" + HEADERLESS,
        "track name=targets\n" + HEADERLESS,
        "chrom\tstart\tend\n" + HEADERLESS,
        "browser position chr1:1-100\n" + HEADERLESS,
        "# a comment\n" + HEADERLESS,
        "",
    ],
)
def test_a_first_line_that_is_not_an_interval_passes(tmp_path, text):
    decision, result = bed_call(tmp_path, text)
    assert decision.action == "allow" and result.verdict == "pass"


@pytest.mark.parametrize("text", ["chr1 99 120\nchr2 49 70\n", "chr1\t99\t120\tname\t0\t+\n"])
def test_space_separated_and_bed6_first_lines_are_intervals(tmp_path, text):
    assert bed_call(tmp_path, text)[1].code == "bed_first_line_skipped"


@pytest.mark.parametrize("option", ["--bed", "--exclude-bed"])
def test_both_bed_options_skip_the_first_line(tmp_path, option):
    assert bed_call(tmp_path, HEADERLESS, option)[1].verdict == "fail"


def test_the_equals_spelling_is_seen(tmp_path):
    (tmp_path / "t.bed").write_text(HEADERLESS)
    argv = ["--vcf", "v.vcf", "--bed=t.bed", "--recode"]
    assert verdict("vcftools.bed_header", "vcftools", argv, tmp_path)[1].verdict == "fail"


def test_a_missing_file_is_unknown_and_other_vcftools_calls_are_not_matched(tmp_path):
    argv = ["--vcf", "v.vcf", "--bed", "absent.bed", "--recode"]
    decision, result = verdict("vcftools.bed_header", "vcftools", argv, tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"
    for argv in (
        ["--vcf", "v.vcf", "--positions", "pos.txt", "--recode"],
        ["--vcf", "v.vcf", "--freq", "--stdout"],
        ["--version"],
    ):
        assert verdict("vcftools.bed_header", "vcftools", argv, tmp_path)[1] is None


def test_two_bed_files_one_of_them_headerless(tmp_path):
    (tmp_path / "a.bed").write_text("#h\n" + HEADERLESS)
    (tmp_path / "b.bed").write_text(HEADERLESS)
    argv = ["--vcf", "v.vcf", "--bed", "a.bed", "--exclude-bed", "b.bed"]
    result = verdict("vcftools.bed_header", "vcftools", argv, tmp_path)[1]
    assert result.verdict == "fail" and "b.bed" in result.message


def test_the_checker_reads_files_so_a_t0_context_refuses(tmp_path):
    (tmp_path / "t.bed").write_text(HEADERLESS)
    call = Call("vcftools", ("--bed", "t.bed"), tmp_path)
    match = Match(RULES["vcftools.bed_header"])
    check = known_checkers()["vcftools_bed_header"]
    with pytest.raises(TierViolation):
        check(call, match, Ctx(tmp_path, "T0"))
    assert check(call, match, Ctx(tmp_path, "T1")).verdict == "fail"


# ---- evidence against vcftools 0.1.17 and the F53 workspaces ----------------------------------


def kept_sites(tools, work, *options):
    argv = ["vcftools", "--gzvcf", "variants.vcf.gz", *options, "--recode", "--stdout"]
    proc = tools.run(argv, work)
    assert proc.returncode == 0
    sites = [line for line in proc.stdout.splitlines() if line and line[0] != "#"]
    return len(sites), proc.stderr


def test_vcftools_skips_the_first_line_of_the_bed_file(tmp_path, tools):
    task = build_task("F53", 4, "faulted", tmp_path / "f", tools, "implicit")
    work = tmp_path / "f"
    n_naive, stderr = kept_sites(tools, work, "--bed", "targets.bed")
    assert n_naive == len(task.truth["naive"]) < len(task.truth["expected"])
    assert "Read 8 BED file entries" in stderr  # the skipped line is counted
    bed = (work / "targets.bed").read_text()
    (work / "hdr.bed").write_text("#chrom\tstart\tend\n" + bed)
    assert kept_sites(tools, work, "--bed", "hdr.bed")[0] == len(task.truth["expected"])
    n_excl, _ = kept_sites(tools, work, "--exclude-bed", "targets.bed")
    n_all = kept_sites(tools, work)[0]
    assert n_all - n_excl == len(task.truth["naive"])  # the first interval excluded nothing


def test_the_rule_fails_the_faulted_workspace_and_passes_the_clean_one(tmp_path, tools):
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F53", 6, twin, work, tools, "implicit")
        argv = ["--gzvcf", "variants.vcf.gz", "--bed", "targets.bed", "--recode", "--stdout"]
        assert verdict("vcftools.bed_header", "vcftools", argv, work)[1].verdict == expected


# ---- admission ---------------------------------------------------------------------------------


def test_admission_bed_header(tools, tmp_path):
    stats = measure("vcftools.bed_header", [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    # every undeclared call that reads the headerless file as it is gets caught
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 8
    assert promotion_is_consistent(stats)


# ---- vcftools.keep_filtered_dot ----------------------------------------------------------------

KEEP = "vcftools.keep_filtered_dot"
KEY = "unfiltered_records"
VCF_HEAD = (
    "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
    '##FILTER=<ID=LowQual,Description="l">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
)


def filtered_vcf(tmp_path, values, name="f.vcf"):
    rows = "".join(
        f"chr1\t{100 + 10 * i}\tf{i}\tA\tC\t50\t{v}\t.\tGT\t0/1\t0/0\n"
        for i, v in enumerate(values)
    )
    cols = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
    (tmp_path / name).write_text(VCF_HEAD + cols + rows)
    return name


def keep_call(tmp_path, values, options, declared=None, flag="--vcf"):
    name = filtered_vcf(tmp_path, values)
    argv = [flag, name, *options, "--recode", "--stdout"]
    return verdict(KEEP, "vcftools", argv, tmp_path, declared)


def test_keep_filtered_pass_fails_with_a_dot_record_and_the_fix_is_remove_filtered_all(tmp_path):
    decision, result = keep_call(tmp_path, WITH_DOT_V, ["--keep-filtered", "PASS"])
    assert decision.action == "block" and result.code == "dot_filter_dropped"
    text = render_block(decision)
    assert text.startswith("FIX: use --remove-filtered-all") and "1 of 4 records" in text


WITH_DOT_V = ["PASS", ".", "LowQual", "PASS"]


@pytest.mark.parametrize(
    ("options", "verdict_name"),
    [
        (["--keep-filtered", "PASS"], "fail"),
        (["--keep-filtered", "PASS", "--keep-filtered", "."], "fail"),
        (["--remove-filtered-all"], "pass"),
        (["--remove-filtered", "LowQual"], "pass"),
        (["--remove-filtered-all", "--maf", "0"], "pass"),
    ],
)
def test_vcftools_filter_options(tmp_path, options, verdict_name):
    assert keep_call(tmp_path, WITH_DOT_V, options)[1].verdict == verdict_name


def test_gzvcf_declarations_and_no_dot_records(tmp_path):
    import gzip

    name = filtered_vcf(tmp_path, WITH_DOT_V)
    with gzip.open(tmp_path / "f.vcf.gz", "wb") as handle:
        handle.write((tmp_path / name).read_bytes())
    argv = ["--gzvcf", "f.vcf.gz", "--keep-filtered", "PASS", "--recode", "--stdout"]
    assert verdict(KEEP, "vcftools", argv, tmp_path)[1].verdict == "fail"
    assert verdict(KEEP, "vcftools", argv, tmp_path, {KEY: "drop"})[1].verdict == "pass"
    decision, result = verdict(KEEP, "vcftools", argv, tmp_path, {KEY: "keep"})
    assert result.code == "declared_keep_but_dropped" and decision.action == "block"
    keep = ["--gzvcf", "f.vcf.gz", "--remove-filtered-all", "--stdout", "--recode"]
    assert verdict(KEEP, "vcftools", keep, tmp_path, {KEY: "drop"})[1].code == (
        "declared_drop_but_kept"
    )
    # no record has "." -> nothing to say
    clean = ["PASS", "LowQual", "PASS"]
    for declared in (None, {KEY: "keep"}, {KEY: "drop"}):
        assert (
            keep_call(tmp_path, clean, ["--keep-filtered", "PASS"], declared)[1].verdict == "pass"
        )


def test_vcftools_ignores_filter_unless_asked_and_keep_filtered_drops_dot(tmp_path, tools):
    task = build_task("F55", 4, "faulted", tmp_path, tools, "implicit")

    def ids(*options):
        argv = ["vcftools", "--gzvcf", "variants.vcf.gz", *options, "--recode", "--stdout"]
        proc = tools.run(argv, tmp_path)
        assert proc.returncode == 0
        return [line.split("\t")[2] for line in proc.stdout.splitlines() if line and line[0] != "#"]

    assert len(ids()) > len(task.truth["expected"])  # LowQual & co. stay without an option
    assert sorted(ids("--keep-filtered", "PASS")) == task.truth["naive"]
    assert sorted(ids("--remove-filtered-all")) == task.truth["expected"]
    assert sorted(ids("--keep-filtered", "PASS", "--keep-filtered", ".")) == task.truth["naive"]


def test_admission_keep_filtered_dot(tools, tmp_path):
    stats = measure(KEEP, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 4
    assert promotion_is_consistent(stats)


# ---- vcftools.subset_monomorphic ---------------------------------------------------------------

SUBSET = "vcftools.subset_monomorphic"
SUBSET_KEY = "subset_sites"
SUB_HEAD = (
    "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
)
SUB_ROWS = (
    "chr1\t100\tm0\tA\tC\t50\tPASS\t.\tGT\t0/1\t0/0\t1/1\t0/0\n"
    "chr1\t200\tm1\tA\tC\t50\tPASS\t.\tGT\t0/0\t0/0\t0/1\t0/0\n"
)


def sub_call(tmp_path, options, declared=None, rows=SUB_ROWS):
    (tmp_path / "s.vcf").write_text(SUB_HEAD + rows)
    (tmp_path / "keep.txt").write_text("s1\ns2\n")
    argv = ["--vcf", "s.vcf", *options, "--recode", "--stdout"]
    return verdict(SUBSET, "vcftools", argv, tmp_path, declared)


def test_keep_without_a_filter_fails_with_the_monomorphic_site_named(tmp_path):
    decision, result = sub_call(tmp_path, ["--keep", "keep.txt"])
    assert decision.action == "block" and result.code == "monomorphic_sites_kept"
    text = render_block(decision)
    assert text.startswith("FIX: add --mac 1") and "1 of 2 sites" in text and "m1" in text


@pytest.mark.parametrize(
    ("options", "verdict_name"),
    [
        (["--keep", "keep.txt"], "fail"),
        (["--indv", "s1", "--indv", "s2"], "fail"),
        (["--remove", "keep.txt"], "pass"),  # s3 and s4 remain: m1 is variable there
        (["--remove-indv", "s3", "--remove-indv", "s4"], "fail"),
        (["--keep", "keep.txt", "--mac", "1"], "pass"),
        (["--keep", "keep.txt", "--mac", "0"], "fail"),
        (["--keep", "keep.txt", "--non-ref-ac-any", "1"], "pass"),
        (["--keep", "keep.txt", "--non-ref-ac", "1"], "pass"),
        (["--keep", "keep.txt", "--maf", "0.0001"], "pass"),
        (["--keep", "keep.txt", "--maf", "0"], "fail"),
        (["--keep", "keep.txt", "--minQ", "10"], "fail"),
        (["--keep", "absent.txt"], "pass"),
    ],
)
def test_vcftools_subset_options(tmp_path, options, verdict_name):
    assert sub_call(tmp_path, options)[1].verdict == verdict_name


def test_vcftools_subset_declarations(tmp_path):
    options = ["--keep", "keep.txt"]
    assert sub_call(tmp_path, options, {SUBSET_KEY: "keep_all_sites"})[1].verdict == "pass"
    assert sub_call(tmp_path, options, {SUBSET_KEY: "filtered_downstream"})[1].verdict == "pass"
    assert sub_call(tmp_path, options, {SUBSET_KEY: "variant_sites_only"})[1].verdict == "fail"
    assert (
        sub_call(tmp_path, [*options, "--mac", "1"], {SUBSET_KEY: "variant_sites_only"})[1].verdict
        == "pass"
    )


def test_vcftools_keeps_monomorphic_sites_after_keep(tmp_path, tools):
    task = build_task("F57", 3, "faulted", tmp_path, tools, "implicit")
    base = ["vcftools", "--gzvcf", "cohort.vcf.gz", "--keep", "group_b.txt"]

    def sites(*options):
        proc = tools.run([*base, *options, "--recode", "--stdout"], tmp_path)
        assert proc.returncode == 0
        return sum(1 for line in proc.stdout.splitlines() if line and line[0] != "#")

    assert sites() == task.truth["naive"] > task.truth["expected"]
    for options in (["--mac", "1"], ["--non-ref-ac-any", "1"], ["--non-ref-ac", "1"]):
        assert sites(*options) == task.truth["expected"], options
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F57", 5, twin, work, tools, "implicit")
        argv = ["--gzvcf", "cohort.vcf.gz", "--keep", "group_b.txt", "--recode", "--stdout"]
        assert verdict(SUBSET, "vcftools", argv, work)[1].verdict == expected


def test_admission_vcftools_subset_monomorphic(tools, tmp_path):
    stats = measure(SUBSET, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].failed == 0 and stats.cells[("right", False)].n > 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= ADMISSION_SEEDS * 4
    assert promotion_is_consistent(stats)


# ---- var.missing_threshold ---------------------------------------------------------------------

MISS = "var.missing_threshold"
MISS_KEY = "missing_threshold_meaning"


def miss_call(tool, argv, declared=None):
    return verdict(MISS, tool, argv, Path("/nonexistent"), declared)


def test_missing_threshold_spec_is_a_required_t0_builtin():
    rule = RULES[MISS]
    assert (rule.kind, rule.tier, rule.slot) == ("call", "T0", "metric_scale")
    assert rule.check == "option_implies_declared" and rule.incidents == ["GEN-31"]
    assert rule.declare is not None and rule.declare.required
    assert rule.declare.values == ["completeness", "missing_rate"]
    assert rule.applies.tools == ["vcftools", "plink", "plink2"]


@pytest.mark.parametrize(
    ("tool", "argv", "meaning"),
    [
        ("vcftools", ["--vcf", "v.vcf", "--max-missing", "0.9"], "completeness"),
        ("vcftools", ["--gzvcf", "v.vcf.gz", "--max-missing=0.9", "--recode"], "completeness"),
        ("plink", ["--vcf", "v.vcf", "--geno", "0.1", "--make-bed"], "missing_rate"),
        ("plink", ["--bfile", "c", "--mind", "0.1"], "missing_rate"),
        ("plink2", ["--vcf", "v.vcf", "--geno", "0.1"], "missing_rate"),
    ],
)
def test_the_option_decides_the_meaning_and_the_declaration_must_agree(tool, argv, meaning):
    assert miss_call(tool, argv, {MISS_KEY: meaning})[1].verdict == "pass"
    other = "missing_rate" if meaning == "completeness" else "completeness"
    decision, result = miss_call(tool, argv, {MISS_KEY: other})
    assert result.code == "declared_mismatch" and decision.action == "block"
    text = render_block(decision)
    assert f"you declared {MISS_KEY}={other}" in text
    assert "--max-missing" in text or "--geno" in text
    decision, result = miss_call(tool, argv)
    assert result.code == "declaration_missing" and decision.action == "block"


def test_fix_names_the_conversion_of_the_rate():
    text = render_block(
        miss_call(
            "vcftools", ["--vcf", "v.vcf", "--max-missing", "0.1"], {MISS_KEY: "missing_rate"}
        )[0]
    )
    assert text.startswith(
        "FIX: use --geno <rate> in plink or --max-missing <1 - rate> in vcftools"
    )
    text = render_block(
        miss_call("plink", ["--vcf", "v.vcf", "--geno", "0.9"], {MISS_KEY: "completeness"})[0]
    )
    assert text.startswith("FIX: use --max-missing <fraction called> in vcftools")


def test_other_options_are_not_matched():
    for tool, argv in (
        ("vcftools", ["--vcf", "v.vcf", "--max-missing-count", "2"]),
        ("vcftools", ["--vcf", "v.vcf", "--maf", "0.05"]),
        ("plink", ["--bfile", "c", "--maf", "0.05"]),
    ):
        assert miss_call(tool, argv)[1] is None


def test_admission_missing_threshold(tools, tmp_path):
    stats = measure(MISS, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    assert not stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    # a wrong call declared with the meaning its author had in mind (the opposite of its option)
    assert stats.cells[("wrong", True)].failed >= ADMISSION_SEEDS * 4
    assert promotion_is_consistent(stats)
