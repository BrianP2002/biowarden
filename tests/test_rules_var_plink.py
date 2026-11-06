"""Rules of the variant pack for plink 1.9 and plink2: unit tests, evidence, admission.

plink.subset_monomorphic (F57), plink.allele_order (F58), plink.vcf_multiallelic (F59). The unit
tests write small VCFs; the evidence tests run the installed plink 1.9 and plink2 2.0.0-a.7 on the
family workspaces; the admission tests measure the rules over ADMISSION_SEEDS seeds with calls
labelled by running plink (bench/admission/cases_var_plink.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import var_plink  # noqa: F401  (registers the checkers)
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("var_plink")
RULES = {rule.id: rule for rule in SPEC.rules}
MEM = ["--memory", "1000", "--silent"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or {"plink", "plink2"} - set(env.real_bins(["plink", "plink2"])):
        pytest.skip("samtools, bcftools and plink are not available")
    return env


def verdict(rule_id, tool, argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def write_vcf(path: Path, samples, rows) -> Path:
    head = (
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
    )
    path.write_text(head + "".join(rows))
    return path


def row(pos, alt, *gts):
    return f"chr1\t{pos}\tp{pos}\tA\t{alt}\t50\tPASS\t.\tGT\t" + "\t".join(gts) + "\n"


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_declares_the_rules_with_the_expected_shape():
    assert set(RULES) == {
        "plink.subset_monomorphic",
        "plink.allele_order",
        "plink.vcf_multiallelic",
    }
    for rule in RULES.values():
        assert (rule.kind, rule.tier) == ("call", "T1") and rule.check in known_checkers()
        assert rule.declare is not None and not rule.declare.required
    assert RULES["plink.subset_monomorphic"].applies.tools == ["plink", "plink2"]
    assert RULES["plink.allele_order"].applies.tools == ["plink"]
    assert RULES["plink.allele_order"].incidents == ["GEN-30", "R2-OMX-25"]
    assert RULES["plink.vcf_multiallelic"].declare.values == ["first_alt_only", "biallelic_only"]


# ---- plink.subset_monomorphic ------------------------------------------------------------------

SUB = "plink.subset_monomorphic"
SUB_KEY = "subset_sites"
SUB_ROWS = [
    row(100, "C", "0/1", "0/0", "1/1", "0/0"),
    row(200, "C", "0/0", "0/0", "0/1", "0/0"),
    row(300, "C", "1/1", "1/1", "0/1", "0/0"),
]


def sub_call(tmp_path, options, declared=None, tool="plink", rows=SUB_ROWS):
    write_vcf(tmp_path / "s.vcf", ["s1", "s2", "s3", "s4"], rows)
    (tmp_path / "keep.txt").write_text("s1 s1\ns2 s2\n")
    argv = [*MEM, "--vcf", "s.vcf", *options, "--make-bed"]
    return verdict(SUB, tool, argv, tmp_path, declared)


def test_keep_without_a_filter_fails_and_the_fix_is_mac_1(tmp_path):
    decision, result = sub_call(tmp_path, ["--keep", "keep.txt"])
    assert decision.action == "block" and result.code == "monomorphic_sites_kept"
    text = render_block(decision)
    # p200 has no alternate allele in the two kept samples, p300 only the same one (plink counts
    # it too); the allele rule no longer adds an "Also:" line to a call that only writes a fileset
    assert text.startswith("FIX: add --mac 1") and "2 of 3 sites" in text and "p200" in text
    assert "Also:" not in text


@pytest.mark.parametrize(
    ("options", "tool", "verdict_name"),
    [
        (["--keep", "keep.txt"], "plink", "fail"),
        (["--keep", "keep.txt"], "plink2", "fail"),
        (["--remove", "keep.txt"], "plink", "pass"),  # s3 and s4 vary at every site
        (["--keep", "keep.txt", "--mac", "1"], "plink", "pass"),
        (["--keep", "keep.txt", "--mac", "0"], "plink", "fail"),
        (["--keep", "keep.txt", "--maf", "0.01"], "plink", "pass"),
        (["--keep", "keep.txt", "--maf", "0"], "plink", "fail"),
        (["--keep", "keep.txt", "--geno", "0.1"], "plink", "fail"),
        (["--keep", "keep.txt", "--mac", "1"], "plink2", "pass"),
    ],
)
def test_plink_subset_options(tmp_path, options, tool, verdict_name):
    assert sub_call(tmp_path, options, tool=tool)[1].verdict == verdict_name


def test_fixed_alternate_in_the_subset_counts_as_monomorphic_for_plink(tmp_path):
    # s1 and s2 are 1/1 at p300: the minor allele count is 0 there, --mac 1 removes it
    assert sub_call(tmp_path, ["--keep", "keep.txt"])[1].verdict == "fail"
    rows = [row(100, "C", "0/1", "0/0", "1/1", "0/0"), row(300, "C", "1/1", "1/1", "0/1", "0/0")]
    result = sub_call(tmp_path, ["--keep", "keep.txt"], rows=rows)[1]
    assert result.verdict == "fail" and "p300" in result.message


def test_subset_declarations_and_calls_that_write_nothing(tmp_path):
    options = ["--keep", "keep.txt"]
    for declared in ("keep_all_sites", "filtered_downstream"):
        assert sub_call(tmp_path, options, {SUB_KEY: declared})[1].verdict == "pass"
    assert sub_call(tmp_path, options, {SUB_KEY: "variant_sites_only"})[1].verdict == "fail"
    write_vcf(tmp_path / "s.vcf", ["s1", "s2", "s3", "s4"], SUB_ROWS)
    quiet = verdict(SUB, "plink", [*MEM, "--vcf", "s.vcf", "--keep", "keep.txt"], tmp_path)[1]
    assert quiet.verdict == "pass"  # no output option: nothing is written


# ---- plink.allele_order ------------------------------------------------------------------------

A1 = "plink.allele_order"
A1_KEY = "plink_a1_allele"
ALT_MAJOR = [row(100, "C", "0/1", "0/0", "1/1", "0/0"), row(200, "C", "1/1", "1/1", "0/1", "1/1")]
ALT_MINOR = [row(100, "C", "0/1", "0/0", "1/1", "0/0"), row(200, "C", "0/1", "0/0", "0/0", "0/0")]


def a1_call(tmp_path, options, declared=None, rows=ALT_MAJOR):
    write_vcf(tmp_path / "a.vcf", ["s1", "s2", "s3", "s4"], rows)
    return verdict(A1, "plink", [*MEM, "--vcf", "a.vcf", *options, "--freq"], tmp_path, declared)


def test_freq_on_a_vcf_with_alt_major_sites_fails_and_names_the_site(tmp_path):
    decision, result = a1_call(tmp_path, [])
    assert decision.action == "block" and result.code == "a1_is_minor_allele"
    text = render_block(decision)
    assert text.startswith("FIX: add --keep-allele-order") and "p200" in text
    assert "ALT frequency 0.88" in text and "1 of 2 sites" in text


def test_the_allele_order_declarations(tmp_path):
    assert a1_call(tmp_path, [], {A1_KEY: "minor_allele"})[1].verdict == "pass"
    decision, result = a1_call(tmp_path, [], {A1_KEY: "alt_allele"})
    assert result.code == "declared_alt_but_minor" and decision.action == "block"
    assert "you declared plink_a1_allele=alt_allele" in render_block(decision)
    keep = ["--keep-allele-order"]
    assert a1_call(tmp_path, keep)[1].verdict == "pass"
    assert a1_call(tmp_path, keep, {A1_KEY: "alt_allele"})[1].verdict == "pass"
    assert a1_call(tmp_path, keep, {A1_KEY: "minor_allele"})[1].code == "declared_minor_but_alt"


def test_alt_minor_everywhere_and_controlled_alleles_are_silent(tmp_path):
    for declared in (None, {A1_KEY: "alt_allele"}, {A1_KEY: "minor_allele"}):
        assert a1_call(tmp_path, [], declared, ALT_MINOR)[1].verdict == "pass"
    for option in (["--a1-allele", "f.txt"], ["--a2-allele", "f.txt"], ["--real-ref-alleles"]):
        assert a1_call(tmp_path, option)[1].verdict == "pass"


@pytest.mark.parametrize(
    "report", ["--freq", "--freqx", "--assoc", "--linear", "--logistic", "--score", "--recode"]
)
def test_the_options_that_use_a1_are_seen(tmp_path, report):
    write_vcf(tmp_path / "a.vcf", ["s1", "s2", "s3", "s4"], ALT_MAJOR)
    argv = [*MEM, "--vcf", "a.vcf", report]
    assert verdict(A1, "plink", argv, tmp_path)[1].code == "a1_is_minor_allele"
    quiet = verdict(A1, "plink", [*MEM, "--vcf", "a.vcf", "--missing"], tmp_path)[1]
    assert quiet.verdict == "pass"
    assert verdict(A1, "plink2", argv, tmp_path)[1] is None  # plink2 keeps REF/ALT


@pytest.mark.parametrize("write", ["--make-bed", "--make-just-bim"])
def test_a_call_that_only_filters_and_writes_a_fileset_is_not_judged(tmp_path, write):
    # E-12: these two used to make the rule fire on every call of a family that is not F58
    write_vcf(tmp_path / "a.vcf", ["s1", "s2", "s3", "s4"], ALT_MAJOR)
    for options in ([], ["--geno", "0.1"], ["--maf", "0.01"], ["--keep-allele-order"]):
        argv = [*MEM, "--vcf", "a.vcf", *options, write]
        result = verdict(A1, "plink", argv, tmp_path)[1]
        assert (result.verdict, result.code) == ("pass", "ok"), options
    both = [*MEM, "--vcf", "a.vcf", "--geno", "0.1", write, "--freq"]
    assert verdict(A1, "plink", both, tmp_path)[1].code == "a1_is_minor_allele"  # --freq reports A1


def test_a_tie_keeps_alt_as_a1(tmp_path):
    rows = [row(100, "C", "0/1", "0/1", "0/0", "0/0")]  # ALT frequency 0.25
    tie = [row(100, "C", "0/1", "0/1", "0/1", "0/1")]  # exactly 0.5
    assert a1_call(tmp_path, [], rows=rows)[1].verdict == "pass"
    assert a1_call(tmp_path, [], rows=tie)[1].verdict == "pass"


# ---- plink.vcf_multiallelic --------------------------------------------------------------------

MULTI = "plink.vcf_multiallelic"
MULTI_KEY = "vcf_multiallelic"
MULTI_ROWS = [
    row(100, "C", "0/1", "0/0", "1/1", "0/0"),
    row(200, "C,T", "0/1", "0/2", "1/1", "0/0"),
]
SECOND_UNUSED = [row(100, "C,T", "0/1", "0/0", "1/1", "0/0")]


def multi_call(tmp_path, options=(), declared=None, rows=MULTI_ROWS, tool="plink"):
    write_vcf(tmp_path / "m.vcf", ["s1", "s2", "s3", "s4"], rows)
    argv = [*MEM, "--vcf", "m.vcf", *options, "--geno", "0.1"]
    return verdict(MULTI, tool, argv, tmp_path, declared)


def test_a_second_alt_call_fails_and_the_fix_points_to_plink2(tmp_path):
    decision, result = multi_call(tmp_path)
    assert decision.action == "block" and result.code == "second_alt_set_missing"
    text = render_block(decision)
    assert (
        text.startswith("FIX: run plink2") and "1 multiallelic sites with 1 genotype calls" in text
    )
    assert "at p200" in text


def test_biallelic_only_declarations_and_plink2(tmp_path):
    assert multi_call(tmp_path, ["--biallelic-only"])[1].verdict == "pass"
    assert multi_call(tmp_path, [], {MULTI_KEY: "first_alt_only"})[1].verdict == "pass"
    result = multi_call(tmp_path, [], {MULTI_KEY: "biallelic_only"})[1]
    assert result.code == "declared_biallelic_only_without_option"
    result = multi_call(tmp_path, ["--biallelic-only"], {MULTI_KEY: "first_alt_only"})[1]
    assert result.code == "declared_first_alt_only_with_option"
    assert multi_call(tmp_path, [], tool="plink2")[1] is None


def test_multiallelic_sites_without_second_alt_calls_are_silent(tmp_path):
    assert multi_call(tmp_path, rows=SECOND_UNUSED)[1].verdict == "pass"
    assert multi_call(tmp_path, rows=[MULTI_ROWS[0]])[1].verdict == "pass"


def test_the_checkers_read_files(tmp_path):
    write_vcf(tmp_path / "m.vcf", ["s1", "s2", "s3", "s4"], MULTI_ROWS)
    call = Call("plink", ("--vcf", "m.vcf", "--geno", "0.1"), tmp_path)
    check = known_checkers()["plink_vcf_multiallelic"]
    with pytest.raises(TierViolation):
        check(call, Match(RULES[MULTI]), Ctx(tmp_path, "T0"))


# ---- evidence against plink 1.9 / plink2 and the family workspaces -----------------------------


def test_plink_keep_make_bed_keeps_monomorphic_variants(tmp_path, tools):
    task = build_task("F57", 3, "faulted", tmp_path, tools, "implicit")
    names = (tmp_path / "group_b.txt").read_text().split()
    (tmp_path / "keep_fid.txt").write_text("".join(f"{n} {n}\n" for n in names))

    def n_bim(*options):
        argv = ["plink", *MEM, "--vcf", "cohort.vcf.gz", "--keep", "keep_fid.txt", *options]
        argv += ["--make-just-bim", "--out", "k"]
        proc = tools.run(argv, tmp_path)
        assert proc.returncode == 0 and proc.stderr == ""
        return len((tmp_path / "k.bim").read_text().splitlines())

    assert n_bim() == task.truth["naive"] > task.truth["expected"]
    assert n_bim("--mac", "1") == n_bim("--maf", "0.0001") == task.truth["expected"]


def test_plink_makes_the_minor_allele_a1_and_keep_allele_order_keeps_alt(tmp_path, tools):
    task = build_task("F58", 3, "faulted", tmp_path, tools, "implicit")
    from biowarden.bench.tasks.extra import f58_plink_a1 as f58

    default = f58.run_frq(tmp_path, tools)
    ordered = f58.run_frq(tmp_path, tools, "--keep-allele-order")
    assert ordered == task.truth["expected"] and default == task.truth["naive"] != ordered
    rows = [r.split() for r in (tmp_path / "frq.frq").read_text().splitlines()[1:]]
    assert all(r[2] == alt for r, alt in zip(rows, alt_alleles(tmp_path), strict=True))
    for twin, expected in (("clean", "fail"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F58", 5, twin, work, tools, "implicit")
        argv = [*MEM, "--vcf", "cohort.vcf.gz", "--freq"]
        assert verdict(A1, "plink", argv, work)[1].verdict == expected
        argv = [*MEM, "--vcf", "cohort.vcf.gz", "--keep-allele-order", "--freq"]
        assert verdict(A1, "plink", argv, work)[1].verdict == "pass"


def alt_alleles(work: Path) -> list[str]:
    """The ALT column of cohort.vcf.gz, in file order."""
    import gzip

    with gzip.open(work / "cohort.vcf.gz", "rt") as handle:
        return [line.split("\t")[4] for line in handle if not line.startswith("#")]


def test_the_f59_calls_are_not_judged_by_the_allele_rule_but_a_frequency_report_still_is(
    tmp_path, tools
):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F59", 5, twin, work, tools, "implicit")
        for write in ("--make-just-bim", "--make-bed"):
            argv = [*MEM, "--vcf", "cohort.vcf.gz", "--geno", "0.1", write]
            assert verdict(A1, "plink", argv, work)[1].verdict == "pass", (twin, write)
        argv = [*MEM, "--vcf", "cohort.vcf.gz", "--geno", "0.1", "--freq"]
        assert verdict(A1, "plink", argv, work)[1].code == "a1_is_minor_allele"  # ALT is major


def test_plink_reads_a_second_alt_as_missing_and_geno_drops_the_site(tmp_path, tools):
    task = build_task("F59", 3, "faulted", tmp_path, tools, "implicit")
    from biowarden.bench.tasks.extra import f59_plink_multiallelic as f59

    proc = tools.run(
        ["plink", *MEM[:2], "--vcf", "cohort.vcf.gz", "--geno", "0.1", "--make-just-bim"]
        + ["--out", "w"],
        tmp_path,
    )
    assert proc.returncode == 0 and proc.stderr == ""
    assert "warn" not in proc.stdout.lower()  # nothing says the second ALT calls were dropped
    assert f59.plink_geno_count(tmp_path, tools) == task.truth["naive"]
    argv = ["plink2", *MEM[:2], "--vcf", "cohort.vcf.gz", "--geno", "0.1", "--make-just-pvar"]
    assert tools.run([*argv, "--out", "p2"], tmp_path).returncode == 0
    pvar = (tmp_path / "p2.pvar").read_text().splitlines()
    assert sum(1 for line in pvar if not line.startswith("#")) == task.truth["expected"]
    for twin, expected in (("clean", "pass"), ("faulted", "fail")):
        work = tmp_path / twin
        build_task("F59", 5, twin, work, tools, "implicit")
        argv = [*MEM, "--vcf", "cohort.vcf.gz", "--geno", "0.1", "--make-just-bim"]
        assert verdict(MULTI, "plink", argv, work)[1].verdict == expected


# ---- admission ---------------------------------------------------------------------------------


def check_stats(stats, wrong_undeclared_min):
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= MIN_RIGHT_CALLS
    assert stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= wrong_undeclared_min
    assert promotion_is_consistent(stats)


def test_admission_plink_subset_monomorphic(tools, tmp_path):
    stats = measure(SUB, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    check_stats(stats, ADMISSION_SEEDS * 6)


def test_admission_plink_allele_order(tools, tmp_path):
    stats = measure(A1, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    check_stats(stats, ADMISSION_SEEDS * 6)


def test_admission_plink_vcf_multiallelic(tools, tmp_path):
    stats = measure(MULTI, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)
    check_stats(stats, ADMISSION_SEEDS * 4)
