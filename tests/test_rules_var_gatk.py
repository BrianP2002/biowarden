"""Rules of the variant pack for GATK: unit tests, evidence against GATK 4.6.2.0, admission.

gatk.missing_annotation: VariantFiltration does not filter a record that lacks the annotation.
GATK is a JVM program, so the evidence and admission tests run few seeds (a call takes 2-8 s).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import var_gatk  # noqa: F401  (registers the checkers)
from biowarden.engine.spec import load_packaged
from helpers import MIN_RIGHT_CALLS

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("var_gatk")
RULES = {rule.id: rule for rule in SPEC.rules}
GATK_ADMISSION_SEEDS = 2  # about 10 JVM runs per seed
KEY = "missing_annotation"
MISSING = "gatk.missing_annotation"
JAVA = ["--java-options", "-Xmx512m"]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "gatk" not in env.real_bins(["gatk"]):
        pytest.skip("samtools, bcftools and gatk are not available")
    return env


def verdict(argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call("gatk", tuple(argv), Path(cwd), declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == MISSING]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def write_vcf(path: Path, values) -> Path:
    head = (
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
        '##INFO=<ID=QD,Number=1,Type=Float,Description="q">\n'
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="d">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
    )
    rows = ""
    for i, qd in enumerate(values):
        info = "DP=30" if qd is None else f"QD={qd};DP=30"
        rows += f"chr1\t{100 + 10 * i}\tg{i}\tA\tC\t50\tPASS\t{info}\tGT\t0/1\n"
    path.write_text(head + rows)
    return path


def vf(extra, name="g.vcf", expression="QD < 2.0"):
    return [
        *JAVA,
        "VariantFiltration",
        "-V",
        name,
        "--filter-expression",
        expression,
        "--filter-name",
        "lowQD",
        *extra,
        "-O",
        "o.vcf",
    ]


PARTIAL = [5.0, None, 1.0, 9.0]


def test_spec_shape():
    rule = RULES[MISSING]
    assert (rule.kind, rule.tier, rule.slot) == ("call", "T1", "missing_value")
    assert rule.applies.tools == ["gatk"] and rule.applies.subcommands == ["VariantFiltration"]
    assert rule.applies.options_any == ["--filter-expression", "-filter"]
    assert rule.declare is not None and not rule.declare.required
    assert (rule.declare.key, rule.declare.values) == (KEY, ["record_kept", "record_removed"])
    assert rule.check in known_checkers()


def test_variant_filtration_on_a_partly_missing_tag_fails_with_the_flag_as_the_fix(tmp_path):
    write_vcf(tmp_path / "g.vcf", PARTIAL)
    decision, result = verdict(vf([]), tmp_path)
    assert decision.action == "block" and result.code == "missing_values_kept"
    text = render_block(decision)
    assert text.startswith("FIX: add --missing-values-evaluate-as-failing")
    assert "1 of 4 records" in text and "no QD annotation" in text and "PASSES" in text


def test_the_flag_passes_and_declarations_are_checked(tmp_path):
    write_vcf(tmp_path / "g.vcf", PARTIAL)
    flag = ["--missing-values-evaluate-as-failing"]
    assert verdict(vf(flag), tmp_path)[1].verdict == "pass"
    assert verdict(vf(flag), tmp_path, {KEY: "record_removed"})[1].verdict == "pass"
    result = verdict(vf(flag), tmp_path, {KEY: "record_kept"})[1]
    assert result.code == "declared_missing_mismatch"
    assert verdict(vf([]), tmp_path, {KEY: "record_kept"})[1].verdict == "pass"
    decision, result = verdict(vf([]), tmp_path, {KEY: "record_removed"})
    assert result.code == "declared_missing_mismatch" and decision.action == "block"
    assert "--missing-values-evaluate-as-failing" in render_block(decision)


@pytest.mark.parametrize(
    "argv",
    [
        ["VariantFiltration", "-V", "g.vcf", "-filter", "QD < 2", "--filter-name", "x"],
        ["VariantFiltration", "--variant", "g.vcf", "--filter-expression=QD < 2"],
        ["VariantFiltration", "-V", "g.vcf", "--filter-expression", "QD < 2 || DP > 5000"],
        ["VariantFiltration", "-V", "g.vcf", "--filter-expression", "QD >= 2"],
    ],
)
def test_spellings_are_seen(tmp_path, argv):
    write_vcf(tmp_path / "g.vcf", PARTIAL)
    assert verdict([*JAVA, *argv], tmp_path)[1].code == "missing_values_kept"


def test_complete_missing_or_unrelated_attributes_are_silent(tmp_path):
    write_vcf(tmp_path / "g.vcf", [5.0, 1.0, 9.0])
    assert verdict(vf([]), tmp_path)[1].verdict == "pass"  # no record lacks QD
    write_vcf(tmp_path / "g.vcf", [None, None])
    assert verdict(vf([]), tmp_path)[1].verdict == "pass"  # QD is missing everywhere
    write_vcf(tmp_path / "g.vcf", PARTIAL)
    assert verdict(vf([], expression="DP < 10"), tmp_path)[1].verdict == "pass"
    # a JEXL test of the attribute itself compares nothing: the rule has no attribute to check
    has_attribute = vf([], expression="vc.hasAttribute('QD') && QUAL < 10")
    assert verdict(has_attribute, tmp_path)[1].verdict == "pass"
    other = ["SelectVariants", "-V", "g.vcf", "-select", "QD > 2", "-O", "s.vcf"]
    assert verdict(other, tmp_path)[1] is None
    absent = [*JAVA, "VariantFiltration", "-V", "absent.vcf", "-filter", "QD < 2"]
    assert verdict(absent, tmp_path)[1].verdict == "pass"  # unreadable input: nothing to compare


def test_the_checker_reads_files(tmp_path):
    write_vcf(tmp_path / "g.vcf", PARTIAL)
    call = Call("gatk", tuple(vf([])), tmp_path)
    check = known_checkers()["gatk_missing_annotation"]
    with pytest.raises(TierViolation):
        check(call, Match(RULES[MISSING]), Ctx(tmp_path, "T0"))


# ---- evidence against GATK 4.6.2.0 and the F56 workspaces --------------------------------------


def test_variant_filtration_passes_records_without_the_annotation(tmp_path, tools):
    task = build_task("F56", 3, "faulted", tmp_path, tools, "implicit")

    def passing(*extra):
        argv = ["gatk", *vf(extra, "variants.vcf.gz", "QD < 2.0")]
        argv[-1] = "o.vcf.gz"
        proc = tools.run(argv, tmp_path)
        assert proc.returncode == 0
        if not extra:  # the default run says nothing about the records without QD
            assert "missing" not in proc.stderr.lower() and "no qd" not in proc.stderr.lower()
        out = tools.run(["bcftools", "view", "-H", "-f", "PASS", "o.vcf.gz"], tmp_path)
        return sorted(line.split("\t")[2] for line in out.stdout.splitlines())

    assert passing() == task.truth["naive"]
    assert passing("--missing-values-evaluate-as-failing") == task.truth["expected"]
    argv = ["VariantFiltration", "-V", "variants.vcf.gz", "-filter", "QD < 2.0"]
    clean_verdict = verdict([*JAVA, *argv], tmp_path)[1]
    assert clean_verdict.verdict == "fail"


def test_admission_missing_annotation(tools, tmp_path):
    stats = measure(MISSING, [SPEC], range(GATK_ADMISSION_SEEDS), tools, tmp_path)
    assert stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS * GATK_ADMISSION_SEEDS // 40
    assert stats.right_declared.failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", False)].failed >= GATK_ADMISSION_SEEDS * 2
