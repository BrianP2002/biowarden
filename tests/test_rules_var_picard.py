"""Rule of the variant pack for Picard: unit tests, evidence against Picard 3.5.0, admission.

picard.markduplicates_remove (T0, declaration required): MarkDuplicates flags the duplicates and
keeps them unless REMOVE_DUPLICATES=true. Picard is a JVM program (about 8 s per call), so the
evidence and admission tests run few calls and few seeds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine
from biowarden.engine.checks import known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import var_picard
from biowarden.engine.spec import load_packaged
from helpers import MIN_RIGHT_CALLS

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("var_picard")
(RULE,) = SPEC.rules
KEY = "duplicate_output"
PICARD_ADMISSION_SEEDS = 2


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "picard" not in env.real_bins(["picard"]):
        pytest.skip("samtools, bcftools and picard are not available")
    return env


def verdict(tool, argv, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path("/nonexistent"), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


FLAG = ["-Xmx512m", "MarkDuplicates", "I=a.bam", "O=b.bam", "M=m.txt"]
REMOVE = [*FLAG, "REMOVE_DUPLICATES=true"]


def test_spec_shape():
    assert RULE.id == "picard.markduplicates_remove"
    assert (RULE.kind, RULE.tier, RULE.slot) == ("call", "T0", "default_param")
    assert RULE.applies.tools == ["picard", "gatk"]
    assert RULE.applies.subcommands == ["MarkDuplicates", "MarkDuplicatesSpark"]
    assert RULE.declare is not None and RULE.declare.required
    assert (RULE.declare.key, RULE.declare.values) == (KEY, ["flagged", "removed"])
    assert RULE.check in known_checkers()


def test_a_missing_declaration_is_refused_by_the_engine():
    decision, result = verdict("picard", FLAG)
    assert decision.action == "block" and result.code == "declaration_missing"


@pytest.mark.parametrize(
    ("argv", "tool", "declared", "ok"),
    [
        (FLAG, "picard", "flagged", True),
        (FLAG, "picard", "removed", False),
        (REMOVE, "picard", "removed", True),
        (REMOVE, "picard", "flagged", False),
        ([*FLAG, "--REMOVE_DUPLICATES", "true"], "picard", "removed", True),
        ([*FLAG, "-REMOVE_DUPLICATES", "true"], "picard", "removed", True),
        ([*FLAG, "REMOVE_DUPLICATES=false"], "picard", "removed", False),
        ([*FLAG, "--REMOVE_DUPLICATES", "false"], "picard", "flagged", True),
        ([*FLAG, "--REMOVE_SEQUENCING_DUPLICATES", "true"], "picard", "removed", False),
        (["MarkDuplicates", "-I", "a.bam", "-O", "b.bam", "-M", "m.txt"], "gatk", "removed", False),
        (
            [
                "MarkDuplicates",
                "-I",
                "a.bam",
                "-O",
                "b.bam",
                "-M",
                "m",
                "--REMOVE_DUPLICATES",
                "true",
            ],
            "gatk",
            "removed",
            True,
        ),
    ],
)
def test_the_option_against_the_declaration(argv, tool, declared, ok):
    decision, result = verdict(tool, argv, {KEY: declared})
    assert (result.verdict == "pass") is ok
    assert decision.action == ("allow" if ok else "block")


def test_the_message_names_the_flag_and_the_fix():
    decision, result = verdict("picard", FLAG, {KEY: "removed"})
    text = render_block(decision)
    assert result.code == "duplicates_flagged_not_removed"
    assert text.startswith("FIX: add REMOVE_DUPLICATES=true") and "0x400" in text
    assert "you declared" in text
    decision, result = verdict("picard", REMOVE, {KEY: "flagged"})
    assert render_block(decision).startswith("FIX: drop REMOVE_DUPLICATES=true")


def test_other_picard_tools_are_not_matched():
    for argv in (
        ["-Xmx512m", "SortVcf", "I=a.vcf", "O=b.vcf"],
        ["-Xmx512m", "CollectInsertSizeMetrics", "I=a.bam", "O=m.txt", "H=h.pdf"],
        ["-Xmx512m", "ValidateSamFile", "I=a.bam"],
    ):
        assert verdict("picard", argv, {KEY: "removed"})[1] is None


def test_removes_duplicates_reads_the_last_setting():
    assert var_picard.removes_duplicates(("MarkDuplicates", "REMOVE_DUPLICATES=true")) is True
    assert var_picard.removes_duplicates(("MarkDuplicates",)) is False
    assert (
        var_picard.removes_duplicates(
            ("--REMOVE_DUPLICATES", "true", "--REMOVE_DUPLICATES", "false")
        )
        is False
    )


def test_markduplicates_flags_and_remove_duplicates_removes(tmp_path, tools):
    task = build_task("F64", 3, "faulted", tmp_path, tools, "implicit")

    def run(*extra):
        argv = ["picard", "-Xmx512m", "MarkDuplicates", "I=reads.bam", "O=o.bam", "M=m.txt", *extra]
        proc = tools.run(argv, tmp_path)
        assert proc.returncode == 0
        assert (
            "Marking" in proc.stderr or extra
        )  # the INFO log says "Marking N records as duplicates"
        assert "removed" not in proc.stderr.lower()
        count = tools.run(["samtools", "view", "-c", "-f", "0x40", "o.bam"], tmp_path)
        flagged = tools.run(["samtools", "view", "-c", "-f", "0x400", "o.bam"], tmp_path)
        return int(count.stdout), int(flagged.stdout)

    pairs, flagged = run()
    assert pairs == task.truth["naive"] > task.truth["expected"] and flagged > 0
    metrics = [
        line.split("\t")
        for line in (tmp_path / "m.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    header, row = metrics[0], metrics[1]
    assert int(row[header.index("READ_PAIR_DUPLICATES")]) == pairs - task.truth["expected"]
    assert flagged == 2 * (pairs - task.truth["expected"])  # reads, while the metrics count pairs
    pairs, flagged = run("REMOVE_DUPLICATES=true")
    assert pairs == task.truth["expected"] and flagged == 0


def test_admission_markduplicates_remove(tools, tmp_path):
    stats = measure(RULE.id, [SPEC], range(PICARD_ADMISSION_SEEDS), tools, tmp_path)
    assert not stats.conditional and stats.admitted()
    assert stats.right_declared.n >= MIN_RIGHT_CALLS * PICARD_ADMISSION_SEEDS // 40
    assert stats.right_declared.failed == 0
    assert stats.killed() > 0 and all(cell.crashed == 0 for cell in stats.cells.values())
    assert stats.cells[("wrong", True)].failed >= PICARD_ADMISSION_SEEDS * 2
