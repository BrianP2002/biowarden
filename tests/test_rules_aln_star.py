"""Rule star.genecounts_column (family F31): unit tests, evidence against STAR, admission.

A fact rule over ReadsPerGene.out.tab files. The unit tests write small tables; the family tests use
F31 workspaces, where STAR 2.7.11b is run for real to tie the module docstring's claims (which
column holds the pairs for each library) to the tool; the admission test measures the rule over
ADMISSION_SEEDS seeds with calls labelled by running them on STAR's own table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f29_mapq_unique import star_map
from biowarden.bench.tasks.extra.f31_star_genecounts import column, parse_counts
from biowarden.engine import Call, Engine
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.rules import aln_star as rules
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_packaged("aln_star")
(RULE,) = SPEC.rules
KEY = "genecounts_column"
HEAD = (
    "N_unmapped\t0\t0\t0\nN_multimapping\t0\t0\t0\n"
    "N_noFeature\t0\t{nf3}\t{nf4}\nN_ambiguous\t5\t0\t0\n"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "STAR" not in env.real_bins(["STAR"]):
        pytest.skip("samtools and STAR are not available")
    return env


def table(rows: list[tuple[str, int, int, int]], nf3: int = 0, nf4: int = 0) -> str:
    return HEAD.format(nf3=nf3, nf4=nf4) + "".join(f"{g}\t{a}\t{b}\t{c}\n" for g, a, b, c in rows)


DUTP = table([("g1", 60, 1, 59), ("g2", 40, 2, 38), ("g3", 80, 0, 80)], nf3=177, nf4=3)
FORWARD = table([("g1", 60, 59, 1), ("g2", 40, 38, 2), ("g3", 80, 80, 0)], nf3=3, nf4=177)
UNSTRANDED = table([("g1", 60, 31, 29), ("g2", 40, 18, 22), ("g3", 80, 41, 39)], 90, 90)


def run(tool, argv, cwd, declared=None):
    eng = Engine([SPEC], mode="block", retry_cap=10**9)
    decision = eng.evaluate(Call(tool, tuple(argv), Path(cwd), declared or {}))
    return decision, (decision.outcomes[0].result if decision.outcomes else None)


def test_spec_is_a_conditional_t1_fact_rule():
    assert RULE.id == "star.genecounts_column" and (RULE.kind, RULE.tier) == ("fact", "T1")
    assert (
        RULE.applies.tools == []
        and RULE.slot == "strand"
        and RULE.incidents == ["GEN-20", "GEN-22"]
    )
    assert RULE.declare.key == KEY and not RULE.declare.required
    assert RULE.declare.values == ["unstranded", "first_read_strand", "second_read_strand"]
    assert RULE.check in known_checkers()


def test_the_table_is_read_with_t1_only(tmp_path):
    (tmp_path / "x_ReadsPerGene.out.tab").write_text(DUTP)
    call = Call("awk", ("{print $1,$3}", "x_ReadsPerGene.out.tab"), tmp_path)
    match = Match(RULE, (tmp_path / "x_ReadsPerGene.out.tab",))
    with pytest.raises(TierViolation):
        known_checkers()[RULE.check](call, match, Ctx(tmp_path, "T0"))


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [(5, 150, 4), (150, 5, 3), (100, 100, None), (60, 40, None), (2, 10, None), (0, 0, None)],
)
def test_stranded_column_needs_a_clear_bias_and_enough_pairs(first, second, expected):
    assert rules.stranded_column(first, second) == expected


@pytest.mark.parametrize(
    ("command", "columns"),
    [
        ("awk '{print $1, $2}' t", {2}),
        ("awk '{print $1, $4}' t", {4}),
        ("awk '{s += $3; t += $4} END {print s, t}' t", {3, 4}),
        ("awk '{print ${3}}' t", {3}),
        ("cut -f1,3 t", {3}),
        ("cut -f 1,4 t", {4}),
        ("cut -f1-4 t", {2, 3, 4}),
        ("cut --fields=1,2 t", {2}),
        ("python3 -c 'print(1)'", set()),
    ],
)
def test_columns_named_in_awk_and_cut(command, columns):
    assert rules.columns_in_code(command) == columns


@pytest.mark.parametrize(
    ("tool", "text", "expected"),
    [
        ("head", "head -5 t", False),
        ("cat", "cat t", False),
        ("wc", "wc -l t", False),
        ("grep", "grep N_ambiguous t", False),
        ("awk", "awk {print $3}", True),
        ("cut", "cut -f1,4", True),
        ("sort", "sort -k4,4nr", True),
        ("python3", "rows = [l.split('\\t') for l in open(f)]", True),
        ("python3", "print(open(f).read())", False),
        ("awk", "awk END{print NR}", False),
    ],
)
def test_which_commands_choose_a_column(tool, text, expected):
    assert rules.reads_a_column(tool, text) is expected


def test_dutp_table_read_through_the_wrong_column_fails_with_the_right_one(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(DUTP)
    for column_used in (2, 3):
        decision, result = run(
            "awk", [f"{{print $1, ${column_used}}}", "ReadsPerGene.out.tab"], tmp_path
        )
        assert result.verdict == "fail" and result.code == "wrong_column"
        assert "column 4" in result.fix[0] and "second_read_strand" in result.fix[0]
        assert decision.action == "block"
    _, ok = run("awk", ["{print $1, $4}", "ReadsPerGene.out.tab"], tmp_path)
    assert ok.verdict == "pass"
    _, cut = run("cut", ["-f1,4", "ReadsPerGene.out.tab"], tmp_path)
    assert cut.verdict == "pass"
    _, bad_cut = run("cut", ["-f1,3", "ReadsPerGene.out.tab"], tmp_path)
    assert bad_cut.verdict == "fail"


def test_forward_table_wants_column_3(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(FORWARD)
    assert run("awk", ["{print $1,$3}", "ReadsPerGene.out.tab"], tmp_path)[1].verdict == "pass"
    assert run("awk", ["{print $1,$4}", "ReadsPerGene.out.tab"], tmp_path)[1].verdict == "fail"
    assert run("awk", ["{print $1,$2}", "ReadsPerGene.out.tab"], tmp_path)[1].verdict == "fail"


def test_unstranded_and_tiny_tables_are_silent(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(UNSTRANDED)
    for column_used in (2, 3, 4):
        assert (
            run("awk", [f"{{print $1, ${column_used}}}", "ReadsPerGene.out.tab"], tmp_path)[
                1
            ].verdict
            == "pass"
        )
    (tmp_path / "tiny_ReadsPerGene.out.tab").write_text(table([("g1", 5, 0, 5), ("g2", 3, 1, 2)]))
    assert (
        run("awk", ["{print $1, $2}", "tiny_ReadsPerGene.out.tab"], tmp_path)[1].verdict == "pass"
    )


def test_viewers_and_inspections_never_fire(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(DUTP)
    for tool, argv in (
        ("head", ["-8", "ReadsPerGene.out.tab"]),
        ("cat", ["ReadsPerGene.out.tab"]),
        ("wc", ["-l", "ReadsPerGene.out.tab"]),
        ("grep", ["N_noFeature", "ReadsPerGene.out.tab"]),
        ("awk", ["END{print NR}", "ReadsPerGene.out.tab"]),
        ("awk", ["{a+=$3; b+=$4} END{print a, b}", "ReadsPerGene.out.tab"]),  # compares columns
    ):
        assert run(tool, argv, tmp_path)[1].verdict == "pass", (tool, argv)


def test_declaration_must_match_the_data_and_replaces_reading_the_code(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(DUTP)
    argv = ["{print $1, $3}", "ReadsPerGene.out.tab"]
    assert run("awk", argv, tmp_path, {KEY: "second_read_strand"})[1].verdict == "pass"
    wrong = run("awk", argv, tmp_path, {KEY: "first_read_strand"})[1]
    assert wrong.verdict == "fail" and wrong.code == "declared_column_contradicts_data"
    assert run("awk", argv, tmp_path, {KEY: "unstranded"})[1].verdict == "fail"
    (tmp_path / "t.py").write_text("rows = [l.split() for l in open('ReadsPerGene.out.tab')]\n")
    assert (
        run("python3", ["t.py", "ReadsPerGene.out.tab"], tmp_path, {KEY: "second_read_strand"})[
            1
        ].verdict
        == "pass"
    )


def test_a_column_the_code_does_not_show_only_warns(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(DUTP)
    (tmp_path / "t.py").write_text("rows = [l.split() for l in open('ReadsPerGene.out.tab')]\n")
    decision, result = run("python3", ["t.py", "ReadsPerGene.out.tab"], tmp_path)
    assert result.verdict == "unknown" and decision.action == "warn"
    assert "column 4" in result.fix[0]


def test_other_files_and_malformed_tables_are_left_alone(tmp_path):
    (tmp_path / "SJ.out.tab").write_text("chr1\t100\t200\t1\t1\t1\n")
    assert run("awk", ["{print $2}", "SJ.out.tab"], tmp_path)[1] is None
    (tmp_path / "odd_ReadsPerGene.out.tab").write_text("not\ta\ttable\n")
    assert run("awk", ["{print $2}", "odd_ReadsPerGene.out.tab"], tmp_path)[1].verdict == "pass"
    (tmp_path / "empty_ReadsPerGene.out.tab").write_text("")
    assert run("awk", ["{print $2}", "empty_ReadsPerGene.out.tab"], tmp_path)[1].verdict == "pass"


def test_a_table_inside_a_pipeline_string_is_found(tmp_path):
    (tmp_path / "ReadsPerGene.out.tab").write_text(DUTP)
    decision, result = run("bash", ["-c", "cut -f1,2 ReadsPerGene.out.tab | tail -n +5"], tmp_path)
    assert result.verdict == "fail"


# ---- F31 workspaces and STAR ------------------------------------------------------------------


@pytest.fixture(scope="module")
def f31(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f31")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        task = build_task("F31", 3, twin, work, tools, "implicit")
        star_map(
            work,
            tools,
            ["reads_1.fastq", "reads_2.fastq"],
            "run_",
            ("--quantMode", "GeneCounts", "--outSAMtype", "None"),
        )
        out[twin] = (task, work)
    return out


def test_star_puts_the_pairs_in_column_3_for_forward_and_column_4_for_dutp(f31):
    for twin, index in (("clean", 1), ("faulted", 2)):
        task, work = f31[twin]
        counts = parse_counts((work / "run_ReadsPerGene.out.tab").read_text())
        assert column(counts, index) == task.truth["expected"]
        other = column(counts, 3 - index)
        assert sum(other.values()) < sum(task.truth["expected"].values()) * 0.4
        assert column(counts, 0) != task.truth["expected"]  # the shared exon costs column 2 pairs


def test_the_rule_reads_the_real_table_for_each_library(f31):
    for twin, right in (("clean", 3), ("faulted", 4)):
        _, work = f31[twin]
        for used in (2, 3, 4):
            verdict = run("awk", [f"{{print $1, ${used}}}", "run_ReadsPerGene.out.tab"], work)[
                1
            ].verdict
            assert verdict == ("pass" if used == right else "fail"), (twin, used)


# ---- admission ---------------------------------------------------------------------------------


def test_admission(tools, tmp_path):
    seeds = max(10, ADMISSION_SEEDS // 2)
    stats = measure(RULE.id, [SPEC], range(seeds), tools, tmp_path)
    assert stats.admitted() and stats.conditional
    assert stats.cells[("right", True)].n >= MIN_RIGHT_CALLS * seeds // ADMISSION_SEEDS
    assert stats.cells[("right", True)].failed == 0 and stats.cells[("right", False)].failed == 0
    assert stats.killed() > 0 and all(c.crashed == 0 for c in stats.cells.values())
    assert promotion_is_consistent(stats)
    # undeclared reads of the wrong column that the code shows (awk and cut): every one is caught
    assert stats.cells[("wrong", False)].failed >= seeds * 8
