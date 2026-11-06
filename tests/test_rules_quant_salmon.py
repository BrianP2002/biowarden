"""Rules salmon.duplicate_transcripts (call on `salmon index`) and salmon.quant_duplicates (fact).

The call rule is checked on hand-written FASTA files (plain, gzip, several files) and on the F45
workspaces built with the real tools; the fact rule on quant folders written here (cmd_info.json,
duplicate_clusters.tsv, quant.sf) and on real salmon 2.8.0 runs. salmon costs about 2 s per index
build, so the admission tests use 5 seeds.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from test_rules_quant_common import engine_for, outcome, quant_tools, workspace

from biowarden.bench.admission import measure
from biowarden.engine import Call, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import quant_salmon as qs
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, promotion_is_consistent

DUP, QUANT = "salmon.duplicate_transcripts", "salmon.quant_duplicates"
SEEDS = min(ADMISSION_SEEDS, 5)
INDEX = ["index", "-t", "transcripts.fa", "-i", "idx"]


@pytest.fixture(scope="module")
def tools():
    return quant_tools()


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("quant_salmon")


@pytest.fixture(scope="module")
def eng():
    return engine_for("quant_salmon")


def write_fasta(path: Path, records: dict[str, str], wrap: int = 0) -> Path:
    text = ""
    for name, seq in records.items():
        text += f">{name} some description\n"
        text += (
            "\n".join(seq[i : i + wrap] for i in range(0, len(seq), wrap)) + "\n"
            if wrap
            else seq + "\n"
        )
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_declares_a_call_rule_on_index_and_a_fact_rule_on_quant_tables(spec):
    call_rule, fact_rule = spec.rules
    assert (call_rule.id, call_rule.kind, call_rule.tier) == (DUP, "call", "T1")
    assert call_rule.applies.tools == ["salmon"] and call_rule.applies.subcommands == ["index"]
    assert (fact_rule.id, fact_rule.kind, fact_rule.tier) == (QUANT, "fact", "T1")
    for rule in spec.rules:
        assert rule.slot == "completeness" and rule.check in known_checkers()
        assert (rule.declare.key, rule.declare.values, rule.declare.required) == (
            "duplicate_transcripts",
            ["keep", "drop"],
            False,
        )
    assert spec.declaration_schema()["required"] == []


# ---- salmon.duplicate_transcripts --------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "declared", "verdict", "code"),
    [
        (INDEX, None, "fail", "duplicates_collapsed"),
        ([*INDEX, "-p", "1"], None, "fail", "duplicates_collapsed"),
        (
            ["index", "--transcripts", "transcripts.fa", "--index", "idx"],
            None,
            "fail",
            "duplicates_collapsed",
        ),
        (
            ["index", "--transcripts=transcripts.fa", "--index=idx"],
            None,
            "fail",
            "duplicates_collapsed",
        ),
        (["index", "-ttranscripts.fa", "-i", "idx"], None, "fail", "duplicates_collapsed"),
        (["-q", *INDEX], None, "fail", "duplicates_collapsed"),
        ([*INDEX, "--keepDuplicates"], None, "pass", None),
        ([*INDEX, "--keepDuplicates"], {"duplicate_transcripts": "keep"}, "pass", None),
        (
            [*INDEX, "--keepDuplicates"],
            {"duplicate_transcripts": "drop"},
            "fail",
            "declared_drop_but_kept",
        ),
        (INDEX, {"duplicate_transcripts": "drop"}, "pass", None),
        (INDEX, {"duplicate_transcripts": "keep"}, "fail", "duplicates_collapsed"),
    ],
)
def test_duplicate_transcripts_verdicts(tmp_path, eng, argv, declared, verdict, code):
    write_fasta(
        tmp_path / "transcripts.fa",
        {"a": "ACGTACGTAGCT" * 5, "b": "TTGACCA" * 9, "c": "ACGTACGTAGCT" * 5},
    )
    _, result = outcome(eng, DUP, "salmon", argv, tmp_path, declared)
    assert result is not None and result.verdict == verdict, result
    if code:
        assert result.code == code


def test_the_message_names_the_pair_and_the_fix_comes_first(tmp_path, eng):
    write_fasta(
        tmp_path / "transcripts.fa", {"tx1": "ACGT" * 20, "tx9": "acgt" * 20, "tx2": "GGCC" * 20}
    )
    decision, result = outcome(eng, DUP, "salmon", INDEX, tmp_path)
    assert "tx9 = tx1" in result.message and "1 transcript(s)" in result.message
    assert result.fix[0] == "add --keepDuplicates to salmon index to keep every transcript"
    assert render_block(decision, eng.retry_cap, eng.style).startswith("FIX: add --keepDuplicates")


def test_no_duplicates_is_silent_with_any_declaration_and_other_subcommands_are_ignored(
    tmp_path, eng
):
    write_fasta(tmp_path / "transcripts.fa", {"a": "ACGT" * 20, "b": "GGCC" * 20}, wrap=60)
    for declared in (None, {"duplicate_transcripts": "keep"}, {"duplicate_transcripts": "drop"}):
        _, result = outcome(eng, DUP, "salmon", INDEX, tmp_path, declared)
        assert result.verdict == "pass"
    _, quant = outcome(
        eng, DUP, "salmon", ["quant", "-i", "idx", "-l", "A", "-r", "x.fq", "-o", "q"], tmp_path
    )
    assert quant is None


def test_wrapped_gzip_and_several_fasta_files(tmp_path, eng):
    seq = "ACGTTGCA" * 30
    write_fasta(tmp_path / "t1.fa.gz", {"a": seq, "b": "GGAT" * 40}, wrap=70)
    write_fasta(tmp_path / "t2.fa", {"c": seq}, wrap=50)  # the same sequence, another wrapping
    _, one = outcome(eng, DUP, "salmon", ["index", "-t", "t1.fa.gz", "-i", "idx"], tmp_path)
    assert one.verdict == "pass"
    _, two = outcome(
        eng, DUP, "salmon", ["index", "-t", "t1.fa.gz", "t2.fa", "-i", "idx"], tmp_path
    )
    assert two.verdict == "fail" and "c = a" in two.message
    _, gen = outcome(eng, DUP, "salmon", ["index", "-t", "missing.fa", "-i", "idx"], tmp_path)
    assert gen.verdict == "pass"  # nothing on disk to look at


def test_empty_sequences_are_not_duplicates_of_each_other(tmp_path, eng):
    (tmp_path / "transcripts.fa").write_text(">a\n>b\nACGT\n")
    _, result = outcome(eng, DUP, "salmon", INDEX, tmp_path)
    assert result.verdict == "pass"


def test_the_family_workspaces(tools, tmp_path, eng):
    clean, _ = workspace(tools, tmp_path, "F45", 0, "clean")
    faulted, _ = workspace(tools, tmp_path, "F45", 0, "faulted")
    assert outcome(eng, DUP, "salmon", INDEX, clean)[1].verdict == "pass"
    _, bad = outcome(eng, DUP, "salmon", INDEX, faulted)
    assert bad.verdict == "fail" and "tx06 = tx02" in bad.message
    assert outcome(eng, DUP, "salmon", [*INDEX, "--keepDuplicates"], faulted)[1].verdict == "pass"


def test_the_checker_reads_the_fasta_so_a_t0_context_refuses(tmp_path, spec):
    write_fasta(tmp_path / "transcripts.fa", {"a": "ACGT" * 20})
    rule = spec.rules[0]
    call = Call("salmon", tuple(INDEX), tmp_path)
    check = known_checkers()["salmon_duplicate_transcripts"]
    with pytest.raises(TierViolation):
        check(call, Match(rule), Ctx(tmp_path, "T0"))
    assert check(call, Match(rule), Ctx(tmp_path, "T1")).verdict == "pass"


# ---- salmon.quant_duplicates (fact) ------------------------------------------------------------


def fake_quant(
    base: Path, folder: str, index: str, kept: list[str], gone: list[str], cwd_rel: bool = True
):
    """A quant folder next to an index folder: clusters list `gone`, quant.sf has `kept` rows."""
    idx = base / index
    idx.mkdir(exist_ok=True)
    clusters = "RetainedRef\tDuplicateRef\n" + "".join(f"{kept[0]}\t{g}\n" for g in gone)
    (idx / "duplicate_clusters.tsv").write_text(clusters)
    out = base / folder
    out.mkdir(exist_ok=True)
    (out / "cmd_info.json").write_text(json.dumps({"index": index if cwd_rel else str(idx)}))
    (out / "quant.sf").write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\n"
        + "".join(f"{t}\t9\t9\t1\t5\n" for t in kept)
    )
    (out / "quant.genes.sf").write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\ng1\t9\t9\t1\t10\n"
    )
    return out


SUM_GENES = ["{s+=$5} END{print s}", "q/quant.genes.sf"]


@pytest.mark.parametrize(
    ("tool", "argv", "declared", "verdict"),
    [
        ("awk", SUM_GENES, None, "fail"),
        ("awk", ["{s+=$5} END{print s}", "q/quant.sf"], None, "fail"),
        ("python3", ["-c", "print(open('q/quant.genes.sf').read())"], None, "fail"),
        ("Rscript", ["-e", "read.delim('q/quant.sf')"], None, "fail"),
        ("bash", ["-c", "awk '{s+=$5}' q/quant.sf"], None, "fail"),
        ("awk", SUM_GENES, {"duplicate_transcripts": "keep"}, "fail"),
        ("awk", SUM_GENES, {"duplicate_transcripts": "drop"}, "pass"),
        (
            "awk",
            ['{ if ($1 == "tx6") s += $5 }', "q/quant.sf"],
            None,
            "pass",
        ),  # names the removed one
        ("python3", ["-c", "open('idx/duplicate_clusters.tsv'); open('q/quant.sf')"], None, "pass"),
        ("head", ["-3", "q/quant.sf"], None, "pass"),
        ("cat", ["q/quant.genes.sf"], None, "pass"),
        ("wc", ["-l", "q/quant.sf"], None, "pass"),
        ("cut", ["-f1,5", "q/quant.sf"], None, "pass"),
    ],
)
def test_quant_duplicates_verdicts(tmp_path, eng, tool, argv, declared, verdict):
    fake_quant(tmp_path, "q", "idx", ["tx1", "tx2"], ["tx6"])
    _, result = outcome(eng, QUANT, tool, argv, tmp_path, declared)
    if verdict == "pass":
        assert result is None or result.verdict == "pass", result
    else:
        assert result is not None and result.verdict == "fail", result


def test_a_table_whose_duplicates_were_kept_is_silent(tmp_path, eng):
    """The clusters file also lists duplicates that --keepDuplicates retained: quant.sf has them."""
    fake_quant(tmp_path, "q", "idx", ["tx1", "tx2", "tx6"], ["tx6"])
    _, result = outcome(eng, QUANT, "awk", SUM_GENES, tmp_path)
    assert result.verdict == "pass"


def test_index_paths_are_resolved_from_the_call_the_parent_or_as_absolute(tmp_path, eng):
    work = tmp_path / "work"
    (work / "runs").mkdir(parents=True)
    fake_quant(work, "q", "idx", ["tx1"], ["tx6"])  # relative to the working directory
    assert outcome(eng, QUANT, "awk", SUM_GENES, work)[1].verdict == "fail"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    fake_quant(elsewhere, "q", "idx", ["tx1"], ["tx6"], cwd_rel=False)  # absolute in cmd_info.json
    assert (
        outcome(eng, QUANT, "awk", ["{s+=$5}", str(elsewhere / "q/quant.sf")], work)[1].verdict
        == "fail"
    )
    lonely = tmp_path / "lonely"
    (lonely / "q").mkdir(parents=True)
    (lonely / "q" / "quant.sf").write_text("Name\tLength\tEffectiveLength\tTPM\tNumReads\n")
    assert (
        outcome(eng, QUANT, "awk", ["{s+=$5}", "q/quant.sf"], lonely)[1].verdict == "pass"
    )  # no cmd_info
    (lonely / "q" / "cmd_info.json").write_text("not json")
    assert outcome(eng, QUANT, "awk", ["{s+=$5}", "q/quant.sf"], lonely)[1].verdict == "pass"
    (lonely / "q" / "cmd_info.json").write_text(json.dumps({"index": 5}))
    assert outcome(eng, QUANT, "awk", ["{s+=$5}", "q/quant.sf"], lonely)[1].verdict == "pass"


def test_real_salmon_runs_on_the_family_workspace(tools, tmp_path, eng):
    work, _ = workspace(tools, tmp_path, "F45", 1, "faulted")
    for name, keep in (("default", False), ("keep", True)):
        idx = ["salmon", "index", "-t", "transcripts.fa", "-i", f"idx_{name}", "-p", "1"]
        assert tools.run([*idx, *(["--keepDuplicates"] if keep else [])], work).returncode == 0
        quant = ["salmon", "quant", "-i", f"idx_{name}", "-l", "A", "-1", "reads_1.fastq"]
        assert (
            tools.run(
                [*quant, "-2", "reads_2.fastq", "-o", f"q_{name}", "-p", "1"], work
            ).returncode
            == 0
        )
    _, default = outcome(eng, QUANT, "awk", ["{s+=$5}", "q_default/quant.sf"], work)
    assert default.verdict == "fail" and "tx06" in default.message
    assert "q_default" not in default.message or "quant.sf" in default.message
    _, kept = outcome(eng, QUANT, "awk", ["{s+=$5}", "q_keep/quant.sf"], work)
    assert kept.verdict == "pass"
    removed = qs.removed_duplicates(Ctx(work, "T1"), work / "q_default" / "quant.sf", work)
    assert removed == ["tx06"]


# ---- admission ---------------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", [DUP, QUANT])
def test_admission(spec, tools, tmp_path, rule_id):
    stats = measure(rule_id, [spec], range(SEEDS), tools, tmp_path)
    needed = max(1, 200 * SEEDS // 300)
    assert stats.conditional and stats.admitted() and stats.killed() > 0
    assert stats.cells[("right", True)].n >= needed and stats.cells[("right", True)].failed == 0
    assert stats.cells[("right", False)].n >= needed and stats.cells[("right", False)].failed == 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    assert (
        stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    )  # every wrong call
    assert promotion_is_consistent(stats)
