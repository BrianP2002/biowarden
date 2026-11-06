"""Rule matrix.log_scale_state: the sniffer, the checker on real argv, the spec, admission."""

from __future__ import annotations

import gzip
import math
import random
import shutil
from pathlib import Path

import pytest

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv
from biowarden.engine import Call, Engine, Match, known_checkers, load_spec
from biowarden.engine.checks import Ctx, TierViolation
from biowarden.engine.render import render_block
from biowarden.engine.rules.expression_matrix import (
    GREY_HIGH,
    LOG_MAX,
    MAX_ROWS,
    MIN_ROWS,
    sniff,
)
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
RULE = "matrix.log_scale_state"
SPEC = load_packaged("expression_matrix")
RULE_SPEC = SPEC.rules[0]


# -- synthetic tables ---------------------------------------------------------------------------


def counts(n: int = 40, k: int = 4, seed: int = 1) -> tuple[list[str], list[list[int]]]:
    rng = random.Random(seed)
    ids = [f"ENSG{1000 + i:011d}" for i in range(n)]
    data = [[int(rng.lognormvariate(4.5, 1.6)) for _ in range(k)] for _ in ids]
    data[0] = [60000, 82000, 71000, 90000][:k] + [50000] * max(k - 4, 0)  # one deep gene
    return ids, data


def write(path: Path, rows: list[list[object]], sep: str = "\t") -> Path:
    path.write_text("".join(sep.join(str(c) for c in row) + "\n" for row in rows))
    return path


def table(
    path: Path,
    kind: str,
    n: int = 40,
    k: int = 4,
    sep: str = "\t",
    corner: str = "gene",
    prefix: str = "S",
) -> Path:
    ids, data = counts(n, k)
    totals = [sum(row[j] for row in data) for j in range(k)]
    if kind == "raw":
        body = data
    elif kind == "log":
        body = [[f"{math.log2(c + 1):.4f}" for c in row] for row in data]
    elif kind == "cpm":
        body = [[f"{c / totals[j] * 1e6:.2f}" for j, c in enumerate(row)] for row in data]
    else:
        raise ValueError(kind)
    header = [corner, *[f"{prefix}{j + 1}" for j in range(k)]]
    return write(path, [header, *[[i, *row] for i, row in zip(ids, body, strict=True)]], sep)


def engine(real: bool = False) -> Engine:
    return Engine([SPEC], mode="block", retry_cap=10**9)


def run(tmp_path: Path, tool: str, argv: list[str], declared: dict[str, str] | None = None):
    decision = engine().evaluate(Call(tool, tuple(argv), tmp_path, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == RULE]
    return decision, (own[0].result if own else None)


AWK_MEAN = "NR>1{s=0;for(i=2;i<=NF;i++)s+=$i; if(s/(NF-1)>100)n++} END{print n}"


# -- the spec ------------------------------------------------------------------------------------


def test_spec_is_a_valid_conditional_t1_fact_rule_with_a_registered_checker():
    assert SPEC.name == "expression_matrix" and [r.id for r in SPEC.rules] == [RULE]
    assert (RULE_SPEC.kind, RULE_SPEC.tier, RULE_SPEC.slot) == ("fact", "T1", "matrix_state")
    assert RULE_SPEC.declare is not None and RULE_SPEC.declare.required is False
    assert RULE_SPEC.declare.key == "matrix_state"
    assert RULE_SPEC.declare.values == ["raw_counts", "log_transformed", "normalized"]
    assert RULE_SPEC.check in known_checkers()
    assert load_spec(ROOT / "src/biowarden/engine/specs/expression_matrix.yaml") == SPEC
    assert SPEC.declaration_schema()["required"] == []  # a fact rule never forces a declaration
    Engine([SPEC])  # all checker names resolve


def test_tier_honesty_the_checker_reads_files_so_only_t1_may_run_it(tmp_path):
    path = table(tmp_path / "expr.tsv", "log")
    call = Call("awk", (AWK_MEAN, "expr.tsv"), tmp_path)
    check = known_checkers()[RULE_SPEC.check]
    for tier in ("T0", "T2"):
        with pytest.raises(TierViolation):
            check(call, Match(RULE_SPEC, (path,)), Ctx(tmp_path, tier))
    assert check(call, Match(RULE_SPEC, (path,)), Ctx(tmp_path, "T1")).verdict == "fail"


@pytest.mark.parametrize(
    "tool,argv",
    [
        ("awk", ["head", "expr.tsv"]),
        ("head", ["-3", "expr.tsv"]),
        ("cat", ["expr.tsv"]),
        ("wc", ["-l", "expr.tsv"]),
        ("cut", ["-f1", "expr.tsv"]),
        ("grep", ["-c", "ENSG", "expr.tsv"]),
    ],
)
def test_only_computing_tools_are_watched_not_looking_tools(tmp_path, tool, argv):
    table(tmp_path / "expr.tsv", "log")
    decision, result = run(tmp_path, tool, argv)
    if tool == "awk":
        assert result is not None and result.verdict == "fail"
    else:
        assert decision.outcomes == ()


# -- the sniffer ---------------------------------------------------------------------------------


def lines_of(path: Path) -> list[str]:
    return path.read_text().splitlines()


@pytest.mark.parametrize(
    "kind,state,top",
    [("raw", "raw_counts", 90000), ("log", "log_transformed", 16.5), ("cpm", "normalized", 1e5)],
)
def test_sniff_classifies_the_three_states(tmp_path, kind, state, top):
    profile = sniff(lines_of(table(tmp_path / "m.tsv", kind)), "m.tsv")

    assert profile is not None and profile.state == state
    assert profile.sample_columns == 4 and profile.rows == 40
    assert profile.maximum > (top * 0.5 if state != "log_transformed" else 10)
    assert profile.ambiguous is False


def test_log_matrix_integer_values_do_not_make_it_raw():
    # log2(count+1) of counts 0, 1, 3, 7, 15 prints 0.0000, 1.0000, 2.0000, 3.0000, 4.0000:
    # many cells are whole numbers, a column still shows its fractions
    rows = [["gene", "S1", "S2", "S3"]]
    for i in range(12):
        counts_row = [(0, 1, 3, 7)[i % 4], i + 2, (15, 2, 0, 5)[i % 4]]
        rows.append([f"g{i}", *[f"{math.log2(c + 1):.4f}" for c in counts_row]])
    assert sum(c.endswith(".0000") for r in rows[1:] for c in r[1:]) >= 12
    profile = sniff(["\t".join(map(str, r)) for r in rows], "x.tsv")
    assert profile is not None and profile.state == "log_transformed"


def test_a_minority_of_fractional_columns_among_integer_ones_stays_raw(tmp_path):
    ids, data = counts()
    rows = [["gene", "S1", "S2", "S3", "S4", "frac_mito"]]
    rows += [
        [g, *r, f"{0.01 * (i % 7):.3f}"] for i, (g, r) in enumerate(zip(ids, data, strict=True))
    ]
    profile = sniff(lines_of(write(tmp_path / "c.tsv", rows)), "c.tsv")
    assert profile is not None and profile.state == "raw_counts"


def test_pandas_floats_that_are_whole_numbers_are_still_raw_counts(tmp_path):
    ids, data = counts()
    rows = [
        ["gene", "S1", "S2", "S3", "S4"],
        *[[i, *[f"{c}.0" for c in r]] for i, r in zip(ids, data, strict=True)],
    ]
    profile = sniff(lines_of(write(tmp_path / "c.tsv", rows)), "c.tsv")
    assert profile is not None and profile.state == "raw_counts"


@pytest.mark.parametrize("token", ["NA", "NaN", "", "nan", "N/A"])
def test_missing_values_never_decide_the_state(tmp_path, token):
    for kind, state in (("raw", "raw_counts"), ("log", "log_transformed")):
        path = table(tmp_path / f"{kind}.tsv", kind)
        text = lines_of(path)
        for i in (3, 9, 17):
            cells = text[i].split("\t")
            cells[2] = token
            text[i] = "\t".join(cells)
        profile = sniff(text, "m.tsv")
        assert profile is not None and profile.state == state


def test_annotation_columns_around_the_samples_are_ignored(tmp_path):
    ids, data = counts()
    for kind in ("raw", "log"):
        rows = [["gene_id", "gene_name", "biotype", "Length", "S1", "S2", "S3", "S4"]]
        for i, (g, r) in enumerate(zip(ids, data, strict=True)):
            vals = r if kind == "raw" else [f"{math.log2(c + 1):.4f}" for c in r]
            rows.append([g, f"GENE{i}", "protein_coding", 1000 + 37 * i, *vals])
        profile = sniff(lines_of(write(tmp_path / f"{kind}.tsv", rows)), "x.tsv")
        assert profile is not None
        assert profile.state == ("raw_counts" if kind == "raw" else "log_transformed")
        assert profile.maximum < 1e5 if kind == "raw" else profile.maximum < 20


def test_an_unnamed_integer_annotation_column_does_not_hide_a_log_matrix(tmp_path):
    ids, data = counts()
    rows = [["gene_id", "span", "S1", "S2", "S3", "S4"]]
    for i, (g, r) in enumerate(zip(ids, data, strict=True)):
        rows.append([g, 500 + 11 * i, *[f"{math.log2(c + 1):.4f}" for c in r]])
    profile = sniff(lines_of(write(tmp_path / "m.tsv", rows)), "m.tsv")
    assert profile is not None and profile.state == "log_transformed"
    assert profile.maximum < 20  # the span column (up to 940) did not set the range


def test_delimiters_quoting_comments_and_row_name_headers(tmp_path):
    base = table(tmp_path / "base.tsv", "log")
    text = lines_of(base)
    csv_lines = [
        ",".join(f'"{c}"' if i == 0 or j == 0 else c for j, c in enumerate(r.split("\t")))
        for i, r in enumerate(text)
    ]
    space_lines = [" ".join(r.split("\t")) for r in text]
    r_style = [" ".join(text[0].split("\t")[1:]), *[r.replace("\t", " ") for r in text[1:]]]
    featurecounts = ["# Program:featureCounts v2.0.1; Command: 'x'", *text]
    for name, lines in (
        ("a.csv", csv_lines),
        ("a.txt", space_lines),
        ("a.txt", r_style),
        ("a.tsv", featurecounts),
        ("a.csv", text),  # tab-separated data in a .csv: the delimiter is inferred
    ):
        profile = sniff(lines, name)
        assert profile is not None and profile.state == "log_transformed", name


def test_only_a_bounded_prefix_is_needed(tmp_path):
    path = table(tmp_path / "big.tsv", "log", n=2000)
    profile = sniff(lines_of(path)[: MAX_ROWS + 1], "big.tsv")
    assert profile is not None and profile.rows == MAX_ROWS


@pytest.mark.parametrize(
    "rows",
    [
        # sample sheet: first column samples, clinical numbers
        [["sample", "condition", "age", "weight", "bmi"]]
        + [[f"S{i}", "case", 30 + i, 60.5 + i, 22.1 + i / 10] for i in range(40)],
        # DESeq2 results
        [["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]]
        + [[f"g{i}", 100.5 + i, 1.2, 0.3, 4.0, 0.001, 0.01] for i in range(30)],
        # limma topTable
        [["gene", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B"]]
        + [[f"g{i}", 1.1 + i / 10, 8.2, 3.3, 0.01, 0.04, 1.5] for i in range(30)],
        # proportions (relative abundance)
        [["taxon", "S1", "S2", "S3"]] + [[f"t{i}", 0.1 + i / 100, 0.2, 0.3] for i in range(20)],
        # long format
        [["gene", "sample", "value"]] + [[f"g{i}", "S1", 5.5 + i] for i in range(30)],
        # fewer rows than MIN_ROWS
        [["gene", "S1", "S2", "S3"]] + [[f"g{i}", 1.5, 2.5, 3.5] for i in range(MIN_ROWS - 2)],
        # time series: fractional first column
        [["t", "x1", "x2", "x3"]] + [[i / 10, 1.5 + i, 2.5, 3.5] for i in range(30)],
        # columns of unlike scale (QC table)
        [["lib", "reads1", "rate2", "gc3"]]
        + [[f"L{i}", 12000000.5 + i, 0.5, 41.5] for i in range(30)],
        # ids are numeric, first column is not called an id
        [["x", "s1", "s2", "s3"]] + [[i + 0.5, 1.5 + i, 2.5, 3.5] for i in range(30)],
    ],
)
def test_tables_that_are_not_expression_matrices_are_no_match(tmp_path, rows):
    assert sniff(lines_of(write(tmp_path / "t.tsv", rows)), "t.tsv") is None


def test_the_sniffer_never_raises_on_garbage(tmp_path):
    rng = random.Random(9)
    alphabet = [
        "\t",
        ",",
        ";",
        " ",
        "\n",
        '"',
        "'",
        "#",
        "1",
        "2.5",
        "-3",
        "NA",
        "abc",
        "1e5",
        "S1",
        "",
        "\r",
    ]
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        sniff(text.split("\n"), rng.choice(["a.tsv", "a.csv", "a.txt", "a.tsv.gz", ""]))
    # shaped like a table, with damaged rows
    base = lines_of(table(tmp_path / "m.tsv", "log"))
    for _ in range(500):
        rows = list(base)
        for _ in range(rng.randint(1, 8)):
            i = rng.randrange(len(rows))
            rows[i] = rng.choice(
                ["", "\t\t", "x", rows[i][: rng.randint(0, len(rows[i]))], '"' + rows[i]]
            )
        sniff(rows, "m.tsv")


def test_square_correlation_and_distance_matrices_are_no_match(tmp_path):
    names = [f"S{i}" for i in range(12)]
    rows = [["", *names]]
    for i, a in enumerate(names):
        rows.append([a, *[f"{abs(i - j) * 3.5:.3f}" for j in range(12)]])
    assert sniff(lines_of(write(tmp_path / "d.tsv", rows)), "d.tsv") is None


def test_scaled_matrix_with_negative_values_is_a_guess_not_a_certainty(tmp_path):
    rng = random.Random(3)
    rows = [["gene", "S1", "S2", "S3", "S4"]]
    rows += [[f"g{i}", *[f"{rng.gauss(0, 1):.3f}" for _ in range(4)]] for i in range(30)]
    profile = sniff(lines_of(write(tmp_path / "z.tsv", rows)), "z.tsv")
    assert profile is not None and profile.ambiguous and profile.minimum < 0


def test_gray_zone_is_flagged_ambiguous_and_the_cut_is_between_log_and_normalised(tmp_path):
    for top, state, ambiguous in (
        (19.9, "log_transformed", False),
        (25.0, "log_transformed", True),
        (35.0, "normalized", True),
        (150.0, "normalized", False),
    ):
        rows = [["gene", "S1", "S2", "S3"]]
        rows += [
            [f"g{i}", f"{top * (i + 1) / 20:.3f}", f"{top / 2:.3f}", f"{top / 3 + i / 7:.3f}"]
            for i in range(20)
        ]
        profile = sniff(lines_of(write(tmp_path / "g.tsv", rows)), "g.tsv")
        assert profile is not None and (profile.state, profile.ambiguous) == (state, ambiguous), top
    assert 20 < LOG_MAX < GREY_HIGH


# -- the checker on real argv ---------------------------------------------------------------------

LOG_AWK_VARIANTS = [
    ("awk", ["-F\\t", AWK_MEAN, "expr.tsv"]),
    ("awk", ["-F", "\t", AWK_MEAN, "./expr.tsv"]),
    ("gawk", ["-F\\t", AWK_MEAN, "expr.tsv"]),
    ("awk", [AWK_MEAN, "expr.tsv", "other.txt"]),
    ("python3", ["-c", "print(sum(float(x) for l in open('expr.tsv') for x in l.split()[1:2]))"]),
    ("python3.12", ["-c", "import pandas as pd; d = pd.read_csv('expr.tsv', sep='\\t'); d.mean()"]),
    ("python3", ["-c", 'import csv; r = csv.reader(open("expr.tsv"), delimiter="\\t")']),
    ("perl", ["-F\\t", "-lane", "$s += $F[1]; END { print $s }", "expr.tsv"]),
    ("Rscript", ["-e", 'x <- read.table("expr.tsv", header=TRUE); mean(x$S1)']),
    ("datamash", ["-H", "mean", "2", "<", "expr.tsv"]),
    ("sort", ["-t", "\t", "-k2,2gr", "expr.tsv"]),
    ("sort", ["-nrk2", "expr.tsv"]),
    ("sort", ["--key=2,2n", "expr.tsv"]),
    ("sort", ["-k", "2n", "expr.tsv"]),
    ("sqlite3", [":memory:", ".import expr.tsv t"]),
]


@pytest.mark.parametrize("tool,argv", LOG_AWK_VARIANTS)
def test_an_undeclared_log_matrix_fails_for_every_spelling_of_the_file_and_tool(
    tmp_path, tool, argv
):
    table(tmp_path / "expr.tsv", "log")
    (tmp_path / "other.txt").write_text("not a matrix\n")

    decision, result = run(tmp_path, tool, argv)

    assert result is not None and result.verdict == "fail", (tool, argv)
    assert result.code == "matrix_state_undeclared"
    assert decision.action == "block"


@pytest.mark.parametrize(
    "tool,argv",
    [("sort", ["expr.tsv"]), ("sort", ["-u", "-k1,1", "expr.tsv"]), ("sort", ["-t,", "expr.tsv"])],
)
def test_lexical_sort_does_not_interpret_values(tmp_path, tool, argv):
    table(tmp_path / "expr.tsv", "log")
    _, result = run(tmp_path, tool, argv)
    assert result is not None and result.verdict == "pass"


def test_message_states_the_range_why_it_matters_and_the_back_transform(tmp_path):
    path = table(tmp_path / "expr.tsv", "log")
    values = [float(c) for line in lines_of(path)[1:] for c in line.split("\t")[1:]]
    decision, result = run(tmp_path, "awk", [AWK_MEAN, "expr.tsv"])

    assert result is not None
    assert f"{min(values):.6g} to {max(values):.6g}" in result.message
    assert "expr.tsv" in result.message and "log-transformed" in result.message
    assert "not raw-count statistics" in result.message
    assert "thresholds, means or fold changes" in result.message
    text = render_block(decision)
    assert text.splitlines()[0].startswith("FIX:") and "2^x-1" in text.splitlines()[0]
    assert "--declare matrix_state=log_transformed" in text


def test_normalised_message_names_the_scale_and_offers_the_declaration(tmp_path):
    table(tmp_path / "expr.tsv", "cpm")
    decision, result = run(tmp_path, "awk", [AWK_MEAN, "expr.tsv"])

    assert result is not None and result.verdict == "fail"
    assert "normalised" in result.message and "CPM/TPM/FPKM" in result.message
    assert "--declare matrix_state=normalized" in render_block(decision)


@pytest.mark.parametrize(
    "kind,declared,verdict,code",
    [
        ("raw", None, "pass", "ok"),
        ("raw", "raw_counts", "pass", "ok"),
        ("raw", "normalized", "pass", "ok"),
        ("raw", "log_transformed", "fail", "declared_log_but_integers"),
        ("log", None, "fail", "matrix_state_undeclared"),
        ("log", "log_transformed", "pass", "ok"),
        ("log", "raw_counts", "fail", "declared_raw_not_integer"),
        ("log", "normalized", "fail", "declared_state_mismatch"),
        ("cpm", None, "fail", "matrix_state_undeclared"),
        ("cpm", "normalized", "pass", "ok"),
        ("cpm", "raw_counts", "fail", "declared_raw_not_integer"),
        ("cpm", "log_transformed", "fail", "declared_state_mismatch"),
    ],
)
def test_verdict_table_for_every_state_and_declaration(tmp_path, kind, declared, verdict, code):
    table(tmp_path / "expr.tsv", kind)

    _, result = run(
        tmp_path, "awk", [AWK_MEAN, "expr.tsv"], {"matrix_state": declared} if declared else None
    )

    assert result is not None
    assert (result.verdict, result.code) == (verdict, code)


def test_raw_counts_cost_zero_friction_whatever_the_call(tmp_path):
    table(tmp_path / "expr.tsv", "raw")
    table(tmp_path / "counts.csv", "raw", sep=",")
    gz = tmp_path / "counts.txt.gz"
    gz.write_bytes(gzip.compress((tmp_path / "expr.tsv").read_bytes()))
    for tool, argv in (
        ("awk", ["-F\\t", AWK_MEAN, "expr.tsv"]),
        ("python3", ["-c", "import gzip; gzip.open('counts.txt.gz')"]),
        ("python3", ["-c", "open('counts.csv'); open('expr.tsv')"]),
        ("perl", ["-lane", "print", "expr.tsv"]),
        ("sort", ["-nrk2", "expr.tsv"]),
    ):
        decision, result = run(tmp_path, tool, argv)
        assert result is not None and result.verdict == "pass", (tool, argv)
        assert decision.action == "allow"


def test_gzip_csv_and_space_delimited_files_are_read(tmp_path):
    table(tmp_path / "m.csv", "log", sep=",")
    gz = tmp_path / "m2.tsv.gz"
    gz.write_bytes(gzip.compress((tmp_path / "m.csv").read_text().replace(",", "\t").encode()))
    table(tmp_path / "m3.txt", "cpm", sep=" ")
    for name, code in (("m.csv", "fail"), ("m2.tsv.gz", "fail"), ("m3.txt", "fail")):
        _, result = run(tmp_path, "python3", ["-c", f"open('{name}')"])
        assert result is not None and result.verdict == code, name


def test_unrelated_and_missing_files_are_no_match(tmp_path):
    (tmp_path / "notes.txt").write_text("remember to normalise the data\n" * 30)
    table(tmp_path / "expr.tsv", "log")

    assert run(tmp_path, "awk", ["{print}", "notes.txt"])[1].verdict == "pass"
    assert run(tmp_path, "awk", ["{print}", "absent.tsv"])[0].outcomes == ()
    assert run(tmp_path, "awk", ["{print $1}", "README.md"])[0].outcomes == ()


def test_an_unreadable_file_is_unknown_and_never_blocks(tmp_path):
    (tmp_path / "broken.tsv.gz").write_bytes(b"\x1f\x8b" + b"not really gzip data" * 5)

    decision, result = run(tmp_path, "python3", ["-c", "open('broken.tsv.gz')"])

    assert result is not None and result.verdict == "unknown" and result.code == "unreadable"
    assert decision.action == "warn"


def test_several_files_in_one_call_each_risky_one_needs_the_declaration(tmp_path):
    table(tmp_path / "raw.tsv", "raw")
    table(tmp_path / "log.tsv", "log")
    argv = ["join", "raw.tsv", "log.tsv"]
    code = "import pandas; pandas.read_csv('raw.tsv'); pandas.read_csv('log.tsv')"

    assert run(tmp_path, "python3", ["-c", code])[1].verdict == "fail"
    assert (
        run(tmp_path, "python3", ["-c", code], {"matrix_state": "log_transformed"})[1].verdict
        == "pass"
    )
    assert (
        run(tmp_path, "python3", ["-c", code], {"matrix_state": "raw_counts"})[1].verdict == "fail"
    )
    assert argv  # join itself is not a watched tool


def test_the_contract_data_in_a_risky_state_need_the_declaration_whatever_the_call_computes(
    tmp_path,
):
    # a correct back-transform on log data still fails without the declaration: a fact rule fires
    # on the data, not on the arithmetic (the admission cases do not count this as friction)
    table(tmp_path / "expr.tsv", "log")
    back = "NR>1{s=0;for(i=2;i<=NF;i++)s+=2^$i-1; if(s/(NF-1)>100)n++} END{print n}"

    undeclared = run(tmp_path, "awk", ["-F\\t", back, "expr.tsv"])[1]
    declared = run(
        tmp_path, "awk", ["-F\\t", back, "expr.tsv"], {"matrix_state": "log_transformed"}
    )[1]

    assert undeclared.verdict == "fail" and declared.verdict == "pass"


def test_a_checker_failure_cannot_pass_silently_a_directory_named_like_a_table_is_ignored(tmp_path):
    (tmp_path / "dir.tsv").mkdir()
    decision, _ = run(tmp_path, "awk", ["1", "dir.tsv"])
    assert decision.outcomes == () and decision.action == "allow"


# -- admission over 300 seeds ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not all(shutil.which(t) for t in ("awk", "perl", "python3")):
        pytest.skip("samtools, bcftools, bedtools, awk, perl or python3 are not available")
    return env


def test_admission(tmp_path, tools):
    stats = measure(RULE, [SPEC], range(ADMISSION_SEEDS), tools, tmp_path)

    right_declared = stats.right_declared
    right_undeclared = stats.cells[("right", False)]
    assert stats.admitted()
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert (
        right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    )  # zero friction on raw counts
    assert stats.killed() > 0
    assert stats.recall() is not None and stats.recall() > 0.3
    for cell in stats.cells.values():
        assert cell.crashed == 0 and cell.unknown == 0
    assert promotion_is_consistent(stats)
