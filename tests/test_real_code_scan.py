"""tools/real_code_scan.py: the parsers, the safe unpacking, the bookkeeping and one guarded run."""

from __future__ import annotations

import importlib.util
import io
import re
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "real_code_scan", ROOT / "tools" / "real_code_scan.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["real_code_scan"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan():
    return load()


BEDTOOLS_OUTPUT = """\
    merge.t01...ok
    merge.t02...ok
    merge.t03...1c1
< chr1\t10\t20
---
> chr1\t10\t25
fail
    merge.t04...ok
    merge.t05...
    merge.t06...ok
"""

TOTALS = """\

Number of tests:
    total            .. 1032
    passed           .. 952
    failed           .. 48
    expected failure .. 32
    unexpected pass  .. 0
"""


def test_a_failing_bedtools_test_prints_its_diff_before_the_fail(scan):
    labels = scan.parse_labels(BEDTOOLS_OUTPUT)

    assert labels == {
        "merge.t01": "ok",
        "merge.t02": "ok",
        "merge.t03": "fail",
        "merge.t04": "ok",
        "merge.t05": "unfinished",
        "merge.t06": "ok",
    }


def test_the_totals_of_test_pl(scan):
    assert scan.parse_totals(TOTALS) == {
        "total": 1032,
        "passed": 952,
        "failed": 48,
        "expected failure": 32,
        "unexpected pass": 0,
    }
    assert scan.parse_totals("nothing to see") == {}


def row(action, tool="bedtools", fired=()):
    outcomes = [
        {"rule": r, "code": c, "effective": e, "verdict": "fail", "note": ""} for r, c, e in fired
    ]
    return {"action": action, "tool": tool, "argv": ["merge", "-i", "a.bed"], "outcomes": outcomes}


def test_the_ledger_of_a_guarded_run_is_summarised(scan):
    rows = [
        row("allow"),
        row("warn", fired=[("bedtools.merge_bookended", "merge_adjacent_undeclared", "warn")]),
        row("block", fired=[("bedtools.prec_significant", "declaration_missing", "block")]),
        row("allow", tool="sort"),
    ]

    result = scan.summarize_ledger(rows)

    assert result["calls"] == 4
    assert result["by_tool"] == {"bedtools": 3, "sort": 1}
    assert result["by_action"] == {"allow": 2, "warn": 1, "block": 1}
    assert result["firing"] == {
        "bedtools.merge_bookended|merge_adjacent_undeclared|warn": 1,
        "bedtools.prec_significant|declaration_missing|block": 1,
    }
    assert [x["action"] for x in result["non_allowed"]] == ["warn", "block"]
    merged = scan.merge_ledgers([result, result])
    assert merged["calls"] == 8 and merged["by_action"]["block"] == 2
    assert len(merged["non_allowed"]) == 4


def test_a_blocked_call_keeps_a_bounded_number_of_short_words(scan):
    long_call = {**row("block", fired=[("r", "c", "block")]), "argv": ["x" * 500] * 40}

    kept = scan.summarize_ledger([long_call])["non_allowed"][0]["argv"]

    assert len(kept) == scan.MAX_ARGS and all(len(w) <= scan.MAX_WORD + 1 for w in kept)


def test_the_digest_of_a_tree_follows_names_and_contents(scan, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("one")
    first = scan.tree_digest(tmp_path)

    assert scan.tree_digest(tmp_path) == first
    (tmp_path / "a" / "x.txt").write_text("two")
    assert scan.tree_digest(tmp_path) != first
    (tmp_path / "a" / "x.txt").write_text("one")
    (tmp_path / "a" / "y.txt").write_text("")
    assert scan.tree_digest(tmp_path) != first


def make_tar(path: Path, members: list[tarfile.TarInfo], payload: bytes = b"data") -> None:
    with tarfile.open(path, "w:gz") as tar:
        for info in members:
            tar.addfile(info, io.BytesIO(payload) if info.isfile() else None)


def member(name: str, kind=tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = 4 if kind == tarfile.REGTYPE else 0
    info.linkname = link
    return info


def test_unpacking_stays_below_the_target(scan, tmp_path):
    good = tmp_path / "good.tgz"
    make_tar(good, [member("top/test/a.txt")])
    scan.safe_unpack(good, tmp_path / "out")
    assert (tmp_path / "out" / "top" / "test" / "a.txt").read_text() == "data"

    for bad_member in (
        member("../escape.txt"),
        member("/abs.txt"),
        member("top/link", tarfile.SYMTYPE, "../../outside"),
        member("top/dev", tarfile.CHRTYPE),
    ):
        bad = tmp_path / "bad.tgz"
        make_tar(bad, [bad_member])
        with pytest.raises(ValueError):
            scan.safe_unpack(bad, tmp_path / "out2")


def test_the_pinned_trees_and_urls_are_complete(scan):
    for suite in scan.SUITES.values():
        assert re.fullmatch(r"[0-9a-f]{64}", suite.test_tree), suite.name
        assert suite.url.startswith("https://github.com/")
        assert suite.version in suite.url and suite.top.endswith(suite.version)


def test_a_bedtools_script_loses_its_errexit_line_only(scan, tmp_path):
    test = tmp_path / "bedtools2-2.31.1" / "test" / "merge"
    test.mkdir(parents=True)
    (test / "test-merge.sh").write_text("set -e;\nBT=${BT-x}\nset -e\necho hi\n")
    (test / "a.bed").write_text("chr1\t1\t2\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    workdir, command = scan.prepare("bedtools", "merge", tmp_path, scratch)

    assert command == ["bash", "run.sh"]
    assert (workdir / "run.sh").read_text() == "BT=${BT-x}\nset -e\necho hi\n"
    assert (workdir / "a.bed").is_file()
    assert scan.bedtools_units(tmp_path) == ["merge"]


SAMTOOLS_CMD = """sub cmd
{
    my ($cmd, $args) = @_;
    my ($ret,$out,$err) = _cmd($cmd, $args);
    if ( $ret ) { error("The command failed [$ret]: $cmd\\n", "out:$out\\n", "err:$err\\n"); }
    return $out;
}
"""
BCFTOOLS_CMD = """sub cmd
{
    my ($cmd) = @_;
    my ($ret,$out) = _cmd($cmd);
    if ( $ret ) { error("The command failed: $cmd\\n", $out); }
    return $out;
}
"""


@pytest.mark.parametrize(
    ("suite", "perl"), [("samtools", SAMTOOLS_CMD), ("bcftools", BCFTOOLS_CMD)]
)
def test_the_perl_suites_link_the_program_and_cmd_no_longer_ends_the_script(
    scan, tmp_path, suite, perl
):
    test = tmp_path / f"{suite}-{scan.SUITES[suite].version}" / "test"
    test.mkdir(parents=True)
    (test / "test.pl").write_text(perl)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    workdir, command = scan.prepare(suite, None, tmp_path, scratch)

    patched = (workdir / "test" / "test.pl").read_text()
    assert 'if ( $ret ) { print STDERR "The command failed' in patched
    assert "error(" not in patched and "return $out;" in patched
    assert command[:2] == ["bash", "-c"] and f"command -v {suite}" in command[2]
    assert command[2].endswith("perl test/test.pl")


def test_a_test_pl_without_the_expected_cmd_is_refused(scan, tmp_path):
    test = tmp_path / "samtools-1.24" / "test"
    test.mkdir(parents=True)
    (test / "test.pl").write_text("1;\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(ValueError, match="expected one"):
        scan.prepare("samtools", None, tmp_path, scratch)


def test_a_guarded_run_leaves_a_ledger_and_an_unguarded_one_does_not(scan, tmp_path):
    """awk is one of the programs the shims stand in for, so the call needs no bio tool."""
    work = tmp_path / "ws"
    work.mkdir()
    command = ["bash", "-c", "awk 'BEGIN { print 41 + 1 }' > out.txt"]
    env = scan.tool_environment(None)

    plain = scan.run_command(
        command, workdir=work, guarded=False, env=env, bwrap=None, extra_env={}
    )
    guarded = scan.run_command(
        command, workdir=work, guarded=True, env=env, bwrap=None, extra_env={}
    )

    assert plain.code == 0 and plain.ledger == []
    assert guarded.code == 0 and (work / "out.txt").read_text() == "42\n"
    assert [r["tool"] for r in guarded.ledger] == ["awk"]
    assert guarded.ledger[0]["action"] == "allow"
    assert scan.summarize_ledger(guarded.ledger)["calls"] == 1
