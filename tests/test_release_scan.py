"""The publishable tree carries no private paths, addresses, keys or unpublished-work names."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "tools" / "release_scan.py"


def load_scanner():
    spec = importlib.util.spec_from_file_location("release_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Strings that the scanner must catch, split so that this file does not contain them whole.
HOME = "/gpfs/" + "home/someone/x"
PROJECT = "/projects/" + "p12345/work"
MAIL = "a.b@" + "example.org"
KEY = "sk-" + "live-" + "a" * 24
LINK = "claude.ai/code/" + "session_01abc"
# What only a release scan reports: unpublished documents, incident ids, private environments.
PREP = "see docs/" + "research/x.md, .lo" + "cal/tools.json, GEN-" + "24 and bio-" + "sc"
# Owner entries for these tests are made up: the real ones live in an untracked file.
DENY = "fictional-user\nre:zz[c]ode\nallow:zzcoded\nallow:azzcode\n"


def test_the_scanner_finds_what_it_should(tmp_path):
    scanner = load_scanner()
    bad = tmp_path / "bad.txt"
    bad.write_text(" ; ".join([HOME, PROJECT, MAIL, KEY, LINK]) + "\n")
    ok = tmp_path / "ok.txt"
    ok.write_text(
        "noreply@anthropic.com 123+name@users.noreply.github.com sk-test-0123456789abcdefghij\n"
        "disk-usage-analyzer-0123456789 /opt/tools/bin /home\n"
    )

    hits = scanner.scan([bad], scanner.Denylist())
    assert {label for _, _, label, _ in hits} == {
        "home path",
        "project path",
        "e-mail address",
        "api key",
        "private link",
    }
    assert scanner.scan([ok], scanner.Denylist()) == []


def test_release_prep_patterns_are_reported_only_on_request(tmp_path):
    scanner = load_scanner()
    note = tmp_path / "note.txt"
    note.write_text(PREP + "\n")

    assert scanner.scan([note], scanner.Denylist()) == []  # not what the tree test asks for
    hits = scanner.scan([note], scanner.Denylist(), release=True)
    assert [(label, found) for _, _, label, found in hits] == [
        ("unpublished document", "docs/" + "research/"),
        ("unpublished document", ".lo" + "cal/"),
        ("incident id", "GEN-" + "24"),
        ("private environment", "bio-" + "sc"),
    ]
    assert scanner.scan([SCANNER], scanner.Denylist(), release=True) == []  # itself included
    assert scanner.scan([Path(__file__)], scanner.Denylist(), release=True) == []


def test_owner_entries_come_from_a_denylist_that_is_not_in_the_tree(tmp_path):
    scanner = load_scanner()
    denylist = scanner.parse_denylist(DENY)
    text = tmp_path / "notes.txt"
    text.write_text(
        "written by Fictional-User\n"  # any case
        "see the ZZCODE notes and zzcode_v2 and zzcode-x\n"  # substring, any case
        "a zzcoded value and the azzcode word\n"  # inside an allowed word: no hit
        "nothing here\n"
    )

    hits = [(n, found) for _, n, _, found in scanner.scan([text], denylist)]

    assert hits == [(1, "Fictional-User"), (2, "ZZCODE"), (2, "zzcode"), (2, "zzcode")]
    # without the entries the same file is clean: the generic patterns know nothing of the owner
    assert scanner.scan([text], scanner.Denylist()) == []
    assert scanner.load_denylist(tmp_path / "missing.txt") == scanner.Denylist()


def test_an_allowed_word_only_covers_its_own_match():
    scanner = load_scanner()
    denylist = scanner.parse_denylist("code\nallow:decode\n")

    assert denylist.find("decode the code") == ["code"]  # the second one is not inside 'decode'
    assert denylist.find("DECODE") == []
    assert denylist.find("Barcodes") == ["code"]


def test_the_path_of_a_file_is_scanned_too(tmp_path):
    scanner = load_scanner()
    denylist = scanner.parse_denylist(DENY)

    hits = scanner.scan_text("docs/fictional-user-notes.md", "clean text\n", denylist)
    assert hits == [("docs/fictional-user-notes.md", 0, "owner entry", "fictional-user")]
    # a file outside the repository is named by an absolute path, which says nothing
    assert scanner.scan_text(str(tmp_path / "fictional-user.md"), "clean\n", denylist) == []


def test_the_scanner_does_not_exempt_itself(tmp_path):
    scanner = load_scanner()
    # the old scanner skipped any file called release_scan.py
    copy = tmp_path / "release_scan.py"
    copy.write_text(SCANNER.read_text() + f'\nLEAK = "{HOME}"\n')

    hits = scanner.scan([copy], scanner.Denylist())
    assert [(h[0], h[2]) for h in hits] == [(str(copy), "home path")]
    assert hits[0][1] == len(copy.read_text().splitlines())
    # and the real one matches none of its own patterns, with or without the owner's entries
    assert scanner.scan([SCANNER], scanner.Denylist()) == []
    assert scanner.scan([SCANNER]) == []
    assert scanner.scan([Path(__file__)]) == []


def test_text_that_is_not_utf8_is_still_scanned(tmp_path):
    scanner = load_scanner()
    path = tmp_path / "latin.txt"
    path.write_bytes(b"caf\xe9 " + HOME.encode() + b"\n")
    binary = tmp_path / "blob.dat"
    binary.write_bytes(b"\0\1\2" + HOME.encode())

    assert [h[2] for h in scanner.scan([path], scanner.Denylist())] == ["home path"]
    assert scanner.scan([binary], scanner.Denylist()) == []


def git(repo: Path, *args: str, email: str = "someone@" + "example.org") -> None:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "A Name",
        "GIT_COMMITTER_NAME": "A Name",
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_EMAIL": email,
    }
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_committed_tree_and_the_commit_metadata_are_scanned(tmp_path):
    scanner = load_scanner()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    assert not scanner.has_commit(repo)
    assert scanner.scan_committed(repo, scanner.Denylist()) == []  # nothing committed yet
    assert scanner.scan_metadata(repo, scanner.Denylist()) == []

    (repo / "a.txt").write_text(f"fine\nleak: {HOME}\n")
    (repo / "dir").mkdir()
    (repo / "dir" / "fictional-user.txt").write_text("fine\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", f"first\n\nsee {LINK}\n")
    git(repo, "tag", "-a", "-m", "frozen by fictional-user", "stage2-freeze")
    assert scanner.has_commit(repo)

    denylist = scanner.parse_denylist(DENY)
    files = {(n, ln, label) for n, ln, label, _ in scanner.scan_committed(repo, denylist)}
    assert files == {
        ("a.txt", 2, "home path"),
        ("dir/fictional-user.txt", 0, "owner entry"),
    }
    meta = scanner.scan_metadata(repo, denylist)
    assert {(name.split()[0], label) for name, _, label, _ in meta} == {
        ("commit", "e-mail address"),  # the author and the committer
        ("commit", "private link"),
        ("tag", "owner entry"),
        ("tag", "e-mail address"),
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_clean_history_with_a_noreply_identity_passes(tmp_path):
    scanner = load_scanner()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "ok.txt").write_text("fine\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first", email="1+name" + "@users.noreply.github.com")

    denylist = scanner.parse_denylist(DENY)
    assert scanner.scan_committed(repo, denylist) == []
    assert scanner.scan_metadata(repo, denylist) == []


def test_the_publishable_tree_is_clean():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    scanner = load_scanner()
    assert scanner.scan() == []


def test_the_path_of_a_file_whose_content_is_not_read_is_checked_too(tmp_path):
    scanner = load_scanner()
    scanner.ROOT = tmp_path  # so that the files are named by their path inside the "repository"
    names = ["fictional-user.png", "fictional-user.dat", "fictional-user.bam"]
    files = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"\0\0\0\x01")  # not text, or a suffix the scanner does not open
        files.append(path)

    hits = scanner.scan(files, scanner.parse_denylist(DENY))

    assert sorted(h[0].rpartition("/")[2] for h in hits) == sorted(names)
    assert {h[2] for h in hits} == {"owner entry"} and {h[1] for h in hits} == {0}


def test_a_long_line_without_an_address_is_scanned_in_linear_time(tmp_path):
    import time

    scanner = load_scanner()
    started = time.monotonic()
    hits = scanner.scan_line("x.txt", 1, "a" * 400_000, scanner.Denylist())

    assert hits == [] and time.monotonic() - started < 5


def test_a_sentence_final_full_stop_does_not_turn_an_allowed_address_into_a_hit():
    scanner = load_scanner()

    assert scanner.scan_line("x.md", 1, "write to noreply@anthropic.com.", scanner.Denylist()) == []
    assert scanner.scan_line("x.md", 1, "write to " + MAIL + ".", scanner.Denylist())


def test_a_key_variable_with_a_value_is_flagged_under_its_current_and_its_former_name(tmp_path):
    scanner = load_scanner()
    env = tmp_path / "env.txt"
    env.write_text(
        "BIOWARDEN_" + "OPENROUTER_API_KEY=abc123\n"
        "HARNESS_" + "ZAI_API_KEY = abc123\n"
        "BIOWARDEN_" + "ZAI_API_KEY=\n"  # an empty template line is no leak
    )

    hits = scanner.scan([env], scanner.Denylist())

    assert [(number, label) for _, number, label, _ in hits] == [(1, "api key"), (2, "api key")]


def test_a_fastq_read_name_is_not_an_address(tmp_path):
    scanner = load_scanner()
    reads = tmp_path / "reads.json"
    reads.write_text(r"\n@SRR14781203.2 1 length=75\n@SRR14781203.3 2 name@lab" + ".org\n")

    hits = scanner.scan([reads], scanner.Denylist())

    assert [(label, found) for _, _, label, found in hits] == [
        ("e-mail address", "name@lab" + ".org")
    ]
