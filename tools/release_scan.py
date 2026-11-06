"""Scan what would be published for things that must not leave the machine.

    python tools/release_scan.py            # exits 1 and lists every hit

Checked: every file that would be committed (tracked, or untracked and not ignored) and its path;
once there is a commit, also `git archive HEAD` and the author, committer and message of every
commit and annotated tag. The generic patterns below cover home and project paths, e-mail
addresses, API-key shapes and private links. What belongs to one person (a user name, the handle
of an e-mail address, an allocation, the names of unpublished work) is not written down here, since
this file is published too: it is read from `release_denylist.txt` in the untracked
`.local` directory.

    # one entry per line; blank lines and lines starting with '#' are ignored
    some-user-name        a plain entry: matched as a substring, ignoring case
    re:some-code[n]ame    a regular expression (ignoring case); the bracket keeps this very line
                          from matching itself
    allow:some-codenames  an ordinary word that contains an entry; a match inside it is ignored

Nothing is exempt, this file included: the patterns are written so that they do not match their
own source. To check each commit, make `python tools/release_scan.py` the pre-commit hook.

`--release` also lists what is not private but does not belong in a public repository: references
to documents that are not published, internal incident ids and the names of private conda
environments. They are for the owner to settle before the first public commit (editing a rule
spec changes its hash, so that is a re-admission), and the test of the tree does not ask for them.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = ROOT / ".local" / "release_denylist.txt"

Hit = tuple[str, int, str, str]  # (file or commit, line, label, matched text); line 0: the path

# A bracket in a pattern keeps the pattern from matching its own text in this file.
PATTERNS = {
    "home path": re.compile(r"/gpfs/hom[e]/|/hom[e]/[a-z][\w.-]*/|/User[s]/\w+"),
    "project path": re.compile(r"/project[s]/p\d{4,}\b"),
    "e-mail address": re.compile(
        r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63})*\.[A-Za-z][\w-]{0,62}"
    ),
    "api key": re.compile(
        r"(?<![\w-])(?:sk-[A-Za-z0-9_-]{16,}|sk-or-[\w-]{8,})|(?:BIOWARDEN|HARNESS)_\w+_API_KEY\s*=\s*\S"
    ),
    "private link": re.compile(r"claude\.ai/(?:code/)?(?:session|artifact)\S*"),
}
RELEASE_PATTERNS = {  # only with --release
    "unpublished document": re.compile(r"docs/researc[h]/|\.loca[l]/"),
    "incident id": re.compile(r"\bGEN-\d{2}\b"),
    "private environment": re.compile(r"\bbio-(?:a|py|struct|sc)\b"),
}
ALLOWED_EMAIL = re.compile(r"noreply@anthropic\.com|[\w.+-]{1,64}@users\.noreply\.github\.com")
FAKE_KEY = re.compile(r"sk-test-")  # fixtures that prove keys are never leaked
SKIP_SUFFIXES = {".gz", ".bam", ".bai", ".csi", ".png", ".pdf", ".pyc"}


class Denylist(NamedTuple):
    """The owner's entries: `names` match as substrings (case-insensitive); a match that lies
    inside one of the `words` (ordinary words that contain an entry) is not a hit."""

    names: tuple[re.Pattern[str], ...] = ()
    words: tuple[re.Pattern[str], ...] = ()

    def find(self, text: str) -> list[str]:
        covered = [m.span() for word in self.words for m in word.finditer(text)]
        found = []
        for name in self.names:
            for m in name.finditer(text):
                start, end = m.span()
                if not any(a <= start and end <= b for a, b in covered):
                    found.append(m.group(0))
        return found


def parse_denylist(text: str) -> Denylist:
    names, words = [], []
    for raw in text.splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("allow:"):
            words.append(re.compile(re.escape(entry[6:].strip()), re.IGNORECASE))
        elif entry.startswith("re:"):
            names.append(re.compile(entry[3:].strip(), re.IGNORECASE))
        else:
            names.append(re.compile(re.escape(entry), re.IGNORECASE))
    return Denylist(tuple(names), tuple(words))


def load_denylist(path: Path | None = None) -> Denylist:
    """The owner's entries; empty (only the generic patterns run) when the file is not there."""
    path = DENYLIST if path is None else path
    try:
        return parse_denylist(path.read_text())
    except OSError:
        return Denylist()


def _name(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def scan_line(
    name: str, number: int, line: str, denylist: Denylist, release: bool = False
) -> list[Hit]:
    patterns = {**PATTERNS, **RELEASE_PATTERNS} if release else PATTERNS
    hits: list[Hit] = []
    for label, pattern in patterns.items():
        for match in pattern.finditer(line):
            found = match.group(0)
            if label == "e-mail address" and ALLOWED_EMAIL.fullmatch(found.rstrip(".-")):
                continue
            if label == "api key" and FAKE_KEY.match(found):
                continue
            hits.append((name, number, label, found))
    hits += [(name, number, "owner entry", found) for found in denylist.find(line)]
    return hits


def scan_text(name: str, text: str, denylist: Denylist, release: bool = False) -> list[Hit]:
    hits = scan_line(name, 0, name, denylist) if not name.startswith("/") else []  # the path
    for number, line in enumerate(text.splitlines(), 1):
        hits += scan_line(name, number, line, denylist, release)
    return hits


def is_text(data: bytes) -> bool:
    return b"\0" not in data[:8192]


def _path_hits(name: str, denylist: Denylist) -> list[Hit]:
    """Hits in the path of a file whose content is not read (line 0, as for any path)."""
    return [] if name.startswith("/") else scan_line(name, 0, name, denylist)


def candidate_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    return [ROOT / name for name in out if name and (ROOT / name).is_file()]


def scan(
    paths: list[Path] | None = None, denylist: Denylist | None = None, release: bool = False
) -> list[Hit]:
    """Hits in the working-tree files that would be published (or in `paths`)."""
    denylist = load_denylist() if denylist is None else denylist
    hits: list[Hit] = []
    for path in paths if paths is not None else candidate_files():
        name = _name(path)
        if path.suffix in SKIP_SUFFIXES:
            hits += _path_hits(name, denylist)
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if is_text(data):
            hits += scan_text(name, data.decode("utf-8", errors="replace"), denylist, release)
        else:
            hits += _path_hits(name, denylist)
    return hits


def git(args: list[str], root: Path, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=not binary, check=False
    )


def has_commit(root: Path = ROOT) -> bool:
    return git(["rev-parse", "--verify", "--quiet", "HEAD"], root).returncode == 0


def scan_committed(
    root: Path = ROOT, denylist: Denylist | None = None, release: bool = False
) -> list[Hit]:
    """Hits in `git archive HEAD`, the tree a clone of the published repository would have."""
    denylist = load_denylist() if denylist is None else denylist
    archive = git(["archive", "--format=tar", "HEAD"], root, binary=True)
    if archive.returncode != 0:
        return []
    hits: list[Hit] = []
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        for member in tar:
            handle = tar.extractfile(member) if member.isfile() else None
            if handle is None:
                continue
            if Path(member.name).suffix in SKIP_SUFFIXES:
                hits += _path_hits(member.name, denylist)
                continue
            data = handle.read()
            if is_text(data):
                text = data.decode("utf-8", errors="replace")
                hits += scan_text(member.name, text, denylist, release)
            else:
                hits += _path_hits(member.name, denylist)
    return hits


def scan_metadata(root: Path = ROOT, denylist: Denylist | None = None) -> list[Hit]:
    """Hits in the author, committer and message of every commit, and in annotated tags."""
    denylist = load_denylist() if denylist is None else denylist
    hits: list[Hit] = []
    log = git(["log", "--format=%H%n%an <%ae>%n%cn <%ce>%n%B%x00", "HEAD"], root)
    for block in log.stdout.split("\x00") if log.returncode == 0 else []:
        lines = block.strip("\n").splitlines()
        for number, line in enumerate(lines[1:], 1):
            hits += scan_line(f"commit {lines[0][:10]}", number, line, denylist)
    tags = git(
        [
            "for-each-ref",
            "refs/tags",
            "--format=%(refname:short)%09%(taggername) %(taggeremail)%09%(contents)%00",
        ],
        root,
    )
    for block in tags.stdout.split("\x00") if tags.returncode == 0 else []:
        name, _, text = block.strip("\n").partition("\t")
        for number, line in enumerate(text.splitlines(), 1):
            hits += scan_line(f"tag {name}", number, line.replace("\t", " "), denylist)
    return hits


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--release",
        action="store_true",
        help="also list unpublished documents, incident ids and private environment names",
    )
    release = parser.parse_args(argv).release
    denylist = load_denylist()
    if not (denylist.names or denylist.words):
        print(f"note: no entries in {_name(DENYLIST)}; the owner-specific check did not run")
    paths = candidate_files()
    hits = scan(paths, denylist, release)
    committed = has_commit()
    if committed:
        hits += scan_committed(denylist=denylist, release=release)
        hits += scan_metadata(denylist=denylist)
    hits = list(dict.fromkeys(hits))
    for name, number, label, text in hits:
        print(f"{name}:{number}: {label}: {text}")
    where = "files, HEAD and commit metadata" if committed else "files (no commit yet)"
    print(f"{len(hits)} hit(s) in {len(paths)} {where}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
