"""Shared readers for the `seq` rules (sequence files, FASTQ reads and command lines).

Nothing here is a rule. It holds what several rules need: stdlib readers of FASTA and FASTQ (plain
or gzip, through `Ctx.lines`, so they obey the tier and the byte limit), the reverse complement,
the positional arguments of a command line given the options that take a value, and a helper that
runs a real tool on a copy of a call's arguments (outputs redirected to a temporary directory) for
the rules that compare two settings of the same tool.
"""

from __future__ import annotations

import subprocess
import tempfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

from biowarden.engine.checks import Ctx

_COMP = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")
NUCLEOTIDE = frozenset("ACGTUN")


class DryRunFailed(Exception):
    """A read-only rerun of a tool stopped with an error."""


MAX_RECORDS = 20_000
MAX_FILE_BYTES = 5_000_000  # inputs larger than this are not dry-run or sniffed completely
# a dry run that cannot be made is no evidence (a TierViolation is not one of these)
RUN_ERRORS = (OSError, subprocess.SubprocessError, DryRunFailed)


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def is_nucleotide(seqs: Iterable[str], share: float = 0.9) -> bool:
    """At least `share` of the letters of the sequences are A C G T U N."""
    total = good = 0
    for seq in seqs:
        up = seq.upper()
        total += len(up)
        good += sum(c in NUCLEOTIDE for c in up)
    return total > 0 and good / total >= share


def read_fasta(ctx: Ctx, path: Path, limit: int = MAX_RECORDS) -> list[tuple[str, str]] | None:
    """(name, sequence) of the records of a FASTA file; None when it is not FASTA or unreadable."""
    records: list[tuple[str, list[str]]] = []
    try:
        for line in ctx.lines(path):
            if line.startswith(">"):
                if len(records) >= limit:
                    break
                records.append((line[1:].split(None, 1)[0] if line[1:].strip() else "", []))
            elif records:
                records[-1][1].append(line.strip())
            elif line.strip():
                return None
    except (OSError, EOFError, zlib.error, UnicodeError):
        return None
    return [(name, "".join(parts)) for name, parts in records]


def read_fastq(ctx: Ctx, path: Path, limit: int = MAX_RECORDS) -> list[tuple[str, str, str]] | None:
    """(name, sequence, quality) of the first `limit` reads; None when not FASTQ or unreadable."""
    reads: list[tuple[str, str, str]] = []
    try:
        lines = ctx.lines(path, limit=4 * limit)
        block: list[str] = []
        for line in lines:
            block.append(line)
            if len(block) < 4:
                continue
            if not block[0].startswith("@") or not block[2].startswith("+"):
                return None if not reads else reads
            if len(block[1].strip()) != len(block[3].strip()):
                return None if not reads else reads
            reads.append((block[0][1:].split(None, 1)[0], block[1].strip(), block[3].strip()))
            block = []
    except (OSError, EOFError, zlib.error, UnicodeError):
        return reads or None
    return reads


def positionals(argv: Sequence[str], value_options: frozenset[str]) -> list[str]:
    """Arguments that are not options and not the value of an option in `value_options`.

    `--name=value` forms are options with their value attached. A lone `-` is a positional.
    """
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token.startswith("-") and len(token) > 1:
            skip = token in value_options
            continue
        out.append(token)
    return out


def existing(ctx: Ctx, name: str) -> Path | None:
    path = ctx.resolve(name)
    return path if path.is_file() else None


def small_file(path: Path, limit: int = MAX_FILE_BYTES) -> bool:
    try:
        return path.stat().st_size <= limit
    except OSError:
        return False


def drop_options(
    argv: Sequence[str], with_value: Iterable[str] = (), flags: Iterable[str] = ()
) -> list[str]:
    """`argv` without the options named (those in `with_value` also lose their value)."""
    with_value, flags = set(with_value), set(flags)
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        name = token.split("=", 1)[0] if token.startswith("--") else token
        if len(token) > 2 and not token.startswith("--") and token[:2] in with_value:
            continue  # the value attached to a short option: -l30
        if name in with_value:
            skip = "=" not in token or not token.startswith("--")
            continue
        if name in flags:
            continue
        out.append(token)
    return out


def run_in_scratch(
    ctx: Ctx, tool: str, argv: Sequence[str], build: Callable[[Path], list[str]] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real `tool` with `argv` in the call's directory, outputs under a temp directory.

    `build(tmp)` may return extra arguments that name output files inside the temp directory.
    """
    with tempfile.TemporaryDirectory(prefix="biowarden-dry-") as tmp:
        extra = build(Path(tmp)) if build else []
        return ctx.run_real(tool, [*argv, *extra])


_MEMO: dict[tuple, object] = {}
MEMO_SIZE = 512


def stamp(paths: Iterable[Path]) -> tuple:
    """Identity of input files (path, size, mtime) for the memo key of a dry run."""
    out = []
    for path in paths:
        try:
            info = path.stat()
            out.append((str(path), info.st_size, info.st_mtime_ns))
        except OSError:
            out.append((str(path), -1, -1))
    return tuple(out)


def memo(key: tuple, compute: Callable[[], object]) -> object:
    """Remember a dry run: the same arguments on unchanged files give the same answer."""
    if key not in _MEMO:
        if len(_MEMO) >= MEMO_SIZE:
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = compute()
    return _MEMO[key]


def kmers(seq: str, k: int) -> Iterator[str]:
    for i in range(len(seq) - k + 1):
        yield seq[i : i + k]
