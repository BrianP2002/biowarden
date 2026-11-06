"""STAR rule of the alignment pack: the three count columns of ReadsPerGene.out.tab (strand).

star.genecounts_column (T1, fact, declaration optional)
    `STAR --quantMode GeneCounts` writes `ReadsPerGene.out.tab` with, per gene, three counts:
    column 2 unstranded, column 3 reads whose first mate is on the gene's strand (htseq-count -s
    yes, featureCounts -s 1), column 4 reads whose first mate is on the opposite strand (-s reverse,
    featureCounts -s 2, the dUTP protocol of TruSeq Stranded). Nothing in the file says which column
    suits the library. On a dUTP library column 3 is the leakage (here 1-3 pairs per gene, the
    genes look lowly expressed) and column 2 loses every pair that overlaps a gene on the other
    strand: they are counted in the `N_ambiguous` row.

Verified here (STAR 2.7.11b), 5 genes (two overlapping on opposite
strands over 160 bases of exon), 2x75 pairs, one thread, 40 seeds of the F31 family:
    forward library  column 3 = the sense pairs per gene exactly; column 4 = the 1-3 leaked pairs
    dUTP library     column 4 = the sense pairs per gene exactly; column 3 = the leaked pairs, plus
                     the pairs of g4 and g5 that lie in the shared exon and map to the other gene
                     (seed 0: g1 1, g2 2, g3 3, g4 31, g5 49 against the truth 57, 81, 78, 68, 51)
    column 2 (any library) = the sense pairs per gene minus the pairs in the shared exon: seed 0
                     g4 21 against 68, N_ambiguous 78.
    The N_noFeature row of the wrong stranded column holds most pairs (257 against 8).

What the rule does. It is a FACT rule on every existing file named `*ReadsPerGene.out.tab` in a
command (inside awk or python code, a `bash -c` pipeline, a script the command runs). Commands that
only look at the table (head, tail, cat, less, wc, ls, file, grep, ...) never fire it. The data
test: over the gene rows (not `N_*`), a = column 3 total and b = column 4 total; the library is
stranded when |a - b| / (a + b) >= 0.3 and a + b >= 20, first_read_strand when a > b, else
second_read_strand. Unstranded data (a close to b) and tiny tables pass silently.
In the stranded state:
    declared == the column the data point to       -> pass
    declared anything else                          -> fail (contradicts the data)
    no declaration, the command's code shows one column (awk `$N`, `cut -f` list) -> pass when it
        is the stranded column the data point to, fail when it is column 2 or the other one
    no declaration, the column cannot be read from the code (python, R, perl, ...) -> unknown
Declaration `genecounts_column`: unstranded | first_read_strand | second_read_strand = the column
the command reads (2, 3 or 4). Seen: awk/cut/python/perl/R/sort/join/paste commands naming the
file, a script file the command runs. Not seen: a table renamed so that it does not end in
`ReadsPerGene.out.tab`, counts merged from several tables by a script that names none of them.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed

KEY = "genecounts_column"
MIN_PAIRS = 20
MIN_BIAS = 0.3
COLUMN_NAME = {2: "unstranded", 3: "first_read_strand", 4: "second_read_strand"}
NAME_COLUMN = {v: k for k, v in COLUMN_NAME.items()}

NOT_READERS = frozenset(
    "head tail cat less more zless wc ls file stat du md5sum sha1sum sha256sum cksum nl tac rev "
    "fold column xxd od hexdump cp mv rm ln diff cmp touch mkdir basename dirname readlink "
    "realpath tree echo printf test find vi vim nano emacs scp rsync tar zip unzip grep egrep "
    "fgrep zgrep gzip gunzip".split()
)
_SCRIPT_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:py|awk|pl|sh|rb|R)(?![\w.])")
_TABLE_NAME = re.compile(r"(?<![\w.+/@~-])[\w.+/@~-]*ReadsPerGene\.out\.tab(?![\w-])")
_AWK_COLUMN = re.compile(r"\$\s*\{?\s*([2-4])\b")
_CUT_FIELDS = re.compile(r"\bcut\b[^|;&\n]*?(?:-f\s*|--fields[=\s]\s*)([0-9,\-]+)")
_ANY_COLUMN_USE = re.compile(
    r"\$\s*\{?\s*[2-4]\b|\bcut\b[^|;&\n]*-f|\bsplit\s*\(|\.split\b|\bread[._]?(?:csv|table|delim)"
    r"|\busecols\b|\biloc\b|\[\s*[1-3]\s*\]|\bF\[|\bcolumns?\b|\bnames\s*=|\bsort\b[^|;&\n]*-k"
    r"|\bjoin\b|\bpaste\b|\bdatamash\b",
    re.IGNORECASE,
)


def parse_table(ctx: Ctx, path: Path) -> tuple[int, int, int] | None:
    """(column 2, 3, 4 totals over the gene rows), or None if the file is not that table."""
    totals = [0, 0, 0]
    genes = 0
    for line in ctx.lines(path, limit=200_000):
        f = line.split("\t")
        if len(f) != 4 or not all(x.isdigit() for x in f[1:]):
            return None
        if f[0].startswith("N_"):
            continue
        genes += 1
        for i in range(3):
            totals[i] += int(f[i + 1])
    return (totals[0], totals[1], totals[2]) if genes else None


def stranded_column(first: int, second: int) -> int | None:
    """3 or 4 when the totals say the library is stranded, else None."""
    if first + second < MIN_PAIRS:
        return None
    if abs(first - second) / (first + second) < MIN_BIAS:
        return None
    return 3 if first > second else 4


def _script_texts(call: Call, ctx: Ctx) -> list[str]:
    found: list[str] = []
    for token in call.argv:
        for hit in _SCRIPT_NAME.finditer(token):
            path = ctx.resolve(hit.group(0))
            if path.is_file() and len(found) < 5:
                try:
                    found.append("\n".join(ctx.lines(path, limit=2000)))
                except (OSError, EOFError, zlib.error):
                    continue
    return found


def command_text(call: Call, ctx: Ctx) -> str:
    text = "\n".join([" ".join([call.tool, *call.argv]), *_script_texts(call, ctx)])
    return _TABLE_NAME.sub(" ", text)


def reads_a_column(tool: str, text: str) -> bool:
    """Does this command pick counts out of the table (and so choose a column)?"""
    if tool in NOT_READERS:
        return False
    if tool in ("sort", "join", "paste", "datamash"):
        return True
    return _ANY_COLUMN_USE.search(text) is not None


def columns_in_code(text: str) -> set[int]:
    """The columns (2-4) an awk or cut command names, ignoring column 1."""
    found = {int(c) for c in _AWK_COLUMN.findall(text)}
    for fields in _CUT_FIELDS.findall(text):
        for part in fields.split(","):
            lo, _, hi = part.partition("-")
            if lo.isdigit() and hi.isdigit():
                found |= {c for c in range(int(lo), int(hi) + 1) if 2 <= c <= 4}
            elif part.isdigit() and 2 <= int(part) <= 4:
                found.add(int(part))
    return found


@checker("star_genecounts_column")
def star_genecounts_column(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    text = command_text(call, ctx)
    if not reads_a_column(call.tool, text):
        return passed()
    declared = call.declared.get(KEY)
    shown = columns_in_code(text)
    for path in match.files:
        try:
            totals = parse_table(ctx, path)
        except (OSError, EOFError, zlib.error, UnicodeError):
            continue
        if totals is None:
            continue
        right = stranded_column(totals[1], totals[2])
        if right is None:
            continue
        if declared == COLUMN_NAME[right]:
            continue
        sums = f"column 3 {totals[1]}, column 4 {totals[2]}, column 2 {totals[0]}"
        where = (
            "to the gene strand (forward, htseq-count -s yes)"
            if right == 3
            else "to the opposite strand (dUTP, htseq-count -s reverse)"
        )
        profile = (
            f"{path.name} is from a stranded library: over the genes the totals are {sums}, so "
            f"pairs map {where}"
        )
        fix = (
            f"read column {right} ({COLUMN_NAME[right]}) of {path.name} and declare it: "
            f"--declare {KEY}={COLUMN_NAME[right]}",
        )
        if declared is not None:
            return CheckResult(
                "fail",
                "declared_column_contradicts_data",
                f"you declared {KEY}={declared} but {profile}; column {right} holds the counts "
                "for this protocol.",
                expected=COLUMN_NAME[right],
                got=declared,
                fix=fix,
            )
        if len(shown) == 1:
            (used,) = shown
            if used == right:
                continue
            what = "the unstranded column" if used == 2 else f"column {used}"
            return CheckResult(
                "fail",
                "wrong_column",
                f"this command reads {what} of {path.name}, but {profile}: {what} "
                + (
                    "loses every pair that overlaps a gene on the other strand (N_ambiguous)."
                    if used == 2
                    else "holds only the pairs that leak the wrong way."
                ),
                expected=right,
                got=used,
                fix=fix,
            )
        if len(shown) > 1:
            continue  # compares or sums columns: an inspection
        return CheckResult(
            "unknown",
            "column_not_visible",
            f"{profile}; the code does not show which column it reads, and column {right} is the "
            "one for this protocol.",
            fix=fix,
        )
    return passed()
