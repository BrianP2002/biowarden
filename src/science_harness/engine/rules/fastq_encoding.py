"""Rule fastq.phred_offset (slot units, family F13): Phred+33 versus Phred+64 FASTQ qualities.

The trap (GEN-38; Cock et al. 2010, PMID 20015970). FASTQ exists in incompatible variants: Sanger
and Illumina 1.8+ write Phred+33 (quality characters '!'..'J'), Illumina 1.3-1.7 write Phred+64
('@'..'h') and Solexa+64 reaches down to ';'. A +64 file read as +33 parses fine and every quality
is 31 too high, so a Q30 cutoff written for +33 (character '?' or above) accepts every base.

Verified here (samtools 1.24, GNU awk 4.2.1) on the F13 files built from the same Phred scores in
both encodings (seed 3, 57 reads of 50 bases): `samtools import reads.fastq -o reads.bam` exits 0
with empty stderr on both and has no quality-offset option (`samtools import` usage);
`samtools stats` then reports `average quality 32.7` for the +33 file and `63.7` for the +64 file,
so the +64 qualities were stored 31 too high. The awk ord-table count with `-33` gives 2850 (every
base) on the +64 file, with `-64` 2256 (the bases that are Q>=30); on the +33 file `-33` gives 2256.

What the rule does. It is a FACT rule: it applies to every existing file whose name looks like
`*.fastq`, `*.fq`, with an optional `.gz`/`.bgz`, mentioned anywhere in a command (inside `awk`
or `python3 -c` code, a `bash -c` pipeline, a script the command runs). It then sniffs the quality
lines of the first `max_reads` records (default 1000; plain or gzip text, at most Ctx.max_bytes of
decoded text; a record needs `@` header, `+` line and a quality line as long as the sequence, the
sniff stops at the first record that does not):
  any quality character below ';' (ASCII 59)             -> Phred+33 proven (Phred+64 and Solexa+64
                                                             never go below ';')
  all in ';'..'h' (59..104) and at least one above 'J'    -> Phred+64 (the risky state)
  anything else (only ';'..'J', or something above 'h')    -> ambiguous
  no complete quality line (empty file, shorter than a read, not FASTQ) -> nothing to say, pass
  unreadable (truncated or corrupt gzip before any read)   -> unknown
Caveat on the +64 proof: long-read +33 files (PacBio HiFi) can have only high qualities that pass
'J'; such a file would be taken for +64 if no character below ';' occurs in its first reads.

Declaration `fastq_quality_offset` (optional), `33` or `64`:
  data Phred+64 : no declaration -> fail (unless the command shows a 64 offset itself, below);
                  declared 64 -> pass; declared 33 -> fail (contradicts the data).
  data Phred+33 : pass silently; declared 64 -> fail (contradicts the data). An extension of the
                  brief: the same data-versus-declaration contradiction, mirrored.
  ambiguous     : no declaration and no 64 offset in the command -> unknown (a warning only);
                  any declaration -> pass.

When the rule looks at a command (zero friction on everything else). It is a heuristic over the
argument text, FASTQ names removed, plus the text of script files the command runs:
  scripting tools (awk family, python*, perl*, ruby, Rscript, R, julia, node, shells, env, xargs,
      parallel, timeout, time, nohup, srun): only when the code reads qualities as numbers and so
      sees a default of +33: `ord(`, awk `sprintf("%c"` tables, `unpack`, a `-33` shift or
      `offset = 33`, a `>= "?"` / `>= "^"` character cutoff, the words phred / quality / qual,
      `Q30`, Biopython SeqIO, or `samtools import`. `head`, `awk 'END{print NR/4}'` and
      `awk 'NR%4==0' reads.fastq | head` do not match.
  samtools: only `samtools import` (the verified one).
  any other program: unless it is on a closed list of viewers and file utilities (head, tail, cat,
      less, wc, ls, file, stat, grep, sed, cut, zcat, gzip, cp, mv, rm, md5sum, ...), it is assumed
      to read qualities (seqtk, fastp, cutadapt, bwa, ...). Their own offset options were not
      verified (not installed); the check for a 64 offset below applies to them too.
  A command that shows a 64 offset (a standalone `64`, `-64`, `offset = 64`, `ord('@')`,
  `>= 94`, `>= "^"`, `--phred64`, `-Q64`, Biopython `fastq-illumina`/`fastq-solexa`) is taken to
  have handled the encoding and stays silent without a declaration. That test is deliberately loose.
Not seen: a quality-aware program inside a `bash -c` pipeline that has no quality arithmetic, a
script written by one command and run by another when the second names no script file, FASTQ names
given as globs, and any arithmetic that is not written as one of the patterns above. The rule
cannot verify that the code applies the offset it declares.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed

DEFAULT_MAX_READS = 1000
MAX_SCRIPTS = 5
PHRED33_BELOW = 59  # ';': nothing below it exists in a Phred+64 or Solexa+64 file
PHRED64_TOP = 104  # 'h': Illumina 1.3-1.7 maximum
PHRED33_TOP = 74  # 'J': Illumina 1.8+ maximum

_FASTQ_NAME = re.compile(r"(?<![\w.+/@~-])[\w.+/@~-]*\.f(?:ast)?q(?:\.b?gz)?(?![\w-]|\.\w)")
_SCRIPT_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:py|awk|pl|sh|rb|R)(?![\w.])")
_SCRIPTING = re.compile(
    r"[gmn]?awk|python[\d.]*|perl[\d.]*|ruby|Rscript|R|julia|node"
    r"|bash|sh|zsh|dash|ksh|env|xargs|parallel|timeout|time|nohup|srun"
)
VIEWERS = frozenset(
    "head tail cat less more zless zmore wc ls file stat du md5sum sha1sum sha256sum sha512sum "
    "cksum zcat gzcat bzcat xzcat gunzip gzip bgzip nl tac rev fold cut paste column sort uniq tr "
    "sed grep egrep fgrep zgrep xxd od hexdump cp mv rm ln diff cmp touch mkdir basename dirname "
    "readlink realpath tree echo printf test find vi vim nano emacs scp rsync tar zip unzip".split()
)


def _shift(n: int) -> str:
    """`x-33`, `ord(c)-33`, `x - 33` (a shift written as code); `head -33` is not one."""
    return rf"[\w)\]]-\s*{n}\b|[\w)\]]\s+-\s+{n}\b"


_QUALITY_AS_NUMBER = re.compile(
    r"\bord\s*\(|sprintf\s*\(\s*[\"']%c|\bunpack\b"
    rf"|{_shift(33)}"
    r"|(?i:\b(?:off\w*|base|shift|delta|q0|sanger)\s*[=:+_-]?\s*33\b)"
    r"|>=?\s*[\"'][?^][\"']"
    r"|(?i:\bphred)|(?i:\bqual(?:ity|ities)?\b)|\bQ\d{2}\b|\bSeqIO\b|\bletter_annotations\b"
    r"|\bsamtools\s+import\b"
)
_HANDLES_64 = re.compile(
    rf"{_shift(64)}"
    r"|(?i:\b(?:off\w*|base|shift|delta|q0|illumina)\s*[=:+_-]?\s*64\b)"
    r"|(?<![\d.])64(?![\d.])"
    r"|ord\(\s*['\"]@['\"]\s*\)|>=?\s*94\b|>=?\s*[\"']\^[\"']"
    r"|fastq-illumina|fastq-solexa"
)


def _script_texts(call: Call, ctx: Ctx) -> list[str]:
    found: list[str] = []
    seen: set[Path] = set()
    for token in call.argv:
        for hit in _SCRIPT_NAME.finditer(token):
            path = ctx.resolve(hit.group(0))
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                found.append("\n".join(ctx.lines(path, limit=2000)))
            except (OSError, EOFError, zlib.error):
                continue
            if len(found) >= MAX_SCRIPTS:
                return found
    return found


def command_text(call: Call, ctx: Ctx) -> str:
    """The command as code to look at: tool, arguments and the scripts it runs, FASTQ names out."""
    text = "\n".join([call.tool, *call.argv, *_script_texts(call, ctx)])
    return _FASTQ_NAME.sub(" ", text)


def reads_qualities(tool: str, text: str) -> bool:
    """Does this command interpret quality characters (and so assume an offset)?"""
    if _SCRIPTING.fullmatch(tool):
        return _QUALITY_AS_NUMBER.search(text) is not None
    if tool == "samtools":
        return re.search(r"\bimport\b", text) is not None
    return tool not in VIEWERS


def shows_64_offset(text: str) -> bool:
    return _HANDLES_64.search(text) is not None


# -- the data test ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualitySniff:
    kind: str  # phred33 | phred64 | ambiguous | none | unreadable
    low: int = 0
    high: int = 0
    reads: int = 0
    detail: str = ""


def classify(low: int, high: int) -> str:
    if low < PHRED33_BELOW:
        return "phred33"
    if PHRED33_TOP < high <= PHRED64_TOP:
        return "phred64"
    return "ambiguous"


def sniff_qualities(ctx: Ctx, path: Path, max_reads: int) -> QualitySniff:
    """Range of the quality characters in the first `max_reads` records of a plain/gzip FASTQ."""
    low, high, reads = 127, 0, 0
    seq = ""
    lines = ctx.lines(path, limit=4 * max_reads)
    try:
        for i, line in enumerate(lines):
            slot = i % 4
            if slot == 0:
                if not line.startswith("@"):
                    break
            elif slot == 1:
                seq = line.strip()
            elif slot == 2:
                if not line.startswith("+"):
                    break
            else:
                qual = line.strip()
                if len(qual) != len(seq):
                    break
                reads += 1
                codes = [c for c in map(ord, set(qual)) if 33 <= c <= 126]
                if codes:
                    low, high = min(low, min(codes)), max(high, max(codes))
    except (OSError, EOFError, zlib.error) as exc:
        if reads == 0:
            return QualitySniff("unreadable", detail=f"{type(exc).__name__}: {exc}")
    finally:
        lines.close()  # type: ignore[attr-defined]  # a generator: release the file now
    if high == 0:
        return QualitySniff("none", reads=reads)
    return QualitySniff(classify(low, high), low, high, reads)


# -- the checker -----------------------------------------------------------------------------


def _span(sniff: QualitySniff) -> str:
    return (
        f"{chr(sniff.low)!r}..{chr(sniff.high)!r} (ASCII {sniff.low}..{sniff.high}, "
        f"{sniff.reads} reads looked at)"
    )


def _fail_undeclared_64(name: str, key: str, sniff: QualitySniff) -> CheckResult:
    effect = (
        f"even the lowest character {chr(sniff.low)!r} would read as Q{sniff.low - 33}, so a "
        "cutoff written for +33 accepts every base"
        if sniff.low >= 63
        else f"the lowest character {chr(sniff.low)!r} would read as Q{sniff.low - 33} instead of "
        f"Q{sniff.low - 64}"
    )
    message = (
        f"{name} is Phred+64, not Phred+33: its quality characters span {_span(sniff)}, none "
        "below ';' (59) and some above 'J' (74); read as Phred+33 every quality is 31 too high "
        f"({effect}; Q30 is '?' or above for +33 but '^' or above for +64)."
    )
    return CheckResult(
        "fail",
        "phred64_undeclared",
        message,
        expected=["33", "64"],
        got=None,
        fix=(
            f"re-run with --declare {key}=64 and compute Q as ord(c) - 64 (Q30 is the character "
            "'^' or above)",
            "or convert the file to Phred+33 first (subtract 31 from every quality character) "
            "and use the converted file",
        ),
    )


def _fail_contradiction(name: str, key: str, declared: str, sniff: QualitySniff) -> CheckResult:
    if declared == "33":
        message = (
            f"you declared {key}=33 but {name} is Phred+64: its quality characters span "
            f"{_span(sniff)}, none below ';' (59) and some above 'J' (74), which a Phred+33 "
            "Illumina file does not have."
        )
        fix = (f"declare {key}=64 and compute Q as ord(c) - 64 (Q30 is '^' or above)",)
    else:
        message = (
            f"you declared {key}=64 but {name} is Phred+33: its quality characters span "
            f"{_span(sniff)} and go below ';' (59), which a Phred+64 file never does."
        )
        fix = (f"declare {key}=33 and compute Q as ord(c) - 33 (Q30 is '?' or above)",)
    return CheckResult(
        "fail",
        "declared_offset_contradicts_data",
        message,
        expected="64" if declared == "33" else "33",
        got=declared,
        fix=fix,
    )


@checker("fastq_phred_offset")
def fastq_phred_offset(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text = command_text(call, ctx)
    if not reads_qualities(call.tool, text):
        return passed()
    declared = call.declared.get(key)
    handled = shows_64_offset(text)
    max_reads = int(rule.params.get("max_reads", DEFAULT_MAX_READS))
    unknown: list[str] = []
    for path in match.files:
        sniff = sniff_qualities(ctx, path, max_reads)
        if sniff.kind == "phred33":
            if declared == "64":
                return _fail_contradiction(path.name, key, declared, sniff)
        elif sniff.kind == "phred64":
            if declared == "33":
                return _fail_contradiction(path.name, key, declared, sniff)
            if declared is None and not handled:
                return _fail_undeclared_64(path.name, key, sniff)
        elif sniff.kind == "ambiguous":
            if declared is None and not handled:
                unknown.append(
                    f"{path.name}: quality characters span {_span(sniff)}, which both Phred+33 "
                    "and Phred+64 can produce"
                )
        elif sniff.kind == "unreadable":
            unknown.append(f"{path.name}: could not read it ({sniff.detail})")
    if unknown:
        return CheckResult(
            "unknown",
            "offset_not_provable",
            "; ".join(unknown) + ": the offset cannot be read from the data.",
            fix=(
                f"state the offset from the README or the sequencer documentation: "
                f"--declare {key}=33 or --declare {key}=64",
            ),
        )
    return passed()
