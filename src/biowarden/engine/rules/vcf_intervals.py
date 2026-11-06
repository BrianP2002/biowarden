"""Rule vcf.indel_interval_anchor (slot interval_closure, family F10): the VCF anchor base.

The trap. A VCF indel is written with an anchor: the REF allele of a deletion starts with a base
the deletion does not remove, and ALT is that base alone. Turning records into BED intervals from
POS and the length of REF (`bcftools query -f '%POS0\\t%END'`, awk `$2-1 .. $2-1+length($4)`) gives
the span of the REF allele, anchor included, so for a deletion the interval starts one base before
the bases that are actually deleted. For an SNV or an MNV the same arithmetic is right. Nothing
fails: the output is a perfectly valid BED.

Verified on bcftools 1.24 (htslib 1.24) and GNU awk 4.2.1 with the records `chr1 10 del ACG A`
(the deleted bases are positions 11 and 12, BED 10..12), `chr1 20 snv C T`, `chr1 30 ins A ACT`,
`chr1 40 mnv AC GT`, `chr1 50 multi AC A,ACC`:

    bcftools query -f '%CHROM\\t%ID\\t%POS\\t%POS0\\t%END\\t%REF\\t%ALT\\n'
      chr1 del 10 9 12 ACG A      %POS0 = POS-1, %END = POS-1+len(REF): the REF span, BED 9..12
      chr1 snv 20 19 20 C T       one base, right
      chr1 ins 30 29 30 A ACT     the REF span is the anchor alone
      chr1 mnv 40 39 41 AC GT     right: every REF base is replaced
    awk '{print $1,$2-1,$2-1+length($4)}'          -> chr1 9 12 for the deletion (anchor included)
    awk '{s=$2-1; if (length($4)>length($5) && substr($4,1,1)==substr($5,1,1)) s++;
          print $1,s,$2-1+length($4)}'             -> chr1 10 12 (anchor skipped), rest unchanged
    bcftools query -f '%POS0\\t%END' gives the same numbers on .vcf, .vcf.gz and .bcf input, and
    `bcftools head -h 0 -n N FILE` prints the first N records of all three (used to sample a BCF).

What the rule does. It is a FACT rule and fires only when both hold:
  data     the VCF has a record whose REF and ALT share the first base while their lengths differ
           (an anchored deletion or insertion), among its first `max_records` records (default
           10000, and at most Ctx.max_bytes of text): anchored records further on are not seen.
           Symbolic (`<DEL>`), `*`, `.` and breakend alleles do not count, and neither do
           equal-length records (SNV, MNV).
  command  the command looks like it builds intervals (heuristic, below) and does not show that it
           looked at the ALT allele.
Ordinary reads (view, head, counting, `awk 'END{print NR}'`) never match the command test, and a
VCF without an anchored indel (the clean twin of F10) never matches the data test: zero friction.

The command test is a regex heuristic over the argument text plus the text of script files the
command runs (`python3 convert.py f.vcf.gz`), and it errs towards silence:
  builds intervals  `%POS0` or `%END` in a bcftools format; or POS minus one (`$2-1`, `pos - 1`,
                    `f[1]) - 1`) together with the length of something (`length(`, `len(`,
                    `nchar(`); or POS plus a length (`$2+length(`, `start + len(`); or pysam-style
                    `.start` and `.stop`/`.end` attributes (not method calls) together.
  looked at ALT     `$5`, ALT/alt, `[4]` (a field index into ALT), `substr(`, or two different
                    `length(...)` arguments (REF against ALT). Printing %ALT in a `-f` format
                    string and then measuring REF alone is the trap itself and does not count. A
                    bcftools type restriction (`-v snps`, `-V indels`, a TYPE/INDEL filter) counts
                    too: an interval for SNVs only has no anchor.
Declaration `interval_meaning` (optional): `ref_span` (the REF allele span, anchor included) or
`replaced_bases` (the bases the record substitutes or deletes). In the risky state no declaration
fails; any declaration passes, because it forces the decision and the failure message already
carried the knowledge.

Honest limits. The checker cannot verify the interval arithmetic itself: a declaration is accepted
whatever the command computes, and a command that reads ALT but gets the arithmetic wrong (for
example skipping the anchor of every record with a multi-base REF, MNVs included) is invisible to
it. The engine sees one tool call. In a pipeline whose stages are separate calls, a stage that
names the VCF but has no interval arithmetic (`bcftools query -f '%POS\\t%REF' f.vcf.gz`) and a
stage that has the arithmetic but never names the file (`awk '...'` on stdin) are each invisible.
Seen are: one `bash -c` pipeline, `bcftools query -f '%POS0\\t%END' f.vcf.gz`, `python3 -c` code
that opens the file. Anchors at the end of REF (VCF allows them at position 1) and equal-length
records that merely share a first base are not treated as anchors.
"""

from __future__ import annotations

import gzip
import re
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path

from biowarden.engine.argv import option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed

DEFAULT_MAX_RECORDS = 10_000
MAX_SCRIPTS = 5

# -- the command test ------------------------------------------------------------------------

_FORMAT_FIELD = re.compile(r"%POS0|%END\b")
_ZERO_BASED_START = re.compile(
    r"\$2\s*-\s*1\b"  # awk
    r"|(?<![\w$])[Pp][Oo][Ss]\s*-\s*1\b"  # pos - 1
    r"|\[1\]\)?\s*-\s*1\b"  # int(f[1]) - 1, $F[1]-1
)
_LENGTH_CALL = re.compile(r"\b(?:length|len|strlen|nchar)\s*\(")
_POS_PLUS_LENGTH = re.compile(
    r"(?:\$2|(?<![\w$])[Pp][Oo][Ss]|(?<![\w$])start|\[1\]\)?)\s*\+\s*(?:length|len|nchar)\s*\("
)
_ATTR_START = re.compile(r"\.start\b(?!\s*\()")
_ATTR_STOP = re.compile(r"\.(?:stop|end)\b(?!\s*\()")

_ALT_USED = re.compile(r"\$5\b|\b[Aa][Ll][Tt]s?\b|\[4\]|\bsubstr\s*\(")
_LENGTH_ARG = re.compile(r"\b(?:length|len|strlen|nchar)\s*\(\s*([^)]*?)\s*\)")
_FORMAT_STRING = re.compile(r"(?:-f|--format)[\s=]*(?:'[^']*'|\"[^\"]*\")")
_TYPE_FILTER = re.compile(r"\bTYPE\b|\bINDEL\b|\bis_indel\b|\bis_snp\b")
_ONLY_SUBSTITUTIONS = re.compile(
    r"(?:^|\s)(?:-v|--types)[\s=]+(?:snps|mnps|ref|bnd)(?:,(?:snps|mnps|ref|bnd))*(?=\s|$)"
    r"|(?:^|\s)(?:-V|--exclude-types)[\s=]+[a-z,]*indels"
)
_SCRIPT_NAME = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:py|awk|pl|sh|rb|R)(?![\w.])")


def _script_texts(call: Call, ctx: Ctx) -> list[str]:
    """Text of script files the command names (`python3 convert.py f.vcf`, `awk -f prog.awk`)."""
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
            except OSError:
                continue
            if len(found) >= MAX_SCRIPTS:
                return found
    return found


def command_text(call: Call, ctx: Ctx) -> tuple[str, str]:
    """(text, text_without_format_strings): argv, then the scripts it runs."""
    scripts = _script_texts(call, ctx)
    formats = set(option_values(call.argv, "-f", "--format"))
    plain = [t for t in call.argv if t not in formats]
    full = "\n".join([call.tool, *call.argv, *scripts])
    bare = _FORMAT_STRING.sub(" ", "\n".join([call.tool, *plain, *scripts]))
    return full, bare


def builds_intervals(text: str) -> bool:
    """The command turns records into intervals from POS and the length of REF (heuristic)."""
    if _FORMAT_FIELD.search(text):
        return True
    has_length = _LENGTH_CALL.search(text) is not None
    if has_length and _ZERO_BASED_START.search(text):
        return True
    if _POS_PLUS_LENGTH.search(text):
        return True
    return bool(_ATTR_START.search(text) and _ATTR_STOP.search(text))


def looks_at_alt(bare_text: str) -> bool:
    """The command shows that the ALT allele (or the variant type) took part in the computation."""
    if _ALT_USED.search(bare_text) or _TYPE_FILTER.search(bare_text):
        return True
    if _ONLY_SUBSTITUTIONS.search(bare_text):
        return True
    # two different things measured (REF against ALT); the length of a whole line says nothing
    return len(set(_LENGTH_ARG.findall(bare_text)) - {"$0"}) >= 2


# -- the data test ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchored:
    chrom: str
    pos: int
    ref: str
    alt: str


@dataclass(frozen=True)
class VcfSample:
    scanned: int
    anchored: int
    example: Anchored | None


def shares_anchor(ref: str, alt: str) -> bool:
    """REF and ALT start with the same base and differ in length: an anchored indel allele."""
    if not ref or not alt or alt[0] in "<*." or "[" in alt or "]" in alt:
        return False
    return len(ref) != len(alt) and ref[0].upper() == alt[0].upper()


def _is_bcf(path: Path) -> bool:
    try:
        with path.open("rb") as raw:
            if raw.read(2) != b"\x1f\x8b":
                return False
        with gzip.open(path, "rb") as handle:
            return handle.read(3) == b"BCF"
    except (OSError, EOFError, zlib.error):
        return False


def _record_lines(ctx: Ctx, path: Path, limit: int) -> list[str]:
    """The first `limit` data lines of a VCF (plain or gzip) or BCF, header lines left out."""
    if _is_bcf(path):
        proc = ctx.run_real("bcftools", ["head", "-h", "0", "-n", str(limit), str(path)])
        if proc.returncode != 0:
            raise OSError(proc.stderr.strip() or f"bcftools head exited {proc.returncode}")
        return proc.stdout.splitlines()
    records: list[str] = []
    lines = ctx.lines(path)
    try:
        for line in lines:
            if line and not line.startswith("#"):
                records.append(line)
                if len(records) >= limit:
                    break
    finally:
        lines.close()  # type: ignore[attr-defined]  # a generator: release the file now
    return records


def sample_vcf(ctx: Ctx, path: Path, max_records: int) -> VcfSample:
    """Scan the first `max_records` records for anchored indel alleles."""
    scanned = anchored = 0
    example: Anchored | None = None
    example_is_deletion = False
    for line in _record_lines(ctx, path, max_records):
        if line.startswith("#"):
            continue
        fields = line.split("\t", 5)
        if len(fields) < 5:
            continue
        scanned += 1
        ref = fields[3]
        hits = [alt for alt in fields[4].split(",") if shares_anchor(ref, alt)]
        if not hits:
            continue
        anchored += 1
        plain_deletion = len(hits[0]) == 1 and len(ref) > 1  # the clearest thing to show
        if example is None or (plain_deletion and not example_is_deletion):
            try:
                example = Anchored(fields[0], int(fields[1]), ref, hits[0])
            except ValueError:
                continue
            example_is_deletion = plain_deletion
    return VcfSample(scanned, anchored, example)


# -- the checker -----------------------------------------------------------------------------


def _example_text(sample: VcfSample) -> tuple[str, str]:
    """(record description, what the REF span gets wrong for it)."""
    ex = sample.example
    assert ex is not None
    record = f"{ex.chrom}:{ex.pos} REF={ex.ref} ALT={ex.alt}"
    start0 = ex.pos - 1
    end = start0 + len(ex.ref)
    if len(ex.alt) == 1 and len(ex.ref) > 1:
        return record, (
            f"the REF span of that record is {ex.chrom}:{start0}-{end} (0-based, end exclusive) "
            f"but the deleted bases are {ex.chrom}:{start0 + 1}-{end}"
        )
    return record, (
        f"the REF span of that record is {ex.chrom}:{start0}-{end} (0-based, end exclusive) and "
        "starts at the unchanged anchor base"
    )


@checker("vcf_indel_interval_anchor")
def vcf_indel_interval_anchor(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    text, bare = command_text(call, ctx)
    if not builds_intervals(text) or looks_at_alt(bare):
        return passed()
    declared = call.declared.get(key)
    if declared is not None:
        return passed(
            f"{key}={declared} declared; the interval arithmetic itself is not verified by the rule"
        )
    max_records = int(rule.params.get("max_records", DEFAULT_MAX_RECORDS))
    unreadable: list[str] = []
    for path in match.files:
        try:
            sample = sample_vcf(ctx, path, max_records)
        except (OSError, EOFError, zlib.error, subprocess.SubprocessError) as exc:
            unreadable.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if sample.anchored == 0 or sample.example is None:
            continue
        record, effect = _example_text(sample)
        message = (
            f"{path.name} has {sample.anchored} of its first {sample.scanned} records with an "
            f"indel allele whose REF and ALT share the first base (the VCF anchor, e.g. {record}), "
            "and this command builds intervals from POS and the length of REF, which gives the "
            f"REF allele span with the anchor included: {effect}."
        )
        fix = (
            f"declare what the intervals mean: --declare {key}=ref_span (the whole REF allele, "
            f"anchor included) or --declare {key}=replaced_bases (anchor excluded: a deletion "
            "starts at POS, not POS-1, and an insertion replaces no base).",
            "For replaced_bases: bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' "
            f"{path.name} | awk -F'\\t' -v OFS='\\t' '{{s=$2-1; if (length($3)>length($4) && "
            "substr($3,1,1)==substr($4,1,1)) s++; print $1,s,$2-1+length($3)}'",
        )
        return CheckResult(
            "fail",
            "anchor_base_in_interval",
            message,
            expected=list(rule.declare.values),
            got=None,
            fix=fix,
        )
    if unreadable:
        return CheckResult(
            "unknown",
            "vcf_unreadable",
            "could not read the VCF to look for anchored indels: " + "; ".join(unreadable),
        )
    return passed()
