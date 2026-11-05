"""Rules for UCSC liftOver (pack iv): carried columns, the discarded unmapped file, contig names.

Verified on UCSC liftOver (bio-a) with hand-built chain files. Every call quoted here exits 0 and
prints only "Reading liftover chains" and "Mapping coordinates" on stderr, unless said otherwise.

liftover.stale_columns (slot derived_column, CALL rule, tier T1, optional declaration
`summit_offset`; F74). `liftOver -bedPlus=6` lifts the first six columns of a narrowPeak file and
copies columns 7-10 unchanged; column 10 is the summit as an offset from the start. With a chain
that has a 100-base insertion at old position 500 (chr1 2000 -> 2050) the peak `chr1 450 550 pk
500 . 5.0 10.0 8.0 80` is written as `chr1 450 650 ... 80`: the offset is still 80, the lifted
start + 80 = 530 is not the summit (old 530 lifts to 630, as the one-base interval `chr1 530 531`
shows). Without -bedPlus the ten-column file stops liftOver (exit 255, "Expecting integer field
7"), which is loud. The rule reads the input (first 5000 lines); if it has rows of ten or more
columns with an integer column 10 and no `-multiple`/non-BED format option, it dry-runs the call's
own options through `liftOver` (rows renamed to their index, output to /dev/stdout) and lifts the
summit bases `start + offset` as one-base intervals; it fails when lifted start + column 10
differs from the lifted summit for at least one peak. `summit_offset=not_used` (only the peak
intervals are used) or `relifted` (the summits are lifted separately) passes. A file without a
spanning indel (every lifted start shifted like its summit) is silent: zero friction there.

liftover.unmapped_discarded (slot completeness, CALL rule, tier T1, optional declaration
`dropped_records`; F74). The fourth argument receives every interval that does not lift, with a
`#Deleted in new` or `#Partially deleted in new` comment line before it: chr1 1200 1400 (50 of 200
bases deleted in the new assembly) is written there under the default `-minMatch=0.95` and not
lifted; `-minMatch=0.5` lifts it as 1300 1450. When the fourth argument is /dev/null the main
output is simply shorter, exit 0. The rule dry-runs the call (main output to /dev/null, unmapped
to /dev/stdout) and fails when the unmapped file is /dev/null and at least one interval does not
lift; `dropped_records=accepted` passes. A real unmapped file is not judged: the caller can read
it.

liftover.chain_contigs (slot contig_naming, CALL rule, tier T1, no declaration; F74 files). A BED
whose contig names are not the chain's `tName`s (`1` against `chr1`) lifts nothing: with a
synthetic chain on `chr1` every record of `1 100 200 pa` goes to the unmapped file as `#Deleted in
new`, the main output is empty, exit 0 and the same two stderr lines. The rule reads the `chain`
header lines of the chain file (plain or gzip, at most 4 MB of text) and the contig names of the
input; it fails when no contig of the input is a `tName` of the chain. It stays silent when only
some contigs are missing (unplaced contigs are routinely absent from a chain) and when the chain
is too large to read completely."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed

NON_BED_OPTIONS = ("-gff", "-genePred", "-positions", "-pslT", "-multiple", "-hasBin", "-ends")
SAMPLE_LINES = 5000


def split_call(call: Call) -> tuple[list[str], list[str]]:
    """(options, positionals) of a liftOver call; the options are single-dash and use `=`."""
    options = [t for t in call.argv if t.startswith("-") and t != "-"]
    positionals = [t for t in call.argv if not t.startswith("-") or t == "-"]
    return options, positionals


def plain_bed_call(options: list[str]) -> bool:
    return not any(o.split("=")[0] in NON_BED_OPTIONS for o in options)


def input_rows(ctx: Ctx, name: str, limit: int = SAMPLE_LINES) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in ctx.lines(ctx.resolve(name), limit=limit):
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        rows.append(line.split("\t") if "\t" in line else line.split())
    return rows


def dry_run(ctx: Ctx, argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return ctx.run_real("liftOver", argv)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def summit_rows(rows: list[list[str]]) -> list[list[str]] | None:
    """Rows of narrowPeak shape (>= 10 columns, integer start, end and summit offset) or None."""
    peaks = [r for r in rows if len(r) >= 10]
    if not peaks:
        return None
    try:
        for r in peaks:
            int(r[1]), int(r[2]), int(r[9])
    except ValueError:
        return None
    return peaks


# ---- liftover.stale_columns ------------------------------------------------------------------


@checker("iv_liftover_stale_columns")
def iv_liftover_stale_columns(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    options, positionals = split_call(call)
    if len(positionals) < 4 or not plain_bed_call(options):
        return passed()
    if call.declared.get(rule.declare.key) in ("not_used", "relifted"):
        return passed()
    source, chain = positionals[0], positionals[1]
    if not ctx.resolve(source).is_file() or not ctx.resolve(chain).is_file():
        return passed()
    peaks = summit_rows(input_rows(ctx, source))
    if peaks is None:
        return passed()
    names = {str(i): r[3] if len(r) > 3 else str(i) for i, r in enumerate(peaks)}
    with tempfile.TemporaryDirectory() as tmp:
        whole, point = Path(tmp) / "peaks.bed", Path(tmp) / "summits.bed"
        whole.write_text(
            "".join("\t".join([*r[:3], str(i), *r[4:]]) + "\n" for i, r in enumerate(peaks))
        )
        point.write_text(
            "".join(
                f"{r[0]}\t{int(r[1]) + int(r[9])}\t{int(r[1]) + int(r[9]) + 1}\t{i}\n"
                for i, r in enumerate(peaks)
            )
        )
        chain_path = str(ctx.resolve(chain))
        lifted = dry_run(ctx, [*options, str(whole), chain_path, "/dev/stdout", "/dev/null"])
        summits = dry_run(
            ctx,
            [*[o for o in options if not o.startswith("-bedPlus")], str(point), chain_path]
            + ["/dev/stdout", "/dev/null"],
        )
    if lifted is None or summits is None or lifted.returncode != 0 or summits.returncode != 0:
        return passed()  # liftOver stops on its own (loud) or cannot be run here
    by_peak = {
        r[3]: r for r in (line.split("\t") for line in lifted.stdout.splitlines()) if len(r) >= 10
    }
    by_summit = {
        r[3]: int(r[1]) for r in (line.split("\t") for line in summits.stdout.splitlines())
    }
    stale = []
    for index, row in by_peak.items():
        if index in by_summit and int(row[1]) + int(row[9]) != by_summit[index]:
            stale.append((index, int(row[1]) + int(row[9]), by_summit[index]))
    if not stale:
        return passed()
    index, naive, true = stale[0]
    key = rule.declare.key
    return CheckResult(
        "fail",
        "summit_offset_stale",
        f"liftOver copies the narrowPeak columns after the sixth unchanged, so the summit offset "
        f"(column 10) is stale after an indel: for {len(stale)} of {len(by_peak)} lifted peaks of "
        f"{Path(source).name} lifted start + offset is not the summit base (peak {names[index]}: "
        f"{naive}, while the summit base itself lifts to {true}).",
        fix=(
            f'lift the summit bases as intervals of their own: awk \'BEGIN{{OFS="\\t"}}'
            f"{{print $1,$2+$10,$2+$10+1,$4}}' {Path(source).name} > summits.bed; liftOver "
            f"summits.bed {chain} summits.lifted.bed summits.unmapped.bed",
            f"or, if only the peak intervals matter, re-run with --declare {key}=not_used",
        ),
    )


# ---- liftover.unmapped_discarded -------------------------------------------------------------


@checker("iv_liftover_unmapped_discarded")
def iv_liftover_unmapped_discarded(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    options, positionals = split_call(call)
    if len(positionals) < 4 or positionals[3] != "/dev/null":
        return passed()
    if call.declared.get(rule.declare.key) == "accepted":
        return passed()
    source, chain = positionals[0], positionals[1]
    if not ctx.resolve(source).is_file() or not ctx.resolve(chain).is_file():
        return passed()
    proc = dry_run(ctx, [*options, source, chain, "/dev/null", "/dev/stdout"])
    if proc is None or proc.returncode != 0:
        return passed()
    lines = proc.stdout.splitlines()
    dropped = [line for line in lines if line.strip() and not line.startswith("#")]
    if not dropped:
        return passed()
    reasons = sorted({line.lstrip("#").strip() for line in lines if line.startswith("#")})
    first = dropped[0].split("\t")
    label = first[3] if len(first) > 3 else "\t".join(first[:3])
    return CheckResult(
        "fail",
        "dropped_records_discarded",
        f"liftOver does not lift {len(dropped)} interval(s) of {Path(source).name} "
        f"(first: {label}; {', '.join(reasons) or 'unmapped'}) and this call sends the unmapped "
        "file to /dev/null, "
        "so the output is silently shorter than the input.",
        fix=(
            "write the fourth argument to a file (for example lifted.unmapped.bed), count it and "
            "report the intervals that did not lift",
            f"or declare --declare {rule.declare.key}=accepted if losing them is intended",
        ),
    )


# ---- liftover.chain_contigs ------------------------------------------------------------------


@checker("iv_liftover_chain_contigs")
def iv_liftover_chain_contigs(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    options, positionals = split_call(call)
    if len(positionals) < 4 or not plain_bed_call(options):
        return passed()
    source, chain = ctx.resolve(positionals[0]), ctx.resolve(positionals[1])
    if not source.is_file() or not chain.is_file():
        return passed()
    if chain.stat().st_size > ctx.max_bytes // 4:
        return passed()  # too large to read completely: cannot tell
    targets = {
        line.split()[2]
        for line in ctx.lines(chain)
        if line.startswith("chain ") and len(line.split()) > 2
    }
    contigs = {r[0] for r in input_rows(ctx, str(source), 20000)}
    if not targets or not contigs or contigs & targets:
        return passed()
    shown = ", ".join(sorted(contigs)[:3])
    known = ", ".join(sorted(targets)[:3])
    if {f"chr{c}" for c in contigs} <= targets:
        fix = (
            'awk \'BEGIN{OFS="\\t"}{$1="chr"$1; print}\' ' + f"{source.name} > chr_{source.name}",
        )
    elif {c.removeprefix("chr") for c in contigs} <= targets:
        fix = ("sed 's/^chr//' " + f"{source.name} > nochr_{source.name}",)
    else:
        fix = (f"rename the contigs of {source.name} to the names in the chain ({known}, ...)",)
    return CheckResult(
        "fail",
        "no_contig_in_chain",
        f"none of the contigs of {source.name} ({shown}) is a source contig of {chain.name} "
        f"({known}): liftOver puts every interval in the unmapped file and exits 0 with an empty "
        "output.",
        fix=fix,
    )
