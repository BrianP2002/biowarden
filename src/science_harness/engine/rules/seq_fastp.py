"""fastp rules of the sequence pack: `-q` filters reads (it trims nothing) and SE adapter detection.

fastp.quality_trimming (family F83, slot flag_semantics; T1, call, declaration optional)
    `fastp -q 20` sets the "qualified base" cut-off (`--qualified_quality_phred`); with `-u`
    (`--unqualified_percent_limit`, default 40) it DROPS whole reads that have more than 40% of
    their bases below Q20. It trims nothing: the 3' tails of the reads that survive keep their
    low-quality bases. Tail trimming is `--cut_right` / `--cut_tail` / `--cut_front` (window `-W`,
    mean quality `-M`). Exit 0; the summary on stderr says how many reads failed due to low
    quality, not that nothing was trimmed.

    Verified here (fastp 1.4.0; 50 reads of 80-100 bases, about half with a Q14-18 tail after a
    Q38-40 head): `--cut_right -W 4 -M 20` returned exactly the heads of all 50 reads; `-q 20`
    returned 35-40 of the 50 reads with the tails of the rest attached (seed 0: 3210 bases in 35
    reads against 3581 in 50); `--cut_tail -W 4 -M 20` cut one base further than the 5' window
    scan on some reads.

    What the rule does. It applies to `fastp` calls that set a quality option (`-q`, `-u`, `-e` or
    their long forms) and no tail-trimming option (`--cut_right`, `--cut_tail`, `--cut_front`,
    `-r`, `-3`, `-5`). It reads the first 5000 reads of `-i`/`--in1` (Phred+33 assumed) and counts
    reads whose last 4 bases have a mean quality below the call's `-q` (15 when absent); with at
    least 3% of such reads it fails. Declaration `low_quality_goal`: filter_reads (dropping reads
    with many low-quality bases is what is wanted: pass) or trim_tails (the tails must go: the
    call trims nothing, fail whatever the data).

fastp.adapter_detection (family F87, slot default_param; T1, call, declaration optional)
    For single-end data fastp detects the adapter automatically, but only from 10,000 reads on: a
    smaller file is never trimmed, and the only trace is "No adapter detected for read1" on stderr
    and "reads with adapter trimmed: 0" in the report. Paired-end data are trimmed by overlap
    analysis without detection.

    Verified here (fastp 1.4.0; 100-base reads, 75% of them insert + TruSeq adapter): detected at
    10,000 and 10,001 reads, not at 9,999, 9,900, 9,500, 9,000 and below (also with 30% adapter
    reads: 10,000 detected, 8,000 not); with `-a AGATCGGAAGAGC` the adapter is cut from every read
    of 250-600 reads exactly (F87, seeds 0-39).

    What the rule does. It applies to single-end `fastp` calls (no `-I`) without `-a`,
    `--adapter_sequence`, `--adapter_fasta`, `-A` or `--disable_adapter_trimming`. It reads `-i`
    (up to 10,000 reads); when the file has fewer than 10,000 reads and at least 5% of the reads
    (and at least 5 reads) contain the first 12 bases of a known adapter (TruSeq/universal, Nextera,
    small-RNA 3' adapter, MGI) at position 10 or later, it fails. Declaration `adapter_trimming`:
    none_needed (the adapter stays on purpose: pass) or adapter_given (the adapter sequence is
    named in the call: fail when the call has no -a).
"""

from __future__ import annotations

from science_harness.engine.argv import option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.seq_common import existing, read_fastq, small_file

QUALITY_OPTIONS = ("-q", "--qualified_quality_phred", "-u", "--unqualified_percent_limit")
QUALITY_OPTIONS += ("-e", "--average_qual")
TRIM_FLAGS = ("--cut_right", "-r", "--cut_tail", "-3", "--cut_front", "-5")
ADAPTER_OPTIONS = ("-a", "--adapter_sequence", "--adapter_fasta")
ADAPTER_OFF = ("-A", "--disable_adapter_trimming")
SAMPLE = 5000
MIN_TAILED = 0.03
DETECTION_READS = 10_000
KNOWN_ADAPTERS = {
    "TruSeq / universal": "AGATCGGAAGAGC",
    "Nextera": "CTGTCTCTTATACACATCT",
    "small RNA 3'": "TGGAATTCTCGGGTGCCAAGG",
    "MGI": "AAGTCGGAGGCCAAGCGGTC",
}


def has_flag(argv: tuple[str, ...], names: tuple[str, ...]) -> bool:
    for token in argv:
        if token == "--":
            break
        if token in names or token.split("=", 1)[0] in names:
            return True
    return False


def _threshold(argv: tuple[str, ...]) -> int:
    values = option_values(argv, "-q", "--qualified_quality_phred")
    try:
        return int(values[-1]) if values else 15
    except ValueError:
        return 15


def _tail_means(reads: list[tuple[str, str, str]]) -> list[float]:
    means = []
    for _, _, qual in reads:
        tail = qual[-4:]
        if len(tail) == 4:
            means.append(sum(ord(c) - 33 for c in tail) / 4)
    return means


@checker("fastp_quality_trimming")
def fastp_quality_trimming(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    if not has_flag(argv, QUALITY_OPTIONS) or has_flag(argv, TRIM_FLAGS):
        return passed()
    declared = call.declared.get(key)
    if declared == "filter_reads":
        return passed()
    threshold = _threshold(argv)
    options = [o for o in QUALITY_OPTIONS if has_flag(argv, (o,))]
    used = "/".join(options)
    if declared == "trim_tails":
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}=trim_tails but this fastp call only sets {used}, the read "
            "filter: it drops reads with many low-quality bases and trims no tail.",
            expected="trim_tails",
            got="filter_reads",
            fix=(
                f"add --cut_right --cut_right_window_size 4 --cut_right_mean_quality {threshold}",
            ),
        )
    names = option_values(argv, "-i", "--in1")
    path = existing(ctx, names[-1]) if names else None
    if path is None or not small_file(path):
        return passed()
    reads = read_fastq(ctx, path, limit=SAMPLE)
    if not reads:
        return passed()
    means = _tail_means(reads)
    tailed = [m for m in means if m < threshold]
    if not means or len(tailed) < max(2, MIN_TAILED * len(means)):
        return passed()
    return CheckResult(
        "fail",
        "filter_not_trimmer",
        f"fastp {used} is a read filter, not a trimmer: it drops reads with more than 40% of "
        f"their bases below Q{threshold} and leaves every low-quality tail of the reads it keeps; "
        f"{len(tailed)} of the {len(means)} reads sampled from {path.name} end in a 4-base window "
        f"with a mean quality below {threshold}.",
        expected="trim_tails",
        got="filter_reads",
        fix=(
            f"add --cut_right --cut_right_window_size 4 --cut_right_mean_quality {threshold} "
            "(5' to 3' window scan; --cut_tail scans from the 3' end)",
            f"or, if dropping low-quality reads is what is wanted: --declare {key}=filter_reads",
        ),
    )


@checker("fastp_adapter_detection")
def fastp_adapter_detection(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    if has_flag(argv, ("-I", "--in2")) or has_flag(argv, ADAPTER_OFF):
        return passed()
    declared = call.declared.get(key)
    if declared == "none_needed":
        return passed()
    if has_flag(argv, ADAPTER_OPTIONS):
        return passed()
    names = option_values(argv, "-i", "--in1")
    path = existing(ctx, names[-1]) if names else None
    if path is None or not small_file(path):
        return passed()
    reads = read_fastq(ctx, path, limit=DETECTION_READS)
    if not reads or len(reads) >= DETECTION_READS:
        return passed()
    sampled = sum(len(name) + 2 * len(seq) + 6 for name, seq, _ in reads)
    if sampled > 0.9 * ctx.max_bytes:  # the byte limit, not the end of the file, stopped the read
        return passed()
    found: dict[str, int] = {}
    for _, seq, _ in reads:
        for label, adapter in KNOWN_ADAPTERS.items():
            if seq.find(adapter[:12], 10) >= 0:
                found[label] = found.get(label, 0) + 1
    if not found:
        return passed()
    label, count = max(found.items(), key=lambda item: item[1])
    if count < max(5, 0.05 * len(reads)):
        return passed()
    adapter = KNOWN_ADAPTERS[label]
    return CheckResult(
        "fail",
        "adapter_not_detected_in_small_file",
        f"{count} of the {len(reads)} reads in {path.name} contain the {label} adapter "
        f"({adapter[:12]}...), and fastp detects a single-end adapter automatically only from "
        f"{DETECTION_READS:,} reads on: this file will not be trimmed ('No adapter detected' on "
        "stderr, nothing else).",
        expected="adapter_given",
        got="auto",
        fix=(
            f"add -a {adapter} (the adapter sequence)",
            f"or, if the adapter must stay: --declare {key}=none_needed",
        ),
    )
