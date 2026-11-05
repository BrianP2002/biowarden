"""Multiple-alignment rule of the sequence pack: MAFFT and MUSCLE align sequences as written.

msa.sequence_orientation (family F86, slot strand; T1, call, declaration optional)
    `mafft seqs.fasta` and `muscle -align seqs.fasta` assume every sequence is on the same strand.
    A member written as the reverse complement of the others is aligned as unrelated sequence:
    exit 0, the right number of rows, almost no conserved column. `mafft --adjustdirection` (or
    `--adjustdirectionaccurately`) flips such sequences and renames them with a `_R_` prefix;
    MUSCLE 5 has no such option.

    Verified here (MAFFT v7.526, MUSCLE 5.3; 8 DNA sequences of 300 bases at 92% identity, two
    reverse-complemented): `mafft --quiet` 357 columns, 64 conserved; `--auto` 417 columns, 41
    conserved; `--adjustdirection` and `--adjustdirectionaccurately` 300 columns, 163 conserved,
    rows renamed `_R_seq3` and `_R_seq6`; plain runs keep the input order, `--reorder` does not.
    MUSCLE writes its rows in guide-tree order (not input order) and "Unknown option stable" is
    printed for `-stable` in 5.3. On the F86 sets (seeds 0-59) `--adjustdirectionaccurately`
    returns exactly the analytic count of conserved columns and the plain call is 88 or more
    columns short.

    What the rule does. It applies to `mafft` and `muscle`. For mafft it takes the last argument
    that is an existing file, for muscle the value of `-align` or `-super5`. It reads up to 200
    records (5 MB); for nucleotides it counts, per sequence, the 10-mers shared with the forward
    10-mers of the others (summed over the others) and those shared by its reverse complement, and
    a sequence with at least as much reverse-complement as forward sharing (and at least 10 shared
    10-mers) is reversed. A
    mafft call with `--adjustdirection[accurately]` passes. Declaration `sequence_orientation`:
    same_strand (fails when the data hold a reversed sequence), mixed_strands (fails when the call
    cannot flip: muscle, or mafft without the option).
    Not seen: protein files (pass), input on stdin, fragments shorter than 30 bases.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from science_harness.engine.argv import option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.seq_common import (
    existing,
    is_nucleotide,
    kmers,
    read_fasta,
    revcomp,
    small_file,
)

K = 10
MIN_SHARED = 10
MAX_RECORDS = 200
MAFFT_ADJUST = ("--adjustdirection", "--adjustdirectionaccurately")


def reversed_members(records: list[tuple[str, str]]) -> list[str]:
    """Names of the sequences that share more 10-mers with the others as reverse complement.

    For each sequence the 10-mers it shares with the forward 10-mers of the other sequences are
    summed over those sequences (a 10-mer found in three others counts three), and so are the
    10-mers of its reverse complement. At least as much reverse-complement as forward support
    marks it as reversed (a tie, as in a set of two, marks every member: orientations are mixed).
    """
    seqs = [seq.upper().replace("U", "T") for _, seq in records]
    forward = [set(kmers(s, K)) for s in seqs]
    counts: Counter[str] = Counter()
    for kms in forward:
        counts.update(kms)
    flagged: list[str] = []
    for (name, _), seq, own in zip(records, seqs, forward, strict=True):
        direct = sum(counts[km] - 1 for km in own)
        flipped = sum(counts[km] - (1 if km in own else 0) for km in set(kmers(revcomp(seq), K)))
        if flipped >= MIN_SHARED and flipped >= direct:
            flagged.append(name)
    return flagged


def _input_file(call: Call, ctx: Ctx) -> Path | None:
    if call.tool == "muscle":
        names = option_values(call.argv, "-align", "-super5")
        return existing(ctx, names[-1]) if names else None
    for token in reversed(call.argv):
        if not token.startswith("-"):
            path = existing(ctx, token)
            if path is not None:
                return path
    return None


@checker("msa_sequence_orientation")
def msa_sequence_orientation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    declared = call.declared.get(key)
    flips = call.tool == "mafft" and any(a in call.argv for a in MAFFT_ADJUST)
    if call.tool == "muscle" and not (
        "-align" in call.argv or "-super5" in call.argv or option_values(call.argv, "-align")
    ):
        return passed()
    if flips:
        return passed()
    path = _input_file(call, ctx)
    cannot = (
        "muscle -align has no option that flips sequences"
        if call.tool == "muscle"
        else "mafft without --adjustdirection aligns sequences as written"
    )
    fix_flip = (
        "orient the reversed sequences first (reverse-complement them, e.g. seqkit seq -r -p) "
        "or align with mafft --adjustdirectionaccurately"
        if call.tool == "muscle"
        else "add --adjustdirectionaccurately (flipped sequences are renamed with an _R_ prefix)"
    )
    if declared == "mixed_strands":
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}=mixed_strands but this {call.tool} call does not flip sequences: "
            f"{cannot}.",
            expected="mixed_strands",
            got="same_strand",
            fix=(fix_flip,),
        )
    if path is None or not small_file(path):
        return passed()
    records = read_fasta(ctx, path, limit=MAX_RECORDS)
    if not records or len(records) < 3 or not is_nucleotide(s for _, s in records):
        return passed()
    flagged = reversed_members(records)
    if not flagged:
        return passed()
    shown = ", ".join(flagged[:3]) + (", ..." if len(flagged) > 3 else "")
    prefix = (
        f"you declared {key}=same_strand but {len(flagged)} of {len(records)} sequences"
        if declared == "same_strand"
        else f"{len(flagged)} of {len(records)} sequences"
    )
    return CheckResult(
        "fail",
        "declared_mismatch" if declared == "same_strand" else "reverse_complemented_input",
        f"{prefix} in {path.name} ({shown}) share more 10-mers with the others as reverse "
        f"complement than as written, and {cannot}: they would be aligned as unrelated sequence "
        "(few conserved columns, long gaps).",
        expected="mixed_strands" if declared == "same_strand" else "oriented sequences",
        got=declared or "as written",
        fix=(fix_flip,),
    )
