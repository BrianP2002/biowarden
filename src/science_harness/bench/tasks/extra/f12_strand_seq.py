"""F12 strand_seq (strand): features on the - strand need the reverse complement of the interval.

The BED6 strand column is ignored by `bedtools getfasta` unless `-s` is given (its help says "By
default, strand information is ignored"), and by `samtools faidx` unless `-i` is given. The answer
is the same in both twins: the clean genome has every feature on +; in the faulted genome the
- features are stored as the reverse complement of that answer, so the strand-blind sequence
differs from the right one on every - feature (palindromic sequences are excluded).
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

N_FEATURES = 8
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def _features(rnd: random.Random, genome: dict[str, str]) -> list[tuple[str, int, int]]:
    """N_FEATURES non-overlapping (chrom, start0, end0), 20-45 bp, sorted, none palindromic."""
    found: list[tuple[str, int, int]] = []
    while len(found) < N_FEATURES:
        chrom = rnd.choice(list(genome))
        start = rnd.randint(15, len(genome[chrom]) - 70)
        end = start + rnd.randint(20, 45)
        seq = genome[chrom][start:end]
        clear = all(c != chrom or end + 10 < s or start > e + 10 for c, s, e in found)
        if clear and seq != revcomp(seq):
            found.append((chrom, start, end))
    order = {c: i for i, c in enumerate(genome)}
    return sorted(found, key=lambda f: (order[f[0]], f[1]))


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rnd = random.Random(f"f12-{seed}")
    lengths = {"chr1": rnd.randint(800, 1200), "chr2": rnd.randint(600, 1000)}
    clean = {c: "".join(rnd.choice("ACGT") for _ in range(n)) for c, n in lengths.items()}
    features = _features(rnd, clean)
    minus = set(rnd.sample(range(N_FEATURES), rnd.randint(3, 5)))  # drawn for both twins
    expected = [clean[c][s:e] for c, s, e in features]
    genome = {c: list(seq) for c, seq in clean.items()}
    if twin == "faulted":
        for i in minus:  # store the - features as the reverse complement of the answer
            c, s, e = features[i]
            genome[c][s:e] = revcomp(expected[i])
    genome_text = {c: "".join(seq) for c, seq in genome.items()}
    strands = ["-" if twin == "faulted" and i in minus else "+" for i in range(N_FEATURES)]
    bed = "".join(
        f"{c}\t{s}\t{e}\tf{i + 1:02d}\t0\t{strands[i]}\n" for i, (c, s, e) in enumerate(features)
    )
    common.write_fasta(work, genome_text, tools)
    (work / "features.bed").write_text(bed)
    stated = (
        " Column 6 is the strand of the feature; the sequence of a feature on the - strand is "
        "the reverse complement of its reference interval."
    )
    if tier == "explicit":
        feature_line = "features.bed: features in BED6 format (start 0-based, end exclusive)."
        feature_line += stated
    else:
        feature_line = "features.bed: features in BED6 format."
    readme = ["ref.fa (with ref.fa.fai): reference sequence, contigs chr1 and chr2", feature_line]
    (work / "README.md").write_text(common.readme(readme))
    if twin == "clean":
        evidence = ("features.bed column 6 is + for every feature",)
    else:
        evidence = (
            "features.bed column 6 holds - for some features",
            "bedtools getfasta help: by default, strand information is ignored (-s forces it)",
        )
    prompt = common.prompt_for(
        route,
        "report the sequence of each feature in features.bed on the feature's own strand, "
        "uppercase, in the order of the file",
        "use `bedtools getfasta` to report the sequence of each feature in features.bed on the "
        "feature's own strand, uppercase, in the order of the file",
        "a JSON list of uppercase sequence strings, one per feature",
    )
    blind = [genome_text[c][s:e] for c, s, e in features]  # strand-blind: equals expected if clean
    truth = {"kind": "seq_list", "expected": expected, "naive": blind}
    task_id = common.task_id("F12", seed, twin, tier, route)
    return Task(task_id, "F12", "strand", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _sequences(stdout: str) -> list[str]:
    seqs: list[str] = []
    for line in stdout.splitlines():
        if line.startswith(">"):
            seqs.append("")
        elif seqs:
            seqs[-1] += line.strip().upper()
    return seqs


def _getfasta(work: Path, tools: ToolEnv, *flags: str) -> list[str]:
    argv = ["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "features.bed", *flags]
    return _sequences(tools.run(argv, work).stdout)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_getfasta(work, tools, "-s"), "bedtools getfasta -s (strand-aware)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_getfasta(work, tools), "bedtools getfasta on features.bed")


# Exposure: a strand-blind extraction from the features, i.e. `bedtools getfasta` without -s or
# `samtools faidx` on regions without -i / --reverse-complement, in a command that does not also
# reverse-complement by itself (rev, tr, revcomp, complement, [::-1]).
SEGMENT = re.compile(r"[|;&\n]")
GETFASTA = re.compile(r"\bbedtools\s+getfasta\b")
FAIDX = re.compile(r"\bsamtools\s+faidx\b")
GETFASTA_STRAND = re.compile(r"(?<!\S)-s(?!\S)")
FAIDX_STRAND = re.compile(r"(?<!\S)(?:-i|--reverse-complement)(?!\S)")
REGION = re.compile(r"\w:\d+-\d+")
REVCOMP = re.compile(r"\brev\b|\btr\b[^|;&]*ACGT|revcomp|reverse_?comp|complement|\[::-1\]", re.I)


def _blind(command: str) -> bool:
    if REVCOMP.search(command):
        return False
    segments = SEGMENT.split(command)
    if "features.bed" in command and any(
        GETFASTA.search(s) and not GETFASTA_STRAND.search(s) for s in segments
    ):
        return True
    regions = "features.bed" in command or REGION.search(command)
    return bool(regions) and any(FAIDX.search(s) and not FAIDX_STRAND.search(s) for s in segments)


def exposed(commands: list[str], work: Path) -> bool:
    return any(_blind(c) for c in commands)


SPEC = FamilySpec(
    id="F12",
    slot="strand",
    source="bedtools getfasta ignores the BED6 strand column unless -s (cf. GEN-21, GEN-22)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"ref.fa", "ref.fa.fai", "features.bed", "README.md"}),
)
