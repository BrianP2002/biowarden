"""F13 phred_offset: FASTQ qualities written as Phred+33 (clean) or Phred+64 (faulted).

Source GEN-38 (Cock et al. 2010): FASTQ exists in incompatible variants and a +64 file read as +33
shifts every quality by 31 while parsing fine. The twins hold the same reads and the same underlying
Phred scores; only the quality characters differ. Task: count bases with Q >= 30. Read as +33, every
base of the +64 file passes, so the naive answer is the total number of bases.

  explicit  the README states the encoding of reads.fastq
  implicit  the README only says "Illumina reads"; the character range is the evidence
            (+33 here spans '#'..'I'; +64 spans 'B'..'h' with nothing below ASCII 59)
  forced    the task sentence dictates awk, whose usual ord-table script subtracts 33
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

THRESHOLD = 30
READ_LEN = 50
LOW_Q = 2  # Phred 2 is the 'B' read-segment indicator of Illumina 1.5 and later a +33 '#'


def _qualities(rng: random.Random, n_reads: int) -> list[list[int]]:
    """Phred scores per read: high at the start, decaying; some reads end in a Q2 run."""
    reads: list[list[int]] = []
    for i in range(n_reads):
        top, drop = rng.uniform(36, 40), rng.uniform(5, 20)
        quals = [
            max(3, min(40, round(top - drop * (j / (READ_LEN - 1)) ** 1.5 + rng.gauss(0, 2.5))))
            for j in range(READ_LEN)
        ]
        quals[0] = rng.randint(34, 40)  # no quality line starts with '@' or '+' in either encoding
        if i == 0 or rng.random() < 0.2:
            tail = rng.randint(4, 14)
            quals[READ_LEN - tail :] = [LOW_Q] * tail
        reads.append(quals)
    reads[1][0] = 40  # the top of the range is present
    return reads


def fastq_text(seqs: list[str], quals: list[list[int]], offset: int) -> str:
    return "".join(
        f"@read_{i + 1:03d}\n{s}\n+\n{''.join(chr(q + offset) for q in qs)}\n"
        for i, (s, qs) in enumerate(zip(seqs, quals, strict=True))
    )


def _fastq_quality_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()[3::4]


def count_at_least(lines: list[str], offset: int) -> int:
    return sum(ord(c) - offset >= THRESHOLD for line in lines for c in line)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = random.Random(f"f13-{seed}")
    n_reads = rng.randint(40, 60)
    seqs = ["".join(rng.choices("ACGT", k=READ_LEN)) for _ in range(n_reads)]
    quals = _qualities(rng, n_reads)
    offset = 33 if twin == "clean" else 64
    text = fastq_text(seqs, quals, offset)
    (work / "reads.fastq").write_text(text)
    line = "reads.fastq: Illumina reads, FASTQ (4 lines per read)."
    if tier == "explicit":
        line += f" Base qualities are Phred+{offset}."
    (work / "README.md").write_text(common.readme([line]))
    chars = sorted(set("".join(text.splitlines()[3::4])))
    lo, hi = chars[0], chars[-1]
    if twin == "clean":
        evidence = ("README states Phred+33",) if tier == "explicit" else ()
    else:
        seen = f"quality characters span {lo!r}..{hi!r}, none below ASCII 59"
        evidence = ("README states Phred+64", seen) if tier == "explicit" else (seen,)
    prompt = common.prompt_for(
        route,
        f"count the bases with a Phred quality score of at least {THRESHOLD} across all reads "
        "in reads.fastq",
        f"use awk to count the bases with a Phred quality score of at least {THRESHOLD} across "
        "all reads in reads.fastq",
        "a JSON integer",
    )
    expected = sum(q >= THRESHOLD for qs in quals for q in qs)
    naive = expected if twin == "clean" else count_at_least(text.splitlines()[3::4], 33)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F13", seed, twin, tier, route)
    return Task(task_id, "F13", "units", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    lines = _fastq_quality_lines(work / "reads.fastq")
    codes = [ord(c) for line in lines for c in line]
    offset = 33 if min(codes) < 59 else 64  # below 59 only occurs in +33; above 74 only in +64
    flags = [f"lowest quality character is ASCII {min(codes)}: read as Phred+64"]
    flags = flags if offset == 64 else []
    return _result(count_at_least(lines, offset), f"Phred+{offset} from the character range", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    lines = _fastq_quality_lines(work / "reads.fastq")
    return _result(count_at_least(lines, 33), "ord(c) - 33 on every quality character")


def _offset(n: int) -> re.Pattern[str]:
    """`x - n`, `ord(c)-n`, `offset=n`, `phredn`, `Phred+n`: the shift written as code."""
    return re.compile(
        rf"[\w)\]]-\s*{n}\b|[\w)\]]\s+-\s+{n}\b"
        rf"|(?i:\b(?:off\w*|base|shift|delta|q0|phred\w*|sanger)\s*[=:+_-]?\s*{n}\b)"
    )


SHIFT_33 = _offset(33)
SHIFT_64 = _offset(64)
# "ord('!')" is the +33 zero; ">= 63" or >= "?" is the +33 character of Q30; ">= 94" or "^" the
# +64 one
ZERO_33 = re.compile(
    r"ord\(\s*['\"]!['\"]\s*\)|>=?\s*63\b|>=?\s*['\"]\?['\"]|\bsamtools\s+import\b"
)
ZERO_64 = re.compile(
    r"ord\(\s*['\"]@['\"]\s*\)|>=?\s*94\b|>=?\s*['\"]\^['\"]|(?<![\w.])64(?![\w.])"
)
READS = re.compile(r"\breads\.fastq\b|\*\.fastq\b")
COMPUTE = re.compile(r"\b(?:awk|gawk|mawk|python3?|perl|Rscript|samtools|seqtk|seqkit)\b")
# the Q30 cutoff written as a number: a script that scores the file with Q >= 30 and never writes
# an offset still reads the characters as the default +33 (index into a table that starts at 33)
CUTOFF_30 = re.compile(r"(?<![\w.])30(?![\w.])")


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff some command scores reads.fastq assuming +33 and mentions no offset of 64.

    Commands that handle a 64 offset are set aside one by one, not the whole session: a model that
    first runs the +33 reading and then discovers the encoding still executed the trapped route
    (exposure is "ran it", not "kept the answer"). The remaining commands are searched as one text,
    so a script written by one command and run by another counts. A +33 reading is a written shift,
    a +33 character cutoff, or a Q30 cutoff in an awk/python/perl command over the file.
    """
    text = "\n".join(c for c in commands if not (SHIFT_64.search(c) or ZERO_64.search(c)))
    if not READS.search(text) or not COMPUTE.search(text):
        return False
    return bool(SHIFT_33.search(text) or ZERO_33.search(text) or CUTOFF_30.search(text))


SPEC = FamilySpec(
    id="F13",
    slot="units",
    source="GEN-38",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.fastq", "README.md"}),
)
