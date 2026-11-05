"""F22 stmt_encoding (locus task_statement): the sentence says the FASTQ qualities are Phred+33 (or
+64), and the characters in the faulted twin say the other.

The task sentence states the encoding of reads.fastq: Phred+33 on even seeds, Phred+64 on odd seeds.
The clean twin follows it. The faulted twin holds the same Phred scores written with the other
offset (+64 under a +33 claim, +33 under a +64 claim). Task: how many reads have a mean Phred score
of at least 25. Read as stated, a +64 file gives every read a mean 31 too high (all reads pass), and
a +33 file read as +64 gives every read a mean 31 too low (none passes).

Evidence a careful agent can find: the set of quality characters. A +33 file has characters below
ASCII 59 (';', down to '#' for Q2) that Phred+64 cannot hold, a +64 file has characters above 'J'
(ASCII 74, lowercase letters for Q >= 11) that Phred+33 cannot hold (Cock et al. 2010). The file
head does not show it: the first HEAD_READS reads (40 lines) are written so that their characters
are valid under both encodings. Under a +33 claim they are junk reads of Q2-Q9 (+64: 'B'..'I', which
read like Q33-Q40 in +33); under a +64 claim they are good reads of Q32-Q40 (+33: 'A'..'I', which
read like Q1-Q9 in +64). The same Phred scores sit in both twins, so only the characters change.

  explicit  the README states the true encoding
  implicit  the README only says "Illumina reads"
  forced    the task sentence dictates awk

Mean quality is the arithmetic mean of the Phred scores. No read's mean lies within 0.2 of the
cutoff, so float rounding cannot move a read across it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

THRESHOLD = 25
READ_LEN = 50
HEAD_READS = 10  # reads whose characters are valid under both encodings: the head hides the fault
HEAD_Q = {33: (2, 9), 64: (32, 40)}  # Phred range of the head reads, by the STATED encoding
MARGIN = 10  # |sum(Q) - THRESHOLD * length| at least this, i.e. a mean at least 0.2 away
OTHER = {33: 64, 64: 33}
TWIN_FILES = frozenset({"reads.fastq", "README.md"})


@dataclass(frozen=True)
class Plan:
    seqs: list[str]
    quals: list[list[int]]  # Phred scores per read
    stated: int  # the offset the task sentence claims

    def offset(self, twin: str) -> int:
        return self.stated if twin == "clean" else OTHER[self.stated]


def _read_quals(rnd: random.Random) -> list[int]:
    """Phred scores of an ordinary read: a mean level, a decay towards the 3' end, sometimes a Q2
    tail; the first base is Q34-40 (no quality line starts with '@' or '+' in either encoding)."""
    level, decay = rnd.uniform(14, 38), rnd.uniform(0, 10)
    quals = [
        max(3, min(40, round(level + rnd.gauss(0, 3) - decay * (j / (READ_LEN - 1)) ** 1.5)))
        for j in range(READ_LEN)
    ]
    quals[0] = rnd.randint(34, 40)
    if rnd.random() < 0.25:
        tail = rnd.randint(4, 14)
        quals[READ_LEN - tail :] = [2] * tail
    return quals


def passes(quals: list[int]) -> bool:
    return sum(quals) >= THRESHOLD * len(quals)


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f22-{seed}")
    stated = 33 if seed % 2 == 0 else 64
    for _ in range(200):
        n_reads = rnd.randint(40, 56)
        lo, hi = HEAD_Q[stated]
        quals = [[rnd.randint(lo, hi) for _ in range(READ_LEN)] for _ in range(HEAD_READS)]
        while len(quals) < n_reads:
            candidate = _read_quals(rnd)
            if abs(sum(candidate) - THRESHOLD * READ_LEN) >= MARGIN:
                quals.append(candidate)
        if 0.25 * n_reads <= sum(map(passes, quals)) <= 0.75 * n_reads:
            seqs = ["".join(rnd.choices("ACGT", k=READ_LEN)) for _ in range(n_reads)]
            return Plan(seqs, quals, stated)
    raise RuntimeError(f"no valid F22 plan for seed {seed}")


def fastq_text(plan: Plan, offset: int) -> str:
    return "".join(
        f"@read_{i + 1:03d}\n{s}\n+\n{''.join(chr(q + offset) for q in qs)}\n"
        for i, (s, qs) in enumerate(zip(plan.seqs, plan.quals, strict=True))
    )


def count_passing(lines: list[str], offset: int) -> int:
    """Reads whose mean Phred score is at least THRESHOLD when the characters are read at offset."""
    return sum(sum(ord(c) - offset for c in line) >= THRESHOLD * len(line) for line in lines)


def truth_answers(plan: Plan, twin: str) -> tuple[int, int]:
    """(expected, naive): reads passing under the real offset, and under the stated one."""
    lines = fastq_text(plan, plan.offset(twin)).splitlines()[3::4]
    expected = sum(map(passes, plan.quals))
    return expected, count_passing(lines, plan.stated)


def quality_chars(plan: Plan, twin: str) -> list[str]:
    return sorted({chr(q + plan.offset(twin)) for qs in plan.quals for q in qs})


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    real = plan.offset(twin)
    (work / "reads.fastq").write_text(fastq_text(plan, real))
    line = "reads.fastq: Illumina reads, FASTQ (4 lines per read)."
    if tier == "explicit":
        line += f" Base qualities are Phred+{real}."
    (work / "README.md").write_text(common.readme([line]))
    chars = quality_chars(plan, twin)
    lo, hi = chars[0], chars[-1]
    if twin == "clean":
        evidence = (f"quality characters span {lo!r}..{hi!r}, as Phred+{real} allows",)
        if tier == "explicit":
            evidence = ("README states the same encoding as the task sentence",) + evidence
    else:
        seen = f"quality characters span {lo!r}..{hi!r}, " + (
            "none below ASCII 59: not Phred+33 (the task sentence says +33)"
            if real == 64
            else "many below ASCII 59 ('#' is Q2): not Phred+64 (the task sentence says +64)"
        )
        evidence = (f"README states Phred+{real}, against the task sentence", seen)
        evidence = evidence if tier == "explicit" else (seen,)
    what = (
        f"reads in reads.fastq whose mean base quality is at least {THRESHOLD} (the arithmetic "
        "mean of the Phred scores of the read's bases)"
    )
    stated = f"the base qualities in reads.fastq are Phred+{plan.stated} encoded"
    prompt = common.prompt_for(
        route,
        f"count the {what}; {stated}",
        f"use awk to count the {what}; {stated}",
        "a JSON integer",
    )
    expected, naive = truth_answers(plan, twin)
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F22", seed, twin, tier, route)
    return Task(task_id, "F22", "units", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _quality_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()[3::4]


def real_offset(work: Path) -> int | None:
    """33 if a quality character lies below ASCII 59, 64 if one lies above 'J', else None."""
    try:
        codes = [ord(c) for line in _quality_lines(work / "reads.fastq") for c in line]
    except OSError:
        return None
    if not codes:
        return None
    if min(codes) < 59:
        return 33
    return 64 if max(codes) > 74 else None


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def _stated(task: Task) -> int:
    match = re.search(r"Phred\+(33|64)", task.prompt)
    return int(match.group(1)) if match else 33


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    lines = _quality_lines(work / "reads.fastq")
    offset = real_offset(work)
    if offset is None:
        return {
            "status": "needs_input",
            "answer": None,
            "flags": ["encoding unclear"],
            "method": "",
        }
    flags = []
    if offset != _stated(task):
        flags = [f"quality characters show Phred+{offset}, not the stated Phred+{_stated(task)}"]
    return _result(count_passing(lines, offset), f"mean Phred from the Phred+{offset} range", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    lines = _quality_lines(work / "reads.fastq")
    offset = _stated(task)
    return _result(count_passing(lines, offset), f"ord(c) - {offset}, as the task says")


# ---- exposure ----------------------------------------------------------------------------------
# A command that scores reads.fastq writes an offset: `ord(c) - 33`, `offset = 64`, `phred33`, a
# '!' zero, `samtools import`; with none written, a Q25 cutoff in awk/python still reads the
# characters as the default +33. Commands that write the data's own offset, or that look at the
# character range (min/max, sort -u, uniq), are set aside one by one, not the whole session: a model
# that first scores with the stated offset and later finds the encoding still executed the trapped
# route. The remaining commands are searched as one text, so a script written by one command and
# run by another counts. The data's offset is found from the characters in the workspace, so on
# the clean twin the stated offset is the right one and is not exposure.


def _offset(n: int) -> re.Pattern[str]:
    """`x - n`, `ord(c)-n`, `offset=n`, `phredn`, `Phred+n`: the shift written as code."""
    return re.compile(
        rf"[\w)\]]-\s*{n}\b|[\w)\]]\s+-\s+{n}\b"
        rf"|(?i:\b(?:off\w*|base|shift|delta|q0|phred\w*|sanger)\s*[=:+_-]?\s*{n}\b)"
    )


MARKERS = {
    33: (_offset(33), re.compile(r"ord\(\s*['\"]!['\"]\s*\)|\bsamtools\s+import\b")),
    64: (_offset(64), re.compile(r"ord\(\s*['\"]@['\"]\s*\)|(?<![\w.])64(?![\w.])")),
}
READS = re.compile(r"\breads\.fastq\b|\*\.fastq\b")
COMPUTE = re.compile(r"\b(?:awk|gawk|mawk|python3?|perl|Rscript|samtools|seqtk|seqkit)\b")
CUTOFF = re.compile(r"(?<![\w.])25(?![\w.])")
LOOKS = re.compile(r"\bmin\(|\bmax\(|\bsort\s+-\w*u|\buniq\b|<\s*59\b|>\s*74\b")


def writes_offset(text: str, offset: int) -> bool:
    return any(pattern.search(text) for pattern in MARKERS[offset])


def exposed(commands: list[str], work: Path) -> bool:
    real = real_offset(work)
    if real is None:
        return False
    wrong = OTHER[real]
    kept = [c for c in commands if not (writes_offset(c, real) or LOOKS.search(c))]
    text = "\n".join(kept)
    if not (READS.search(text) and COMPUTE.search(text)):
        return False
    return writes_offset(text, wrong) or (wrong == 33 and bool(CUTOFF.search(text)))


SPEC = FamilySpec(
    id="F22",
    slot="units",
    source="GEN-38 (Cock et al. 2010) with the offset claimed in the task sentence",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="task_statement",
)
