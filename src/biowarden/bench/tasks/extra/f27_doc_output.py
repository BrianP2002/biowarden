"""F27 doc_output (output_format, locus documentation): the notes read the mapped count from line 5
of `samtools flagstat`, where releases up to 1.12 printed it; since 1.13 line 5 is "duplicates".

Source: samtools NEWS, release 1.13: "Three new counts have been added to samtools flagstat:
primary, mapped primary and duplicate primary". The block grew from 13 to 16 lines and the old
fifth line moved to the seventh.

Verified by running the installed 1.24 and the older 1.10.1 on one BAM (67 records: 60 mapped,
25 of them duplicate-flagged, 7 unmapped; no secondary or supplementary records):

  samtools 1.10.1  line 4 "25 + 0 duplicates"       line 5 "60 + 0 mapped (89.55% : N/A)"
  samtools 1.24    line 2 "67 + 0 primary"          line 5 "25 + 0 duplicates"
                   line 6 "25 + 0 primary duplicates"  line 7 "60 + 0 mapped (89.55% : N/A)"
                   line 8 "60 + 0 primary mapped (89.55% : N/A)"
  `samtools view -c -F 4` gives 60 and `samtools view -c -f 1024` gives 25 on the same file.

`samtools flagstat FILE | sed -n 5p | cut -d' ' -f1` exits 0 with an empty stderr and prints a
plausible integer (the duplicate count) on 1.24. The mapped count includes the duplicate-flagged
reads (60 of 67 above, 25 of them duplicates), so the two numbers are far apart.

Twin axis: `twin_files` is NOTES.md. The three BAMs are the same in both twins. The notes say the
mapped count is on line 7 (clean; with the release that added the lines) or on line 5 (faulted,
the layout of the older releases). Task: the mapped read count of each of three BAMs, in order.
Each BAM has 50-80 mapped reads of which 12-25 are duplicates, so naive and expected differ by at
least 28 on every file over seeds 0-29. A loop over three files that cuts line N and prints only
the number never shows the label, which is how the stale line survives a quick look.

  explicit  the README adds that the notes can be out of date and should be checked
  implicit  the README says nothing about the notes
  forced    the task sentence dictates `samtools flagstat`
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

CONTIGS = {"chr1": 520, "chr2": 430}
SAMPLES = ("sample1.bam", "sample2.bam", "sample3.bam")
READ_LEN = 36
STALE_LINE = 5  # where releases up to 1.12 printed the mapped count
MAPPED_LINE = 7  # where 1.13 and later print it
TWIN_FILES = frozenset({"NOTES.md"})

CHECK_SENTENCE = (
    "The notes may be out of date: check what they say against the installed tools (their --help "
    "output, or a small test on the data) before relying on it."
)


@dataclass(frozen=True)
class Sample:
    mapped: int
    duplicates: int  # among the mapped ones
    unmapped: int
    sam_body: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    samples: tuple[Sample, ...]

    def sam(self, i: int) -> str:
        lines = ["@HD\tVN:1.6\tSO:coordinate"]
        lines += [f"@SQ\tSN:{c}\tLN:{n}" for c, n in CONTIGS.items()]
        return "\n".join([*lines, *self.samples[i].sam_body]) + "\n"


def _sample(rnd: random.Random, index: int) -> Sample:
    qual = "I" * READ_LEN
    mapped = rnd.randint(50, 80)
    duplicates = rnd.randint(12, 25)
    unmapped = rnd.randint(3, 9)
    dup_slots = set(rnd.sample(range(mapped), duplicates))
    rows: list[tuple[int, int, str]] = []
    for k in range(mapped):
        c = rnd.randrange(len(CONTIGS))
        length = list(CONTIGS.values())[c]
        flag = rnd.choice((0, 16)) + (1024 if k in dup_slots else 0)
        seq = "".join(rnd.choice("ACGT") for _ in range(READ_LEN))
        pos = rnd.randint(1, length - READ_LEN + 1)
        name = f"s{index}m{k:03d}"
        line = f"{name}\t{flag}\t{list(CONTIGS)[c]}\t{pos}\t60\t{READ_LEN}M\t*\t0\t0\t{seq}\t{qual}"
        rows.append((c, pos, line))
    rows.sort(key=lambda t: (t[0], t[1]))
    body = [r[2] for r in rows]
    for k in range(unmapped):
        seq = "".join(rnd.choice("ACGT") for _ in range(READ_LEN))
        body.append(f"s{index}u{k:03d}\t4\t*\t0\t0\t*\t*\t0\t0\t{seq}\t{qual}")
    return Sample(mapped, duplicates, unmapped, tuple(body))


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f27-{seed}")
    return Plan(tuple(_sample(rnd, i) for i in range(len(SAMPLES))))


def truth_answers(plan: Plan) -> tuple[list[int], list[int]]:
    """(expected, naive): the mapped counts, and the duplicate counts that line 5 now holds."""
    return [s.mapped for s in plan.samples], [s.duplicates for s in plan.samples]


def notes_text(twin: str) -> str:
    if twin == "clean":
        where = (
            f"The mapped count is on line {MAPPED_LINE} (the block has had the primary, primary "
            "duplicates and primary mapped lines since samtools 1.13)."
        )
    else:
        where = f"The mapped count is on line {STALE_LINE}."
    lines = [
        "# Lab notes: flagstat summaries",
        "",
        '- `samtools flagstat FILE.bam` prints one statistic per line as "N + M label" '
        "(QC-passed + QC-failed reads); the first number is the one we report.",
        f"- {where}",
        "- Duplicates are flagged (0x400) but kept in the BAM; they are counted as mapped.",
        "- Unmapped reads are kept in the BAM (flag 0x4) and are not counted as mapped.",
        "- None of our BAMs has secondary, supplementary or QC-failed records.",
    ]
    return "\n".join(lines) + "\n"


def readme_lines(tier: str) -> list[str]:
    lines = [
        "sample1.bam, sample2.bam, sample3.bam: alignments of three small runs on chr1 and chr2 "
        "(coordinate-sorted, no index).",
        "NOTES.md: the lab's notes on flagstat summaries; the task says to follow them.",
    ]
    return [*lines, CHECK_SENTENCE] if tier == "explicit" else lines


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    for i, name in enumerate(SAMPLES):
        stem = name.removesuffix(".bam")
        (work / f"{stem}.sam").write_text(plan.sam(i))
        argv = ["samtools", "view", "-b", "--no-PG", "-o", name, f"{stem}.sam"]
        common.check(tools.run(argv, work), argv)
        (work / f"{stem}.sam").unlink()
    (work / "NOTES.md").write_text(notes_text(twin))
    (work / "README.md").write_text(common.readme(readme_lines(tier)))
    if twin == "clean":
        evidence = (
            "NOTES.md puts the mapped count on line 7, where `samtools flagstat` prints "
            "its 'mapped' line",
        )
    else:
        evidence = (
            "NOTES.md puts the mapped count on line 5; `samtools flagstat` prints '... duplicates' "
            "on line 5 and '... mapped (...)' on line 7",
            "a small test: print the whole flagstat block of one BAM and read the labels",
        )
    prompt = common.prompt_for(
        route,
        "following the lab notes in NOTES.md, report how many reads are mapped in each of "
        "sample1.bam, sample2.bam and sample3.bam, in that order",
        "following the lab notes in NOTES.md, use `samtools flagstat` to report how many reads "
        "are mapped in each of sample1.bam, sample2.bam and sample3.bam, in that order",
        "a JSON list of three integers, in the order sample1.bam, sample2.bam, sample3.bam",
    )
    expected, naive_answer = truth_answers(plan)
    truth = {
        "kind": "int_list",
        "expected": expected,
        "naive": expected if twin == "clean" else naive_answer,
    }
    task_id = common.task_id("F27", seed, twin, tier, route)
    return Task(task_id, "F27", "output_format", twin, seed, prompt, truth, evidence, tier, route)


def _ok(answer: Any, method: str) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": [], "method": method}


def _flagstat(tools: ToolEnv, work: Path, name: str) -> list[str]:
    return tools.run(["samtools", "flagstat", name], work).stdout.splitlines()


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Ignores the notes: reads the line labelled 'mapped'."""
    counts = []
    for name in SAMPLES:
        for line in _flagstat(tools, work, name):
            match = re.match(r"(\d+) \+ \d+ mapped \(", line)
            if match:
                counts.append(int(match.group(1)))
    return _ok(counts, "samtools flagstat, the line labelled mapped")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    """Cuts the line number the notes give."""
    text = (work / "NOTES.md").read_text()
    line_no = int(re.search(r"mapped count is on line (\d+)", text).group(1))  # type: ignore[union-attr]
    counts = [int(_flagstat(tools, work, name)[line_no - 1].split()[0]) for name in SAMPLES]
    return _ok(counts, f"samtools flagstat, line {line_no}")


# Exposure: a command that runs samtools flagstat and picks out line 5 by number (sed -n 5p, awk
# NR==5, head -5 | tail -1, a [4] index into the split lines), with no selection by label (grep
# mapped, /mapped/ in awk, "mapped" in line) in the same command. A command that prints the whole
# block, or cuts line 7, is not counted. Per command, not per session: a later correction does
# not undo an earlier trapped call.
FLAGSTAT = re.compile(r"\bsamtools\s+flagstat\b|\bflagstat\b|['\"]flagstat['\"]")
LINE_FIVE = re.compile(
    r"\bsed\s+(?:-n\s+)?['\"]?5\s*(?:p|,\s*5\s*p|\{)"
    r"|\bsed\s+(?:-n\s+)?['\"]5!d['\"]"
    r"|\b(?:NR|FNR)\s*==\s*5\b"
    r"|\bhead\s+(?:-n\s*)?-?5\b[^\n]*\btail\s+(?:-n\s*)?-?1\b"
    r"|\[4\]"
)
BY_LABEL = re.compile(
    r"\bgrep\b[^|\n]*\bmapped\b|/[^/\n]*\bmapped\b[^/\n]*/|\$4\s*==\s*['\"]?mapped"
    r"|['\"]mapped['\"]\s+in\b|\.find\(\s*['\"]mapped"
)


def exposed(commands: list[str], work: Path) -> bool:
    return any(
        FLAGSTAT.search(c) and LINE_FIVE.search(c) and not BY_LABEL.search(c) for c in commands
    )


SPEC = FamilySpec(
    id="F27",
    slot="output_format",
    source="samtools 1.13 NEWS: flagstat gained the primary, primary mapped and primary duplicates "
    "lines",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="documentation",
)
