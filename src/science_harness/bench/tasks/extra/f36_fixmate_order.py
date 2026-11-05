"""F36 fixmate_order (record_order, data): `samtools fixmate` on a BAM whose mates are not adjacent.

`samtools fixmate` pairs mates by looking at consecutive records: it needs the file grouped by read
name (`samtools sort -n` or `collate`). It refuses a header that says SO:coordinate, but a BAM whose
header says SO:unsorted (or nothing) and whose records are in any other order is accepted: exit 0,
empty stderr, and every record is treated as an unpaired read. The mate fields (RNEXT, PNEXT, TLEN)
and the proper-pair flag are cleared. `fixmate -m` then adds ms tags on reads without a mate,
and `samtools markdup` runs on the damaged file and marks duplicates by single-end rules.

Task: mark the PCR duplicates with the usual samtools workflow (fixmate -m, sort, markdup) and
report how many reads carry the duplicate flag 0x400. The BAM has exact duplicate pairs (flagged
as pairs by a correct run: both mates of the copy) and groups of pairs that share the start of
read 1 but not the fragment length (not duplicates as pairs; when the mates are lost, every read 1
after the first of a group looks like a duplicate). Clean twin: records grouped by name, the way
the aligner wrote them. Faulted twin: the same records in a scrambled order with the same header.

Verified here (samtools 1.24, 160 pairs from bwa, 20 duplicate pairs added): fixmate -m on the
shuffled BAM (header SO:unsorted GO:query): exit 0, stderr empty, proper pairs 280 -> 2 of 320 reads
(RNEXT '*', PNEXT 0, TLEN 0 on every record), markdup marked 49 reads where the grouped BAM
gives 40.
The same fixmate on a coordinate-sorted BAM stops with "Coordinate sorted, require grouped/sorted by
queryname"; markdup on a BAM without ms tags stops with "no ms score tag"; markdup on a name-grouped
BAM stops with "not in coordinate sorted order". Only the fixmate step is silent.

  explicit  the README says the BAM is unsorted (clean: straight from the aligner, mates adjacent)
  implicit  the README only says the BAM is unsorted paired-end data
  forced    the sentence asks for the fixmate -m / sort / markdup workflow
"""

from __future__ import annotations

import posixpath
import random
import re
import shlex
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

READ_LEN = 50
CONTIGS = {"chr1": 3200, "chr2": 2600}
DUP = 0x400


@dataclass(frozen=True)
class Pair:
    name: str
    contig: str
    pos1: int  # leftmost base of the forward read (1-based)
    pos2: int  # leftmost base of the reverse read
    first_forward: bool  # read 1 is the forward read
    quality: str  # base quality character of both mates
    seqs: tuple[str, str]  # (forward read, reverse read)

    def records(self) -> list[str]:
        fwd_flag, rev_flag = (99, 147) if self.first_forward else (163, 83)
        tlen = self.pos2 + READ_LEN - self.pos1
        rows = []
        for flag, pos, mate, sign, seq in (
            (fwd_flag, self.pos1, self.pos2, 1, self.seqs[0]),
            (rev_flag, self.pos2, self.pos1, -1, self.seqs[1]),
        ):
            rows.append(
                f"{self.name}\t{flag}\t{self.contig}\t{pos}\t60\t{READ_LEN}M\t=\t{mate}\t"
                f"{sign * tlen}\t{seq}\t{self.quality * READ_LEN}"
            )
        return rows if self.first_forward else rows[::-1]


@dataclass(frozen=True)
class Plan:
    pairs: list[Pair]
    duplicates: int  # exact duplicate pairs
    shared_extra: int  # reads that look like duplicates once the mates are lost


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f36-{seed}")
    taken: set[tuple[str, str, int]] = set()  # (contig, strand, 5' coordinate)

    def free(contig: str, pos1: int, pos2: int) -> bool:
        return (contig, "+", pos1) not in taken and (contig, "-", pos2 + READ_LEN - 1) not in taken

    def claim(contig: str, pos1: int, pos2: int) -> None:
        taken.add((contig, "+", pos1))
        taken.add((contig, "-", pos2 + READ_LEN - 1))

    def seq() -> str:
        return "".join(rng.choices("ACGT", k=READ_LEN))

    pairs: list[Pair] = []

    def new_pair(contig: str | None = None, pos1: int | None = None) -> Pair:
        while True:
            c = contig or rng.choice(list(CONTIGS))
            frag = rng.randint(150, 320)
            p1 = pos1 or rng.randint(1, CONTIGS[c] - frag)
            if p1 + frag > CONTIGS[c]:
                continue
            p2 = p1 + frag - READ_LEN
            if free(c, p1, p2):
                claim(c, p1, p2)
                name = f"pair{len(pairs):03d}"
                return Pair(name, c, p1, p2, rng.random() < 0.5, "I", (seq(), seq()))

    for _ in range(rng.randint(60, 75)):
        pairs.append(new_pair())
    originals = rng.sample(pairs, rng.randint(8, 12))
    for orig in originals:  # exact duplicate pairs: same positions and orientation, lower quality
        pairs.append(
            Pair(
                f"pair{len(pairs):03d}",
                orig.contig,
                orig.pos1,
                orig.pos2,
                orig.first_forward,
                "F",
                (seq(), seq()),
            )
        )
    extra = 0
    for _ in range(rng.randint(3, 4)):  # pairs sharing the start of the forward read
        size = rng.randint(3, 4)
        anchor = new_pair()
        pairs.append(anchor)
        extra += size - 1
        for _ in range(size - 1):
            while True:
                frag = rng.randint(150, 320)
                p2 = anchor.pos1 + frag - READ_LEN
                end = (anchor.contig, "-", p2 + READ_LEN - 1)
                if p2 + READ_LEN - 1 <= CONTIGS[anchor.contig] and end not in taken:
                    taken.add(end)
                    break
            pairs.append(
                Pair(
                    f"pair{len(pairs):03d}",
                    anchor.contig,
                    anchor.pos1,
                    p2,
                    anchor.first_forward,
                    "I",
                    (seq(), seq()),
                )
            )
    rng.shuffle(pairs)
    return Plan(pairs, len(originals), extra)


def sam_text(plan: Plan, twin: str, seed: int) -> str:
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in CONTIGS.items()
    )
    rows = [row for pair in plan.pairs for row in pair.records()]
    if twin == "faulted":
        rng = random.Random(f"f36-order-{seed}")
        while True:  # mates that land next to each other would be paired by fixmate after all
            rng.shuffle(rows)
            if all(a.split("\t", 1)[0] != b.split("\t", 1)[0] for a, b in pairwise(rows)):
                break
    return header + "\n".join(rows) + "\n"


def write_unsorted_bam(work: Path, text: str, tools: ToolEnv, name: str = "reads") -> None:
    (work / f"{name}.sam").write_text(text)
    argv = ["samtools", "view", "--no-PG", "-b", "-o", f"{name}.bam", f"{name}.sam"]
    common.check(tools.run(argv, work), argv)
    (work / f"{name}.sam").unlink()


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_unsorted_bam(work, sam_text(plan, twin, seed), tools)
    explicit = tier == "explicit"
    if explicit:
        order = (
            "records are grouped by read name, mates next to each other, as the aligner wrote them"
            if twin == "clean"
            else "records are in no particular order, mates can be anywhere in the file"
        )
        line = (
            f"reads.bam: paired-end reads of 50 bases aligned to chr1 and chr2, unsorted; {order}."
        )
    else:
        line = "reads.bam: paired-end reads of 50 bases aligned to chr1 and chr2, unsorted."
    (work / "README.md").write_text(common.readme([line]))
    if twin == "clean":
        evidence = ("in reads.bam every read name appears in two consecutive records",)
    else:
        evidence = (
            "in reads.bam the two records of a read name are not next to each other",
            "samtools view reads.bam | head shows different read names in consecutive records",
        )
    prompt = common.prompt_for(
        route,
        "mark the PCR duplicates in reads.bam (duplicate read pairs: the same start and "
        "orientation for both mates) and report how many reads carry the duplicate flag",
        "use the samtools fixmate -m, sort and markdup workflow to mark the PCR duplicates in "
        "reads.bam (duplicate read pairs: the same start and orientation for both mates) and "
        "report how many reads carry the duplicate flag",
        "a JSON integer",
    )
    expected = 2 * plan.duplicates
    naive = expected if twin == "clean" else expected + plan.shared_extra
    truth = {"kind": "int", "expected": expected, "naive": naive}
    task_id = common.task_id("F36", seed, twin, tier, route)
    return Task(task_id, "F36", "record_order", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def duplicates(work: Path, tools: ToolEnv, prefix: str, source: str) -> int:
    steps = [
        ["samtools", "fixmate", "-m", source, f"{prefix}_fx.bam"],
        ["samtools", "sort", "-o", f"{prefix}_fxs.bam", f"{prefix}_fx.bam"],
        ["samtools", "markdup", f"{prefix}_fxs.bam", f"{prefix}_md.bam"],
    ]
    for argv in steps:
        common.check(tools.run(argv, work), argv)
    proc = tools.run(["samtools", "view", "-c", "-f", str(DUP), f"{prefix}_md.bam"], work)
    common.check(proc, ["samtools", "view", "-c"])
    return int(proc.stdout.strip())


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["samtools", "sort", "-n", "-o", "ref_names.bam", "reads.bam"]
    common.check(tools.run(argv, work), argv)
    n = duplicates(work, tools, "ref", "ref_names.bam")
    return _result(n, "samtools sort -n, fixmate -m, sort, markdup")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(duplicates(work, tools, "naive", "reads.bam"), "fixmate -m, sort, markdup")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: `samtools fixmate` whose input is not grouped by read name. A file is grouped when an
# earlier call of the session wrote it with `samtools sort -n` / `-N` or `samtools collate` (its
# `-o FILE`, a `> FILE` redirect, the collate prefix), or when the input is `-` and the previous
# stage of the pipeline is such a call writing to stdout. A coordinate sort of the file takes that
# back. The options of each call are parsed (`-@ N`, `-o FILE`, bundles such as `-no n.bam`), so
# their order does not matter, and the name sort is tied to fixmate's own input: sorting another
# file by name does not protect `fixmate reads.bam`.
CALL = re.compile(r"\bsamtools\s+(sort|collate|fixmate)\b")
REDIRECT = re.compile(r"(?<![\w&])1?>>?\s*([^\s|;&<>]+)")  # the stdout redirect of a call
REDIRECTS = re.compile(r"\d*>>?&?\s*[^\s|;&<>]*|<\s*[^\s|;&<>]+")  # every redirect, to drop
SEGMENT = re.compile(r"&&|\|\||;|\n")
# short options that take a value, per subcommand (`-O` is a flag of collate, a format elsewhere)
VALUE = {"sort": "lmKIwtoTO@", "collate": "olnrT@", "fixmate": "Oz@"}
LONG_VALUE = frozenset(
    "--threads --verbosity --reference --sanitize --input-fmt-option --output-fmt "
    "--output-fmt-option".split()
)


def _options(words: list[str], valued: str) -> tuple[dict[str, str], list[str]]:
    """(option letter -> value, positional words) of the words after a samtools subcommand."""
    opts: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        i += 1
        if word.startswith("--"):
            if word in LONG_VALUE:  # `--threads 4`; `--threads=4` is one word
                i += 1
        elif word.startswith("-") and len(word) > 1:
            for k, letter in enumerate(word[1:], start=2):
                if letter not in valued:
                    opts[letter] = ""
                elif word[k:]:  # `-@4`, `-no n.bam`: the rest of the word is the value
                    opts[letter] = word[k:]
                    break
                else:  # `-o n.bam`: the next word is
                    opts[letter] = words[i] if i < len(words) else ""
                    i += 1
                    break
        else:
            positional.append(word)
    return opts, positional


def _name(path: str) -> str:
    return posixpath.normpath(path.strip("\"'")).removeprefix("/work/")


def _call(stage: str) -> tuple[str, dict[str, str], list[str], str | None] | None:
    """(subcommand, options, positional words, stdout redirect) of the sort, collate or fixmate
    in one pipeline stage, or None."""
    match = CALL.search(stage)
    if match is None:
        return None
    text = stage[match.end() :]
    redirect = REDIRECT.search(text)
    text = REDIRECTS.sub(" ", text)
    try:
        words = shlex.split(text)
    except ValueError:  # a quote that closes outside the stage, as in `bash -c "...|..."`
        words = text.replace('"', " ").replace("'", " ").split()
    opts, positional = _options(words, VALUE[match.group(1)])
    return match.group(1), opts, positional, redirect.group(1) if redirect else None


def exposed(commands: list[str], work: Path) -> bool:
    grouped: set[str] = set()  # files written by a name sort or collate, not sorted since
    for command in commands:
        for statement in SEGMENT.split(command):
            piped = False  # the previous stage writes name-grouped records to stdout
            for stage in statement.split("|"):
                call = _call(stage)
                if call is None:
                    continue
                sub, opts, positional, redirect = call
                if sub == "fixmate":
                    source = positional[0] if positional else "-"
                    if not (piped if source == "-" else _name(source) in grouped):
                        return True
                    continue
                by_name = sub == "collate" or bool({"n", "N"} & opts.keys())
                stdout = "O" in opts if sub == "collate" else "o" not in opts
                if stdout:
                    target = redirect
                elif "o" in opts:
                    target = opts["o"]
                else:  # `collate in.bam prefix` writes prefix.bam
                    target = positional[1] + ".bam" if len(positional) > 1 else None
                if target:
                    (grouped.add if by_name else grouped.discard)(_name(target))
                piped = by_name and stdout and not target
    return False


SPEC = FamilySpec(
    id="F36",
    slot="record_order",
    source="samtools fixmate manual (input must be sorted by name); GEN-32 neighbourhood",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.bam", "README.md"}),
    locus="data",
)
