"""F65 split_reciprocal (tool_logic, tool default): `bedtools intersect -split -r -f` drops pairs.

The incident (GEN-40, arq5x/bedtools2#1142, 2026): with `-split` and a reciprocal fraction
(`-r -f`, or `-F`), a perfect transcript pair can be dropped when ANOTHER record of the b file lies
inside the span of the same a record, even in an intron where it touches no exon. Nothing is
printed: exit 0, empty stderr, a plausible BED12 listing that is merely shorter.

Verified here (bedtools 2.31.1, BED12 transcripts of 2-3 exons of 150-260 bases, 600-700 exon
bases): one a record A1 and its exact copy P in b: `-a a.bed -b b.bed -split -r -f 0.9 -wa -wb`
reports the pair. Add a second b record D of 300 bases inside an intron of A1 (no exon touched):
the pair disappears, for `-split -r -f 0.9`, `-F 0.9` and `-f 0.9 -F 0.9` (with or without `-u`);
`-split -f 0.9` alone and `-split -f 0.5 -r` keep it. D of 20 or 60 bases (under about 10 percent
of the exon length) does not trigger it, D of 150 bases and more does. D placed 500 bases past the
end of A1, or touching its end, changes nothing. Running one `intersect` per b record gives the
same pairs as an exon-level Python oracle on all seeds (the reference solver does it).

The twins hold the same transcripts and partners. About half of the a transcripts that have a true
partner get an extra single-exon b record of 260-330 bases: in the faulted twin inside an intron
of that transcript, in the clean twin 500 bases past its end. No decoy touches an exon of any a
record, so the exon-level answer is identical. The records of tx_b.bed are named tx_b_001, ... in
position order, partners and extras alike. The builder runs the one-call route on both twins and
fails when the faulted twin returns the expected names (the trap is a bug of this bedtools version)
or the clean twin does not.

  explicit  the README says that the exons are the BED12 blocks
  implicit  the README only says "transcript models"
  forced    the sentence asks for `bedtools intersect -split -r -f 0.9`
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
from biowarden.bench.tasks.extra.f53_vcftools_bed import normalise, pipelines
from biowarden.bench.tasks.tools import ToolEnv

FRACTION = 0.9
SPACING = 7000
Exons = list[tuple[int, int]]


@dataclass(frozen=True)
class Tx:
    name: str
    chrom: str
    exons: Exons

    @property
    def start(self) -> int:
        return self.exons[0][0]

    @property
    def end(self) -> int:
        return self.exons[-1][1]


def bed12_line(tx: Tx, strand: str = "+") -> str:
    sizes = ",".join(str(b - a) for a, b in tx.exons)
    starts = ",".join(str(a - tx.start) for a, b in tx.exons)
    return (
        f"{tx.chrom}\t{tx.start}\t{tx.end}\t{tx.name}\t0\t{strand}\t{tx.start}\t{tx.end}\t0\t"
        f"{len(tx.exons)}\t{sizes},\t{starts},\n"
    )


def bed12_text(records: list[Tx]) -> str:
    ordered = sorted(records, key=lambda t: (t.chrom, t.start, t.name))
    return "".join(bed12_line(t) for t in ordered)


def shared_bases(a: Exons, b: Exons) -> int:
    return sum(max(0, min(ae, be) - max(as_, bs)) for as_, ae in a for bs, be in b)


def reciprocal(a: Tx, b: Tx) -> bool:
    """Exon-level reciprocal overlap: shared exon bases >= FRACTION of each transcript's exons."""
    if a.chrom != b.chrom:
        return False
    shared = shared_bases(a.exons, b.exons)
    total_a = sum(e - s for s, e in a.exons)
    total_b = sum(e - s for s, e in b.exons)
    return shared >= FRACTION * total_a and shared >= FRACTION * total_b


@dataclass(frozen=True)
class Plan:
    a: list[Tx]
    partners: list[Tx]  # real b records (match or near miss), not decoys
    decoy_targets: list[str]  # names of the a records that get a decoy
    decoys_faulted: list[Tx]
    decoys_clean: list[Tx]


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f65-{seed}")
    n_a = rng.randint(26, 34)
    a: list[Tx] = []
    for i in range(n_a):
        chrom = "chr1" if i % 2 == 0 else "chr2"
        cursor = 2000 + (i // 2) * SPACING + rng.randint(0, 300)
        exons: Exons = []
        for _ in range(rng.choice([2, 3, 3])):
            size = rng.randint(150, 260)
            exons.append((cursor, cursor + size))
            cursor += size + rng.randint(560, 900)
        a.append(Tx(f"tx_a_{i + 1:03d}", chrom, exons))
    kinds = rng.choices(["match", "near", "none"], weights=[5, 2, 3], k=n_a)
    partners: list[Tx] = []
    for tx, kind in zip(a, kinds, strict=True):
        name = tx.name.replace("tx_a", "tx_b")
        if kind == "match":
            jitter = [(s + rng.randint(-4, 4), e + rng.randint(-4, 4)) for s, e in tx.exons]
            partners.append(Tx(name, tx.chrom, jitter))
        elif kind == "near":
            if rng.random() < 0.5:  # the last exon moved 130 bases away
                last = (tx.exons[-1][0] + 130, tx.exons[-1][1] + 130)
                partners.append(Tx(name, tx.chrom, [*tx.exons[:-1], last]))
            else:  # the first exon is missing
                partners.append(Tx(name, tx.chrom, tx.exons[1:]))
    matched = [tx for tx in a if any(reciprocal(tx, p) for p in partners)]
    count = max(4, round(0.55 * len(matched)))
    chosen = rng.sample(matched, min(count, len(matched)))
    chosen.sort(key=lambda t: t.name)
    faulted, clean = [], []
    for k, tx in enumerate(chosen):
        size = rng.randint(260, 330)
        gaps = [(tx.exons[i][1], tx.exons[i + 1][0]) for i in range(len(tx.exons) - 1)]
        gap_start, gap_end = max(gaps, key=lambda g: g[1] - g[0])
        start = gap_start + rng.randint(80, gap_end - gap_start - size - 80)
        faulted.append(Tx(f"tx_b_decoy_{k + 1:03d}", tx.chrom, [(start, start + size)]))
        past = tx.end + 500
        clean.append(Tx(f"tx_b_decoy_{k + 1:03d}", tx.chrom, [(past, past + size)]))
    return Plan(a, partners, [t.name for t in chosen], faulted, clean)


def expected_names(plan: Plan) -> list[str]:
    return sorted(tx.name for tx in plan.a if any(reciprocal(tx, p) for p in plan.partners))


def b_records(plan: Plan, twin: str) -> list[Tx]:
    """The records of tx_b.bed, named tx_b_001... in position order: one name pattern, so a name
    does not tell a partner from an extra record."""
    records = plan.partners + (plan.decoys_clean if twin == "clean" else plan.decoys_faulted)
    ordered = sorted(records, key=lambda t: (t.chrom, t.start, t.end, t.name))
    return [Tx(f"tx_b_{i + 1:03d}", t.chrom, t.exons) for i, t in enumerate(ordered)]


def single_call(tools: ToolEnv, work: Path) -> list[str]:
    """The names of the a records reported by the one-call route (real bedtools)."""
    argv = [
        "bedtools", "intersect", "-a", "tx_a.bed", "-b", "tx_b.bed",
        "-split", "-r", "-f", str(FRACTION), "-wa", "-u",
    ]  # fmt: skip
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sorted({line.split("\t")[3] for line in proc.stdout.splitlines() if line.strip()})


def per_record(tools: ToolEnv, work: Path) -> list[str]:
    """One `intersect` per b record: the pairs do not depend on the other b records."""
    names: set[str] = set()
    lines = [x for x in (work / "tx_b.bed").read_text().splitlines() if x.strip()]
    for i, line in enumerate(lines):
        (work / f"_one_{i}.bed").write_text(line + "\n")
        argv = [
            "bedtools", "intersect", "-a", "tx_a.bed", "-b", f"_one_{i}.bed",
            "-split", "-r", "-f", str(FRACTION), "-wa", "-u",
        ]  # fmt: skip
        proc = tools.run(argv, work)
        (work / f"_one_{i}.bed").unlink()
        common.check(proc, argv)
        names |= {x.split("\t")[3] for x in proc.stdout.splitlines() if x.strip()}
    return sorted(names)


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    (work / "tx_a.bed").write_text(bed12_text(plan.a))
    (work / "tx_b.bed").write_text(bed12_text(b_records(plan, twin)))
    lines = [
        "tx_a.bed: transcript models of annotation A, BED12 (sorted).",
        "tx_b.bed: transcript models of annotation B, BED12 (sorted).",
    ]
    if tier == "explicit":
        lines.append(
            "The exons of a transcript are its BED12 blocks; the gaps between them are introns."
        )
    (work / "README.md").write_text(common.readme(lines))
    expected = expected_names(plan)
    naive = single_call(tools, work)
    if (naive == expected) != (twin == "clean"):
        version = tools.run(["bedtools", "--version"], work).stdout.strip()
        raise RuntimeError(
            f"F65 seed {seed}: with {version} the one-call route "
            + ("drops pairs of the clean twin" if twin == "clean" else "keeps every pair")
            + f", so the {twin} twin does not behave as built (bedtools 2.31.1 drops the pairs "
            "of the faulted twin only)"
        )
    if twin == "clean":
        evidence: tuple[str, ...] = ()
    else:
        k = len(plan.decoy_targets)
        evidence = (
            f"{k} transcripts of tx_a.bed have a second tx_b.bed record inside an intron: "
            "bedtools intersect -split -r -f loses their matches",
        )
    what = (
        "the names of the transcripts in tx_a.bed that have a transcript in tx_b.bed whose exons "
        "overlap theirs reciprocally by at least 90% (the shared exon bases are at least 90% of "
        "the total exon length of each of the two transcripts)"
    )
    prompt = common.prompt_for(
        route,
        f"list {what}",
        f"use `bedtools intersect -split -r -f 0.9` to list {what}",
        "a JSON list of transcript names",
    )
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F65", seed, twin, tier, route)
    return Task(task_id, "F65", "tool_logic", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    names = per_record(tools, work)
    return _result(names, "bedtools intersect -split -r -f 0.9 run once per tx_b record")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(single_call(tools, work), "bedtools intersect -split -r -f 0.9 -wa -u")


SPLIT = re.compile(r"(?<![\w-])-split\b")
FRACTION_OPT = re.compile(r"(?<![\w-])-(?:r|F)\b")
LOOP_WORD = re.compile(r"\b(for|while|until|do|done)\b")
PER_RECORD = re.compile(r"\bxargs\b|\bparallel\b")


def without_loops(text: str) -> str:
    """`text` without its `for`/`while`/`until` ... `do` ... `done` blocks, nested ones included; a
    loop that is never closed stays (so does a Python `for`, which has no `do`)."""
    kept: list[str] = []
    last, start, depth = 0, -1, 0
    for hit in LOOP_WORD.finditer(text):
        word = hit.group(1)
        if word in ("for", "while", "until"):
            start = hit.start() if depth == 0 else start
        elif word == "do":
            start = hit.start() if depth == 0 and start < 0 else start
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                kept += [text[last:start], " "]
                last, start = hit.end(), -1
    return "".join([*kept, text[last:]])


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff one pipeline runs `bedtools intersect -split` with -r or -F on the two files in
    a single call, outside a loop over the b records: a for/while/until block, or xargs/parallel in
    the pipeline, is the per-record route. Judged per pipeline, so a loop in another statement of
    the command does not hide the single call."""
    for command in commands:
        for part in pipelines(without_loops(normalise(command))):
            if "intersect" not in part or "bedtools" not in part or PER_RECORD.search(part):
                continue
            if SPLIT.search(part) and FRACTION_OPT.search(part):
                return True
    return False


SPEC = FamilySpec(
    id="F65",
    slot="tool_logic",
    source="GEN-40 (bedtools2#1142: intersect -split -r -f drops a perfect pair)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"tx_b.bed"}),
    locus="tool_default",
)
