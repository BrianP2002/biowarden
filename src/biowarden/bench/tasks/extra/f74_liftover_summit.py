"""F74 liftover_summit (locus tool_default): liftOver does not touch the extra BED columns.

UCSC `liftOver -bedPlus=6` lifts the first six columns of a narrowPeak file and copies columns 7-10
unchanged. Column 10 is the summit as an offset from the peak start; after an indel inside the peak
the lifted start and the old offset no longer point at the summit base, so `start + offset` is off
by the size of the indel. Two more things happen on the same route: an interval spanning an
insertion in the new assembly simply grows, and one with more than 5 percent of its bases deleted in
the new assembly is not lifted at all (default `-minMatch=0.95`) but written to the unmapped file.

Verified on UCSC liftOver (bio-a) with hand-built chain files (exit 0, stderr only "Reading liftover
chains" and "Mapping coordinates"):
  chain chr1 2000 -> 2050 with a 100-base insertion at old 500: the peak chr1 450 550 (summit offset
    80, old summit 530) is written as chr1 450 650 with the offset still 80; the summit is at new
    630, `start + offset` says 530;
  the peak chr1 1200 1400 (50 deleted bases, 25 percent) lands in the unmapped file
    ("#Partially deleted in new"); with -minMatch=0.5 it is lifted as 1300 1450;
  `-bedPlus=6` is needed for a ten-column file (without it liftOver stops: "Expecting integer field
    7", exit 255);
  lifting the summit as its own one-base interval (start + offset, +1) gives the right new position.

The twins hold the same summits, the same chain and the same peak names; only the peak extents (and
so the offset column) differ. Clean: every peak lies inside one aligned block, a shift is all
liftOver has to apply and `start + offset` is right. Faulted: every peak starts before an indel of
the chain and its summit lies after it, so the stale offset is wrong by the size of the indel (an
insertion of 30-80 bases or a deletion of 3-8 bases), and the peaks with a large deletion (30-45
bases, more than 5 percent) are not lifted by default.

  explicit  the README says that column 10 is the summit offset from the start
  implicit  the README only says "narrowPeak"
  forced    the task sentence dictates liftOver
"""

from __future__ import annotations

import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.extra.f53_vcftools_bed import normalise
from biowarden.bench.tasks.tools import ToolEnv

TWIN_FILES = frozenset({"peaks_old.narrowPeak"})
KINDS = {
    "chr1": ["ins", "del_small", "ins", "del_large", "ins"],
    "chr2": ["ins", "del_small", "del_large", "ins"],
}
TAIL = 150  # unaligned bases after the last chain block, on both assemblies


@dataclass(frozen=True)
class PeakPlan:
    chrom: str
    name: str
    kind: str
    k: int  # size of the indel before the summit's block
    block_t: int  # old start of the block that holds the summit
    shift: int  # new minus old coordinate inside that block
    summit_old: int
    start_clean: int
    start_faulted: int
    end: int
    values: tuple[float, int, float, float]  # fold, scaled score, -log10 p, -log10 q

    @property
    def summit_new(self) -> int:
        return self.summit_old + self.shift


def make_plan(seed: int) -> tuple[list[PeakPlan], str]:
    """(peaks of both twins, chain file text)."""
    rng = random.Random(f"f74-{seed}")
    peaks: list[PeakPlan] = []
    chains: list[str] = []
    number = 0
    for chain_id, (chrom, kinds) in enumerate(KINDS.items(), start=1):
        kinds = list(kinds)
        rng.shuffle(kinds)
        t = q = 0
        length = rng.randint(300, 500)
        lines: list[str] = []
        for kind in kinds:
            k = {
                "ins": rng.randint(30, 80),
                "del_small": rng.randint(3, 8),
                "del_large": rng.randint(30, 45),
            }[kind]
            dt, dq = (0, k) if kind == "ins" else (k, 0)
            lines.append(f"{length}\t{dt}\t{dq}")
            t += length + dt
            q += length + dq
            d = rng.randint(110, 160)
            back_clean = rng.randint(70, 90)
            back_insertion = rng.randint(30, 70)
            back_deletion = k + rng.randint(10, 40)
            fold = round(rng.uniform(2.0, 14.0), 2)
            qval = round(rng.uniform(4.0, 38.0), 3)
            pval = round(qval + rng.uniform(0.5, 3.0), 3)
            number += 1
            summit = t + d
            peaks.append(
                PeakPlan(
                    chrom,
                    f"peak{number:02d}",
                    kind,
                    k,
                    t,
                    q - t,
                    summit,
                    summit - back_clean,
                    t - (back_insertion if kind == "ins" else back_deletion),
                    summit + 100,
                    (fold, min(1000, int(10 * qval)), pval, qval),
                )
            )
            length = d + 100 + 140 + rng.randint(0, 150)
        lines.append(str(length))
        t += length
        q += length
        header = f"chain {1000 * chain_id} {chrom} {t + TAIL} + 0 {t} {chrom} {q + TAIL} + 0 {q}"
        chains.append("\n".join([f"{header} {chain_id}", *lines]) + "\n")
    return peaks, "\n".join(chains)


def narrow_peak_text(peaks: list[PeakPlan], twin: str) -> str:
    rows = []
    for p in sorted(peaks, key=lambda p: (p.chrom, p.summit_old)):
        start = p.start_clean if twin == "clean" else p.start_faulted
        fold, score, pval, qval = p.values
        rows.append(
            f"{p.chrom}\t{start}\t{p.end}\t{p.name}\t{score}\t.\t{fold}\t{pval}\t{qval}\t"
            f"{p.summit_old - start}\n"
        )
    return "".join(rows)


def lift(work: Path, tools: ToolEnv, options: list[str], text: str) -> dict[str, list[str]]:
    """Run liftOver on `text` (BED lines) with the workspace chain; name -> lifted columns."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out, unmapped = (Path(tmp) / n for n in ("in.bed", "out.bed", "unmapped.bed"))
        src.write_text(text)
        argv = ["liftOver", *options, str(src), "old_to_new.chain", str(out), str(unmapped)]
        proc = tools.run(argv, work)
        if proc.returncode != 0:
            raise RuntimeError(f"liftOver failed: {proc.stderr}")
        rows = [line.split("\t") for line in out.read_text().splitlines()]
    return {r[3]: r for r in rows}


def naive_positions(work: Path, tools: ToolEnv, name: str = "peaks_old.narrowPeak") -> dict:
    """What lifting the narrowPeak file and adding column 10 to the lifted start gives."""
    lifted = lift(work, tools, ["-bedPlus=6"], (work / name).read_text())
    return {n: int(r[1]) + int(r[9]) for n, r in lifted.items()}


def summit_positions(work: Path, tools: ToolEnv, name: str = "peaks_old.narrowPeak") -> dict:
    """The summit lifted as a one-base interval of its own."""
    rows = [line.split("\t") for line in (work / name).read_text().splitlines()]
    text = "".join(
        f"{r[0]}\t{int(r[1]) + int(r[9])}\t{int(r[1]) + int(r[9]) + 1}\t{r[3]}\n" for r in rows
    )
    return {n: int(r[1]) for n, r in lift(work, tools, [], text).items()}


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    peaks, chain = make_plan(seed)
    (work / "old_to_new.chain").write_text(chain)
    (work / "peaks_old.narrowPeak").write_text(narrow_peak_text(peaks, twin))
    peaks_line = (
        "peaks_old.narrowPeak: MACS3 peaks on the OLD assembly, narrowPeak format (BED6+4)."
    )
    if tier == "explicit":
        peaks_line += " Column 10 is the summit position as an offset from the peak start."
    lines = [
        peaks_line,
        "old_to_new.chain: UCSC chain file that converts OLD assembly coordinates (chr1, chr2) to "
        "NEW assembly coordinates.",
    ]
    (work / "README.md").write_text(common.readme(lines))
    expected = {p.name: p.summit_new for p in peaks}
    naive = expected if twin == "clean" else naive_positions(work, tools)
    if twin == "faulted":
        hit = [p.name for p in peaks if p.kind != "del_large"]
        lost = [p.name for p in peaks if p.kind == "del_large"]
        evidence = (
            f"every peak starts before an indel of the chain and its summit lies after it; "
            f"liftOver -bedPlus=6 copies column 10, so the offset is stale for {', '.join(hit)}",
            f"{', '.join(lost)} lose more than 5 percent of their bases to a deletion and go to "
            "the unmapped file",
        )
    else:
        evidence = ("no peak spans an indel of the chain: one shift applies to the whole peak",)
    if tier == "explicit":
        evidence = ("README says that column 10 is the summit offset from the start", *evidence)
    what = (
        "lift the peaks to the new assembly and report, for each peak, the position of its summit "
        "base in the new assembly"
    )
    prompt = common.prompt_for(
        route,
        what,
        "use liftOver with the chain file to " + what,
        "a JSON object mapping each peak name to the 0-based position of its summit base in the "
        "new assembly (an integer)",
    )
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0.5}
    task_id = common.task_id("F74", seed, twin, tier, route)
    return Task(task_id, "F74", "derived_column", twin, seed, prompt, truth, evidence, tier, route)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    flags = ["summit lifted as its own 1-base interval: liftOver does not update column 10"]
    return _result(summit_positions(work, tools), "liftOver of the summit bases", flags)


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(naive_positions(work, tools), "liftOver -bedPlus=6, then start + column 10")


# ---- exposure --------------------------------------------------------------------------------

_STAGE = re.compile(r"\|\||&&|[|;&\n]")
_LIFTOVER = re.compile(r"(?<![\w.-])liftOver\b")
_RAW = re.compile(r"(?<![\w./-])(?:\./)?peaks_old\.narrowPeak(?![\w.-])")


def has_spanning_peak(work: Path) -> bool:
    """Does some peak of the workspace start before the summit's block (the offset goes stale)?"""
    try:
        rows = [
            line.split("\t") for line in (work / "peaks_old.narrowPeak").read_text().splitlines()
        ]
        chain = (work / "old_to_new.chain").read_text()
    except OSError:
        return False
    blocks: dict[str, list[tuple[int, int]]] = {}
    chrom, t = "", 0
    for line in chain.splitlines():
        if line.startswith("chain"):
            chrom, t = line.split()[2], 0
            blocks[chrom] = []
        elif line.strip():
            fields = [int(x) for x in line.split("\t")]
            blocks[chrom].append((t, t + fields[0]))
            t += fields[0] + (fields[1] if len(fields) > 1 else 0)
    for r in rows:
        start, summit = int(r[1]), int(r[1]) + int(r[9])
        home = next((b for b in blocks.get(r[0], []) if b[0] <= summit < b[1]), None)
        if home is not None and start < home[0]:
            return True
    return False


def exposed(commands: list[str], work: Path) -> bool:
    """Trapped iff a liftOver stage lifts peaks_old.narrowPeak itself (the summit offset then rides
    along unchanged) in a workspace where a peak spans an indel. Lifting a file derived from it
    (the summit bases) is the safe route; judged per stage."""
    if not has_spanning_peak(work):
        return False
    for command in commands:
        for stage in _STAGE.split(normalise(command)):
            if _LIFTOVER.search(stage) and _RAW.search(stage):
                return True
    return False


SPEC = FamilySpec(
    id="F74",
    slot="derived_column",
    source="UCSC liftOver: columns after the first six are copied, a summit offset goes stale",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="tool_default",
)
