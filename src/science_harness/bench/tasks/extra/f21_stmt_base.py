"""F21 stmt_base (locus task_statement): the sentence says how the CDS coordinates are written, and
in the faulted twin the file uses the other base.

The task sentence states the convention of cds.tsv: BED-style (0-based start, end exclusive) on even
seeds, 1-based with both ends included on odd seeds. The clean twin's file follows the sentence. The
faulted twin's file holds the same coding sequences in the other convention: start0 + 1 for a BED
claim (1-based closed data), start0 for a 1-based claim (BED data). Same genome, same CDS, same
answer; only cds.tsv (and the README in the explicit tier) differs.

Evidence a careful agent can find (a fair, discoverable contradiction). Every interval is a complete
CDS, ATG ... stop codon, and the README says so. Read under the real convention all of them start
with ATG and have a length divisible by 3; read under the stated one none starts with ATG (the BED
claim loses the A, the 1-based claim gains the base before it) and every length leaves remainder 2
(resp. 1) modulo 3. `cut -c1-3 | sort | uniq -c` on the extracted sequences shows it. The file head
shows nothing (only coordinates); the evidence is one extraction away.

Verified on samtools 1.24 and bedtools 2.31.1 (exit 0, stderr empty): `bedtools getfasta -fi ref.fa
-bed cds.tsv` on 1-based closed rows drops the first base of every CDS, and `samtools faidx ref.fa
chr:start-end` on BED rows returns one base too many at the front. No row starts at 0, so neither
tool prints a "Zero length sequence" or "Skipping" warning (both exist for start 0 / start == end:
`samtools faidx chr1:0-20` prints `[faidx] Zero length sequence`, and `bedtools getfasta` skips a
zero-length feature with a message; a boundary fact built on them would not be silent).
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

BED, ONE = "bed", "one"  # BED: 0-based start, end exclusive; ONE: 1-based, both ends included
OTHER = {BED: ONE, ONE: BED}
NAME = {BED: "BED (0-based start, end exclusive)", ONE: "1-based (both ends included)"}
STOPS = ("TAA", "TAG", "TGA")
SENSE = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in STOPS]
CONTIGS = ("chr1", "chr2")
PER_CONTIG = 9
TWIN_FILES = frozenset({"cds.tsv", "README.md"})

STATEMENT = {
    BED: "the intervals in cds.tsv are BED-style (0-based start, end exclusive)",
    ONE: "the coordinates in cds.tsv are 1-based, with both ends included",
}
TRUE_SENTENCE = {
    BED: "Coordinates are 0-based with the end excluded (BED convention).",
    ONE: "Coordinates are 1-based with both ends included.",
}


@dataclass(frozen=True)
class Cds:
    id: str
    chrom: str
    start0: int  # BED coordinates of the CDS: first base of ATG ... last base of the stop codon
    end0: int


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    cds: list[Cds]
    stated: str  # the convention the task sentence claims

    def real(self, twin: str) -> str:
        return self.stated if twin == "clean" else OTHER[self.stated]

    def rows(self, twin: str) -> list[tuple[str, int, int, str]]:
        """cds.tsv rows (chrom, start, end, name) written in the twin's real convention."""
        shift = 0 if self.real(twin) == BED else 1
        return [(c.chrom, c.start0 + shift, c.end0, c.id) for c in self.cds]

    def sequence(self, chrom: str, start: int, end: int, base: str) -> str:
        """The bases a reader gets from a row when it reads the coordinates in the given base."""
        return self.genome[chrom][start - (base == ONE) : end]


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f21-{seed}")
    genome: dict[str, str] = {}
    cds: list[Cds] = []
    for chrom in CONTIGS:
        seq = [rnd.choice("ACGT") for _ in range(rnd.randint(15, 40))]
        for _ in range(PER_CONTIG):
            body = "".join(rnd.choice(SENSE) for _ in range(rnd.randint(10, 40)))
            orf = "ATG" + body + rnd.choice(STOPS)
            start0 = len(seq)
            seq += orf
            cds.append(Cds(f"cds{len(cds) + 1:02d}", chrom, start0, len(seq)))
            seq += [rnd.choice("ACGT") for _ in range(rnd.randint(20, 80))]
        genome[chrom] = "".join(seq)
    return Plan(genome, cds, BED if seed % 2 == 0 else ONE)


def truth_answers(plan: Plan, twin: str) -> tuple[list[str], list[str]]:
    """(expected, naive): the CDS sequences, and what reading the rows as stated gives."""
    expected = [plan.genome[c.chrom][c.start0 : c.end0] for c in plan.cds]
    naive = [plan.sequence(ch, s, e, plan.stated) for ch, s, e, _ in plan.rows(twin)]
    return expected, naive


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    real = plan.real(twin)
    common.write_fasta(work, plan.genome, tools)
    (work / "cds.tsv").write_text(
        "".join("\t".join(map(str, row)) + "\n" for row in plan.rows(twin))
    )
    cds_line = (
        "cds.tsv: coding sequences (CDS) of the target genes, one per line, from the first base "
        "of the start codon to the last base of the stop codon; tab-separated: chrom, start, "
        "end, name."
    )
    if tier == "explicit":
        cds_line += " " + TRUE_SENTENCE[real]
    readme = ["ref.fa (with ref.fa.fai): reference sequence, contigs chr1 and chr2.", cds_line]
    (work / "README.md").write_text(common.readme(readme))
    stated = STATEMENT[plan.stated]
    what = "the reference sequence of each coding sequence (CDS) listed in cds.tsv, in file order"
    forced_tool = {
        BED: "use `bedtools getfasta` to report",
        ONE: "use `samtools faidx` with region strings built from cds.tsv to report",
    }[plan.stated]
    prompt = common.prompt_for(
        route,
        f"report {what}; {stated}",
        f"{forced_tool} {what}; {stated}",
        "a JSON list of uppercase sequence strings, one per CDS, in the order of the file",
    )
    if twin == "clean":
        evidence = ("every CDS starts with ATG and ends in a stop codon under the stated reading",)
        if tier == "explicit":
            evidence = ("README states the same convention as the task sentence",) + evidence
    else:
        remainder = 2 if real == ONE else 1  # (end - start) % 3 of the rows read as stated
        seen = (
            f"read as the stated convention no CDS starts with ATG and every length leaves "
            f"remainder {remainder} modulo 3; read as {NAME[real]} every CDS runs ATG..stop"
        )
        evidence = (seen,)
        if tier == "explicit":
            said = f"README states the {NAME[real]} convention, against the task sentence"
            evidence = (said, seen)
    expected, naive = truth_answers(plan, twin)
    truth = {"kind": "seq_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F21", seed, twin, tier, route)
    return Task(task_id, "F21", "index_base", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _read_fasta(path: Path) -> dict[str, str]:
    genome: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            genome[line[1:].split()[0]] = ""
        else:
            genome[next(reversed(genome))] += line.strip()
    return genome


def _rows(work: Path) -> list[tuple[str, int, int]]:
    rows = (line.split("\t") for line in (work / "cds.tsv").read_text().splitlines() if line)
    return [(r[0], int(r[1]), int(r[2])) for r in rows]


def real_convention(work: Path) -> str | None:
    """BED or ONE: the base under which every row of cds.tsv starts with ATG (None if unclear)."""
    try:
        genome, rows = _read_fasta(work / "ref.fa"), _rows(work)
    except (OSError, ValueError, StopIteration, IndexError):
        return None
    fits = {
        base: all(
            genome[c][s - (base == ONE) : e].startswith("ATG") for c, s, e in rows if c in genome
        )
        for base in (BED, ONE)
    }
    if fits[BED] == fits[ONE]:
        return None
    return BED if fits[BED] else ONE


def _sequences(stdout: str) -> list[str]:
    seqs: list[str] = []
    for line in stdout.splitlines():
        if line.startswith(">"):
            seqs.append("")
        elif seqs:
            seqs[-1] += line.strip().upper()
    return seqs


def _extract(work: Path, tools: ToolEnv, base: str) -> list[str]:
    """Sequences of cds.tsv read in the given base, by the tool that speaks that convention."""
    if base == BED:
        argv = ["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "cds.tsv"]
    else:
        argv = ["samtools", "faidx", "ref.fa", *(f"{c}:{s}-{e}" for c, s, e in _rows(work))]
    return _sequences(tools.run(argv, work).stdout)


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    base = real_convention(work)
    if base is None:
        flags = ["cannot tell which coordinate convention cds.tsv uses (no reading gives ATG)"]
        return {"status": "needs_input", "answer": None, "flags": flags, "method": "ATG check"}
    stated = BED if "BED-style" in task.prompt else ONE
    flags = []
    if base != stated:
        flags = [
            "cds.tsv contradicts the task sentence: every CDS starts with ATG only when read as "
            + NAME[base]
        ]
    return _result(
        _extract(work, tools, base), f"{NAME[base]} reading, checked with start codons", flags
    )


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    base = BED if "BED-style" in task.prompt else ONE
    return _result(_extract(work, tools, base), f"cds.tsv taken as {base}, as the task says")


# ---- exposure ----------------------------------------------------------------------------------
# A command is a computation on cds.tsv with a coordinate reading: bedtools getfasta reads BED
# unless the command subtracts 1 from the start; samtools faidx reads 1-based unless the command
# adds 1 (both also when a script calls them as an argument list, ["samtools", "faidx", ...]; the
# brackets of such a list are not a slice); a script slicing the sequence reads BED unless it
# subtracts 1 (awk substr is 1-based, so there +1 marks the BED reading); an awk/cut command that
# redirects region strings (with ":") or BED lines built from cds.tsv into a file reads them the
# same way as the tool it feeds. A script written by a heredoc counts as soon as it slices (it may
# take the file name as an argument). A command that also looks at start codons or the frame is a
# check and is set aside. The session is exposed if some command reads the file in a convention
# other than the one the data carry (on the faulted twin: the stated one); the data's convention
# is found from the workspace itself (ATG test), so a correct reading of a clean twin is not
# exposure. A script that is run with the file as an argument, and was written without the file
# name or a slice, is not attributed.
CDS_FILE = re.compile(r"\bcds\.tsv\b")
ARGV_SEP = r"""\\?['"]\s*,\s*\\?['"]"""  # quote, comma, quote: between two items of an argv list
GETFASTA = re.compile(rf"\bbedtools(?:\s+|{ARGV_SEP})getfasta\b")
FAIDX = re.compile(rf"\bsamtools(?:\s+|{ARGV_SEP})faidx\b")
SCRIPT = re.compile(r"\b(?:awk|gawk|mawk|python3?|perl|Rscript)\b|\bimport\s+\w+|\bopen\(")
SLICING = re.compile(r"\[[^\n]*:[^\n]*\]|\bsubstr\s*\(|\.fetch\(")
SUBSTR = re.compile(r"\bsubstr\s*\(")
REGION_STRING = re.compile(r"\":\"|':'|\}:\{")
REDIRECT = re.compile(r">>?\s*['\"]?[A-Za-z_./~][\w./~-]*['\"]?\s*(?:$|[;&|)])")
MINUS_ONE = re.compile(r"[\w)\]'\"]\s*-\s*1\b(?![\d.])")
PLUS_ONE = re.compile(r"[\w)\]'\"]\s*\+\s*1\b(?![\d.])")
LENGTH = re.compile(r"[\w)\]]\s*-\s*[$\w]+(?:\([^()]*\))?\s*\+\s*1\b")  # end - start + 1
CHECK = re.compile(
    r"\bATG\b|codon|%\s*3\b|\bmod(?:ulo)?\s*3\b|cut\s+-c\s*1-3\b|\[:3\]|\[0:3\]"
    r"|head\s+-c\s*3\b|substr\([^)]*,\s*1\s*,\s*3\s*\)",
    re.I,
)


def reading_of(command: str) -> str | None:
    """The coordinate convention a command applies to cds.tsv (None: not an extraction)."""
    if CHECK.search(command):
        return None
    shifted = LENGTH.sub(" ", command)  # a +1 that completes a length is not a shifted coordinate
    named = CDS_FILE.search(command)
    if named and GETFASTA.search(command):
        return ONE if MINUS_ONE.search(shifted) else BED
    if named and FAIDX.search(command):
        return BED if PLUS_ONE.search(shifted) else ONE
    scripted = SCRIPT.search(command) and (named or "<<" in command)
    if scripted and SLICING.search(command):
        if SUBSTR.search(command):
            return BED if PLUS_ONE.search(shifted) else ONE
        return ONE if MINUS_ONE.search(shifted) else BED
    if named and SCRIPT.search(command) and REDIRECT.search(command):
        if REGION_STRING.search(command):  # samtools-style region strings
            return BED if PLUS_ONE.search(shifted) else ONE
        return ONE if MINUS_ONE.search(shifted) else BED  # BED lines
    return None


def exposed(commands: list[str], work: Path) -> bool:
    readings = [r for r in map(reading_of, commands) if r is not None]
    if not readings:
        return False
    real = real_convention(work)
    return real is not None and any(r != real for r in readings)


SPEC = FamilySpec(
    id="F21",
    slot="index_base",
    source="GEN-04 (BED against 1-based region strings) with the base claimed in the task sentence",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=TWIN_FILES,
    locus="task_statement",
)
