"""F29 mapq_unique (mapq_semantics, tool default): in STAR output MAPQ > 0 does not mean unique.

The folklore from bwa and bowtie2 is "MAPQ 0 means the read maps to several places, so `-q 1` keeps
the uniquely mapped reads". STAR does not follow it: a uniquely mapped read has MAPQ 255, a read
that maps to 2 loci has MAPQ 3, to 3-4 loci MAPQ 1 and only to 5-10 loci MAPQ 0. A MAPQ filter of
1, 2 or 3 therefore keeps the multi-mapped reads (and `-q 1` is what most scripts write). Nothing
warns: STAR exits 0, the SAM is valid, `samtools view -c -q 1` prints a plausible count. (The same
folklore breaks on bowtie2 and HISAT2, where multi-mapped reads get MAPQ 1; STAR is used here
because its progress output says nothing about the number of multi-mapped reads.)

Task: how many reads of reads.fastq map to exactly one place of the genome. Both twins carry the
same genome (two repeat families: a 300 bp segment present twice and one present three times) and
the same STAR index, and the same unique reads. The faulted twin adds reads drawn from the repeats.

Verified here (STAR 2.7.11b, samtools 1.24; `--genomeDir .`, one thread, SAM output; seed 0 has 73
unique reads and 19 multi-mapped ones): the unique reads have MAPQ 255 (NH:i:1); a read that maps to
2 loci has MAPQ 3 on every record (primary flag 0 or 16, secondary 256 or 272, NH:i:2) and a read
that maps to 3 loci MAPQ 1 (NH:i:3). `samtools view -c -q 1 -F 0x100` counts every primary record,
so the answer is 92 = 73 + 19; `-q 4` or higher, `-q 255` and `-d NH:1` give 73. The alignments, not
the reads, are what `-c` counts: without -F 0x100 a 2-locus read counts twice. (An earlier 140-read
test set with the same repeat layout gave the same table for 2-locus reads.)

  explicit  the README names the repeats and says that reads from them map to 2 or 3 places
  implicit  the README only describes the files; Log.final.out (second run) and the genome carry it
  forced    the sentence asks for STAR followed by a `samtools view -c` MAPQ filter
"""

from __future__ import annotations

import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.tools import ToolEnv

READ_LEN = 100
UNIT = 300  # length of a repeat unit
UNSAFE_MAPQ = (1, 2, 3)  # STAR multi-mapper MAPQ values 3 and 1 survive a cutoff of 1..3

# ---- shared kit for the alignment pack (F29-F40) -----------------------------------------------

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choices("ACGT", k=n))


def mutate(rng: random.Random, seq: str, k: int, edge: int = 0) -> str:
    """`k` substitutions at distinct random positions, none within `edge` bases of either end.

    STAR can dodge mismatches in the first or last ~10 bases of a read by a spliced alignment with
    a 6-base anchor and the same score, which makes a unique read look multi-mapped (seen on seeds
    with 3-4 mismatches near an end); `edge=12` keeps the unique reads unambiguous.
    """
    out = list(seq)
    for i in rng.sample(range(edge, len(seq) - edge), k):
        out[i] = rng.choice([b for b in "ACGT" if b != out[i]])
    return "".join(out)


def fastq_text(reads: list[tuple[str, str]], qual: str = "I") -> str:
    return "".join(f"@{name}\n{seq}\n+\n{qual * len(seq)}\n" for name, seq in reads)


def fasta_text(genome: dict[str, str], width: int = 60) -> str:
    return "".join(f">{c}\n{textwrap.fill(s, width)}\n" for c, s in genome.items())


def write_genome(
    work: Path, genome: dict[str, str], tools: ToolEnv, name: str = "genome.fa"
) -> None:
    """Write the FASTA and its .fai."""
    (work / name).write_text(fasta_text(genome))
    argv = ["samtools", "faidx", name]
    common.check(tools.run(argv, work), argv)


def star_index(
    work: Path,
    tools: ToolEnv,
    fasta: str = "genome.fa",
    gtf: str | None = None,
    overhang: int = 99,
) -> None:
    """STAR index files directly in `work` (use --genomeDir .), logs and paths removed.

    Verified byte-reproducible across runs once Log.out (a timestamp) is deleted and the command
    line in the first line of genomeParameters.txt (it holds the path of the STAR binary) is
    replaced. STAR does not read that line. `--genomeChrBinNbits 8` and `--genomeSAindexNbases 5`
    keep a 6 kb genome at about 50 KB.
    """
    argv = [
        "STAR",
        "--runMode",
        "genomeGenerate",
        "--genomeDir",
        ".",
        "--genomeFastaFiles",
        fasta,
        "--genomeSAindexNbases",
        "5",
        "--genomeChrBinNbits",
        "8",
        "--runThreadN",
        "1",
        "--outFileNamePrefix",
        "gen_",
    ]
    if gtf:
        argv += ["--sjdbGTFfile", gtf, "--sjdbOverhang", str(overhang)]
    common.check(tools.run(argv, work), argv)
    for name in ("Log.out", "gen_Log.out"):
        (work / name).unlink(missing_ok=True)
    params = work / "genomeParameters.txt"
    lines = params.read_text().splitlines()
    lines[0] = "### STAR --runMode genomeGenerate --genomeDir . --genomeFastaFiles " + fasta
    params.write_text("\n".join(lines) + "\n")


def star_map(
    work: Path, tools: ToolEnv, reads: list[str], prefix: str, extra: tuple[str, ...] = ()
):
    """STAR against the index in `work`; outputs are `<prefix>Aligned.out.sam`, logs, tab files."""
    argv = [
        "STAR",
        "--genomeDir",
        ".",
        "--readFilesIn",
        *reads,
        "--runThreadN",
        "1",
        "--outFileNamePrefix",
        prefix,
        *extra,
    ]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return proc


def sam_records(text: str) -> list[list[str]]:
    return [line.split("\t") for line in text.splitlines() if line and not line.startswith("@")]


def tag_value(fields: list[str], tag: str) -> str | None:
    prefix = tag + ":"
    for field in fields[11:]:
        if field.startswith(prefix):
            return field.split(":", 2)[2]
    return None


def clean_up(work: Path, *patterns: str) -> None:
    for pattern in patterns:
        for path in work.glob(pattern):
            if path.is_file():
                path.unlink()


# ---- the plan ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    genome: dict[str, str]
    unique: list[tuple[str, str]]  # (name, sequence) of reads from unique regions
    repeats: list[tuple[str, str]]  # reads drawn from the repeat families
    order: list[int]  # file order over unique + repeats (faulted); clean keeps unique ones only
    loci: dict[str, int]  # repeat read name -> number of loci it maps to


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f29-{seed}")
    two, three = rand_seq(rng, UNIT), rand_seq(rng, UNIT)
    chr1_parts = [
        ("u", rand_seq(rng, rng.randint(620, 700))),
        ("2", two),
        ("u", rand_seq(rng, rng.randint(300, 400))),
        ("3", three),
        ("u", rand_seq(rng, rng.randint(300, 400))),
        ("3", three),
        ("u", rand_seq(rng, rng.randint(400, 520))),
    ]
    chr2_parts = [
        ("u", rand_seq(rng, rng.randint(500, 600))),
        ("2", two),
        ("u", rand_seq(rng, rng.randint(300, 380))),
        ("3", three),
        ("u", rand_seq(rng, rng.randint(550, 650))),
    ]
    genome = {
        "chr1": "".join(seq for _, seq in chr1_parts),
        "chr2": "".join(seq for _, seq in chr2_parts),
    }
    segments = [seq for parts in (chr1_parts, chr2_parts) for kind, seq in parts if kind == "u"]
    unique: list[tuple[str, str]] = []
    for i in range(rng.randint(60, 80)):
        seg = rng.choices(segments, weights=[len(s) for s in segments])[0]
        start = rng.randint(0, len(seg) - READ_LEN)
        read = mutate(
            rng, seg[start : start + READ_LEN], rng.choice([0, 0, 1, 1, 2, 3, 4]), edge=12
        )
        unique.append((f"read{i:03d}", revcomp(read) if rng.random() < 0.5 else read))
    repeats: list[tuple[str, str]] = []
    loci: dict[str, int] = {}
    for kind, unit, count in (("2", two, rng.randint(9, 13)), ("3", three, rng.randint(7, 10))):
        for _ in range(count):
            start = rng.randint(0, UNIT - READ_LEN)
            read = unit[start : start + READ_LEN]
            name = f"read{len(unique) + len(repeats):03d}"
            repeats.append((name, revcomp(read) if rng.random() < 0.5 else read))
            loci[name] = int(kind)
    order = list(range(len(unique) + len(repeats)))
    rng.shuffle(order)
    return Plan(genome, unique, repeats, order, loci)


def reads_for(plan: Plan, twin: str) -> list[tuple[str, str]]:
    everything = plan.unique + plan.repeats
    keep = len(plan.unique) if twin == "clean" else len(everything)
    return [everything[i] for i in plan.order if i < keep]


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    write_genome(work, plan.genome, tools)
    star_index(work, tools)
    (work / "reads.fastq").write_text(fastq_text(reads_for(plan, twin)))
    explicit = tier == "explicit"
    lines = [
        "reads.fastq: single-end reads of 100 bases (FASTQ).",
        "genome.fa (with genome.fa.fai): the reference, contigs chr1 and chr2. The STAR 2.7.11b "
        "index of genome.fa is in this directory (Genome, SA, SAindex, ...): use --genomeDir .",
    ]
    n_unique = len(plan.unique)
    n_repeat = len(plan.repeats)
    if twin == "clean":
        if explicit:
            lines.append("Every read comes from a region of the genome that occurs once.")
        evidence = (
            ("README states that every read comes from a region that occurs once",)
            if explicit
            else ()
        ) + ("Log.final.out of a STAR run reports 0 reads mapped to multiple loci",)
    else:
        if explicit:
            lines.append(
                "Some reads come from two repeat families of the genome: a 300 bp segment that "
                "occurs twice (chr1 and chr2) and another that occurs three times (twice on chr1, "
                "once on chr2). Those reads map to 2 or 3 places."
            )
        evidence = (
            ("README names the repeat families that some reads come from",) if explicit else ()
        ) + (
            f"Log.final.out of a STAR run reports {n_repeat} reads mapped to multiple loci",
            "STAR gives multi-mapped reads MAPQ 3 (2 loci) or 1 (3-4 loci), not 0",
        )
    (work / "README.md").write_text(common.readme(lines))
    prompt = common.prompt_for(
        route,
        "count the reads in reads.fastq that map to exactly one place in the genome "
        "(uniquely mapped reads)",
        "use STAR to align reads.fastq to the genome and `samtools view -c` with a MAPQ filter "
        "(-q) to count the reads that map to exactly one place in the genome (uniquely mapped "
        "reads)",
        "a JSON integer",
    )
    naive = n_unique if twin == "clean" else n_unique + n_repeat
    truth = {"kind": "int", "expected": n_unique, "naive": naive}
    task_id = common.task_id("F29", seed, twin, tier, route)
    return Task(task_id, "F29", "mapq_semantics", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    star_map(work, tools, ["reads.fastq"], "ref_")
    text = (work / "ref_Aligned.out.sam").read_text()
    unique = {f[0] for f in sam_records(text) if not int(f[1]) & 4 and tag_value(f, "NH") == "1"}
    return _result(len(unique), "STAR, reads whose records carry NH:i:1")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    star_map(work, tools, ["reads.fastq"], "naive_")
    argv = ["samtools", "view", "-c", "-q", "1", "-F", "0x100", "naive_Aligned.out.sam"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return _result(int(proc.stdout.strip()), "samtools view -c -q 1 -F 0x100 (MAPQ > 0 is unique)")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a command that keeps alignments by a MAPQ cutoff of 1, 2 or 3 (samtools view -q /
# --min-MQ, awk on column 5) without looking at NH or at MAPQ 255. A cutoff of 4 or more, `-q 255`,
# `-d NH:1` and NH:i:1 greps are the safe routes. Applied per command.
VIEW = re.compile(r"\bsamtools\s+view\b")
Q_OPTION = re.compile(r"(?:(?<=\s)-[A-Za-z0-9]*q\s*|--min-MQ[\s=]\s*)(\d+)")
AWK_MAPQ = re.compile(r"\$5\s*(>=|>)\s*(\d+)")
SAFE = re.compile(r"\bNH\b|\b255\b|XS:i")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def _cutoffs(segment: str) -> list[int]:
    found: list[int] = []
    if VIEW.search(segment):
        found += [int(v) for v in Q_OPTION.findall(segment)]
    for op, value in AWK_MAPQ.findall(segment):
        found.append(int(value) + (1 if op == ">" else 0))
    return found


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if SAFE.search(segment):
                continue
            if any(c in UNSAFE_MAPQ for c in _cutoffs(segment)):
                return True
    return False


SPEC = FamilySpec(
    id="F29",
    slot="mapq_semantics",
    source="STAR manual (MAPQ 255 unique, 3 for 2 loci, 1 for 3-4, 0 above); bwa/bowtie2 folklore",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"reads.fastq", "README.md"}),
    locus="tool_default",
)
