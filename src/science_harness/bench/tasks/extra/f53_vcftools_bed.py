"""F53 vcftools_bed (input_format, tool default): `vcftools --bed` skips the first line.

`vcftools --bed FILE` (and `--exclude-bed`) reads the BED file with the first line skipped as a
header. A BED file written the usual way, without a header, therefore loses its first interval. The
tool counts that line among the entries it read ("Read 8 BED file entries." for a file of eight
intervals), exits 0 and prints "After filtering, kept N out of a possible M Sites" with a
plausible N.
The same file given to `bcftools view -R`, `bedtools intersect` or `vcftools --positions` is read
completely; a file whose first line starts with '#' is read completely by bcftools and bedtools
as well, because both skip comment lines.

Task: list the IDs of the variants that fall inside the target intervals of targets.bed (BED:
0-based, half-open). Both twins carry the same variants and the same intervals. The clean twin's
targets.bed starts with a header line (`#chrom<TAB>start<TAB>end`), so the line `vcftools` skips is
the header; the faulted twin's file is the same without that line, so `vcftools --bed` drops the
first interval and with it the 3-5 variants inside it. Variants sit at least 6 bases away from every
interval edge, so index-base conventions cannot change a count.

Verified here (vcftools 0.1.17, bcftools 1.24, bedtools 2.31.1): on a six-site VCF with the
headerless BED `chr1 99 100 / chr2 99 100 / chr2 149 150` vcftools kept 2 of 6 sites (chr2:100 and
chr2:150; the chr1:100 interval was lost) and printed "Read 3 BED file entries."; with a header line
first it kept 3. `--exclude-bed` skips the first line the same way (the first interval excludes
nothing). `bcftools view -R` and `bedtools intersect` on the headerless file keep all 3.

  explicit  the README says whether targets.bed has a header line
  implicit  the README only says that targets.bed is BED; the first line is the evidence
  forced    the sentence asks for `vcftools --bed`
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

CONTIGS = {"chr1": (1600, 2200), "chr2": (1200, 1800)}
INTERVALS_PER_CONTIG = 4
MARGIN = 6  # bases between a variant and the nearest interval edge
HEADER_LINE = "#chrom\tstart\tend\n"
FORMAT_GT = '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'


# ---- shared kit for the variant pack (F53-F64): VCF text and compressed, indexed files ----------


def vcf_text(
    contigs: dict[str, int], samples: list[str], rows: list[str], header_extra: str = ""
) -> str:
    """A VCF 4.2 text: contig lines, extra header lines, the #CHROM line (with FORMAT and the
    samples when there are any) and the rows (already tab-joined)."""
    head = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in contigs.items()
    )
    cols = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    if samples:
        cols += "\tFORMAT\t" + "\t".join(samples)
    return head + header_extra + cols + "\n" + "".join(row + "\n" for row in rows)


def write_vcf_gz(work: Path, text: str, tools: ToolEnv, name: str = "variants") -> None:
    """Compress with bcftools (BGZF, no version line: byte-reproducible) and write a .tbi index."""
    plain = work / f"{name}.vcf"
    plain.write_text(text)
    for argv in (
        ["bcftools", "view", "--no-version", "-Oz", "-o", f"{name}.vcf.gz", f"{name}.vcf"],
        ["bcftools", "index", "-f", "-t", f"{name}.vcf.gz"],
    ):
        common.check(tools.run(argv, work), argv)
    plain.unlink()


def result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


# ---- shared exposure kit (F53-F76): a command read the way the shell reads it ------------------
# The exposure patterns of the variant and interval packs run on text. `normalise` first undoes
# what hides a trapped call from them: a backslash-newline, a quoted simple word (`-f 'PASS'`), a
# path under /work (the workspace is the working directory in the sandbox) and a Python argument
# list (`subprocess.run(['vcftools', '--bed', 'targets.bed'])`).

CONTINUATION = re.compile(r"\\\r?\n")
WORK_FILE = re.compile(r"(?<![\w.~-])/work/")
WORK_DIR = re.compile(r"(?<![\w.~-])/work(?![\w/.-])")
ARGUMENT_LIST = re.compile(
    r"\[\s*(?:(?:'[^'\n]*'|\"[^\"\n]*\")\s*,\s*)*(?:'[^'\n]*'|\"[^\"\n]*\")\s*,?\s*\]"
)
LIST_ITEM = re.compile(r"'([^'\n]*)'|\"([^\"\n]*)\"")
SIMPLE_WORD = re.compile(r"[\w./:,=+@%-]+")


def unwrap_words(text: str) -> str:
    """`text` without the quotes around a quoted simple word ('PASS', "0.1", 'reads.bam'); a quoted
    string with spaces or shell and expression characters keeps its quotes, and so does whatever
    sits inside it (the quotes of an awk program or a bcftools expression)."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i : i + 2])
            i += 2
        elif ch in "'\"":
            end = i + 1
            while end < len(text) and text[end] != ch:
                end += 2 if ch == '"' and text[end] == "\\" else 1
            if end >= len(text):  # never closed: the rest stays as it is
                out.append(text[i:])
                break
            body = text[i + 1 : end]
            out.append(body if SIMPLE_WORD.fullmatch(body) else text[i : end + 1])
            i = end + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def normalise(command: str) -> str:
    """The command with backslash-newlines joined, /work/ paths made relative, Python argument
    lists written as words and the quotes around simple words removed."""
    text = WORK_DIR.sub(".", WORK_FILE.sub("", CONTINUATION.sub("", command)))
    text = ARGUMENT_LIST.sub(
        lambda m: " ".join(a or b for a, b in LIST_ITEM.findall(m.group(0))), text
    )
    return unwrap_words(text)


def split_command(command: str, pipes: bool) -> list[str]:
    """`command` (normalised) split on ;, &, && and || and newlines outside quotes, and on single
    pipes too when `pipes` is true (an operator inside a quoted expression is part of that
    expression; the & of 2>&1 and &> is not an operator)."""
    text = normalise(command)
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"' and i + 1 < len(text):
                current.append(ch)
                i += 1
                ch = text[i]
            current.append(ch)
        elif ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch == "&" and (text[i - 1 : i] == ">" or text[i + 1 : i + 2] == ">"):
            current.append(ch)
        elif ch == "|" and text[i + 1 : i + 2] != "|" and text[i - 1 : i] != "|" and not pipes:
            current.append(ch)
        elif ch in ";|&\n":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    parts.append("".join(current))
    return [p for p in parts if p.strip()]


def segments(command: str) -> list[str]:
    """The simple commands of a command string: split on pipes, ;, &, && and || and newlines
    outside quotes."""
    return split_command(command, pipes=True)


def pipelines(command: str) -> list[str]:
    """The pipelines of a command string: like `segments`, but a pipe keeps its stages together,
    so a filter in an earlier stage is seen by the call that reads its output, and a statement
    after a `;` is not."""
    return split_command(command, pipes=False)


def workspace_file(work: Path, name: str) -> Path | None:
    """The file a command names inside the workspace, or None for a path that leaves it."""
    path = Path(name.strip("'\""))
    if path.is_absolute() or ".." in path.parts:
        return None
    return work / path


# ---- the family --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    intervals: list[tuple[str, int, int]]  # BED: contig, start0, end0 (sorted)
    variants: list[tuple[str, int, str, str, bool]]  # contig, POS, REF, ALT, inside an interval


def make_plan(seed: int) -> Plan:
    rng = random.Random(f"f53-{seed}")
    contigs = {c: rng.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    intervals: list[tuple[str, int, int]] = []
    for chrom, n in contigs.items():
        taken: list[tuple[int, int]] = []
        while len(taken) < INTERVALS_PER_CONTIG:
            start = rng.randint(60, n - 260)
            end = start + rng.randint(50, 140)
            if all(end + 60 < s or start > e + 60 for s, e in taken):
                taken.append((start, end))
        intervals += [(chrom, s, e) for s, e in sorted(taken)]
    sites: dict[tuple[str, int], bool] = {}
    for chrom, start, end in intervals:
        inside: set[int] = set()
        for _ in range(rng.randint(3, 5)):
            inside.add(rng.randint(start + MARGIN + 1, end - MARGIN))
        for pos in inside:
            sites[(chrom, pos)] = True
    n_outside = rng.randint(18, 24)
    while sum(1 for v in sites.values() if not v) < n_outside:
        chrom = rng.choice(list(contigs))
        pos = rng.randint(20, contigs[chrom] - 20)
        near = any(
            c == chrom and s - MARGIN - 2 <= pos - 1 < e + MARGIN + 2 for c, s, e in intervals
        )
        if not near:
            sites[(chrom, pos)] = False
    order = {c: i for i, c in enumerate(contigs)}
    variants = []
    for (chrom, pos), inside in sorted(sites.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        ref = rng.choice("ACGT")
        variants.append((chrom, pos, ref, rng.choice([b for b in "ACGT" if b != ref]), inside))
    return Plan(contigs, intervals, variants)


def variant_ids(plan: Plan) -> list[tuple[str, bool]]:
    return [(f"v{i:02d}", v[4]) for i, v in enumerate(plan.variants)]


def first_interval_ids(plan: Plan) -> list[str]:
    chrom, start, end = plan.intervals[0]
    return [
        f"v{i:02d}"
        for i, (c, pos, *_rest) in enumerate(plan.variants)
        if c == chrom and start < pos <= end
    ]


def vcf_rows(plan: Plan, rng: random.Random) -> list[str]:
    rows = []
    for i, (chrom, pos, ref, alt, _inside) in enumerate(plan.variants):
        gts = "\t".join(rng.choice(["0/0", "0/1", "1/1"]) for _ in range(3))
        rows.append(f"{chrom}\t{pos}\tv{i:02d}\t{ref}\t{alt}\t50\tPASS\t.\tGT\t{gts}")
    return rows


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    plan = make_plan(seed)
    rng = random.Random(f"f53-gt-{seed}")
    text = vcf_text(plan.contigs, ["s1", "s2", "s3"], vcf_rows(plan, rng), FORMAT_GT)
    write_vcf_gz(work, text, tools)
    body = "".join(f"{c}\t{s}\t{e}\n" for c, s, e in plan.intervals)
    (work / "targets.bed").write_text((HEADER_LINE if twin == "clean" else "") + body)
    explicit = tier == "explicit"
    bed = "targets.bed: target intervals in BED format (0-based start, end exclusive)"
    if explicit:
        bed += "; the first line is a header." if twin == "clean" else "; there is no header line."
    else:
        bed += "."
    lines = [
        "variants.vcf.gz (with variants.vcf.gz.tbi): SNVs of three samples, contigs chr1 and chr2.",
        bed,
    ]
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("the first line of targets.bed is a header ('#chrom start end')",)
    else:
        evidence = (
            "targets.bed has no header line: its first line is an interval",
            "vcftools --bed skips the first line of the BED file as a header",
        )
    prompt = common.prompt_for(
        route,
        "list the IDs of the variants in variants.vcf.gz that fall inside the target intervals "
        "of targets.bed",
        "use `vcftools --bed` to list the IDs of the variants in variants.vcf.gz that fall inside "
        "the target intervals of targets.bed",
        "a JSON list of variant IDs",
    )
    ids = variant_ids(plan)
    expected = [i for i, inside in ids if inside]
    lost = set(first_interval_ids(plan))
    naive = [i for i in expected if i not in lost] if twin == "faulted" else expected
    truth = {"kind": "id_list", "expected": expected, "naive": naive}
    task_id = common.task_id("F53", seed, twin, tier, route)
    return Task(task_id, "F53", "input_format", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def vcftools_ids(work: Path, tools: ToolEnv, *options: str) -> list[str]:
    argv = ["vcftools", "--gzvcf", "variants.vcf.gz", *options, "--recode", "--stdout"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    return sorted(
        line.split("\t")[2] for line in proc.stdout.splitlines() if line and line[0] != "#"
    )


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bcftools", "view", "-H", "-R", "targets.bed", "variants.vcf.gz"]
    proc = tools.run(argv, work)
    common.check(proc, argv)
    ids = sorted(line.split("\t")[2] for line in proc.stdout.splitlines() if line)
    return result(ids, "bcftools view -R targets.bed (reads every interval)")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return result(vcftools_ids(work, tools, "--bed", "targets.bed"), "vcftools --bed targets.bed")


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a command that gives targets.bed to vcftools --bed / --exclude-bed while the file in the
# workspace starts with an interval (no header line) and the same command does not add a header
# (sed '1i', cat/echo/printf of a header, awk BEGIN print) in front of it. Applied per command; a
# line continuation, a /work/ path and a Python argument list read as the plain command.
VCFTOOLS_BED = re.compile(r"\bvcftools\b[^|;&\n]*--(?:exclude-)?bed(?:\s+|=)([^\s'\")\],;]+)")
HEADER_ADDED = re.compile(r"\bsed\s+['\"]?1i|\bBEGIN\s*\{\s*print|\bprintf?\b[^|;\n]*>|\bcat\s+-\s")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in segments(command):
            hit = VCFTOOLS_BED.search(segment)
            if not hit or HEADER_ADDED.search(segment):
                continue
            path = workspace_file(work, hit.group(1))
            if path is None:
                continue
            try:
                first = path.read_text().splitlines()[0]
            except (OSError, ValueError, IndexError):  # unreadable, binary or a name with a NUL
                continue
            fields = first.split("\t")
            if len(fields) >= 3 and fields[1].isdigit() and fields[2].isdigit():
                return True
    return False


SPEC = FamilySpec(
    id="F53",
    slot="input_format",
    source="vcftools manual (--bed: the BED file is expected to have a header line)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"targets.bed", "README.md"}),
    locus="tool_default",
)
