"""Workspace helpers shared by the task families."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from science_harness.bench.tasks.instance import Instance
from science_harness.bench.tasks.tools import ToolEnv


def wrap_fasta(genome: dict[str, str]) -> str:
    return "".join(f">{c}\n{textwrap.fill(s, 60)}\n" for c, s in genome.items())


def check(proc: Any, argv: list[str]) -> None:
    if proc.returncode != 0:
        raise RuntimeError(f"setup failed: {' '.join(argv)}\n{proc.stderr}")


def write_vcf_text(
    work: Path, header: str, body: str, tools: ToolEnv, name: str = "variants"
) -> None:
    """Write a plain VCF, compress and index it (byte-reproducible: no version or date lines)."""
    plain = work / f"{name}.vcf"
    plain.write_text(header + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + body)
    for argv in (
        ["bcftools", "view", "--no-version", "-Oz", "-o", f"{name}.vcf.gz", f"{name}.vcf"],
        ["bcftools", "index", f"{name}.vcf.gz"],
    ):
        check(tools.run(argv, work), argv)
    plain.unlink()


def write_vcf(work: Path, inst: Instance, tools: ToolEnv) -> None:
    header = "##fileformat=VCFv4.2\n" + "".join(
        f"##contig=<ID={c},length={n}>\n" for c, n in inst.contigs.items()
    )
    body = "".join(
        f"{v.chrom}\t{v.pos}\t{v.id}\t{v.ref}\t{v.alt}\t.\t.\t.\n" for v in inst.variants
    )
    write_vcf_text(work, header, body, tools)


def write_fasta(work: Path, genome: dict[str, str], tools: ToolEnv) -> None:
    (work / "ref.fa").write_text(wrap_fasta(genome))
    check(tools.run(["samtools", "faidx", "ref.fa"], work), ["samtools", "faidx"])


def write_bam(work: Path, sam_text: str, tools: ToolEnv, name: str = "reads") -> None:
    """Sort and index SAM text into <name>.bam (no @PG lines, so the bytes are reproducible)."""
    (work / f"{name}.sam").write_text(sam_text)
    for argv in (
        ["samtools", "sort", "--no-PG", "-o", f"{name}.bam", f"{name}.sam"],
        ["samtools", "index", f"{name}.bam"],
    ):
        check(tools.run(argv, work), argv)
    (work / f"{name}.sam").unlink()


def bed_text(inst: Instance, rename: Callable[[str], str] = str) -> str:
    return "".join(f"{rename(r.chrom)}\t{r.start0}\t{r.end0}\n" for r in inst.regions)


def task_id(family: str, seed: int, twin: str, tier: str, route: str) -> str:
    extra = ("" if tier == "explicit" else f"-{tier}") + ("" if route == "free" else f"-{route}")
    return f"{family}-s{seed:02d}-{twin}{extra}"


def readme(lines: list[str]) -> str:
    return "# Project files\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"


# Agents often wrote a mapping answer as the whole of result.json, without the "answer" key
# (protocol_fail "answer is missing"), so its sentence also shows the nesting. "as in", not "must
# be": the full contract keeps status, flags and method next to "answer".
MAPPING_ANSWER = "a JSON object"
MAPPING_SHAPE = '{"answer": {<key>: <value>, ...}}'


def answer_sentence(answer: str) -> str:
    """The sentence that says what `answer` must be; the stage-1 wording unless it is a mapping."""
    shape = f", as in {MAPPING_SHAPE}" if answer.startswith(MAPPING_ANSWER) else ""
    return f"In result.json, `answer` must be {answer}{shape}."


def prompt_for(route: str, free: str, forced: str, answer: str) -> str:
    """Task sentence for the free or forced route, then the answer format."""
    sentence = free if route == "free" else forced
    return f"Task: {sentence} (see README.md for the files). {answer_sentence(answer)}"
