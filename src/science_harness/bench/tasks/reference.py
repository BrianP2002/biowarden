"""Scripted solvers for the validity gates: a reference solution and a naive one per family.

The reference solver knows the trap and handles it. The naive solver uses the obvious tool call and
falls into the trap on the faulted twin. Both run outside the sandbox and return the dict an agent
would write to result.json.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks.families import Task
from science_harness.bench.tasks.instance import FLANK
from science_harness.bench.tasks.tools import ToolEnv


def _regions_file(work: Path) -> str:
    return "regions.bed" if (work / "regions.bed").exists() else "regions.tsv"


def _result(answer: Any, method: str, flags: list[str] | None = None, status: str = "ok") -> dict:
    return {"status": status, "answer": answer, "flags": flags or [], "method": method}


def _ids(stdout: str, column: int) -> list[str]:
    return sorted(line.split("\t")[column] for line in stdout.splitlines() if line.strip())


def _fasta_sequences(stdout: str) -> list[str]:
    seqs: list[str] = []
    for line in stdout.splitlines():
        if line.startswith(">"):
            seqs.append("")
        elif seqs:
            seqs[-1] += line.strip()
    return seqs


def _read_fasta(path: Path) -> dict[str, str]:
    genome: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            genome[line[1:].split()[0]] = ""
        else:
            genome[next(reversed(genome))] += line.strip()
    return genome


def _vcf_variants(work: Path, tools: ToolEnv) -> list[tuple[str, int, str]]:
    proc = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work)
    rows = [line.split("\t") for line in proc.stdout.splitlines()]
    return [(r[2], int(r[1]), r[0]) for r in rows]


def _contexts(genome: dict[str, str], variants: list[tuple[str, int, str]]) -> dict[str, str]:
    out = {}
    for vid, pos, chrom in variants:
        seq, i = genome[chrom], pos - 1
        out[vid] = f"{seq[i - FLANK : i]}[{seq[i]}]{seq[i + 1 : i + 1 + FLANK]}"
    return out


def _vcf_contig_lengths(work: Path, tools: ToolEnv) -> dict[str, int]:
    proc = tools.run(["bcftools", "view", "-h", "variants.vcf.gz"], work)
    pairs = re.findall(r"##contig=<ID=([^,>]+),length=(\d+)", proc.stdout)
    return {name: int(length) for name, length in pairs}


def _fai_lengths(work: Path) -> dict[str, int]:
    rows = (line.split("\t") for line in (work / "ref.fa.fai").read_text().splitlines())
    return {r[0]: int(r[1]) for r in rows}


def solve(task: Task, work: Path, tools: ToolEnv, *, reference: bool) -> dict[str, Any]:
    if task.family not in _reference:
        from science_harness.bench.tasks.extra import load

        spec = load()[task.family]
        return (spec.reference if reference else spec.naive)(task, work, tools)
    return (_reference if reference else _naive)[task.family](task, work, tools)


def _f1_reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", _regions_file(work), "-u"]
    return _result(_ids(tools.run(argv, work).stdout, 2), "bedtools intersect (BED semantics)")


def _f1_naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bcftools", "view", "-H", "-R", _regions_file(work), "variants.vcf.gz"]
    return _result(_ids(tools.run(argv, work).stdout, 2), "bcftools view -R")


def _f2_reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    shift = 0 if task.twin == "clean" else 1  # faulted strings carry a 0-based start
    regions = []
    for line in (work / "regions.txt").read_text().split():
        chrom, span = line.split(":")
        start, end = span.split("-")
        regions.append(f"{chrom}:{int(start) + shift}-{end}")
    proc = tools.run(["samtools", "faidx", "ref.fa", *regions], work)
    return _result(_fasta_sequences(proc.stdout), "samtools faidx, start converted to 1-based")


def _f2_naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    regions = (work / "regions.txt").read_text().split()
    proc = tools.run(["samtools", "faidx", "ref.fa", *regions], work)
    return _result(_fasta_sequences(proc.stdout), "samtools faidx on the region strings as given")


def _f3_reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    text = "".join(
        line if line.startswith("chr") else f"chr{line}"
        for line in (work / "regions.bed").read_text().splitlines(keepends=True)
    )
    (work / "regions_chr.bed").write_text(text)
    argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "regions_chr.bed", "-u"]
    count = len(_ids(tools.run(argv, work).stdout, 2))
    flags = ["contig names differ between VCF and BED; renamed 1/2 to chr1/chr2"]
    return _result(
        count, "bedtools intersect after renaming contigs", flags if task.twin == "faulted" else []
    )


def _f3_naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "regions.bed", "-u"]
    return _result(len(_ids(tools.run(argv, work).stdout, 2)), "bedtools intersect")


def _f4_reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    in_vcf, in_fai = _vcf_contig_lengths(work, tools), _fai_lengths(work)
    if in_vcf != in_fai:
        flags = [
            f"reference build mismatch: VCF contig lengths {in_vcf} differ from ref.fa {in_fai}"
        ]
        return _result(None, "compared contig lengths", flags, status="needs_input")
    contexts = _contexts(_read_fasta(work / "ref.fa"), _vcf_variants(work, tools))
    return _result(contexts, "contexts from ref.fa after a contig length check")


def _f4_naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    contexts = _contexts(_read_fasta(work / "ref.fa"), _vcf_variants(work, tools))
    return _result(contexts, "contexts read directly from ref.fa")


_reference = {"F1": _f1_reference, "F2": _f2_reference, "F3": _f3_reference, "F4": _f4_reference}
_naive = {"F1": _f1_naive, "F2": _f2_naive, "F3": _f3_naive, "F4": _f4_naive}
