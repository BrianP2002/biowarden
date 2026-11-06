"""Admission cases for seqkit.rmdup_strands (family F82, implicit tier).

`seqkit rmdup <options> guides.fasta`; the answer is the number of records left (the number of
distinct sequences as written). A call without `-P` is right on the clean twin (no reverse-
complement pair) and wrong on the faulted one; calls with `-P` / `--only-positive-strand` are
right on both. Removal by name (`-n`) keeps every record and is wrong on both twins (the file
holds exact duplicates); the rule does not apply to it.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_seq_common import Tpl, cases_for
from biowarden.bench.tasks import ToolEnv

KEY = "duplicate_strand_policy"


def records(work: Path, proc) -> int:
    return proc.stdout.count(">")


def _rmdup(name: str, *options: str, positive: bool) -> Tpl:
    if positive:
        right, wrong = ("same_strand_only",), ("same_strand_only",)
    else:
        right, wrong = ("same_strand_only", "both_strands"), ("both_strands", "same_strand_only")
    return Tpl(name, "seqkit", ("rmdup", *options, "guides.fasta"), records, right, wrong)


TEMPLATES: tuple[Tpl, ...] = (
    # --- both strands compared
    _rmdup("by_seq", "-s", positive=False),
    _rmdup("by_seq_long", "--by-seq", positive=False),
    _rmdup("by_seq_ignore_case", "-s", "-i", positive=False),
    _rmdup("by_seq_combined", "-si", positive=False),
    _rmdup("by_seq_threads", "-j", "2", "-s", positive=False),
    _rmdup("by_seq_quiet", "--quiet", "-s", positive=False),
    _rmdup("by_seq_dups", "-s", "-d", "dups.fasta", positive=False),
    _rmdup("global_threads", "-s", "-w", "0", positive=False),
    # --- positive strand only
    _rmdup("positive", "-s", "-P", positive=True),
    _rmdup("positive_long", "--by-seq", "--only-positive-strand", positive=True),
    _rmdup("positive_combined", "-sP", positive=True),
    _rmdup("positive_ignore_case", "-s", "-i", "-P", positive=True),
    _rmdup("positive_threads", "-j", "2", "-s", "-P", positive=True),
    # --- by name: keeps every record, the rule does not apply
    _rmdup("by_name", "-n", positive=True),
    _rmdup("by_id_default", positive=True),
)


@case_source("seqkit.rmdup_strands")
def rmdup_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F82", KEY, seed, scratch, tools, TEMPLATES)
