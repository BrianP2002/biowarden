"""Admission cases for msa.sequence_orientation (family F86, implicit tier).

`mafft --quiet --thread 1 <options> seqs.fasta` (alignment on stdout) and `muscle -align seqs.fasta
-output out.afa -threads 1`; the answer is the number of fully conserved columns. A call is right
when it equals the count on the correctly oriented sequences. Without
`--adjustdirection[accurately]`
(and for every muscle call) the clean twin is right and the faulted twin, which holds 2-3
reverse-complemented members, is wrong; the mafft calls with the option are right on both.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_seq_common import Tpl, cases_for
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.extra.f86_mafft_orient import conserved_columns

KEY = "sequence_orientation"
AS_WRITTEN = (("same_strand",), ("same_strand", "mixed_strands"))
FLIPPING = (("mixed_strands", "same_strand"), ("mixed_strands", "same_strand"))


def from_stdout(work: Path, proc) -> int:
    return conserved_columns(proc.stdout)


def from_file(name: str):
    def read(work: Path, proc) -> int:
        return conserved_columns((work / name).read_text())

    return read


def _mafft(name: str, flips: bool, *options: str) -> Tpl:
    argv = ("--quiet", "--thread", "1", *options, "seqs.fasta")
    return Tpl(name, "mafft", argv, from_stdout, *(FLIPPING if flips else AS_WRITTEN))


def _muscle(name: str, *options: str) -> Tpl:
    out = f"out_{name}.afa"
    argv = (*options, "seqs.fasta", "-output", out, "-threads", "1")
    return Tpl(name, "muscle", argv, from_file(out), *AS_WRITTEN)


TEMPLATES: tuple[Tpl, ...] = (
    # --- sequences aligned as written
    _mafft("default", False),
    _mafft("auto", False, "--auto"),
    _mafft("localpair", False, "--localpair", "--maxiterate", "100"),
    _mafft("retree", False, "--retree", "2"),
    _mafft("reorder", False, "--reorder", "--auto"),
    _muscle("muscle_align", "-align"),
    _muscle("muscle_super5", "-super5"),
    # --- sequences flipped when needed
    _mafft("adjust", True, "--adjustdirection"),
    _mafft("adjust_accurately", True, "--adjustdirectionaccurately"),
    _mafft("adjust_auto", True, "--adjustdirection", "--auto"),
)


@case_source("msa.sequence_orientation")
def orientation_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F86", KEY, seed, scratch, tools, TEMPLATES)
