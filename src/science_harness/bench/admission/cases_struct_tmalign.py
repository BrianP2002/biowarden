"""Admission cases for tmalign.normalisation (family F93: TM-score of a model against a reference).

Workspaces are the benchmark's own (implicit tier). The clean twin holds two structures of equal
length (both normalisations coincide), the faulted twin a model shorter than the reference. Every
call below is a real `TMalign model.pdb reference.pdb ...` run in the workspace (the reference is
the second file, the order the task wording induces); the output is parsed by the READING the
caller adopts and compared with `task.truth` (bench.score.answer_matches, tolerance 0.02):

  undeclared   the reader that does not think about the normalisation takes the first
               "TM-score=" line
  declared v   the reader takes the line v names: structure_1 and structure_2 are the two
               "normalized by length of Structure_n" lines, average the "-a T" line and
               given_length the "user-specified L" line; a line the call did not print makes the
               reading wrong (it cannot have read it)

Calls: plain, -fast, -a T, -u L and -L L with L the reference length, -d 5, -m matrix, -o
superposition, -ter 3, and ./-prefixed file names. Reference-first calls are not generated (the rule
flags them until they declare structure_1; see the rule's docstring).
"""

from __future__ import annotations

import re
from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call

KEY = "tmscore_normalisation"
VALUES = ("structure_1", "structure_2", "average", "given_length")
LINE = re.compile(r"TM-score=\s*([0-9.]+)\s*\((.*?)\)")
MARK = {
    "structure_1": "normalized by length of Structure_1",
    "structure_2": "normalized by length of Structure_2",
    "average": "average length",
    "given_length": "user-specified L",
}


def reading(output: str, which: str) -> float | None:
    """The TM-score on the line a reader of kind `which` keeps (None: the line is not there)."""
    lines = LINE.findall(output)
    if which == "first":
        return float(lines[0][0]) if lines else None
    for value, note in lines:
        if MARK[which] in note:
            return float(value)
    return None


def templates(ref_len: int) -> list[tuple[str, tuple[str, ...]]]:
    base = ("model.pdb", "reference.pdb")
    return [
        ("plain", base),
        ("fast", (*base, "-fast")),
        ("average", (*base, "-a", "T")),
        ("user_length", (*base, "-u", str(ref_len))),
        ("user_length_L", (*base, "-L", str(ref_len))),
        ("d0", (*base, "-d", "5")),
        ("matrix", (*base, "-m", "matrix.txt")),
        ("superpose", (*base, "-o", "sup.pdb")),
        ("ter", (*base, "-ter", "3")),
        ("dot_slash", ("./model.pdb", "./reference.pdb")),
        ("options_first", ("-fast", *base)),
    ]


def label(truth, value: float | None) -> Label:
    return "right" if value is not None and answer_matches(truth, value) else "wrong"


@case_source("tmalign.normalisation")
def normalisation_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F93", seed, twin, work, tools, "implicit", "free")
        ref_len = sum(
            1 for line in (work / "reference.pdb").read_text().splitlines() if " CA " in line[:20]
        )
        for name, argv in templates(ref_len):
            proc = tools.run(["TMalign", *argv], work)
            call = Call("TMalign", argv, work)
            first = (
                label(task.truth, reading(proc.stdout, "first"))
                if proc.returncode == 0
                else "wrong"
            )
            out.append(Case(call, first, False, "F93", seed, twin, name))
            for value in VALUES:
                got = reading(proc.stdout, value) if proc.returncode == 0 else None
                decl = Call("TMalign", argv, work, {KEY: value})
                out.append(Case(decl, label(task.truth, got), True, "F93", seed, twin, name))
    return out
