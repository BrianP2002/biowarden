"""Admission cases for the liftOver rules (pack iv), built on the F74 task.

For one seed both twins of F74 are built (implicit tier): the same summits and the same chain, with
peaks that lie inside one aligned block (clean) or start before an indel of the chain while their
summit lies after it (faulted). The task asks for the position of every summit in the new assembly.
Next to the task files the workspace gets `summits.bed` (the summit bases as one-base intervals)
and copies of both inputs with the `chr` prefix removed from the contig names. Every call below is
run with the real `liftOver` in the workspace; its output file is turned into the task's answer
(`start + column 10` of the lifted narrowPeak, or the start of the lifted summit) and compared with
`task.truth` (bench.score.answer_matches); the label never comes from a rule.

Calls: `-bedPlus=6` on the narrowPeak file (right on the clean twin, wrong on the faulted one:
stale offsets, two peaks dropped), the same with `-minMatch=0.5` (nothing dropped, still stale),
each with a real unmapped file and with /dev/null; the summit bases lifted as their own intervals
(right, also with /dev/null and `-minMatch`); and the contig-stripped copies of both inputs (every
interval is unmapped: wrong). Declarations: liftover.stale_columns takes `summit_offset=relifted`
on the right calls and `not_used` or `relifted` on the wrong ones (the rule takes the word of the
caller); liftover.unmapped_discarded takes `dropped_records=accepted`; liftover.chain_contigs has
no declaration, so every call is emitted in both cells with a note the rule ignores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from science_harness.bench.admission import Case, Label, case_source
from science_harness.bench.score import answer_matches
from science_harness.bench.tasks import ToolEnv, build_task
from science_harness.engine import Call

CHAIN = "old_to_new.chain"


@dataclass(frozen=True)
class Template:
    name: str
    options: tuple[str, ...]
    source: str  # the BED file lifted
    unmapped: str  # the fourth argument
    answer: str  # offset (lifted start + column 10) | start (lifted start)

    def argv(self) -> tuple[str, ...]:
        return (*self.options, self.source, CHAIN, f"{self.name}.out.bed", self.unmapped)


PEAKS, SUMMITS = "peaks_old.narrowPeak", "summits.bed"
PEAKS_NOCHR, SUMMITS_NOCHR = "peaks_nochr.narrowPeak", "summits_nochr.bed"
UNM = "lifted.unmapped.bed"
TEMPLATES: tuple[Template, ...] = (
    Template("peaks_plus6", ("-bedPlus=6",), PEAKS, UNM, "offset"),
    Template("peaks_plus6_devnull", ("-bedPlus=6",), PEAKS, "/dev/null", "offset"),
    Template("peaks_plus6_minmatch", ("-bedPlus=6", "-minMatch=0.5"), PEAKS, UNM, "offset"),
    Template(
        "peaks_plus6_minmatch_devnull",
        ("-bedPlus=6", "-minMatch=0.5"),
        PEAKS,
        "/dev/null",
        "offset",
    ),
    Template(
        "peaks_plus6_minmatch9", ("-minMatch=0.9", "-bedPlus=6"), PEAKS, "/dev/null", "offset"
    ),
    Template("summits", (), SUMMITS, UNM, "start"),
    Template("summits_devnull", (), SUMMITS, "/dev/null", "start"),
    Template("summits_minmatch", ("-minMatch=0.5",), SUMMITS, UNM, "start"),
    Template("summits_minmatch_devnull", ("-minMatch=0.5",), SUMMITS, "/dev/null", "start"),
    Template("peaks_nochr", ("-bedPlus=6",), PEAKS_NOCHR, UNM, "offset"),
    Template("peaks_nochr_devnull", ("-bedPlus=6",), PEAKS_NOCHR, "/dev/null", "offset"),
    Template("summits_nochr", (), SUMMITS_NOCHR, UNM, "start"),
)


def prepare(work: Path) -> None:
    """summits.bed and the contig-stripped copies of both inputs."""
    rows = [line.split("\t") for line in (work / PEAKS).read_text().splitlines()]
    summits = [(r[0], int(r[1]) + int(r[9]), r[3]) for r in rows]
    (work / SUMMITS).write_text("".join(f"{c}\t{p}\t{p + 1}\t{n}\n" for c, p, n in summits))
    strip = lambda text: "".join(  # noqa: E731
        line.removeprefix("chr") + "\n" for line in text.splitlines()
    )
    (work / PEAKS_NOCHR).write_text(strip((work / PEAKS).read_text()))
    (work / SUMMITS_NOCHR).write_text(strip((work / SUMMITS).read_text()))


def label_of(template: Template, task, tools: ToolEnv, work: Path) -> Label:
    proc = tools.run(["liftOver", *template.argv()], work)
    if proc.returncode != 0:
        return "wrong"
    out = work / f"{template.name}.out.bed"
    rows = [line.split("\t") for line in out.read_text().splitlines()]
    if template.answer == "offset":
        answer = {r[3]: int(r[1]) + int(r[9]) for r in rows}
    else:
        answer = {r[3]: int(r[1]) for r in rows}
    return "right" if answer_matches(task.truth, answer) else "wrong"


def build_cases(seed: int, scratch: Path, tools: ToolEnv, declarations) -> list[Case]:
    """`declarations(label)` -> the declarations of the declared cells for a call of that label."""
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F74", seed, twin, work, tools, "implicit", "free")
        prepare(work)
        for template in TEMPLATES:
            label = label_of(template, task, tools, work)
            argv = template.argv()
            out.append(
                Case(Call("liftOver", argv, work), label, False, "F74", seed, twin, template.name)
            )
            for declared in declarations(label):
                call = Call("liftOver", argv, work, declared)
                out.append(Case(call, label, True, "F74", seed, twin, template.name))
    return out


@case_source("liftover.stale_columns")
def stale_columns_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def declarations(label: Label):
        if label == "right":
            return [{"summit_offset": "relifted"}]
        return [{"summit_offset": "not_used"}, {"summit_offset": "relifted"}]

    return build_cases(seed, scratch, tools, declarations)


@case_source("liftover.unmapped_discarded")
def unmapped_discarded_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return build_cases(seed, scratch, tools, lambda label: [{"dropped_records": "accepted"}])


@case_source("liftover.chain_contigs")
def chain_contigs_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return build_cases(seed, scratch, tools, lambda label: [{"note": "no declaration exists"}])
