"""Admission cases for hisat2.rna_strandness (family F34, slot strand).

For one seed both twins of the F34 task are built (implicit tier: the README says only
"strand-specific"): the same fragments as a forward-stranded library (clean, FR is right) and as a
dUTP library (faulted, RF is right). Each `hisat2` call is run for real into its own SAM; the SAM
is sorted, assembled with StringTie and the strand of the transcript at each gene region is read off
(the family's own routine); that is compared with `task.truth["expected"]` (bench.score.
answer_matches). The label never comes from the rule. FR and F are right on the clean twin, RF and R
on the faulted one (R and F on read 1 alone); the one-letter values on pairs and the opposite value
are wrong; calls without the option are right on both twins and are not matched by the rule.

Each call is emitted undeclared and with declarations: the library the call's value stands for (its
author's belief) and, for wrong calls, the library the data really are.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_aln_samtools import Template
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f34_hisat2_strand import assembled_strands
from biowarden.engine import Call

KEY = "library_strandedness"
PAIRS = ("-1", "reads_1.fastq", "-2", "reads_2.fastq")
SINGLE = ("-U", "reads_1.fastq")
IMPLIED = {"F": "forward", "FR": "forward", "R": "reverse", "RF": "reverse"}


def _hs(name: str, value: str | None, *options: str, reads: tuple[str, ...] = PAIRS) -> Template:
    flag = ("--rna-strandness", value) if value else ()
    return Template(
        name, "hisat2", ("-p", "1", *options, *flag, "-x", "hs2idx", *reads), value or ""
    )


TEMPLATES: tuple[Template, ...] = (
    _hs("pairs_fr", "FR"),
    _hs("pairs_rf", "RF"),
    _hs("pairs_fr_threads", "FR", "--threads", "1", "--dta"),
    _hs("pairs_rf_dta", "RF", "--dta"),
    _hs("pairs_fr_equals", "FR", "--no-unal"),
    _hs("pairs_f_single_letter", "F"),
    _hs("pairs_r_single_letter", "R"),
    _hs("single_f", "F", reads=SINGLE),
    _hs("single_r", "R", reads=SINGLE),
    _hs("no_option", None),
    _hs("no_option_dta", None, "--dta"),
)


@case_source("hisat2.rna_strandness")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F34", seed, twin, work, tools, "implicit", "free")
        library = "forward" if twin == "clean" else "reverse"
        for template in TEMPLATES:
            sam = f"t_{template.name}"
            argv = [template.tool, *template.argv, "-S", f"{sam}.sam"]
            proc = tools.run(argv, work)
            label: Label = "wrong"
            if proc.returncode == 0:
                answer = assembled_strands(work, tools, sam)
                label = "right" if answer_matches(task.truth, answer) else "wrong"
            own = IMPLIED.get(template.meaning)
            if own is None:  # no --rna-strandness: the rule is not matched
                out.append(
                    Case(
                        Call(template.tool, template.argv, work, {}),
                        label,
                        False,
                        "F34",
                        seed,
                        twin,
                        template.name,
                    )
                )
                continue
            declarations = {own} if label == "right" else {own, library}
            calls = [(False, {})] + [(True, {KEY: v}) for v in sorted(declarations)]
            for declared, decl in calls:
                call = Call(template.tool, template.argv, work, decl)
                out.append(Case(call, label, declared, "F34", seed, twin, template.name))
    return out
