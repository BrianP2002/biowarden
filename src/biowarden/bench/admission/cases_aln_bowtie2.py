"""Admission cases for bowtie2.maxins_default (family F33, slot default_param).

For one seed both twins of the F33 task are built (implicit tier): the same pairs, fragments of
250-450 bases (clean) or 30% of 600-800 bases (faulted). Each paired `bowtie2` call is run for
real; the first-mate records with the proper-pair flag (0x43) are counted and compared with
`task.truth["expected"]` (bench.score.answer_matches). The label never comes from the rule. A call
with the default -X 500 (or any -X below 800) is right on the clean twin and wrong on the
faulted one;
a call with -X 800 or more is right on both.

The rule is a required-declaration rule: every call is emitted with the declaration its author would
give it (`up_to_500` if the call keeps the default, `over_500` if it raises -X) and with the other
one. Right calls carry the value that matches the twin's data; the wrong ones are emitted with both.
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.admission.cases_aln_samtools import Template, run_all
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call

KEY = "fragment_length_max"
BASE = ("-x", "bt2idx", "-1", "reads_1.fastq", "-2", "reads_2.fastq")


def _bt2(name: str, *options: str, tail: tuple[str, ...] = ()) -> Template:
    return Template(name, "bowtie2", (*options, *BASE, *tail), "")


TEMPLATES: tuple[Template, ...] = (
    # --- the default -X 500, or a value under the longest fragment: long pairs are discordant
    _bt2("default"),
    _bt2("threads", "-p", "2"),
    _bt2("very_sensitive", "--very-sensitive", "-p", "2"),
    _bt2("explicit_500", "-X", "500"),
    _bt2("x700_attached", "-X700"),
    _bt2("maxins_700", "--maxins", "700"),
    _bt2("no_unal", "--no-unal"),
    _bt2("local", "--local", "-p", "2"),
    _bt2("mixed_off", "--no-mixed", "--no-discordant"),
    _bt2("fr_orientation", "--fr", "-p", "2"),
    # --- -X at or above the longest fragment (800): every pair is concordant
    _bt2("x1000", "-X", "1000"),
    _bt2("x800", "-X", "800"),
    _bt2("maxins_equals", "--maxins=1200"),
    _bt2("x2000_threads", "-p", "2", "-X", "2000"),
    _bt2("x1000_very_sensitive", "--very-sensitive", "-X", "1000"),
    _bt2("x900_no_unal", "--no-unal", "-X900"),
    _bt2("x1000_minins", "-I", "100", "-X", "1000"),
)


def _count(proc) -> int | None:
    if proc.returncode != 0:
        return None
    count = 0
    for line in proc.stdout.splitlines():
        if not line.startswith("@") and (int(line.split("\t")[1]) & 0x43) == 0x43:
            count += 1
    return count


def _declared_value(argv: tuple[str, ...]) -> str:
    from biowarden.engine.rules.aln_bowtie2 import maxins

    return "over_500" if maxins(Call("bowtie2", argv, Path("."))) > 500 else "up_to_500"


@case_source("bowtie2.maxins_default")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F33", seed, twin, work, tools, "implicit", "free")
        procs = run_all(tools, work, TEMPLATES)
        for template, proc in zip(TEMPLATES, procs, strict=True):
            n = _count(proc)
            label: Label = "right" if n is not None and answer_matches(task.truth, n) else "wrong"
            own = _declared_value(template.argv)
            # the value that matches the twin's data, and the author's own belief about the call
            true_value = "up_to_500" if twin == "clean" else "over_500"
            values = {own, true_value} if label == "wrong" else {true_value}
            if label == "right" and own != true_value:
                values.add(own)  # a call that raises -X on the clean twin is right whatever it says
            calls = [(True, {KEY: v}) for v in sorted(values)]
            if label == "right":
                calls.append((False, {}))
            for declared, decl in calls:
                call = Call(template.tool, template.argv, work, decl)
                out.append(Case(call, label, declared, "F33", seed, twin, template.name))
            if label == "wrong":
                out.append(
                    Case(
                        Call(template.tool, template.argv, work, {}),
                        label,
                        False,
                        "F33",
                        seed,
                        twin,
                        template.name,
                    )
                )
    return out
