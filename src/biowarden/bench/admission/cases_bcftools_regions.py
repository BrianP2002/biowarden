"""Admission cases for bcftools.regions_file_base (family F1) and bcftools.region_semantics (F7).

Both case sources build the benchmark workspaces for one seed (clean and faulted twin, implicit
tier) and return real bcftools calls. The label of every call comes from running it in the
workspace and comparing its record IDs with `task.truth["expected"]` (bench.score.answer_matches);
the rules never see it. Calls are emitted with and without a declaration.

F1 (base of the regions file). The BED intervals sit in `regions.bed` (clean) or `regions.tsv`
(faulted), and the call reads them through -R/-T in several spellings and subcommands. Beside the
task's own file the workspace gets the files an agent would make itself: a `.bed` copy, an
upper-case `.BED` copy, gzip copies, and a copy converted to 1-based coordinates under a `.tsv`
name. Right calls: any file whose name makes bcftools read BED (clean twin, or the copies) and the
converted file. Wrong calls: the BED intervals under a name that is not .bed/.bed.gz.
Declarations: an informed agent declares what its file holds (zero_based_half_open for BED content,
one_based_closed for the converted file); a mistaken agent declares one_based_closed for BED
content. The converted file is emitted only with its declaration: BED-looking content under a
non-.bed name is exactly the state in which the rule asks for one, so an undeclared call there is
friction by design, not a measured cell.

F7 (record overlap versus start-only). The gVCF-style records of the F7 task are selected by -R/-r
(overlap), -T/-t (start only) and the --regions-overlap/--targets-overlap variants, through view,
query, filter, annotate and norm. The clean twin's reference blocks start inside their regions, so
every call is right there; on the faulted twin start-only selection loses the trap blocks, and so
does `query -f '%ID\\n' -R` on a text VCF (a verified bcftools surprise the rule can only warn
about). Declarations: the honest one matches the options (overlap or start_only); a mistaken agent
that believes -T/-t overlap declares overlap for a start-only call, emitted only where the result
is wrong as well. A declaration that contradicts the options of a call whose result is right (-T on
the clean twin) is not emitted: the rule fails it by design although the result is right, and the
cells are labelled by result.
"""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import Task, ToolEnv, build_task
from biowarden.engine import Call

TAB = "\\t"
VCF = "variants.vcf.gz"


@dataclass(frozen=True)
class Template:
    name: str
    argv: tuple[str, ...]  # {R}: regions file, {S}: comma-separated region strings
    id_col: int  # column of the record ID in the call's output
    semantic: str = ""  # F7: what the options imply, overlap | start_only


def _ids(tools: ToolEnv, work: Path, argv: Sequence[str], id_col: int) -> list[str] | None:
    """IDs the call returns, or None when it fails (then it is no case)."""
    proc = tools.run(["bcftools", *argv], work)
    if proc.returncode != 0:
        return None
    rows = (line.split("\t") for line in proc.stdout.splitlines() if line and line[0] != "#")
    return sorted(dict.fromkeys(row[id_col] for row in rows))


def _fill(template: Template, regions: str, strings: str = "") -> tuple[str, ...]:
    return tuple(a.replace("{R}", regions).replace("{S}", strings) for a in template.argv)


def _rotate(seed: int, offset: int, n: int, k: int) -> list[int]:
    """k template indexes that move with the seed, so 300 seeds cover every template."""
    return list(dict.fromkeys((seed + offset + step * 7) % n for step in range(k)))


@dataclass(frozen=True)
class Run:
    """One call to run in a workspace, and where its output keeps the record ID."""

    work: Path
    argv: tuple[str, ...]
    id_col: int


def _run_all(tools: ToolEnv, runs: Sequence[Run]) -> list[list[str] | None]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda r: _ids(tools, r.work, r.argv, r.id_col), runs))


def _build_twins(family: str, seed: int, scratch: Path, tools: ToolEnv) -> dict[str, Task]:
    twins = ("clean", "faulted")
    with ThreadPoolExecutor(max_workers=2) as pool:
        built = pool.map(
            lambda twin: build_task(family, seed, twin, scratch / twin, tools, tier="implicit"),
            twins,
        )
    return dict(zip(twins, built, strict=True))


# ---- F1: the base of the regions file ----------------------------------------------------------

BASE_RULE = "bcftools.regions_file_base"
BASE_KEY = "region_file_base"
BED, ONE = "zero_based_half_open", "one_based_closed"

F1_TEMPLATES = (
    Template("view -R", ("view", "-H", "-R", "{R}", VCF), 2),
    Template("view -T", ("view", "-H", "-T", "{R}", VCF), 2),
    Template("view --regions-file=", ("view", "-H", "--regions-file={R}", VCF), 2),
    Template("view --targets-file", ("view", "-H", "--targets-file", "{R}", VCF), 2),
    Template("view -R attached", ("view", "-H", "-R{R}", VCF), 2),
    Template("query -R", ("query", "-f", "%ID\\n", "-R", "{R}", VCF), 0),
    Template("query -T", ("query", "-f", f"%CHROM{TAB}%POS{TAB}%ID\\n", "-T", "{R}", VCF), 2),
    Template("filter -R", ("filter", "-i", "POS>0", "-R", "{R}", VCF), 2),
    Template("annotate -R", ("annotate", "-R", "{R}", VCF), 2),
    Template("norm -T", ("norm", "-m-any", "-T", "{R}", VCF), 2),
)


@dataclass(frozen=True)
class RegionsFile:
    name: str
    honest: str  # what an informed agent declares for this file's content
    mistaken: str | None  # a declaration that is wrong about the content
    undeclared: bool = True  # emit the call without a declaration too


def _add_files(work: Path, twin: str) -> list[RegionsFile]:
    own = "regions.bed" if twin == "clean" else "regions.tsv"
    text = (work / own).read_text()
    gz = gzip.compress(text.encode(), mtime=0)
    (work / "regions.bed.gz").write_bytes(gz)
    files = [RegionsFile(own, BED, ONE), RegionsFile("regions.bed.gz", BED, ONE)]
    if twin == "faulted":
        (work / "regions.tsv.bed").write_text(text)
        (work / "regions_up.BED").write_text(text)
        (work / "regions.tsv.gz").write_bytes(gz)
        lines = (line.split("\t") for line in text.splitlines())
        converted = "".join(f"{c}\t{int(s) + 1}\t{e}\n" for c, s, e in lines)
        (work / "regions_1based.tsv").write_text(converted)
        files += [
            RegionsFile("regions.tsv.bed", BED, ONE),
            RegionsFile("regions_up.BED", BED, ONE),
            RegionsFile("regions.tsv.gz", BED, ONE),
            RegionsFile("regions_1based.tsv", ONE, None, undeclared=False),
        ]
    return files


@case_source(BASE_RULE)
def regions_file_base_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    tasks = _build_twins("F1", seed, scratch, tools)
    plan: list[tuple[str, RegionsFile, Template, Run]] = []
    for twin in tasks:
        work = scratch / twin
        for j, rf in enumerate(_add_files(work, twin)):
            for t in _rotate(seed, 3 * j, len(F1_TEMPLATES), 2 if j == 0 else 1):
                template = F1_TEMPLATES[t]
                plan.append(
                    (twin, rf, template, Run(work, _fill(template, rf.name), template.id_col))
                )
    cases: list[Case] = []
    for (twin, rf, template, run), ids in zip(
        plan, _run_all(tools, [p[3] for p in plan]), strict=True
    ):
        if ids is None:
            continue
        label: Label = "right" if answer_matches(tasks[twin].truth, ids) else "wrong"
        declarations: list[str | None] = [rf.honest]
        if rf.undeclared:
            declarations.insert(0, None)
        if rf.mistaken:
            declarations.append(rf.mistaken)
        for value in declarations:
            declared = {BASE_KEY: value} if value else {}
            note = f"{template.name} {rf.name} {value or 'undeclared'}"
            call = Call("bcftools", run.argv, run.work, declared)
            cases.append(Case(call, label, value is not None, "F1", seed, twin, note))
    return cases


# ---- F7: record overlap versus start-only ------------------------------------------------------

SEMANTICS_RULE = "bcftools.region_semantics"
SEMANTICS_KEY = "region_semantics"
OVERLAP, START = "overlap", "start_only"
FMT_END = f"%CHROM{TAB}%POS{TAB}%ID{TAB}%END\\n"

F7_TEMPLATES = (
    Template("view -R", ("view", "-H", "-R", "{R}", VCF), 2, OVERLAP),
    Template("view -T", ("view", "-H", "-T", "{R}", VCF), 2, START),
    Template("view --regions-file", ("view", "-H", "--regions-file", "{R}", VCF), 2, OVERLAP),
    Template(
        "view --regions-file= --regions-overlap=1",
        ("view", "-H", "--regions-file={R}", "--regions-overlap=1", VCF),
        2,
        OVERLAP,
    ),
    Template(
        "view -R --regions-overlap record",
        ("view", "-H", "-R", "{R}", "--regions-overlap", "record", VCF),
        2,
        OVERLAP,
    ),
    Template("view -r strings", ("view", "-H", "-r", "{S}", VCF), 2, OVERLAP),
    Template("view --targets-file=", ("view", "-H", "--targets-file={R}", VCF), 2, START),
    Template(
        "view -T --targets-overlap 0",
        ("view", "-H", "-T", "{R}", "--targets-overlap", "0", VCF),
        2,
        START,
    ),
    Template(
        "view -T --targets-overlap pos",
        ("view", "-H", "-T", "{R}", "--targets-overlap=pos", VCF),
        2,
        START,
    ),
    Template("view -t strings", ("view", "-H", "-t", "{S}", VCF), 2, START),
    Template(
        "view -R --regions-overlap 0",
        ("view", "-H", "-R", "{R}", "--regions-overlap", "0", VCF),
        2,
        START,
    ),
    Template(
        "view -T --targets-overlap 1",
        ("view", "-H", "-T", "{R}", "--targets-overlap", "1", VCF),
        2,
        OVERLAP,
    ),
    Template(
        "view -T --targets-overlap=2",
        ("view", "-H", "-T", "{R}", "--targets-overlap=2", VCF),
        2,
        OVERLAP,
    ),
    Template(
        "view -t strings --targets-overlap record",
        ("view", "-H", "-t", "{S}", "--targets-overlap", "record", VCF),
        2,
        OVERLAP,
    ),
    Template("query %END -R", ("query", "-f", FMT_END, "-R", "{R}", VCF), 2, OVERLAP),
    Template("query %ID -R", ("query", "-f", "%ID\\n", "-R", "{R}", VCF), 0, OVERLAP),
    Template("query %ID -T", ("query", "-f", "%ID\\n", "-T", "{R}", VCF), 0, START),
    Template(
        "query %INFO/END -T --targets-overlap 1",
        ("query", "-f", f"%ID{TAB}%INFO/END\\n", "-T", "{R}", "--targets-overlap", "1", VCF),
        0,
        OVERLAP,
    ),
    Template("filter -R", ("filter", "-i", "POS>0", "-R", "{R}", VCF), 2, OVERLAP),
    Template("filter -T", ("filter", "-i", "POS>0", "-T", "{R}", VCF), 2, START),
    Template("annotate -R", ("annotate", "-R", "{R}", VCF), 2, OVERLAP),
    Template("norm -m-any -T", ("norm", "-m-any", "-T", "{R}", VCF), 2, START),
)
F7_ANCHORS = (0, 1)  # view -R and view -T are in every seed


def _region_strings(work: Path) -> str:
    rows = (line.split("\t") for line in (work / "regions.bed").read_text().splitlines())
    return ",".join(f"{c}:{int(s) + 1}-{e}" for c, s, e in rows)


@case_source(SEMANTICS_RULE)
def region_semantics_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    tasks = _build_twins("F7", seed, scratch, tools)
    rest = [i for i in range(len(F7_TEMPLATES)) if i not in F7_ANCHORS]
    chosen = [*F7_ANCHORS, *(rest[i] for i in _rotate(seed, 0, len(rest), 4))]
    plan: list[tuple[str, Template, Run]] = []
    for twin in tasks:
        work = scratch / twin
        strings = _region_strings(work)
        for t in chosen:
            template = F7_TEMPLATES[t]
            argv = _fill(template, "regions.bed", strings)
            plan.append((twin, template, Run(work, argv, template.id_col)))
    cases: list[Case] = []
    for (twin, template, run), ids in zip(plan, _run_all(tools, [p[2] for p in plan]), strict=True):
        if ids is None:
            continue
        label: Label = "right" if answer_matches(tasks[twin].truth, ids) else "wrong"
        declarations: list[str | None] = [None, template.semantic]
        if template.semantic == START and label == "wrong":
            declarations.append(OVERLAP)  # believes -T/-t overlap: declared, and mistaken
        for value in declarations:
            declared = {SEMANTICS_KEY: value} if value else {}
            call = Call("bcftools", run.argv, run.work, declared)
            note = f"{template.name} {value or 'undeclared'}"
            cases.append(Case(call, label, value is not None, "F7", seed, twin, note))
    return cases
