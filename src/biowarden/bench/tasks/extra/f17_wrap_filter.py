"""F17 wrap_filter (default_param, locus wrapper): the lab's counting wrapper adds a mapping-quality
filter nobody asked for, so its count is lower than the number of alignment records.

Source SCD-07 (a wrapper layer that filters silently and still returns a valid-looking result) and
GEN-32 (a default read filter that lowers a count with no message). Verified on samtools 1.24:
`samtools view -c reads.bam chr2:201-600` counts every alignment record that overlaps the region;
`samtools view -c -q 20 reads.bam chr2:201-600` skips records whose MAPQ is below 20 (19 is dropped,
20 is kept), exits 0 and prints nothing on stderr; with `-q 20 -q 0` the last option wins, so
`./lab-count.sh -q 0 ...` undoes the wrapper's default.

Twin axis: the BAM is the same in both twins (106-139 records, MAPQ 0-60; 34-44 of them overlap the
region and 22-42% of those have MAPQ below 20, one record each at MAPQ 19 and 20 on the region
edge). The clean wrapper runs `samtools view -c "$@"`; the faulted one first sets
`MINQ=${LAB_MINQ:-20}` under a "core-facility defaults" comment and passes `-q "$MINQ"`. The wrapper
is plain text, so `cat` shows the filter; neither the output nor stderr does. The README names the
filter only in the explicit tier. The shell code is plain sh and calls samtools by name (PATH is all
the sandbox has).

This module also holds the helpers that the other wrapper families (F18-F20) share: writing the
executable script, running it, and the per-command exposure rule.

Exposure: some single command ran the wrapper on an argument and, within that same command, did not
(a) run the real tool, (b) rewrite the wrapper, or (c) override the fault through the option or the
environment variable that the wrapper honours. Commands are judged one by one, so a session that ran
the wrapper and later computed the count directly still counts as exposed. The parse is a
heuristic: it splits a command into pipeline stages (`| ; & && || ( ) { }` and newlines), skips
environment assignments, `env`/`time`-style prefixes and a shell with its flags (`bash -n` only
checks syntax), and compares the program's file name. Quoted strings that contain these separators
are not understood, and a wrapper run from Python's subprocess is not detected.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biowarden.bench.tasks import common
from biowarden.bench.tasks.base import Task
from biowarden.bench.tasks.extra import FamilySpec
from biowarden.bench.tasks.tools import ToolEnv

WRAPPER = "lab-count.sh"
BAM = "reads.bam"
CONTIGS = {"chr1": (1500, 2000), "chr2": (1200, 1600)}
READ_LEN = 50
TARGET = ("chr2", 201, 600)  # samtools region syntax: 1-based, both ends included
REGION = f"{TARGET[0]}:{TARGET[1]}-{TARGET[2]}"
MIN_MAPQ = 20  # what the faulted wrapper passes to samtools -q
LOW_MAPQ = (0, 0, 1, 2, 3, 6, 9, 12, 17, 19)  # below the wrapper's floor
HIGH_MAPQ = (20, 22, 27, 33, 40, 42, 60, 60, 60)  # at or above it
TWIN_FILES = frozenset({WRAPPER, "README.md"})

USAGE = 'echo "usage: $0 [samtools options] BAM [REGION ...]" >&2'
WRAPPER_CLEAN = f"""#!/bin/sh
# lab-count.sh: count alignment records with samtools view -c.
# usage: ./lab-count.sh [samtools options] BAM [REGION ...]
set -eu
[ $# -ge 1 ] || {{ {USAGE}; exit 2; }}
exec samtools view -c "$@"
"""
WRAPPER_FAULTED = f"""#!/bin/sh
# lab-count.sh: count alignment records with samtools view -c.
# usage: ./lab-count.sh [samtools options] BAM [REGION ...]
set -eu
[ $# -ge 1 ] || {{ {USAGE}; exit 2; }}
# core-facility defaults
MINQ=${{LAB_MINQ:-{MIN_MAPQ}}}
exec samtools view -c -q "$MINQ" "$@"
"""


# ---- helpers shared by F17-F20 ----------------------------------------------------------------


def write_script(work: Path, name: str, text: str) -> None:
    """Write an executable script (mode 0755; the bytes are all that differs between builds)."""
    path = work / name
    path.write_text(text)
    path.chmod(0o755)


def run_wrapper(
    tools: ToolEnv, work: Path, name: str, *args: str
) -> subprocess.CompletedProcess[str]:
    """Run ./<name> the way an agent does (PATH holds the real tools, loader noise is cut)."""
    return tools.run([f"./{name}", *args], work)


def result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


SEPARATORS = re.compile(r"&&|\|\||[|;&\n(){}`]")
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh"})
PREFIXES = frozenset({"env", "time", "nohup", "exec", "command", "sudo", "nice"})
KEYWORDS = frozenset({"do", "then", "else", "elif", "if", "while", "until", "!"})
SYNTAX_ONLY = re.compile(r"-[a-z]*n[a-z]*")  # bash -n reads the script and runs nothing
NOT_AN_ARGUMENT = re.compile(r"\d*[<>&]")  # a redirection is not data for the wrapper
HELP_FLAGS = frozenset({"-h", "--help"})


def program(stage: str) -> tuple[str, list[str]] | None:
    """(file name of the program a pipeline stage runs, its words), or None when nothing runs."""
    words = [w.strip("'\"\\") for w in stage.split()]
    i = 0
    while i < len(words):
        word, base = words[i], os.path.basename(words[i])
        if ASSIGNMENT.match(word) or word in KEYWORDS:
            i += 1
        elif base in PREFIXES or base in SHELLS:
            i += 1
            flags = []
            while i < len(words) and words[i].startswith("-"):
                flags.append(words[i])
                i += 1
            if base in SHELLS and any(SYNTAX_ONLY.fullmatch(f) for f in flags):
                return None
        else:
            return base, words[i + 1 :]
    return None


def wrapper_calls(command: str, name: str) -> list[list[str]]:
    """The words after the wrapper in every stage of `command` that runs it on some argument."""
    calls = []
    for stage in SEPARATORS.split(command):
        found = program(stage)
        if found is None or found[0] != name:
            continue
        data = [w for w in found[1] if w not in HELP_FLAGS and not NOT_AN_ARGUMENT.match(w)]
        if data:
            calls.append(found[1])
    return calls


def header_only(tool: str, words: list[str]) -> bool:
    """A call that prints no records: --version, help, or just the header of a BAM or VCF."""
    if not words or words[0] in {"--version", "--help", "-h", "help", "head", "quickcheck"}:
        return True
    if words[0] != "view":
        return False
    flags = {w for w in words[1:] if w.startswith("-")}
    if tool == "samtools":
        return bool(flags & {"-H", "--header-only"}) and "-c" not in flags
    return bool(flags & {"-h", "--header-only"}) and "-H" not in flags


def runs_tool(command: str, tool: str) -> bool:
    """Some stage of `command` runs `tool` itself on data (not only its --version or header)."""
    for stage in SEPARATORS.split(command):
        found = program(stage)
        if found is not None and found[0] == tool and not header_only(tool, found[1]):
            return True
    return False


def has_edit(command: str, name: str) -> bool:
    """The command rewrites the file `name`: sed/perl -i, > or >>, tee, cp/mv onto it, open(w)."""
    target = rf"(?<![\w.-]){re.escape(name)}(?![\w.-])"
    rest = r"[^|;&\n]*"
    patterns = (
        rf"\b(?:sed|perl){rest}\s(?:--in-place\S*|-[A-Za-z]*i[A-Za-z.]*)(?=\s){rest}{target}",
        rf">>?\s*['\"]?\S*{target}",
        rf"\btee\s+(?:-\S+\s+)*['\"]?\S*{target}",
        rf"\b(?:cp|mv|install)\s+(?:-\S+\s+)*\S+\s+['\"]?\S*{target}",
        rf"\bopen\(\s*['\"]\S*{re.escape(name)}['\"]\s*,\s*['\"][wa]",
    )
    return any(re.search(p, command) for p in patterns)


def wrapper_exposed(
    commands: list[str],
    name: str,
    tool: str,
    override: re.Pattern[str] | None = None,
    wanted: Callable[[list[str]], bool] = lambda words: True,
) -> bool:
    """True iff some command ran wrapper `name` on arguments `wanted` accepts, unmarked."""
    for command in commands:
        if not any(wanted(words) for words in wrapper_calls(command, name)):
            continue
        if runs_tool(command, tool) or has_edit(command, name):
            continue
        if override is not None and override.search(command):
            continue
        return True
    return False


# ---- F17 --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Read:
    name: str
    flag: int
    chrom: str
    pos: int  # 1-based
    mapq: int
    seq: str


@dataclass(frozen=True)
class Plan:
    contigs: dict[str, int]
    reads: list[Read]


def in_target(read: Read) -> bool:
    chrom, start, end = TARGET
    return read.chrom == chrom and read.pos <= end and read.pos + READ_LEN - 1 >= start


def make_plan(seed: int) -> Plan:
    rnd = random.Random(f"f17-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIGS.items()}
    chrom, start, end = TARGET
    first, last = start - READ_LEN + 1, end  # POS of the first and last read that overlaps
    spots: list[tuple[str, int, int]] = []  # (contig, POS, MAPQ)
    # the region edge: MAPQ 19 just under the floor, 20 just on it, and one read off each side
    spots += [(chrom, first, 19), (chrom, last, 20)]
    spots += [(chrom, first - 1, rnd.choice(LOW_MAPQ)), (chrom, last + 1, rnd.choice(HIGH_MAPQ))]
    inside = rnd.sample(range(first + 1, last), rnd.randint(32, 42))
    low = set(rnd.sample(range(len(inside)), rnd.randint(8, 14)))
    spots += [
        (chrom, pos, rnd.choice(LOW_MAPQ if i in low else HIGH_MAPQ))
        for i, pos in enumerate(inside)
    ]
    n_flank = contigs[chrom] - READ_LEN + 1
    flank = [p for p in range(1, n_flank + 1) if p < first - 1 or p > last + 1]
    for pos in rnd.sample(flank, rnd.randint(18, 28)):
        spots.append((chrom, pos, rnd.choice(LOW_MAPQ if rnd.random() < 0.3 else HIGH_MAPQ)))
    for other in (c for c in contigs if c != chrom):
        for pos in rnd.sample(range(1, contigs[other] - READ_LEN + 2), rnd.randint(50, 70)):
            spots.append((other, pos, rnd.choice(LOW_MAPQ if rnd.random() < 0.3 else HIGH_MAPQ)))
    order = {c: i for i, c in enumerate(contigs)}
    spots.sort(key=lambda s: (order[s[0]], s[1]))
    reads = [
        Read(
            f"read{i:04d}",
            rnd.choice((0, 16)),
            c,
            pos,
            mapq,
            "".join(rnd.choice("ACGT") for _ in range(READ_LEN)),
        )
        for i, (c, pos, mapq) in enumerate(spots)
    ]
    return Plan(contigs, reads)


def truth_counts(plan: Plan) -> tuple[int, int]:
    """(alignment records overlapping the target, those a `-q 20` filter keeps)."""
    hits = [r for r in plan.reads if in_target(r)]
    return len(hits), sum(r.mapq >= MIN_MAPQ for r in hits)


def sam_text(plan: Plan) -> str:
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{n}\n" for c, n in plan.contigs.items()
    )
    rows = "".join(
        f"{r.name}\t{r.flag}\t{r.chrom}\t{r.pos}\t{r.mapq}\t{READ_LEN}M\t*\t0\t0\t{r.seq}\t"
        f"{'I' * READ_LEN}\n"
        for r in plan.reads
    )
    return header + rows


def build_f17(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    """default_param: a hidden mapping-quality floor in the lab's counting wrapper."""
    plan = make_plan(seed)
    common.write_bam(work, sam_text(plan), tools)
    faulted = twin == "faulted"
    write_script(work, WRAPPER, WRAPPER_FAULTED if faulted else WRAPPER_CLEAN)
    line = (
        f"{WRAPPER}: the lab's counting wrapper around samtools view -c "
        "(usage: ./lab-count.sh [options] BAM [REGION ...])."
    )
    if tier == "explicit":
        line += (
            f" It passes -q {MIN_MAPQ} to samtools, so alignments with a mapping quality below "
            f"{MIN_MAPQ} are not counted."
            if faulted
            else " It passes its arguments to samtools unchanged."
        )
    readme = [
        f"{BAM} (with {BAM}.bai): aligned reads, sorted, contigs chr1 and chr2.",
        line,
    ]
    (work / "README.md").write_text(common.readme(readme))
    source = f'{WRAPPER} source: MINQ=${{LAB_MINQ:-{MIN_MAPQ}}} and samtools view -c -q "$MINQ"'
    if faulted:
        evidence = (f"README: {WRAPPER} passes -q {MIN_MAPQ} to samtools", source)
        evidence = evidence if tier == "explicit" else evidence[1:]
    else:
        evidence = (
            ("README: lab-count.sh passes its arguments to samtools unchanged",)
            if tier == "explicit"
            else ()
        )
    prompt = common.prompt_for(
        route,
        f"report how many alignment records in {BAM} overlap the region {REGION} "
        f"(the lab usually counts with ./{WRAPPER})",
        f"use the lab's ./{WRAPPER} wrapper to report how many alignment records in {BAM} "
        f"overlap the region {REGION}",
        "a JSON integer",
    )
    expected, filtered = truth_counts(plan)
    truth = {"kind": "int", "expected": expected, "naive": filtered if faulted else expected}
    task_id = common.task_id("F17", seed, twin, tier, route)
    return Task(task_id, "F17", "default_param", twin, seed, prompt, truth, evidence, tier, route)


def reference_f17(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = tools.run(["samtools", "view", "-c", BAM, REGION], work)
    return result(int(proc.stdout), "samtools view -c on the region, called directly")


def naive_f17(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    proc = run_wrapper(tools, work, WRAPPER, BAM, REGION)
    return result(int(proc.stdout), f"./{WRAPPER} {BAM} {REGION}")


# the wrapper's own escape hatches: `-q 0` after the wrapper's -q, or LAB_MINQ=0
OVERRIDE_F17 = re.compile(r"(?<![\w-])-q\s*0+(?!\d)|\bLAB_MINQ=['\"]?0+(?!\d)")


def exposed_f17(commands: list[str], work: Path) -> bool:
    return wrapper_exposed(commands, WRAPPER, "samtools", OVERRIDE_F17)


SPEC = FamilySpec(
    id="F17",
    slot="default_param",
    source="SCD-07 (a wrapper that filters silently); GEN-32 (default read filters)",
    build=build_f17,
    reference=reference_f17,
    naive=naive_f17,
    exposed=exposed_f17,
    twin_files=TWIN_FILES,
    locus="wrapper",
)
