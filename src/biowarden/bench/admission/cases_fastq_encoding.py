"""Admission cases for fastq.phred_offset (family F13, slot units).

For one seed both twins of the F13 task are built (implicit tier: the README does not state the
encoding): the same reads and Phred scores written as Phred+33 (clean) and Phred+64 (faulted). The
task is to count the bases with Q >= 30. Every call below is run with the real tools in the
workspace and its output is compared with `task.truth["expected"]` (bench.score.answer_matches);
the label never comes from the rule. A call that shifts by 33 is right on the clean twin and counts
every base on the faulted one, a call that shifts by 64 is the reverse, and a call that detects the
offset from the data is right on both. The calls are awk, python3 and perl, written as a direct
tool call or as a `bash -c` pipeline (an ord table, an offset variable, a character cutoff, a
`sed | awk` stage, a heredoc), plus one wrong for another reason (a strict `>` cutoff). Inspection
calls (head, wc, cat, grep, a min/max of the quality characters) produce no count; they are labelled
right (they cannot be wrong) and are the zero-friction probes on the faulted twin.

Each call is emitted without a declaration and with one. Right calls carry the true offset of the
twin. Wrong calls carry the true offset (declared correctly, code botched: the rule cannot see it)
and the offset their code uses (declared wrongly: contradicts the data on the faulted twin).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call

KEY = "fastq_quality_offset"

AWK_TABLE = (
    'BEGIN{for(i=33;i<127;i++)o[sprintf("%c",i)]=i} NR%4==0{for(j=1;j<=length($0);j++)'
    "if(o[substr($0,j,1)]-{off}>=30)n++} END{print n+0}"
)
AWK_VAR = (
    'BEGIN{for(i=33;i<127;i++)o[sprintf("%c",i)]=i} NR%4==0{for(j=1;j<=length($0);j++)'
    "if(o[substr($0,j,1)]-off>=30)n++} END{print n}"
)
AWK_CHAR = 'NR%4==0{for(i=1;i<=length($0);i++)if(substr($0,i,1)>="{char}")c++}END{print c+0}'
PY_ORD = (
    "print(sum(ord(c)-{off}>=30 for l in open('reads.fastq').read().split('\\n')[3::4] for c in l))"
)
PY_AUTO = (
    "q = ''.join(open('reads.fastq').read().split('\\n')[3::4])\n"
    "o = 33 if min(map(ord, q)) < 59 else 64\n"
    "print(sum(ord(c) - o >= 30 for c in q))"
)
PY_STDIN = "import sys; print(sum(ord(c)-{off}>=30 for l in sys.stdin for c in l.strip()))"
PY_HEREDOC = (
    "python3 - <<'EOF'\n"
    "n = 0\n"
    "for i, line in enumerate(open('reads.fastq')):\n"
    "    if i % 4 == 3:\n"
    "        n += sum(ord(c) - {off} >= 30 for c in line.strip())\n"
    "print(n)\n"
    "EOF"
)
PERL_ORD = (
    "my $n=0; while(<>){ if ($.%4==0){ chomp; "
    "for my $c (split //){ $n++ if ord($c)-{off}>=30 } } } print $n+0;"
)
PY_MINMAX = "q=open('reads.fastq').read().split('\\n')[3::4]; print(min(q), max(q))"


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    offset: str  # the offset the code uses: "33", "64", "auto" or "none" (inspection)
    inspection: bool = False


def _awk(name: str, program: str, offset: str, *flags: str) -> Template:
    return Template(name, "awk", (*flags, program, "reads.fastq"), offset)


def _sh(name: str, pipeline: str, offset: str, inspection: bool = False) -> Template:
    return Template(name, "bash", ("-c", pipeline), offset, inspection)


def _pair(build, offsets=("33", "64")) -> list[Template]:
    return [build(off) for off in offsets]


TEMPLATES: tuple[Template, ...] = (
    *_pair(lambda o: _awk(f"awk_table_{o}", AWK_TABLE.replace("{off}", o), o)),
    *_pair(lambda o: _awk(f"awk_var_{o}", AWK_VAR, o, "-v", f"off={o}")),
    *_pair(
        lambda o: _awk(
            f"awk_char_cutoff_{o}", AWK_CHAR.replace("{char}", "?" if o == "33" else "^"), o
        )
    ),
    *_pair(lambda o: Template(f"python_ord_{o}", "python3", ("-c", PY_ORD.replace("{off}", o)), o)),
    Template("python_autodetect", "python3", ("-c", PY_AUTO), "auto"),
    *_pair(
        lambda o: _sh(
            f"sed_pipe_python_{o}",
            "sed -n '4~4p' reads.fastq | python3 -c \"" + PY_STDIN.replace("{off}", o) + '"',
            o,
        )
    ),
    *_pair(lambda o: _sh(f"python_heredoc_{o}", PY_HEREDOC.replace("{off}", o), o)),
    *_pair(
        lambda o: Template(
            f"perl_ord_{o}", "perl", ("-e", PERL_ORD.replace("{off}", o), "reads.fastq"), o
        )
    ),
    # wrong for another reason: a strict cutoff with the offset of 33
    _awk("awk_table_33_strict", AWK_TABLE.replace("{off}", "33").replace(">=30", ">30"), "33"),
    # inspection: no count produced, cannot be wrong
    Template("head", "head", ("-8", "reads.fastq"), "none", True),
    Template("wc", "wc", ("-l", "reads.fastq"), "none", True),
    Template("cat", "cat", ("reads.fastq",), "none", True),
    Template("grep_count", "grep", ("-c", "^@", "reads.fastq"), "none", True),
    Template("awk_quality_lines", "awk", ("NR%4==0", "reads.fastq"), "none", True),
    Template("python_minmax", "python3", ("-c", PY_MINMAX), "none", True),
    _sh("quality_lines_head", "awk 'NR%4==0' reads.fastq | head -3", "none", True),
    _sh("read_count_awk", "awk 'END{print NR/4}' reads.fastq", "none", True),
)


def _label(template: Template, task, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    if template.inspection:
        return "right"
    return "right" if answer_matches(task.truth, proc.stdout.strip()) else "wrong"


def _declarations(template: Template, label: Label, twin: str) -> list[dict[str, str]]:
    true_offset = "33" if twin == "clean" else "64"
    if template.inspection:
        return [{KEY: true_offset}]
    if label == "right":
        return [{KEY: true_offset}]
    options = [true_offset]
    if template.offset in ("33", "64") and template.offset != true_offset:
        options.append(template.offset)
    return [{KEY: o} for o in options]


def _run(tools: ToolEnv, work: Path, template: Template):
    return tools.run([template.tool, *template.argv], work)


@case_source("fastq.phred_offset")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F13", seed, twin, work, tools, "implicit", "free")
        with ThreadPoolExecutor(max_workers=16) as pool:
            procs = list(pool.map(partial(_run, tools, work), TEMPLATES))
        for template, proc in zip(TEMPLATES, procs, strict=True):
            label = _label(template, task, proc)
            out.append(
                Case(
                    Call(template.tool, template.argv, work),
                    label,
                    False,
                    "F13",
                    seed,
                    twin,
                    template.name,
                )
            )
            for decl in _declarations(template, label, twin):
                call = Call(template.tool, template.argv, work, decl)
                out.append(Case(call, label, True, "F13", seed, twin, template.name))
    return out
