"""Admission cases for vcf.indel_interval_anchor (family F10, slot interval_closure).

For one seed both twins of the F10 task are built (implicit tier: the README does not describe the
anchor). The faulted twin has 6-8 deletions written the VCF way, the clean twin has the same sites
as substitutions, so the correct BED is the same in both and a REF-span interval is right only for
the clean twin. Every call below is run with the real tools in the workspace and its output is
compared with `task.truth["expected"]` (bench.score.answer_matches); the label never comes from the
rule. Calls that compute REF spans (`%POS0/%END`, `$2-1 .. $2-1+length(REF)`, with or without
printing ALT) are right on the clean twin and wrong on the faulted one; calls that compare REF with
ALT are right on both; two calls are wrong for other reasons (they skip the first base of every
multi-base REF, or of every record) and show what the rule cannot see. Inspection calls (a plain
view, a count) produce no interval; they are labelled right (they cannot be wrong) and are the
zero-friction probes on the faulted twin.

Each call is emitted without a declaration and with one: right calls carry the meaning they
compute (`ref_span` for a REF span, `replaced_bases` for an anchor-aware one); wrong calls carry
both meanings in turn (declared but botched, or the wrong meaning chosen knowingly).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call

KEY = "interval_meaning"
TAB = "\\t"


@dataclass(frozen=True)
class Template:
    name: str
    tool: str
    argv: tuple[str, ...]
    meaning: str  # what the call computes: ref_span | replaced_bases | none (inspection)
    inspection: bool = False


def _sh(name: str, pipeline: str, meaning: str, inspection: bool = False) -> Template:
    return Template(name, "bash", ("-c", pipeline), meaning, inspection)


FMT_SPAN = f"%CHROM{TAB}%POS0{TAB}%END\\n"
FMT_ALLELES = f"%CHROM{TAB}%POS{TAB}%REF{TAB}%ALT\\n"

# a Python program that prints POS-1 and POS-1+len(REF), with an optional anchor skip
PY_HEAD = (
    "import gzip\n"
    "for line in gzip.open('variants.vcf.gz', 'rt'):\n"
    "    if line.startswith('#'):\n"
    "        continue\n"
    "    f = line.rstrip('\\n').split('\\t')\n"
    "    s = int(f[1]) - 1\n"
)
PY_WRONG = PY_HEAD + "    print(f[0], s, s + len(f[3]), sep='\\t')\n"
PY_RIGHT = (
    PY_HEAD
    + "    ref, alt = f[3], f[4]\n"
    + "    if len(ref) > 1 and len(alt) == 1 and ref[0] == alt:\n"
    + "        s += 1\n"
    + "    print(f[0], s, int(f[1]) - 1 + len(ref), sep='\\t')\n"
)
PY_COUNT = (
    "import gzip\n"
    "print(sum(1 for l in gzip.open('variants.vcf.gz', 'rt') if not l.startswith('#')))\n"
)

AWK_VIEW = "bcftools view -H variants.vcf.gz | awk -F'\\t' -v OFS='\\t' "
AWK_QUERY = f"bcftools query -f '{FMT_ALLELES}' variants.vcf.gz | awk -F'\\t' -v OFS='\\t' "

TEMPLATES: tuple[Template, ...] = (
    # --- REF-span calls: right on the clean twin, one base early for deletions on the faulted one
    Template(
        "query_pos0_end", "bcftools", ("query", "-f", FMT_SPAN, "variants.vcf.gz"), "ref_span"
    ),
    Template(
        "query_regions_pos0_end",
        "bcftools",
        ("query", "-r", "chr1,chr2", "--format", FMT_SPAN, "variants.vcf.gz"),
        "ref_span",
    ),
    _sh("view_awk_ref_span", AWK_VIEW + "'{print $1,$2-1,$2-1+length($4)}'", "ref_span"),
    _sh(
        "query_awk_alt_printed_ref_measured",
        AWK_QUERY + "'{print $1,$2-1,$2-1+length($3)}'",
        "ref_span",
    ),
    _sh(
        "zcat_grep_awk_ref_span",
        "zcat variants.vcf.gz | grep -v '^#' | "
        'awk \'{print $1"\\t"($2-1)"\\t"($2-1+length($4))}\'',
        "ref_span",
    ),
    Template(
        "awk_plain_vcf_ref_span",
        "awk",
        ("-F", "\t", "-v", "OFS=\t", "!/^#/ {print $1, $2-1, $2+length($4)-1}", "variants.vcf"),
        "ref_span",
    ),
    Template("python_ref_span", "python3", ("-c", PY_WRONG), "ref_span"),
    _sh(
        "perl_ref_span",
        "zcat variants.vcf.gz | perl -F'\\t' -lane 'next if /^#/; "
        'print join("\\t", $F[0], $F[1]-1, $F[1]-1+length($F[3]))\'',
        "ref_span",
    ),
    # --- anchor-aware calls: right on both twins
    _sh(
        "query_awk_compare_lengths",
        AWK_QUERY + "'{s=$2-1; if (length($3)>length($4) && substr($3,1,1)==substr($4,1,1)) s++; "
        "print $1,s,$2-1+length($3)}'",
        "replaced_bases",
    ),
    _sh(
        "view_awk_alt_is_anchor",
        AWK_VIEW + "'{s=$2-1; if (length($4)>1 && length($5)==1 && substr($4,1,1)==$5) s++; "
        "print $1,s,$2-1+length($4)}'",
        "replaced_bases",
    ),
    Template(
        "awk_plain_vcf_compare",
        "awk",
        (
            "-F",
            "\t",
            "-v",
            "OFS=\t",
            "!/^#/ {s=$2-1; if (length($4)>length($5)) s++; print $1,s,$2-1+length($4)}",
            "variants.vcf",
        ),
        "replaced_bases",
    ),
    Template("python_anchor_skip", "python3", ("-c", PY_RIGHT), "replaced_bases"),
    _sh(
        "perl_anchor_skip",
        "zcat variants.vcf.gz | perl -F'\\t' -lane 'next if /^#/; $s=$F[1]-1; "
        "$s++ if length($F[3])>length($F[4]); "
        'print join("\\t", $F[0], $s, $F[1]-1+length($F[3]))\'',
        "replaced_bases",
    ),
    # --- wrong for another reason: the rule is not meant to see these
    _sh(  # skips the first base of every multi-base REF, MNVs included, and never reads ALT
        "view_awk_skip_all_multibase",
        AWK_VIEW + "'{s=$2-1; if (length($4)>1) s++; print $1,s,$2-1+length($4)}'",
        "replaced_bases",
    ),
    _sh(  # reads ALT but skips the first base of every record with a one-base ALT, SNVs included
        "view_awk_skip_all_snv_alt",
        AWK_VIEW + "'{s=$2-1; if (length($5)==1) s++; print $1,s,$2-1+length($4)}'",
        "replaced_bases",
    ),
    # --- inspection: no interval produced, cannot be wrong
    _sh("view_head", "bcftools view -H variants.vcf.gz | head -5", "none", True),
    _sh("view_count", "bcftools view -H variants.vcf.gz | wc -l", "none", True),
    _sh("zcat_count", "zcat variants.vcf.gz | grep -vc '^#'", "none", True),
    Template("python_count", "python3", ("-c", PY_COUNT), "none", True),
    Template(
        "query_alleles",
        "bcftools",
        ("query", "-f", FMT_ALLELES, "variants.vcf.gz"),
        "none",
        True,
    ),
    Template("view_header", "bcftools", ("view", "-h", "variants.vcf.gz"), "none", True),
)


def _label(template: Template, task, proc) -> Label:
    if proc.returncode != 0:
        return "wrong"
    if template.inspection:
        return "right"
    answer = [line for line in proc.stdout.splitlines() if line]
    return "right" if answer_matches(task.truth, answer) else "wrong"


def _declarations(template: Template, label: Label) -> list[dict[str, str]]:
    if template.inspection:
        return [{KEY: "replaced_bases"}]
    if label == "right":
        return [{KEY: template.meaning}]
    return [{KEY: "ref_span"}, {KEY: "replaced_bases"}]


def _run(tools: ToolEnv, work: Path, template: Template):
    return tools.run([template.tool, *template.argv], work)


@case_source("vcf.indel_interval_anchor")
def cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F10", seed, twin, work, tools, "implicit", "free")
        plain = tools.run(
            ["bcftools", "view", "--no-version", "-o", "variants.vcf", "variants.vcf.gz"], work
        )
        if plain.returncode != 0:
            raise RuntimeError(f"could not write variants.vcf: {plain.stderr}")
        with ThreadPoolExecutor(max_workers=16) as pool:
            procs = list(pool.map(partial(_run, tools, work), TEMPLATES))
        for template, proc in zip(TEMPLATES, procs, strict=True):
            label = _label(template, task, proc)
            calls: Sequence[tuple[bool, dict[str, str]]] = [(False, {})] + [
                (True, d) for d in _declarations(template, label)
            ]
            for declared, decl in calls:
                call = Call(template.tool, template.argv, work, decl)
                out.append(Case(call, label, declared, "F10", seed, twin, template.name))
    return out
