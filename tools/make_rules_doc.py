"""Regenerate docs/rules.md, the catalogue of the checking layer's rules.

    python tools/make_rules_doc.py > docs/rules.md
    python tools/make_rules_doc.py --check        # exit 1 when docs/rules.md is out of date

One row per rule, grouped by domain. The data of the rules are not typed in by hand:

    rules        biowarden.engine.spec.packaged_specs()
    calibration  biowarden.engine.Promotion.packaged(), the promotion table written by
                 `python -I -m biowarden.bench.admission`
    families     bench/tasks/family_rules.json (bench.stage2.load_family_rules), reversed

`--check` writes nothing and exits 1 when the page differs from what the code gives now;
tests/test_generated_docs.py runs it. A rule whose spec file belongs to no domain, a family map
that names a rule no spec defines, and a rule without a declaration whose two cells of wrong calls
differ in the promotion table stop the generator with a message (exit 2) rather than producing a
wrong page. The page holds no filesystem path.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import textwrap
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

from biowarden.bench.admission import MIN_CALLS, UPPER_LIMIT
from biowarden.bench.stage2 import FamilyRule, load_family_rules
from biowarden.engine import Promotion, Rule
from biowarden.engine.precheck import MAX_CHECKED_COMMANDS
from biowarden.engine.spec import packaged_specs
from biowarden.engine.tools import INTERPRETERS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "tools/make_rules_doc.py"
PAGE = ROOT / "docs" / "rules.md"
SPECS_LINK = "../src/biowarden/engine/specs"
WIDTH = 100  # prose is wrapped here; table rows are not
SUBCOMMANDS_SHOWN = 6  # a rule with more subcommands shows their number
OPTIONS_SHOWN = 8  # a rule with more options than this shows their number
PROGRAMS_SHOWN = 8  # a fact rule watching more programs shows three of them and a count
COMPRESSED = (".gz", ".bgz")

# The domains are the packs of docs/families.md: the titles must equal those of DOMAINS in
# tools/make_families_doc.py (tools/make_supported_tools_doc.py checks this). A rule is filed under
# the domain of the tool it watches, found by the name of its spec file. The last domain holds the
# rules about data formats (FASTQ, VCF, expression tables) rather than one tool; they are fact
# rules written for families of the original pack (tests/test_generated_docs.py checks this).
DOMAINS = (
    ("Read alignment", ("aln_",)),
    ("RNA-seq quantification", ("quant_",)),
    ("Variants and genotypes", ("var_", "bcftools_regions")),
    ("Intervals, peaks, liftover, coverage tracks", ("iv_",)),
    ("Sequence processing, search and multiple alignment", ("seq_",)),
    ("Protein structure and protein-model plumbing", ("struct_", "structure_")),
    ("Single-cell and virtual-cell inputs", ("sc_",)),
    ("Original families", ("expression_matrix", "fastq_encoding", "vcf_intervals")),
)
DATA_FORMAT_DOMAIN = "Original families"

HEADERS = (
    "Rule",
    "Watches",
    "Convention",
    "Tier",
    "Kind",
    "Declaration",
    "Families",
    "Mode",
    "Seeds",
    "False blocks",
    "Upper bound",
    "Recall",
)
ALIGNMENT = ("---",) * 9 + ("---:",) * 3  # the last three columns are numbers


class GenerationError(Exception):
    """The page cannot be built from the current specs and tables."""


class Row(NamedTuple):
    domain: str
    spec: str
    rule: Rule
    families: tuple[str, ...]  # "F13", "F22 (even seeds)", "F23 (free route)"
    mode: str  # what a failing rule does in promoted mode: block or warn (and why not block)
    entry: Mapping[str, Any] | None  # the rule's line of the promotion table


def domain_of(spec_name: str) -> str:
    hits = [title for title, prefixes in DOMAINS if spec_name.startswith(prefixes)]
    if len(hits) != 1:
        raise GenerationError(
            f"spec file {spec_name}.yaml belongs to {len(hits)} domains; name it in DOMAINS "
            f"of {SCRIPT}"
        )
    return hits[0]


# -- the files a fact rule reads -------------------------------------------------------------------


def _group(text: str, i: int) -> tuple[list[str], int]:
    """The strings that the group starting at text[i] can match, and the index after its ')'.

    Handles what the packaged file patterns use: literals and escaped dots, `(?:a|b)` and `(?i:x)`
    groups, an optional `?` after a group or a character, and a run of name characters
    (`[...]*` or `[...]+`, written `*`). Anything else is an error: the generator should be taught
    the new construct, not guess.
    """
    branches: list[list[list[str]]] = [[]]
    while i < len(text):
        ch = text[i]
        if ch == ")":
            return [s for b in branches for s in map("".join, product(*b))], i + 1
        if ch == "|":
            branches.append([])
            i += 1
        elif text.startswith(("(?:", "(?i:"), i):
            inner, i = _group(text, text.index(":", i) + 1)
            if text[i : i + 1] == "?":
                inner, i = ["", *inner], i + 1
            branches[-1].append(inner)
        elif ch == "[":
            end = text.index("]", i) + 1
            if text[end : end + 1] not in ("*", "+"):
                raise GenerationError(f"file pattern {text!r}: a character class that is not a run")
            branches[-1].append(["*"])
            i = end + 1
        else:
            if ch == "\\":
                ch, i = text[i + 1], i + 1
                if ch != ".":
                    raise GenerationError(f"file pattern {text!r}: the escape \\{ch}")
            elif ch in "()*+?{}^$":
                raise GenerationError(f"file pattern {text!r}: unsupported {ch!r}")
            i += 1
            optional = text[i : i + 1] == "?"
            branches[-1].append(["", ch] if optional else [ch])
            i += optional
    raise GenerationError(f"file pattern {text!r}: a group is not closed")


def file_names(patterns: Sequence[str]) -> list[str]:
    """The file names a fact rule's patterns match, as globs; compressed copies are left out."""
    names: list[str] = []
    for pattern in patterns:
        body = re.sub(r"\(\?![^)]*\)$", "", re.sub(r"^\(\?<![^)]*\)", "", pattern))
        found, _ = _group(body + ")", 0)
        for name in found:
            if not re.search(pattern, name.replace("*", "x")):  # every name shown must match
                raise GenerationError(
                    f"file pattern {pattern!r} does not match its own name {name}"
                )
            if name not in names:
                names.append(name)
    return [n for n in names if not (n.endswith(COMPRESSED) and n.rsplit(".", 1)[0] in names)]


# -- the cells -------------------------------------------------------------------------------------


def code_list(items: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in items)


def watches(rule: Rule) -> str:
    applies = rule.applies
    if rule.kind == "call":
        tools, subs, options = applies.tools, applies.subcommands, applies.options_any
        if len(tools) == 1 and len(subs) == 1:
            what = f"`{tools[0]} {subs[0]}`"
        elif len(subs) > SUBCOMMANDS_SHOWN:
            what = f"{code_list(tools)} ({len(subs)} subcommands)"
        else:
            what = f"{code_list(tools)} ({code_list(subs)})" if subs else code_list(tools)
        if len(options) == 1:  # the rule applies only to calls that have one of its options
            what += f", with `{options[0]}`"
        elif len(options) > OPTIONS_SHOWN:
            what += f", with any of {len(options)} options"
        elif options:
            what += f", with any of {code_list(options)}"
        return what
    tools = applies.tools
    if not tools:
        programs = "any program"
    elif len(tools) <= PROGRAMS_SHOWN:
        programs = code_list(tools)
    else:
        programs = f"{code_list(tools[:3])} and {len(tools) - 3} more"
    return f"files {code_list(file_names(applies.file_patterns))} ({programs})"


def declaration(rule: Rule) -> str:
    """required: a call without it fails the rule; optional: asked for when the call is risky."""
    if rule.declare is None:
        return "none"
    return "required" if rule.kind == "call" and rule.declare.required else "optional"


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def calibration(rule: Rule, entry: Mapping[str, Any] | None) -> tuple[str, str, str, str]:
    """Seeds, false blocks, upper bound and recall, as the promotion table has them.

    The table counts every call of a rule that has no declaration twice, once as declared and once
    as undeclared (the case sources emit it in both cells), so the recall counts of such a rule are
    those of one cell; the generator stops when the two cells differ.
    """
    if entry is None:
        return ("-",) * 4
    failed, calls = entry["right_declared"]
    upper = entry["false_block_upper"]
    recall = entry["recall"]
    cells = [entry["cells"][cell] for cell in ("wrong_declared", "wrong_undeclared")]
    if rule.declare is None:
        if cells[0] != cells[1]:
            raise GenerationError(
                f"{rule.id}: no declaration, but the promotion table has different wrong calls "
                "with and without one"
            )
        cells = cells[:1]
    wrong = sum(cell["n"] for cell in cells)
    killed = sum(cell["failed"] for cell in cells)
    return (
        str(entry["seeds"]),
        f"{failed:,} of {calls:,}",
        "-" if upper is None else percent(upper),
        "-" if recall is None else f"{percent(recall)} ({killed:,} of {wrong:,})",
    )


# -- the data --------------------------------------------------------------------------------------


def families_by_rule(family_rules: Mapping[str, FamilyRule]) -> dict[str, list[str]]:
    """rule id -> the families it is written for, in family order, with where it is restricted."""

    def label(fid: str, *notes: str) -> str:
        return fid + (f" ({', '.join(notes)})" if notes else "")

    found: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for fid, family in family_rules.items():
        number = int(fid[1:])
        parity = (f"{family.seed_parity} seeds",) if family.seed_parity else ()
        for rule_id in family.rules:
            found[rule_id].append((number, label(fid, *parity)))
        for route, rule_ids in family.route_rules.items():
            for rule_id in rule_ids:
                if rule_id not in family.rules:
                    found[rule_id].append((number, label(fid, *parity, f"{route} route")))
    return {rule_id: [text for _, text in sorted(items)] for rule_id, items in found.items()}


def collect() -> list[Row]:
    """Every rule with its domain, families, effective mode and promotion entry."""
    promotion = Promotion.packaged()
    by_rule = families_by_rule(load_family_rules())
    rows: list[Row] = []
    seen: set[str] = set()
    for spec in packaged_specs():
        domain = domain_of(spec.name)
        spec_hash = spec.content_hash()
        for rule in spec.rules:
            if rule.id in seen:
                raise GenerationError(f"rule id {rule.id} is defined in more than one spec")
            seen.add(rule.id)
            entry = promotion.entries.get(rule.id)
            mode = "warn" if rule.tier == "T2" else promotion.mode_of(rule.id, spec_hash)
            if entry is None:
                mode += " (not calibrated)"
            elif entry.get("spec_hash") not in (None, spec_hash):
                mode += " (stale)"
            rows.append(Row(domain, spec.name, rule, tuple(by_rule.get(rule.id, ())), mode, entry))
    unknown = sorted(set(by_rule) - seen)
    if unknown:
        raise GenerationError(f"family_rules.json names rules that no spec defines: {unknown}")
    return rows


def span(values: Sequence[int]) -> str:
    low, high = min(values), max(values)
    return f"{low:,}" if low == high else f"{low:,} to {high:,}"


# -- the page --------------------------------------------------------------------------------------


def paragraph(text: str, bullet: bool = False) -> list[str]:
    """`text` wrapped at WIDTH, without a break inside a code span or a link text."""
    kept = re.sub(r"`[^`]*`|\[[^\]]*\]", lambda m: m.group(0).replace(" ", "\0"), text)
    lines = textwrap.wrap(
        kept,
        width=WIDTH,
        initial_indent="- " if bullet else "",
        subsequent_indent="  " if bullet else "",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return [line.replace("\0", " ") for line in lines]


def table(rows: Sequence[Row]) -> list[str]:
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join(ALIGNMENT) + "|"]
    for row in sorted(rows, key=lambda r: r.rule.id):
        rule = row.rule
        cells = [
            f"[`{rule.id}`]({SPECS_LINK}/{row.spec}.yaml)",
            watches(rule),
            rule.slot,
            rule.tier,
            rule.kind,
            declaration(rule),
            ", ".join(row.families) or "none",
            row.mode,
            *calibration(rule, row.entry),
        ]
        if any("|" in cell for cell in cells):
            raise GenerationError(f"{rule.id}: a table cell holds a pipe")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def ignores_case(rule: Rule) -> bool:
    return any("(?i" in pattern for pattern in rule.applies.file_patterns)


def legend(rows: Sequence[Row]) -> list[str]:
    with_options = sum(1 for row in rows if row.rule.applies.options_any)
    caseless = [row.rule.id for row in rows if ignores_case(row.rule)]
    interpreters = code_list(sorted(INTERPRETERS, key=str.lower))
    case_note = (
        f" Where a pattern ignores case ({code_list(caseless)}), the glob is shown in lower case."
        if caseless
        else ""
    )
    items = [
        "**Rule**: the rule id, `<tool or topic>.<name>`, linked to the spec file that defines it.",
        "**Watches**: for a call rule, the programs it applies to, with their subcommands (their "
        f"number when there are more than {SUBCOMMANDS_SHOWN}) and, for {with_options} rules, the "
        "options it needs: it applies only to a call that has at least one of them (their number "
        f"when there are more than {OPTIONS_SHOWN}). For a fact rule, the file names it reads, as "
        "globs (a compressed copy, `.gz` or `.bgz`, is matched too where the pattern in the spec "
        "allows it), and the programs it applies to." + case_note,
        "**Programs**: `any program` means that the rule names no program. The PATH shim and "
        "`biowarden` ask the engine only about the programs that rules name and the interpreters "
        f"{interpreters}; the whole-command pre-check asks about every simple command of the "
        f"string (the first {MAX_CHECKED_COMMANDS}). `python-sc`, `python-struct` and `python-bio` "
        "are the Python interpreters of the single-cell, protein-structure and sequencing-tool "
        "environments of [supported tools](supported-tools.md).",
        "**Convention**: the convention slot the rule guards, with the names used in the "
        "Convention column of [task families](families.md).",
        "**Tier**: T0 decides from the call alone and reads no file; T1 may read files and run "
        "the real tool for a dry run; T2 can only warn.",
        "**Kind**: `call`: the rule applies to a call of a listed program and decides from its "
        "arguments and, in tier T1, from files. `fact`: the rule applies when an argument, or "
        "code inside one, names an existing file of a listed kind; it reads the file and fires "
        "only when the data are in a risky state.",
        "**Declaration**: `required`: a call without `--declare key=value` fails the rule. "
        "`optional`: the rule asks for the declaration only when the call or the data are risky. "
        "`none`: the rule has no declaration.",
        "**Families**: the families whose trap the rule is written for, from "
        "[`family_rules.json`](../src/biowarden/bench/tasks/family_rules.json), with the "
        "seeds or the route where the rule covers only part of a family. `none`: no family lists "
        "the rule as its own.",
        "**Mode**: what a failing rule does in the default `promoted` mode: `block`, or `warn` "
        "(the message is shown and the call runs). A rule blocks only when calibration admits "
        "it: it fails at least one wrong call, fails none of the correct calls tested "
        "(those that carry the declaration and, for a rule that does not require one, those "
        f"without it), and the upper bound below is at most {UPPER_LIMIT:.0%} over at least "
        f"{MIN_CALLS} such calls. A T2 rule, or a rule whose spec changed after it was measured, "
        "only warns.",
        "**Seeds**: the seeds of the generated instances the rule was measured on.",
        "**False blocks**: correct calls that carry the rule's declaration and that the rule "
        "failed, out of the correct declared calls tested (for a rule without a declaration, all "
        "its correct calls). A call tested with several values of the declaration counts once "
        "per value.",
        "**Upper bound**: the upper end of the exact (Clopper-Pearson) two-sided 95% confidence "
        "interval of the false-block rate, which treats the tested calls as independent.",
        "**Recall**: the wrong calls the rule failed, out of all wrong calls tested, with and "
        "without the declaration (a rule without a declaration counts each call once). Some of "
        "these calls are wrong for a reason the rule is not meant to catch, so recall is not the "
        "share of the rule's own faults that it catches.",
    ]
    return [line for item in items for line in paragraph(item, bullet=True)]


def render() -> str:
    rows = collect()
    by_domain: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_domain[row.domain].append(row)
    kinds = Counter(row.rule.kind for row in rows)
    tiers = Counter(row.rule.tier for row in rows)
    declared = Counter(declaration(row.rule) for row in rows)
    modes = Counter(row.mode.split(" ")[0] for row in rows)
    entries = [row.entry for row in rows if row.entry is not None]
    lines = [
        "# Rule catalogue",
        "",
        f"<!-- Generated by {SCRIPT} from the rule specs; do not edit by hand. -->",
        "",
        *paragraph(
            f"One row per rule of the checking layer ({len(rows)} rules in "
            f"{len({row.spec for row in rows})} spec files), with the calibration that decides "
            f"whether a rule may block a call. Generated by `{SCRIPT}` from the specs and the "
            f"promotion table in [`src/biowarden/engine/specs/`]({SPECS_LINK}/) and from "
            "[`family_rules.json`](../src/biowarden/bench/tasks/family_rules.json); do not "
            f"edit by hand. `python {SCRIPT} --check` fails when this page is out of date."
        ),
        "",
        *paragraph(
            "What a rule is, the tiers, and how verdicts become allow, warn or block: "
            "[architecture](architecture.md). The benchmark tasks the rules are written for: "
            "[task families](families.md)."
        ),
        "",
        *paragraph(
            f"{len(rows)} rules: {kinds['call']} call rules and {kinds['fact']} fact rules; "
            f"{tiers['T0']} in tier T0, {tiers['T1']} in T1, {tiers['T2']} in T2. A declaration "
            f"is required by {declared['required']} rules, optional for {declared['optional']} "
            f"and absent for {declared['none']}."
        ),
        "",
    ]
    if entries:
        failed = [e["right_declared"][0] for e in entries]
        calls = [e["right_declared"][1] for e in entries]
        uppers = [e["false_block_upper"] for e in entries if e["false_block_upper"] is not None]
        lines += [
            *paragraph(
                f"Calibration: in the default `promoted` mode {modes['block']} rules block and "
                f"{modes['warn']} only warn. Per rule, {span(calls)} correct "
                f"declared calls were tested and {span(failed)} of them failed; the upper bound of "
                f"the false-block rate is at most {percent(max(uppers))}. **The instances are "
                "generated by the benchmark's own code, so these false-block rates are "
                "optimistic.**"
            ),
            "",
        ]
    lines += [
        "## Columns",
        "",
        *paragraph(
            "Calibration builds benchmark workspaces from seeds, makes calls on them and labels "
            "each call by running the real tool: a correct call has a result that is right for "
            "the workspace, a wrong call one that is not. The engine judges the calls both with "
            "the rule's declaration and without it."
        ),
        "",
        *legend(rows),
        "",
    ]
    for title, _ in DOMAINS:
        chosen = by_domain.get(title, [])
        if not chosen:
            continue
        specs = sorted({row.spec for row in chosen})
        text = (
            f"{len(chosen)} {'rule' if len(chosen) == 1 else 'rules'} from the spec "
            f"{'file' if len(specs) == 1 else 'files'} {code_list(specs)}."
        )
        if title == DATA_FORMAT_DOMAIN:
            text += (
                " These are fact rules about data formats rather than about one tool; they are "
                "filed here because the families they are written for are in the Original "
                "families section of [task families](families.md)."
            )
        lines += [f"## {title}", "", *paragraph(text), "", *table(chosen), ""]
    marked = [row.rule.id for row in rows if "(" in row.mode]
    if marked:
        lines += [
            *paragraph(
                f"Rules marked in the Mode column ({', '.join(marked)}) have no promotion entry "
                "for their current spec: they only warn until "
                "`python -I -m biowarden.bench.admission` is run again."
            ),
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


# -- the command line ------------------------------------------------------------------------------


def check_page(text: str, page: Path, script: str) -> int:
    """0 when `page` is `text`; otherwise 1, with the first differences on stderr."""
    name = page.relative_to(ROOT).as_posix() if page.is_relative_to(ROOT) else page.name
    who = Path(script).stem
    current = page.read_text(encoding="utf-8") if page.is_file() else None
    if current == text:
        return 0
    if current is None:
        print(f"{who}: {name} does not exist", file=sys.stderr)
    else:
        diff = difflib.unified_diff(
            current.splitlines(),
            text.splitlines(),
            f"{name} (committed)",
            "generated",
            n=0,
            lineterm="",
        )
        for line in list(diff)[:24]:
            print(line[:240], file=sys.stderr)
    print(
        f"{who}: {name} is out of date; regenerate it with: python {script} > {name}",
        file=sys.stderr,
    )
    return 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_rules_doc.py",
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if docs/rules.md differs"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        text = render()
    except GenerationError as exc:
        print(f"make_rules_doc: {exc}", file=sys.stderr)
        return 2
    if args.check:
        return check_page(text, PAGE, SCRIPT)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
