"""bcftools rules of the variant pack (view, filter, query, isec, merge).

Rules: bcftools.format_filter_scope (F54), bcftools.filter_pass_dot (F55),
bcftools.missing_value_filter (F56), bcftools.subset_monomorphic (F57),
bcftools.isec_representation (F62), bcftools.merge_absent_sites (F63). The evidence of the rules
after the first one is quoted in the comment block above each of them.

bcftools.format_filter_scope (T1, call, declaration optional `format_filter_scope`)
    A FORMAT field in a -i/-e expression is one test per record, true when ANY sample satisfies
    it. `-i 'FMT/GQ>=20'` keeps a site if at least one sample has GQ >= 20; `-e 'FMT/GQ<20'`
    drops a site if at least one sample has GQ < 20, so it keeps the sites where EVERY sample has
    GQ >= 20. The two are not complements on a multi-sample file, and neither prints a warning.

Verified here (bcftools 1.24), four samples, GQ 50/50/50/50 (site a), 50/50/-/50 (b), 50/50/15/50
(c), 50/50/50/50 (d); `view -H`, IDs kept:
    -i 'FMT/GQ>=20'            a b c d      (c kept: three samples have GQ >= 20)
    -e 'FMT/GQ<20'             a b d        (c dropped: one sample has GQ 15)
    -i 'MIN(FMT/GQ)>=20'       a b d        (same as -e; the missing GQ of b is ignored)
    -i 'F_PASS(FMT/GQ>=20)==1' a d          (a missing value does not pass)
    -i 'GQ>=20', -i 'FORMAT/GQ>=20': as FMT/GQ; -i 'N_PASS(FMT/GQ>=20)==N_SAMPLES' a d
    -i 'FMT/DP>5 && FMT/GQ>20' tests each term for any sample separately; '&' requires the same
    sample (`-i 'FMT/DP<5 & FMT/GQ>40'` found the one site where one sample has both).
    `bcftools filter -S . -e 'FMT/GQ<20'` keeps every site and sets the failing genotypes to ./.
    `bcftools view -s s3 -i 'FMT/GQ>=20'` evaluates the test on ALL samples and subsets afterwards
    (site c, where s3 has GQ 15, is kept because the others have 50); `bcftools query -s s3 -i`
    subsets first and drops c.

What the rule does. It applies to `bcftools view|filter|query` calls with -i/--include or
-e/--exclude. It is silent unless the expression tests a FORMAT field without an aggregate
(MIN, MAX, AVG, SUM, N_PASS, F_PASS, COUNT, ...) or an index ([0]) and the input VCF (read from the
file named in the call; stdin and BCF are not read) has two or more samples after any -s/-S subset.
It then reads the records and evaluates each FORMAT term (numeric comparisons and GT classes
ref/alt/het/hom/mis or a literal genotype) per sample. It acts only if at least one record is MIXED
for a term: some samples satisfy it and others do not. There the two readings of "all samples" and
"any sample" differ, and the call must declare which one it means:
    any_sample    a site is kept (-i) when at least one sample satisfies the condition
    all_samples   a site is kept only when every sample does
`-i E` means any_sample; `-e E` means all_samples (kept when no sample satisfies E, that is when all
satisfy not-E). A declaration must agree with that. Without one, `-i E` on mixed data fails (the
mistake the rule exists for) while `-e E` passes, because "drop the sites where a sample fails" is
the usual reading of -e; if no record is mixed every reading gives the same sites and the rule is
silent. An expression with a FORMAT
reference the rule cannot evaluate (a vector such as AD, a non-numeric literal) gives `unknown`.
Not seen: the VCF produced by an earlier pipeline stage, expressions inside a bcftools plugin.
"""

from __future__ import annotations

import operator
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from science_harness.engine.argv import has_option, option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.var_common import (
    Rec,
    Vcf,
    alleles,
    existing_vcfs,
    fail,
    read_list,
    read_vcf,
    subset_verdict,
    unknown,
)

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
}
NEGATED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "=": "!=", "==": "!=", "!=": "=="}
AGGREGATES = re.compile(
    r"\b(?:MIN|MAX|AVG|MEAN|SUM|MEDIAN|STDEV|SD|N_PASS|F_PASS|COUNT|N_MISSING|F_MISSING|SMPL_\w+"
    r"|BINOM|ABS|STRLEN|PHRED)\s*\(",
    re.IGNORECASE,
)
TERM = re.compile(
    r"(?P<field>(?:(?:FMT|FORMAT|INFO)/)?[A-Za-z_][\w.]*)\s*(?P<op><=|>=|==|!=|=|<|>)\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
FORMAT_REF = re.compile(r"\b(?:FMT|FORMAT)/\w+")
MIN_SAMPLES = 2
MAX_MIXED_SHOWN = 3


@dataclass(frozen=True)
class Term:
    tag: str
    op: str
    value: str  # numeric text or a quoted/bare string
    raw: str


def selected_indices(call: Call, ctx: Ctx, vcf: Vcf) -> list[int]:
    """Indices of the samples left by -s/--samples/-S/--samples-file (all when not given)."""
    everyone = list(range(len(vcf.samples)))
    names: list[str] = []
    for value in option_values(call.argv, "-s", "--samples"):
        names += value.split(",")
    for value in option_values(call.argv, "-S", "--samples-file"):
        names += read_list(ctx, value) or []
    if not names:
        return everyone
    exclude = any(n.startswith("^") for n in names)
    chosen = {n.lstrip("^") for n in names}
    return [i for i in everyone if (vcf.samples[i] in chosen) != exclude]


def subset_indices(call: Call, ctx: Ctx, vcf: Vcf) -> list[int]:
    """Indices of the samples a -i/-e expression is evaluated on. `bcftools view` evaluates the
    expression on ALL samples of the record and subsets afterwards (-s/-S do not narrow a FORMAT
    test); `bcftools query -s/-S` subsets first."""
    if call.subcommand != "query":
        return list(range(len(vcf.samples)))
    return selected_indices(call, ctx, vcf)


def format_terms(expression: str, vcf: Vcf) -> tuple[list[Term], bool]:
    """The comparisons on FORMAT fields of an expression, and whether a FORMAT reference was left
    that could not be read as such a comparison."""
    bare = (vcf.format_ids - vcf.info_ids) | {"GT"}
    terms: list[Term] = []
    rest = expression
    for hit in TERM.finditer(expression):
        field = hit.group("field")
        prefix, _, name = field.rpartition("/")
        if prefix in ("FMT", "FORMAT") or (not prefix and name in bare):
            terms.append(Term(name, hit.group("op"), hit.group("value"), hit.group(0)))
            rest = rest.replace(hit.group(0), " ", 1)
    leftover = bool(FORMAT_REF.search(rest)) or any(
        re.search(rf"(?<![\w/]){re.escape(tag)}(?![\w(])", rest) for tag in bare
    )
    return terms, leftover


def gt_class(genotype: list[int | None], word: str) -> bool | None:
    known = [a for a in genotype if a is not None]
    if word == "mis":
        return len(known) < len(genotype)
    if len(known) < len(genotype):
        return False
    if word == "ref":
        return all(a == 0 for a in known)
    if word == "alt":
        return any(a > 0 for a in known)
    if word == "hom":
        return len(set(known)) == 1
    if word == "het":
        return len(set(known)) > 1
    return None


def sample_truth(rec: Rec, sample: int, term: Term) -> bool | None:
    """Whether one sample satisfies a FORMAT comparison (None: cannot be evaluated here)."""
    raw = rec.value(sample, term.tag)
    if term.tag == "GT":
        word = term.value.strip("'\"")
        if word in ("ref", "alt", "hom", "het", "mis"):
            hit = gt_class(rec.genotype(sample), word)
            return None if hit is None else (hit if term.op in ("=", "==") else not hit)
        same = raw is not None and alleles(raw) == alleles(word)
        return same if term.op in ("=", "==") else not same
    if term.value[0] in "'\"":
        return None
    if raw is None or raw in (".", ""):
        return False
    if "," in raw:
        return None
    try:
        return OPS[term.op](float(raw), float(term.value))
    except ValueError:
        return None


def mixed_sites(
    vcf: Vcf, terms: list[Term], which: list[int]
) -> tuple[list[tuple[Rec, Term]], bool]:
    """Records where a term holds for some samples and not for others; True if a term could not
    be evaluated on some record."""
    found: list[tuple[Rec, Term]] = []
    blind = False
    for rec in vcf.records:
        for term in terms:
            if term.tag not in rec.fmt:
                continue
            truth = [sample_truth(rec, i, term) for i in which]
            if any(t is None for t in truth):
                blind = True
                continue
            if 0 < sum(bool(t) for t in truth) < len(truth):
                found.append((rec, term))
    return found, blind


def plain_fix(call: Call, term: Term, scope: str) -> tuple[str, ...]:
    """Fix lines for the declared scope the call does not deliver."""
    tag = f"FMT/{term.tag}"
    if scope == "all_samples":  # the call uses -i: every sample must satisfy the term
        flip = NEGATED[term.op]
        agg = "MIN" if term.op in (">", ">=") else "MAX" if term.op in ("<", "<=") else None
        lines = [f"drop the sites where any sample fails: -e '{tag}{flip}{term.value}'"]
        if agg:
            lines.append(f"or require it of every sample: -i '{agg}({tag}){term.op}{term.value}'")
        else:
            lines.append(f"or: -i 'F_PASS({tag}{term.op}{term.value})==1'")
        return tuple(lines)
    flip = NEGATED[term.op]  # the call uses -e: it keeps the sites where NO sample matches
    return (
        f"keep the sites where at least one sample satisfies it: -i '{tag}{flip}{term.value}'",
        "or say all_samples with --declare format_filter_scope=all_samples if that is the intent",
    )


@checker("bcftools_format_filter_scope")
def bcftools_format_filter_scope(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    includes = option_values(call.argv, "-i", "--include")
    excludes = option_values(call.argv, "-e", "--exclude")
    if call.subcommand == "filter" and option_values(call.argv, "-S", "--set-GTs"):
        return passed()  # -S evaluates the expression per sample
    if not (includes or excludes):
        return passed()
    if includes and excludes:
        return unknown("both_include_and_exclude", "both -i and -e are given; scope not checked")
    expression = (includes or excludes)[-1]
    if AGGREGATES.search(expression) or "[" in expression:
        return passed()
    files = existing_vcfs(ctx, call.argv)
    if len(files) != 1:
        return passed()
    vcf = read_vcf(ctx, files[0])
    if vcf is None:
        return passed()
    which = subset_indices(call, ctx, vcf)
    if len(which) < MIN_SAMPLES:
        return passed()
    terms, leftover = format_terms(expression, vcf)
    if not terms and not leftover:
        return passed()
    mixed, blind = mixed_sites(vcf, terms, which)
    if not mixed:
        if leftover or blind:
            return unknown(
                "scope_unverified",
                f"the expression '{expression}' tests a FORMAT field that could not be evaluated "
                f"here; bcftools reads a FORMAT test as true when ANY of the {len(which)} "
                "samples satisfies it.",
                fix=("use MIN()/MAX()/F_PASS() to state the scope per site",),
            )
        return passed()
    implied = "any_sample" if includes else "all_samples"
    declared = call.declared.get(key)
    term = mixed[0][1]
    shown = ", ".join(f"{r.chrom}:{r.pos}" for r, _ in mixed[:MAX_MIXED_SHOWN])
    facts = (
        f"in {files[0].name} {len({id(r) for r, _ in mixed})} of {len(vcf.records)} records have "
        f"samples that satisfy {term.raw} and samples that do not (for example {shown})"
    )
    reading = (
        f"-i keeps a site when ANY of the {len(which)} samples satisfies the FORMAT test"
        if includes
        else f"-e drops a site when ANY of the {len(which)} samples satisfies the FORMAT "
        "test, so it keeps only the sites where no sample does"
    )
    if declared is None:
        if excludes:
            return passed()  # -e E is the idiom for "drop the sites where any sample fails"
        return fail(
            "scope_undeclared",
            f"{reading}; {facts}, so 'any sample' and 'all samples' select different sites.",
            fix=(
                f"re-run with --declare {key}=any_sample if keeping sites where one sample "
                "satisfies the condition is what you want",
                *plain_fix(call, term, "all_samples"),
            ),
            expected=["any_sample", "all_samples"],
        )
    if declared == implied:
        return passed()
    return fail(
        "declared_scope_mismatch",
        f"you declared {key}={declared} but {reading}; {facts}.",
        fix=plain_fix(call, term, declared),
        expected=declared,
        got=implied,
    )


# ---- bcftools.filter_pass_dot --------------------------------------------------------------------
#
# bcftools.filter_pass_dot (T1, call, declaration optional `unfiltered_records`)
#     A record whose FILTER column is "." has not been through any filter. `bcftools view -f PASS`
#     (and `isec`, `merge` with -f PASS) drops it, as do `-i 'FILTER="PASS"'` and
#     `-e 'FILTER!="PASS"'`; `-f .,PASS` keeps it. The records of a caller that applies no filters
#     (or of a merge of two callsets) are all "." and silently vanish. No message.
#
# Verified here (bcftools 1.24), records a PASS, b LowQual, c PASS, d PASS, e ".", f PASS:
#     view -f PASS                      a c d f        view -f .,PASS            a c d e f
#     view -i 'FILTER="PASS"'           a c d f        view -e 'FILTER!="PASS"'  a c d f
#     view -i 'FILTER="PASS" || FILTER="."'  a c d e f
# What the rule does. For view/filter/query/isec/merge it finds the filter selection of the call:
# -f/--apply-filters lists (view, isec, merge) and -i/-e expressions that test FILTER against
# "PASS". A selection that names PASS and not "." drops the "." records (implied `drop`); one that
# names "." keeps them (`keep`). It then reads the VCF named in the call; if no record has FILTER
# "." the call is silent. Otherwise, undeclared, a call that drops them fails; declared `keep` with
# a call that drops them fails; declared `drop` with a call that keeps them fails.
# Not seen: -f on a stream, FILTER tests combined with other fields in ways other than ||.

FILTER_KEY = "unfiltered_records"
APPLY_SUBCOMMANDS = ("view", "isec", "merge")
PASS_TEST = re.compile(r"FILTER\s*(==?|!=)\s*[\"']PASS[\"']")
DOT_TEST = re.compile(r"FILTER\s*(==?|!=)\s*[\"']\.[\"']")


def dot_selection(call: Call) -> str | None:
    """`drop`, `keep` or None (no PASS selection) for the "." records, from the call's options."""
    lists = option_values(call.argv, "--apply-filters")
    if call.subcommand in APPLY_SUBCOMMANDS:
        lists += option_values(call.argv, "-f")
    for value in lists:
        names = {n.strip() for n in value.split(",")}
        if "PASS" in names and "." not in names:
            return "drop"
        if "." in names:
            return "keep"
    includes = option_values(call.argv, "-i", "--include")
    excludes = option_values(call.argv, "-e", "--exclude")
    for expression in (*includes, *excludes):
        if DOT_TEST.search(expression):
            return "keep"
    for expression in includes:
        hit = PASS_TEST.search(expression)
        if hit and hit.group(1) in ("=", "=="):
            return "drop"
    for expression in excludes:
        hit = PASS_TEST.search(expression)
        if hit and hit.group(1) == "!=":
            return "drop"
    return None


def dot_records(vcf: Vcf) -> list[Rec]:
    return [r for r in vcf.records if r.filt == "."]


def filter_dot_fix(call: Call) -> tuple[str, ...]:
    if option_values(call.argv, "-i", "--include"):
        return ('use -i \'FILTER="PASS" || FILTER="."\'',)
    if option_values(call.argv, "-e", "--exclude"):
        return ('use -e \'FILTER!="PASS" && FILTER!="."\'',)
    return ("use -f .,PASS (the records with FILTER '.' are kept)",)


def dot_facts(vcf: Vcf, name: str) -> str:
    dots = dot_records(vcf)
    first = dots[0]
    return (
        f"{len(dots)} of {len(vcf.records)} records of {name} have FILTER '.' "
        f"(for example {first.id or first.chrom + ':' + str(first.pos)})"
    )


@checker("bcftools_filter_pass_dot")
def bcftools_filter_pass_dot(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    selection = dot_selection(call)
    if selection is None:
        return passed()
    files = existing_vcfs(ctx, call.argv)
    if len(files) != 1:
        return passed()
    vcf = read_vcf(ctx, files[0])
    if vcf is None or not dot_records(vcf):
        return passed()
    declared = call.declared.get(rule.declare.key)
    facts = dot_facts(vcf, files[0].name)
    if selection == "drop" and declared is None:
        return fail(
            "dot_filter_dropped",
            f"this call keeps only FILTER=PASS records and drops the records whose FILTER is '.' "
            f"(no filter was applied to them): {facts}.",
            fix=(
                *filter_dot_fix(call),
                "or --declare unfiltered_records=drop if dropping them is intended",
            ),
            expected=["keep", "drop"],
        )
    if declared is None or declared == selection:
        return passed()
    if declared == "keep":
        return fail(
            "declared_keep_but_dropped",
            f"you declared {rule.declare.key}=keep but this call drops the records whose FILTER "
            f"is '.': {facts}.",
            fix=filter_dot_fix(call),
            expected="keep",
            got="drop",
        )
    return fail(
        "declared_drop_but_kept",
        f"you declared {rule.declare.key}=drop but this call keeps the records whose FILTER is "
        f"'.': {facts}.",
        fix=("use -f PASS (or -i 'FILTER=\"PASS\"') to drop the records with FILTER '.'",),
        expected="drop",
        got="keep",
    )


# ---- bcftools.missing_value_filter ---------------------------------------------------------------
#
# bcftools.missing_value_filter (T1, call, declaration optional `missing_annotation`)
#     A comparison on a missing value (QUAL "." or an INFO tag the record lacks) is false for every
#     operator. So `-i 'INFO/QD>=2'` removes the records without QD, while `-e 'INFO/QD<2'` keeps
#     them: the pair looks like a negation of each other and is not. No message either way.
#
# Verified here (bcftools 1.24), records a b c f with QUAL 50, d with QUAL 10, e with QUAL ".":
#     -i 'QUAL>=30'  a b c f           -e 'QUAL<30'  a b c e f          (e = QUAL ".")
#     -i 'QUAL>30 || QUAL="."'  a b c e f
# and on the F56 data (INFO/QD present at most records, missing at the others): -i 'INFO/QD>=2'
# removes the records without QD, -e 'INFO/QD<2' keeps them (tests/test_rules_var_bcftools.py).
# What the rule does. For view/filter/query it reads a call whose only expression is one
# comparison of QUAL or of an INFO tag with a number. -i E means the records with a missing value
# are removed (`record_removed`), -e E that they stay (`record_kept`). It reads the VCF of the call
# and is silent unless some records have the value missing and others have it. Undeclared, only
# the -e form fails (the surprising one); declared, either form must agree with the declaration.
# Not seen: expressions with several terms, FORMAT fields (missing per sample), stdin.

MISSING_KEY = "missing_annotation"
SINGLE_TERM = re.compile(
    r"^\s*(?P<field>QUAL|(?:INFO/)?[A-Za-z_]\w*)\s*(?P<op><=|>=|==|!=|=|<|>)\s*"
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def missing_records(vcf: Vcf, field: str) -> list[Rec] | None:
    """Records where QUAL or the INFO tag is missing (None: not a field this rule reads)."""
    if field == "QUAL":
        missing = [r for r in vcf.records if r.qual == "."]
    else:
        tag = field.removeprefix("INFO/")
        if tag not in vcf.info_ids:
            return None
        missing = [r for r in vcf.records if r.info.get(tag, ".") in (".", "")]
    return missing if len(missing) < len(vcf.records) else None  # missing everywhere: other issue


@checker("bcftools_missing_value_filter")
def bcftools_missing_value_filter(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    includes = option_values(call.argv, "-i", "--include")
    excludes = option_values(call.argv, "-e", "--exclude")
    if len(includes) + len(excludes) != 1:
        return passed()
    expression = (includes or excludes)[0]
    term = SINGLE_TERM.match(expression)
    if term is None:
        return passed()
    files = existing_vcfs(ctx, call.argv)
    if len(files) != 1:
        return passed()
    vcf = read_vcf(ctx, files[0])
    if vcf is None:
        return passed()
    field, op, value = term.group("field"), term.group("op"), term.group("value")
    missing = missing_records(vcf, field)
    if not missing:
        return passed()
    implied = "record_removed" if includes else "record_kept"
    declared = call.declared.get(rule.declare.key)
    name = field if field == "QUAL" else field.removeprefix("INFO/")
    first = missing[0]
    facts = (
        f"{len(missing)} of {len(vcf.records)} records of {files[0].name} have no {name} value "
        f"(for example {first.id or first.chrom + ':' + str(first.pos)})"
    )
    flipped = NEGATED[op]
    if includes:
        reading = f"-i '{expression}' is false for a missing value, so those records are removed"
        fix = (f"use -e '{field}{flipped}{value}' to keep the records without {name}",)
    else:
        reading = f"-e '{expression}' is false for a missing value, so those records are kept"
        fix = (
            f"use -i '{field}{flipped}{value}' to keep only records that have {name} "
            f"and meet the cutoff",
        )
    if declared is None:
        if includes:
            return passed()
        return fail(
            "missing_values_kept",
            f"{reading}: {facts}.",
            fix=(*fix, f"or --declare {rule.declare.key}=record_kept if keeping them is intended"),
            expected=["record_kept", "record_removed"],
        )
    if declared == implied:
        return passed()
    return fail(
        "declared_missing_mismatch",
        f"you declared {rule.declare.key}={declared} but {reading}: {facts}.",
        fix=fix,
        expected=declared,
        got=implied,
    )


# ---- bcftools.subset_monomorphic -----------------------------------------------------------------
#
# bcftools.subset_monomorphic (T1, call, declaration optional `subset_sites`)
#     `bcftools view -s/-S` writes every site of the input for the selected samples. It recomputes
#     INFO/AC and AN (AC=0 at a site nobody carries) but leaves the other INFO fields such as AF
#     alone, and keeps the sites. `-c 1` (--min-ac) drops them; a test such as -e 'AC==0' does
#     not (see below). A sample list with a missing name stops ("subset called for sample that
#     does not exist"); --force-samples only warns.
#
# Verified here (bcftools 1.24), sites a b c with INFO AC/AN/AF and four samples, subset s1,s2:
#     view -s s1,s2          3 sites, b: AC=0;AN=4;AF=0.125 (stale AF)
#     view -I -s s1,s2       3 sites, INFO untouched      view -s s1,s2 -c1   2 sites (b gone)
#     view -s s1,s2 -i 'AF>0.1'  keeps b: the filter reads the stale AF of the full cohort
#     view -s s1,s2 -e 'AC==0'  and  -i 'AC>0'  and  -i 'COUNT(GT="alt")>0'  keep b too: -i/-e
#     are evaluated on the full cohort BEFORE the samples are selected (only -c/--min-ac, which
#     uses the recomputed AC, drops b). On a file without INFO/AC the AC tests do nothing at all.
# What the rule does. A `bcftools view` call with -s/--samples/-S/--samples-file and no
# -c/--min-ac (>= 1); an -i/-e test on AC or GT does not count (see above). It reads the VCF,
# selects the samples as bcftools would (^ excludes) and counts the sites without an alternate
# allele among them. If there are none it is silent.
# Undeclared it fails; `keep_all_sites` and `filtered_downstream` (a later pipe stage drops them)
# pass; `variant_sites_only` fails because the call keeps them.
# Not seen: a subset applied by an earlier pipeline stage, BCF input, stdin.

SUBSET_KEY = "subset_sites"


def view_min_ac(call: Call) -> bool:
    """Only -c/--min-ac counts alleles after the subset; -i/-e tests (AC, COUNT(GT="alt")) are
    evaluated on the full cohort before it."""
    for value in option_values(call.argv, "-c", "--min-ac"):
        head = value.split(":")[0]
        if head.isdigit() and int(head) >= 1:
            return True
    return False


@checker("bcftools_subset_monomorphic")
def bcftools_subset_monomorphic(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.subcommand != "view" or not option_values(
        call.argv, "-s", "--samples", "-S", "--samples-file"
    ):
        return passed()
    files = existing_vcfs(ctx, call.argv)
    if len(files) != 1:
        return passed()
    vcf = read_vcf(ctx, files[0])
    if vcf is None:
        return passed()
    which = selected_indices(call, ctx, vcf)
    if not which:
        return passed()
    return subset_verdict(
        call,
        rule.declare.key,
        "bcftools view",
        files[0].name,
        vcf,
        which,
        view_min_ac(call),
        fix=("add -c 1 to drop the sites without an alternate allele in the selected samples",),
    )


# ---- bcftools.isec_representation ----------------------------------------------------------------
#
# bcftools.isec_representation (T1, call, declaration optional `isec_matching`)
#     `bcftools isec` matches records by POS and alleles (-c none, the default) or by POS only with
#     some allele test (-c some|indels|all); it never moves an indel. The same deletion in a
#     homopolymer run written at the left end by one caller and at the right end by another is two
#     different records for isec, merge and every join on CHROM-POS-REF-ALT. `bcftools norm -f
#     ref.fa` left-aligns both files first. isec needs bgzip-compressed, indexed input (a plain
#     .vcf stops: "not compressed with bgzip").
#
# Verified here (bcftools 1.24), reference ACGTACGTTTTTTGACGT..., a one-base deletion in the T run
# (8-13) written `GT>G` at POS 7 by caller A and `TT>T` at POS 12 by caller B, plus one SNV both
# write the same way: `isec -n=2 -w1` with -c none, some, indels or all finds the SNV only; after
# `norm -f ref.fa` on both files (stderr "Lines total/split/joined/realigned/mismatch_removed/
# dup_removed/skipped: 2/0/0/1/0/0/0" for B) it finds both. A record whose REF disagrees with the
# FASTA is reported ("Reference allele mismatch") and, with -c e, stops.
# What the rule does. For `isec` calls with two or more VCFs it looks for one FASTA (.fa, .fasta,
# .fna) in the working directory, runs `bcftools norm -f FASTA -c w -o /dev/null FILE` read-only on
# every input (through links in a scratch directory: norm would write FASTA.fai beside the FASTA)
# and reads the "realigned" count. Nothing realigned: silent. Realigned records: fails
# undeclared; `as_written` passes (the author compares the records as they are written);
# `normalized` fails (the inputs are not normalised). No FASTA or a failing dry run: `unknown`.
# Not seen: inputs piped in, more than one FASTA in the directory, BCF inputs (dry-run only).

ISEC_KEY = "isec_matching"
NORM_SUMMARY = re.compile(r"Lines\s+total/split/joined/realigned[^\n]*:\s*(\d+)/(\d+)/(\d+)/(\d+)")


def find_fasta(ctx: Ctx) -> Path | None:
    found = [
        p
        for pattern in ("*.fa", "*.fasta", "*.fna")
        for p in sorted(ctx.cwd.glob(pattern))
        if p.is_file()
    ]
    return found[0] if len(found) == 1 else None


def realigned_records(ctx: Ctx, fasta: Path, vcf: Path) -> tuple[int, int] | None:
    """(records norm would realign, records) from a read-only `bcftools norm -f FASTA` run.

    norm writes `FASTA.fai` next to a FASTA that has no index, which would leave a new file in the
    agent's workspace (E-8). It runs on links in a scratch directory instead: it reads the FASTA
    (and its index, when there is one) through them and writes the index it needs beside the links.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="tbm-norm-") as scratch:
            link = Path(scratch) / fasta.name
            for suffix in ("", ".fai"):
                if Path(f"{fasta}{suffix}").is_file():
                    Path(f"{link}{suffix}").symlink_to(f"{fasta}{suffix}")
            proc = ctx.run_real(
                "bcftools", ["norm", "-f", str(link), "-c", "w", "-o", "/dev/null", str(vcf)]
            )
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):  # no evidence either way
        return None
    hit = NORM_SUMMARY.search(proc.stderr)
    if proc.returncode != 0 or hit is None:
        return None
    return int(hit.group(4)), int(hit.group(1))


@checker("bcftools_isec_representation")
def bcftools_isec_representation(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.subcommand != "isec":
        return passed()
    files = existing_vcfs(ctx, call.argv)
    if len(files) < 2:
        return passed()
    declared = call.declared.get(rule.declare.key)
    if declared == "as_written":
        return passed()
    fasta = find_fasta(ctx)
    counts = None if fasta is None else {f.name: realigned_records(ctx, fasta, f) for f in files}
    if counts is None or any(v is None for v in counts.values()):
        if declared is None:
            return unknown(
                "normalisation_not_checked",
                "isec matches records by position and alleles as written, and an indel in a "
                "repeat can be written at another position by another caller; the inputs could "
                "not be checked against a reference here.",
                fix=("run bcftools norm -f ref.fa on every input before isec",),
            )
        return passed()
    moved = {name: c for name, c in counts.items() if c and c[0] > 0}
    if not moved:
        return passed()
    name, (n, total) = next(iter(moved.items()))
    facts = f"bcftools norm -f {fasta.name} would realign {n} of {total} records of {name}"  # type: ignore[union-attr]
    fix = (
        f"normalise every input first: bcftools norm -f {fasta.name} -Oz -o <in>.norm.vcf.gz "  # type: ignore[union-attr]
        "<in>.vcf.gz && bcftools index -t <in>.norm.vcf.gz, then isec the normalised files",
        f"or --declare {rule.declare.key}=as_written to compare the records as they are written",
    )
    if declared == "normalized":
        return fail(
            "declared_normalized_but_not",
            f"you declared {rule.declare.key}=normalized but {facts}: it is not left-aligned.",
            fix=fix[:1],
            expected="normalized",
            got="not normalized",
        )
    return fail(
        "isec_unnormalised_inputs",
        f"isec compares POS and alleles as written, so the same indel written at another position "
        f"in the other file is not shared: {facts}.",
        fix=fix,
        expected=["normalized", "as_written"],
    )


# ---- bcftools.merge_absent_sites -----------------------------------------------------------------
#
# bcftools.merge_absent_sites (T1, call, declaration optional `absent_sites`)
#     `bcftools merge` writes ./. for the samples of an input that does not list a site; `-0`
#     (--missing-to-ref) writes 0/0. For single-sample VCFs that list every site the two agree (no
#     site is absent). For variants-only calls, where a sample's unlisted site is homozygous
#     reference, the default makes every allele frequency (INFO/AF or AC/AN from +fill-tags, or
#     counts of non-missing genotypes) divide by the called alleles only.
#
# Verified here (bcftools 1.24), three single-sample VCFs, a in all, b in smp2 and smp3, c in smp3:
#     merge     b: ./. 1/1 0/1, c: ./. ./. 0/1;  +fill-tags: b F_MISSING 0.333333 AN 4 AF 0.75,
#               c F_MISSING 0.666667 AN 2 AF 0.5
#     merge -0  b: 0/0 1/1 0/1, c: 0/0 0/0 0/1;  AN 6 and AF 0.5 for b, 0.166667 for c
# What the rule does. For `merge` calls with two or more single-sample VCFs (positional or listed
# with -l/--file-list), it compares the site sets (CHROM, POS, REF, ALT) of the inputs. If every
# input lists every site, or an input is a gVCF (INFO/END or <NON_REF>/<*> alleles: absent means no
# data there), it is silent. Otherwise the call implies `missing` without -0 and `reference` with
# it. Undeclared, the default form fails (the surprising one); declared, the declaration must agree.
# Not seen: inputs with several samples, BCF, stdin lists.

ABSENT_KEY = "absent_sites"


def merge_inputs(call: Call, ctx: Ctx) -> list[Path]:
    lists = option_values(call.argv, "-l", "--file-list")
    if lists:
        names = read_list(ctx, lists[0]) or []
        return [ctx.resolve(n) for n in names]
    return existing_vcfs(ctx, call.argv)


def site_key(rec: Rec) -> tuple[str, int, str, tuple[str, ...]]:
    return rec.chrom, rec.pos, rec.ref, rec.alts


def is_gvcf(vcf: Vcf) -> bool:
    return any(
        "END" in r.info or any(a in ("<NON_REF>", "<*>") for a in r.alts) for r in vcf.records
    )


@checker("bcftools_merge_absent_sites")
def bcftools_merge_absent_sites(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    if call.subcommand != "merge":
        return passed()
    files = merge_inputs(call, ctx)
    if len(files) < 2:
        return passed()
    vcfs = [read_vcf(ctx, f) for f in files]
    if any(v is None or len(v.samples) != 1 or v.truncated or is_gvcf(v) for v in vcfs):
        return passed()
    sites = [{site_key(r) for r in v.records} for v in vcfs if v is not None]
    union = set().union(*sites)
    absent = [(f, union - s) for f, s in zip(files, sites, strict=True) if union - s]
    if not absent:
        return passed()
    implied = "reference" if has_option(call.argv, "-0", "--missing-to-ref") else "missing"
    declared = call.declared.get(rule.declare.key)
    name, lacking = absent[0]
    example = sorted(lacking)[0]
    pairs = sum(len(a) for _, a in absent)
    facts = (
        f"{len(absent)} of the {len(files)} inputs lack some of the {len(union)} sites "
        f"({pairs} sample-site pairs; {name.name} lacks {example[0]}:{example[1]})"
    )
    if implied == "missing":
        reading = "bcftools merge writes ./. (missing) for a site an input does not list"
        fix = (
            "add -0 (--missing-to-ref) if the files are variants-only calls, where an unlisted "
            "site is homozygous reference",
        )
    else:
        reading = "-0 writes 0/0 (homozygous reference) for a site an input does not list"
        fix = ("drop -0 to leave the unlisted sites missing (./.)",)
    if declared is None:
        if implied == "reference":
            return passed()
        return fail(
            "absent_sites_missing",
            f"{reading}, so an allele frequency or call rate computed on the merged file leaves "
            f"those samples out: {facts}.",
            fix=(*fix, f"or --declare {rule.declare.key}=missing if absence means no data"),
            expected=["missing", "reference"],
        )
    if declared == implied:
        return passed()
    return fail(
        "declared_absent_mismatch",
        f"you declared {rule.declare.key}={declared} but {reading}: {facts}.",
        fix=fix if declared == "reference" else ("drop -0 to leave the unlisted sites missing",),
        expected=declared,
        got=implied,
    )
