"""HMMER rule of the sequence pack: E-values scale with the size of the file searched.

hmmer.split_database (family F88, slot metric_scale; T1, call, declaration optional)
    A HMMER E-value is the P-value times Z, the number of target sequences in the file (`hmmsearch`,
    `phmmer`, `nhmmer`) or of profiles in the file (`hmmscan`); `-Z` overrides Z. When a database
    arrives in pieces and each piece is searched alone, every E-value is divided by the number of
    pieces, and a threshold such as `-E 1e-5` is that many times more permissive than on the whole
    database. Exit 0; the Z that was used is printed in the report, nothing says the E-values are
    not those of the whole database.

    Verified here (HMMER 3.4; a 110-residue family of 10 members at 72% identity, `hmmbuild`;
    160-224 target proteins in eight equal parts): every hit has an E-value 8 times smaller in its
    part (Z = 20-28) than in the whole file (Z = 160-224), within the 7% that the two printed
    digits allow (396 hits, F88 seeds 0-29); `-Z` set to the whole size gives the whole-file
    value on a part; the hits passing `-E 1e-5` on the parts are those with E <= 8e-5 on the whole
    file (F88, seeds 0-29: 3-4 extra proteins on every faulted twin).
    `hmmsearch --cut_ga` on a model without a GA line stops with "Error: GA bit thresholds
    unavailable on model" (loud, not covered).

    What the rule does. It applies to hmmsearch, phmmer, nhmmer and hmmscan. It finds the target
    file (hmmsearch, phmmer, nhmmer: the second argument; hmmscan: the first) and its siblings:
    files of the same directory whose names are equal once digit runs are replaced (proteome_part1
    .fasta, proteome_part2.fasta, ...), at least two. When the call has no `-Z` and uses an E-value
    (`-E`, `--incE`, `--domE`, `--incdomE`, or none of the bit-score thresholds `-T`, `--incT`,
    `--domT`, `--incdomT`, `--cut_ga`, `--cut_nc`, `--cut_tc`), it fails with the total count of the
    siblings (sequences by '>' lines, profiles by 'HMMER3' headers; groups over 3 MB are not
    counted). Declaration `evalue_database`: whole_collection (E-values must be those of the
    complete collection: needs -Z) or this_file (this file is the whole database: pass).
    Not seen: input on stdin, parts in other directories, names whose numbers are not digits.
"""

from __future__ import annotations

import re
from pathlib import Path

from science_harness.engine.argv import option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.seq_common import existing, positionals

VALUE_OPTIONS = frozenset(
    "-o -A --tblout --domtblout --pfamtblout -E -T --domE --domT --incE --incT --incdomE "
    "--incdomT -Z --domZ --seed --tformat --qformat --cpu --textw --F1 --F2 --F3 --popen "
    "--pextend --mx --mxfile --EmL --EmN --EvL --EvN --EfL --EfN --Eft --w_beta --w_length "
    "--block_length --hmmout --alimask -N --formula --informat --hmmin".split()
)
E_OPTIONS = ("-E", "--incE", "--domE", "--incdomE")
SCORE_OPTIONS = ("-T", "--incT", "--domT", "--incdomT", "--cut_ga", "--cut_nc", "--cut_tc")
COUNT_LIMIT = 3_000_000


def _has(argv: tuple[str, ...], names: tuple[str, ...]) -> bool:
    return any(t in names or t.split("=", 1)[0] in names for t in argv)


def siblings(path: Path) -> list[Path]:
    """Files next to `path` whose names differ from it only in digit runs (itself included)."""
    pattern = re.sub(r"\d+", "#", path.name)
    if "#" not in pattern:
        return [path]
    try:
        group = [
            p
            for p in sorted(path.parent.iterdir())
            if p.is_file() and re.sub(r"\d+", "#", p.name) == pattern
        ]
    except OSError:
        return [path]
    return group


def count_units(ctx: Ctx, path: Path) -> int | None:
    """Sequences ('>' lines) or profiles ('HMMER3' header lines) in a plain-text file."""
    try:
        if path.stat().st_size > COUNT_LIMIT:
            return None
        marker = "HMMER3/" if path.suffix == ".hmm" else ">"
        return sum(1 for line in ctx.lines(path) if line.startswith(marker))
    except (OSError, UnicodeError):
        return None


@checker("hmmer_split_database")
def hmmer_split_database(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    declared = call.declared.get(key)
    if declared == "this_file" or option_values(argv, "-Z"):
        return passed()
    if _has(argv, SCORE_OPTIONS) and not _has(argv, E_OPTIONS):
        return passed()
    args = positionals(argv, VALUE_OPTIONS)
    index = 0 if call.tool == "hmmscan" else 1
    if len(args) <= index or args[index] == "-":
        return passed()
    target = existing(ctx, args[index])
    if target is None:
        return passed()
    group = siblings(target)
    if len(group) < 2:
        return passed()
    counts = [count_units(ctx, p) for p in group]
    if any(c is None for c in counts):
        return passed()
    here, total = counts[group.index(target)], sum(counts)  # type: ignore[arg-type]
    if not here or here >= total:
        return passed()
    unit = "profiles" if target.suffix == ".hmm" else "sequences"
    ratio = total / here
    names = f"{group[0].name} ... {group[-1].name}"
    return CheckResult(
        "fail",
        "evalue_from_one_part",
        f"{call.tool} computes E-values from the {here} {unit} of {target.name}, but "
        f"{target.name} is one of {len(group)} sibling files ({names}, {total} {unit} together): "
        f"every E-value is {ratio:.1f} times smaller than in a search of the whole collection, "
        "so an E-value threshold lets through hits that fail it there.",
        expected=total,
        got=here,
        fix=(
            f"add -Z {total} (the number of {unit} in all {len(group)} files)",
            f"or, if {target.name} is the whole database: --declare {key}=this_file",
        ),
    )
