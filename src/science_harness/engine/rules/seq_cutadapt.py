"""cutadapt rule of the sequence pack: one adapter is removed per read unless `-n` says more.

cutadapt.adapters_per_read (family F84, slot default_param; T1, call, declaration optional)
    `cutadapt -g PRIMER -a ADAPTER` looks like "remove the 5' primer and the 3' adapter", but
    `-n/--times` defaults to 1: each read is trimmed once, with the best-scoring of ALL listed
    adapters, so a read that carries both keeps one of them. The report only says "Reads with
    adapters: N". `-n 2`, or a linked adapter (`-g PRIMER...ADAPTER`), removes both.

    Verified here (cutadapt 5.2): 30 amplicon reads (19-base primer, 20-35 base insert, 13-base
    adapter, random tail; `-g GTGCCAGCAGCCGCGGTAA -a AGATCGGAAGAGC`): 0 of 30 reads trimmed to
    the insert; with `-n 2` 30 of 30; with the linked adapter 30 of 30; with `-a ADAPTER` alone or
    `-g PRIMER` alone 0 of 30. Exit 0, empty stderr. On the F84 sets (seeds 0-99) `-n 2` returns
    exactly the inserts, the plain call leaves one adapter on every read of the faulted twin.

    What the rule does. It applies to `cutadapt`. It lists the adapters of a read side (R1: `-a`,
    `-g`, `-b` and the long forms; R2: `-A`, `-G`, `-B`; a linked adapter `A...B` counts as one;
    `file:` adapters are read from the FASTA), and when a side has more adapters than `-n`/`--times`
    it samples 5000 reads of the matching input file and fails when some read contains two of the
    adapter sequences (exact match, anchors and parameters removed, IUPAC adapters skipped).
    Declaration `adapters_per_read`: one (the adapters are alternatives, a read has one of them:
    pass) or all (every listed adapter must go from each read that has it: fail when -n is too
    small). Silent when no read carries two adapters, and for standard input.
"""

from __future__ import annotations

import re

from science_harness.engine.argv import option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.seq_common import (
    existing,
    positionals,
    read_fasta,
    read_fastq,
    small_file,
)

R1 = ("-a", "--adapter", "-g", "--front", "-b", "--anywhere")
R2 = ("-A", "-G", "-B")
VALUE_OPTIONS = frozenset(
    "-a --adapter -g --front -b --anywhere -A -G -B -u --cut -U -n --times -o --output -p "
    "--paired-output -e --error-rate -O --overlap -m --minimum-length -M --maximum-length -q "
    "--quality-cutoff -j --cores -l --length -x --prefix -y --suffix --json --report --format "
    "--rename --action --max-n --max-expected-errors --trim-n".split()
)
PLAIN = re.compile(r"[ACGT]+")
SAMPLE = 5000


def _clean(spec: str) -> str:
    """The adapter sequence of a specification: anchors, parameters and the name removed."""
    spec = spec.split(";", 1)[0]
    if "=" in spec and not spec.startswith("file:"):
        spec = spec.split("=", 1)[1]
    return spec.strip("^$X").upper()


def _specs(ctx: Ctx, argv: tuple[str, ...], names: tuple[str, ...]) -> list[str]:
    """One entry per adapter specification of the side (sequences; a linked adapter once)."""
    out: list[str] = []
    for raw in option_values(argv, *names):
        if raw.startswith("file:"):
            path = existing(ctx, raw[5:])
            records = read_fasta(ctx, path) if path else None
            out += [seq.upper() for _, seq in records or []]
        elif "..." in raw:
            out.append("LINKED:" + raw)
        else:
            out.append(_clean(raw))
    return out


def _times(argv: tuple[str, ...]) -> int:
    values = option_values(argv, "-n", "--times")
    try:
        return max(int(v) for v in values) if values else 1
    except ValueError:
        return 1


def _side_check(ctx: Ctx, specs: list[str], input_name: str | None, times: int):
    """(reads with two adapters, example read) for one side, or None when nothing to report."""
    plain = [s for s in dict.fromkeys(specs) if PLAIN.fullmatch(s) and len(s) >= 8]
    if len(plain) < 2 or len(specs) <= times or input_name is None:
        return None
    path = existing(ctx, input_name)
    if path is None or not small_file(path):
        return None
    reads = read_fastq(ctx, path, limit=SAMPLE)
    if reads is None:
        records = read_fasta(ctx, path, limit=SAMPLE)
        reads = [(n, s, "") for n, s in records] if records else None
    if not reads:
        return None
    both = [(n, [p for p in plain if p in s.upper()]) for n, s, _ in reads]
    both = [(n, found) for n, found in both if len(found) >= 2]
    return (len(both), len(reads), both[0], path.name) if both else None


@checker("cutadapt_adapters_per_read")
def cutadapt_adapters_per_read(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    times = _times(argv)
    declared = call.declared.get(key)
    if declared == "one":
        return passed()
    files = [f for f in positionals(argv, VALUE_OPTIONS) if f != "-"]
    sides = ((_specs(ctx, argv, R1), files[0] if files else None, "R1"),)
    sides += ((_specs(ctx, argv, R2), files[1] if len(files) > 1 else None, "R2"),)
    for specs, input_name, label in sides:
        found = _side_check(ctx, specs, input_name, times)
        if found is None:
            continue
        n_both, n_reads, (example, adapters), file_name = found
        count = len(specs)
        return CheckResult(
            "fail",
            "one_adapter_per_read",
            f"this cutadapt call lists {count} adapters for {label} but removes only "
            f"{times} per read (-n/--times, default 1): {n_both} of the {n_reads} reads sampled "
            f"from {file_name} contain two of them (for example {example}: "
            f"{' and '.join(adapters)}), and one of the two stays on the read.",
            expected="all",
            got=f"-n {times}",
            fix=(
                f"add -n {count} (remove every listed adapter from each read), or link a 5' and "
                "a 3' adapter as -g FRONT...BACK",
                f"or, if a read has only one of the adapters: --declare {key}=one",
            ),
        )
    return passed()
