"""seqtk rule of the sequence pack: `trimfq` keeps at least 30 bases of every read by default.

seqtk.trimfq_floor (family F81, slot default_param; T1, call, declaration optional)
    `seqtk trimfq` trims low-quality ends with the Phred algorithm (`-q`, an ERROR RATE, default
    0.05) and has a second default, `-l INT  maximally trim down to INT bp [30]`. A read whose good
    part is shorter than 30 bases is cut at 30 bases and keeps a low-quality tail of up to
    30 - g bases (g = the good part). Exit 0, empty stderr. (`-q 20` is a separate trap: the
    threshold is an error rate, so 20 trims nothing; the rule does not look at it.)

    Verified here (seqtk 1.5-r133; 60-base reads with a Q32-40 head of g bases and a Q2-8 tail):
    `trimfq -q 0.01` gave 30 33 40 30 30 30 30 30 57 44 38 30 bases for g = 25 33 40 23 17 18 19 29
    57 44 38 20, `trimfq -q 0.01 -l 1` (and `-l 0`) gave g exactly; `-q 20` returned all 60 bases of
    every read. On the F81 sets (seeds 0-59) the default floor changes the total on every faulted
    twin and on no clean one.

    What the rule does. It applies to `seqtk trimfq` calls that trim by quality (no non-zero
    `-b`, `-e` or `-L`, which switch quality trimming off) and have a floor of 2 or more (the
    default 30, or `-l N`). It reads the FASTQ named last on the command line (up to 5 MB), runs
    the call and the same call with `-l 1`, and fails when some read comes out longer with the
    floor. Declaration `trim_floor`: keep_minimum (reads must keep the floor length: pass) or
    no_minimum (trim every tail whatever is left: fail when the floor binds).
    Not seen: reads on stdin, FASTA input (no qualities: pass).
"""

from __future__ import annotations

from science_harness.engine.argv import option_values
from science_harness.engine.call import Call
from science_harness.engine.checks import CheckResult, Ctx, Match, checker, passed
from science_harness.engine.rules.seq_common import (
    RUN_ERRORS,
    drop_options,
    existing,
    positionals,
    small_file,
)

VALUE_OPTIONS = frozenset({"-q", "-l", "-b", "-e", "-L"})
DEFAULT_FLOOR = 30


def _lengths(stdout: str) -> list[int]:
    return [len(line) for line in stdout.splitlines()[1::4]]


@checker("seqtk_trimfq_floor")
def seqtk_trimfq_floor(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = list(call.argv)
    if "trimfq" not in argv:
        return passed()
    args = argv[argv.index("trimfq") + 1 :]
    for name in ("-b", "-e", "-L"):
        if any(v not in ("0", "") for v in option_values(args, name)):
            return passed()
    floors = option_values(args, "-l")
    try:
        floor = int(floors[-1]) if floors else DEFAULT_FLOOR
    except ValueError:
        return passed()
    declared = call.declared.get(key)
    if floor <= 1 or declared == "keep_minimum":
        return passed()
    files = positionals(args, VALUE_OPTIONS)
    path = existing(ctx, files[-1]) if files and files[-1] != "-" else None
    if path is None or not small_file(path):
        return passed()
    rest = drop_options(args, with_value={"-l"})
    try:
        current = ctx.run_real("seqtk", ["trimfq", *args])
        full = ctx.run_real("seqtk", ["trimfq", "-l", "1", *rest])
    except RUN_ERRORS:  # a failing dry run is not evidence
        return passed()
    if current.returncode != 0 or full.returncode != 0:
        return passed()
    kept, wanted = _lengths(current.stdout), _lengths(full.stdout)
    if len(kept) != len(wanted):
        return passed()
    names = [line[1:].split()[0] for line in full.stdout.splitlines()[0::4]]
    held = [(n, w, k) for n, w, k in zip(names, wanted, kept, strict=True) if k > w]
    if not held:
        return passed()
    name, good, shown = held[0]
    used = f"-l {floor}" if floors else f"the default -l {DEFAULT_FLOOR}"
    return CheckResult(
        "fail",
        "trim_floor_binds",
        f"seqtk trimfq with {used} never cuts a read below {floor} bases: {len(held)} of "
        f"{len(wanted)} reads in {path.name} keep low-quality bases to reach it (for example "
        f"{name}: {good} good bases, {shown} kept).",
        expected=1,
        got=floor,
        fix=(
            "add -l 1 (trim every low-quality tail whatever is left)",
            f"or, if every read must keep at least {floor} bases: --declare {key}=keep_minimum",
        ),
    )
