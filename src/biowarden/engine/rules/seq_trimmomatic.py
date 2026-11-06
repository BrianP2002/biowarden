"""Trimmomatic rule of the sequence pack: steps run in the order given, MINLEN included.

trimmomatic.minlen_order (family F85, slot step_order; T1, call, declaration optional)
    Trimmomatic's manual: "the trimming steps are performed in the order in which they are
    specified on the command line". `MINLEN:36` filters on the length of the read AT THAT POINT.
    Listed before `SLIDINGWINDOW` and `TRAILING` (for example because a protocol sheet lists the
    parameters alphabetically) it checks the untrimmed length, and the reads that the later steps
    cut under 36 bases are kept. Exit 0; the summary says "Surviving: 40 (100.00%)".

    Verified here (Trimmomatic 0.41, `SE -phred33`; 40 reads of 60 bases with a Q32-40 head of
    10-60 bases and a Q2-8 tail): `LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36` kept 19
    reads (shortest 40 bases); `MINLEN:36 SLIDINGWINDOW:4:15` kept all 40 (shortest 10);
    `SLIDINGWINDOW:4:15 MINLEN:36` kept 19; `MINLEN:36 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15`
    kept 40. A missing ILLUMINACLIP file and an unreadable quality encoding stop with a Java
    stack trace or "Unable to detect quality encoding" (loud, not covered).

    What the rule does. It applies to `trimmomatic` (SE and PE; `java -jar trimmomatic.jar` is not
    seen). It reads the step names in argument order and acts only when a MINLEN comes before a
    step that shortens reads (ILLUMINACLIP, SLIDINGWINDOW, LEADING, TRAILING, CROP, HEADCROP,
    MAXINFO). Declaration `minlen_stage`: before_trimming (the untrimmed length is what is
    filtered: pass); after_trimming with such an order fails at once (a contradiction in the
    call). Without a declaration it reruns the call on its input reads (up to 5 MB, outputs in a
    temporary directory) twice, as written and with MINLEN moved to the end, and fails when the
    number of reads kept differs. Silent when the two agree (a MINLEN that cannot matter on this
    file); `unknown` when the input cannot be found or the rerun fails.
    Not seen: `-baseout` runs, input on a pipe.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.seq_common import (
    RUN_ERRORS,
    DryRunFailed,
    drop_options,
    existing,
    memo,
    small_file,
    stamp,
)

STEP = re.compile(
    r"^(ILLUMINACLIP|SLIDINGWINDOW|LEADING|TRAILING|CROP|HEADCROP|MAXINFO|MINLEN|AVGQUAL|TOPHRED33|TOPHRED64):"
)
SHORTENING = ("ILLUMINACLIP", "SLIDINGWINDOW", "LEADING", "TRAILING", "CROP", "HEADCROP", "MAXINFO")


def steps_of(argv: tuple[str, ...]) -> list[str]:
    return [m.group(1) for token in argv if (m := STEP.match(token))]


VALUE_OPTIONS = frozenset({"-threads", "-trimlog", "-summary", "-basein", "-baseout"})


def _layout(argv: tuple[str, ...]) -> tuple[str, list[str], list[str], list[str]] | None:
    """(mode, options, files before the steps, steps) of a trimmomatic call."""
    tokens = list(argv)
    mode = next((t for t in tokens if t in ("SE", "PE")), None)
    if mode is None:
        return None
    rest = tokens[tokens.index(mode) + 1 :]
    options: list[str] = []
    files: list[str] = []
    steps: list[str] = []
    skip = False
    for token in rest:
        if skip:
            options.append(token)
            skip = False
        elif STEP.match(token):
            steps.append(token)
        elif token.startswith("-"):
            options.append(token)
            skip = token in VALUE_OPTIONS
        else:
            files.append(token)
    return mode, options, files, steps


def _kept(ctx: Ctx, mode: str, options: list[str], inputs: list[Path], steps: list[str]) -> int:
    """Reads kept by `trimmomatic <mode> <options> <inputs> <temp outputs> <steps>`."""
    with tempfile.TemporaryDirectory(prefix="biowarden-trimmomatic-") as tmp:
        count = 4 if mode == "PE" else 1
        outs = [str(Path(tmp) / f"out{i}.fastq") for i in range(count)]
        cmd = [mode, *options, *map(str, inputs), *outs, *steps]
        proc = ctx.run_real("trimmomatic", cmd)
        if proc.returncode != 0:
            raise DryRunFailed(proc.stderr[-200:])
        total = 0
        for out in outs:
            path = Path(out)
            if path.is_file():
                total += len(path.read_text().splitlines()) // 4
        return total


@checker("trimmomatic_minlen_order")
def trimmomatic_minlen_order(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    steps = steps_of(call.argv)
    if "MINLEN" not in steps:
        return passed()
    later = [s for s in steps[steps.index("MINLEN") + 1 :] if s in SHORTENING]
    declared = call.declared.get(key)
    if not later or declared == "before_trimming":
        return passed()
    full = [t for t in call.argv if STEP.match(t)]
    order = " ".join(full)
    moved = [t for t in full if not t.startswith("MINLEN:")]
    moved += [t for t in full if t.startswith("MINLEN:")]
    fix = (
        f"put MINLEN after the trimming steps (step order: {' '.join(moved)})",
        f"or, if the untrimmed length is the filter's target: --declare {key}=before_trimming",
    )
    if declared == "after_trimming":
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}=after_trimming but MINLEN comes before {', '.join(later)} in "
            f"the step order ({order}), so it filters the untrimmed reads.",
            expected="MINLEN after the trimming steps",
            got=order,
            fix=fix,
        )
    layout = _layout(call.argv)
    if layout is None or "-baseout" in layout[1]:
        return CheckResult("unknown", "cannot_rerun", f"MINLEN comes before {', '.join(later)}.")
    mode, options, files, raw_steps = layout
    n_in = 2 if mode == "PE" else 1
    inputs = [existing(ctx, f) for f in files[:n_in]]
    if len(inputs) < n_in or any(p is None or not small_file(p) for p in inputs):
        return CheckResult(
            "unknown",
            "cannot_rerun",
            f"MINLEN comes before {', '.join(later)} ({order}) and the input reads could not be "
            "found to test the effect.",
            fix=fix,
        )
    kept_options = drop_options(options, with_value={"-trimlog", "-summary", "-threads"})
    kept_options += ["-threads", "1"]
    reordered = [s for s in raw_steps if not s.startswith("MINLEN:")]
    reordered += [s for s in raw_steps if s.startswith("MINLEN:")]
    files_key = tuple(p for p in inputs if p is not None)
    try:
        as_written = memo(
            ("trim", str(ctx.cwd), mode, tuple(kept_options), tuple(raw_steps), stamp(files_key)),
            lambda: _kept(ctx, mode, kept_options, list(files_key), raw_steps),
        )
        after = memo(
            ("trim", str(ctx.cwd), mode, tuple(kept_options), tuple(reordered), stamp(files_key)),
            lambda: _kept(ctx, mode, kept_options, list(files_key), reordered),
        )
    except RUN_ERRORS as exc:  # a failing rerun is not evidence
        return CheckResult("unknown", "cannot_rerun", f"the rerun failed: {exc}", fix=fix)
    if as_written == after:
        return passed()
    return CheckResult(
        "fail",
        "minlen_before_trimming",
        f"Trimmomatic runs its steps in the order given, and MINLEN comes before "
        f"{', '.join(later)} ({order}): it filters the length of the reads BEFORE those steps "
        f"shorten them. On {files_key[0].name} the order as written keeps {as_written} reads, "
        f"with MINLEN last {after}.",
        expected=after,
        got=as_written,
        fix=fix,
    )
