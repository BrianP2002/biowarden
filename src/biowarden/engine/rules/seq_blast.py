"""BLAST+ rules of the sequence pack: the megablast default and a database that lags its FASTA.

blastn.megablast_default (family F77, slot default_param; T1, call, declaration optional)
    `blastn` without `-task` runs megablast: word size 28, match/mismatch 1/-2, built for
    near-identical sequences. A query that is a diverged copy of a reference region (80-90%
    identity) rarely holds 28 exact bases in a row, so it gets no hit at all; the run exits 0 with
    an empty stderr. `-task blastn` (word size 11) or `-task dc-megablast` finds it.

    Verified here (BLAST+ 2.17.0; 120-base queries cut from a 4 kb reference with substitutions
    only): default blastn found the queries at 100% and 95% identity, one of two at 90% and none at
    85, 80, 75 and 70%; `-task blastn` found every one down to 70%; exact 20- and 25-base queries
    were missed by the default and found by `-task blastn`, exact 35-base queries found by both.
    On the F77 sets (seeds 0-29) the default call missed 2-10 of 7-10 homologous queries on the
    faulted twin and none on the clean one.

    What the rule does. It applies to `blastn`. A call with `-task blastn`, `blastn-short` or
    `rmblastn` passes. A declaration `divergent` with the megablast task and a word size above
    16 (or none) fails at once. Otherwise, when `-query` and `-subject` (a FASTA file) or `-db`
    (files or a FASTA file) name small existing inputs (query up to 5 MB, database up to 20 MB),
    it runs the call (outputs redirected, `-outfmt "6 qseqid evalue"`, one thread) and the same
    call with `-task blastn`, and fails when a query has a hit with E <= min(1e-3, the call's
    -evalue) under `-task blastn` and no hit under the call. The default E-value 10 is not used
    for the extra hits: random 11-mer matches would make every unrelated query look like a miss.
    Declaration `blast_sensitivity`: near_identical (megablast is intended: pass); divergent
    (the queries may be diverged copies: pass with a sensitive task, fail with the default task
    or a task that still loses a query, see the rerun). Silent when nothing is lost, and on
    missing, unreadable or large inputs (`unknown` is never returned: a missing input is the
    tool's own error).
    Not seen: queries on stdin, `-import_search_strategy`, remote searches, `-db` aliases.

blast.stale_database (family F78, slot index_state; T1, call, declaration optional)
    `makeblastdb -in proteins.fasta` writes proteins.fasta.p* next to the FASTA and `blastp -db
    proteins.fasta` reads those files; proteins appended to the FASTA afterwards are invisible, and
    a query whose true partner is new gets its best remaining paralog. Exit 0, empty stderr.

    Verified here (BLAST+ 2.17.0): a database of 40 proteins next to a FASTA of 47: `blastdbcmd
    -info` and the `number-of-sequences` field of the .pjs file say 40, `grep -c '>'` says 47,
    `blastp -db proteins.fasta` searches the 40 without a message. The database title (the
    `description` field of the .pjs file, also in the .pin header) is the FASTA file name.

    What the rule does. It applies to blastp, blastn, blastx, tblastn, tblastx and psiblast. For
    each name in `-db` it reads NAME.pjs (protein) or NAME.njs (nucleotide), takes the source FASTA
    from the `description` and `dbname` fields (then the name itself), and compares the number of
    sequences (and the number of letters when every sequence is plain letters) with the FASTA as it
    is now. A difference fails the call unless the declaration `database_currency` is `frozen` (the
    old snapshot is intended). Declared `current` with a stale database also fails. Databases with
    several volumes or aliases, a missing .pjs/.njs or a missing FASTA are not analysed (pass).
"""

from __future__ import annotations

import json
from pathlib import Path

from biowarden.engine.argv import option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.seq_common import (
    RUN_ERRORS,
    drop_options,
    existing,
    memo,
    read_fasta,
    small_file,
    stamp,
)

SENSITIVE_TASKS = {"blastn", "blastn-short", "rmblastn"}
EXTRA_EVALUE = 1e-3
DB_LIMIT = 20_000_000


# ---- blastn.megablast_default -----------------------------------------------------------------


def _db_arguments(ctx: Ctx, argv: tuple[str, ...]) -> list[str] | None:
    """The `-subject FILE` or `-db NAME` pair of the call when it can be searched, else None."""
    subjects = option_values(argv, "-subject")
    if subjects:
        path = existing(ctx, subjects[-1])
        return ["-subject", str(path)] if path and small_file(path) else None
    names = option_values(argv, "-db")
    if not names:
        return None
    name = names[-1]
    for suffix in (".nin", ".nal", ".00.nin"):
        index = ctx.resolve(name + suffix)
        if index.is_file():
            data = ctx.resolve(name + ".nsq")
            if data.is_file() and not small_file(data, DB_LIMIT):
                return None
            return ["-db", str(ctx.resolve(name))]
    path = existing(ctx, name)  # a FASTA file given as the "database"
    return ["-subject", str(path)] if path and small_file(path) else None


def _hits(stdout: str) -> dict[str, float]:
    """query -> best (smallest) E-value from `-outfmt "6 qseqid evalue"` output."""
    best: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 2:
            try:
                value = float(fields[1])
            except ValueError:
                continue
            best[fields[0]] = min(value, best.get(fields[0], value))
    return best


@checker("blastn_megablast_default")
def blastn_megablast_default(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv
    tasks = option_values(argv, "-task")
    task = tasks[-1] if tasks else "megablast"
    declared = call.declared.get(key)
    if task in SENSITIVE_TASKS or declared == "near_identical":
        return passed()
    words = option_values(argv, "-word_size")
    small_word = bool(words) and words[-1].isdigit() and int(words[-1]) <= 16
    if declared == "divergent" and task == "megablast" and not small_word:
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}=divergent but this blastn call runs "
            f"{'the default task' if not tasks else '-task megablast'}, which needs an exact "
            "28-base seed and misses diverged copies (80-90% identity).",
            expected="divergent",
            got="near_identical",
            fix=("add -task blastn (word size 11), or -task dc-megablast",),
        )
    queries = option_values(argv, "-query")
    if not queries or queries[-1] == "-" or option_values(argv, "-import_search_strategy"):
        return passed()
    query = existing(ctx, queries[-1])
    target = _db_arguments(ctx, argv)
    if query is None or target is None or not small_file(query):
        return passed()
    base = drop_options(
        argv, with_value={"-out", "-outfmt", "-num_threads", "-export_search_strategy"},
        flags={"-html", "-remote"},
    )  # fmt: skip
    tail = ["-outfmt", "6 qseqid evalue", "-num_threads", "1"]
    files = [query]
    if target[0] == "-subject":
        files.append(Path(target[1]))
    else:
        files += [Path(target[1] + suffix) for suffix in (".nin", ".nsq", ".nhr")]
    sensitive_base = drop_options(base, with_value={"-task", "-word_size"})

    def run_both():
        try:
            return (
                ctx.run_real("blastn", [*base, *tail]),
                ctx.run_real("blastn", [*sensitive_base, "-task", "blastn", *tail]),
            )
        except RUN_ERRORS:  # a failing dry run is not evidence
            return None

    runs = memo(("blastn", str(ctx.cwd), tuple(base), stamp(files)), run_both)
    if runs is None:
        return passed()
    current, sensitive = runs  # type: ignore[misc]
    if current.returncode != 0 or sensitive.returncode != 0:
        return passed()
    evalues = option_values(argv, "-evalue")
    try:
        limit = min(EXTRA_EVALUE, float(evalues[-1])) if evalues else EXTRA_EVALUE
    except ValueError:
        limit = EXTRA_EVALUE
    have = set(_hits(current.stdout))
    gained = sorted(q for q, e in _hits(sensitive.stdout).items() if e <= limit and q not in have)
    if not gained:
        return passed()
    records = read_fasta(ctx, query) or []
    shown = ", ".join(gained[:3])
    used = "the default task (megablast)" if not tasks else f"-task {task}"
    return CheckResult(
        "fail",
        "megablast_misses_hits",
        f"{len(gained)} of {len(records) or '?'} queries in {query.name} ({shown}"
        f"{', ...' if len(gained) > 3 else ''}) have no hit with {used} but a hit with E <= "
        f"{limit:g} with -task blastn: megablast needs an exact 28-base seed and misses queries "
        "that are diverged copies (80-90% identity) or shorter than about 28 bases.",
        expected="-task blastn",
        got=task,
        fix=(
            "add -task blastn (word size 11; -task blastn-short for queries under 30 bases)",
            f"or, if only near-identical matches are wanted: --declare {key}=near_identical",
        ),
    )


# ---- blast.stale_database ---------------------------------------------------------------------


def _metadata(ctx: Ctx, name: str) -> tuple[Path, dict] | None:
    for suffix in (".pjs", ".njs"):
        path = ctx.resolve(name + suffix)
        if path.is_file():
            try:
                return path, json.loads("\n".join(ctx.lines(path)))
            except (OSError, ValueError):
                return None
    return None


def _source_fasta(ctx: Ctx, name: str, meta: dict) -> Path | None:
    candidates = [meta.get("description"), meta.get("dbname"), name]
    base = ctx.resolve(name).parent
    for item in candidates:
        if not item or not isinstance(item, str):
            continue
        for path in (ctx.resolve(item), base / Path(item).name):
            if path.is_file() and not any(
                path.name.endswith(s) for s in (".pjs", ".njs", ".pin", ".nin", ".phr", ".nhr")
            ):
                with path.open("rb") as handle:
                    if handle.read(1) == b">":
                        return path
    return None


@checker("blast_stale_database")
def blast_stale_database(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    declared = call.declared.get(key)
    if declared == "frozen":
        return passed()
    for value in option_values(call.argv, "-db"):
        for name in value.split():
            found = _metadata(ctx, name)
            if found is None:
                continue
            meta_path, meta = found
            try:
                n_db = int(meta["number-of-sequences"])
                letters_db = int(meta["number-of-letters"])
            except (KeyError, TypeError, ValueError):
                continue
            if int(meta.get("number-of-volumes", 1) or 1) > 1:
                continue  # several volumes
            fasta = _source_fasta(ctx, name, meta)
            if fasta is None or not small_file(
                fasta, ctx.max_bytes // 2
            ):  # read in full or not at all
                continue
            records = read_fasta(ctx, fasta, limit=10**7)
            if records is None:
                continue
            n_fasta = len(records)
            plain = all(seq.isalpha() for _, seq in records)
            letters = sum(len(seq) for _, seq in records)
            if n_fasta == n_db and (not plain or letters == letters_db):
                continue
            what = (
                f"{n_fasta} sequences"
                if n_fasta != n_db
                else f"{letters} letters in {n_fasta} sequences"
            )
            built = f"{n_db} sequences" if n_fasta != n_db else f"{letters_db} letters"
            program = "makeblastdb"
            kind = "nucl" if meta_path.suffix == ".njs" else "prot"
            return CheckResult(
                "fail",
                "database_older_than_fasta",
                f"the BLAST database {name} (title {meta.get('description', name)!r}) holds "
                f"{built}, but {fasta.name} now holds {what}: the search never sees the "
                "sequences added or changed since the database was built.",
                expected=n_fasta,
                got=n_db,
                fix=(
                    f"rebuild it first: {program} -in {fasta.name} -dbtype {kind} -out {name}",
                    f"or, if the old snapshot is intended: --declare {key}=frozen",
                ),
            )
    return passed()
