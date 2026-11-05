"""Admission cases for the BLAST rules (families F77 and F78, implicit tier).

blastn.megablast_default (F77): `blastn -query queries.fasta -subject ref.fasta -evalue 1e-5` with
many option sets. A call is right when the queries it reports (E < 1e-5) are exactly the
homologous ones. Megablast calls (the default task, `-task megablast`, a word size above 11)
are right on the clean twin and wrong on the faulted one; the calls that run `-task blastn`,
`dc-megablast`, or `blastn-short` are labelled by running them.

blast.stale_database (F78): `blastp -query queries.fasta -db proteins.fasta` and variants against
the shipped database (right on the clean twin, wrong on the faulted one), against a database
rebuilt under another name, rebuilt in place (a subdirectory copy), and `-subject` (no database).
The answer is the best hit by bit score for every query in file order.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_seq_common import Tpl, cases_for
from science_harness.bench.tasks import ToolEnv
from science_harness.bench.tasks.base import Task

# ---- blastn.megablast_default (F77) -----------------------------------------------------------

KEY_BLASTN = "blast_sensitivity"
SUBJECT = ("-query", "queries.fasta", "-subject", "ref.fasta")
REST = ("-evalue", "1e-5", "-outfmt", "6", "-num_threads", "1")
DB = ("-query", "queries.fasta", "-db", "refdb")
CLOSE = ("near_identical",)
CLOSE_WRONG = ("near_identical", "divergent")
OPEN = ("divergent",)
OPEN_WRONG = ("divergent", "near_identical")


def hit_ids(work: Path, proc) -> list[str]:
    return sorted({line.split("\t")[0] for line in proc.stdout.splitlines() if line.strip()})


def _blastn(name: str, base: tuple[str, ...], *options: str, sensitive: bool) -> Tpl:
    right, wrong = (OPEN, OPEN_WRONG) if sensitive else (CLOSE, CLOSE_WRONG)
    return Tpl(name, "blastn", (*base, *options, *REST), hit_ids, right, wrong)


BLASTN_TEMPLATES: tuple[Tpl, ...] = (
    # --- megablast: needs an exact 28-base seed
    _blastn("default", SUBJECT, sensitive=False),
    _blastn("megablast", SUBJECT, "-task", "megablast", sensitive=False),
    _blastn("word28", SUBJECT, "-word_size", "28", sensitive=False),
    _blastn("max_target", SUBJECT, "-max_target_seqs", "5", sensitive=False),
    _blastn("strand_plus", SUBJECT, "-strand", "plus", sensitive=False),
    _blastn("db_default", DB, sensitive=False),
    _blastn("db_megablast", DB, "-task", "megablast", sensitive=False),
    # --- more sensitive settings
    _blastn("blastn", SUBJECT, "-task", "blastn", sensitive=True),
    _blastn("blastn_identity", SUBJECT, "-task", "blastn", "-perc_identity", "80", sensitive=True),
    _blastn("blastn_dust_off", SUBJECT, "-task", "blastn", "-dust", "no", sensitive=True),
    _blastn("dc_megablast", SUBJECT, "-task", "dc-megablast", sensitive=True),
    _blastn("blastn_short", SUBJECT, "-task", "blastn-short", sensitive=True),
    _blastn("word11", SUBJECT, "-word_size", "11", sensitive=True),
    _blastn("word16", SUBJECT, "-word_size", "16", sensitive=True),
    _blastn("db_blastn", DB, "-task", "blastn", sensitive=True),
)


@case_source("blastn.megablast_default")
def megablast_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def prepare(work: Path, task: Task) -> None:
        proc = tools.run(
            ["makeblastdb", "-in", "ref.fasta", "-dbtype", "nucl", "-out", "refdb"], work
        )
        assert proc.returncode == 0, proc.stderr

    return cases_for("F77", KEY_BLASTN, seed, scratch, tools, BLASTN_TEMPLATES, prepare)


# ---- blast.stale_database (F78) ---------------------------------------------------------------

KEY_DB = "database_currency"
FORMAT = ("-outfmt", "6 qseqid sseqid bitscore", "-num_threads", "1")
CURRENT = ("current",)
SHIPPED_WRONG = ("frozen", "current")  # the snapshot (consistent with the call) / the FASTA


def best_hits(work: Path, proc) -> list[str]:
    order = [
        line[1:].split()[0]
        for line in (work / "queries.fasta").read_text().splitlines()
        if line.startswith(">")
    ]
    best: dict[str, tuple[float, str]] = {}
    for line in proc.stdout.splitlines():
        query, subject, score = line.split("\t")
        if query not in best or float(score) > best[query][0]:
            best[query] = (float(score), subject)
    return [best[q][1] if q in best else "none" for q in order]


def _blastp(name: str, db: tuple[str, ...], *options: str, subdir: str = "", wrong=SHIPPED_WRONG):
    argv = ("-query", "queries.fasta", *db, *options, *FORMAT)
    return Tpl(name, "blastp", argv, best_hits, CURRENT, wrong, subdir)


SHIPPED = ("-db", "proteins.fasta")
STALE_TEMPLATES: tuple[Tpl, ...] = (
    # --- the shipped database: right only when it is current
    _blastp("shipped", SHIPPED),
    _blastp("shipped_evalue", SHIPPED, "-evalue", "1e-3"),
    _blastp("shipped_max_target", SHIPPED, "-max_target_seqs", "5"),
    _blastp("shipped_no_cbs", SHIPPED, "-comp_based_stats", "0"),
    _blastp("shipped_quoted_prefix", ("-db", "./proteins.fasta")),
    # --- a database made from the current FASTA
    _blastp("fresh", ("-db", "fresh"), wrong=CURRENT),
    _blastp("fresh_evalue", ("-db", "fresh"), "-evalue", "1e-3", wrong=CURRENT),
    _blastp("subject", ("-subject", "proteins.fasta"), wrong=CURRENT),
    _blastp("rebuilt_in_place", SHIPPED, subdir="rebuilt", wrong=CURRENT),
)


@case_source("blast.stale_database")
def stale_database_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    def prepare(work: Path, task: Task) -> None:
        build = ["makeblastdb", "-dbtype", "prot", "-in", "proteins.fasta"]
        assert tools.run([*build, "-out", "fresh"], work).returncode == 0
        again = work / "rebuilt"
        again.mkdir()
        for name in ("proteins.fasta", "queries.fasta"):
            shutil.copy(work / name, again / name)
        assert tools.run(build, again).returncode == 0  # rebuilt in place, same name

    return cases_for("F78", KEY_DB, seed, scratch, tools, STALE_TEMPLATES, prepare)
