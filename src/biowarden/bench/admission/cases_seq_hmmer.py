"""Admission cases for hmmer.split_database (family F88, implicit tier).

The task asks for the proteins with an E-value of at most 1e-5 in the complete proteome. The faulted
twin ships the proteome as eight files of equal size; the clean twin as one file. A "call" is one
`hmmsearch ... fam.hmm <file>`, but the answer needs every file, so each option set is run on all
files and the union of the hits with E <= 1e-5 is compared with `task.truth`; the case emitted for
the option set is the call on the first file (and on the fifth, a second representative). An option
set is right when the union is the truth: `-Z <total>` on each part (or a concatenated file); wrong
on the faulted twin when E-values come from the parts. On the clean twin the plain call is right.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from biowarden.bench.admission import Case, Label, case_source
from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra.f88_hmmer_split import THRESHOLD, parse_tblout
from biowarden.engine import Call

KEY = "evalue_database"


def run_union(tools: ToolEnv, work: Path, tag: str, files: list[str], options: tuple[str, ...]):
    """Union of the hits with E <= THRESHOLD over `files`, and the calls that were run."""
    calls = []
    found: set[str] = set()
    for k, name in enumerate(files):
        table = f"hits_{tag}_{k}.tbl"
        argv = ("--cpu", "1", "--tblout", table, "-o", "/dev/null", *options, "fam.hmm", name)
        proc = tools.run(["hmmsearch", *argv], work)
        if proc.returncode != 0:
            return None, calls
        found |= {i for i, e in parse_tblout((work / table).read_text()).items() if e <= THRESHOLD}
        calls.append(argv)
    return sorted(found), calls


@case_source("hmmer.split_database")
def split_database_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    out: list[Case] = []
    for twin in ("clean", "faulted"):
        work = scratch / twin
        task = build_task("F88", seed, twin, work, tools, "implicit", "free")
        parts = sorted(p.name for p in work.glob("proteome_part*.fasta"))
        files = parts or ["proteome.fasta"]
        total = sum((work / n).read_text().count(">") for n in files)
        if parts:  # the concatenated proteome, for the "merge first" route
            text = "".join((work / n).read_text() for n in parts)
            (work / "proteome_all.fasta").write_text(text)
        e = ("-E", "1e-5")
        both = ("this_file", "whole_collection")
        whole = ("whole_collection",)
        # name, options, files, declarations if right, declarations if wrong
        variants = [
            ("e_threshold", e, files, both, both),
            ("e_default", (), files, both, both),
            ("e_dom", (*e, "--domE", "1e-5"), files, both, both),
            ("e_inc", ("--incE", "1e-5", "-E", "1e-5"), files, both, both),
            ("z_total", (*e, "-Z", str(total)), files, whole, whole),
            ("z_default", ("-Z", str(total)), files, whole, whole),
            ("z_attached", (*e, f"-Z{total}"), files, whole, whole),
        ]
        if parts:
            variants.append(("merged", e, ["proteome_all.fasta"], whole, whole))

        def one(variant, work=work):
            return run_union(tools, work, variant[0], variant[2], variant[1])

        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(one, variants))
        for (name, _options, _files, right_decl, wrong_decl), (answer, calls) in zip(
            variants, runs, strict=True
        ):
            label: Label = (
                "right" if answer is not None and answer_matches(task.truth, answer) else "wrong"
            )
            declared = right_decl if label == "right" else wrong_decl
            picked = calls[:1] if len(calls) <= 1 else [calls[0], calls[4]]
            for argv in picked:
                out.append(
                    Case(Call("hmmsearch", argv, work), label, False, "F88", seed, twin, name)
                )
                for value in declared:
                    call = Call("hmmsearch", argv, work, {KEY: value})
                    out.append(Case(call, label, True, "F88", seed, twin, f"{name}:{value}"))
    return out
