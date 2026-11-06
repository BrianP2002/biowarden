"""Family-specific tests for the `seq` pack (F77-F88): the trap size on seeds 0-29 from the recorded
answers, the twins and tiers, exposure on realistic commands, the real tools against the recorded
naive answer (exit 0 and no warning), and byte reproducibility across processes.

The generic gates (reference passes, naive falls into the trap on seeds 0-4, forced route, truth out
of the workspace, locus) live in test_extra_families.py. Skipped when the tools are missing.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.score import answer_matches
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.reference import solve
from biowarden.engine.spec import packaged_specs

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [f"F{n}" for n in range(77, 89)]
SEEDS = range(30)
# builds that validate the data with the real tool cost 1-3 s per seed: fewer seeds (the trap
# sizes over 100 and 40 seeds are recorded in the pack report)
SEEDS_BY_FAMILY = {"F78": range(20), "F83": range(20), "F84": range(10), "F87": range(10)}
WORK = Path("/nonexistent-workspace")  # the exposure checks read only the command text
NEEDS = {
    "seqtk", "seqkit", "fastp", "cutadapt", "trimmomatic", "makeblastdb", "blastn", "blastp",
    "diamond", "mafft", "muscle", "hmmbuild", "hmmsearch", "hmmscan",
}  # fmt: skip
SPECS = load()
_cache: dict[tuple, tuple[Path, object]] = {}


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not set(env.real_bins(sorted(NEEDS))) >= NEEDS:
        pytest.skip("the extended sequence environment is not available")
    return env


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return tmp_path_factory.mktemp("seq")


def built(base, tools, family, seed, twin, tier="implicit", route="free"):
    key = (family, seed, twin, tier, route)
    if key not in _cache:
        work = base / f"{family}-{seed}-{twin}-{tier}-{route}"
        _cache[key] = (work, build_task(family, seed, twin, work, tools, tier, route))
    return _cache[key]


# ---- registration and design ---------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_is_registered_with_a_locus_a_source_and_flat_twin_files(family):
    spec = SPECS[family]
    assert spec.locus in ("tool_default", "data") and spec.source and spec.slot
    assert spec.twin_files and all("/" not in name for name in spec.twin_files)
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")


def test_the_pack_touches_every_tool_and_covers_both_loci():
    rule_tools = {
        t
        for s in packaged_specs()
        if s.name.startswith("seq_")
        for r in s.rules
        for t in r.applies.tools
    }
    assert {
        "seqtk",
        "seqkit",
        "fastp",
        "cutadapt",
        "trimmomatic",
        "blastn",
        "blastp",
        "diamond",
    } <= rule_tools
    assert {"mafft", "muscle", "hmmsearch", "hmmscan"} <= rule_tools
    sources = " ".join(
        (ROOT / "src/biowarden/bench/tasks/extra" / f"{path.stem}.py").read_text()
        for path in (ROOT / "src/biowarden/bench/tasks/extra").glob("f[78][0-9]_*.py")
        if 77 <= int(path.stem.split("_")[0][1:]) <= 88
    )
    assert "makeblastdb" in sources and "hmmbuild" in sources  # used by builders and solvers
    loci = {SPECS[f].locus for f in FAMILIES}
    assert loci == {"tool_default", "data"}
    assert sum(len(s.rules) for s in packaged_specs() if s.name.startswith("seq_")) >= 8


@pytest.mark.parametrize("family", FAMILIES)
def test_trap_size_from_the_recorded_answers_over_30_seeds_and_both_tiers(base, tools, family):
    # the tiers differ only in README.md (next test), so the data are built once per seed and
    # the explicit tier is checked on the first seeds
    for seed in SEEDS_BY_FAMILY.get(family, SEEDS):
        tiers = ("implicit", "explicit") if seed < 5 else ("implicit",)
        for tier in tiers:
            _, clean = built(base, tools, family, seed, "clean", tier)
            _, faulted = built(base, tools, family, seed, "faulted", tier)
            as_naive = {**faulted.truth, "expected": faulted.truth["naive"]}
            assert answer_matches(clean.truth, clean.truth["naive"]), (family, seed, "clean")
            assert not answer_matches(faulted.truth, faulted.truth["naive"]), (family, seed, tier)
            assert answer_matches(as_naive, faulted.truth["naive"])
            assert clean.truth["expected"] == faulted.truth["expected"]


@pytest.mark.parametrize("family", FAMILIES)
def test_twins_differ_only_in_twin_files_and_tiers_only_in_the_readme(base, tools, family):
    spec = SPECS[family]
    for seed in (0, 7, 19):
        work_c, clean = built(base, tools, family, seed, "clean")
        work_f, faulted = built(base, tools, family, seed, "faulted")
        assert (
            clean.prompt == faulted.prompt and clean.truth["expected"] == faulted.truth["expected"]
        )
        names = {p.name for p in work_c.iterdir()} | {p.name for p in work_f.iterdir()}
        changed = {
            n
            for n in names
            if not (work_c / n).exists()
            or not (work_f / n).exists()
            or (work_c / n).read_bytes() != (work_f / n).read_bytes()
        }
        assert changed and changed <= spec.twin_files, (family, seed, changed)
        assert all((work_f / n).is_file() for n in names if (work_f / n).exists())  # no subdirs
        explicit, _ = built(base, tools, family, seed, "faulted", "explicit")
        diff = {
            n
            for n in names
            if (work_f / n).exists()
            and (explicit / n).exists()
            and (work_f / n).read_bytes() != (explicit / n).read_bytes()
        }
        assert diff <= {"README.md"}, (family, diff)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_prompt_names_the_files_and_the_forced_route_names_the_tool(base, tools, family):
    work, free = built(base, tools, family, 3, "faulted", "explicit", "free")
    _, forced = built(base, tools, family, 3, "faulted", "explicit", "forced")
    files = {p.name for p in work.iterdir()}
    mentioned = [n for n in files if n in free.prompt or n in "README.md"]
    assert mentioned and ("fam.hmm" in free.prompt or any(n in free.prompt for n in files))
    assert forced.prompt != free.prompt and forced.truth == free.truth
    assert free.task_id.startswith(f"{family}-s03-faulted")


# ---- the real tools against the recorded naive answer --------------------------------------

#  family: (argv of the naive call, files it needs to exist first)
NAIVE_CALLS = {
    "F77": ["blastn", "-query", "queries.fasta", "-subject", "ref.fasta", "-evalue", "1e-5",
            "-outfmt", "6 qseqid"],
    "F78": ["blastp", "-query", "queries.fasta", "-db", "proteins.fasta", "-outfmt", "6"],
    "F81": ["seqtk", "trimfq", "-q", "0.01", "reads.fastq"],
    "F82": ["seqkit", "rmdup", "-s", "guides.fasta"],
    "F84": ["cutadapt", "-g", "GTGCCAGCAGCCGCGGTAA", "-a", "AGATCGGAAGAGCACACGTC",
            "-o", "out.fastq", "reads.fastq"],
    "F86": ["mafft", "--quiet", "--auto", "seqs.fasta"],
}  # fmt: skip
QUIET_STDERR = {"F77", "F78", "F81", "F84", "F86"}


@pytest.mark.parametrize("family", sorted(NAIVE_CALLS))
def test_the_naive_call_exits_zero_and_says_nothing_on_stderr(base, tools, family):
    work, task = built(base, tools, family, 2, "faulted")
    proc = tools.run(NAIVE_CALLS[family], work)
    assert proc.returncode == 0
    if family in QUIET_STDERR:
        assert proc.stderr == "", proc.stderr
    else:  # seqkit: only INFO lines
        assert all(line.startswith("[INFO]") or "INFO" in line for line in proc.stderr.splitlines())


def test_the_chatty_tools_print_progress_but_no_warning_on_the_naive_route(base, tools):
    work, _ = built(base, tools, "F79", 2, "faulted")
    assert tools.run(["diamond", "makedb", "--in", "db.fasta", "-d", "db"], work).returncode == 0
    proc = tools.run(["diamond", "blastp", "-q", "query.fasta", "-d", "db", "-o", "o.tsv"], work)
    assert proc.returncode == 0 and "Reported 25 pairwise alignments" in proc.stderr
    assert "warn" not in proc.stderr.lower() and "error" not in proc.stderr.lower()
    work, _ = built(base, tools, "F80", 2, "faulted")
    assert tools.run(["diamond", "makedb", "--in", "db.fasta", "-d", "db"], work).returncode == 0
    proc = tools.run(["diamond", "blastp", "-q", "queries.fasta", "-d", "db", "-o", "o.tsv"], work)
    assert proc.returncode == 0 and "warn" not in proc.stderr.lower()
    work, _ = built(base, tools, "F83", 2, "faulted")
    proc = tools.run(["fastp", "-i", "reads.fastq", "-o", "o.fq", "-q", "20", "-j", "/dev/null",
                      "-h", "/dev/null"], work)  # fmt: skip
    assert proc.returncode == 0 and "reads failed due to low quality" in proc.stderr
    assert "warn" not in proc.stderr.lower() and "error" not in proc.stderr.lower()
    work, _ = built(base, tools, "F87", 2, "faulted")
    proc = tools.run(["fastp", "-i", "reads.fastq", "-o", "o.fq", "-j", "/dev/null",
                      "-h", "/dev/null"], work)  # fmt: skip
    assert proc.returncode == 0 and "No adapter detected" in proc.stderr
    assert "warn" not in proc.stderr.lower() and "error" not in proc.stderr.lower()
    work, _ = built(base, tools, "F85", 2, "faulted")
    argv = ["trimmomatic", "SE", "-Xmx512m", "-phred33", "reads.fastq", "o.fq", "LEADING:3",
            "MINLEN:36", "SLIDINGWINDOW:4:15", "TRAILING:3"]  # fmt: skip
    proc = tools.run(argv, work)
    assert proc.returncode == 0 and "Completed successfully" in proc.stderr
    assert "warn" not in proc.stderr.lower() and "exception" not in proc.stderr.lower()
    work, _ = built(base, tools, "F88", 2, "faulted")
    argv = ["hmmsearch", "--tblout", "t.tbl", "-E", "1e-5", "fam.hmm", "proteome_part1.fasta"]
    proc = tools.run(argv, work)
    assert proc.returncode == 0 and proc.stderr == ""


@pytest.mark.parametrize("family", FAMILIES)
def test_the_naive_solver_reproduces_the_recorded_answer_silently(base, tools, family):
    for seed in range(3):
        work, task = built(base, tools, family, seed, "faulted")
        naive = solve(task, work, tools, reference=False)
        recorded = {**task.truth, "expected": task.truth["naive"]}
        assert answer_matches(recorded, naive["answer"]), (family, seed, naive["answer"])
        assert not answer_matches(task.truth, naive["answer"])
        assert naive["status"] == "ok" and naive["flags"] == []
        work, task = built(base, tools, family, seed, "clean")
        assert answer_matches(task.truth, solve(task, work, tools, reference=False)["answer"])


# ---- exposure ------------------------------------------------------------------------------

P = "GTGCCAGCAGCCGCGGTAA"
A = "AGATCGGAAGAGCACACGTC"
EXPOSED = {
    "F77": [
        "blastn -query queries.fasta -subject ref.fasta -evalue 1e-5 -outfmt 6",
        "makeblastdb -in ref.fasta -dbtype nucl -out refdb && blastn -query q.fasta -db refdb",
        "blastn -task megablast -query queries.fasta -subject ref.fasta",
        "blastn -query queries.fasta -subject ref.fasta -word_size 20",
    ],
    "F78": [
        "blastp -query queries.fasta -db proteins.fasta -outfmt '6 qseqid sseqid bitscore'",
        "blastp -db ./proteins.fasta -query queries.fasta | sort -k3,3nr",
    ],
    "F79": [
        "diamond blastp -d db -q query.fasta -o out.tsv",
        "diamond blastp -d db -q query.fasta -k 10 -o out.tsv",
        "diamond blastp -d db -q query.fasta --max-target-seqs 25 -f 6 sseqid bitscore",
        "diamond blastp -d db -q query.fasta -e 1e-5 --sensitive -o out.tsv",
    ],
    "F80": [
        "diamond blastp -d db -q queries.fasta -o out.tsv",
        "diamond blastp -d db -q queries.fasta --fast -o out.tsv",
        "diamond blastp -d db -q queries.fasta --mid-sensitive -k 0",
        "diamond blastp -d db -q queries.fasta --sensitive -o out.tsv",
        "diamond blastp -d db -q queries.fasta --more-sensitive",
        "diamond blastp -d db -q queries.fasta --very-sensitive",
    ],
    "F81": [
        "seqtk trimfq -q 0.01 reads.fastq > trimmed.fq",
        "seqtk trimfq reads.fastq | seqtk seq -A",
        "seqtk trimfq -q 0.01 -l 30 reads.fastq",
        "seqtk trimfq -l 20 -q 0.05 reads.fastq",
    ],
    "F82": [
        "seqkit rmdup -s guides.fasta | grep -c '>'",
        "seqkit rmdup --by-seq guides.fasta -o out.fa",
        "seqkit rmdup -si guides.fasta",
        "seqkit rmdup -j 2 -s guides.fasta | seqkit stats",
    ],
    "F83": [
        "fastp -i reads.fastq -o out.fq -q 20",
        "fastp -i reads.fastq -o out.fq -q 20 -u 30 -j r.json",
        "fastp -i reads.fastq -o out.fq",
    ],
    "F84": [
        f"cutadapt -g {P} -a {A} -o out.fastq reads.fastq",
        f"cutadapt -a {A} -g {P} -m 10 -o out.fastq reads.fastq",
        f"cutadapt -g {P} -a {A} -n 1 -o out.fastq reads.fastq",
    ],
    "F85": [
        "trimmomatic SE -phred33 r.fastq o.fq LEADING:3 MINLEN:36 SLIDINGWINDOW:4:15 TRAILING:3",
        "trimmomatic SE reads.fastq out.fq MINLEN:36 SLIDINGWINDOW:4:15",
    ],
    "F86": [
        "mafft --auto seqs.fasta > aln.fasta",
        "mafft --quiet --localpair --maxiterate 100 seqs.fasta",
        "muscle -align seqs.fasta -output aln.afa",
    ],
    "F87": [
        "fastp -i reads.fastq -o out.fastq",
        "fastp -i reads.fastq -o out.fastq -l 20 -j r.json -h r.html",
        "fastp -i reads.fastq -o out.fastq -a auto",
        "fastp -i reads.fastq -o out.fastq --adapter_sequence auto",
        "fastp -i reads.fastq -o out.fastq --adapter_sequence=auto -l 20",
    ],
    "F88": [
        "hmmsearch -E 1e-5 --tblout t.tbl fam.hmm proteome_part1.fasta",
        "for f in proteome_part*.fasta; do hmmsearch -E 1e-5 --tblout $f.tbl fam.hmm $f; done",
        "hmmsearch --tblout t.tbl fam.hmm proteome_part3.fasta",
    ],
}
SAFE = {
    "F77": [
        "blastn -task blastn -query queries.fasta -subject ref.fasta",
        "blastn -task blastn-short -query queries.fasta -subject ref.fasta",
        "blastn -task dc-megablast -query queries.fasta -subject ref.fasta",
        "blastn -query queries.fasta -subject ref.fasta -word_size 11",
        "blastp -query queries.fasta -db proteins.fasta",
    ],
    "F78": [
        "makeblastdb -in proteins.fasta -dbtype prot -out fresh && blastp -query q.fasta -db fresh",
        "blastp -query queries.fasta -subject proteins.fasta",
        "makeblastdb -in proteins.fasta -dbtype prot; blastp -db proteins.fasta -query q.fasta",
        "blastdbcmd -db proteins.fasta -info",
    ],
    "F79": [
        "diamond blastp -d db -q query.fasta -k 0 -o out.tsv",
        "diamond blastp -d db -q query.fasta --max-target-seqs 0",
        "diamond blastp -d db -q query.fasta -k 500",
        "diamond makedb --in db.fasta -d db",
    ],
    "F80": [
        "diamond blastp -d db -q queries.fasta --ultra-sensitive -o out.tsv",
        "diamond blastp -d db -q queries.fasta --threads 4 --ultra-sensitive -k 0",
        "diamond makedb --in db.fasta -d db",
    ],
    "F81": [
        "seqtk trimfq -q 0.01 -l 1 reads.fastq",
        "seqtk trimfq -b 5 reads.fastq",
        "seqtk comp reads.fastq | head",
        "seqkit stats reads.fastq",
    ],
    "F82": [
        "seqkit rmdup -s -P guides.fasta",
        "seqkit rmdup -sP guides.fasta",
        "seqkit rmdup guides.fasta",
        "sort -u guides.fasta | wc -l",
    ],
    "F83": [
        "fastp -i reads.fastq -o out.fq --cut_right -W 4 -M 20",
        "fastp -i reads.fastq -o out.fq -q 20 -r",
        "fastp -i reads.fastq -o out.fq --cut_tail",
        "fastp -i reads.fastq -o out.fq -5 -3",
    ],
    "F84": [
        f"cutadapt -g {P} -a {A} -n 2 -o out.fastq reads.fastq",
        f"cutadapt -g {P} -a {A} --times 2 -o out.fastq reads.fastq",
        f"cutadapt -g {P}...{A} -o out.fastq reads.fastq",
        f"cutadapt -g {P} -o out.fastq reads.fastq",
    ],
    "F85": [
        "trimmomatic SE -phred33 r.fastq o.fq LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36",
        "trimmomatic SE reads.fastq out.fq SLIDINGWINDOW:4:15 MINLEN:36",
    ],
    "F86": [
        "mafft --adjustdirection seqs.fasta > aln.fasta",
        "mafft --adjustdirectionaccurately --auto seqs.fasta",
        "seqkit stats seqs.fasta",
    ],
    "F87": [
        "fastp -i reads.fastq -o out.fastq -a AGATCGGAAGAGC",
        "fastp -i reads.fastq -o out.fastq --adapter_sequence AGATCGGAAGAGC -l 20",
        "fastp -i reads.fastq -o out.fastq -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",
        "fastp -i reads.fastq -o out.fastq --adapter_sequence=AGATCGGAAGAGCACACGTC",
        "fastp -i reads.fastq -o out.fastq -a $ADAPTER",
        "fastp -i reads.fastq -o out.fastq --adapter_fasta adapters.fa",
    ],
    "F88": [
        "hmmsearch -E 1e-5 -Z 200 --tblout t.tbl fam.hmm proteome_part1.fasta",
        "cat proteome_part*.fasta > all.fasta && hmmsearch -E 1e-5 fam.hmm all.fasta",
        "hmmsearch -E 1e-5 --tblout t.tbl fam.hmm proteome_all.fasta",
        "hmmbuild fam.hmm fam.sto",
    ],
}


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_sees_the_trapped_route_per_command_and_nothing_else(family):
    spec = SPECS[family]
    for command in EXPOSED[family]:
        assert spec.exposed([command], WORK) is True, (family, command)
    for command in SAFE[family]:
        assert spec.exposed([command], WORK) is False, (family, command)
    assert spec.exposed(["ls -la; cat README.md", "head -4 reads.fastq"], WORK) is False
    # an agent that tried the naive route and then corrected still executed the trap
    assert spec.exposed([SAFE[family][0], EXPOSED[family][0]], WORK) is True


# ---- fixes of the final review: F80 (FD-2), F81 (FD-12), F87 (FD-1) -------------------------

TRUSEQ_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"  # TruSeq Read 1 adapter as the textbooks give it
ADAPTER_LENGTHS = (12, 13, 20, 33, 34)  # the 33 bases, then the C that follows them in the library


def fresh(tools, tmp_path, family, seed, twin, tier="implicit", route="free"):
    """A workspace in its own directory, so that a tool run leaves the cached ones alone."""
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def test_f80_only_ultra_sensitive_finds_every_partner_and_exposure_agrees(tools, tmp_path):
    from biowarden.bench.tasks.extra import f80_diamond_sens as f80

    # seeds 5 and 1001 are the sets where --very-sensitive also loses a partner
    missed = {"--sensitive": 0, "--more-sensitive": 0, "--very-sensitive": 0}
    for seed in (0, 1, 5, 1001, 1003):
        work, task = fresh(tools, tmp_path, "F80", seed, "faulted")
        expected = task.truth["expected"]
        assert f80.found(tools, work, "--ultra-sensitive") == expected, seed
        for mode in missed:
            lost = set(expected) - set(f80.found(tools, work, mode))
            missed[mode] += bool(lost)
            command = f"diamond blastp -d db -q queries.fasta {mode} -o out.tsv"
            assert SPECS["F80"].exposed([command], WORK) is True
    # DIAMOND 2.2.8: the first two lose a partner on all five sets, --very-sensitive on 5 and 1001
    assert missed["--sensitive"] >= 4 and missed["--more-sensitive"] >= 4
    assert missed["--very-sensitive"] >= 1
    ultra = "diamond blastp -d db -q queries.fasta --ultra-sensitive -o out.tsv"
    assert SPECS["F80"].exposed([ultra], WORK) is False


@pytest.mark.parametrize("seed", range(30))
def test_f81_readme_states_the_read_lengths_of_its_twin_and_what_counts(base, tools, seed):
    texts = {}
    for twin, shortest in (("clean", 10), ("faulted", 60)):
        work, _ = built(base, tools, "F81", seed, twin)
        lengths = [len(line) for line in (work / "reads.fastq").read_text().splitlines()[1::4]]
        readme = (work / "README.md").read_text()
        assert f"{shortest}-90 bases" in readme, (seed, twin)
        assert shortest <= min(lengths) and max(lengths) <= 90, (seed, twin)
        assert "counts only the good part of each read" in readme, (seed, twin)
        texts[twin] = readme
    assert texts["clean"].replace("10-90", "60-90") == texts["faulted"]  # no other twin cue


def test_f81_explicit_readme_keeps_its_own_sentence_on_top_of_the_intent(base, tools):
    for twin, needle in (("faulted", "however short that part is"), ("clean", "trimmed before")):
        work, _ = built(base, tools, "F81", 2, twin, "explicit")
        readme = (work / "README.md").read_text()
        assert needle in readme and "counts only the good part of each read" in readme
    implicit = (built(base, tools, "F81", 2, "faulted", "implicit")[0] / "README.md").read_text()
    assert "however short" not in implicit  # the floor stays unnamed in the implicit tier


def test_f87_the_faulted_reads_run_into_the_whole_adapter(base, tools):
    carried = TRUSEQ_R1 + "C"
    work, _ = built(base, tools, "F87", 2, "faulted")
    reads = (work / "reads.fastq").read_text().splitlines()[1::4]
    stem = [r for r in reads if "AGATCGGAAGAGC" in r]
    assert len(stem) > 0.6 * len(reads)  # the 75% of the reads with an insert shorter than 100
    for r in stem:
        tail = r[r.index("AGATCGGAAGAGC") :]
        k = min(len(tail), len(carried))
        assert tail[:k] == carried[:k] and len(r) == 100
    implicit = (work / "README.md").read_text()
    assert "AGATCGG" not in implicit and "TruSeq library" in implicit  # the tier stays implicit
    explicit = (built(base, tools, "F87", 2, "faulted", "explicit")[0] / "README.md").read_text()
    assert "TruSeq adapter sequence: AGATCGGAAGAGC." in explicit


def fastp_total(tools, work, *options):
    argv = ["fastp", "-i", "reads.fastq", "-o", "o.fastq", "-j", "o.json", "-h", "o.html"]
    proc = tools.run([*argv, "--thread", "1", *options], work)
    assert proc.returncode == 0, proc.stderr
    return sum(len(line) for line in (work / "o.fastq").read_text().splitlines()[1::4])


@pytest.mark.parametrize("seed", [0, 1, 7, 1000, 1004])
def test_f87_every_textbook_adapter_length_trims_exactly_and_the_default_call_does_not(
    tools, tmp_path, seed
):
    work_f, faulted = fresh(tools, tmp_path, "F87", seed, "faulted")
    work_c, clean = fresh(tools, tmp_path, "F87", seed, "clean")
    expected = faulted.truth["expected"]
    assert clean.truth["expected"] == expected
    for length in ADAPTER_LENGTHS:
        adapter = (TRUSEQ_R1 + "C")[:length]
        assert fastp_total(tools, work_f, "-a", adapter) == expected, (seed, length)
        assert fastp_total(tools, work_c, "-a", adapter) == expected, (seed, length, "clean")
    naive = faulted.truth["naive"]
    assert naive != expected and fastp_total(tools, work_f) == naive
    assert fastp_total(tools, work_f, "-a", "auto") == naive  # `-a auto` is the detection
    assert fastp_total(tools, work_c) == clean.truth["naive"] == expected


# ---- byte reproducibility across processes -------------------------------------------------

SNIPPET = """
import hashlib, sys
from pathlib import Path
from biowarden.bench.tasks import ToolEnv, build_task
tools = ToolEnv.discover(Path(sys.argv[1]))
work = Path(sys.argv[2])
build_task(sys.argv[3], int(sys.argv[4]), "faulted", work, tools, "implicit", "free")
for p in sorted(work.iterdir()):
    print(p.name, hashlib.sha256(p.read_bytes()).hexdigest())
"""


@pytest.mark.parametrize("family", ["F78", "F84", "F87", "F88"])
def test_builds_are_byte_reproducible_across_processes(tools, tmp_path, family):
    outputs = []
    for i in range(2):
        proc = subprocess.run(
            [sys.executable, "-I", "-c", SNIPPET, str(ROOT), str(tmp_path / f"w{i}"), family, "5"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] and outputs[0].count("\n") >= 2
    assert hashlib.sha256(outputs[0].encode()).hexdigest()
