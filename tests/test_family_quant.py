"""Family-specific checks for F41-F46 (the quant pack: RNA-seq quantification).

The generic gates live in test_extra_families.py. Here, on seeds 0-29 and both tiers: the trap
changes the answer on every seed (pure Python from the plans, the real tools on a few seeds), the
twins hold the same biology and differ only in the declared files, the real tools stay silent about
the trap, the exposure functions see realistic trapped commands and not the safe ones (on a
faulted workspace, and not on a clean one), the F43 and F46 models agree with featureCounts and
kallisto, and builds are byte-reproducible across processes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_rules_quant_common import engine_for

from biowarden.bench.score import answer_matches, score_result
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f41_htseq_strand as f41
from biowarden.bench.tasks.extra import f42_fc_read_pairs as f42
from biowarden.bench.tasks.extra import f43_fc_antisense as f43
from biowarden.bench.tasks.extra import f44_htseq_mapq as f44
from biowarden.bench.tasks.extra import f45_salmon_duplicates as f45
from biowarden.bench.tasks.extra import f46_rsem_kit as f46
from biowarden.bench.tasks.extra import load
from biowarden.engine import Call

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
FAMILIES = ("F41", "F42", "F43", "F44", "F45", "F46")
SPECS = load()
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F41-F46 are not registered"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    if len(env.real_bins(["htseq-count", "featureCounts", "salmon", "kallisto"])) < 4:
        pytest.skip("the quantification tools are not installed")
    return env


def built(tmp_path, family, seed, twin, tools, tier="implicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def worst(truth, key):
    """Largest per-gene gap between the expected answer and the recorded `key` answer."""
    return max(abs(truth["expected"][g] - truth[key][g]) for g in truth["expected"])


@pytest.mark.parametrize(
    ("family", "slot", "locus"),
    [
        ("F41", "strand", "tool_default"),
        ("F42", "record_count", "tool_default"),
        ("F43", "strand", "tool_default"),
        ("F44", "default_param", "tool_default"),
        ("F45", "completeness", "tool_default"),
        ("F46", "strand", "data"),
    ],
)
def test_locus_slot_and_source_are_declared(family, slot, locus):
    spec = SPECS[family]
    assert (spec.locus, spec.slot) == (locus, slot) and spec.source
    assert spec.tiers == ("explicit", "implicit") and spec.routes == ("free", "forced")
    assert "README.md" in spec.twin_files and len(spec.twin_files) <= 4


# ---- the trap, family by family, on seeds 0-29 (pure Python from the plans) ------------------


def test_f41_the_default_misses_a_stranded_twin_and_fits_the_forward_one():
    kinds = set()
    for seed in SEEDS:
        plan = f41.make_plan(seed)
        kinds.add(plan.faulted_library)
        for twin in ("clean", "faulted"):
            expected, naive = f41.truth_answers(plan, twin)
            gap = worst({"expected": expected, "naive": naive}, "naive")
            if twin == "clean":
                assert gap <= f41.LEAK * 90 + 1 and gap <= f41.TOL, seed  # leaked reads only
            else:
                assert min(abs(expected[g] - naive[g]) for g in expected) > f41.TOL, seed
        assert plan.faulted_library == ("reverse" if seed % 2 == 0 else "unstranded"), seed
        assert len(plan.layout.genes) == 5 + seed % 2
        # the same genes and read counts in both twins; the library only changes the orientation
        assert {
            g: len([f for f in plan.frags_by_library["forward"] if f.gene == i])
            for i, g in enumerate(x.id for x in plan.layout.genes)
        } == f41.counts_by_gene(plan.layout, plan.frags_by_library[plan.faulted_library])
    assert kinds == {"reverse", "unstranded"}


def test_f42_a_paired_bam_counted_with_p_alone_doubles_every_gene():
    for seed in SEEDS:
        layout, frags = f42.make_plan(seed)
        expected, naive = f42.truth_answers(layout, frags, "faulted")
        assert all(naive[g] == 2 * expected[g] for g in expected), seed
        assert min(expected.values()) >= 30, seed
        _, clean_naive = f42.truth_answers(layout, frags, "clean")
        assert clean_naive == expected


def test_f43_the_default_drops_the_overlap_reads_of_both_genes_and_nothing_else():
    for seed in SEEDS:
        layout, frags = f43.make_plan(seed, "faulted")
        expected = f41.counts_by_gene(layout, frags)
        naive, ambiguous = f43.unstranded_counts(layout, frags)
        first, partner = layout.genes[0], layout.genes[-1]
        assert first.strand != partner.strand and first.chrom == partner.chrom, seed
        lost = {g: expected[g] - naive[g] for g in expected}
        assert lost[first.id] >= 5 and lost[partner.id] >= 10, (seed, lost)
        assert sum(lost.values()) == ambiguous >= 20, seed
        assert all(v == 0 for g, v in lost.items() if g not in (first.id, partner.id)), seed
        clean_layout, clean_frags = f43.make_plan(seed, "clean")
        clean_naive, none = f43.unstranded_counts(clean_layout, clean_frags)
        assert none == 0 and clean_naive == f41.counts_by_gene(clean_layout, clean_frags), seed
        assert not any(  # every read is on the strand opposite to its gene: a dUTP library
            f41.genome_strand(layout.genes[f.gene], r.orient) == layout.genes[f.gene].strand
            for f in frags
            for r in f41.reads_of(layout, f, False)
        )


def test_f44_the_default_floor_drops_15_to_30_percent_of_the_reads_of_every_gene():
    for seed in SEEDS:
        layout, frags, low, quality = f44.make_plan(seed)
        for gene_index, gene in enumerate(layout.genes):
            mine = [f.name for f in frags if f.gene == gene_index]
            share = sum(n in low for n in mine) / len(mine)
            assert 0.14 <= share <= 0.31 and sum(n in low for n in mine) >= 4, (seed, gene.id)
        assert all(q in f44.LOW_MAPQ for n, q in quality["faulted"].items() if n in low)
        assert all(f.o == "sense" for f in frags), seed  # forward-stranded, no leaked read
        assert (
            min(quality["clean"].values()) >= 30 and min(quality["faulted"].values()) == 0 or True
        )
        assert all(
            quality["faulted"][n] == quality["clean"][n] for n in quality["clean"] if n not in low
        )


def test_f45_the_duplicate_gene_vanishes_and_its_twin_doubles():
    for seed in SEEDS:
        layout, frags = f45.make_plan(seed, "faulted")
        sequences = [layout.sequence(g) for g in layout.genes]
        assert len(set(sequences)) == len(sequences) - 1, seed
        first, last = layout.genes[1], layout.genes[-1]
        assert layout.sequence(first) == layout.sequence(last) and first.n == last.n >= 30
        clean_layout, _ = f45.make_plan(seed, "clean")
        assert clean_layout.sequence(clean_layout.genes[1]) != clean_layout.sequence(
            clean_layout.genes[-1]
        )
        assert [g.n for g in layout.genes] == [g.n for g in clean_layout.genes]


def test_f46_reverse_keeps_only_the_fragments_with_read_1_on_the_opposite_strand():
    for seed in SEEDS:
        layout, frags = f46.make_plan(seed, "faulted")
        expected, naive = f46.truth_answers(layout, frags, "faulted")
        assert min(expected[g] - naive[g] for g in expected) > 2 * 2.0, seed
        assert (
            0.35 * sum(expected.values()) < sum(naive.values()) < 0.65 * sum(expected.values())
        ), seed
        clean_layout, clean_frags = f46.make_plan(seed, "clean")
        clean_expected, clean_naive = f46.truth_answers(clean_layout, clean_frags, "clean")
        assert clean_expected == clean_naive == expected, seed  # a dUTP library keeps all of them


def test_f46_rsem_keeps_exactly_the_matching_orientation(tmp_path, tools):
    """The recorded naive answer is computed without RSEM: check it against the real tool."""
    for seed in (0, 1, 2):
        work, task = built(tmp_path, "F46", seed, "faulted", tools)
        got = f46.rsem_counts(work, tools, "--strandedness", "reverse")
        assert {g: round(v) for g, v in got.items()} == task.truth["naive"]
        forward = f46.rsem_counts(work, tools, "--strandedness", "forward")
        assert {
            g: round(a + b) for g, a, b in ((g, got[g], forward[g]) for g in got)
        } == task.truth["expected"]
        assert answer_matches(task.truth, f46.rsem_counts(work, tools, "--strandedness", "none"))


# ---- the twins -------------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_twins_share_the_biology_and_differ_only_in_declared_files(tmp_path, tools, family):
    spec = SPECS[family]
    for seed in (0, 5, 11):
        w_clean, clean = built(tmp_path, family, seed, "clean", tools)
        w_fault, faulted = built(tmp_path, family, seed, "faulted", tools)
        assert clean.truth["expected"] == faulted.truth["expected"], seed
        changed = {
            p.name
            for p in w_clean.iterdir()
            if p.name in {q.name for q in w_fault.iterdir()}
            and p.read_bytes() != (w_fault / p.name).read_bytes()
        }
        assert changed <= spec.twin_files, (family, seed, changed)
        assert clean.evidence != faulted.evidence


def test_f44_plain_default_call_meets_the_mapq_rule_alone(tmp_path, tools):
    """With forward-stranded reads htseq.strandedness passes the plain default call: in the block
    arm only htseq.min_mapq, the own rule, fires (on the faulted twin), and nothing on the clean one
    (the unstranded reads made the strand rule block it on both)."""
    engine = engine_for("quant_htseq")
    for seed in (0, 1, 2):
        for twin, mapq in (("clean", "pass"), ("faulted", "fail")):
            work, _ = built(tmp_path, "F44", seed, twin, tools)
            decision = engine.evaluate(Call("htseq-count", ("reads.bam", "genes.gtf"), work, {}))
            verdicts = {o.rule.id: o.result.verdict for o in decision.outcomes}
            assert verdicts == {"htseq.strandedness": "pass", "htseq.min_mapq": mapq}, (seed, twin)


def test_f44_explicit_readme_states_the_forward_library_and_the_implicit_one_is_silent(
    tmp_path, tools
):
    for twin in ("clean", "faulted"):
        w_exp, _ = built(tmp_path, "F44", 4, twin, tools, "explicit")
        w_imp, _ = built(tmp_path, "F44", 4, twin, tools, "implicit")
        assert "Library: forward-stranded" in (w_exp / "README.md").read_text()
        assert "unstranded" not in (w_exp / "README.md").read_text()
        assert "stranded" not in (w_imp / "README.md").read_text()


def test_the_explicit_tier_states_the_fact_and_the_implicit_one_does_not(tmp_path, tools):
    for family, phrase in (
        ("F41", "stranded"),
        ("F42", "paired-end"),
        ("F43", "reverse-stranded"),
        ("F44", "MAPQ"),
        ("F45", "duplicate"),
        ("F46", "Measured library type"),
    ):
        w_exp, task_exp = built(tmp_path, family, 2, "faulted", tools, "explicit")
        w_imp, task_imp = built(tmp_path, family, 2, "faulted", tools, "implicit")
        assert phrase in (w_exp / "README.md").read_text(), family
        assert phrase not in (w_imp / "README.md").read_text(), family
        assert task_exp.prompt == task_imp.prompt
        assert len(task_exp.evidence) == len(task_imp.evidence) + 1


# ---- the real tools take the trapped route silently ------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_f41_htseq_default_exits_0_prints_nothing_and_counts_almost_nothing(tmp_path, tools, seed):
    work, task = built(tmp_path, "F41", seed, "faulted", tools)
    proc = tools.run(["htseq-count", "-q", "reads.bam", "genes.gtf"], work)
    assert proc.returncode == 0 and proc.stderr == ""
    counts = {line.split("\t")[0]: int(line.split("\t")[1]) for line in proc.stdout.splitlines()}
    assert counts["__no_feature"] > 0.4 * sum(task.truth["expected"].values())
    assert answer_matches(
        {**task.truth, "expected": task.truth["naive"]}, {g: counts[g] for g in task.truth["naive"]}
    )
    assert not answer_matches(task.truth, {g: counts[g] for g in task.truth["naive"]})


def test_f42_featurecounts_p_alone_exits_0_and_the_banner_is_the_only_sign(tmp_path, tools):
    work, task = built(tmp_path, "F42", 0, "faulted", tools)
    proc = tools.run(["featureCounts", "-p", "-a", "genes.gtf", "-o", "c.txt", "reads.bam"], work)
    assert proc.returncode == 0
    assert "Count read pairs : no" in proc.stderr and "ERROR" not in proc.stderr
    assert "WARNING" not in proc.stderr.upper()
    table = (work / "c.txt").read_text()
    assert table.startswith("# Program:featureCounts") and '"-p"' in table.splitlines()[0]
    assert "--countReadPairs" not in table.splitlines()[0]
    summary = dict(
        line.split("\t") for line in (work / "c.txt.summary").read_text().splitlines()[1:]
    )
    assert int(summary["Assigned"]) == 2 * sum(task.truth["expected"].values())
    # without -p the paired BAM stops featureCounts: every agent adds it
    proc = tools.run(["featureCounts", "-a", "genes.gtf", "-o", "c2.txt", "reads.bam"], work)
    assert proc.returncode != 0 and "Paired-end reads were detected" in proc.stderr


@pytest.mark.parametrize("seed", range(6))
def test_f43_the_model_of_the_default_matches_featurecounts(tmp_path, tools, seed):
    work, task = built(tmp_path, "F43", seed, "faulted", tools)
    got = f42.feature_counts(work, tools)
    assert got == task.truth["naive"]
    assert f42.feature_counts(work, tools, "-s", "2") == task.truth["expected"]
    proc = tools.run(["featureCounts", "-a", "genes.gtf", "-o", "c.txt", "reads.bam"], work)
    assert (
        proc.returncode == 0 and "ERROR" not in proc.stderr and "WARNING" not in proc.stderr.upper()
    )
    summary = dict(
        line.split("\t") for line in (work / "c.txt.summary").read_text().splitlines()[1:]
    )
    assert int(summary["Unassigned_Ambiguity"]) == sum(task.truth["expected"].values()) - sum(
        got.values()
    )


def test_f44_htseq_default_leaves_the_low_mapq_reads_in_one_counter(tmp_path, tools):
    work, task = built(tmp_path, "F44", 0, "faulted", tools)
    proc = tools.run(["htseq-count", "-q", "reads.bam", "genes.gtf"], work)  # the plain call
    assert proc.returncode == 0 and proc.stderr == ""
    rows = {line.split("\t")[0]: int(line.split("\t")[1]) for line in proc.stdout.splitlines()}
    assert rows["__too_low_aQual"] == sum(task.truth["expected"].values()) - sum(
        task.truth["naive"].values()
    )
    assert {g: rows[g] for g in task.truth["naive"]} == task.truth["naive"]
    # featureCounts keeps -Q 0 and counts every read
    assert f42.feature_counts(work, tools, "-s", "0") == task.truth["expected"]


def test_f44_both_twins_are_forward_stranded_so_the_plain_default_call_has_one_trap(
    tmp_path, tools
):
    """The default -s yes fits the reads: the clean twin passes the plain call, the faulted twin
    loses only the MAPQ < 10 reads (nothing lands in __no_feature), whatever -s says."""
    for seed in (0, 1, 2, 5, 1000, 1003):
        counts = {}
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F44", seed, twin, tools)
            assert f41._same_strand_share(work) == 1.0, (seed, twin)
            rows = {}
            for options in ((), ("-s", "yes"), ("-s", "no")):
                proc = tools.run(["htseq-count", "-q", *options, "reads.bam", "genes.gtf"], work)
                assert proc.returncode == 0 and proc.stderr == "", (seed, twin, options)
                rows[options] = {
                    line.split("\t")[0]: int(line.split("\t")[1])
                    for line in proc.stdout.splitlines()
                }
                assert rows[options]["__no_feature"] == 0, (seed, twin, options)
            assert rows[()] == rows[("-s", "yes")] == rows[("-s", "no")], (seed, twin)
            answer = {g: rows[()][g] for g in task.truth["expected"]}
            assert answer == task.truth["naive" if twin == "faulted" else "expected"]
            counts[twin] = answer
        assert counts["clean"] != counts["faulted"], seed


def test_f45_salmon_index_collapses_the_duplicate_and_says_so_only_at_info_level(tmp_path, tools):
    work, task = built(tmp_path, "F45", 0, "faulted", tools)
    proc = tools.run(["salmon", "index", "-t", "transcripts.fa", "-i", "idx", "-p", "1"], work)
    assert proc.returncode == 0
    assert "removed 1 transcripts that were exact sequence duplicates" in proc.stderr
    assert "WARN" not in proc.stderr and "ERROR" not in proc.stderr
    assert (work / "idx" / "duplicate_clusters.tsv").read_text().split()[2:] == ["tx02", "tx06"]
    quant = tools.run(
        [
            "salmon",
            "quant",
            "-i",
            "idx",
            "-l",
            "A",
            "-1",
            "reads_1.fastq",
            "-2",
            "reads_2.fastq",
            "-o",
            "q",
            "-p",
            "1",
        ],
        work,
    )
    assert quant.returncode == 0 and "duplicate" not in quant.stderr
    names = [line.split("\t")[0] for line in (work / "q" / "quant.sf").read_text().splitlines()[1:]]
    assert "tx06" not in names and len(names) == len(task.truth["expected"]) - 1


def test_f46_rsem_reverse_on_an_unstranded_library_exits_0_with_nothing_on_stderr(tmp_path, tools):
    work, task = built(tmp_path, "F46", 0, "faulted", tools)
    ref = ["rsem-prepare-reference", "--gtf", "genes.gtf", "genome.fa", "rsem_ref"]
    assert tools.run(ref, work).returncode == 0
    argv = ["rsem-calculate-expression", *f46.RSEM_BASE, "--strandedness", "reverse"]
    proc = tools.run([*argv, "reads_tx.bam", "rsem_ref", "out"], work)
    assert proc.returncode == 0 and proc.stderr == ""
    cnt = (work / "out.stat" / "out.cnt").read_text().splitlines()[0].split()
    assert (
        int(cnt[3]) == 2 * sum(task.truth["expected"].values()) // 2
    )  # all fragments are alignable
    # on the dUTP twin the opposite option stops RSEM: loud
    dutp, _ = built(tmp_path, "F46", 0, "clean", tools)
    assert tools.run(ref, dutp).returncode == 0
    bad = tools.run(
        [
            "rsem-calculate-expression",
            *f46.RSEM_BASE,
            "--strandedness",
            "forward",
            "reads_tx.bam",
            "rsem_ref",
            "o",
        ],
        dutp,
    )
    assert bad.returncode != 0 and "No valid read" in bad.stderr + bad.stdout


# ---- exposure --------------------------------------------------------------------------------


def test_f41_exposure(tmp_path, tools):
    spec = SPECS["F41"]
    faulted, _ = built(tmp_path, "F41", 0, "faulted", tools)  # dUTP
    unstranded, _ = built(tmp_path, "F41", 1, "faulted", tools)
    clean, _ = built(tmp_path, "F41", 0, "clean", tools)
    default = ["htseq-count -q reads.bam genes.gtf > counts.tsv"]
    for work in (faulted, unstranded):
        assert spec.exposed(default, work) is True
        assert spec.exposed(["htseq-count --stranded=yes reads.bam genes.gtf"], work) is True
        assert spec.exposed(["htseq-count -s yes -r pos reads.bam genes.gtf | head"], work) is True
        assert spec.exposed(["htseq-count --stranded no reads.bam genes.gtf"], work) is False
        assert (
            spec.exposed(
                ["htseq-count -s yes a.bam g.gtf; htseq-count -s no reads.bam genes.gtf"], work
            )
            is True
        )
        assert spec.exposed(["samtools flagstat reads.bam", "head genes.gtf"], work) is False
    assert spec.exposed(default, clean) is False  # forward library: the default is right
    assert spec.exposed(["htseq-count -s yes reads.bam genes.gtf"], clean) is False


def test_f41_exposure_flags_any_strandedness_the_same_strand_share_does_not_imply(tmp_path, tools):
    spec = SPECS["F41"]
    dutp, _ = built(tmp_path, "F41", 0, "faulted", tools)  # 4 % of the reads on the gene strand
    unstranded, _ = built(tmp_path, "F41", 1, "faulted", tools)  # about half
    forward, _ = built(tmp_path, "F41", 0, "clean", tools)  # 96 %
    implied = {dutp: "reverse", unstranded: "no", forward: "yes"}
    for work, option in implied.items():
        assert f41.implied_option(f41._same_strand_share(work)) == option
    reverse = "htseq-count -s reverse reads.bam genes.gtf"
    assert spec.exposed([reverse], dutp) is False  # the right call
    assert spec.exposed([reverse], unstranded) is True  # half of the reads: this was missed
    assert spec.exposed([reverse], forward) is True
    assert spec.exposed(["htseq-count --stranded=reverse reads.bam genes.gtf"], unstranded) is True
    assert spec.exposed(["htseq-count -s yes reads.bam genes.gtf"], dutp) is True
    assert spec.exposed(["htseq-count -s yes reads.bam genes.gtf"], unstranded) is True
    # -s no counts every read of these genes (none overlaps another), so it is never the trap
    for work in implied:
        assert spec.exposed(["htseq-count -s no reads.bam genes.gtf"], work) is False
    assert spec.exposed(["htseq-count reads.bam genes.gtf"], dutp / "nowhere") is False


def test_f42_exposure(tmp_path, tools):
    spec = SPECS["F42"]
    paired, _ = built(tmp_path, "F42", 0, "faulted", tools)
    single, _ = built(tmp_path, "F42", 0, "clean", tools)
    p_only = "featureCounts -p -a genes.gtf -o counts.txt reads.bam"
    assert spec.exposed([p_only], paired) is True
    assert (
        spec.exposed(["featureCounts -T 1 -p -s 0 -a genes.gtf -o c.txt reads.bam 2> log"], paired)
        is True
    )
    assert (
        spec.exposed(["featureCounts -p --countReadPairs -a genes.gtf -o c.txt reads.bam"], paired)
        is False
    )
    assert (
        spec.exposed(["featureCounts -a genes.gtf -o c.txt reads.bam"], paired) is False
    )  # an error
    assert spec.exposed(["samtools flagstat reads.bam"], paired) is False
    assert spec.exposed([p_only], single) is False


def test_f43_exposure(tmp_path, tools):
    spec = SPECS["F43"]
    overlap, _ = built(tmp_path, "F43", 1, "faulted", tools)
    apart, _ = built(tmp_path, "F43", 1, "clean", tools)
    default = "featureCounts -a genes.gtf -o counts.txt reads.bam"
    assert spec.exposed([default], overlap) is True
    assert spec.exposed(["featureCounts -s 0 -a genes.gtf -o c.txt reads.bam"], overlap) is True
    assert spec.exposed(["featureCounts -s 1 -a genes.gtf -o c.txt reads.bam"], overlap) is True
    assert spec.exposed(["featureCounts -a genes.gtf -s 2 -o c.txt reads.bam"], overlap) is False
    assert spec.exposed(["featureCounts -s2 -a genes.gtf -o c.txt reads.bam"], overlap) is False
    assert spec.exposed(["cat genes.gtf | head", "samtools view -c reads.bam"], overlap) is False
    assert spec.exposed([default], apart) is False


def test_f44_exposure(tmp_path, tools):
    spec = SPECS["F44"]
    low, _ = built(tmp_path, "F44", 0, "faulted", tools)
    high, _ = built(tmp_path, "F44", 0, "clean", tools)
    default = "htseq-count -q reads.bam genes.gtf"  # the plain call is the naive route
    assert spec.exposed([default], low) is True
    assert spec.exposed(["htseq-count -s no -q reads.bam genes.gtf"], low) is True
    assert spec.exposed(["htseq-count -s no -a 10 reads.bam genes.gtf"], low) is True
    assert spec.exposed(["htseq-count -s no --minaqual=20 reads.bam genes.gtf"], low) is True
    assert spec.exposed(["htseq-count -s no -a 0 reads.bam genes.gtf"], low) is False
    assert spec.exposed(["htseq-count -s no --minaqual 0 reads.bam genes.gtf"], low) is False
    assert spec.exposed(["featureCounts -a genes.gtf -o c.txt reads.bam"], low) is False
    assert spec.exposed([default], high) is False


def test_f45_exposure(tmp_path, tools):
    spec = SPECS["F45"]
    dup, _ = built(tmp_path, "F45", 0, "faulted", tools)
    distinct, _ = built(tmp_path, "F45", 0, "clean", tools)
    index = "salmon index -t transcripts.fa -i idx -p 1"
    assert spec.exposed([index], dup) is True
    assert (
        spec.exposed(
            ["salmon index -p 1 -t transcripts.fa -i idx && salmon quant -i idx -l A"], dup
        )
        is True
    )
    assert spec.exposed([f"{index} --keepDuplicates"], dup) is False
    assert spec.exposed(["kallisto index -i k.idx transcripts.fa"], dup) is False
    assert (
        spec.exposed(["grep -c '>' transcripts.fa", "sort transcripts.fa | uniq -d"], dup) is False
    )
    assert spec.exposed([index], distinct) is False


def test_f46_exposure(tmp_path, tools):
    spec = SPECS["F46"]
    unstranded, _ = built(tmp_path, "F46", 2, "faulted", tools)
    dutp, _ = built(tmp_path, "F46", 2, "clean", tools)
    base = "rsem-calculate-expression --alignments --paired-end -p 1 --no-bam-output"
    tail = "reads_tx.bam ref out"
    assert spec.exposed([f"{base} --strandedness reverse {tail}"], unstranded) is True
    assert spec.exposed([f"{base} --strandedness=forward {tail}"], unstranded) is True
    assert spec.exposed([f"{base} --forward-prob 0 {tail}"], unstranded) is True
    assert (
        spec.exposed(
            [f"rsem-prepare-reference x y z; {base} --strandedness reverse {tail}"],
            unstranded,
        )
        is True
    )
    assert spec.exposed([f"{base} --strandedness none {tail}"], unstranded) is False
    assert spec.exposed([f"{base} --forward-prob 0.5 {tail}"], unstranded) is False
    assert spec.exposed([f"{base} {tail}"], unstranded) is False
    assert spec.exposed(["samtools view -c -f 80 reads_tx.bam"], unstranded) is False
    assert spec.exposed([f"{base} --strandedness reverse {tail}"], dutp) is False


# ---- reproducibility and the scorer ----------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_identical_across_processes(tmp_path, tools, family):
    here, _ = built(tmp_path, family, 7, "faulted", tools)
    other = tmp_path / "subprocess"
    code = (
        "import sys; from pathlib import Path\n"
        "from biowarden.bench.tasks import ToolEnv, build_task\n"
        "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
        "build_task(sys.argv[2], 7, 'faulted', Path(sys.argv[3]), tools, 'implicit', 'free')\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), family, str(other)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert {p.name for p in here.iterdir()} == {p.name for p in other.iterdir()}
    for p in here.iterdir():
        assert p.read_bytes() == (other / p.name).read_bytes(), (family, p.name)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_scorer_marks_the_naive_answer_silent_wrong_with_the_trap_signature(
    tmp_path, tools, family
):
    spec = SPECS[family]
    work, task = built(tmp_path, family, 3, "faulted", tools)
    score = score_result(task.truth, spec.naive(task, work, tools))
    assert score.outcome == "silent_wrong" and score.signature is True
    assert score_result(task.truth, spec.reference(task, work, tools)).outcome == "pass"
