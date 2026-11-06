"""Family-specific checks for F21-F24 (locus task_statement: the sentence disagrees with the data).

The generic gates live in test_extra_families.py. Here, on seeds 0-29 and both tiers: the trap
changes the answer (pure Python from the plans), the twins hold the same biology and differ only in
the declared files, the evidence is where the family says it is and not in the head of the data,
the exposure functions see realistic trapped commands and not the safe ones (on a faulted workspace,
and not on a clean one), the stated route really produces the recorded wrong answer with the real
tools (silently), and builds are byte-reproducible across processes.

Seeds alternate the claim: even seeds claim BED / Phred+33 / reverse-stranded, odd seeds claim
1-based / Phred+64 / forward-stranded (F24 always claims TPM).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f21_stmt_base as f21
from biowarden.bench.tasks.extra import f22_stmt_encoding as f22
from biowarden.bench.tasks.extra import f23_stmt_strand as f23
from biowarden.bench.tasks.extra import f24_stmt_units as f24
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.tools import strip_loader_noise

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
FAMILIES = ("F21", "F22", "F23", "F24")
SPECS = load()
PLAIN_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F21-F24 are not registered"
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def built(tmp_path, family, seed, twin, tools, tier="explicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def build_in(tmp_path, family, seed, twin, tier="explicit", route="free"):
    """Build a family that needs no tools (F22, F24)."""
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    work.mkdir(parents=True, exist_ok=True)
    return work, SPECS[family].build(seed, twin, work, None, tier, route)


def shell(command, work, env):
    """Run a shell command; (returncode, stdout, stderr without dynamic-loader noise)."""
    proc = subprocess.run(
        ["bash", "-c", command], cwd=work, env=env, capture_output=True, text=True, check=False
    )
    return proc.returncode, proc.stdout, strip_loader_noise(proc.stderr)


def fasta_sequences(stdout: str) -> list[str]:
    seqs: list[str] = []
    for line in stdout.splitlines():
        if line.startswith(">"):
            seqs.append("")
        elif seqs:
            seqs[-1] += line.strip().upper()
    return seqs


def commands_of(item) -> list[str]:
    return item if isinstance(item, list) else [item]


@pytest.mark.parametrize(
    ("family", "slot"),
    [("F21", "index_base"), ("F22", "units"), ("F23", "strand"), ("F24", "units")],
)
def test_locus_slot_and_source_are_declared(family, slot):
    spec = SPECS[family]
    assert spec.locus == "task_statement" and spec.slot == slot and spec.source
    assert "README.md" in spec.twin_files and len(spec.twin_files) <= 4


# ---- the trap, family by family, on seeds 0-29 (pure Python from the plans) --------------------


def test_f21_trap_is_observable_on_every_seed_and_every_cds_is_affected():
    claims = set()
    for seed in SEEDS:
        plan = f21.make_plan(seed)
        claims.add(plan.stated)
        c_exp, c_naive = f21.truth_answers(plan, "clean")
        f_exp, f_naive = f21.truth_answers(plan, "faulted")
        assert c_exp == f_exp and c_naive == c_exp, seed
        assert len(plan.cds) == 18 and all(a != b for a, b in zip(f_exp, f_naive, strict=True))
        assert plan.stated == (f21.BED if seed % 2 == 0 else f21.ONE), seed
        # the evidence: complete ORFs under the real reading, none starts with ATG under the stated
        for seq in f_exp:
            codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
            assert codons[0] == "ATG" and codons[-1] in f21.STOPS and len(seq) % 3 == 0
            assert not any(c in f21.STOPS for c in codons[:-1]), seed
        assert not any(s.startswith("ATG") for s in f_naive), seed
        assert {len(s) % 3 for s in f_naive} == {2 if plan.stated == f21.BED else 1}, seed
        # the twins differ by exactly the start shift; no row starts at 0 (so the tools stay silent)
        shift = 1 if plan.stated == f21.BED else -1
        for clean, faulted in zip(plan.rows("clean"), plan.rows("faulted"), strict=True):
            assert (faulted[1] - clean[1], faulted[2], faulted[3]) == (shift, clean[2], clean[3])
        assert all(min(r[1], r[2] - r[1]) >= 1 for t in ("clean", "faulted") for r in plan.rows(t))
        assert sum(len(s) for s in plan.genome.values()) <= 5000, seed
    assert claims == {f21.BED, f21.ONE}


def test_f22_trap_is_observable_on_every_seed_and_the_head_hides_it(tmp_path):
    for seed in SEEDS:
        plan = f22.make_plan(seed)
        n = len(plan.seqs)
        c_exp, c_naive = f22.truth_answers(plan, "clean")
        f_exp, f_naive = f22.truth_answers(plan, "faulted")
        assert c_exp == f_exp == c_naive and 40 <= n <= 56, seed
        assert 0.25 * n <= f_exp <= 0.75 * n, seed
        assert f_naive == (n if plan.stated == 33 else 0), seed  # every read passes / none does
        assert abs(f_naive - f_exp) >= 10, seed
        assert plan.stated == (33 if seed % 2 == 0 else 64), seed
        assert all(abs(sum(q) - f22.THRESHOLD * len(q)) >= f22.MARGIN for q in plan.quals), seed
        # the same scores sit in both twins; only the offset of the characters differs
        clean = f22.fastq_text(plan, plan.offset("clean")).splitlines()
        faulted = f22.fastq_text(plan, plan.offset("faulted")).splitlines()
        for lc, lf in zip(clean[3::4], faulted[3::4], strict=True):
            assert [ord(c) - plan.offset("clean") for c in lc] == [
                ord(c) - plan.offset("faulted") for c in lf
            ]
        assert clean[0::4] == faulted[0::4] and clean[1::4] == faulted[1::4]
        assert not any(q[0] in "@+" for q in clean[3::4] + faulted[3::4])
        # the faulted head (the first 10 reads = 40 lines) holds only characters valid in both
        head = "".join(faulted[3 : 4 * f22.HEAD_READS : 4])
        assert head and all(59 <= ord(c) <= 74 for c in head), seed
        rest = "".join(faulted[4 * f22.HEAD_READS + 3 :: 4])
        real = plan.offset("faulted")
        decisive = [c for c in rest if (ord(c) > 74 if real == 64 else ord(c) < 59)]
        assert len(decisive) >= 100, seed
        # ... while the whole file is impossible under the stated encoding
        codes = [ord(c) for c in "".join(faulted[3::4])]
        if plan.stated == 33:
            assert min(codes) >= 64 and max(codes) > 74
        else:
            assert min(codes) < 59 and max(codes) <= 74
        # F22 needs no tool: the same builder runs without them
        work = tmp_path / f"f22-{seed}"
        work.mkdir()
        task = SPECS["F22"].build(seed, "faulted", work, None, "implicit")
        assert task.truth["naive"] == f_naive and task.truth["expected"] == f_exp


def test_f23_trap_is_observable_on_every_seed_and_every_gene_is_affected():
    claims = set()
    for seed in SEEDS:
        plan = f23.make_plan(seed)
        claims.add(plan.stated)
        c_exp, c_naive = f23.truth_answers(plan, "clean")
        f_exp, f_naive = f23.truth_answers(plan, "faulted")
        assert c_exp == f_exp and c_naive == c_exp, seed
        assert list(f_exp) == [g.id for g in plan.genes] and len(plan.genes) == 8
        assert all(f_exp[g] >= 12 and f_naive[g] <= 3 for g in f_exp), seed  # all 8 genes differ
        assert 3 <= sum(g.strand == "-" for g in plan.genes) <= 5, seed
        assert plan.stated == (f23.REVERSE if seed % 2 == 0 else f23.FORWARD), seed
        positions = [(r.chrom, r.pos) for r in plan.reads]
        assert len(positions) == len(set(positions)) < 400, seed
        for r in plan.reads:
            lo, hi = r.pos - 1, r.pos - 1 + f23.READ_LEN
            assert 0 <= lo and hi <= plan.contigs[r.chrom], (seed, r)
            near = [
                g
                for g in plan.genes
                if g.chrom == r.chrom and lo < g.end0 + 100 and hi > g.start0 - 100
            ]
            if r.kind == "inter":  # at least 100 bp from every gene: no overlap rule can matter
                assert not near, (seed, r)
            else:
                gene = plan.genes[r.gene]
                assert near == [gene] and gene.start0 <= lo and hi <= gene.end0, (seed, r)
        # the stated count and the right count are the two strand counts of the same BAM
        real = plan.real("faulted")
        assert plan.counts(real, real) == f_exp and plan.counts(real, plan.stated) == f_naive
    assert claims == {f23.REVERSE, f23.FORWARD}


def test_f23_bam_flags_carry_the_real_library_and_the_head_shows_only_intergenic_reads(
    tmp_path, tools
):
    for seed in SEEDS:
        plan = f23.make_plan(seed)
        for twin in ("clean", "faulted"):
            work, _ = built(tmp_path, "F23", seed, twin, tools, "implicit")
            assert f23.real_library(work) == plan.real(twin), (seed, twin)
            _, out, _ = shell("samtools view reads.bam | head -6", work, tools.process_env())
            head = [line.split("\t") for line in out.splitlines()]
            assert len(head) == 6 and all(r[2] == "chr1" and int(r[3]) <= 150 for r in head)
    # the stdlib BAM reader used by the exposure check agrees with samtools on every record
    work, _ = built(tmp_path, "F23", 4, "faulted", tools)
    _, out, _ = shell("samtools view reads.bam", work, tools.process_env())
    rows = [line.split("\t") for line in out.splitlines()]
    assert f23.bam_reads(work / "reads.bam") == [
        (r[2], int(r[3]) - 1, len(r[9]), int(r[1])) for r in rows
    ]


def test_f24_trap_is_observable_on_every_seed(tmp_path):
    too_high = set()
    for seed in SEEDS:
        plan = f24.make_plan(seed)
        expected, naive = f24.truth_counts(plan)
        assert abs(expected - naive) >= f24.MIN_TRAP, seed
        too_high.add(naive > expected)
        assert 60 <= len(plan.genes) <= 90 and 0.35 <= expected / len(plan.genes) <= 0.85, seed
        for s in range(f24.N_SAMPLES):
            assert sum(row[s] for row in plan.tpm) == f24.PER_MILLION, seed
        means = [plan.mean_tpm(g) for g in range(len(plan.genes))]
        assert min(abs(m - f24.CUTOFF) for m in means) >= f24.MARGIN, seed
        tables = {}
        for twin in ("clean", "faulted"):
            work = tmp_path / f"{seed}-{twin}"
            work.mkdir()
            task = SPECS["F24"].build(seed, twin, work, None, "implicit")
            tables[twin] = f24.read_table(work / "expr.tsv")
            assert task.truth["expected"] == expected
            assert task.truth["naive"] == (expected if twin == "clean" else naive)
        assert f24.is_tpm(tables["clean"]) and not f24.is_tpm(tables["faulted"])
        assert all(abs(t - 1e6) < 0.5 for t in f24.column_sums(tables["clean"])), seed
        assert all(abs(t / 1e6 - 1) >= 0.3 for t in f24.column_sums(tables["faulted"])), seed
        # normalising the FPKM columns to a million gives the TPM table back (rounding only)
        sums = f24.column_sums(tables["faulted"])
        for c_row, f_row in zip(tables["clean"], tables["faulted"], strict=True):
            for s, (c, f) in enumerate(zip(c_row, f_row, strict=True)):
                assert abs(f / sums[s] * 1e6 - c) <= 0.01 + 1e-6 * c, seed
        assert f24._count(tables["faulted"], True) == expected
        assert f24._count(tables["faulted"], False) == naive
    assert too_high == {
        True,
        False,
    }  # the stated count is too high on some seeds, too low on others


# ---- the twins hold the same biology and differ only where the family says ---------------------


def lines_of(work: Path, name: str) -> list[str]:
    return (work / name).read_text().splitlines()


def test_twins_share_everything_but_the_convention(tmp_path, tools):
    for seed in (0, 1, 7):
        wc, _ = built(tmp_path, "F21", seed, "clean", tools, "implicit")
        wf, _ = built(tmp_path, "F21", seed, "faulted", tools, "implicit")
        for name in ("ref.fa", "ref.fa.fai", "README.md"):
            assert (wc / name).read_bytes() == (wf / name).read_bytes(), name
        assert (wc / "cds.tsv").read_bytes() != (wf / "cds.tsv").read_bytes()
        wc, _ = built(tmp_path, "F22", seed, "clean", tools, "implicit")
        wf, _ = built(tmp_path, "F22", seed, "faulted", tools, "implicit")
        for part in (0, 1):
            assert lines_of(wc, "reads.fastq")[part::4] == lines_of(wf, "reads.fastq")[part::4]
        assert lines_of(wc, "reads.fastq")[3::4] != lines_of(wf, "reads.fastq")[3::4]
        assert (wc / "README.md").read_bytes() == (wf / "README.md").read_bytes()
        wc, _ = built(tmp_path, "F23", seed, "clean", tools, "implicit")
        wf, _ = built(tmp_path, "F23", seed, "faulted", tools, "implicit")
        assert (wc / "genes.bed").read_bytes() == (wf / "genes.bed").read_bytes()
        assert (wc / "README.md").read_bytes() == (wf / "README.md").read_bytes()
        assert (wc / "reads.bam").read_bytes() != (wf / "reads.bam").read_bytes()
        wc, _ = built(tmp_path, "F24", seed, "clean", tools, "implicit")
        wf, _ = built(tmp_path, "F24", seed, "faulted", tools, "implicit")
        genes_c = [ln.split("\t")[0] for ln in lines_of(wc, "expr.tsv")]
        genes_f = [ln.split("\t")[0] for ln in lines_of(wf, "expr.tsv")]
        assert genes_c == genes_f and lines_of(wc, "expr.tsv") != lines_of(wf, "expr.tsv")
        assert (wc / "README.md").read_bytes() == (wf / "README.md").read_bytes()


# ---- tiers and routes: where the convention is stated ------------------------------------------

README_SENTENCE = {
    "F21": {f21.BED: "0-based with the end excluded", f21.ONE: "1-based with both ends included"},
    "F22": {33: "Phred+33", 64: "Phred+64"},
    "F23": {f23.REVERSE: "reverse-stranded (dUTP)", f23.FORWARD: "forward-stranded"},
    "F24": {"clean": "Values are TPM", "faulted": "Values are FPKM"},
}
PLANS = {"F21": f21.make_plan, "F22": f22.make_plan, "F23": f23.make_plan}


def real_key(family, seed, twin):
    if family == "F24":
        return twin
    plan = PLANS[family](seed)
    return plan.offset(twin) if family == "F22" else plan.real(twin)


def test_explicit_readme_states_the_real_convention_and_implicit_does_not(tmp_path, tools):
    for family in FAMILIES:
        for seed in (2, 3):
            for twin in ("clean", "faulted"):
                work_e, explicit = built(tmp_path, family, seed, twin, tools, "explicit")
                work_i, implicit = built(tmp_path, family, seed, twin, tools, "implicit")
                key = real_key(family, seed, twin)
                text_e = (work_e / "README.md").read_text()
                text_i = (work_i / "README.md").read_text()
                assert README_SENTENCE[family][key] in text_e, (family, seed, twin)
                others = [v for k, v in README_SENTENCE[family].items() if k != key]
                assert not any(o in text_e for o in others), (family, seed, twin)
                assert not any(v in text_i for v in README_SENTENCE[family].values())
                assert explicit.truth == implicit.truth and explicit.evidence != implicit.evidence
                assert implicit.evidence or twin == "clean"


def test_f24_readme_says_no_gene_was_filtered_out_in_every_twin_and_tier(tmp_path):
    """Column sums below a million must not read as the sums of a gene panel."""
    for seed in (0, 3, 1020, 1021, 1022, 1023, 1024):
        readmes = {}
        for twin in ("clean", "faulted"):
            for tier in ("explicit", "implicit"):
                work, task = build_in(tmp_path, "F24", seed, twin, tier)
                readmes[twin, tier] = (work / "README.md").read_text()
                assert "One row per annotated gene, none filtered out." in readmes[twin, tier]
                assert not re.search(r"panel|subset|selected", readmes[twin, tier]), (seed, twin)
                if twin == "faulted":
                    assert any("no gene filtered out" in line for line in task.evidence), seed
        # the clause is the same in both twins: they differ only by the unit the explicit tier names
        assert readmes["clean", "implicit"] == readmes["faulted", "implicit"]
        assert readmes["clean", "explicit"].replace("TPM", "FPKM") == readmes["faulted", "explicit"]
        assert readmes["clean", "explicit"].startswith(readmes["clean", "implicit"].rstrip("\n"))


CLAIMS = {  # what the sentence says, by seed parity
    "F21": ("BED-style (0-based start, end exclusive)", "1-based, with both ends included"),
    "F22": ("Phred+33 encoded", "Phred+64 encoded"),
    "F23": ("reverse-stranded (dUTP)", "forward-stranded"),
    "F24": ("the table holds TPM values", "the table holds TPM values"),
}
DICTATED = {
    "F21": ("bedtools getfasta", "samtools faidx"),
    "F22": ("awk", "awk"),
    "F23": ("bedtools coverage", "bedtools coverage"),
    "F24": ("awk", "awk"),
}


def test_the_sentence_makes_the_claim_in_both_twins_and_forced_names_the_tool(tmp_path, tools):
    for family in FAMILIES:
        for seed in (0, 1):
            free_prompts = []
            for twin in ("clean", "faulted"):
                _, free = built(tmp_path, family, seed, twin, tools, "explicit", "free")
                _, forced = built(tmp_path, family, seed, twin, tools, "explicit", "forced")
                assert CLAIMS[family][seed] in free.prompt and CLAIMS[family][seed] in forced.prompt
                assert DICTATED[family][seed] in forced.prompt
                assert DICTATED[family][seed] not in free.prompt
                free_prompts.append(free.prompt)
            assert free_prompts[0] == free_prompts[1]


# ---- exposure ----------------------------------------------------------------------------------
# F21: seed 0 claims BED, so its faulted file is 1-based closed; seed 1 claims 1-based, so its
# faulted file is BED. Trapped = a command reading cds.tsv under the claim; safe = under the data's
# convention, or a command that also looks at start codons.

F21_BED_READING = [  # trapped on seed 0 (file is 1-based), right on seed 1 (file is BED)
    "bedtools getfasta -fi ref.fa -bed cds.tsv -fo cds.fa",
    "bedtools getfasta -fi ref.fa -bed cds.tsv -tab | cut -f2",
    'awk \'{print $1":"$2+1"-"$3}\' cds.tsv | xargs samtools faidx ref.fa',
    'samtools faidx ref.fa $(awk \'{print $1":"($2+1)"-"$3}\' cds.tsv)',
    "python3 -c \"s=open('ref.fa').read(); print([s[int(l.split()[1]):int(l.split()[2])] "
    "for l in open('cds.tsv')])\"",
    "cat > extract.py <<'EOF'\nfor line in open('cds.tsv'):\n    c, a, b, n = line.split()\n"
    "    print(seq[int(a):int(b)])\nEOF",
    "awk -v OFS='\\t' '{print $1,$2,$3}' cds.tsv > cds.bed",
    'awk \'{print $1":"$2+1"-"$3}\' cds.tsv > regions.txt',
]
F21_ONE_READING = [  # trapped on seed 1 (file is BED), right on seed 0 (file is 1-based)
    'samtools faidx ref.fa $(awk \'{print $1":"$2"-"$3}\' cds.tsv)',
    'awk \'{print $1":"$2"-"$3}\' cds.tsv | xargs samtools faidx ref.fa',
    "awk -v OFS='\\t' '{print $1,$2-1,$3}' cds.tsv | bedtools getfasta -fi ref.fa -bed stdin",
    "python3 -c \"s=open('ref.fa').read(); print([s[int(l.split()[1])-1:int(l.split()[2])] "
    "for l in open('cds.tsv')])\"",
    'awk \'{print $1":"$2"-"$3}\' cds.tsv > regions.txt',
    "awk -v OFS='\\t' '{print $1,$2-1,$3}' cds.tsv > cds.bed",
]
F21_CHECKS_AND_BYSTANDERS = [  # not an extraction, or an extraction that looks at start codons
    "head cds.tsv; ls -la; cat README.md",
    "wc -l cds.tsv; cut -f1 cds.tsv | sort | uniq -c",
    "awk '{print $1,$2,$3}' cds.tsv | head",
    "bedtools getfasta -fi ref.fa -bed cds.tsv -tab | cut -f2 | cut -c1-3 | sort | uniq -c",
    "bedtools getfasta -fi ref.fa -bed cds.tsv -tab | cut -f2 | grep -c '^ATG'",
    'samtools faidx ref.fa $(awk \'{print $1":"$2"-"$3}\' cds.tsv) | grep -c ATG',
    "bedtools getfasta -fi ref.fa -bed cds.bed -fo cds.fa",
    "samtools faidx ref.fa chr1:1-60",
]


def test_f21_exposure_on_both_claims(tmp_path, tools):
    work0, _ = built(tmp_path, "F21", 0, "faulted", tools)
    work1, _ = built(tmp_path, "F21", 1, "faulted", tools)
    assert f21.real_convention(work0) == f21.ONE and f21.real_convention(work1) == f21.BED
    expose = SPECS["F21"].exposed
    for command in F21_BED_READING:
        assert expose([command], work0) is True, command
        assert expose([command], work1) is False, command
    for command in F21_ONE_READING:
        assert expose([command], work1) is True, command
        assert expose([command], work0) is False, command
    for command in F21_CHECKS_AND_BYSTANDERS:
        assert expose([command], work0) is False and expose([command], work1) is False, command


def test_f21_exposure_is_per_command_and_follows_the_data_not_the_claim(tmp_path, tools):
    work, _ = built(tmp_path, "F21", 0, "faulted", tools)
    safe, trapped = F21_ONE_READING[0], F21_BED_READING[0]
    assert SPECS["F21"].exposed([safe, trapped], work) is True  # tried the claim, then corrected
    assert SPECS["F21"].exposed([trapped, safe], work) is True
    assert SPECS["F21"].exposed([safe], work) is False
    assert SPECS["F21"].exposed([], work) is False
    clean, _ = built(tmp_path, "F21", 0, "clean", tools)  # here the claim (BED) is right
    assert SPECS["F21"].exposed([trapped], clean) is False
    assert SPECS["F21"].exposed([safe], clean) is True


def heredoc(*lines):
    """A python script written to the interpreter's stdin, as an agent does."""
    return "python3 - <<'EOF'\n" + "\n".join(lines) + "\nEOF"


ROWS = ("import subprocess", "for line in open('cds.tsv'):", "    c, s, e, n = line.split()")
F21_ARGV_ONE_READING = [  # samtools faidx as an argument list: region strings read 1-based
    heredoc(*ROWS, "    subprocess.run(['samtools', 'faidx', 'ref.fa', f'{c}:{s}-{e}'])"),
    "python3 -c \"import subprocess; [subprocess.check_output(['samtools','faidx','ref.fa',"
    "'%s:%s-%s' % (c, s, e)]) for c, s, e, n in map(str.split, open('cds.tsv'))]\"",
    "python3 -c \"import subprocess; r = [c+':'+s+'-'+e for c, s, e, n in map(str.split, "
    "open('cds.tsv'))]; subprocess.run(['samtools', 'faidx', 'ref.fa'] + r)\"",
    'python3 -c "import subprocess; subprocess.run([\\"samtools\\", \\"faidx\\", \\"ref.fa\\", '
    "'chr1:5-50']); open('cds.tsv')\"",
    heredoc(  # an argument list over several lines
        "import subprocess",
        "rows = [l.split() for l in open('cds.tsv')]",
        "subprocess.run(",
        "    ['samtools', 'faidx', 'ref.fa',",
        "     f'{rows[0][0]}:{rows[0][1]}-{rows[0][2]}'],",
        "    capture_output=True)",
    ),
    heredoc(  # the length of the region (end - start + 1) is not a shifted start
        *ROWS,
        "    seq = subprocess.run(['samtools', 'faidx', 'ref.fa', f'{c}:{s}-{e}'])",
        "    assert len(seq.stdout) >= int(e) - int(s) + 1",
    ),
]
F21_ARGV_BED_READING = [  # bedtools getfasta as an argument list, or faidx fed the start plus one
    heredoc(
        "import subprocess",
        "subprocess.run(['bedtools', 'getfasta', '-fi', 'ref.fa', '-bed', 'cds.tsv'])",
    ),
    'python3 -c "import subprocess; subprocess.run([\\"bedtools\\", \\"getfasta\\", \\"-fi\\", '
    '\\"ref.fa\\", \\"-bed\\", \\"cds.tsv\\"])"',
    heredoc(*ROWS, "    subprocess.run(['samtools', 'faidx', 'ref.fa', f'{c}:{int(s) + 1}-{e}'])"),
]


def test_f21_exposure_reads_a_tool_called_with_an_argument_list_like_the_command_line(
    tmp_path, tools
):
    """The brackets of ['samtools', 'faidx', ...] are not a slice: it is the 1-based reading."""
    work0, _ = built(tmp_path, "F21", 0, "faulted", tools)  # claims BED; the file is 1-based
    work1, _ = built(tmp_path, "F21", 1, "faulted", tools)  # claims 1-based; the file is BED
    clean1, _ = built(tmp_path, "F21", 1, "clean", tools)  # claims 1-based and the file is
    expose = SPECS["F21"].exposed
    for command in F21_ARGV_ONE_READING:
        reads_cds = "cds.tsv" in command  # the one without it is not a computation on the file
        assert f21.reading_of(command) == (f21.ONE if reads_cds else None), command
        assert expose([command], work1) is reads_cds, command
        assert expose([command], work0) is False, command
        assert expose([command], clean1) is False, command  # a right reading is no exposure
    for command in F21_ARGV_BED_READING:
        assert f21.reading_of(command) == f21.BED, command
        assert expose([command], work0) is True, command
        assert expose([command], work1) is False, command
        assert expose([command], clean1) is True, command  # BED rows read as BED, file is 1-based
    # the same tools on the command line say the same, so an agent that tries both is exposed once
    assert expose([F21_ARGV_ONE_READING[0], F21_BED_READING[0]], work0) is True


def test_f21_length_arithmetic_is_not_a_shifted_coordinate():
    """`end - start + 1` is the length of a region; only a start moved by one is a reading."""
    region = 'samtools faidx ref.fa $(awk \'{print $1":"$2"-"$3}\' cds.tsv)'
    for length in ("$3-$2+1", "$3 - $2 + 1", "int($3)-int($2)+1", "($3 - $2 + 1)"):
        command = f"{region} | awk '{{n += {length}}} END{{print n}}'"
        assert f21.reading_of(command) == f21.ONE, command
    assert f21.reading_of("awk '{print $3-$2+1}' cds.tsv") is None
    shifted = 'samtools faidx ref.fa $(awk \'{print $1":"$2+1"-"$3}\' cds.tsv)'
    assert f21.reading_of(shifted) == f21.BED
    assert f21.reading_of(f"{shifted} | awk '{{n += $3-$2+1}}'") == f21.BED


F22_AWK = (
    'awk \'BEGIN{for(i=33;i<127;i++)o[sprintf("%%c",i)]=i} NR%%4==0{s=0;'
    "for(j=1;j<=length($0);j++)s+=o[substr($0,j,1)]-%d; if(s/length($0)>=25)n++} "
    "END{print n+0}' reads.fastq"
)
F22_PLUS_33 = [  # trapped on seed 0 (file is +64), right on seed 1 (file is +33)
    F22_AWK % 33,
    'python3 -c "q=[sum(ord(c)-33 for c in l.strip())/50 for l in '
    "open('reads.fastq').read().split('\\n')[3::4] if l]; print(sum(m>=25 for m in q))\"",
    "samtools import reads.fastq -o reads.bam",
    ["cat > mean.py <<'EOF'\nOFFSET = 33\nEOF", "python3 mean.py reads.fastq"],
    "awk 'BEGIN{for(i=33;i<127;i++)ord[sprintf(\"%c\",i)]=i-33} NR%4==0{n++}' reads.fastq",
    "python3 -c \"print(sum(ord(c)-ord('!') for c in open('reads.fastq').read()))\"",
]
F22_PLUS_64 = [  # trapped on seed 1 (file is +33), right on seed 0 (file is +64)
    F22_AWK % 64,
    "python3 -c \"q=[ord(c)-64 for l in open('reads.fastq') for c in l.strip()]\"",
    ["cat > mean.py <<'EOF'\nOFFSET = 64\nEOF", "python3 mean.py reads.fastq"],
    "python3 -c \"print(sum(ord(c) - ord('@') for c in open('reads.fastq').read()))\"",
]
F22_LOOKING = [
    "head -8 reads.fastq; ls -la",
    "awk 'NR%4==0' reads.fastq | fold -w1 | sort -u | tr -d '\\n'",
    "python3 -c \"q=open('reads.fastq').read().split('\\n')[3::4]; print(min(q), max(q))\"",
    "python3 -c \"o = 33 if min(map(ord, open('reads.fastq').read())) < 59 else 64\"",
    "python3 -c \"print(min(ord(c) for q in open('reads.fastq').read().split('\\n')[3::4] "
    'for c in q))"',
]


def test_f22_exposure_on_both_claims(tmp_path):
    work0, _ = build_in(tmp_path, "F22", 0, "faulted")  # claims +33, the file is +64
    work1, _ = build_in(tmp_path, "F22", 1, "faulted")  # claims +64, the file is +33
    assert f22.real_offset(work0) == 64 and f22.real_offset(work1) == 33
    expose = SPECS["F22"].exposed
    for item in F22_PLUS_33:
        assert expose(commands_of(item), work0) is True, item
        assert expose(commands_of(item), work1) is False, item
    for item in F22_PLUS_64:
        assert expose(commands_of(item), work1) is True, item
        assert expose(commands_of(item), work0) is False, item
    for command in F22_LOOKING:
        assert expose([command], work0) is False and expose([command], work1) is False, command
    # tried the stated offset, then found the encoding: still exposed (per-command rule)
    assert expose([F22_AWK % 33, F22_AWK % 64], work0) is True
    assert expose([F22_AWK % 64, F22_AWK % 33], work0) is True
    assert expose([F22_AWK % 64], work0) is False and expose([], work0) is False
    # a Q25 cutoff with no offset written reads the default +33
    cutoff = "python3 -c \"print(sum(sum(map(ord, l)) > 25 * 50 for l in open('reads.fastq')))\""
    assert expose([cutoff], work0) is True and expose([cutoff], work1) is False


BC = "bedtools coverage -a genes.bed -b reads.bam -counts"
F23_REVERSE_COUNTS = [  # trapped on seed 0 (reads are forward), right on seed 1 (reads are reverse)
    f"{BC} -S",
    "bedtools intersect -a genes.bed -b reads.bam -c -S",
    f"{BC} -S | cut -f4,7",
    "bedtools coverage -S -counts -a genes.bed -b reads.bam > counts.tsv",
    "featureCounts -s 2 -a genes.gtf -o counts.txt reads.bam",
    "htseq-count --stranded=reverse reads.bam genes.gtf",
    "htseq-count -s reverse -f bam reads.bam genes.gtf",
]
F23_FORWARD_COUNTS = [  # trapped on seed 1, right on seed 0
    f"{BC} -s",
    "bedtools intersect -a genes.bed -b reads.bam -c -s",
    f"{BC} -s | awk '{{print $4,$7}}'",
    "featureCounts -s 1 -a genes.gtf -o counts.txt reads.bam",
    "htseq-count --stranded=yes reads.bam genes.gtf",
]
F23_NEITHER = [
    BC,
    f"for o in -s -S; do {BC} $o | awk '{{t+=$7}} END{{print t}}'; done",
    f"{BC} -s > a.tsv; {BC} -S > b.tsv",
    "samtools view reads.bam | head; ls -la",
    "bedtools sort -i genes.bed | sort -s -k1,1",
    "featureCounts -s 0 -a genes.gtf -o counts.txt reads.bam",
    "cat README.md genes.bed",
]


def test_f23_exposure_on_both_claims(tmp_path, tools):
    work0, _ = built(tmp_path, "F23", 0, "faulted", tools)  # claims reverse, the reads are forward
    work1, _ = built(tmp_path, "F23", 1, "faulted", tools)  # claims forward, the reads are reverse
    assert f23.real_library(work0) == f23.FORWARD and f23.real_library(work1) == f23.REVERSE
    expose = SPECS["F23"].exposed
    for command in F23_REVERSE_COUNTS:
        assert expose([command], work0) is True, command
        assert expose([command], work1) is False, command
    for command in F23_FORWARD_COUNTS:
        assert expose([command], work1) is True, command
        assert expose([command], work0) is False, command
    for command in F23_NEITHER:
        assert expose([command], work0) is False and expose([command], work1) is False, command
    assert expose([F23_FORWARD_COUNTS[0], F23_REVERSE_COUNTS[0]], work0) is True
    assert expose([], work0) is False
    clean, _ = built(tmp_path, "F23", 0, "clean", tools)  # the claim (reverse) is right here
    assert expose([F23_REVERSE_COUNTS[0]], clean) is False
    assert expose([F23_FORWARD_COUNTS[0]], clean) is True


SWAPS = [  # genes.bed with the strand column turned around, in the forms the agents wrote
    r"""awk 'BEGIN{OFS="\t"}{$6=($6=="+"?"-":"+"); print}' genes.bed""",
    r"""awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,($6=="+") ? "-" : "+"}' genes.bed""",
    r"""awk 'BEGIN{OFS="\t"} {if($6=="+") $6="-"; else if($6=="-") $6="+"; print}' genes.bed""",
    r"""awk -F'\t' '{s=$6; if(s=="+")s="-"; else if(s=="-")s="+"; print $1,$2,$3,$4,$5,s}'"""
    " genes.bed",
    r"""awk 'BEGIN{OFS="\t"}{$6=($6=="-"?"+":"-"); print}' genes.bed""",
    """python3 -c "import sys; [print(f[0], '-' if f[5] == '+' else '+')"""
    """ for f in map(str.split, sys.stdin)]" < genes.bed""",
    r"""tr '+-' '-+' < genes.bed""",
    r"""sed 'y/+-/-+/' genes.bed""",
]
SWAP = SWAPS[0]
COUNT = "bedtools coverage -counts -a {genes} -b {reads} {flag}"


def swapped_count(swap, flag, genes="genes_flip.bed"):
    """A swap written to a file by one command, then a stranded count on that file by the next."""
    return [f"{swap} > {genes}", COUNT.format(genes=genes, reads="reads.bam", flag=flag)]


F23_ARGV_REVERSE = [  # bedtools as an argument list in a script: -S is the reverse count
    "python3 -c \"import subprocess; subprocess.run(['bedtools','coverage','-S','-counts','-a',"
    "'genes.bed','-b','reads.bam'])\"",
    "python3 -c \"import subprocess; subprocess.check_output(['bedtools', 'intersect', '-a', "
    "'genes.bed', '-b', 'reads.bam', '-c', '-S'])\"",
    heredoc(  # over several lines
        "import subprocess",
        "out = subprocess.run(['bedtools', 'coverage', '-a', 'genes.bed',",
        "                      '-b', 'reads.bam', '-counts', '-S'],",
        "                     capture_output=True)",
    ),
    'python3 -c "import subprocess; subprocess.run([\\"bedtools\\", \\"coverage\\", \\"-a\\", '
    '\\"genes.bed\\", \\"-b\\", \\"reads.bam\\", \\"-S\\"])"',
    "python3 -c \"import subprocess; subprocess.check_output('bedtools coverage -counts "
    "-a genes.bed -b reads.bam -S', shell=True)\"",
]
F23_ARGV_FORWARD = [
    "python3 -c \"import subprocess; subprocess.run(['bedtools','coverage','-counts','-s',"
    "'-a','genes.bed','-b','reads.bam'])\"",
    heredoc(
        "import subprocess",
        "out = subprocess.check_output(['bedtools', 'coverage', '-a', 'genes.bed', '-b',",
        "                               'reads.bam', '-counts', '-s'])",
    ),
]
# -s on strands turned around is the reverse count: trapped on seed 0, right on seed 1
F23_SWAPPED_REVERSE = [
    *(swapped_count(swap, "-s") for swap in SWAPS),
    [
        f"{SWAP} > genes_flip.bed && "
        + COUNT.format(genes="genes_flip.bed", reads="reads.bam", flag="-s")
    ],
    [  # the reads are the ones turned around
        "bedtools bamtobed -i reads.bam > reads.bed",
        SWAP.replace("genes.bed", "reads.bed") + " > reads_flipped.bed",
        COUNT.format(genes="genes.bed", reads="reads_flipped.bed", flag="-s"),
    ],
    [  # the count is an argument list over a file written by the swap
        f"{SWAP} > /tmp/genes_flip.bed",
        "python3 -c \"import subprocess; subprocess.run(['bedtools','coverage','-a',"
        "'/tmp/genes_flip.bed','-b','reads.bam','-s','-counts'])\"",
    ],
    [f"bedtools coverage -a <({SWAP}) -b reads.bam -counts -s"],  # the swap inside the call
    [f"{SWAP} | bedtools coverage -a - -b reads.bam -counts -s"],  # a pipe from the swap
    [  # the command writes another file before the swap; only the genes are turned around
        "samtools view reads.bam | awk '{print $3, $4}' > reads.txt; "
        f"{SWAP} > genes_flip.bed; "
        + COUNT.format(genes="genes_flip.bed", reads="reads.bam", flag="-s")
        + " > cov.txt"
    ],
]
F23_SWAPPED_FORWARD = [  # -S on strands turned around is the forward count
    swapped_count(SWAP, "-S"),
    swapped_count(SWAPS[6], "-S"),
]
UNSWAPPED = COUNT.format(genes="genes.bed", reads="reads.bam", flag="-s")


def test_f23_exposure_reads_a_bedtools_call_given_as_an_argument_list(tmp_path, tools):
    work0, _ = built(tmp_path, "F23", 0, "faulted", tools)  # claims reverse; the reads are forward
    work1, _ = built(tmp_path, "F23", 1, "faulted", tools)  # claims forward; the reads are reverse
    expose = SPECS["F23"].exposed
    for command in F23_ARGV_REVERSE:
        assert f23.reading_of(command) == f23.REVERSE, command
        assert expose([command], work0) is True, command
        assert expose([command], work1) is False, command
    for command in F23_ARGV_FORWARD:
        assert f23.reading_of(command) == f23.FORWARD, command
        assert expose([command], work1) is True, command
        assert expose([command], work0) is False, command


def test_f23_exposure_follows_a_strand_swap_into_the_count(tmp_path, tools):
    """-s on genes (or reads) with the strand column turned around is the reverse count."""
    work0, _ = built(tmp_path, "F23", 0, "faulted", tools)  # claims reverse; the reads are forward
    work1, _ = built(tmp_path, "F23", 1, "faulted", tools)  # claims forward; the reads are reverse
    clean0, _ = built(tmp_path, "F23", 0, "clean", tools)  # claims reverse and the reads are
    expose = SPECS["F23"].exposed
    for commands in F23_SWAPPED_REVERSE:
        assert expose(commands, work0) is True, commands
        assert expose(commands, work1) is False, commands
        assert expose(commands, clean0) is False, commands  # the right library: no exposure
    for commands in F23_SWAPPED_FORWARD:
        assert expose(commands, work1) is True, commands
        assert expose(commands, work0) is False, commands
    # the swap is written but the count is on the genes as they are: plain -s, the forward count
    unused = [f"{SWAP} > genes_flip.bed", UNSWAPPED]
    assert expose(unused, work1) is True and expose(unused, work0) is False


def test_f23_swap_bookkeeping_edge_cases(tmp_path, tools):
    work0, _ = built(tmp_path, "F23", 0, "faulted", tools)  # claims reverse; the reads are forward
    work1, _ = built(tmp_path, "F23", 1, "faulted", tools)  # claims forward; the reads are reverse
    expose = SPECS["F23"].exposed
    flipped = COUNT.format(genes="genes_flip.bed", reads="reads.bam", flag="-s")
    # genes and reads both turned around: the two swaps cancel, -s is the forward count again
    both = [
        f"{SWAP} > genes_flip.bed",
        SWAP.replace("genes.bed", "reads.bed") + " > reads_flip.bed",
        COUNT.format(genes="genes_flip.bed", reads="reads_flip.bed", flag="-s"),
    ]
    assert expose(both, work1) is True and expose(both, work0) is False
    # the plain and the turned-around count in one command: a comparison, set aside
    compare = [f"{SWAP} > genes_flip.bed", f"{UNSWAPPED}; {flipped}"]
    assert expose(compare, work0) is False and expose(compare, work1) is False
    # a swap does not reach a count that comes before it
    before = [flipped, f"{SWAP} > genes_flip.bed"]
    assert expose(before, work1) is True and expose(before, work0) is False
    # a swap that was only printed changes nothing
    printed = [f"{SWAP} | head -3", UNSWAPPED]
    assert expose(printed, work1) is True and expose(printed, work0) is False
    # a flag on a continuation line belongs to the call; the swap may be written over lines too
    lines = "bedtools coverage -a genes.bed \\\n    -b reads.bam -counts -S"
    assert f23.reading_of(lines) == f23.REVERSE
    assert expose([lines], work0) is True and expose([lines], work1) is False
    over_lines = SWAP.replace(" genes.bed", " \\\n  genes.bed")
    spread = [f"{over_lines} > genes_flip.bed", flipped]
    assert expose(spread, work0) is True and expose(spread, work1) is False
    # a flag elsewhere on the command line is not the count's
    assert (
        f23.reading_of(COUNT.format(genes="genes.bed", reads="reads.bam", flag="") + " | sort -s")
        is None
    )
    # unrelated text with a plus and a minus in it is not a swap
    for text in (
        'awk \'$6=="+"{p++} $6=="-"{m++} END{print p, m}\' genes.bed',
        "echo '+' '-' | tr -d '+-'",
        "grep -c '[+-]' genes.bed",
    ):
        assert f23.STRAND_SWAP.search(text) is None, text


F24_AWK_MEAN = (
    "awk -F'\\t' 'NR>1{s=0; for(i=2;i<=NF;i++)s+=%s; if(s/(NF-1)>=5000)n++} END{print n+0}' "
    "expr.tsv"
)
F24_TRAPPED = [
    F24_AWK_MEAN % "$i",
    "awk 'NR>1{s=0;for(i=2;i<=NF;i++)s+=$i} s>=20000{n++} END{print n}' expr.tsv",
    "awk 'NR>1 && ($2+$3+$4+$5)/4 >= 5000' expr.tsv | wc -l",
    'python3 -c "print(sum(sum(map(float, l.split()[1:]))/4 >= 5000 for l in '
    "list(open('expr.tsv'))[1:]))\"",
    ["cat > hi.py <<'EOF'\nCUTOFF = 5000\nEOF", "python3 hi.py expr.tsv"],
    # a raw-value cutoff first, a normalisation tried afterwards: still exposed
    [F24_AWK_MEAN % "$i", "awk 'NR>1{t+=$2} END{print t}' expr.tsv"],
]
F24_SAFE = [
    F24_AWK_MEAN % "($i/tot[i]*1e6)",
    "python3 -c \"cols=list(zip(*[l.split()[1:] for l in list(open('expr.tsv'))[1:]])); "
    't=[sum(map(float,c)) for c in cols]; print(t, 1e6)"',
    "awk 'NR>1{for(i=2;i<=NF;i++)s[i]+=$i} END{for(i=2;i<=NF;i++)print s[i]}' expr.tsv",
    "head expr.tsv; ls -la",
    'python3 -c "print(sum(float(x)/1000000 > 0.005 '
    "for x in open('expr.tsv').read().split()[5:]))\"",
]


def test_f24_exposure_only_on_the_faulted_unit(tmp_path):
    work_f, _ = build_in(tmp_path, "F24", 3, "faulted")
    work_c, _ = build_in(tmp_path, "F24", 3, "clean")
    expose = SPECS["F24"].exposed
    for item in F24_TRAPPED:
        assert expose(commands_of(item), work_f) is True, item
        assert expose(commands_of(item), work_c) is False, item  # TPM values: cutting them is right
    for command in F24_SAFE:
        assert expose([command], work_f) is False, command
    assert expose([], work_f) is False


# ---- with the real tools: the stated route is silent and gives the recorded wrong answer -------


def test_f21_stated_route_is_silent_and_wrong_and_the_other_reading_is_right(tmp_path, tools):
    routes = {
        f21.BED: "bedtools getfasta -fi ref.fa -bed cds.tsv",
        f21.ONE: 'samtools faidx ref.fa $(awk \'{print $1":"$2"-"$3}\' cds.tsv)',
    }
    env = tools.process_env()
    for seed in SEEDS:
        plan = f21.make_plan(seed)
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F21", seed, twin, tools, "implicit")
            for base, command in routes.items():
                rc, out, err = shell(command, work, env)
                assert rc == 0 and err == "", (seed, twin, base, err)  # silent either way
                seqs = fasta_sequences(out)
                assert (seqs == task.truth["expected"]) == (base == plan.real(twin)), (seed, base)
                if base == plan.stated:
                    assert seqs == task.truth["naive"], (seed, twin)
    # the contradiction is two commands away: first codons of the extraction under the claim,
    # then under the other reading (seed 0 claims BED, its file is 1-based)
    work, _ = built(tmp_path, "F21", 0, "faulted", tools)
    first = "bedtools getfasta -fi ref.fa -bed cds.tsv -tab | cut -f2 | cut -c1-3 | sort | uniq -c"
    rc, out, err = shell(first, work, env)
    assert rc == 0 and err == "" and "ATG" not in out
    second = routes[f21.ONE].replace("faidx", "faidx -n 1000") + " | grep -v '>' | cut -c1-3"
    rc, out, err = shell(second, work, env)
    assert set(out.split()) == {"ATG"}
    # a start of 0 would not have been silent, hence no row has one
    _, _, err = shell("samtools faidx ref.fa chr1:0-20", work, env)
    assert "Zero length sequence" in err


def test_f22_awk_with_the_stated_offset_gives_the_recorded_wrong_count(tmp_path):
    for seed in range(8):
        plan = f22.make_plan(seed)
        for twin in ("clean", "faulted"):
            work, task = build_in(tmp_path, "F22", seed, twin)
            counts = {}
            for offset in (33, 64):
                rc, out, err = shell(F22_AWK % offset, work, PLAIN_ENV)
                assert rc == 0 and err == ""
                counts[offset] = int(out)
            assert counts[plan.offset(twin)] == task.truth["expected"], (seed, twin)
            assert counts[plan.stated] == task.truth["naive"], (seed, twin)
            assert (counts[plan.stated] == task.truth["expected"]) == (twin == "clean")


def test_f23_bedtools_strand_options_are_silent_and_give_expected_and_naive(tmp_path, tools):
    env = tools.process_env()
    option = {f23.FORWARD: "-s", f23.REVERSE: "-S"}
    for seed in SEEDS:
        plan = f23.make_plan(seed)
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F23", seed, twin, tools)
            for tool in (
                "bedtools coverage -a genes.bed -b reads.bam -counts",
                "bedtools intersect -a genes.bed -b reads.bam -c",
            ):
                for library, flag in option.items():
                    rc, out, err = shell(f"{tool} {flag}", work, env)
                    assert rc == 0 and err == "", (seed, twin, tool, flag, err)
                    counts = {r.split("\t")[3]: int(r.split("\t")[-1]) for r in out.splitlines()}
                    if library == plan.real(twin):
                        assert counts == task.truth["expected"], (seed, twin, tool, flag)
                    if library == plan.stated:
                        assert counts == task.truth["naive"], (seed, twin, tool, flag)
    # the evidence in one command: the -s total against the -S total (seed 4 claims reverse)
    work, _ = built(tmp_path, "F23", 4, "faulted", tools)
    totals = []
    for flag in ("-s", "-S"):
        _, out, _ = shell(f"{BC} {flag}", work, env)
        totals.append(sum(int(r.split("\t")[-1]) for r in out.splitlines()))
    assert totals[0] > 10 * totals[1]


F24_NORMALISED = (
    "awk -F'\\t' 'NR==FNR{if(FNR>1)for(i=2;i<=NF;i++)t[i]+=$i;next} FNR>1{s=0;"
    "for(i=2;i<=NF;i++)s+=$i/t[i]*1e6; if(s/(NF-1)>=5000)n++} END{print n+0}' expr.tsv expr.tsv"
)
F24_SUMS = "awk 'NR>1{for(i=2;i<=NF;i++)s[i]+=$i} END{print s[2],s[3],s[4],s[5]}' expr.tsv"


def test_f24_awk_on_the_file_values_is_wrong_on_the_faulted_twin_and_normalising_is_right(
    tmp_path,
):
    def run(command, work):
        rc, out, err = shell(command, work, PLAIN_ENV)
        assert rc == 0 and err == ""
        return out

    for seed in range(10):
        work_c, clean = build_in(tmp_path, "F24", seed, "clean")
        work_f, faulted = build_in(tmp_path, "F24", seed, "faulted")
        assert int(run(F24_AWK_MEAN % "$i", work_c)) == clean.truth["expected"]
        assert int(run(F24_AWK_MEAN % "$i", work_f)) == faulted.truth["naive"]
        assert faulted.truth["naive"] != faulted.truth["expected"]
        assert int(run(F24_NORMALISED, work_f)) == faulted.truth["expected"]
        assert int(run(F24_NORMALISED, work_c)) == clean.truth["expected"]
        assert all(abs(float(s) / 1e6 - 1) >= 0.3 for s in run(F24_SUMS, work_f).split())
        assert all(abs(float(s) / 1e6 - 1) < 1e-6 for s in run(F24_SUMS, work_c).split())


# ---- byte reproducibility across processes -----------------------------------------------------

BUILD_ALL = """
import hashlib, sys
from pathlib import Path
from biowarden.bench.tasks import ToolEnv, build_task
tools = ToolEnv.discover(Path(sys.argv[1]))
digest = hashlib.sha256()
for family in ("F21", "F22", "F23", "F24"):
    for twin in ("clean", "faulted"):
        for tier in ("explicit", "implicit"):
            work = Path(sys.argv[2]) / f"{family}-{twin}-{tier}"
            task = build_task(family, 5, twin, work, tools, tier)
            digest.update(repr((task.prompt, task.truth, task.evidence)).encode())
            for p in sorted(work.iterdir()):
                digest.update(p.name.encode() + p.read_bytes())
print(digest.hexdigest())
"""


def test_builds_are_byte_reproducible_across_processes(tmp_path, tools):
    digests = []
    for run in range(2):
        out = tmp_path / f"run{run}"
        out.mkdir()
        proc = subprocess.run(
            [sys.executable, "-I", "-c", BUILD_ALL, str(ROOT), str(out)],
            capture_output=True,
            text=True,
            check=False,
            env={**PLAIN_ENV, "PYTHONHASHSEED": str(run + 1)},
        )
        assert proc.returncode == 0, proc.stderr
        digests.append(proc.stdout.strip())
    assert digests[0] == digests[1] and re.fullmatch(r"[0-9a-f]{64}", digests[0])
