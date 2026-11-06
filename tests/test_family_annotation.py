"""Family-specific checks for F11 (gff_exons), F12 (strand_seq) and F16 (tss_upstream)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.extra.f12_strand_seq import revcomp
from biowarden.bench.tasks.extra.f16_tss_upstream import Gene, hits, window0

SEEDS = range(30)
ROOT = Path(__file__).resolve().parents[1]
SPECS = load()
MIN_TRAP = {"F11": 3, "F12": 3, "F16": 12}  # items that differ, naive vs expected, per seed
GFF_DEFAULT = "awk -v OFS='\\t' '$3==\"exon\"{print $1,$4,$5}' exons.gff3 | bedtools intersect"
EXPOSED = {
    "F11": [
        "awk '$3==\"exon\"{print $1,$4,$5}' exons.gff3 > exons_conv.bed",
        GFF_DEFAULT + " -a variants.vcf.gz -b stdin -u | wc -l",
        "grep -v '^#' exons.gff3 | cut -f1,4,5 > e.bed",
        "awk -F'\\t' '{print $1\"\\t\"$4\"\\t\"$5}' ./exons.gff3 | bedtools sort -i - > e.bed",
    ],
    "F12": [
        "bedtools getfasta -fi ref.fa -bed features.bed",
        "bedtools getfasta -fi ref.fa -bed features.bed -name -tab | cut -f2",
        "samtools faidx ref.fa chr1:208-246 chr1:475-500",
        'awk \'{print $1":"$2+1"-"$3}\' features.bed | xargs samtools faidx ref.fa',
    ],
    "F16": [
        "bedtools flank -i genes.bed -g genome.sizes -l 100 -r 0 > up.bed",
        "awk '{print $1,$2-100,$2}' genes.bed",
        "awk -v OFS='\\t' '{print $1,$2 - 100,$2,$4}' genes.bed > w.bed",
        "python3 - <<'EOF'\nfor l in open('genes.bed'):\n    start = int(l.split()[1])\n"
        "    print(start - 100, start)\nEOF",
    ],
}
SAFE = {
    "F11": [
        "bedtools intersect -a variants.vcf.gz -b exons.gff3 -u | wc -l",
        "awk -v OFS='\\t' '$3==\"exon\"{print $1,$4-1,$5}' exons.gff3 > e.bed",
        "awk '$3==\"exon\"{print $1,$4 - 1,$5}' exons.gff3 | bedtools intersect -a v.vcf -b -",
        "cut -f1,4,5 exons.gff3 | awk '{print $1,$2-1,$3}' > e.bed",
        "grep -v '^#' exons.gff3 | head -3",
        "cut -f1,4,5 exons.gff3 | head",
        "awk '{print $1,$4,$5}' variants.vcf > v.txt",
    ],
    "F12": [
        "bedtools getfasta -fi ref.fa -bed features.bed -s",
        "bedtools getfasta -fi ref.fa -bed features.bed -s -name -tab",
        "samtools faidx ref.fa chr1:208-246 -i",
        "samtools faidx ref.fa chr1:208-246 --reverse-complement",
        "samtools faidx ref.fa chr1:208-246 | rev | tr ACGT TGCA",
        "samtools faidx ref.fa",
        "bedtools getfasta -fi ref.fa -bed plus.bed",
        "sort -s features.bed | head",
    ],
    "F16": [
        "bedtools flank -i genes.bed -g genome.sizes -l 100 -r 0 -s | bedtools intersect "
        "-a variants.vcf.gz -b stdin -u",
        'awk \'$6=="+"{print $1,$2-100,$2} $6=="-"{print $1,$3,$3+100}\' genes.bed',
        "bedtools flank -i other.bed -g genome.sizes -l 100 -r 0",
        "cat genes.bed; bcftools view -H variants.vcf.gz | head",
    ],
}
UNRELATED = ["ls -la; cat README.md", "bcftools view -h variants.vcf.gz | head", "wc -l *", ""]


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


def make(tmp_path, family, seed, twin, tools, tier="explicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    return work, build_task(family, seed, twin, work, tools, tier, route)


def differing(truth: dict) -> int:
    expected, naive = truth["expected"], truth["naive"]
    if truth["kind"] == "int":
        return abs(expected - naive)
    if truth["kind"] == "seq_list":
        return sum(a != b for a, b in zip(expected, naive, strict=True))
    return len(set(expected) ^ set(naive))


@pytest.mark.parametrize("family", ["F11", "F12", "F16"])
def test_exposure_is_true_for_the_trapped_commands_and_false_for_the_safe_route(family):
    exposed = SPECS[family].exposed
    for command in EXPOSED[family]:
        assert exposed([command], Path(".")) is True, command
    assert exposed(["ls", EXPOSED[family][0]], Path(".")) is True
    for command in [*SAFE[family], *UNRELATED]:
        assert exposed([command], Path(".")) is False, command
    assert exposed(SAFE[family], Path(".")) is False


@pytest.mark.parametrize("family", ["F11", "F12", "F16"])
def test_the_trap_changes_the_answer_on_every_seed_and_tier(tmp_path, tools, family):
    for seed in SEEDS:
        for tier in ("explicit", "implicit"):
            _, clean = make(tmp_path, family, seed, "clean", tools, tier)
            _, faulted = make(tmp_path, family, seed, "faulted", tools, tier)
            assert clean.truth["naive"] == clean.truth["expected"], clean.task_id
            assert faulted.truth["naive"] != faulted.truth["expected"], faulted.task_id
            assert clean.truth["expected"] == faulted.truth["expected"], faulted.task_id
            assert differing(faulted.truth) >= MIN_TRAP[family], (faulted.task_id, faulted.truth)


@pytest.mark.parametrize("family", ["F11", "F12", "F16"])
def test_data_is_tiny(tmp_path, tools, family):
    for seed in SEEDS:
        work, _ = make(tmp_path, family, seed, "faulted", tools)
        if family == "F12":
            lengths = [
                int(line.split("\t")[1])
                for line in (work / "ref.fa.fai").read_text().split("\n")
                if line
            ]
            records = len((work / "features.bed").read_text().splitlines())
            assert len(lengths) <= 3 and sum(lengths) <= 3000 and records <= 80
            continue
        rows = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work).stdout.splitlines()
        assert 0 < len(rows) <= 80
        if family == "F16":
            sizes = [line.split("\t") for line in (work / "genome.sizes").read_text().splitlines()]
            assert len(sizes) <= 3 and sum(int(n) for _, n in sizes) <= 3000


def test_f11_gff_carries_the_same_exons_in_one_based_closed_coordinates(tmp_path, tools):
    for seed in range(5):
        work_b, clean = make(tmp_path, "F11", seed, "clean", tools)
        work_g, faulted = make(tmp_path, "F11", seed, "faulted", tools)
        bed = [line.split("\t") for line in (work_b / "exons.bed").read_text().splitlines()]
        gff = [
            line.split("\t")
            for line in (work_g / "exons.gff3").read_text().splitlines()
            if not line.startswith("#")
        ]
        assert [(c, int(s) + 1, int(e)) for c, s, e in bed] == [
            (r[0], int(r[3]), int(r[4])) for r in gff
        ]
        assert all(r[2] == "exon" for r in gff)
        # bedtools reads the GFF3 natively and agrees with the BED; the hand conversion does not
        direct = tools.run(
            ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "exons.gff3", "-u"], work_g
        )
        assert direct.returncode == 0 and direct.stderr == ""
        assert (
            len(direct.stdout.splitlines()) == faulted.truth["expected"] == clean.truth["expected"]
        )


def test_f11_hand_conversion_is_silent_and_drops_the_first_base_of_exons(tmp_path, tools):
    work, task = make(tmp_path, "F11", 7, "faulted", tools)
    awk = ["awk", "-v", "OFS=\t", '$3=="exon"{print $1,$4,$5}', "exons.gff3"]
    converted = tools.run(awk, work)
    run = tools.run(
        ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "stdin", "-u"],
        work,
        stdin=converted.stdout,
    )
    assert converted.returncode == run.returncode == 0 and converted.stderr == run.stderr == ""
    assert len(run.stdout.splitlines()) == task.truth["naive"] < task.truth["expected"]


def test_f12_minus_features_are_the_reverse_complement_only_when_strand_aware(tmp_path, tools):
    for seed in range(5):
        work, task = make(tmp_path, "F12", seed, "faulted", tools)
        rows = [line.split("\t") for line in (work / "features.bed").read_text().splitlines()]
        strands = [r[5] for r in rows]
        assert set(strands) == {"+", "-"} and all(r[4] == "0" and len(r) == 6 for r in rows)
        assert all(int(r[2]) - int(r[1]) >= 20 for r in rows)
        expected, naive = task.truth["expected"], task.truth["naive"]
        for strand, want, got in zip(strands, expected, naive, strict=True):
            assert (got == revcomp(want)) if strand == "-" else (got == want)
            assert want == want.upper() and (want != got) == (strand == "-")
        blind = tools.run(["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "features.bed"], work)
        aware = tools.run(
            ["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "features.bed", "-s"], work
        )
        assert blind.returncode == aware.returncode == 0 and blind.stderr == aware.stderr == ""
        assert blind.stdout.split()[1::2] == naive and aware.stdout.split()[1::2] == expected
        _, clean = make(tmp_path, "F12", seed, "clean", tools)
        assert clean.truth["expected"] == expected


def test_f16_window_follows_the_strand_and_excludes_the_tss_base():
    plus, minus = Gene("p", "chr1", 500, 600, "+"), Gene("m", "chr1", 900, 1000, "-")
    assert window0(plus) == (400, 500) and window0(minus) == (1000, 1100)
    assert window0(minus, strand_aware=False) == (800, 900)
    variants = [(f"p{pos}", "chr1", pos) for pos in (400, 401, 500, 501)]
    variants += [(f"m{pos}", "chr1", pos) for pos in (801, 900, 1000, 1001, 1100, 1101)]
    # 1-based: + window 401..500 (501 is the TSS base); - window 1001..1100 (1000 is the TSS base)
    assert hits(variants, [plus, minus], True) == ["p401", "p500", "m1001", "m1100"]
    # strand-blind: the - gene gets 801..900 on its downstream side instead
    assert hits(variants, [plus, minus], False) == ["p401", "p500", "m801", "m900"]


def test_f16_flank_without_s_is_silently_strand_blind(tmp_path, tools):
    for seed in range(5):
        work, task = make(tmp_path, "F16", seed, "faulted", tools)
        genes = [line.split("\t") for line in (work / "genes.bed").read_text().splitlines()]
        assert {g[5] for g in genes} == {"+", "-"}
        flank = [
            "bedtools",
            "flank",
            "-i",
            "genes.bed",
            "-g",
            "genome.sizes",
            "-l",
            "100",
            "-r",
            "0",
        ]
        blind, aware = tools.run(flank, work), tools.run([*flank, "-s"], work)
        assert blind.returncode == aware.returncode == 0 and blind.stderr == aware.stderr == ""
        for g, line in zip(genes, blind.stdout.splitlines(), strict=True):
            assert line.split("\t")[1:3] == [str(int(g[1]) - 100), g[1]]  # the + window, always
        argv = ["bedtools", "intersect", "-a", "variants.vcf.gz", "-b", "stdin", "-u"]
        found = tools.run(argv, work, stdin=blind.stdout).stdout.splitlines()
        assert sorted(line.split("\t")[2] for line in found) == sorted(task.truth["naive"])
