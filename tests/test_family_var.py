"""Family-specific tests for the variant pack, F53-F64.

The generic gates live in test_extra_families.py. Here, per family: the exposure functions on
concrete command lines, the size of the trap over seeds 0-29 on both tiers (read off the built
truth), twins that differ in the one declared fact, the explicit and implicit READMEs, builds that
are byte-reproducible across processes, and checks that the naive route runs the installed tool
without a warning about the trap (exit 0) and that the tool behaves as the module docstring says.
"""

from __future__ import annotations

import hashlib
import random
import subprocess
import sys
from pathlib import Path

import pytest
from family_forms import typed_forms

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f53_vcftools_bed as f53
from biowarden.bench.tasks.extra import f54_format_any as f54
from biowarden.bench.tasks.extra import f55_filter_dot as f55
from biowarden.bench.tasks.extra import f56_missing_annotation as f56
from biowarden.bench.tasks.extra import f57_subset_monomorphic as f57
from biowarden.bench.tasks.extra import f58_plink_a1 as f58
from biowarden.bench.tasks.extra import f59_plink_multiallelic as f59
from biowarden.bench.tasks.extra import f60_max_missing as f60
from biowarden.bench.tasks.extra import f61_tabix_bed as f61
from biowarden.bench.tasks.extra import f62_isec_repr as f62
from biowarden.bench.tasks.extra import f63_merge_absent as f63
from biowarden.bench.tasks.extra import f64_picard_dups as f64
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.extra.f53_vcftools_bed import (
    normalise,
    pipelines,
    segments,
    unwrap_words,
    workspace_file,
)
from biowarden.bench.tasks.extra.f64_picard_dups import read_names

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
WORK = Path("/nonexistent-workspace")
MODULES = {
    "F53": f53, "F54": f54, "F55": f55, "F56": f56, "F57": f57, "F58": f58,
    "F59": f59, "F60": f60, "F61": f61, "F62": f62, "F63": f63, "F64": f64,
}  # fmt: skip
FAMILIES = sorted(MODULES, key=lambda i: int(i[1:]))


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    needed = ["vcftools", "plink", "plink2", "tabix", "bgzip", "picard", "gatk"]
    if env is None or set(needed) - set(env.real_bins(needed)):
        pytest.skip("samtools, bcftools and the variant tools are not available")
    return env


def silent(proc, words=("warn", "error", "fail", "skip", "drop", "discard")) -> bool:
    return proc.returncode == 0 and not any(w in proc.stderr.lower() for w in words)


# ---- exposure ----------------------------------------------------------------------------------


def test_f53_exposure_needs_vcftools_bed_on_a_headerless_file(tmp_path, tools):
    faulted, clean = tmp_path / "f", tmp_path / "c"
    build_task("F53", 3, "faulted", faulted, tools, "implicit")
    build_task("F53", 3, "clean", clean, tools, "implicit")
    spec = f53.SPEC
    trapped = [
        "vcftools --gzvcf variants.vcf.gz --bed targets.bed --recode --stdout",
        "vcftools --gzvcf variants.vcf.gz --bed=targets.bed --out o --recode",
        "vcftools --gzvcf variants.vcf.gz --exclude-bed targets.bed --recode --stdout | wc -l",
        "cd /w && vcftools --gzvcf variants.vcf.gz --bed targets.bed --freq",
    ]
    safe = [
        "bcftools view -H -R targets.bed variants.vcf.gz",
        "bedtools intersect -a variants.vcf.gz -b targets.bed -u",
        "(printf '#chrom\\tstart\\tend\\n'; cat targets.bed) > h.bed; vcftools --vcf v --bed h.bed",
        "sed '1i #chrom\\tstart\\tend' targets.bed > h.bed && vcftools --vcf v --bed h.bed",
        "vcftools --gzvcf variants.vcf.gz --positions pos.txt --recode --stdout",
        "head -3 targets.bed; ls -la",
    ]
    for command in trapped:
        assert spec.exposed([command], faulted) is True, command
        assert spec.exposed([command], clean) is False, command  # the clean file has a header
    for command in safe:
        assert spec.exposed([command], faulted) is False, command
    assert spec.exposed(["ls", trapped[0]], faulted) is True
    assert spec.exposed(["vcftools --bed absent.bed --vcf v.vcf"], faulted) is False
    assert spec.exposed([], faulted) is False


def test_f53_exposure_reads_continuations_work_paths_quotes_and_argument_lists(tmp_path, tools):
    faulted, clean = tmp_path / "f", tmp_path / "c"
    build_task("F53", 3, "faulted", faulted, tools, "implicit")
    build_task("F53", 3, "clean", clean, tools, "implicit")
    trapped = [
        "vcftools --gzvcf variants.vcf.gz \\\n  --bed targets.bed \\\n  --recode --stdout",
        "cd /work && vcftools --gzvcf /work/variants.vcf.gz --bed /work/targets.bed --recode "
        "--stdout",
        "vcftools --gzvcf variants.vcf.gz --bed 'targets.bed' --recode --stdout",
        "python3 -c \"import subprocess; subprocess.run(['vcftools', '--gzvcf', "
        "'variants.vcf.gz', '--bed', 'targets.bed'])\"",
    ]
    for command in trapped:
        assert f53.SPEC.exposed([command], faulted) is True, command
        assert f53.SPEC.exposed([command], clean) is False, command  # the clean file has a header
    assert f53.SPEC.exposed([trapped[0].replace("targets.bed", "absent.bed")], faulted) is False


def test_f53_exposure_never_reads_a_file_outside_the_workspace(tmp_path, tools):
    faulted = tmp_path / "f"
    build_task("F53", 3, "faulted", faulted, tools, "implicit")
    (tmp_path / "outside.bed").write_text("chr1\t10\t20\n")  # looks like a bed without a header
    for name in (str(tmp_path / "outside.bed"), "../outside.bed"):
        command = f"vcftools --gzvcf variants.vcf.gz --bed {name} --recode --stdout"
        assert f53.SPEC.exposed([command], faulted) is False, name
    inside = "vcftools --gzvcf variants.vcf.gz --bed targets.bed --recode --stdout"
    assert f53.SPEC.exposed([inside], faulted) is True
    # a name with a NUL, and a binary file given as the bed: nothing to read, nothing raised
    for name in ("targ\x00ets.bed", "variants.vcf.gz"):
        command = f"vcftools --gzvcf variants.vcf.gz --bed {name} --recode --stdout"
        assert f53.SPEC.exposed([command], faulted) is False, name


def test_f54_exposure_needs_an_unaggregated_include_on_gq(tmp_path):
    spec = f54.SPEC
    trapped = [
        "bcftools view -H -i 'FMT/GQ>=20' cohort.vcf.gz | wc -l",
        "bcftools view -i 'GQ>=20' cohort.vcf.gz -Ov | grep -vc '^#'",
        "bcftools view -H --include 'FORMAT/GQ>20' cohort.vcf.gz",
        "bcftools filter -i \"GQ>=20\" cohort.vcf.gz | grep -vc '^#'",
        "bcftools query -f '%POS\\n' -i 'FMT/GQ>=20' cohort.vcf.gz | wc -l",
        "bcftools view -H -i 'FMT/GQ>=20 && FMT/DP>10' cohort.vcf.gz",
        "bcftools view -H -i 'GQ[*]>=20' cohort.vcf.gz | wc -l",  # all samples: still "any"
        "bcftools view -H -i 'FMT/GQ[0-5]>=20' cohort.vcf.gz",
        "bcftools view -iFMT/GQ\\>=20 -H cohort.vcf.gz",  # an unquoted, escaped expression
        "bcftools view -i \\\n  'FMT/GQ>=20' \\\n  -H cohort.vcf.gz | wc -l",
        "bcftools view -H -i 'FMT/GQ>=20' /work/cohort.vcf.gz | wc -l",
    ]
    safe = [
        "bcftools view -H -e 'FMT/GQ<20' cohort.vcf.gz | wc -l",
        "bcftools view -H -i 'MIN(FMT/GQ)>=20' cohort.vcf.gz | wc -l",
        "bcftools view -H -i 'F_PASS(FMT/GQ>=20)==1' cohort.vcf.gz",
        "bcftools view -H -i 'N_PASS(GQ>=20)==N_SAMPLES' cohort.vcf.gz",
        "bcftools view -H -i 'FMT/GQ[0]>=20' cohort.vcf.gz",
        "bcftools filter -S . -i 'GQ>=20' cohort.vcf.gz",
        "bcftools query -f '[%GQ ]\\n' cohort.vcf.gz | awk '$1<20{n++} END{print n}'",
        "bcftools view -H -i 'QUAL>=20' cohort.vcf.gz",
        "bcftools stats cohort.vcf.gz",
    ]
    for command in trapped:
        assert spec.exposed([command], WORK) is True, command
    for command in safe:
        assert spec.exposed([command], WORK) is False, command
    assert spec.exposed(["ls -la; cat README.md", *safe], WORK) is False
    assert spec.exposed(["bcftools stats cohort.vcf.gz", trapped[0]], WORK) is True
    assert spec.exposed([], WORK) is False


EXPOSURE = {
    "F55": (
        [
            "bcftools view -H -f PASS variants.vcf.gz | wc -l",
            "bcftools view --apply-filters PASS variants.vcf.gz",
            "bcftools view -H -i 'FILTER=\"PASS\"' variants.vcf.gz",
            "bcftools view -H -e 'FILTER!=\"PASS\"' variants.vcf.gz",
            "vcftools --gzvcf variants.vcf.gz --keep-filtered PASS --recode --stdout",
            "zcat variants.vcf.gz | awk '$7==\"PASS\"' | wc -l",
            "bcftools view -H variants.vcf.gz | grep -c PASS",
            "bcftools view -H -f 'PASS' variants.vcf.gz | wc -l",
            'bcftools view -H -f "PASS" variants.vcf.gz | wc -l',
            "bcftools query -f '%ID\\t%FILTER\\n' variants.vcf.gz | awk '$2==\"PASS\"' | cut -f1",
        ],
        [
            "bcftools view -H -f .,PASS variants.vcf.gz | wc -l",
            "bcftools view -H -f PASS,. variants.vcf.gz",
            'bcftools view -H -i \'FILTER="PASS" || FILTER="."\' variants.vcf.gz',
            "vcftools --gzvcf variants.vcf.gz --remove-filtered-all --recode --stdout",
            "bcftools view -H variants.vcf.gz | cut -f7 | sort | uniq -c",
            "bcftools view -H -e 'FILTER~\"Low\"' variants.vcf.gz",
            "bcftools view -H -f '.,PASS' variants.vcf.gz",
            "bcftools query -f '%ID\\t%FILTER\\n' variants.vcf.gz "
            '| awk \'$2=="PASS" || $2=="."\'',
        ],
    ),
    "F56": (
        [
            "bcftools view -H -e 'INFO/QD<2' variants.vcf.gz | wc -l",
            "bcftools view -H -e 'QD<2' variants.vcf.gz",
            "bcftools view -H --exclude 'INFO/QD<=1.99' variants.vcf.gz",
            "gatk --java-options -Xmx512m VariantFiltration -V variants.vcf.gz "
            "--filter-expression 'QD < 2.0' --filter-name lowQD -O o.vcf.gz",
            "bcftools view -H -e 'INFO/QD<2 && INFO/DP>10' variants.vcf.gz",  # inside a longer test
            "bcftools view -H -e 'QUAL<10 || INFO/QD<2' variants.vcf.gz",
            "gatk VariantFiltration -V variants.vcf.gz --filter-expression 'FS > 60.0 || QD < 2.0' "
            "--filter-name bad -O o.vcf.gz",
            # the -i stage is another statement: the exclusion itself ran
            "bcftools view -H -e 'INFO/QD<2' variants.vcf.gz; "
            "bcftools view -H -i 'INFO/QD>=2' variants.vcf.gz",
        ],
        [
            "bcftools view -H -i 'INFO/QD>=2' variants.vcf.gz | wc -l",
            # the exclusion is followed by an -i stage of the same pipeline, which drops the rest
            "bcftools view -e 'INFO/QD<2' variants.vcf.gz -Ou | bcftools query -i 'INFO/QD>=2' "
            "-f '%ID\\n'",
            "bcftools view -e 'INFO/QD<2' variants.vcf.gz | bcftools view -H -i 'QD>=2' -",
            "bcftools view -H -e 'INFO/QD<2 || INFO/QD=\".\"' variants.vcf.gz",
            "gatk VariantFiltration -V variants.vcf.gz --filter-expression 'QD < 2.0' "
            "--filter-name lowQD --missing-values-evaluate-as-failing -O o.vcf.gz",
            "bcftools view -H -e 'INFO/DP<10' variants.vcf.gz",
        ],
    ),
    "F57": (
        [
            "bcftools view -H -S group_b.txt cohort.vcf.gz | wc -l",
            "bcftools view -H -s s2,s3 cohort.vcf.gz",
            "bcftools view -H -S group_b.txt -e 'AC==0' cohort.vcf.gz | wc -l",
            "vcftools --gzvcf cohort.vcf.gz --keep group_b.txt --recode --stdout | grep -vc '^#'",
            "plink --vcf cohort.vcf.gz --keep keep.txt --make-bed --out sub",
            "bcftools view -Hs s1,s2 cohort.vcf.gz | wc -l",  # bundled short options
            "bcftools query -S group_b.txt -f '%CHROM\\t%POS\\n' cohort.vcf.gz | wc -l",
        ],
        [
            "bcftools view -H -S group_b.txt -c 1 cohort.vcf.gz | wc -l",
            "bcftools view -Hc1 -S group_b.txt cohort.vcf.gz | wc -l",
            "bcftools view -H -s s2,s3 --min-ac 1 cohort.vcf.gz",
            "vcftools --gzvcf cohort.vcf.gz --keep group_b.txt --mac 1 --recode --stdout",
            "vcftools --gzvcf cohort.vcf.gz --keep group_b.txt --non-ref-ac-any 1 --recode",
            "plink --vcf cohort.vcf.gz --keep keep.txt --mac 1 --make-bed --out sub",
            "bcftools view -S group_b.txt cohort.vcf.gz | bcftools view -H -c 1 - | wc -l",
            "bcftools view -H cohort.vcf.gz | wc -l",
        ],
    ),
    "F58": (
        [
            "plink --vcf cohort.vcf.gz --freq --out lab",
            "plink --memory 1000 --vcf cohort.vcf.gz --freq --out lab",
            "plink --vcf cohort.vcf.gz --assoc --out a",
            "plink --vcf cohort.vcf.gz --make-bed --out c",
        ],
        [
            "plink --vcf cohort.vcf.gz --keep-allele-order --freq --out lab",
            "plink2 --vcf cohort.vcf.gz --freq --out lab",
            "bcftools +fill-tags cohort.vcf.gz -- -t AF | bcftools query -f '%ID %AF\\n'",
            "plink --vcf cohort.vcf.gz --missing --out m",
        ],
    ),
    "F59": (
        [
            "plink --vcf cohort.vcf.gz --geno 0.1 --make-bed --out qc",
            "plink --memory 1000 --vcf cohort.vcf.gz --geno 0.1 --make-just-bim",
            "plink --vcf cohort.vcf.gz --missing --out m",
        ],
        [
            "plink --vcf cohort.vcf.gz --biallelic-only --geno 0.1 --make-bed --out qc",
            "plink2 --vcf cohort.vcf.gz --geno 0.1 --make-just-pvar --out qc",
            "bcftools view -H -i 'F_MISSING<=0.1' cohort.vcf.gz | wc -l",
        ],
    ),
    "F60": (
        [
            "vcftools --gzvcf cohort.vcf.gz --max-missing 0.1 --recode --stdout",
            "vcftools --gzvcf cohort.vcf.gz --max-missing=0.2 --recode --stdout | grep -vc '^#'",
            "vcftools --vcf x.vcf --max-missing 0.5 --out o",
        ],
        [
            "vcftools --gzvcf cohort.vcf.gz --max-missing 0.9 --recode --stdout",
            "vcftools --gzvcf cohort.vcf.gz --max-missing 1 --recode --stdout",
            "plink --vcf cohort.vcf.gz --geno 0.1 --make-bed",
            "bcftools view -H -i 'F_MISSING<=0.1' cohort.vcf.gz | wc -l",
        ],
    ),
    "F61": (
        [
            "tabix -R windows.tsv intervals.bed.gz",
            "tabix intervals.bed.gz -T windows.tsv",
            "tabix intervals.bed.gz chr1:100-200 chr2:300-400",
            "bgzip -k intervals.bed && tabix -s 1 -b 2 -e 3 intervals.bed.gz",
            "tabix -f -s1 -b2 -e3 intervals.bed.gz",
            # region strings put together from the three columns of a BED window
            'awk \'{print $1":"$2"-"$3}\' windows.tsv | xargs -I{} tabix intervals.bed.gz {}',
            "while read c s e; do tabix intervals.bed.gz ${c}:${s}-${e}; done < windows.tsv",
        ],
        [
            "while read c s e; do tabix intervals.bed.gz $c:$((s+1))-$e; done < windows.tsv",
            "tabix -p bed intervals.bed.gz",
            "tabix -0 -s 1 -b 2 -e 3 intervals.bed.gz",
            "cp windows.tsv windows.bed && tabix -R windows.bed intervals.bed.gz",
            'awk \'{print $1":"$2+1"-"$3}\' windows.tsv | xargs tabix intervals.bed.gz',
            "bedtools intersect -u -a intervals.bed -b windows.tsv",
            "bgzip -k intervals.bed",
        ],
    ),
    "F62": (
        [
            "bcftools isec -n=2 -w1 caller_a.vcf.gz caller_b.vcf.gz | grep -vc '^#'",
            "bcftools isec -n=2 -c all caller_a.vcf.gz caller_b.vcf.gz",
            "bcftools isec -n=2 caller_b.vcf.gz caller_a.vcf.gz",
            # bcftools stats of two files reports the shared records by position and alleles
            "bcftools stats caller_a.vcf.gz caller_b.vcf.gz | grep '^SN'",
            "bcftools isec -n=2 caller_[ab].vcf.gz",
            "comm -12 <(bcftools query -f '%CHROM:%POS:%REF:%ALT\\n' caller_a.vcf.gz | sort) "
            "<(bcftools query -f '%CHROM:%POS:%REF:%ALT\\n' caller_b.vcf.gz | sort) | wc -l",
        ],
        [
            "bcftools isec -n=2 -w1 caller_a.norm.vcf.gz caller_b.norm.vcf.gz",
            "bcftools norm -f ref.fa -Oz -o a.n.vcf.gz caller_a.vcf.gz",
            "bcftools norm -f ref.fa caller_b.vcf.gz | bcftools view -H | wc -l",
            "bcftools stats caller_a.vcf.gz",
            "comm -12 <(bcftools norm -f ref.fa caller_a.vcf.gz | bcftools query -f '%POS\\n' "
            "| sort) <(bcftools norm -f ref.fa caller_b.vcf.gz | bcftools query -f '%POS\\n' "
            "| sort)",
        ],
    ),
    "F63": (
        [
            "bcftools merge -Oz -o m.vcf.gz sample1.vcf.gz sample2.vcf.gz sample3.vcf.gz",
            "bcftools merge sample*.vcf.gz | bcftools +fill-tags -- -t AF",
            "bcftools merge --force-samples -o m.vcf sample1.vcf.gz sample2.vcf.gz",
        ],
        [
            "bcftools merge -0 -Oz -o m.vcf.gz sample1.vcf.gz sample2.vcf.gz",
            "bcftools merge --missing-to-ref sample*.vcf.gz",
            "bcftools query -f '%ID [%GT]\\n' sample1.vcf.gz",
            "bcftools view -H sample1.vcf.gz | wc -l",
        ],
    ),
    "F64": (
        [
            "picard -Xmx512m MarkDuplicates I=reads.bam O=md.bam M=m.txt",
            "picard MarkDuplicates --INPUT reads.bam --OUTPUT md.bam --METRICS_FILE m.txt",
            "gatk --java-options -Xmx512m MarkDuplicates -I reads.bam -O md.bam -M m.txt",
            "picard MarkDuplicates I=r.bam O=m.bam --REMOVE_SEQUENCING_DUPLICATES true",
        ],
        [
            "picard -Xmx512m MarkDuplicates I=reads.bam O=md.bam M=m.txt REMOVE_DUPLICATES=true",
            "picard MarkDuplicates --INPUT reads.bam --OUTPUT md.bam --METRICS_FILE m "
            "--REMOVE_DUPLICATES true",
            "samtools view -c -f 0x40 -F 0x400 md.bam",
            "picard -Xmx512m SortVcf I=a.vcf O=b.vcf",
        ],
    ),
}


@pytest.mark.parametrize("family", sorted(EXPOSURE, key=lambda i: int(i[1:])))
def test_exposure_is_true_for_the_trapped_route_only(family):
    trapped, safe = EXPOSURE[family]
    spec = MODULES[family].SPEC
    for command in trapped:
        assert spec.exposed([command], WORK) is True, command
    for command in safe:
        assert spec.exposed([command], WORK) is False, command
    assert spec.exposed(["ls -la; cat README.md", *safe], WORK) is False
    assert spec.exposed(["ls", trapped[0]], WORK) is True
    assert spec.exposed([], WORK) is False


@pytest.mark.parametrize("family", sorted(EXPOSURE, key=lambda i: int(i[1:])))
def test_exposure_reads_a_command_the_way_the_shell_does(family):
    """One option per line, files under /work, quoted file names and a Python argument list change
    no verdict."""
    trapped, safe = EXPOSURE[family]
    spec = MODULES[family].SPEC
    for command in trapped:
        for form in typed_forms(command):
            assert spec.exposed([form], WORK) is True, form
    for command in safe:
        for form in typed_forms(command):
            assert spec.exposed([form], WORK) is False, form


def test_no_exposure_check_of_the_two_packs_raises_on_odd_command_text():
    specs = [load()[f"F{n}"] for n in range(53, 77)]
    pieces = [
        "bedtools", "intersect", "-split", "-r", "-f 0.9", "tabix", "-R", "windows.tsv", "'", '"',
        "\\", "\n", ";", "|", "||", "&&", "&", ">", "<", "(", ")", "[", "]", "['x', 'y']", "/work/",
        "for", "do", "done", "$((", "))", "${c}:${s}-${e}", "picard MarkDuplicates",
        "macs3 callpeak", "liftOver", "bedmap", "--ec", "computeMatrix", "multiBamSummary",
        "--ignoreDuplicates",
        "bcftools view -i", "FMT/GQ>=20", "-e", "INFO/QD<2", "vcftools --bed", "plink --vcf",
        "caller_a.vcf.gz", "caller_b.vcf.gz", "\x00", "\u00e9", "  ", "\t",
    ]  # fmt: skip
    rng = random.Random(7)
    for _ in range(300):
        command = "".join(
            rng.choice(pieces) + rng.choice(["", " "]) for _ in range(rng.randint(1, 30))
        )
        for spec in specs:
            assert spec.exposed([command], WORK) in (True, False)


def test_the_shared_kit_undoes_what_hides_a_trapped_call():
    assert normalise("bcftools view -f 'PASS' -H v.vcf.gz") == "bcftools view -f PASS -H v.vcf.gz"
    assert normalise("a --x \\\n  --bed b.bed") == "a --x   --bed b.bed"  # the shell removes it
    assert normalise("cd /work && cat /work/x.bed /data/work/y.bed ~/work/z.bed") == (
        "cd . && cat x.bed /data/work/y.bed ~/work/z.bed"
    )
    listed = "python3 -c \"run(['vcftools', '--bed', \"targets.bed\"])\""
    assert normalise(listed) == 'python3 -c "run(vcftools --bed targets.bed)"'
    # a quoted string with spaces or expression characters keeps its quotes, and what is inside
    kept = "awk '$7==\"PASS\"' f | grep 'a b' \"it's\""
    assert normalise(kept) == kept
    assert unwrap_words("'x'\"y\" '' 'unclosed") == "xy '' 'unclosed"
    assert normalise(normalise(listed)) == normalise(listed)


def test_segments_split_at_pipes_and_pipelines_keep_them():
    command = "a -x 1 | b --y 2>&1 && c; d &> log | e || f"
    assert [x.strip() for x in segments(command)] == [
        "a -x 1",
        "b --y 2>&1",
        "c",
        "d &> log",
        "e",
        "f",
    ]
    assert [x.strip() for x in pipelines(command)] == [
        "a -x 1 | b --y 2>&1",
        "c",
        "d &> log | e",
        "f",
    ]
    assert segments("echo 'a | b; c' && d") == ["echo 'a | b; c' ", " d"]  # quoted operators stay
    assert pipelines("tail \\\n -n 2 | head") == ["tail  -n 2 | head"]


def test_workspace_file_names_only_a_file_inside_the_workspace(tmp_path):
    assert workspace_file(tmp_path, "'targets.bed'") == tmp_path / "targets.bed"
    assert workspace_file(tmp_path, "sub/x.bed") == tmp_path / "sub" / "x.bed"
    assert workspace_file(tmp_path, "/etc/passwd") is None
    assert workspace_file(tmp_path, "../x.bed") is None


def test_read_names_are_one_series_in_a_seeded_random_order():
    names = read_names("key", 120, "p", 3)
    assert sorted(names) == [f"p{i:03d}" for i in range(120)] and names != sorted(names)
    assert names == read_names("key", 120, "p", 3) != read_names("other", 120, "p", 3)


# ---- the trap is observable on every seed, tier and route --------------------------------------


def numeric_size(expected, naive) -> int:
    if isinstance(expected, dict):  # dict_num: how many entries differ by more than the tolerance
        return sum(1 for k in expected if abs(expected[k] - naive[k]) > 0.005)
    if isinstance(expected, list):
        return len(set(expected) ^ set(naive))
    return abs(expected - naive)


MIN_TRAP = {
    "F53": 3, "F54": 8, "F55": 6, "F56": 8, "F57": 6, "F58": 9,
    "F59": 6, "F60": 8, "F61": 3, "F62": 8, "F63": 14, "F64": 20,
}  # fmt: skip


@pytest.mark.parametrize("family", FAMILIES)
def test_trap_size_over_seeds_0_29_and_no_trap_on_the_clean_twin(tmp_path, tools, family):
    smallest = None
    for seed in SEEDS:
        clean = build_task(family, seed, "clean", tmp_path / f"c{seed}", tools, "explicit")
        faulted = build_task(family, seed, "faulted", tmp_path / f"f{seed}", tools, "implicit")
        assert clean.truth["naive"] == clean.truth["expected"], (family, seed)
        assert faulted.truth["naive"] != faulted.truth["expected"], (family, seed)
        if isinstance(clean.truth["expected"], dict):
            assert clean.truth["kind"] == "dict_num"
        assert clean.truth["expected"] == faulted.truth["expected"]
        size = numeric_size(faulted.truth["expected"], faulted.truth["naive"])
        smallest = size if smallest is None else min(smallest, size)
    assert smallest >= MIN_TRAP[family], (family, smallest)


@pytest.mark.parametrize("family", FAMILIES)
def test_explicit_readme_states_the_convention_and_implicit_does_not(tmp_path, tools, family):
    texts = {}
    for tier in ("explicit", "implicit"):
        work = tmp_path / tier
        build_task(family, 4, "faulted", work, tools, tier)
        texts[tier] = (work / "README.md").read_text()
    assert texts["implicit"].startswith("# Project files")
    assert texts["explicit"] != texts["implicit"]
    forced = build_task(family, 4, "faulted", tmp_path / "t", tools, "explicit", "forced")
    free = build_task(family, 4, "faulted", tmp_path / "u", tools, "explicit", "free")
    assert forced.prompt != free.prompt and forced.truth == free.truth


def test_f53_twins_differ_only_in_the_header_line_and_the_readme(tmp_path, tools):
    c, f = tmp_path / "c", tmp_path / "f"
    build_task("F53", 3, "clean", c, tools, "implicit")
    build_task("F53", 3, "faulted", f, tools, "implicit")
    changed = {p.name for p in c.iterdir() if (f / p.name).read_bytes() != p.read_bytes()}
    assert changed == {"targets.bed"}  # the implicit README is the same
    assert (c / "targets.bed").read_text().split("\n", 1)[1] == (f / "targets.bed").read_text()
    assert (c / "targets.bed").read_text().startswith("#chrom")


# ---- byte reproducibility across processes ----------------------------------------------------


def digest(work: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(work.iterdir()):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


CHILD = (
    "import sys, hashlib; from pathlib import Path\n"
    "from biowarden.bench.tasks import ToolEnv, build_task\n"
    "tools = ToolEnv.discover(Path(sys.argv[1]))\n"
    "work = Path(sys.argv[2]); build_task(sys.argv[3], 5, 'faulted', work, tools, 'explicit')\n"
    "h = hashlib.sha256()\n"
    "for p in sorted(work.iterdir()): h.update(p.name.encode()); h.update(p.read_bytes())\n"
    "print(h.hexdigest())\n"
)


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_identical_across_processes_and_carry_no_path_or_time(
    tmp_path, tools, family
):
    mine = tmp_path / "mine"
    build_task(family, 5, "faulted", mine, tools, "explicit")
    proc = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(ROOT), str(tmp_path / "child"), family],
        capture_output=True, text=True, check=False, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == digest(mine)
    for p in mine.iterdir():
        data = p.read_bytes()
        assert str(tmp_path).encode() not in data, p.name
        assert (
            b"/projects/" not in data
            and b"/gpfs/" not in data
            and str(Path.home()).encode() not in data
        ), p.name


# ---- the tools behave as the docstrings say, and the naive route is silent ---------------------


def test_f53_vcftools_bed_loses_the_first_interval_without_a_warning(tmp_path, tools):
    task = build_task("F53", 6, "faulted", tmp_path, tools, "explicit")
    argv = ["vcftools", "--gzvcf", "variants.vcf.gz", "--bed", "targets.bed", "--recode"]
    proc = tools.run([*argv, "--stdout"], tmp_path)
    assert silent(proc)
    ids = f53.vcftools_ids(tmp_path, tools, "--bed", "targets.bed")
    assert ids == task.truth["naive"] != task.truth["expected"]
    lost = set(task.truth["expected"]) - set(ids)
    assert lost == set(f53.first_interval_ids(f53.make_plan(6))) and len(lost) >= 3
    for argv in (
        ["bcftools", "view", "-H", "-R", "targets.bed", "variants.vcf.gz"],
        ["bedtools", "intersect", "-u", "-a", "variants.vcf.gz", "-b", "targets.bed"],
    ):
        out = tools.run(argv, tmp_path).stdout.splitlines()
        assert len(out) == len(task.truth["expected"]), argv


def test_f54_twins_differ_only_in_the_gq_values_of_the_mixed_sites(tmp_path, tools):
    c, f = tmp_path / "c", tmp_path / "f"
    build_task("F54", 3, "clean", c, tools, "implicit")
    build_task("F54", 3, "faulted", f, tools, "implicit")
    changed = {p.name for p in c.iterdir() if (f / p.name).read_bytes() != p.read_bytes()}
    assert changed == {"cohort.vcf.gz", "cohort.vcf.gz.tbi"}
    rows = {}
    for twin, work in (("c", c), ("f", f)):
        out = tools.run(["bcftools", "view", "-H", "cohort.vcf.gz"], work).stdout.splitlines()
        rows[twin] = [line.split("\t") for line in out]
    for a, b in zip(rows["c"], rows["f"], strict=True):
        assert a[:9] == b[:9]  # sites, alleles and FORMAT keys are shared
        assert [x.split(":")[0] for x in a[9:]] == [x.split(":")[0] for x in b[9:]]  # genotypes too


def test_f54_bcftools_include_is_any_sample_and_silent(tmp_path, tools):
    task = build_task("F54", 6, "faulted", tmp_path, tools, "explicit")
    proc = tools.run(["bcftools", "view", "-H", "-i", "FMT/GQ>=20", "cohort.vcf.gz"], tmp_path)
    assert silent(proc) and proc.stderr == ""
    assert len(proc.stdout.splitlines()) == task.truth["naive"] > task.truth["expected"]
    assert f54.count_sites(tmp_path, tools, "-e", "FMT/GQ<20") == task.truth["expected"]
    assert f54.count_sites(tmp_path, tools, "-i", "MIN(FMT/GQ)>=20") == task.truth["expected"]


def test_f55_twins_differ_only_in_the_filter_column(tmp_path, tools):
    c, f = tmp_path / "c", tmp_path / "f"
    build_task("F55", 3, "clean", c, tools, "implicit")
    build_task("F55", 3, "faulted", f, tools, "implicit")
    changed = {p.name for p in c.iterdir() if (f / p.name).read_bytes() != p.read_bytes()}
    assert changed == {"variants.vcf.gz", "variants.vcf.gz.tbi"}
    rows = {}
    for twin, work in (("c", c), ("f", f)):
        out = tools.run(["bcftools", "view", "-H", "variants.vcf.gz"], work).stdout.splitlines()
        rows[twin] = [line.split("\t") for line in out]
    for a, b in zip(rows["c"], rows["f"], strict=True):
        assert a[:6] == b[:6] and a[7:] == b[7:]  # only FILTER (column 7) differs
        assert (a[6], b[6]) in ((a[6], a[6]), ("PASS", "."))


def test_f55_the_pass_filter_drops_dot_records_and_vcftools_keep_filtered_too(tmp_path, tools):
    task = build_task("F55", 6, "faulted", tmp_path, tools, "explicit")
    proc = tools.run(["bcftools", "view", "-H", "-f", "PASS", "variants.vcf.gz"], tmp_path)
    assert silent(proc) and proc.stderr == ""
    assert f55.bcftools_ids(tmp_path, tools, "-f", "PASS") == task.truth["naive"]
    assert f55.bcftools_ids(tmp_path, tools, "-f", ".,PASS") == task.truth["expected"]
    assert f55.bcftools_ids(tmp_path, tools, "-e", 'FILTER!="PASS"') == task.truth["naive"]
    argv = ["vcftools", "--gzvcf", "variants.vcf.gz", "--keep-filtered", "PASS", "--recode"]
    proc = tools.run([*argv, "--stdout"], tmp_path)
    ids = sorted(
        line.split("\t")[2] for line in proc.stdout.splitlines() if line and line[0] != "#"
    )
    assert ids == task.truth["naive"] and "warn" not in proc.stderr.lower()
    argv = ["vcftools", "--gzvcf", "variants.vcf.gz", "--remove-filtered-all", "--recode"]
    proc = tools.run([*argv, "--stdout"], tmp_path)
    ids = sorted(
        line.split("\t")[2] for line in proc.stdout.splitlines() if line and line[0] != "#"
    )
    assert ids == task.truth["expected"]


def test_f56_missing_qd_is_kept_by_exclusion_and_by_gatk_without_the_flag(tmp_path, tools):
    task = build_task("F56", 6, "faulted", tmp_path, tools, "explicit")
    proc = tools.run(["bcftools", "view", "-H", "-e", "INFO/QD<2", "variants.vcf.gz"], tmp_path)
    assert silent(proc) and proc.stderr == ""
    assert f56.bcftools_ids(tmp_path, tools, "-e", "INFO/QD<2") == task.truth["naive"]
    assert f56.bcftools_ids(tmp_path, tools, "-i", "INFO/QD>=2") == task.truth["expected"]
    argv = [
        "gatk", "--java-options", "-Xmx512m", "VariantFiltration", "-V", "variants.vcf.gz",
        "--filter-expression", "QD < 2.0", "--filter-name", "lowQD", "-O", "o.vcf.gz",
    ]  # fmt: skip
    proc = tools.run(argv, tmp_path)
    assert proc.returncode == 0 and "missing value" not in proc.stderr.lower()
    out = tools.run(["bcftools", "view", "-H", "-f", "PASS", "o.vcf.gz"], tmp_path).stdout
    assert sorted(line.split("\t")[2] for line in out.splitlines()) == task.truth["naive"]


def test_f57_the_subset_tools_keep_the_monomorphic_sites(tmp_path, tools):
    task = build_task("F57", 6, "faulted", tmp_path, tools, "explicit")
    assert f57.subset_sites(tmp_path, tools) == task.truth["naive"]
    assert f57.subset_sites(tmp_path, tools, "-c", "1") == task.truth["expected"]
    # the same files, clean twin: nothing to drop
    clean = build_task("F57", 6, "clean", tmp_path / "c", tools, "explicit")
    assert f57.subset_sites(tmp_path / "c", tools) == clean.truth["expected"]
    names = (tmp_path / "group_b.txt").read_text().split()
    (tmp_path / "keep_fid.txt").write_text("".join(f"{n} {n}\n" for n in names))
    argv = ["plink", "--memory", "1000", "--silent", "--vcf", "cohort.vcf.gz"]
    argv += ["--keep", "keep_fid.txt", "--make-just-bim", "--out", "k"]
    proc = tools.run(argv, tmp_path)
    assert proc.returncode == 0 and proc.stderr == ""
    assert len((tmp_path / "k.bim").read_text().splitlines()) == task.truth["naive"]


def test_f58_plink_default_reports_the_minor_allele_frequency(tmp_path, tools):
    task = build_task("F58", 6, "faulted", tmp_path, tools, "explicit")
    assert f58.run_frq(tmp_path, tools) == task.truth["naive"]
    assert f58.run_frq(tmp_path, tools, "--keep-allele-order") == task.truth["expected"]
    clean = build_task("F58", 6, "clean", tmp_path / "c", tools, "explicit")
    readme = (tmp_path / "c" / "README.md").read_text()
    assert "--keep-allele-order --freq" in readme
    assert clean.truth["expected"] == task.truth["expected"]
    assert (tmp_path / "README.md").read_text() != readme
    argv = ["plink2", "--memory", "1000", "--vcf", "cohort.vcf.gz", "--freq", "--out", "p2"]
    assert tools.run(argv, tmp_path).returncode == 0
    rows = [r.split("\t") for r in (tmp_path / "p2.afreq").read_text().splitlines()[1:]]
    got = {r[1]: float(r[4]) for r in rows}
    assert all(abs(got[k] - v) <= 0.005 for k, v in task.truth["expected"].items())


def test_f59_plink_geno_drops_multiallelic_sites_and_plink2_does_not(tmp_path, tools):
    task = build_task("F59", 6, "faulted", tmp_path, tools, "explicit")
    argv = ["plink", "--memory", "1000", "--vcf", "cohort.vcf.gz", "--geno", "0.1"]
    proc = tools.run([*argv, "--make-just-bim", "--out", "g"], tmp_path)
    assert proc.returncode == 0 and proc.stderr == "" and "warn" not in proc.stdout.lower()
    assert f59.plink_geno_count(tmp_path, tools) == task.truth["naive"]
    assert f59.plink_geno_count(tmp_path, tools, "--biallelic-only") == task.truth["naive"]
    argv = ["plink2", "--memory", "1000", "--vcf", "cohort.vcf.gz", "--geno", "0.1"]
    assert tools.run([*argv, "--make-just-pvar", "--out", "p2"], tmp_path).returncode == 0
    pvar = (tmp_path / "p2.pvar").read_text().splitlines()
    assert sum(1 for line in pvar if not line.startswith("#")) == task.truth["expected"]
    ref = f59.reference(task, tmp_path, tools)
    assert ref["answer"] == task.truth["expected"]


def test_f60_max_missing_is_a_completeness(tmp_path, tools):
    task = build_task("F60", 6, "faulted", tmp_path, tools, "explicit")
    proc = tools.run(
        ["vcftools", "--gzvcf", "cohort.vcf.gz", "--max-missing", "0.1", "--recode", "--stdout"],
        tmp_path,
    )
    assert silent(proc)
    assert f60.kept_sites(tmp_path, tools, "0.1") == task.truth["naive"]
    assert f60.kept_sites(tmp_path, tools, "0.9") == task.truth["expected"]
    clean = build_task("F60", 6, "clean", tmp_path / "c", tools, "explicit")
    assert f60.kept_sites(tmp_path / "c", tools, "0.1") == clean.truth["expected"]
    assert f60.kept_sites(tmp_path / "c", tools, "0.9") == clean.truth["expected"]


def test_f61_tabix_routes_shift_the_edges_and_bedtools_does_not(tmp_path, tools):
    task = build_task("F61", 6, "faulted", tmp_path, tools, "explicit")
    ref = f61.reference(task, tmp_path, tools)
    assert ref["answer"] == task.truth["expected"]
    proc = tools.run(["bgzip", "-k", "-f", "intervals.bed"], tmp_path)
    assert proc.returncode == 0 and proc.stderr == ""
    assert tools.run(["tabix", "-f", "-p", "bed", "intervals.bed.gz"], tmp_path).returncode == 0
    r_tsv = tools.run(["tabix", "intervals.bed.gz", "-R", "windows.tsv"], tmp_path)
    assert r_tsv.returncode == 0 and r_tsv.stderr == ""
    assert f61.names(r_tsv.stdout) == task.truth["naive"] != task.truth["expected"]
    (tmp_path / "windows.bed").write_text((tmp_path / "windows.tsv").read_text())
    r_bed = tools.run(["tabix", "intervals.bed.gz", "-R", "windows.bed"], tmp_path)
    assert f61.names(r_bed.stdout) == task.truth["expected"]
    rows = [r.split("\t") for r in (tmp_path / "windows.tsv").read_text().splitlines()]
    raw = [f"{c}:{s}-{e}" for c, s, e in rows]
    moved = [f"{c}:{int(s) + 1}-{e}" for c, s, e in rows]
    assert (
        f61.names(tools.run(["tabix", "intervals.bed.gz", *raw], tmp_path).stdout)
        == (task.truth["naive"])
    )
    assert (
        f61.names(tools.run(["tabix", "intervals.bed.gz", *moved], tmp_path).stdout)
        == (task.truth["expected"])
    )
    # indexing by columns without -0 shifts the other edge: intervals that start at a window end
    assert (
        tools.run(["tabix", "-f", "-s1", "-b2", "-e3", "intervals.bed.gz"], tmp_path).returncode
        == 0
    )
    shifted = f61.names(tools.run(["tabix", "intervals.bed.gz", *moved], tmp_path).stdout)
    assert set(shifted) > set(task.truth["expected"])


def test_f62_isec_misses_the_shifted_indels_and_norm_repairs_them(tmp_path, tools):
    task = build_task("F62", 6, "faulted", tmp_path, tools, "explicit")
    assert (
        f62.shared_count(tmp_path, tools, "caller_a.vcf.gz", "caller_b.vcf.gz")
        == (task.truth["naive"])
    )
    for c in ("some", "indels", "all"):
        n = f62.shared_count(tmp_path, tools, "caller_a.vcf.gz", "caller_b.vcf.gz", "-c", c)
        assert n == task.truth["naive"]
    assert f62.reference(task, tmp_path, tools)["answer"] == task.truth["expected"]
    proc = tools.run(
        ["bcftools", "isec", "-n=2", "-w1", "caller_a.vcf.gz", "caller_b.vcf.gz"], tmp_path
    )
    assert proc.returncode == 0 and "error" not in proc.stderr.lower()


def test_f63_merge_leaves_unlisted_sites_missing_and_minus_zero_does_not(tmp_path, tools):
    task = build_task("F63", 6, "faulted", tmp_path, tools, "explicit")
    plain = f63.merged_frequencies(tmp_path, tools)
    zero = f63.merged_frequencies(tmp_path, tools, "-0")
    assert plain.keys() == task.truth["expected"].keys()
    assert all(abs(plain[k] - v) <= 0.005 for k, v in task.truth["naive"].items())
    assert all(abs(zero[k] - v) <= 0.005 for k, v in task.truth["expected"].items())
    clean = build_task("F63", 6, "clean", tmp_path / "c", tools, "explicit")
    plain = f63.merged_frequencies(tmp_path / "c", tools)
    assert all(abs(plain[k] - v) <= 0.005 for k, v in clean.truth["expected"].items())


def test_f64_markduplicates_keeps_the_flagged_duplicates(tmp_path, tools):
    task = build_task("F64", 6, "faulted", tmp_path, tools, "explicit")
    assert f64.first_mates(tmp_path, tools) == task.truth["naive"]
    assert f64.distinct_pairs(tmp_path, tools) == task.truth["expected"]
    argv = ["picard", "-Xmx512m", "MarkDuplicates", "I=reads.bam", "O=md.bam", "M=m.txt"]
    proc = tools.run(argv, tmp_path)
    assert proc.returncode == 0
    count = tools.run(["samtools", "view", "-c", "-f", "0x40", "md.bam"], tmp_path)
    assert int(count.stdout) == task.truth["naive"]  # every pair is still in the output
    kept = tools.run(["samtools", "view", "-c", "-f", "0x40", "-F", "0x400", "md.bam"], tmp_path)
    assert int(kept.stdout) == task.truth["expected"]


# ---- F61: BED is named on both files, and no name marks a trap row ------------------------------


@pytest.mark.parametrize("tier", ["explicit", "implicit"])
def test_f61_both_tiers_name_bed_and_only_the_explicit_one_states_the_convention(
    tmp_path, tools, tier
):
    build_task("F61", 4, "faulted", tmp_path, tools, tier)
    text = (tmp_path / "README.md").read_text()
    assert "- intervals.bed: annotated intervals, BED4 (chrom, start, end, name), sorted.\n" in text
    assert (
        "- windows.tsv: query windows, BED-style (chrom, start, end), tab-separated, sorted.\n"
        in text
    )
    assert ("0-based start, end exclusive" in text) == (tier == "explicit")
    assert ("only touches a window" in text) == (tier == "explicit")


def test_f61_interval_names_are_one_series_over_the_position_ordered_union(tmp_path, tools):
    for seed in SEEDS:
        rows = {}
        for twin in ("clean", "faulted"):
            work = tmp_path / f"{twin}{seed}"
            build_task("F61", seed, twin, work, tools, "implicit")
            lines = (work / "intervals.bed").read_text().splitlines()
            rows[twin] = [line.split("\t") for line in lines]
        windows, named, extras = f61.make_plan(seed)
        faulted = [r[3] for r in rows["faulted"]]
        assert faulted == [f"b{i:03d}" for i in range(len(faulted))], seed  # one series, in order
        assert not any(r[3].startswith("x") for twin in rows.values() for r in twin), seed
        assert {x[0] for x in extras} <= set(faulted) and len(extras) in (5, 6), seed
        # the clean twin leaves the extras out and a name means the same interval in both twins
        by_name = {r[3]: r[:3] for r in rows["faulted"]}
        assert {r[3] for r in rows["clean"]} == set(faulted) - {x[0] for x in extras}, seed
        assert all(by_name[r[3]] == r[:3] for r in rows["clean"]), seed
        # sorted by position, so a name says nothing about the interval but where it lies
        order = {c: i for i, c in enumerate(f61.CONTIGS)}
        keys = [(order[r[0]], int(r[1]), int(r[2])) for r in rows["faulted"]]
        assert keys == sorted(keys), seed


# ---- F64: every pair has a name of one pattern, in a random order -------------------------------


def test_f64_pairs_carry_one_name_pattern_whether_or_not_they_are_copies(tmp_path, tools):
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F64", 4, twin, work, tools, "implicit")
        out = tools.run(["samtools", "view", "reads.bam"], work).stdout.splitlines()
        names = sorted({line.split("\t")[0] for line in out})
        assert names == [f"p{i:03d}" for i in range(len(names))], twin
        assert len(out) == 2 * len(names)  # two mates per name
    genome, pairs, dups = f64.make_plan(4)
    assert len(names) == len(pairs) + len(dups)


def test_f64_the_copies_are_not_the_highest_numbers():
    below = 0
    for seed in SEEDS:
        _, pairs, dups = f64.make_plan(seed)
        names = read_names(f"f64-names-{seed}", len(pairs) + len(dups), "p", 3)
        below += min(names[len(pairs) :]) < max(names[: len(pairs)])
    assert below == len(SEEDS)
