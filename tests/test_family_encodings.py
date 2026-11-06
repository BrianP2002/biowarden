"""Family-specific checks for F13 phred_offset, F14 id_version and F15 log_scale.

The generic gates live in test_extra_families.py. Here: the trap changes the answer on every seed
without any tool (pure Python), the twins hold the same biology, the exposure regexes see the naive
command lines and not the safe ones, and the naive routes really produce the recorded trap when run
with awk, join and grep (skipped when a tool is missing).
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from biowarden.bench.tasks.extra import load

SEEDS = range(30)
FAMILIES = ("F13", "F14", "F15")
SPECS = load()
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F13-F15 are not registered"
)


def build(tmp_path: Path, family: str, seed: int, twin: str, tier="explicit", route="free"):
    work = tmp_path / f"{family}-{seed}-{twin}-{tier}-{route}"
    work.mkdir(parents=True, exist_ok=True)
    return work, SPECS[family].build(seed, twin, work, None, tier, route)


def run_shell(command: str, work: Path) -> str:
    env = {"PATH": os.environ["PATH"], "LC_ALL": "C"}
    proc = subprocess.run(
        ["bash", "-c", command], cwd=work, env=env, capture_output=True, text=True, check=False
    )
    return proc.stdout.strip()


def table(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines()[1:]]


# ---- the trap is observable on every seed, by pure-Python comparison -------------------------


@pytest.mark.parametrize("tier", ["explicit", "implicit"])
@pytest.mark.parametrize("family", FAMILIES)
def test_trap_changes_the_answer_on_every_seed_and_clean_twin_is_unaffected(tmp_path, family, tier):
    spec = SPECS[family]
    for seed in SEEDS:
        work_c, clean = build(tmp_path, family, seed, "clean", tier)
        work_f, faulted = build(tmp_path, family, seed, "faulted", tier)
        assert clean.truth["kind"] == faulted.truth["kind"] == "int"
        assert clean.truth["naive"] == clean.truth["expected"] == faulted.truth["expected"]
        assert faulted.truth["naive"] != faulted.truth["expected"], (family, seed)
        assert spec.naive(faulted, work_f, None)["answer"] == faulted.truth["naive"]
        assert spec.reference(faulted, work_f, None)["answer"] == faulted.truth["expected"]
        assert spec.naive(clean, work_c, None)["answer"] == clean.truth["expected"]


@pytest.mark.parametrize("family", FAMILIES)
def test_data_is_tiny_and_forced_route_dictates_a_tool(tmp_path, family):
    for seed in SEEDS:
        work, free = build(tmp_path, family, seed, "faulted")
        assert all(p.stat().st_size < 50_000 for p in work.iterdir())
        assert free.truth["expected"] > 0
    _, forced = build(tmp_path, family, 0, "faulted", route="forced")
    assert re.search(r"\b(?:awk|join)\b", forced.prompt)
    assert not re.search(r"\b(?:awk|join)\b", free.prompt)


# ---- F13 -------------------------------------------------------------------------------------


def quality_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()[3::4]


def test_f13_twins_carry_the_same_scores_in_two_encodings(tmp_path):
    for seed in SEEDS:
        work_c, _ = build(tmp_path, "F13", seed, "clean")
        work_f, _ = build(tmp_path, "F13", seed, "faulted")
        clean, faulted = (
            quality_lines(work_c / "reads.fastq"),
            quality_lines(work_f / "reads.fastq"),
        )
        assert [[ord(c) - 33 for c in q] for q in clean] == [
            [ord(c) - 64 for c in q] for q in faulted
        ]
        assert len(clean) <= 200
        assert {c for q in faulted for c in q} <= {chr(i) for i in range(ord("B"), ord("h") + 1)}
        assert min(ord(c) for q in clean for c in q) < 59 < min(ord(c) for q in faulted for c in q)
        assert max(ord(c) for q in faulted for c in q) > 74 >= max(ord(c) for q in clean for c in q)
        scores = [ord(c) - 33 for q in clean for c in q]
        assert min(scores) < 30 <= max(scores)
        assert not any(q[0] in "@+" for q in clean + faulted)


def test_f13_naive_answer_on_the_faulted_twin_is_every_base(tmp_path):
    for seed in SEEDS:
        work, task = build(tmp_path, "F13", seed, "faulted")
        total = sum(len(q) for q in quality_lines(work / "reads.fastq"))
        assert task.truth["naive"] == total > task.truth["expected"]


def test_f13_readme_states_the_encoding_only_in_the_explicit_tier(tmp_path):
    for twin, offset in (("clean", "Phred+33"), ("faulted", "Phred+64")):
        work, _ = build(tmp_path, "F13", 4, twin, "explicit")
        assert offset in (work / "README.md").read_text()
        work, _ = build(tmp_path, "F13", 4, twin, "implicit")
        assert "Phred" not in (work / "README.md").read_text()
    clean, _ = build(tmp_path, "F13", 4, "clean", "implicit")
    faulted, _ = build(tmp_path, "F13", 4, "faulted", "implicit")
    assert (clean / "README.md").read_bytes() == (faulted / "README.md").read_bytes()


F13_NAIVE = [
    'awk \'BEGIN{for(i=33;i<127;i++)o[sprintf("%c",i)]=i} NR%4==0{for(j=1;j<=length($0);j++)'
    "if(o[substr($0,j,1)]-33>=30)n++} END{print n}' reads.fastq",
    "python3 -c \"print(sum(ord(c)-33>=30 for l in open('reads.fastq').read().split('\\n')[3::4] "
    'for c in l))"',
    "python3 - <<'EOF'\nq = [ord(c) - 33 for l in open('reads.fastq') for c in l.strip()]\nEOF",
    ["cat > count.py <<'EOF'\nOFFSET = 33\nEOF", "python3 count.py reads.fastq"],
    "samtools import reads.fastq -o reads.bam",
    "awk 'NR%4==0' reads.fastq | python3 -c \"print(sum(ord(c) >= 63 for c in open(0).read()))\"",
]
F13_NAIVE += [
    # seen in the open-weight runs: a character cutoff, or a table starting at 33, no shift
    "awk 'NR%4==0{for(i=1;i<=length($0);i++)if(substr($0,i,1)>=\"?\")c++}END{print c}' reads.fastq",
    'awk \'BEGIN{for(i=33;i<=126;i++)t=t sprintf("%c",i)} NR%4==0{for(i=1;i<=length($0);i++)'
    "if(index(t,substr($0,i,1))>=30)c++}END{print c}' reads.fastq",
    # tried +33, then found the encoding: the trapped route was still executed
    [
        "awk 'NR%4==0{for(i=1;i<=length($0);i++)if(substr($0,i,1)>=\"?\")c++}END{print c}' "
        "reads.fastq",
        "python3 -c \"print(sum(ord(c) - 64 >= 30 for c in open('reads.fastq')))\"",
    ],
]
F13_SAFE = [
    "head -8 reads.fastq; ls -la",
    "python3 -c \"q=open('reads.fastq').read().split('\\n')[3::4]; print(min(q), max(q))\"",
    'awk \'BEGIN{for(i=33;i<127;i++)o[sprintf("%c",i)]=i} NR%4==0{for(j=1;j<=length($0);j++)'
    "if(o[substr($0,j,1)]-64>=30)n++} END{print n}' reads.fastq",
    "python3 -c \"o = 33 if min(map(ord, open('reads.fastq').read())) < 59 else 64\"",
    ['python3 -c "print(ord(c) - 33)"', "python3 q.py reads.fastq --phred64"],
    "python3 - <<'EOF'\nq = [ord(c) - ord('@') for c in open('reads.fastq').read()]\nEOF",
]


@pytest.mark.parametrize("command", F13_NAIVE)
def test_f13_exposure_true_for_the_naive_route(tmp_path, command):
    work, _ = build(tmp_path, "F13", 1, "faulted")
    commands = command if isinstance(command, list) else [command]
    assert SPECS["F13"].exposed(commands, work) is True


@pytest.mark.parametrize("command", F13_SAFE)
def test_f13_exposure_false_for_the_safe_route_and_unrelated_commands(tmp_path, command):
    work, _ = build(tmp_path, "F13", 1, "faulted")
    commands = command if isinstance(command, list) else [command]
    assert SPECS["F13"].exposed(commands, work) is False


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk missing")
def test_f13_awk_with_33_falls_into_the_trap_and_with_64_does_not(tmp_path):
    script = (
        'awk \'BEGIN{for(i=33;i<127;i++)o[sprintf("%%c",i)]=i} NR%%4==0{for(j=1;j<=length($0);j++)'
        "if(o[substr($0,j,1)]-%d>=30)n++} END{print n}' reads.fastq"
    )
    for seed in range(5):
        work_c, clean = build(tmp_path, "F13", seed, "clean")
        work_f, faulted = build(tmp_path, "F13", seed, "faulted")
        assert int(run_shell(script % 33, work_c)) == clean.truth["expected"]
        assert int(run_shell(script % 33, work_f)) == faulted.truth["naive"]
        assert int(run_shell(script % 64, work_f)) == faulted.truth["expected"]
        assert SPECS["F13"].exposed([script % 33], work_f)
        assert not SPECS["F13"].exposed([script % 64], work_f)


# ---- F14 -------------------------------------------------------------------------------------


def test_f14_only_genes_tsv_is_versioned_on_the_faulted_twin(tmp_path):
    for seed in SEEDS:
        work_c, _ = build(tmp_path, "F14", seed, "clean")
        work_f, _ = build(tmp_path, "F14", seed, "faulted")
        genes_c, genes_f = table(work_c / "genes.tsv"), table(work_f / "genes.tsv")
        expr_c, expr_f = table(work_c / "expr.tsv"), table(work_f / "expr.tsv")
        assert expr_c == expr_f
        assert all(re.fullmatch(r"ENSG\d{11}", r[0]) for r in genes_c + expr_c)
        assert all(re.fullmatch(r"ENSG\d{11}\.\d{1,2}", r[0]) for r in genes_f)
        assert [r[0].split(".")[0] for r in genes_f] == [r[0] for r in genes_c]
        assert [r[1:] for r in genes_f] == [r[1:] for r in genes_c]
        listed = {r[0] for r in genes_c}
        present = listed & {r[0] for r in expr_c}
        assert 10 <= len(listed) - len(present) and len(present) >= 30  # absent genes in both
        assert {r[0] for r in expr_c} - listed  # expr.tsv also holds genes that are not listed
        assert len(genes_c) <= 200 and len(expr_c) <= 200


def test_f14_readme_names_the_suffix_only_in_the_explicit_tier(tmp_path):
    work, _ = build(tmp_path, "F14", 2, "faulted", "explicit")
    assert "with version suffixes" in (work / "README.md").read_text()
    work, _ = build(tmp_path, "F14", 2, "clean", "explicit")
    assert "without version suffixes" in (work / "README.md").read_text()
    for twin in ("clean", "faulted"):
        work, _ = build(tmp_path, "F14", 2, twin, "implicit")
        assert "suffix" not in (work / "README.md").read_text()


F14_JOIN = (
    "join <(tail -n +2 genes.tsv | cut -f1 | sort) <(tail -n +2 expr.tsv | cut -f1 | sort) | wc -l"
)
F14_JOIN_SAFE = (
    "join <(tail -n +2 genes.tsv | cut -f1 | sed 's/\\.[0-9]*$//' | sort)"
    " <(tail -n +2 expr.tsv | cut -f1 | sort) | wc -l"
)
F14_GREP = "grep -c -F -f <(tail -n +2 genes.tsv | cut -f1) expr.tsv"
F14_GREP_SAFE = "grep -c -F -f <(tail -n +2 expr.tsv | cut -f1) genes.tsv"
F14_NAIVE = [
    F14_JOIN,
    F14_GREP,
    "cut -f1 genes.tsv > ids.txt; grep -F -f ids.txt expr.tsv | wc -l",
    "awk 'NR==FNR{a[$1];next} ($1 in a){n++} END{print n}' genes.tsv expr.tsv",
    "awk 'BEGIN{FS=\"\\t\"} NR==FNR{a[$1];next} $1 in a{n++} END{print n}' genes.tsv expr.tsv",
    "python3 -c \"g={l.split()[0] for l in open('genes.tsv')}; "
    "e={l.split()[0] for l in open('expr.tsv')}; print(len(g & e))\"",
    ["cat > m.py <<'EOF'\nimport sys\nEOF", "python3 m.py genes.tsv expr.tsv"],
]
F14_SAFE = [
    "head genes.tsv expr.tsv",
    "cat genes.tsv expr.tsv | head -20",
    F14_JOIN_SAFE,
    F14_GREP_SAFE,
    "awk -F'\\t' 'NR==FNR{split($1,a,\".\"); g[a[1]]; next} $1 in g{n++}' genes.tsv expr.tsv",
    "python3 -c \"g={l.split()[0].split('.')[0] for l in open('genes.tsv')}; open('expr.tsv')\"",
    "cut -c1-15 genes.tsv | sort > g.txt; cut -f1 expr.tsv | sort | join g.txt - | wc -l",
    "python3 -c \"import re; re.sub(r'\\.\\d+$', '', open('genes.tsv').read())\" expr.tsv",
]


@pytest.mark.parametrize("command", F14_NAIVE)
def test_f14_exposure_true_for_exact_string_routes(tmp_path, command):
    work, _ = build(tmp_path, "F14", 1, "faulted")
    commands = command if isinstance(command, list) else [command]
    assert SPECS["F14"].exposed(commands, work) is True


@pytest.mark.parametrize("command", F14_SAFE)
def test_f14_exposure_false_for_the_safe_route_and_for_looking(tmp_path, command):
    work, _ = build(tmp_path, "F14", 1, "faulted")
    assert SPECS["F14"].exposed([command], work) is False
    assert SPECS["F14"].exposed(["ls -la; cat README.md", "wc -l *.tsv"], work) is False


F14_NAIVE_COMBINES = [  # a raw join in one command, in every form the check recognises
    F14_JOIN,
    F14_GREP,
    "comm -12 <(cut -f1 genes.tsv | sort) <(cut -f1 expr.tsv | sort) | wc -l",
    "awk 'NR==FNR{a[$1];next} $1 in a{n++} END{print n}' genes.tsv expr.tsv",
    "python3 -c \"g={l.split()[0] for l in open('genes.tsv')}; "
    "e={l.split()[0] for l in open('expr.tsv')}; print(len(g & e))\"",
    "python3 -c \"print(len(set(open('genes.tsv')).intersection(open('expr.tsv'))))\"",
    "python3 -c \"import pandas as pd; print(len(pd.read_csv('genes.tsv', sep='\\t')"
    ".merge(pd.read_csv('expr.tsv', sep='\\t'), left_on='ensembl_gene_id', right_on='gene')))\"",
]


@pytest.mark.parametrize("raw", F14_NAIVE_COMBINES)
def test_f14_a_raw_join_is_exposure_whatever_else_the_session_does(tmp_path, raw):
    """Judged per command, as F13 and F15 are: handling elsewhere does not excuse the join."""
    work, _ = build(tmp_path, "F14", 1, "faulted")
    expose = SPECS["F14"].exposed
    assert expose([raw], work) is True
    # something unrelated that looks like handling: a substitution, a dot check, an escaped dot
    for other in (
        "sed -i 's/draft/final/' notes.txt",
        "grep -c '\\.' README.md",
        "echo ok | sed 's/o/0/'",
    ):
        assert expose([other, raw], work) is True, other
        assert expose([raw, other], work) is True, other
    # the raw join was run first and the suffix stripped afterwards (or the other way round)
    assert expose([raw, F14_JOIN_SAFE], work) is True
    assert expose([F14_JOIN_SAFE, raw], work) is True


def test_f14_a_session_that_only_handles_the_suffix_is_not_exposed(tmp_path):
    work, _ = build(tmp_path, "F14", 1, "faulted")
    expose = SPECS["F14"].exposed
    stripped_then_joined = [  # the suffix is stripped in one command, the lists joined in another
        "awk -F'\\t' 'NR>1{split($1,a,\".\"); print a[1]}' genes.tsv | sort > /tmp/g.txt",
        "tail -n +2 expr.tsv | cut -f1 | sort > /tmp/e.txt",
        "join /tmp/g.txt /tmp/e.txt | wc -l",
    ]
    assert expose(stripped_then_joined, work) is False
    script = [  # a script that strips, written by one command and run on the files by another
        "cat > m.py <<'EOF'\nimport re, sys\nids = [re.sub(r'\\.\\d+$', '', l.split()[0]) "
        "for l in open(sys.argv[1])]\nEOF",
        "python3 m.py genes.tsv expr.tsv",
    ]
    assert expose(script, work) is False
    for safe in (F14_JOIN_SAFE, F14_GREP_SAFE):
        assert (
            expose(["ls -la", "head genes.tsv expr.tsv", safe, "wc -l genes.tsv expr.tsv"], work)
            is False
        )


def test_f14_the_answer_file_and_str_join_are_not_a_join_of_the_ids(tmp_path):
    work, _ = build(tmp_path, "F14", 1, "faulted")
    expose = SPECS["F14"].exposed
    answer = (
        'echo \'{"status": "ok", "answer": 40, "method": "comm of genes.tsv and expr.tsv"}\''
        " > result.json"
    )
    writes = (
        'cat > result.json <<\'EOF\'\n{"answer": 40, "flags": ["join genes.tsv expr.tsv"]}\nEOF'
    )
    str_join = (
        "python3 -c \"print(','.join(l.split()[0] for l in open('genes.tsv')), "
        "open('expr.tsv').read()[:5])\""
    )
    for command in (answer, writes, str_join):
        assert expose([F14_JOIN_SAFE, command], work) is False, command


@pytest.mark.skipif(shutil.which("join") is None, reason="join missing")
def test_f14_join_and_grep_fall_into_the_trap_and_the_safe_routes_do_not(tmp_path):
    for seed in range(5):
        work_c, clean = build(tmp_path, "F14", seed, "clean")
        work_f, faulted = build(tmp_path, "F14", seed, "faulted")
        expected = faulted.truth["expected"]
        for command in (F14_JOIN, F14_GREP):
            assert int(run_shell(command, work_c)) == clean.truth["expected"]
            assert int(run_shell(command, work_f)) == faulted.truth["naive"] == 0
        for command in (F14_JOIN_SAFE, F14_GREP_SAFE):
            assert int(run_shell(command, work_f)) == expected
            assert int(run_shell(command, work_c)) == expected


# ---- F15 -------------------------------------------------------------------------------------


def test_f15_twins_hold_the_same_counts_and_log_values_round_trip_within_rounding(tmp_path):
    for seed in SEEDS:
        work_c, clean = build(tmp_path, "F15", seed, "clean")
        work_f, _ = build(tmp_path, "F15", seed, "faulted")
        counts, logs = table(work_c / "expr.tsv"), table(work_f / "expr.tsv")
        assert [r[0] for r in counts] == [r[0] for r in logs] and len(counts) <= 200
        peak = 0.0
        for count_row, log_row in zip(counts, logs, strict=True):
            assert all(re.fullmatch(r"\d+", v) for v in count_row[1:])
            assert all(re.fullmatch(r"\d+\.\d{4}", v) for v in log_row[1:])
            values = [float(v) for v in log_row[1:]]
            peak = max(peak, *values)
            for x, c in zip(values, map(int, count_row[1:]), strict=True):
                assert abs(2**x - 1 - c) <= (c + 1) * (2**5e-5 - 1) + 1e-9  # 4-decimal rounding
                assert c >= 14000 or round(2**x - 1) == c  # exact below ~14000 counts
            mean = sum(int(v) for v in count_row[1:]) / len(count_row[1:])
            back = sum(2**x - 1 for x in values) / len(values)
            assert abs(mean - 100) >= 2 and (mean > 100) == (back > 100)
        assert 15 < peak < 18  # "max about 17": the visible evidence
        assert clean.truth["expected"] >= 10


def test_f15_readme_flags_the_transform_by_tier(tmp_path):
    work, _ = build(tmp_path, "F15", 2, "faulted", "explicit")
    assert "log2(count + 1)" in (work / "README.md").read_text()
    work, _ = build(tmp_path, "F15", 2, "clean", "explicit")
    assert "raw read counts" in (work / "README.md").read_text()
    work, _ = build(tmp_path, "F15", 2, "faulted", "implicit")
    assert "normalised expression matrix" in (work / "README.md").read_text()
    work, _ = build(tmp_path, "F15", 2, "clean", "implicit")
    text = (work / "README.md").read_text()
    assert "expression matrix" in text and "normalised" not in text and "log" not in text


F15_AWK = (
    "awk -F'\\t' 'NR>1{s=0; for(i=2;i<=NF;i++)s+=%s; if(s/(NF-1)>100)n++} END{print n}' expr.tsv"
)
F15_NAIVE = [
    F15_AWK % "$i",
    "awk 'NR>1{s=0;for(i=2;i<=NF;i++)s+=$i} s>600{n++} END{print n}' expr.tsv",
    'python3 -c "print(sum(sum(map(float, l.split()[1:]))/6 > 100 for l in '
    "list(open('expr.tsv'))[1:]))\"",
    ["cat > mean.py <<'EOF'\nthreshold = 100\nEOF", "python3 mean.py expr.tsv"],
]
F15_NAIVE += [
    # raw-value cutoff first, then 2^x tried as a hypothesis: the trapped route was still executed
    [
        F15_AWK % "$i",
        "awk -F'\\t' 'NR>1{s=0;for(i=2;i<=NF;i++)s+=2^$i; if(s/(NF-1)>100)c++} END{print c}' "
        "expr.tsv",
    ],
]
F15_SAFE = [
    "head -5 expr.tsv; ls",
    "awk 'NR>1{for(i=2;i<=NF;i++)if($i>m)m=$i} END{print m}' expr.tsv",
    F15_AWK % "(2^$i-1)",
    "python3 -c \"rows = [l.split()[1:] for l in open('expr.tsv')][1:]; "
    'print(sum(sum(2**float(x) - 1 for x in r) > 600 for r in rows))"',
    'python3 -c "import math; print(math.exp(1.0 * math.log(2)) > 100)" expr.tsv',
    "awk 'NR>1{s=0;for(i=2;i<=NF;i++)s+=exp($i*log(2))-1; if(s>600)n++} END{print n}' expr.tsv",
]


@pytest.mark.parametrize("command", F15_NAIVE)
def test_f15_exposure_true_for_thresholding_the_file_values(tmp_path, command):
    work, _ = build(tmp_path, "F15", 1, "faulted")
    commands = command if isinstance(command, list) else [command]
    assert SPECS["F15"].exposed(commands, work) is True


@pytest.mark.parametrize("command", F15_SAFE)
def test_f15_exposure_false_for_back_transforming_and_for_looking(tmp_path, command):
    work, _ = build(tmp_path, "F15", 1, "faulted")
    assert SPECS["F15"].exposed([command], work) is False


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk missing")
def test_f15_awk_mean_of_file_values_is_zero_and_back_transformed_mean_is_right(tmp_path):
    for seed in range(5):
        work_c, clean = build(tmp_path, "F15", seed, "clean", route="forced")
        work_f, faulted = build(tmp_path, "F15", seed, "faulted", route="forced")
        assert int(run_shell(F15_AWK % "$i", work_c)) == clean.truth["expected"]
        assert run_shell(F15_AWK % "$i", work_f) == ""  # no gene passes, n is never set
        assert faulted.truth["naive"] == 0
        assert int(run_shell(F15_AWK % "(2^$i-1)", work_f)) == faulted.truth["expected"]
        assert math.isfinite(faulted.truth["expected"])
