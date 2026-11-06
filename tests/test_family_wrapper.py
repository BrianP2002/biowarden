"""Family-specific checks for F17-F20 (locus wrapper): the lab's shell wrapper around samtools,
bcftools or bedtools silently deviates from the tool.

  F17 wrap_filter    default_param    the counting wrapper adds -q 20
  F18 wrap_truncate  completeness     the ID lister pipes through head -n 50
  F19 wrap_stderr    warning_channel  the sequence extractor sends bedtools' stderr to /dev/null
  F20 wrap_drop      record_count     the record counter only looks at chr1-chr22, chrX, chrY

The generic gates live in test_extra_families.py. Here: the trap size over seeds 0-29 on both tiers,
the wrapper text and mode, the wrapper against the real tool (and the escape hatches it has) with
the minimal environment of the sandbox, a run inside the real sandbox, the tiers and routes, the
exposure rule on realistic command lines (per command, not per session), and byte reproducibility
across processes with different hash seeds.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.bench.tasks.extra import f17_wrap_filter as f17
from biowarden.bench.tasks.extra import f18_wrap_truncate as f18
from biowarden.bench.tasks.extra import f19_wrap_stderr as f19
from biowarden.bench.tasks.extra import f20_wrap_drop as f20
from biowarden.bench.tasks.extra import load
from biowarden.bench.tasks.tools import strip_loader_noise

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30)
FAMILIES = ("F17", "F18", "F19", "F20")
SPECS = load()
WORK = Path("/nonexistent-workspace")  # the exposure checks read only the command text
WRAPPERS = {"F17": f17.WRAPPER, "F18": f18.WRAPPER, "F19": f19.WRAPPER, "F20": f20.WRAPPER}
# the tool whose behaviour each wrapper changes, and a marker only the faulted wrapper has
MARKERS = {
    "F17": ("samtools", '-q "$MINQ"'),
    "F18": ("bcftools", "| head -n"),
    "F19": ("bedtools", "2>/dev/null"),
    "F20": ("bcftools", '-r "$STD"'),
}
# the smallest trap this module promises over seeds 0-29 (see the report)
MIN_TRAP = {"F17": 9, "F18": 6, "F19": 3, "F20": 12}
pytestmark = pytest.mark.skipif(
    not all(f in SPECS for f in FAMILIES), reason="F17-F20 are not registered"
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


def sandbox_env(tools: ToolEnv, work: Path, **extra: str) -> dict[str, str]:
    """The environment of sandbox.bwrap_argv: PATH holds the real tools, little else."""
    env = {"PATH": tools.path_var(), "HOME": str(work), "LC_ALL": "C", "TMPDIR": "/tmp"}
    if tools.lib_dirs:
        env["LD_LIBRARY_PATH"] = tools.lib_var()
    return {**env, **extra}


def run(tools, work, argv, **env):
    """Run argv in `work` with the sandbox's minimal environment; (code, stdout, stderr)."""
    proc = subprocess.run(
        argv,
        cwd=work,
        env=sandbox_env(tools, work, **env),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, strip_loader_noise(proc.stderr)


def call(family: str) -> list[str]:
    """The arguments the task sentence implies for the wrapper."""
    return {
        "F17": [f17.BAM, f17.REGION],
        "F18": ["variants.vcf.gz", f18.QUERY],
        "F19": [f19.BED],
        "F20": ["variants.vcf.gz"],
    }[family]


def answer_of(family: str, stdout: str):
    if family == "F18":
        return stdout.split()
    if family == "F19":
        return sum(f19.MOTIF in seq for seq in f19.fasta_records(stdout))
    return int(stdout)


def trap_size(truth: dict) -> int:
    expected, naive = truth["expected"], truth["naive"]
    return len(expected) - len(naive) if isinstance(expected, list) else expected - naive


# ---- the trap is observable on every seed, on both tiers --------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_plans_give_a_trap_of_the_promised_size_on_every_seed(family):
    module = {"F17": f17, "F18": f18, "F19": f19, "F20": f20}[family]
    sizes = []
    for seed in SEEDS:
        plan = module.make_plan(seed)
        if family == "F18":
            expected, naive = module.truth_answers(plan)
            assert naive == expected[: module.CAP] and len(expected) > module.CAP, seed
            sizes.append(len(expected) - len(naive))
        else:
            expected, naive = module.truth_counts(plan)
            sizes.append(expected - naive)
    assert min(sizes) >= MIN_TRAP[family], (family, sizes)


@pytest.mark.parametrize("tier", ["explicit", "implicit"])
@pytest.mark.parametrize("family", FAMILIES)
def test_trap_changes_the_answer_on_every_seed_and_clean_twin_is_unaffected(
    tmp_path, tools, family, tier
):
    spec = SPECS[family]
    kind = "id_list" if family == "F18" else "int"
    for seed in SEEDS:
        work_c, clean = built(tmp_path, family, seed, "clean", tools, tier)
        work_f, faulted = built(tmp_path, family, seed, "faulted", tools, tier)
        assert clean.truth["kind"] == faulted.truth["kind"] == kind
        assert clean.truth["naive"] == clean.truth["expected"] == faulted.truth["expected"]
        assert trap_size(faulted.truth) >= MIN_TRAP[family], (family, seed)
        assert spec.naive(faulted, work_f, tools)["answer"] == faulted.truth["naive"]
        assert spec.reference(faulted, work_f, tools)["answer"] == faulted.truth["expected"]
        assert spec.naive(clean, work_c, tools)["answer"] == clean.truth["expected"]
        assert spec.reference(clean, work_c, tools)["answer"] == clean.truth["expected"]


# ---- the wrapper: readable, executable, plain sh, and different only where it should be --------


@pytest.mark.parametrize("family", FAMILIES)
def test_wrapper_is_an_executable_plain_sh_script_that_the_agent_can_read(tmp_path, tools, family):
    name = WRAPPERS[family]
    tool, marker = MARKERS[family]
    for twin in ("clean", "faulted"):
        work, task = built(tmp_path, family, 2, twin, tools)
        path = work / name
        text = path.read_text()
        assert os.access(path, os.X_OK) and (path.stat().st_mode & 0o777) == 0o755
        assert text.startswith("#!/bin/sh\n") and text.endswith("\n") and "\r" not in text
        assert "bash" not in text and "[[" not in text and "<(" not in text  # portable sh
        assert re.search(rf"\b{tool}\b", text)  # the tool is called by name, never by path
        assert not re.search(r"/(?:gpfs|usr|opt|home|bin)/", text.replace("#!/bin/sh", ""))
        assert "expected" not in text.lower() and "truth" not in text.lower()
        assert (marker in text) == (twin == "faulted"), (family, twin)
        assert task.truth["kind"] in ("int", "id_list")
    # the faulted wrapper is the clean one plus a small, local edit
    clean_lines = (tmp_path / f"{family}-2-clean-explicit-free" / name).read_text().splitlines()
    faulted_lines = (tmp_path / f"{family}-2-faulted-explicit-free" / name).read_text().splitlines()
    changed = set(clean_lines) ^ set(faulted_lines)
    assert 1 <= len(changed) <= 6, (family, changed)
    assert len(faulted_lines) <= 14  # short enough to read in one `cat`


def test_the_data_files_are_the_same_in_the_twins(tmp_path, tools):
    """Same biology: only the wrapper (and, for F19, the contig names in the BED) differs."""
    for family, data in (
        ("F17", ["reads.bam", "reads.bam.bai"]),
        ("F18", ["variants.vcf.gz", "variants.vcf.gz.csi"]),
        ("F19", ["ref.fa", "ref.fa.fai"]),
        ("F20", ["variants.vcf.gz", "variants.vcf.gz.csi"]),
    ):
        for seed in (0, 7, 19):
            work_c, _ = built(tmp_path, family, seed, "clean", tools)
            work_f, _ = built(tmp_path, family, seed, "faulted", tools)
            for name in data:
                assert (work_c / name).read_bytes() == (work_f / name).read_bytes(), name
            assert (work_c / WRAPPERS[family]).read_bytes() != (
                work_f / WRAPPERS[family]
            ).read_bytes()


def test_f19_bed_differs_only_in_the_names_of_the_last_lines(tmp_path, tools):
    for seed in SEEDS:
        work_c, _ = built(tmp_path, "F19", seed, "clean", tools)
        work_f, _ = built(tmp_path, "F19", seed, "faulted", tools)
        clean = (work_c / "targets.bed").read_text().splitlines()
        faulted = (work_f / "targets.bed").read_text().splitlines()
        plan = f19.make_plan(seed)
        k = len(plan.second)
        assert len(clean) == len(faulted) == len(plan.intervals) and 8 <= k <= 10
        assert clean[:-k] == faulted[:-k]
        assert all(row.startswith("chr") for row in clean)
        for row_c, row_f in zip(clean[-k:], faulted[-k:], strict=True):
            assert row_f == row_c.removeprefix("chr") and not row_f.startswith("chr")
        fai = {line.split("\t")[0] for line in (work_f / "ref.fa.fai").read_text().splitlines()}
        assert fai == {"chr1", "chr2"}
        assert len(clean) <= 40  # tiny data


# ---- the wrapper against the real tool, in the sandbox's minimal environment -------------------


def test_f17_wrapper_filters_by_mapq_silently_and_has_its_two_escape_hatches(tmp_path, tools):
    for seed in SEEDS:
        plan = f17.make_plan(seed)
        at_19 = sum(r.mapq == 19 for r in plan.reads if f17.in_target(r))
        at_20 = sum(r.mapq == 20 for r in plan.reads if f17.in_target(r))
        assert at_19 >= 1 and at_20 >= 1
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F17", seed, twin, tools)
            code, out, err = run(tools, work, ["./lab-count.sh", *call("F17")])
            assert (code, err) == (0, ""), (seed, twin)
            assert int(out) == task.truth["naive"]
            code, out, err = run(tools, work, ["samtools", "view", "-c", *call("F17")])
            assert (code, err) == (0, "") and int(out) == task.truth["expected"]
            # -q 19 against -q 20: exactly the MAPQ-19 records; -q 20 keeps MAPQ 20
            counts = {}
            for q in (0, 19, 20, 21):
                argv = ["samtools", "view", "-c", "-q", str(q), *call("F17")]
                counts[q] = int(run(tools, work, argv)[1])
            assert counts[0] == task.truth["expected"]
            assert counts[19] - counts[20] == at_19 and counts[20] - counts[21] == at_20
            faulted_count = f17.truth_counts(plan)[1]
            assert counts[20] == faulted_count
            # the wrapper's own escape hatches (a later -q wins; the variable is read)
            hatch = run(tools, work, ["./lab-count.sh", "-q", "0", *call("F17")])
            assert hatch[0] == 0 and int(hatch[1]) == task.truth["expected"]
            hatch = run(tools, work, ["./lab-count.sh", *call("F17")], LAB_MINQ="0")
            assert hatch[0] == 0 and int(hatch[1]) == task.truth["expected"]
    # the clean wrapper passes options through, so -q 20 by hand is the faulted wrapper
    work, task = built(tmp_path, "F17", 5, "clean", tools)
    argv = ["./lab-count.sh", "-q", "20", *call("F17")]
    assert int(run(tools, work, argv)[1]) == f17.truth_counts(f17.make_plan(5))[1]


def test_f18_wrapper_prints_the_first_fifty_ids_silently(tmp_path, tools):
    for seed in SEEDS:
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F18", seed, twin, tools)
            code, out, err = run(tools, work, ["./lab-ids.sh", *call("F18")])
            assert (code, err) == (0, ""), (seed, twin)
            assert out.split() == task.truth["naive"]
            assert len(out.splitlines()) == (
                50 if twin == "faulted" else len(task.truth["expected"])
            )
            argv = ["bcftools", "query", "-r", f18.QUERY, "-f", "%ID\n", "variants.vcf.gz"]
            code, direct, err = run(tools, work, argv)
            assert (code, err) == (0, "") and direct.split() == task.truth["expected"]
            assert 56 <= len(direct.split()) <= 72
            assert out.split() == direct.split()[: len(out.split())]
            # the escape hatch, and a whole-file listing (no region) through the wrapper
            all_ids = run(
                tools, work, ["./lab-ids.sh", "variants.vcf.gz", f18.QUERY], LAB_MAXROWS="999"
            )
            assert all_ids[1].split() == task.truth["expected"]
            whole = run(tools, work, ["./lab-ids.sh", "variants.vcf.gz"], LAB_MAXROWS="999")
            assert len(whole[1].split()) == len(f18.make_plan(seed).variants)


def test_f19_wrapper_hides_the_skipped_interval_warnings_that_bedtools_prints(tmp_path, tools):
    warning = re.compile(
        r"WARNING\. chromosome \(([12])\) was not found in the FASTA file\. Skipping\."
    )
    for seed in SEEDS:
        plan = f19.make_plan(seed)
        k, n = len(plan.second), len(plan.intervals)
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F19", seed, twin, tools)
            direct = ["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "targets.bed", "-fo", "-"]
            code, out, said = run(tools, work, direct)
            assert code == 0  # the tool exits 0 either way
            code_w, out_w, said_w = run(tools, work, ["./lab-getseq.sh", "targets.bed"])
            assert (code_w, said_w) == (0, ""), (seed, twin)  # the wrapper's stderr is empty
            assert out_w == out  # the same FASTA: the wrapper only silences the channel
            records = f19.fasta_records(out_w)
            if twin == "faulted":
                lines = said.splitlines()
                assert len(lines) == k and all(warning.fullmatch(x) for x in lines), (seed, said)
                assert len(records) == n - k
                assert answer_of("F19", out_w) == task.truth["naive"] < task.truth["expected"]
            else:
                assert said == "" and len(records) == n
                assert answer_of("F19", out_w) == task.truth["expected"]
        # renaming the second panel and calling the tool directly is the way out
        fixed = "".join(iv.bed(False) for iv in plan.intervals)
        recovered = tools.run(
            ["bedtools", "getfasta", "-fi", "ref.fa", "-bed", "stdin", "-fo", "-"],
            work,
            stdin=fixed,
        )
        assert recovered.stderr == "" and len(f19.fasta_records(recovered.stdout)) == n
        assert answer_of("F19", recovered.stdout) == task.truth["expected"]


def test_f19_hits_decoys_and_the_site_count_are_planted_as_documented(tmp_path):
    for seed in SEEDS:
        plan = f19.make_plan(seed)
        genome_hits = sum(seq.count(f19.MOTIF) for seq in plan.genome.values())
        total, first_only = f19.truth_counts(plan)
        assert 8 <= total <= 13 and 3 <= total - first_only <= 5, seed
        assert first_only >= 5, seed
        # a decoy: a site sitting across an interval end is in the genome but not in the interval
        cut = [
            iv
            for iv in plan.intervals
            if f19.MOTIF not in plan.sequence(iv)
            and f19.MOTIF in plan.genome[iv.chrom][iv.start0 - 1 : iv.end0 + 1]
        ]
        assert len(cut) >= 2, seed
        assert genome_hits >= total, seed
        assert all(24 <= iv.end0 - iv.start0 <= 60 for iv in plan.intervals)


def test_f20_wrapper_counts_only_standard_contigs_silently(tmp_path, tools):
    for seed in SEEDS:
        plan = f20.make_plan(seed)
        for twin in ("clean", "faulted"):
            work, task = built(tmp_path, "F20", seed, twin, tools)
            code, out, err = run(tools, work, ["./lab-nvar.sh", "variants.vcf.gz"])
            assert (code, err) == (0, ""), (seed, twin)
            assert int(out) == task.truth["naive"]
            code, out, err = run(tools, work, ["bcftools", "view", "-H", "variants.vcf.gz"])
            assert (code, err) == (0, "") and len(out.splitlines()) == task.truth["expected"]
            code, out, err = run(tools, work, ["bcftools", "index", "-n", "variants.vcf.gz"])
            assert int(out) == task.truth["expected"] == len(plan.variants)
            # the header declares the contigs the wrapper's list lacks, and the index counts them
            rows = run(tools, work, ["bcftools", "index", "-s", "variants.vcf.gz"])[1]
            per_contig = {r.split("\t")[0]: int(r.split("\t")[2]) for r in rows.splitlines()}
            assert set(per_contig) == set(f20.CONTIGS) and all(per_contig.values())
            outside = sum(n for c, n in per_contig.items() if c not in f20.STANDARD)
            assert task.truth["expected"] - task.truth["naive"] == (
                outside if twin == "faulted" else 0
            )
            assert outside >= 12 and set(per_contig) - f20.STANDARD == {"chrM", "chrUn_KI270742v1"}


@pytest.mark.skipif(shutil.which("ksh") is None, reason="ksh missing")
@pytest.mark.parametrize("family", FAMILIES)
def test_wrappers_give_the_same_answer_under_another_posix_shell(tmp_path, tools, family):
    for seed in range(4):
        work, task = built(tmp_path, family, seed, "faulted", tools)
        code, out, err = run(tools, work, ["ksh", WRAPPERS[family], *call(family)])
        assert (code, err) == (0, ""), (family, seed)
        assert answer_of(family, out) == task.truth["naive"]


def test_wrappers_print_a_usage_line_and_fail_when_called_without_arguments(tmp_path, tools):
    for family in FAMILIES:
        work, _ = built(tmp_path, family, 1, "faulted", tools)
        code, out, err = run(tools, work, [f"./{WRAPPERS[family]}"])
        assert code == 2 and out == "" and "usage:" in err, family


def sandbox_available(tools, tmp_path) -> bool:
    from biowarden.bench.runner import sandbox

    if not sandbox.available():
        return False
    probe = tmp_path / "probe"
    probe.mkdir()
    try:
        return sandbox.run_command(probe, tools, "true", timeout=20).exit_code == 0
    except (OSError, subprocess.SubprocessError):
        return False


def test_wrappers_run_inside_the_real_sandbox_with_the_recorded_answer(tmp_path, tools):
    from biowarden.bench.runner import sandbox

    if not sandbox_available(tools, tmp_path):
        pytest.skip("bubblewrap is not usable here")
    for family in FAMILIES:
        work, task = built(tmp_path, family, 3, "faulted", tools)
        shown = sandbox.run_command(work, tools, f"./{WRAPPERS[family]} {' '.join(call(family))}")
        assert shown.exit_code == 0 and shown.stderr == "", (family, shown)
        assert answer_of(family, shown.stdout) == task.truth["naive"]


# ---- tiers, routes, evidence ----------------------------------------------------------------


QUIRK = {
    "F17": "-q 20",
    "F18": "head -n 50",
    "F19": "discards the stderr",
    "F20": "only counts records on chr1-chr22",
}


@pytest.mark.parametrize("family", FAMILIES)
def test_explicit_readme_states_the_quirk_in_one_sentence_and_implicit_does_not(
    tmp_path, tools, family
):
    work_e, explicit = built(tmp_path, family, 2, "faulted", tools, "explicit")
    work_i, implicit = built(tmp_path, family, 2, "faulted", tools, "implicit")
    text_e, text_i = (work_e / "README.md").read_text(), (work_i / "README.md").read_text()
    assert QUIRK[family] in text_e and QUIRK[family] not in text_i
    # the same lines; the explicit one adds a single sentence to the wrapper's line
    lines_e, lines_i = text_e.splitlines(), text_i.splitlines()
    assert len(lines_e) == len(lines_i)
    pairs = [(e, i) for e, i in zip(lines_e, lines_i, strict=True) if e != i]
    assert len(pairs) == 1 and pairs[0][0].startswith(pairs[0][1])
    assert pairs[0][0][len(pairs[0][1]) :].count(".") == 1  # one sentence
    assert explicit.evidence != implicit.evidence and implicit.evidence
    assert explicit.truth == implicit.truth and explicit.prompt == implicit.prompt
    # the wrapper file itself is the same in both tiers
    name = WRAPPERS[family]
    assert (work_e / name).read_bytes() == (work_i / name).read_bytes()


@pytest.mark.parametrize("family", FAMILIES)
def test_implicit_readme_is_the_same_in_both_twins_and_clean_readme_does_not_describe_a_fault(
    tmp_path, tools, family
):
    clean_i, _ = built(tmp_path, family, 2, "clean", tools, "implicit")
    faulted_i, _ = built(tmp_path, family, 2, "faulted", tools, "implicit")
    assert (clean_i / "README.md").read_bytes() == (faulted_i / "README.md").read_bytes()
    clean_e, _ = built(tmp_path, family, 2, "clean", tools, "explicit")
    faulted_e, _ = built(tmp_path, family, 2, "faulted", tools, "explicit")
    text_c, text_f = ((w / "README.md").read_text() for w in (clean_e, faulted_e))
    assert text_c != text_f and text_c.count("\n") == text_f.count("\n")
    assert QUIRK[family] not in text_c


@pytest.mark.parametrize("family", FAMILIES)
def test_forced_prompt_orders_the_wrapper_and_free_prompt_only_mentions_it(tmp_path, tools, family):
    name = WRAPPERS[family]
    _, free = built(tmp_path, family, 1, "faulted", tools, "explicit", "free")
    _, forced = built(tmp_path, family, 1, "faulted", tools, "explicit", "forced")
    assert f"use the lab's ./{name} wrapper" in forced.prompt
    assert f"./{name}" in free.prompt and "use the lab's" not in free.prompt
    assert "lab usually" in free.prompt and "lab usually" not in forced.prompt
    assert free.slot == forced.slot == SPECS[family].slot and free.truth == forced.truth
    assert forced.task_id.endswith("-forced") and free.evidence == forced.evidence


def test_slots_loci_and_ids_are_as_documented():
    expected = {
        "F17": "default_param",
        "F18": "completeness",
        "F19": "warning_channel",
        "F20": "record_count",
    }
    for family, slot in expected.items():
        spec = SPECS[family]
        assert spec.slot == slot and spec.locus == "wrapper"
        assert WRAPPERS[family] in spec.twin_files and "README.md" in spec.twin_files
    assert SPECS["F19"].twin_files == {"lab-getseq.sh", "targets.bed", "README.md"}
    for family in ("F17", "F18", "F20"):
        assert len(SPECS[family].twin_files) == 2


# ---- exposure: the wrapper ran, and nothing in that same command bypassed it -------------------

F17_TRAPPED = [
    "./lab-count.sh reads.bam chr2:201-600",
    "sh lab-count.sh reads.bam chr2:201-600",
    "bash ./lab-count.sh reads.bam chr2:201-600 2>&1",
    "cd /work && ./lab-count.sh reads.bam chr2:201-600",
    'n=$(./lab-count.sh reads.bam chr2:201-600); echo "count: $n"',
    "for r in chr2:201-600 chr1; do ./lab-count.sh reads.bam $r; done",
    "LAB_THREADS=2 ./lab-count.sh reads.bam chr2:201-600",
    "env LC_ALL=C ./lab-count.sh reads.bam chr2:201-600",
    "/work/lab-count.sh reads.bam chr2:201-600",
    "bash -c './lab-count.sh reads.bam chr2:201-600'",
    "time ./lab-count.sh reads.bam chr2:201-600",
    "./lab-count.sh reads.bam chr2:201-600 | tee count.txt",
    "ls; ./lab-count.sh reads.bam chr2:201-600 && echo done",
    # looking first and running it as is still takes the trapped route
    "cat lab-count.sh; ./lab-count.sh reads.bam chr2:201-600",
    # -q 1 and a LAB_MINQ of 10 are not the way out
    "./lab-count.sh -q 1 reads.bam chr2:201-600",
    "LAB_MINQ=10 ./lab-count.sh reads.bam chr2:201-600",
    # tried the wrapper, then corrected: the first command still executed the trap
    ["./lab-count.sh reads.bam chr2:201-600", "samtools view -c reads.bam chr2:201-600"],
    ["cat lab-count.sh", "./lab-count.sh reads.bam chr2:201-600"],
    # header-only samtools calls are not a direct computation
    "samtools view -H reads.bam | head -3; ./lab-count.sh reads.bam chr2:201-600",
]
F17_SAFE = [
    "cat lab-count.sh",
    "head -20 lab-count.sh",
    "grep -n samtools lab-count.sh",
    "sed -n 1,10p lab-count.sh",
    "ls -l lab-count.sh; file lab-count.sh; chmod +x lab-count.sh",
    "bash -n lab-count.sh",
    "echo ./lab-count.sh is the lab wrapper",
    "./lab-count.sh",
    "./lab-count.sh --help",
    "samtools view -c reads.bam chr2:201-600",
    "samtools view reads.bam chr2:201-600 | wc -l",
    "samtools idxstats reads.bam",
    "./lab-count.sh -q 0 reads.bam chr2:201-600",
    "./lab-count.sh -q0 reads.bam chr2:201-600",
    "LAB_MINQ=0 ./lab-count.sh reads.bam chr2:201-600",
    "env LAB_MINQ=0 ./lab-count.sh reads.bam chr2:201-600",
    # the same command also computes the count directly
    "./lab-count.sh reads.bam chr2:201-600; samtools view -c reads.bam chr2:201-600",
    "samtools view -c reads.bam chr2:201-600 && ./lab-count.sh reads.bam chr2:201-600",
    # the wrapper was rewritten first, or only a copy of it was run
    "sed -i 's/-q \"$MINQ\"//' lab-count.sh && ./lab-count.sh reads.bam chr2:201-600",
    "sed 's/-q \"$MINQ\"//' lab-count.sh > mine.sh && sh mine.sh reads.bam chr2:201-600",
    "cat > lab-count.sh <<'EOF'\nexec samtools view -c \"$@\"\nEOF\n./lab-count.sh reads.bam x",
]
F18_TRAPPED = [
    "./lab-ids.sh variants.vcf.gz chr2",
    "./lab-ids.sh variants.vcf.gz chr2 | wc -l",
    "bash lab-ids.sh variants.vcf.gz chr2 > ids.txt",
    "ids=$(./lab-ids.sh variants.vcf.gz chr2); echo $ids",
    "LAB_MAXROWS=60 ./lab-ids.sh variants.vcf.gz chr2",
    "bcftools view -h variants.vcf.gz | head -4; ./lab-ids.sh variants.vcf.gz chr2",
    ["./lab-ids.sh variants.vcf.gz chr2", "bcftools query -r chr2 -f '%ID\\n' variants.vcf.gz"],
]
F18_SAFE = [
    "cat lab-ids.sh",
    "head lab-ids.sh",
    "./lab-ids.sh",
    "bcftools query -r chr2 -f '%ID\\n' variants.vcf.gz",
    "bcftools view -H -r chr2 variants.vcf.gz | cut -f3",
    "bcftools index -s variants.vcf.gz",
    "LAB_MAXROWS=1000 ./lab-ids.sh variants.vcf.gz chr2",
    "env LAB_MAXROWS=500 ./lab-ids.sh variants.vcf.gz chr2 | wc -l",
    "./lab-ids.sh variants.vcf.gz chr2 | wc -l; "
    "bcftools query -f '%ID\\n' -r chr2 variants.vcf.gz | wc -l",
    "sed -i 's/head -n \"${LAB_MAXROWS:-50}\"/cat/' lab-ids.sh; ./lab-ids.sh variants.vcf.gz chr2",
]
F19_TRAPPED = [
    "./lab-getseq.sh targets.bed",
    "./lab-getseq.sh targets.bed | grep -c GAATTC",
    "./lab-getseq.sh targets.bed | grep -v '>' | grep -c GAATTC",
    "bash lab-getseq.sh targets.bed > seqs.fa",
    "./lab-getseq.sh targets.bed 2>&1 | grep -c '>'",
    "wc -l targets.bed; LAB_REF=ref.fa ./lab-getseq.sh targets.bed",
    ["cat targets.bed", "./lab-getseq.sh targets.bed | grep -c GAATTC"],
    ["./lab-getseq.sh targets.bed > seqs.fa", "bedtools getfasta -fi ref.fa -bed targets.bed"],
    ["sed 's/^1/chr1/' targets.bed > fixed.bed", "./lab-getseq.sh targets.bed"],
]
F19_SAFE = [
    "cat lab-getseq.sh",
    "cat targets.bed ref.fa.fai",
    "bedtools getfasta -fi ref.fa -bed targets.bed -fo -",
    "bedtools getfasta -fi ref.fa -bed targets.bed | grep -c GAATTC",
    "./lab-getseq.sh fixed.bed",
    "sed 's/^\\([12]\\)\\t/chr\\1\\t/' targets.bed > fixed.bed && ./lab-getseq.sh fixed.bed",
    "./lab-getseq.sh targets.bed; bedtools getfasta -fi ref.fa -bed targets.bed",
    "sed -i 's/^\\([12]\\)\\t/chr\\1\\t/' targets.bed && ./lab-getseq.sh targets.bed",
    ["sed -i 's/^1/chr1/' targets.bed", "./lab-getseq.sh targets.bed"],
    [
        "awk '{print \"chr\"$0}' targets.bed > targets.bed.new",
        "mv targets.bed.new targets.bed",
        "./lab-getseq.sh targets.bed",
    ],
    "sed -i 's/2>\\/dev\\/null//' lab-getseq.sh; ./lab-getseq.sh targets.bed",
    "bedtools --version; ls",
]
F20_TRAPPED = [
    "./lab-nvar.sh variants.vcf.gz",
    "sh lab-nvar.sh variants.vcf.gz",
    "n=$(./lab-nvar.sh variants.vcf.gz); echo $n",
    "cd /work && ./lab-nvar.sh variants.vcf.gz",
    "bcftools view -h variants.vcf.gz | grep contig; ./lab-nvar.sh variants.vcf.gz",
    "cat lab-nvar.sh; ./lab-nvar.sh variants.vcf.gz",
    ["./lab-nvar.sh variants.vcf.gz", "bcftools view -H variants.vcf.gz | wc -l"],
]
F20_SAFE = [
    "cat lab-nvar.sh",
    "./lab-nvar.sh",
    "bcftools view -H variants.vcf.gz | wc -l",
    "bcftools index -n variants.vcf.gz",
    "bcftools index -s variants.vcf.gz",
    "bcftools query -f '%CHROM\\n' variants.vcf.gz | wc -l",
    "zcat variants.vcf.gz | grep -vc '^#'",
    "./lab-nvar.sh variants.vcf.gz; bcftools index -n variants.vcf.gz",
    "bcftools view -h variants.vcf.gz | grep contig",
    "sed -i 's/ -r \"$STD\"//' lab-nvar.sh && ./lab-nvar.sh variants.vcf.gz",
]
EXPOSURE = {
    "F17": (F17_TRAPPED, F17_SAFE),
    "F18": (F18_TRAPPED, F18_SAFE),
    "F19": (F19_TRAPPED, F19_SAFE),
    "F20": (F20_TRAPPED, F20_SAFE),
}


def sessions(case):
    return case if isinstance(case, list) else [case]


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_true_when_a_command_ran_the_wrapper_unmarked(family):
    for case in EXPOSURE[family][0]:
        assert SPECS[family].exposed(sessions(case), WORK) is True, case


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_false_for_looking_direct_calls_overrides_and_rewrites(family):
    for case in EXPOSURE[family][1]:
        assert SPECS[family].exposed(sessions(case), WORK) is False, case


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_is_judged_per_command_not_per_session(family):
    trapped = sessions(EXPOSURE[family][0][0])[0]
    safe = f"cat {WRAPPERS[family]}"
    direct = {
        "F17": "samtools view -c reads.bam chr2:201-600",
        "F18": "bcftools query -r chr2 -f '%ID\\n' variants.vcf.gz",
        "F19": "bedtools getfasta -fi ref.fa -bed targets.bed",
        "F20": "bcftools view -H variants.vcf.gz | wc -l",
    }[family]
    spec = SPECS[family]
    assert spec.exposed([trapped], WORK) and spec.exposed([trapped, direct], WORK)
    assert spec.exposed([direct, trapped], WORK)  # a later direct call does not undo an earlier run
    assert spec.exposed([direct], WORK) is False and spec.exposed([], WORK) is False
    assert spec.exposed(["ls -la; cat README.md", safe], WORK) is False


@pytest.mark.parametrize("family", FAMILIES)
def test_exposure_ignores_unrelated_commands_and_other_wrappers(family):
    other = next(w for f, w in WRAPPERS.items() if f != family)
    for command in ("ls -la; cat README.md", "pwd", "echo hello", f"./{other} x y"):
        assert SPECS[family].exposed([command], WORK) is False, command


# ---- reproducibility: the same bytes in another process, whatever the hash seed ------------------


def build_in_new_process(tools, family, seed, twin, work, hashseed):
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from biowarden.bench.tasks import ToolEnv, build_task\n"
        "spec = sys.argv[1:]\n"
        f"tools = ToolEnv(tuple(map(Path, {json.dumps([str(p) for p in tools.bin_dirs])})),"
        f" tuple(map(Path, {json.dumps([str(p) for p in tools.lib_dirs])})))\n"
        "build_task(spec[0], int(spec[1]), spec[2], Path(spec[3]), tools)\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": str(hashseed)}
    proc = subprocess.run(
        [sys.executable, "-c", code, family, str(seed), twin, str(work)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("family", FAMILIES)
def test_workspace_bytes_and_modes_are_reproducible_across_processes(tmp_path, tools, family):
    for twin, hashseed in (("clean", 1), ("faulted", 12345)):
        work, _ = built(tmp_path, family, 7, twin, tools)
        again = tmp_path / f"again-{family}-{twin}"
        build_in_new_process(tools, family, 7, twin, again, hashseed)
        assert {p.name for p in work.iterdir()} == {p.name for p in again.iterdir()}
        for path in work.iterdir():
            assert path.read_bytes() == (again / path.name).read_bytes(), (family, path.name)
            assert path.stat().st_mode & 0o777 == (again / path.name).stat().st_mode & 0o777
        assert (again / WRAPPERS[family]).stat().st_mode & 0o111 == 0o111
