"""Admission cases for the fastp rules (families F83 and F87, implicit tier).

fastp.quality_trimming (F83): `fastp -i reads.fastq -o out ... <options>`; the answer is the total
number of bases in the output. Calls with `-q`/`-u`/`-e` and no `--cut_*` option filter reads and
trim nothing: right on the clean twin (reads delivered already trimmed), wrong on the faulted one.
`--cut_right` (5' window scan) is right on both; `--cut_tail`, `--cut_front` and a fixed trim are
labelled by running them (the 3' scan cuts a base further than the task's 5' scan).

fastp.adapter_detection (F87): the same call shapes against the single-end adapter reads; the
answer is the total number of bases. Without `-a` fastp trims nothing from a file of fewer than
10,000 reads (right on the clean twin, whose reads carry no adapter); `-a AGATCGGAAGAGC` (and the
long form, and a longer prefix of the same adapter) is right on both; another adapter, `-A` and
`--detect_adapter_for_pe` are wrong on the faulted twin.
"""

from __future__ import annotations

from pathlib import Path

from science_harness.bench.admission import Case, case_source
from science_harness.bench.admission.cases_seq_common import Tpl, cases_for
from science_harness.bench.tasks import ToolEnv

# ---- fastp.quality_trimming (F83) -------------------------------------------------------------

KEY_QUALITY = "low_quality_goal"


def total_bases(name: str):
    def read(work: Path, proc) -> int:
        lines = (work / name).read_text().splitlines()
        return sum(len(line) for line in lines[1::4])

    return read


def _fastp(name: str, key_values: tuple[tuple[str, ...], tuple[str, ...]], *options: str) -> Tpl:
    out = f"out_{name}.fastq"
    argv = ("-i", "reads.fastq", "-o", out, "-j", f"{name}.json", "-h", f"{name}.html")
    argv += ("--thread", "1", *options)
    return Tpl(name, "fastp", argv, total_bases(out), *key_values)


FILTER = (("filter_reads",), ("filter_reads", "trim_tails"))
TRIM = (("trim_tails",), ("trim_tails",))

QUALITY_TEMPLATES: tuple[Tpl, ...] = (
    # --- read filters only
    _fastp("q20", FILTER, "-q", "20"),
    _fastp("q15", FILTER, "-q", "15"),
    _fastp("q20_u30", FILTER, "-q", "20", "-u", "30"),
    _fastp("u20", FILTER, "-u", "20"),
    _fastp("avg25", FILTER, "-e", "25"),
    _fastp("long_q", FILTER, "--qualified_quality_phred", "20"),
    _fastp("q20_len", FILTER, "-q", "20", "-l", "30"),
    # --- tail trimming
    _fastp("cut_right", TRIM, "--cut_right", "-W", "4", "-M", "20"),
    _fastp("cut_right_default", TRIM, "--cut_right"),
    _fastp("cut_right_short", TRIM, "-r", "-W", "4", "-M", "20"),
    _fastp("q20_cut_right", TRIM, "-q", "20", "--cut_right", "-W", "4", "-M", "20"),
    _fastp("cut_right_long", TRIM, "--cut_right", "--cut_right_window_size", "4",
           "--cut_right_mean_quality", "20"),
    _fastp("cut_tail", TRIM, "--cut_tail", "-W", "4", "-M", "20"),
    _fastp("cut_tail_short", TRIM, "-3", "-W", "4", "-M", "20"),
    _fastp("cut_front", TRIM, "--cut_front", "-W", "4", "-M", "20"),
    _fastp("fixed_trim", TRIM, "-t", "5"),
)  # fmt: skip


@case_source("fastp.quality_trimming")
def quality_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F83", KEY_QUALITY, seed, scratch, tools, QUALITY_TEMPLATES)


# ---- fastp.adapter_detection (F87) ------------------------------------------------------------

KEY_ADAPTER = "adapter_trimming"
GIVEN = (("adapter_given",), ("adapter_given",))
AUTO = (("none_needed",), ("none_needed", "adapter_given"))

ADAPTER_TEMPLATES: tuple[Tpl, ...] = (
    # --- no adapter named: automatic detection (never succeeds under 10,000 reads)
    _fastp("auto", AUTO),
    _fastp("auto_length", AUTO, "-l", "20"),
    _fastp("auto_quality", AUTO, "-q", "15"),
    _fastp("auto_window", AUTO, "-w", "1"),
    _fastp("auto_pe_flag", AUTO, "--detect_adapter_for_pe"),
    _fastp("auto_dedup", AUTO, "--dedup"),
    _fastp("disabled", AUTO, "-A"),
    # --- the adapter named
    _fastp("explicit", GIVEN, "-a", "AGATCGGAAGAGC"),
    _fastp("explicit_long", GIVEN, "--adapter_sequence", "AGATCGGAAGAGC"),
    _fastp("explicit_longer", GIVEN, "-a", "AGATCGGAAGAGCACACGTC"),
    _fastp("explicit_length", GIVEN, "-a", "AGATCGGAAGAGC", "-l", "20"),
    _fastp("explicit_quality", GIVEN, "-a", "AGATCGGAAGAGC", "-q", "15"),
    _fastp("explicit_other_adapter", GIVEN, "-a", "CTGTCTCTTATACACATCT"),
)


@case_source("fastp.adapter_detection")
def adapter_cases(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return cases_for("F87", KEY_ADAPTER, seed, scratch, tools, ADAPTER_TEMPLATES)
