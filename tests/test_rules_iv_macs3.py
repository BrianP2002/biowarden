"""Rules macs3.paired_end_format, .effective_genome_size, .extsize_needs_nomodel, .peaks_xls_start,
.control_contigs.

Unit tests run on small BAM/SAM files written here and on F72 workspaces (real MACS3 runs for the
xls fact rule); the admission tests measure every rule with calls labelled by running macs3 and
bedtools (bench/admission/cases_iv_macs3.py). MACS3 takes about a second per run, so the admission
tests use 12 seeds instead of 40 (the scaled gate: 200 correct calls per 300 seeds).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import measure
from biowarden.bench.tasks import ToolEnv, build_task, common
from biowarden.bench.tasks.extra import f72_macs3_bampe as f72
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block, render_warning
from biowarden.engine.rules import iv_macs3 as rules
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, promotion_is_consistent

ROOT = Path(__file__).resolve().parents[1]
PAIRED = "macs3.paired_end_format"
GSIZE = "macs3.effective_genome_size"
EXTSIZE = "macs3.extsize_needs_nomodel"
XLS = "macs3.peaks_xls_start"
CONTROL = "macs3.control_contigs"
LAYOUT = "read_layout"
COORD = "peak_coordinates"
FRAGMENT = "fragment_length"


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or "macs3" not in env.real_bins(["macs3"]):
        pytest.skip("macs3 (or samtools/bedtools) is not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("iv_macs3")


def engine_for(spec: Spec) -> Engine:
    return Engine([spec], mode="block", retry_cap=10**9)


def outcome(engine: Engine, rule_id: str, tool: str, argv, cwd: Path, declared=None):
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == rule_id]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


def verdict(engine, rule_id, argv, cwd, declared=None) -> str:
    _, result = outcome(engine, rule_id, "macs3", argv, cwd, declared)
    return "-" if result is None else result.verdict


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_each_rule_has_the_declared_shape(spec):
    rules_by_id = {r.id: r for r in spec.rules}
    assert set(rules_by_id) == {PAIRED, GSIZE, EXTSIZE, XLS, CONTROL}
    shape = {r.id: (r.slot, r.kind, r.tier) for r in spec.rules}
    assert shape[PAIRED] == ("read_layout", "call", "T1")
    assert shape[GSIZE] == ("genome_size", "call", "T1")
    assert shape[EXTSIZE] == ("fragment_length", "call", "T0")
    assert shape[XLS] == ("index_base", "fact", "T1")
    assert shape[CONTROL] == ("completeness", "call", "T1")
    assert rules_by_id[GSIZE].declare is None and rules_by_id[CONTROL].declare is None
    for rule_id, key, values in (
        (PAIRED, LAYOUT, ["fragments", "single_end_tags"]),
        (EXTSIZE, FRAGMENT, ["model", "fixed"]),
        (XLS, COORD, ["one_based", "zero_based"]),
    ):
        declare = rules_by_id[rule_id].declare
        assert declare is not None and (declare.key, declare.values) == (key, values)
        assert declare.required is False  # optional: silent unless the data is risky
    assert rules_by_id[PAIRED].incidents == ["GEN-26"]
    engine_for(spec)  # imports the rule modules and checks every checker name
    assert all(r.check in known_checkers() for r in spec.rules)
    schema = spec.declaration_schema()
    assert schema["required"] == [] and set(schema["properties"]) == {LAYOUT, FRAGMENT, COORD}


def test_spec_rejects_a_t0_rule_without_a_declaration_and_a_fact_rule_at_t0(spec):
    data = spec.model_dump(mode="json")
    extsize = next(r for r in data["rules"] if r["id"] == EXTSIZE)
    extsize["declare"] = None
    with pytest.raises(ValidationError, match="needs a declaration"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    next(r for r in data["rules"] if r["id"] == XLS)["tier"] = "T0"
    with pytest.raises(ValidationError, match="cannot be tier T0"):
        Spec.model_validate(data)


@pytest.mark.parametrize(
    ("name", "matches"),
    [
        ("x_peaks.xls", True),
        ("run/sample_peaks.xls", True),
        ("peaks.xls", True),
        ("/abs/p.xls", True),
        ("x_peaks.xlsx", False),
        ("x_peaks.xls.bak", False),
        ("x_peaks.narrowPeak", False),
        ("x_summits.bed", False),
    ],
)
def test_file_pattern_matches_xls_names(spec, name, matches):
    rule = next(r for r in spec.rules if r.id == XLS)
    (pattern,) = rule.applies.file_patterns
    hits = [m.group(0) for m in re.finditer(pattern, f"awk '{{print}}' {name} | head")]
    assert bool(hits) is matches


# ---- the option parser -------------------------------------------------------------------------


def test_parser_reads_every_spelling():
    p = rules.parse_callpeak(
        ("callpeak", "-t", "a.bam", "b.bam", "-c", "c.bam", "-fBAMPE", "-g", "hs", "-q0.01")
    )
    assert p.treatment == ["a.bam", "b.bam"] and p.values["--control"] == ["c.bam"]
    assert (p.fmt, p.last("--gsize"), p.last("--qvalue")) == ("BAMPE", "hs", "0.01")
    p = rules.parse_callpeak(("callpeak", "--treatment=a.bam", "--format=bam", "--gsize=1e9"))
    assert (p.treatment, p.fmt, p.last("--gsize")) == (["a.bam"], "BAM", "1e9")
    p = rules.parse_callpeak(
        ("callpeak", "-t", "a.bam", "--nomodel", "--ext", "150", "-m", "5", "50")
    )
    assert p.last("--extsize") == "150" and "--nomodel" in p.flags
    assert p.values["--mfold"] == ["5", "50"] and p.fmt == "AUTO"
    p = rules.parse_callpeak(
        ("callpeak", "-t", "a.bam", "-n", "x", "--outdir", "o", "-B", "--SPMR")
    )
    assert p.last("--name") == "x" and {"-B", "--SPMR"} <= p.flags and p.treatment == ["a.bam"]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("hs", 2_913_022_398),
        ("MM", 2_652_783_500),
        ("ce", 100_286_401),
        ("dm", 142_573_017),
        ("1e9", 10**9),
        ("1.0e+9", 10**9),
        ("2.7e9", 2_700_000_000),
        ("50000", 50_000),
        ("50,000", 50_000),
        ("big", None),
    ],
)
def test_gsize_parser(text, value):
    assert rules.parse_gsize(text) == value


# ---- data helpers ------------------------------------------------------------------------------

HUMAN = {"chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559, "chrX": 156_040_895}


def write_sam(path: Path, contigs: dict[str, int], records: list[tuple[str, int, str, int]]):
    """records: (name, flag, contig, pos); 20M, MAPQ 60."""
    lines = ["@HD\tVN:1.6\tSO:coordinate"] + [f"@SQ\tSN:{c}\tLN:{n}" for c, n in contigs.items()]
    for name, flag, contig, pos in records:
        mate = "=" if flag & 1 else "*"
        lines.append(
            f"{name}\t{flag}\t{contig}\t{pos}\t60\t20M\t{mate}\t{pos + 100 if flag & 1 else 0}\t"
            f"{100 if flag & 1 else 0}\t{'A' * 20}\t{'I' * 20}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_bam(tools, tmp: Path, name: str, contigs, records) -> Path:
    sam = write_sam(tmp / f"{name}.sam", contigs, records)
    tools.run(["samtools", "view", "-b", "-o", f"{name}.bam", sam.name], tmp)
    sam.unlink()
    return tmp / f"{name}.bam"


SMALL = {"chr1": 30_000, "chr2": 20_000}
PAIRS = [
    ("a", 99, "chr1", 100),
    ("a", 147, "chr1", 200),
    ("b", 99, "chr1", 500),
    ("b", 147, "chr1", 600),
]
SINGLE = [("a", 0, "chr1", 100), ("b", 16, "chr1", 500), ("c", 0, "chr1", 900)]
BASE = ("callpeak", "-n", "x", "-q", "0.01", "-g", "50000")


@pytest.fixture(scope="module")
def data(tools, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("iv-macs3")
    write_bam(tools, tmp, "paired", SMALL, PAIRS)
    write_bam(tools, tmp, "single", SMALL, SINGLE)
    write_bam(tools, tmp, "mixed", SMALL, PAIRS[:2] + SINGLE)
    write_bam(tools, tmp, "human", HUMAN, PAIRS)
    write_bam(tools, tmp, "empty", SMALL, [])
    write_sam(tmp / "paired_text.sam", SMALL, PAIRS)
    (tmp / "junk.bam").write_bytes(b"\x1f\x8b" + b"not a bam" * 30)
    (tmp / "notes.bed").write_text("chr1\t1\t10\n")
    return tmp


# ---- macs3.paired_end_format -------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt", [[], ["-f", "BAM"], ["-f", "AUTO"], ["--format=bam"], ["-fSAM"], ["--format", "BAM"]]
)
def test_single_end_format_on_a_paired_bam_fails_and_the_declaration_resolves_it(spec, data, fmt):
    eng = engine_for(spec)
    argv = [*BASE, "-t", "paired.bam", "--nomodel", "--extsize", "200", *fmt]
    decision, result = outcome(eng, PAIRED, "macs3", argv, data)
    assert decision.action == "block" and result.code == "paired_end_read_as_single_end"
    text = render_block(decision)
    for needle in ("paired-end", "Paired-End mode is off", "second mate", "-f BAMPE", "FIX:"):
        assert needle in text, needle
    assert verdict(eng, PAIRED, argv, data, {LAYOUT: "single_end_tags"}) == "pass"
    _, contradiction = outcome(eng, PAIRED, "macs3", argv, data, {LAYOUT: "fragments"})
    assert (
        contradiction.verdict == "fail"
        and "you declared read_layout=fragments" in contradiction.message
    )


@pytest.mark.parametrize("fmt", [["-f", "BAMPE"], ["--format=BAMPE"], ["-fbampe"]])
def test_bampe_passes_and_contradicts_only_a_single_end_declaration(spec, data, fmt):
    eng = engine_for(spec)
    argv = [*BASE, "-t", "paired.bam", *fmt]
    assert verdict(eng, PAIRED, argv, data) == "pass"
    assert verdict(eng, PAIRED, argv, data, {LAYOUT: "fragments"}) == "pass"
    _, result = outcome(eng, PAIRED, "macs3", argv, data, {LAYOUT: "single_end_tags"})
    assert result.verdict == "fail" and result.code == "declared_layout_contradicts_call"


def test_single_end_data_unreadable_files_and_other_formats_are_left_alone(spec, data):
    eng = engine_for(spec)
    for argv in (
        [*BASE, "-t", "single.bam", "--nomodel"],
        [*BASE, "-t", "single.bam", "-f", "BAM", "--nomodel"],
        [*BASE, "-t", "single.bam", "-f", "BAMPE"],  # stops with a traceback in MACS3: loud
        [*BASE, "-t", "absent.bam"],
        [*BASE, "-t", "junk.bam"],
        [*BASE, "-t", "notes.bed", "-f", "BED", "--nomodel"],
        [*BASE, "-t", "paired.bam", "-f", "BEDPE"],
        [*BASE, "-t", "empty.bam", "--nomodel"],
        ["callpeak", "--help"],
    ):
        assert verdict(eng, PAIRED, argv, data) in ("pass", "-"), argv


def test_a_mixed_bam_is_unknown_and_a_paired_sam_text_file_fails(spec, data):
    eng = engine_for(spec)
    decision, result = outcome(eng, PAIRED, "macs3", [*BASE, "-t", "mixed.bam", "--nomodel"], data)
    assert result.verdict == "unknown" and result.code == "mixed_layout"
    assert decision.action == "warn"  # unknown never blocks
    assert "UNVERIFIED" in render_warning(decision)
    assert verdict(eng, PAIRED, [*BASE, "-t", "paired_text.sam", "-f", "SAM"], data) == "fail"


def test_several_treatment_files_are_judged_by_the_first_one(spec, data):
    eng = engine_for(spec)
    argv = [*BASE, "-t", "paired.bam", "single.bam", "-c", "single.bam", "--nomodel"]
    assert verdict(eng, PAIRED, argv, data) == "fail"
    argv = [*BASE, "-t", "single.bam", "paired.bam", "--nomodel"]
    assert verdict(eng, PAIRED, argv, data) == "pass"


def test_the_paired_rule_reads_files_so_a_t0_context_refuses(spec, data):
    match = Match(next(r for r in spec.rules if r.id == PAIRED))
    call = Call("macs3", (*BASE, "-t", "paired.bam", "--nomodel"), data)
    with pytest.raises(TierViolation):
        known_checkers()["iv_macs3_paired_format"](call, match, Ctx(data, "T0"))


# ---- macs3.effective_genome_size ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("g", "expected"),
    [
        (["-g", "50000"], "pass"),
        (["-g", "5e4"], "pass"),
        (["--gsize=50000"], "pass"),
        (["-g", "100000"], "pass"),  # a factor 2: inside the tolerance
        (["-g", "250000"], "pass"),  # exactly 5x
        (["-g", "10000"], "pass"),  # exactly one fifth
        (["-g", "hs"], "fail"),
        (["-g", "mm"], "fail"),
        (["-g", "ce"], "fail"),
        (["-g", "dm"], "fail"),
        (["-g", "2.7e9"], "fail"),
        (["-g", "1e6"], "fail"),
        (["-g", "2000"], "fail"),
        ([], "fail"),  # the default is hs
    ],
)
def test_genome_size_against_a_50_kb_header(spec, data, g, expected):
    eng = engine_for(spec)
    argv = ["callpeak", "-n", "x", "-q", "0.01", "-t", "single.bam", "--nomodel", *g]
    assert verdict(eng, GSIZE, argv, data) == expected


def test_genome_size_message_names_the_numbers_and_the_fix(spec, data):
    eng = engine_for(spec)
    argv = ["callpeak", "-t", "single.bam", "-q", "0.01", "-g", "hs", "--nomodel"]
    decision, result = outcome(eng, GSIZE, "macs3", argv, data)
    text = render_block(decision)
    assert decision.action == "block" and result.expected == 50_000 and result.got == 2_913_022_398
    for needle in ("-g hs", "2,913,022,398", "50,000", "58,260 times larger", "FIX: use -g 50000"):
        assert needle in text, needle
    argv = ["callpeak", "-t", "single.bam", "-g", "1e6", "--nomodel"]
    assert "20 times larger" in outcome(eng, GSIZE, "macs3", argv, data)[1].message
    argv = ["callpeak", "-t", "single.bam", "--nomodel"]
    assert "no -g (the default is hs)" in outcome(eng, GSIZE, "macs3", argv, data)[1].message


def test_genome_size_is_silent_on_a_human_size_header_and_on_other_inputs(spec, data):
    eng = engine_for(spec)
    for g in ([], ["-g", "hs"], ["-g", "mm"], ["-g", "2.9e9"], ["-g", "3e9"], ["-g", "1e9"]):
        argv = ["callpeak", "-t", "human.bam", "--nomodel", *g]
        assert verdict(eng, GSIZE, argv, data) == "pass", g
    assert verdict(eng, GSIZE, ["callpeak", "-t", "human.bam", "-g", "ce"], data) == "fail"
    for argv in (
        ["callpeak", "-t", "notes.bed", "-f", "BED"],
        ["callpeak", "-t", "absent.bam"],
        ["callpeak", "-t", "junk.bam"],
        ["callpeak", "-t", "paired_text.sam", "-f", "SAM", "-g", "50000"],
    ):
        assert verdict(eng, GSIZE, argv, data) == "pass", argv
    assert verdict(eng, GSIZE, ["callpeak", "-t", "paired_text.sam", "-f", "SAM"], data) == "fail"


def test_genome_size_has_no_declaration_to_pass_with(spec, data):
    argv = ["callpeak", "-t", "single.bam", "-g", "hs", "--nomodel"]
    _, result = outcome(engine_for(spec), GSIZE, "macs3", argv, data, {"anything": "x"})
    assert result.verdict == "fail"


# ---- macs3.extsize_needs_nomodel ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["callpeak", "-t", "a.bam", "--extsize", "150"], "fail"),
        (["callpeak", "-t", "a.bam", "--extsize=150"], "fail"),
        (["callpeak", "-t", "a.bam", "-f", "BAM", "--extsize", "150", "--shift", "-75"], "fail"),
        (["callpeak", "-t", "a.bam", "--nomodel", "--extsize", "150"], "pass"),
        (["callpeak", "-t", "a.bam", "--extsize", "150", "--nomodel"], "pass"),
        (["callpeak", "-t", "a.bam", "-f", "BAMPE", "--extsize", "150"], "pass"),
        (["callpeak", "-t", "a.bam", "--format=BEDPE", "--extsize", "150"], "pass"),
        (["callpeak", "-t", "a.bam", "--shift", "-50"], "-"),  # --shift is applied without a model
        (["callpeak", "-t", "a.bam", "--nomodel"], "-"),
        (["callpeak", "-t", "a.bam"], "-"),
        (["predictd", "-i", "a.bam", "--extsize", "150"], "-"),
    ],
)
def test_extsize_needs_nomodel(spec, tmp_path, argv, expected):
    assert verdict(engine_for(spec), EXTSIZE, argv, tmp_path) == expected


def test_extsize_declaration_resolves_it_and_the_message_has_the_fix(spec, tmp_path):
    eng = engine_for(spec)
    argv = ["callpeak", "-t", "a.bam", "--extsize", "150"]
    decision, result = outcome(eng, EXTSIZE, "macs3", argv, tmp_path)
    text = render_block(decision)
    assert decision.action == "block" and result.code == "extsize_without_nomodel"
    for needle in ("--extsize 150", "predicted fragment length", "FIX: add --nomodel"):
        assert needle in text, needle
    assert verdict(eng, EXTSIZE, argv, tmp_path, {FRAGMENT: "model"}) == "pass"
    assert verdict(eng, EXTSIZE, argv, tmp_path, {FRAGMENT: "fixed"}) == "fail"
    _, bad = outcome(eng, EXTSIZE, "macs3", argv, tmp_path, {FRAGMENT: "predicted"})
    assert bad.code == "declaration_invalid"


def test_extsize_rule_is_tier_t0_and_reads_no_file(spec, tmp_path):
    rule = next(r for r in spec.rules if r.id == EXTSIZE)
    call = Call("macs3", ("callpeak", "-t", "absent.bam", "--extsize", "150"), tmp_path)
    result = known_checkers()["iv_macs3_extsize_nomodel"](call, Match(rule), Ctx(tmp_path, "T0"))
    assert result.verdict == "fail"  # a T0 context would raise if the checker opened a file


# ---- macs3.peaks_xls_start (needs a real MACS run) ---------------------------------------------


@pytest.fixture(scope="module")
def peaks_dir(tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("iv-macs3-xls")
    build_task("F72", 3, "faulted", work, tools, "implicit", "free")
    argv = ["macs3", "callpeak", "-t", "reads.bam", "-f", "BAMPE", "-g", str(f72.genome_size(work))]
    proc = tools.run([*argv, "-q", "0.01", "-n", "run", "--outdir", "."], work)
    assert proc.returncode == 0
    (work / "targets.bed").write_text("chr1\t100\t200\tt1\n")
    # a table named .xls that MACS did not write
    (work / "table.xls").write_text("gene\tcount\nA\t3\n")
    return work


STRIP = "grep -v '^#' run_peaks.xls | awk '$2 ~ /^[0-9]+$/'"
XLS_WRONG = [
    f"{STRIP} | cut -f1-3 | bedtools intersect -a targets.bed -b - -u",
    "awk '!/^#/ && $2 ~ /^[0-9]+$/ {print $1,$2,$3}' run_peaks.xls | bedtools sort -i -",
    f"{STRIP} | cut -f 1,2,3 | bedtools merge -i -",
    "python3 -c \"import sys; rows=[l.split('\\t') for l in open('run_peaks.xls')]; "
    'print([int(r[1]) for r in rows if r[1].isdigit()])"',
    "python3 -c \"import pandas as pd; d=pd.read_csv('run_peaks.xls', sep='\\t', comment='#'); "
    'print(d.start.min())"',
]
XLS_RIGHT = [
    f"{STRIP} | awk 'BEGIN{{OFS=\"\\t\"}} {{print $1,$2-1,$3}}' | "
    "bedtools intersect -a targets.bed -b -",
    "python3 -c \"rows=[l.split('\\t') for l in open('run_peaks.xls')]; "
    'print([int(r[1]) - 1 for r in rows if r[1].isdigit()])"',
    "bedtools intersect -a targets.bed -b run_peaks.narrowPeak -u",
    "awk '$2 ~ /^[0-9]+$/ {print $2}' run_peaks.xls > /dev/null  # start is one-based here",
    "grep -v '^#' run_peaks.xls | head -3",
    "head -20 run_peaks.xls",
    "wc -l run_peaks.xls",
    "grep -c peak run_peaks.xls",
    "cat run_peaks.xls",
    "awk '!/^#/ && $9 > 10 {n++} END{print n}' run_peaks.xls",  # no start column: fold only
]


@pytest.mark.parametrize("command", XLS_WRONG)
def test_xls_start_used_as_a_bed_start_fails(spec, peaks_dir, command):
    eng = engine_for(spec)
    decision, result = outcome(eng, XLS, "bash", ["-c", command], peaks_dir)
    assert result is not None and result.verdict == "fail", command
    text = render_block(decision)
    for needle in ("1-based", "narrowPeak start + 1", "run_peaks.narrowPeak", "FIX:"):
        assert needle in text, needle
    assert (
        outcome(eng, XLS, "bash", ["-c", command], peaks_dir, {COORD: "one_based"})[1].verdict
        == "pass"
    )
    wrong = outcome(eng, XLS, "bash", ["-c", command], peaks_dir, {COORD: "zero_based"})[1]
    assert wrong.verdict == "fail" and "you declared peak_coordinates=zero_based" in wrong.message


@pytest.mark.parametrize("command", XLS_RIGHT)
def test_commands_that_handle_the_base_or_only_look_stay_silent(spec, peaks_dir, command):
    decision, result = outcome(engine_for(spec), XLS, "bash", ["-c", command], peaks_dir)
    assert decision.action == "allow", command
    assert result is None or result.verdict == "pass"


def test_direct_tool_calls_and_non_macs_tables(spec, peaks_dir):
    eng = engine_for(spec)
    _, result = outcome(eng, XLS, "awk", ["{print $1,$2,$3}", "run_peaks.xls"], peaks_dir)
    assert result.verdict == "fail"
    _, result = outcome(eng, XLS, "awk", ["{print $1,$2-1,$3}", "run_peaks.xls"], peaks_dir)
    assert result.verdict == "pass"
    # a .xls file that MACS did not write is not judged
    _, result = outcome(eng, XLS, "awk", ["{print $1,$2}", "table.xls"], peaks_dir)
    assert result.verdict == "pass"
    assert outcome(eng, XLS, "awk", ["{print $1,$2}", "missing_peaks.xls"], peaks_dir)[1] is None


def test_the_xls_header_and_the_start_column_are_what_the_rule_assumes(peaks_dir):
    xls = (peaks_dir / "run_peaks.xls").read_text().splitlines()
    assert xls[0].startswith("# This file is generated by MACS")
    rows = [line.split("\t") for line in xls if line and not line.startswith("#")][1:]
    narrow = [
        line.split("\t")
        for line in (peaks_dir / "run_peaks.narrowPeak").read_text().split("\n")
        if line
    ]
    assert [int(r[1]) for r in rows] == [int(n[1]) + 1 for n in narrow]
    assert [int(r[2]) for r in rows] == [int(n[2]) for n in narrow]
    summits = [
        int(line.split("\t")[1])
        for line in (peaks_dir / "run_summits.bed").read_text().split("\n")
        if line
    ]
    assert [int(r[4]) for r in rows] == [s + 1 for s in summits]  # abs_summit is 1-based
    assert [s - int(n[1]) for s, n in zip(summits, narrow, strict=True)] == [
        int(n[9]) for n in narrow
    ]


# ---- F72 workspaces with the real call ---------------------------------------------------------


def test_the_family_calls_are_judged_as_the_family_says(spec, tools, tmp_path):
    eng = engine_for(spec)
    for twin in ("clean", "faulted"):
        work = tmp_path / twin
        build_task("F72", 4, twin, work, tools, "implicit", "free")
        size = str(f72.genome_size(work))
        naive = [
            "callpeak",
            "-t",
            "reads.bam",
            "-g",
            size,
            "-q",
            "0.01",
            "--nomodel",
            "--extsize",
            "200",
        ]
        reference = ["callpeak", "-t", "reads.bam", "-g", size, "-q", "0.01", "-f", "BAMPE"]
        assert verdict(eng, PAIRED, naive, work) == "fail", twin
        assert verdict(eng, PAIRED, reference, work) == "pass", twin
        assert verdict(eng, GSIZE, naive, work) == "pass"
        hs = ["callpeak", "-t", "reads.bam", "-q", "0.01", "-f", "BAMPE"]
        assert verdict(eng, GSIZE, hs, work) == "fail"
        assert verdict(eng, EXTSIZE, naive, work) == "pass"  # --nomodel is there


# ---- macs3.control_contigs ---------------------------------------------------------------------


def indexed_bam(tools, tmp: Path, name: str, records) -> Path:
    sam = write_sam(tmp / f"{name}.sam", SMALL, records)
    tools.run(["samtools", "sort", "-o", f"{name}.bam", sam.name], tmp)
    tools.run(["samtools", "index", f"{name}.bam"], tmp)
    sam.unlink()
    return tmp / f"{name}.bam"


def reads(contig: str, n: int, prefix: str, flag: int = 0):
    return [(f"{prefix}{contig}{i}", flag, contig, 100 + 50 * i) for i in range(n)]


@pytest.fixture(scope="module")
def ctl_data(tools, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("iv-macs3-ctl")
    both = reads("chr1", 30, "t") + reads("chr2", 30, "t")
    indexed_bam(tools, tmp, "treat", both)
    indexed_bam(tools, tmp, "ctl_both", reads("chr1", 12, "c") + reads("chr2", 12, "c"))
    indexed_bam(tools, tmp, "ctl_chr1", reads("chr1", 40, "c"))
    indexed_bam(tools, tmp, "ctl_chr2", reads("chr2", 40, "c"))
    indexed_bam(tools, tmp, "treat_few", reads("chr1", 30, "t") + reads("chr2", 5, "t"))
    # the same files without an index: the sniff counts the records instead
    write_bam(tools, tmp, "plain_treat", SMALL, both)
    write_bam(tools, tmp, "plain_ctl_chr1", SMALL, reads("chr1", 40, "c"))
    write_bam(tools, tmp, "plain_ctl_both", SMALL, reads("chr1", 12, "c") + reads("chr2", 12, "c"))
    return tmp


@pytest.mark.parametrize("prefix", ["", "plain_"])
@pytest.mark.parametrize("fmt", [[], ["-f", "BAM"], ["-f", "BAMPE"]])
def test_a_control_without_reads_on_a_treatment_contig_fails(spec, ctl_data, prefix, fmt):
    eng = engine_for(spec)
    argv = ["callpeak", "-t", f"{prefix}treat.bam", "-c", f"{prefix}ctl_chr1.bam", *fmt]
    decision, result = outcome(eng, CONTROL, "macs3", argv, ctl_data)
    assert decision.action == "block" and result.code == "control_lacks_contigs"
    text = render_block(decision)
    assert "chr2 (30 treatment reads)" in text and "dropped without a message" in text
    assert result.expected == ["chr2"]
    argv = ["callpeak", "-t", f"{prefix}treat.bam", "-c", f"{prefix}ctl_both.bam", *fmt]
    assert verdict(eng, CONTROL, argv, ctl_data) == "pass"


def test_control_contigs_edge_cases(spec, ctl_data):
    eng = engine_for(spec)

    def run(*argv):
        return verdict(eng, CONTROL, ["callpeak", *argv], ctl_data)

    assert run("-t", "treat.bam") == "pass"  # no control
    assert run("-t", "treat.bam", "-c", "ctl_chr1.bam", "ctl_chr2.bam") == "pass"  # summed
    assert run("--treatment=treat.bam", "--control=ctl_chr1.bam") == "fail"
    assert run("-t", "treat_few.bam", "-c", "ctl_chr1.bam") == "pass"  # chr2: 5 reads, under 10
    assert run("-t", "treat.bam", "-c", "absent.bam") == "pass"
    assert run("-t", "treat.bam", "-c", "treat.bam", "-f", "BED") == "pass"
    assert run("-t", "ctl_chr1.bam", "-c", "treat.bam") == "pass"  # treatment on fewer contigs


def test_bai_counts_equal_samtools_idxstats(tools, ctl_data):
    names = ["chr1", "chr2"]
    idx = tools.run(["samtools", "idxstats", "treat.bam"], ctl_data).stdout.splitlines()
    wanted = {r.split("\t")[0]: int(r.split("\t")[2]) for r in idx if r.split("\t")[0] in names}
    counts = rules.bai_mapped((ctl_data / "treat.bam.bai").read_bytes())
    assert dict(zip(names, counts, strict=True)) == wanted == {"chr1": 30, "chr2": 30}
    assert rules.bai_mapped(b"BAI\x01\x05\x00") is None and rules.bai_mapped(b"junk") is None


# ---- admission ---------------------------------------------------------------------------------


def admission(spec, tools, tmp_path, rule_id: str, seeds: int):
    seeds = min(seeds, ADMISSION_SEEDS)
    stats = measure(rule_id, [spec], range(seeds), tools, tmp_path)
    needed = max(1, 200 * seeds // 300)  # the gate's 200 correct calls, scaled to the seeds run
    assert stats.admitted() and stats.killed() > 0
    assert stats.right_declared.n >= needed and stats.right_declared.failed == 0
    assert stats.cells[("right", False)].n >= needed and stats.cells[("right", False)].failed == 0
    assert all(cell.crashed == 0 and cell.unknown == 0 for cell in stats.cells.values())
    assert promotion_is_consistent(stats)
    return stats


def test_admission_paired_end_format(spec, tools, tmp_path):
    stats = admission(spec, tools, tmp_path, PAIRED, 12)
    # every single-end call on the paired data is wrong and fails unless it declares that intent
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n
    assert stats.cells[("wrong", True)].failed > 0


def test_admission_effective_genome_size(spec, tools, tmp_path):
    stats = admission(spec, tools, tmp_path, GSIZE, 12)
    wrong = stats.cells[("wrong", False)]
    assert (
        wrong.failed >= wrong.n - 12
    )  # only the factor-2 call per seed is out of the rule's reach


def test_admission_extsize_needs_nomodel(spec, tools, tmp_path):
    stats = admission(spec, tools, tmp_path, EXTSIZE, 12)
    assert stats.cells[("wrong", False)].failed > 0
    assert stats.cells[("wrong", True)].failed > 0  # `fixed` declared over a call without --nomodel


def test_admission_peaks_xls_start(spec, tools, tmp_path):
    stats = admission(spec, tools, tmp_path, XLS, 12)
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n


def test_admission_control_contigs(spec, tools, tmp_path):
    stats = admission(spec, tools, tmp_path, CONTROL, 12)
    assert stats.cells[("wrong", False)].failed == stats.cells[("wrong", False)].n


def test_the_build_helpers_write_flat_workspaces(tools, tmp_path):
    from biowarden.bench.admission import cases_iv_macs3 as cases

    work = tmp_path / "g"
    work.mkdir()
    assert cases.build_gsize(0, work, tools) == 180_000
    assert sorted(p.name for p in work.iterdir() if p.is_dir()) == []
    work = tmp_path / "m"
    work.mkdir()
    assert cases.build_model_data(0, work, tools) == 1_000_000
    common_files = {p.name for p in work.iterdir()}
    assert {"model.bam", "model.bam.bai"} <= common_files
    assert common.task_id("F72", 1, "clean", "implicit", "free") == "F72-s01-clean-implicit"
