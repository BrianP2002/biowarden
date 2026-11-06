"""Rule vcf.indel_interval_anchor: the checker on real argv, the spec, tier honesty, admission.

The unit tests run on small VCFs written here (and on F10 workspaces built with the real
bcftools); the admission test measures the rule over 300 seeds with calls labelled by running the
real tools (bench/admission/cases_vcf_intervals.py).
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from biowarden.bench.admission import measure
from biowarden.bench.admission.cases_vcf_intervals import TEMPLATES
from biowarden.bench.tasks import ToolEnv, build_task
from biowarden.engine import Call, Engine, Spec
from biowarden.engine.checks import Ctx, Match, TierViolation, known_checkers
from biowarden.engine.render import render_block
from biowarden.engine.rules import vcf_intervals as vi
from biowarden.engine.spec import load_packaged
from helpers import ADMISSION_SEEDS, MIN_RIGHT_CALLS

ROOT = Path(__file__).resolve().parents[1]
RULE_ID = "vcf.indel_interval_anchor"
KEY = "interval_meaning"
HEADER = "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n"
COLUMNS = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
SPAN = "%CHROM\\t%POS0\\t%END\\n"  # the shell spelling: backslash-t, backslash-n


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None:
        pytest.skip("samtools, bcftools and bedtools are not available")
    return env


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_packaged("vcf_intervals")


def engine_for(spec: Spec, tools: ToolEnv | None = None) -> Engine:
    return Engine(
        [spec],
        mode="block",
        real_bins=tools.real_bins() if tools else None,
        real_env=tools.process_env() if tools else None,
        retry_cap=10**9,
    )


def write_vcf(path: Path, records: list[tuple[str, str, str, str]]) -> Path:
    """Plain VCF, or gzip when the name ends in .gz; records are (chrom, pos, ref, alt)."""
    body = "".join(f"{c}\t{p}\tr{i}\t{r}\t{a}\t.\t.\t.\n" for i, (c, p, r, a) in enumerate(records))
    text = HEADER + COLUMNS + body
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


SNVS = [("chr1", "10", "C", "T"), ("chr1", "30", "AC", "GT")]  # an SNV and an MNV
DELETION = ("chr1", "50", "ACG", "A")


def outcome(engine: Engine, tool: str, argv: list[str], cwd: Path, declared=None):
    decision = engine.evaluate(Call(tool, tuple(argv), cwd, declared or {}))
    own = [o for o in decision.outcomes if o.rule.id == RULE_ID]
    assert len(own) <= 1
    return decision, (own[0].result if own else None)


# ---- the spec ----------------------------------------------------------------------------------


def test_spec_is_valid_and_declares_an_optional_two_valued_key(spec):
    (rule,) = spec.rules
    assert spec.name == "vcf_intervals" and rule.id == RULE_ID
    assert (rule.slot, rule.kind, rule.tier) == ("interval_closure", "fact", "T1")
    assert rule.declare is not None
    assert (rule.declare.key, rule.declare.values) == (KEY, ["ref_span", "replaced_bases"])
    assert rule.declare.required is False
    engine_for(spec)  # building the engine imports the rule modules and checks the checker name
    assert rule.check in known_checkers()
    schema = spec.declaration_schema()
    assert schema["properties"][KEY]["enum"] == ["ref_span", "replaced_bases"]
    assert schema["required"] == []


def test_spec_rejects_a_wrong_shape(spec):
    data = spec.model_dump(mode="json")
    data["rules"][0]["applies"]["file_patterns"] = []
    with pytest.raises(ValidationError, match="needs applies.file_patterns"):
        Spec.model_validate(data)
    data = spec.model_dump(mode="json")
    data["rules"][0]["tier"] = "T0"
    with pytest.raises(ValidationError, match="cannot be tier T0"):
        Spec.model_validate(data)


@pytest.mark.parametrize(
    ("name", "should_match"),
    [
        ("variants.vcf", True),
        ("variants.vcf.gz", True),
        ("sub/dir/v1.vcf.bgz", True),
        ("calls.bcf", True),
        ("/abs/path/calls.bcf", True),
        ("variants.vcf.gz.csi", False),
        ("variants.vcf.gz.tbi", False),
        ("calls.bcf.csi", False),
        ("variants.vcfx", False),
        ("variants.txt", False),
    ],
)
def test_file_pattern_matches_vcf_names_and_not_their_indexes(spec, name, should_match):
    (pattern,) = spec.rules[0].applies.file_patterns
    hits = [m.group(0) for m in re.finditer(pattern, f"cat {name} | head")]
    assert bool(hits) is should_match
    if should_match:
        assert hits == [name]


# ---- the command test: building intervals, and looking at ALT ------------------------------------

BUILDS = [
    "bcftools query -f '%CHROM\\t%POS0\\t%END\\n' v.vcf.gz",
    "bcftools query --format '%CHROM\\t%POS0\\t%END\\n' v.vcf.gz",
    "bcftools query -f '%END\\n' v.vcf.gz",
    "bcftools view -H v.vcf.gz | awk '{print $1,$2-1,$2-1+length($4)}'",
    "bcftools view -H v.vcf.gz | awk '{print $1,$2-1,$2+length($4)-1}'",
    "zcat v.vcf.gz | awk '{print $1, $2 - 1, $2 - 1 + length($4)}'",
    "zcat v.vcf.gz | perl -lane 'print join(\"\\t\",$F[0],$F[1]-1,$F[1]-1+length($F[3]))'",
    'python3 -c "f=l.split(); print(int(f[1]) - 1, int(f[1]) - 1 + len(f[3]))"',
    'python3 -c "pos = int(f[1]); start = pos - 1; end = start + len(ref)"',
    'python3 -c "print(rec.start, rec.stop)"',
    "Rscript -e 'start <- pos - 1; end <- start + nchar(ref)'",
]
NOT_BUILDS = [
    "bcftools view -H v.vcf.gz | head",
    "bcftools view -H v.vcf.gz | wc -l",
    "zcat v.vcf.gz | grep -vc '^#'",
    "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' v.vcf.gz",
    "bcftools query -f '%INFO/END\\n' v.vcf.gz",
    "awk 'END{print NR}' v.vcf",
    "awk '!/^#/{print length($4)}' v.vcf",  # a length, but no position arithmetic
    "awk '!/^#/{print $2-1}' v.vcf",  # a start, but no length
    "python3 -c \"print(len(open('v.vcf').readlines()))\"",
    "python3 -c \"print(re.search('a','b').start(), re.search('a','b').end())\"",
    "bcftools filter -i 'strlen(REF)>1' v.vcf.gz",
    "bcftools stats v.vcf.gz",
    "bcftools index -n v.vcf.gz",
]


def texts(command: str) -> tuple[str, str]:
    """(text, text without -f format strings) the checker sees for a `bash -c` command."""
    return vi.command_text(
        Call("bash", ("-c", command), Path("/nonexistent")), Ctx(Path("/"), "T0")
    )


@pytest.mark.parametrize("command", BUILDS)
def test_commands_that_build_intervals_are_recognised(command):
    assert vi.builds_intervals(texts(command)[0])


@pytest.mark.parametrize("command", NOT_BUILDS)
def test_ordinary_reads_do_not_look_like_interval_building(command):
    assert not vi.builds_intervals(texts(command)[0])


LOOKS_AT_ALT = [
    "awk '{s=$2-1; if (length($4)>1 && length($5)==1) s++}'",
    "awk '{ if (length($3)>length($4)) s++ }'",  # REF against ALT (query-fed), two lengths
    "awk '{ if (substr($3,1,1)==substr($4,1,1)) s++ }'",
    'python3 -c "ref, alt = f[3], f[4]"',
    'python3 -c "if len(f[3]) > 1: s += f[4] == f[3][0]"',
    "perl -lane '$s++ if length($F[3])>length($F[4])'",
    "bcftools query -f '%POS0\\t%END\\n' -i 'TYPE=\"snp\"' v.vcf.gz",
    "bcftools view -v snps v.vcf.gz",
    "bcftools view --types snps,mnps v.vcf.gz",
    "bcftools view -V indels v.vcf.gz",
    "bcftools view --exclude-types=indels v.vcf.gz",
]
IGNORES_ALT = [
    "bcftools query -f '%CHROM\\t%POS0\\t%END\\t%ALT\\n' v.vcf.gz",  # ALT only printed
    "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' v.vcf.gz | awk '{print $2-1+length($3)}'",
    "awk '{print $2-1, length($4)}'",
    "awk -v OFS='\\t' '{print $1,$2-1,$2-1+length($4)}' v.vcf",  # -v is an awk variable
    "bcftools view -v indels v.vcf.gz",
    "awk '{ if (length($0) > 80 && length($4) > 1) s = $2-1 }'",  # a line length is not ALT
]


@pytest.mark.parametrize("command", LOOKS_AT_ALT)
def test_commands_that_show_the_alt_allele_are_seen_as_handling_it(command):
    assert vi.looks_at_alt(texts(command)[1])


@pytest.mark.parametrize("command", IGNORES_ALT)
def test_printing_alt_or_using_unrelated_options_is_not_handling_it(command):
    assert not vi.looks_at_alt(texts(command)[1])


@pytest.mark.parametrize(
    ("ref", "alt", "anchored"),
    [
        ("ACG", "A", True),  # deletion
        ("A", "ACG", True),  # insertion
        ("acg", "A", True),  # lower case
        ("AC", "A", True),
        ("ACG", "AT", True),  # complex, shares the first base
        ("C", "T", False),  # SNV
        ("AC", "GT", False),  # MNV with no shared base
        ("AC", "AG", False),  # equal length, shares the first base: not an anchored indel
        ("ACG", "T", False),  # complex, no shared first base
        ("A", "<DEL>", False),
        ("A", "*", False),
        ("A", ".", False),
        ("A", "A]chr2:5]", False),  # breakend
        ("A", "]chr2:5]AC", False),
        ("", "A", False),
    ],
)
def test_shares_anchor(ref, alt, anchored):
    assert vi.shares_anchor(ref, alt) is anchored


# ---- the data test on small VCFs ----------------------------------------------------------------


def sample(tmp_path: Path, records, name="v.vcf", max_records=10_000, tools=None):
    path = write_vcf(tmp_path / name, records)
    ctx = Ctx(tmp_path, "T1", tools.real_bins() if tools else {}, None)
    return vi.sample_vcf(ctx, path, max_records)


def test_sample_counts_anchored_records_and_prefers_a_plain_deletion_as_the_example(tmp_path):
    ins = ("chr1", "20", "A", "ACT")
    got = sample(tmp_path, [ins, *SNVS, DELETION, ("chr1", "70", "TC", "T,TCC")])
    assert (got.scanned, got.anchored) == (5, 3)
    assert got.example == vi.Anchored("chr1", 50, "ACG", "A")


def test_sample_sees_nothing_in_substitution_only_and_symbolic_files(tmp_path):
    got = sample(tmp_path, [*SNVS, ("chr1", "40", "A", "<DEL>"), ("chr1", "60", "G", "*")])
    assert (got.scanned, got.anchored, got.example) == (4, 0, None)


def test_sample_is_bounded_by_max_records(tmp_path):
    records = [*SNVS, *SNVS, DELETION]
    assert sample(tmp_path, records, max_records=4).anchored == 0
    assert sample(tmp_path, records, max_records=5).anchored == 1


def test_sample_reads_gzip_and_bcf(tmp_path, tools):
    assert sample(tmp_path, [*SNVS, DELETION], "v.vcf.gz").anchored == 1
    plain = write_vcf(tmp_path / "b.vcf", [*SNVS, DELETION])
    proc = tools.run(["bcftools", "view", "-Ob", "-o", "b.bcf", plain.name], tmp_path)
    assert proc.returncode == 0, proc.stderr
    ctx = Ctx(tmp_path, "T1", tools.real_bins(), tools.process_env())
    got = vi.sample_vcf(ctx, tmp_path / "b.bcf", 100)
    assert (got.scanned, got.anchored) == (3, 1)


def test_header_only_and_empty_files_have_no_records(tmp_path):
    assert sample(tmp_path, []).scanned == 0
    (tmp_path / "e.vcf").write_text("")
    assert vi.sample_vcf(Ctx(tmp_path, "T1"), tmp_path / "e.vcf", 10).scanned == 0


# ---- the checker through the engine --------------------------------------------------------------


def test_deletion_vcf_with_interval_arithmetic_fails_until_declared(spec, tmp_path):
    write_vcf(tmp_path / "v.vcf", [*SNVS, DELETION])
    eng = engine_for(spec)
    argv = ["query", "-f", SPAN, "v.vcf"]
    decision, result = outcome(eng, "bcftools", argv, tmp_path)
    assert decision.action == "block" and result.verdict == "fail"
    assert result.code == "anchor_base_in_interval"
    message = render_block(decision)
    for needle in ("chr1:50", "REF=ACG ALT=A", "chr1:49-52", "chr1:50-52", "anchor"):
        assert needle in message, needle
    assert f"--declare {KEY}=ref_span" in message and f"--declare {KEY}=replaced_bases" in message
    for value in ("ref_span", "replaced_bases"):
        decision, result = outcome(eng, "bcftools", argv, tmp_path, {KEY: value})
        assert decision.action == "allow" and result.verdict == "pass"
        assert "not verified" in result.note


def test_an_invalid_declaration_is_refused_by_the_engine(spec, tmp_path):
    write_vcf(tmp_path / "v.vcf", [DELETION])
    _, result = outcome(
        engine_for(spec), "bcftools", ["query", "-f", SPAN, "v.vcf"], tmp_path, {KEY: "x"}
    )
    assert result.code == "declaration_invalid"


def test_substitution_only_vcf_is_never_flagged_whatever_the_command(spec, tmp_path):
    write_vcf(tmp_path / "v.vcf", SNVS)
    eng = engine_for(spec)
    for command in BUILDS:
        _, result = outcome(eng, "bash", ["-c", command.replace("v.vcf.gz", "v.vcf")], tmp_path)
        assert result is None or result.verdict == "pass", command


def test_ordinary_reads_of_a_deletion_vcf_are_silent_and_read_no_file(spec, tmp_path):
    write_vcf(tmp_path / "v.vcf", [*SNVS, DELETION])
    eng = engine_for(spec)
    for command in NOT_BUILDS:
        argv = ["-c", command.replace("v.vcf.gz", "v.vcf")]
        decision, result = outcome(eng, "bash", argv, tmp_path)
        assert decision.action == "allow", command
        assert result is None or result.verdict == "pass", command
    # tier honesty: such a call is decided from the text; a T0 context would raise if it read a file
    match = Match(spec.rules[0], (tmp_path / "v.vcf",))
    call = Call("bcftools", ("view", "-H", "v.vcf"), tmp_path)
    assert (
        known_checkers()["vcf_indel_interval_anchor"](call, match, Ctx(tmp_path, "T0")).verdict
        == "pass"
    )


def test_the_checker_reads_files_so_it_is_tier_t1_and_a_t0_context_refuses(spec, tmp_path):
    write_vcf(tmp_path / "v.vcf", [DELETION])
    match = Match(spec.rules[0], (tmp_path / "v.vcf",))
    call = Call("bcftools", ("query", "-f", SPAN, "v.vcf"), tmp_path)
    check = known_checkers()["vcf_indel_interval_anchor"]
    with pytest.raises(TierViolation):
        check(call, match, Ctx(tmp_path, "T0"))
    assert check(call, match, Ctx(tmp_path, "T1")).verdict == "fail"


FIRES = [
    ("bcftools", ["query", "-f", SPAN, "v.vcf.gz"]),
    ("bcftools", ["query", "--format", SPAN, "v.vcf.gz"]),
    ("bcftools", ["query", f"--format={SPAN}", "v.vcf.gz"]),
    ("bcftools", ["query", f"-f{SPAN}", "v.vcf.gz"]),
    ("bcftools", ["query", "-f", SPAN + "\t%ALT", "-r", "chr1", "v.vcf.gz"]),
    ("bcftools", ["query", "-r", "chr1:1-900", "-f", "%CHROM\t%POS0\t%END\n", "v.vcf.gz"]),
    ("bash", ["-c", f"bcftools query -f '{SPAN}' v.vcf.gz | sort -k2,2n"]),
    ("bash", ["-c", "zcat v.vcf.gz | awk '!/^#/{print $1,$2-1,$2-1+length($4)}'"]),
    (
        "bash",
        [
            "-c",
            "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' v.vcf.gz | "
            "awk '{print $1,$2-1,$2-1+length($3)}'",
        ],
    ),
    ("gawk", ["!/^#/{print $1,$2-1,$2+length($4)-1}", "v.vcf"]),
    ("mawk", ["!/^#/{print $1,$2-1,$2+length($4)-1}", "v.vcf"]),
    ("perl", ["-lane", "print $F[1]-1, $F[1]-1+length($F[3])", "v.vcf"]),
    ("python3", ["-c", "for l in open('v.vcf'): f=l.split(); print(int(f[1])-1+len(f[3]))"]),
    (
        "python3",
        ["-c", "import pysam\nfor r in pysam.VariantFile('v.vcf.gz'): print(r.start, r.stop)"],
    ),
]
SILENT = [
    ("bcftools", ["view", "-H", "v.vcf.gz"]),
    ("bcftools", ["query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", "v.vcf.gz"]),
    ("bcftools", ["stats", "v.vcf.gz"]),
    ("zcat", ["v.vcf.gz"]),
    ("wc", ["-l", "v.vcf"]),
    ("head", ["-20", "v.vcf"]),
    ("bash", ["-c", "zcat v.vcf.gz | grep -vc '^#'"]),
    ("awk", ["END{print NR}", "v.vcf"]),
    ("python3", ["-c", "print(sum(1 for l in open('v.vcf') if l[0] != '#'))"]),
    ("bcftools", ["query", "-f", SPAN, "-i", 'TYPE="snp"', "v.vcf.gz"]),
    ("bcftools", ["view", "-v", "snps", "-H", "v.vcf.gz"]),
    ("bash", ["-c", "bcftools view -V indels -H v.vcf.gz | awk '{print $2-1+length($4)}'"]),
    ("awk", ["!/^#/{s=$2-1; if (length($4)>length($5)) s++; print $1,s,$2-1+length($4)}", "v.vcf"]),
    ("awk", ["-F", "\t", "!/^#/{s=$2-1; if (length($5)==1) s++; print $2-1+length($4)}", "v.vcf"]),
    (
        "bash",
        [
            "-c",
            "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' v.vcf.gz | awk -v OFS='\\t' "
            "'{s=$2-1; if (substr($3,1,1)==substr($4,1,1) && length($3)>1) s++; print $1,s,$2}'",
        ],
    ),
    ("python3", ["-c", "for l in open('v.vcf'): f=l.split(); print(int(f[1])-1, len(f[3]), f[4])"]),
]


@pytest.fixture
def deletion_dir(tmp_path):
    write_vcf(tmp_path / "v.vcf", [*SNVS, DELETION])
    write_vcf(tmp_path / "v.vcf.gz", [*SNVS, DELETION])
    return tmp_path


@pytest.mark.parametrize(("tool", "argv"), FIRES)
def test_interval_arithmetic_on_a_deletion_vcf_fails_in_every_spelling(
    spec, deletion_dir, tool, argv
):
    decision, result = outcome(engine_for(spec), tool, argv, deletion_dir)
    assert result is not None and result.verdict == "fail", (tool, argv)
    _, declared = outcome(engine_for(spec), tool, argv, deletion_dir, {KEY: "replaced_bases"})
    assert declared.verdict == "pass"


@pytest.mark.parametrize(("tool", "argv"), SILENT)
def test_reads_and_alt_aware_commands_on_a_deletion_vcf_stay_silent(spec, deletion_dir, tool, argv):
    decision, result = outcome(engine_for(spec), tool, argv, deletion_dir)
    assert decision.action == "allow", (tool, argv)
    assert result is None or result.verdict == "pass"


def test_a_script_file_the_command_runs_is_read(spec, deletion_dir):
    (deletion_dir / "span.py").write_text(
        "for l in open('v.vcf'):\n    f = l.split()\n"
        "    print(int(f[1]) - 1, int(f[1]) - 1 + len(f[3]))\n"
    )
    (deletion_dir / "span_alt.py").write_text(
        "for l in open('v.vcf'):\n    f = l.split()\n    s = int(f[1]) - 1 + (f[3][0] == f[4])\n"
        "    print(s, int(f[1]) - 1 + len(f[3]))\n"
    )
    (deletion_dir / "prog.awk").write_text("{print $1, $2-1, $2-1+length($4)}\n")
    eng = engine_for(spec)
    assert outcome(eng, "python3", ["span.py", "v.vcf"], deletion_dir)[1].verdict == "fail"
    assert outcome(eng, "python3", ["span_alt.py", "v.vcf"], deletion_dir)[1].verdict == "pass"
    assert outcome(eng, "awk", ["-f", "prog.awk", "v.vcf"], deletion_dir)[1].verdict == "fail"
    assert outcome(eng, "python3", ["missing.py", "v.vcf"], deletion_dir)[1].verdict == "pass"


def test_bcf_input_is_sampled_with_the_real_bcftools(spec, tools, tmp_path):
    plain = write_vcf(tmp_path / "b.vcf", [*SNVS, DELETION])
    assert (
        tools.run(["bcftools", "view", "-Ob", "-o", "b.bcf", plain.name], tmp_path).returncode == 0
    )
    argv = ["query", "-f", SPAN, "b.bcf"]
    _, result = outcome(engine_for(spec, tools), "bcftools", argv, tmp_path)
    assert result.verdict == "fail"
    _, without_binary = outcome(engine_for(spec), "bcftools", argv, tmp_path)
    assert without_binary.verdict == "unknown" and without_binary.code == "vcf_unreadable"


def test_a_corrupt_or_truncated_gzip_is_unknown_not_a_crash(spec, tmp_path):
    good = write_vcf(tmp_path / "ok.vcf.gz", [*SNVS, DELETION]).read_bytes()
    (tmp_path / "cut.vcf.gz").write_bytes(good[: len(good) // 2])
    (tmp_path / "junk.vcf.gz").write_bytes(b"\x1f\x8b" + b"not a gzip stream" * 20)
    eng = engine_for(spec)
    for name in ("cut.vcf.gz", "junk.vcf.gz"):
        decision, result = outcome(eng, "bcftools", ["query", "-f", SPAN, name], tmp_path)
        assert result.verdict in ("unknown", "fail", "pass"), name
        assert result.code != "checker_crashed", name
        assert decision.action != "block" or result.verdict == "fail"
    _, result = outcome(eng, "bcftools", ["query", "-f", SPAN, "junk.vcf.gz"], tmp_path)
    assert result.verdict == "unknown"


def test_anchored_records_beyond_max_records_are_not_seen(spec, tmp_path):
    write_vcf(tmp_path / "v.vcf", [*SNVS * 3, DELETION])
    data = spec.model_dump(mode="json")
    data["rules"][0]["params"]["max_records"] = 6
    short = Spec.model_validate(data)
    argv = ["query", "-f", SPAN, "v.vcf"]
    assert outcome(engine_for(short), "bcftools", argv, tmp_path)[1].verdict == "pass"
    assert outcome(engine_for(spec), "bcftools", argv, tmp_path)[1].verdict == "fail"


def test_missing_files_and_globs_do_not_match(spec, tmp_path):
    decision, result = outcome(
        engine_for(spec), "bcftools", ["query", "-f", SPAN, "none.vcf.gz"], tmp_path
    )
    assert result is None and decision.action == "allow"
    decision, result = outcome(
        engine_for(spec), "bash", ["-c", f"bcftools query -f '{SPAN}' *.vcf.gz"], tmp_path
    )
    assert result is None


# ---- F10 workspaces built with the real tools ----------------------------------------------------


@pytest.fixture(scope="module")
def workspaces(tools, tmp_path_factory):
    base = tmp_path_factory.mktemp("f10")
    out = {}
    for twin in ("clean", "faulted"):
        work = base / twin
        out[twin] = (build_task("F10", 3, twin, work, tools, "implicit"), work)
    return out


def test_every_template_on_the_clean_twin_passes_without_a_declaration(spec, tools, workspaces):
    _, work = workspaces["clean"]
    plain = tools.run(
        ["bcftools", "view", "--no-version", "-o", "variants.vcf", "variants.vcf.gz"], work
    )
    assert plain.returncode == 0
    eng = engine_for(spec, tools)
    for template in TEMPLATES:
        _, result = outcome(eng, template.tool, list(template.argv), work)
        assert result is None or result.verdict == "pass", template.name


def test_on_the_faulted_twin_ref_span_templates_fail_and_the_rest_stay_silent(
    spec, tools, workspaces
):
    _, work = workspaces["faulted"]
    plain = tools.run(
        ["bcftools", "view", "--no-version", "-o", "variants.vcf", "variants.vcf.gz"], work
    )
    assert plain.returncode == 0
    eng = engine_for(spec, tools)
    for template in TEMPLATES:
        _, result = outcome(eng, template.tool, list(template.argv), work)
        failed = result is not None and result.verdict == "fail"
        assert failed == (
            template.meaning == "ref_span" or template.name == "view_awk_skip_all_multibase"
        ), template.name
        declared = {KEY: "replaced_bases"}
        _, with_decl = outcome(eng, template.tool, list(template.argv), work, declared)
        assert with_decl is None or with_decl.verdict == "pass", template.name


# ---- admission -----------------------------------------------------------------------------------


def test_admission(spec, tools, tmp_path):
    stats = measure(RULE_ID, [spec], range(ADMISSION_SEEDS), tools, tmp_path)
    right_declared = stats.cells[("right", True)]
    right_undeclared = stats.cells[("right", False)]
    assert stats.conditional
    assert stats.admitted()
    assert right_declared.n >= MIN_RIGHT_CALLS and right_declared.failed == 0
    assert right_undeclared.n >= MIN_RIGHT_CALLS and right_undeclared.failed == 0
    assert stats.killed() > 0
    assert all(cell.crashed == 0 for cell in stats.cells.values())
    # undeclared REF-span calls on the faulted twin are caught
    wrong_undeclared = stats.cells[("wrong", False)]
    assert wrong_undeclared.failed >= ADMISSION_SEEDS * 6
    # a declaration passes the rule whatever the arithmetic: the documented limit
    assert stats.cells[("wrong", True)].failed == 0
