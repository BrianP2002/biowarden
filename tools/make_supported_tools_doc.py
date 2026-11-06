"""Regenerate docs/supported-tools.md from the lock files in envs/locks/.

    python tools/make_supported_tools_doc.py > docs/supported-tools.md
    python tools/make_supported_tools_doc.py --check        # exit 1 when the page is out of date

The page lists, per environment, the tools and libraries of the benchmark with their versions, and
the number of rules and task families per domain. The versions are read from the lock files that
tools/env_lock.py writes: `<env>.tsv` (the conda packages of an environment), `tool-versions.txt`
(samtools, bcftools, bedtools, bubblewrap and the harness interpreter) and `venv-pip-freeze.txt`
(the pip packages of the harness). Which package holds the version of a name on the page is written
down below (ENVIRONMENTS, HARNESS, BINARIES), and the generator stops with a message (exit 2) when

    a mapped package is missing from its lock file,
    a lock file has no entry in ENVIRONMENTS, or an entry has no lock file,
    a domain lists a tool that no table has, or its title is not a pack of docs/families.md,
    a lock file cannot be read, is malformed or lacks a line the page needs,

so a mapping that no longer fits the lock files stops the build instead of printing a wrong
version. The url and sha256 columns of the lock files are not used, and the page holds no
filesystem path. The rule and family counts come from the packaged specs (through
tools/make_rules_doc.py) and the family modules (through tools/make_families_doc.py).

`--check` writes nothing and exits 1 when the page differs from what the code gives now;
tests/test_generated_docs.py runs it.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from biowarden.engine import Promotion

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "tools/make_supported_tools_doc.py"
LOCKS = ROOT / "envs" / "locks"
PAGE = ROOT / "docs" / "supported-tools.md"
LOCK_LINK = "../envs/locks"

# Standalone programs: tool-versions.txt has a `name: <--version line>` and a `sha256 name:` line
# for each of them; the promotion table records the versions of the same three.
BINARIES = ("samtools", "bcftools", "bedtools")
SANDBOX = ("bubblewrap", "bwrap")  # the name shown, and its key in tool-versions.txt
INTERPRETER = "python"  # the harness interpreter, same file

# conda environments: lock file stem, heading, (name shown, conda package). samtools, bcftools and
# bedtools are not listed here although an environment holds a samtools too: the tool path puts the
# standalone binaries first.
ENVIRONMENTS = (
    (
        "bio-a",
        "Alignment, quantification, variants, intervals and sequence tools",
        (
            ("bwa", "bwa"),
            ("Bowtie 2", "bowtie2"),
            ("minimap2", "minimap2"),
            ("HISAT2", "hisat2"),
            ("STAR", "star"),
            ("tabix", "htslib"),
            ("featureCounts", "subread"),
            ("Salmon", "salmon"),
            ("kallisto", "kallisto"),
            ("RSEM", "rsem"),
            ("StringTie", "stringtie"),
            ("VCFtools", "vcftools"),
            ("plink", "plink"),
            ("plink2", "plink2"),
            ("GATK", "gatk4"),
            ("Picard", "picard"),
            ("BEDOPS", "bedops"),
            ("liftOver", "ucsc-liftover"),
            ("seqtk", "seqtk"),
            ("seqkit", "seqkit"),
            ("fastp", "fastp"),
            ("Trimmomatic", "trimmomatic"),
            ("BLAST+", "blast"),
            ("DIAMOND", "diamond"),
            ("MAFFT", "mafft"),
            ("MUSCLE", "muscle"),
            ("HMMER", "hmmer"),
        ),
    ),
    (
        "bio-py",
        "Python tools for sequencing data",
        (
            ("HTSeq", "htseq"),
            ("deepTools", "deeptools"),
            ("MACS3", "macs3"),
            ("cutadapt", "cutadapt"),
            ("pysam", "pysam"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scipy", "scipy"),
        ),
    ),
    (
        "bio-sc",
        "Single-cell",
        (
            ("scanpy", "scanpy"),
            ("anndata", "anndata"),
            ("mudata", "mudata"),
            ("loompy", "loompy"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scipy", "scipy"),
        ),
    ),
    (
        "bio-struct",
        "Protein structure",
        (
            ("gemmi", "gemmi"),
            ("Biopython", "biopython"),
            ("ProDy", "prody"),
            ("MDAnalysis", "mdanalysis"),
            ("mdtraj", "mdtraj"),
            ("RDKit", "rdkit"),
            ("pdb-tools", "pdb-tools"),
            ("TM-align", "tmalign"),
            ("MMseqs2", "mmseqs2"),
            ("mkdssp", "dssp"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scipy", "scipy"),
        ),
    ),
)

# The pip packages of the harness that matter for a rerun: the runtime dependencies of
# pyproject.toml, the two model SDKs of its `bench` extra, and numpy (the task builders import it).
HARNESS = (
    ("pydantic", "pydantic"),
    ("PyYAML", "PyYAML"),
    ("gemmi", "gemmi"),
    ("h5py", "h5py"),
    ("numpy", "numpy"),
    ("fastmcp", "fastmcp"),
    ("anthropic", "anthropic"),
    ("openai", "openai"),
)

# The tools of each domain, by the names above. The titles are the packs of docs/families.md.
DOMAIN_TOOLS = {
    "Read alignment": "bwa, Bowtie 2, minimap2, HISAT2, STAR, samtools",
    "RNA-seq quantification": "featureCounts, HTSeq, Salmon, kallisto, RSEM, StringTie",
    "Variants and genotypes": "bcftools, VCFtools, plink, plink2, tabix, GATK, Picard",
    "Intervals, peaks, liftover, coverage tracks": "bedtools, BEDOPS, liftOver, deepTools, MACS3",
    "Sequence processing, search and multiple alignment": (
        "seqtk, seqkit, fastp, cutadapt, Trimmomatic, BLAST+, DIAMOND, MAFFT, MUSCLE, HMMER"
    ),
    "Protein structure and protein-model plumbing": (
        "gemmi, Biopython, ProDy, MDAnalysis, mdtraj, RDKit, pdb-tools, TM-align, MMseqs2, mkdssp"
    ),
    "Single-cell and virtual-cell inputs": "scanpy, anndata, mudata, loompy",
    "Original families": "samtools, bcftools, bedtools",
}

VERSION = re.compile(r"\d+(?:\.\d+)+")


class GenerationError(Exception):
    """The page cannot be built from the lock files and the code as they are."""


class Section(NamedTuple):
    stem: str  # the lock file `<stem>.tsv`
    heading: str
    rows: list[tuple[str, str, str]]  # (name shown, version, conda package)
    packages: int  # packages in the lock file
    python: str | None


def sibling(name: str) -> ModuleType:
    """tools/<name>.py as a module (tools/ is not a package and may not be on sys.path)."""
    path = Path(__file__).with_name(f"{name}.py")
    known = sys.modules.get(name)
    if known is not None and Path(getattr(known, "__file__", "") or "") == path:
        return known
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GenerationError(f"cannot load {name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# -- the lock files --------------------------------------------------------------------------------


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # the message names the file, not its directory
        raise GenerationError(f"{path.name}: {exc.strerror or 'cannot be read'}") from exc


def read_lock(path: Path) -> dict[str, str]:
    """package name -> version from `<env>.tsv`; the url and sha256 columns are not looked at."""
    lines = read_text(path).splitlines()
    header = lines[0].split("\t") if lines else []
    if "name" not in header or "version" not in header:
        raise GenerationError(f"{path.name}: the header has no name and version columns")
    i_name, i_version = header.index("name"), header.index("version")
    packages: dict[str, str] = {}
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(header):
            raise GenerationError(
                f"{path.name}:{number}: {len(cells)} fields, expected {len(header)}"
            )
        if cells[i_name] in packages:
            raise GenerationError(f"{path.name}: package {cells[i_name]!r} is listed twice")
        packages[cells[i_name]] = cells[i_version]
    return packages


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_freeze(path: Path) -> dict[str, str]:
    """normalised distribution name -> version from `pip freeze`; lines that are not
    `name==version` pins (comments, for one) are skipped."""
    pins: dict[str, str] = {}
    for line in read_text(path).splitlines():
        name, sep, version = line.strip().partition("==")
        if sep and not name.startswith("#"):
            pins[normalise(name)] = version
    return pins


def version_in(line: str, what: str) -> str:
    found = VERSION.search(line)
    if found is None:
        raise GenerationError(f"tool-versions.txt: no version in the line for {what}: {line!r}")
    return found.group(0)


def read_binaries(path: Path) -> dict[str, str]:
    """Version of each standalone binary, of the sandbox and of the harness interpreter."""
    text = read_text(path)
    lines = text.splitlines()
    values = {}
    for line in lines:
        name, sep, value = line.partition(": ")
        if sep and name.isidentifier():  # `sha256 samtools: ...` is not an identifier
            values[name] = value.strip()
    versions = {}
    for name in (*BINARIES, SANDBOX[1], INTERPRETER):
        if name not in values:
            raise GenerationError(f"tool-versions.txt has no line for {name}")
        versions[name] = version_in(values[name], name)
    for name in BINARIES:
        if not any(line.startswith(f"sha256 {name}:") for line in lines):
            raise GenerationError(f"tool-versions.txt has no sha256 line for {name}")
    return versions


def read_sections(locks: Path) -> list[Section]:
    mapped = {stem for stem, _, _ in ENVIRONMENTS}
    present = {p.stem for p in locks.glob("*.tsv")}
    if present - mapped:
        raise GenerationError(
            f"envs/locks/{sorted(present - mapped)[0]}.tsv has no entry in ENVIRONMENTS of "
            f"{SCRIPT}; add its tools there"
        )
    if mapped - present:
        raise GenerationError(
            f"ENVIRONMENTS of {SCRIPT} names {sorted(mapped - present)[0]}.tsv, which is not in "
            "envs/locks/"
        )
    sections = []
    for stem, heading, mapping in ENVIRONMENTS:
        packages = read_lock(locks / f"{stem}.tsv")
        rows = []
        for shown, package in mapping:
            if package not in packages:
                raise GenerationError(
                    f"{shown}: the conda package {package!r} is not in envs/locks/{stem}.tsv; fix "
                    f"ENVIRONMENTS of {SCRIPT} or the lock file"
                )
            rows.append((shown, packages[package], package))
        sections.append(Section(stem, heading, rows, len(packages), packages.get("python")))
    return sections


def read_harness(pins: dict[str, str]) -> list[tuple[str, str, str]]:
    rows = []
    for shown, dist in HARNESS:
        if normalise(dist) not in pins:
            raise GenerationError(
                f"{shown}: the pip package {dist!r} is not in envs/locks/venv-pip-freeze.txt; fix "
                f"HARNESS of {SCRIPT} or the lock file"
            )
        rows.append((shown, pins[normalise(dist)], dist))
    return rows


# -- the page --------------------------------------------------------------------------------------


def join_and(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[int] = ()
) -> list[str]:
    align = "|" + "|".join("---:" if i in numeric else "---" for i in range(len(headers))) + "|"
    return [
        "| " + " | ".join(headers) + " |",
        align,
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def calibration_sentence(binaries: dict[str, str]) -> str:
    """The binaries the promotion table was measured with, set against those in the lock."""
    sets = Counter(
        tuple(sorted(entry.get("tools", {}).items()))
        for entry in Promotion.packaged().entries.values()
    )
    if len(sets) != 1:
        return f"The promotion table records {len(sets)} different sets of tool versions."
    ((recorded, count),) = sets.items()
    measured = {name: version_in(line, name) for name, line in recorded}
    shown = [f"{name} {measured[name]}" for name in BINARIES if name in measured]
    if not shown:
        return ""
    sentence = (
        f"The promotion table records {join_and(shown)} as the versions it was measured with, for "
        f"all {count} rules."
    )
    if any(measured.get(name) != binaries[name] for name in BINARIES if name in measured):
        sentence += " The lock lists other versions of these binaries; calibrate again."
    return sentence


def domain_rows(
    rules_doc: ModuleType, families_doc: ModuleType, known: dict[str, set[str]]
) -> tuple[list[tuple[str, str, str, str]], int, str]:
    """The rows of the domain table, the number of rules about data formats (counted under the
    domain of the original families) and the sentence on rules counted away from their families."""
    titles = [title for title, _, _ in families_doc.DOMAINS]
    if not (sorted(titles) == sorted(DOMAIN_TOOLS) == sorted(t for t, _ in rules_doc.DOMAINS)):
        raise GenerationError(
            f"the domains of {SCRIPT}, tools/make_rules_doc.py and tools/make_families_doc.py "
            "must have the same titles"
        )
    for title, names in DOMAIN_TOOLS.items():
        for name in names.split(", "):
            if name not in known:
                raise GenerationError(f"domain {title!r} lists {name!r}, which no table has")
            if len(known[name]) != 1:
                raise GenerationError(
                    f"domain {title!r} lists {name!r}, which has several versions"
                )
    try:
        rules = rules_doc.collect()
    except rules_doc.GenerationError as exc:
        raise GenerationError(str(exc)) from exc
    family_table = families_doc.rows()
    pack_of = {n: title for title, low, high in families_doc.DOMAINS for n in range(low, high + 1)}
    if any(number not in pack_of for number in family_table):
        raise GenerationError("some families are in no pack of tools/make_families_doc.py")
    rules_per_domain = Counter(row.domain for row in rules)
    families_per_pack = Counter(pack_of[number] for number in family_table)
    rows = [
        (
            title,
            DOMAIN_TOOLS[title],
            str(rules_per_domain[title]),
            str(families_per_pack[title]),
        )
        for title, _ in rules_doc.DOMAINS
    ]
    rows.append(("Total", "", str(len(rules)), str(len(family_table))))
    # rules whose families are all in another pack than the domain they are counted under
    apart: dict[tuple[str, tuple[str, ...]], list[tuple[str, list[int]]]] = defaultdict(list)
    for row in rules:
        numbers = [int(label.split(" ")[0][1:]) for label in row.families]
        packs = {pack_of[n] for n in numbers}
        if numbers and row.domain not in packs:
            apart[(row.domain, tuple(sorted(packs)))].append((row.rule.id, numbers))
    parts = []
    for (domain, away), found in apart.items():
        names = join_and(
            [
                f"`{rule_id}` ({', '.join(f'F{n}' for n in sorted(numbers))})"
                for rule_id, numbers in found
            ]
        )
        many = len(found) > 1
        parts.append(
            f" {names} {'are' if many else 'is'} counted under {domain}, but all the families "
            f"{'they are' if many else 'it is'} written for are in {join_and(list(away))}."
        )
    data_formats = sum(1 for row in rules if row.domain == rules_doc.DATA_FORMAT_DOMAIN)
    return rows, data_formats, "".join(parts)


def render(locks: Path = LOCKS) -> str:
    rules_doc = sibling("make_rules_doc")
    families_doc = sibling("make_families_doc")
    paragraph = rules_doc.paragraph
    if not locks.is_dir():
        raise GenerationError("envs/locks/ does not exist")
    binaries = read_binaries(locks / "tool-versions.txt")
    sections = read_sections(locks)
    pins = read_freeze(locks / "venv-pip-freeze.txt")
    harness = read_harness(pins)
    known: dict[str, set[str]] = defaultdict(set)  # name shown -> the versions it has
    for name in BINARIES:
        known[name].add(binaries[name])
    for section in sections:
        for shown, version, _ in section.rows:
            known[shown].add(version)
    domains, data_formats, apart = domain_rows(rules_doc, families_doc, known)

    lines = [
        "# Supported tools",
        "",
        f"<!-- Generated by {SCRIPT} from envs/locks/, the rule specs and the family modules; "
        "do not edit by hand. -->",
        "",
        *paragraph(
            "The programs and libraries of the benchmark environments, with the "
            f"versions recorded in [`envs/locks/`]({LOCK_LINK}/), and the number of rules and task "
            f"families per domain. Generated by `{SCRIPT}` from the lock files, the rule specs and "
            f"the family modules; do not edit by hand. `python {SCRIPT} --check` fails when this "
            "page differs from what the generator writes now."
        ),
        "",
        *paragraph("Rules: [rule catalogue](rules.md). Families: [task families](families.md)."),
        "",
        "## Domains",
        "",
        *table(("Domain", "Tools", "Rules", "Families"), domains, numeric=(2, 3)),
        "",
        *paragraph(
            "Rules are counted under the domain of the tool they watch; the "
            f"{data_formats} rules about data formats, which are not about one tool, are counted "
            f"under {rules_doc.DATA_FORMAT_DOMAIN}. Families are counted under their pack, the "
            "section of [task families](families.md) that lists them. A rule and the families it "
            f"is written for can therefore fall in different rows.{apart}"
        ),
        "",
        *paragraph(
            "Models such as AlphaFold, Boltz, scGPT or GEARS are not run: the rules and families "
            "cover the files, libraries and scripts around them. The one rule that watches a "
            "structure-prediction program, `structure.chain_order`, is calibrated against "
            "`mockfold`, a CPU-only stand-in that ships in "
            "[`src/biowarden/bench/fixtures/`](../src/biowarden/bench/fixtures/)."
        ),
        "",
        "## Versions",
        "",
        *paragraph(
            "Each version below is read from one lock file: the version of the named conda "
            "package, pip distribution or binary. A table lists selected packages of its lock "
            "file, not all of them."
        ),
        "",
        "### Tool binaries and sandbox",
        "",
        *paragraph(
            "These programs are recorded by their `--version` line, not by a conda package "
            f"list, in [`tool-versions.txt`]({LOCK_LINK}/tool-versions.txt), which also holds the "
            f"sha256 of {join_and(list(BINARIES))}. " + calibration_sentence(binaries)
        ),
        "",
        *table(
            ("Name", "Version"),
            [
                *((name, binaries[name]) for name in BINARIES),
                (f"{SANDBOX[0]} (the sandbox of the benchmark runner)", binaries[SANDBOX[1]]),
            ],
        ),
        "",
    ]
    for section in sections:
        facts = f"{section.packages} packages" + (
            f", Python {section.python}" if section.python else ""
        )
        lines += [
            f"### {section.heading}",
            "",
            f"Lock file: [`{section.stem}.tsv`]({LOCK_LINK}/{section.stem}.tsv), {facts}.",
            "",
            *table(
                ("Name", "Version", "Conda package"),
                [(name, version, f"`{package}`") for name, version, package in section.rows],
            ),
            "",
        ]
    lines += [
        "### Harness environment",
        "",
        *paragraph(
            f"Lock file: [`venv-pip-freeze.txt`]({LOCK_LINK}/venv-pip-freeze.txt), {len(pins)} "
            f"pinned packages; the interpreter is Python {binaries[INTERPRETER]}. Selected pins: "
            "the runtime dependencies in `pyproject.toml`, numpy (imported by the task builders) "
            "and the two model SDKs of the `bench` extra."
        ),
        "",
        *table(
            ("Name", "Version", "Pip package"),
            [(name, version, f"`{dist}`") for name, version, dist in harness],
        ),
        "",
        "## How the versions are pinned",
        "",
        *paragraph(
            f"[`envs/locks/`]({LOCK_LINK}/) holds one explicit package list per conda environment "
            "(name, version, build, URL and sha256 of every package, no filesystem path), the "
            "`pip freeze --all` of the harness interpreter, and `tool-versions.txt`: the "
            "`--version` lines of samtools, bcftools, bedtools and bubblewrap, the Python version "
            "and the sha256 of the three binaries. This page prints names and versions only.",
            bullet=True,
        ),
        *paragraph(
            "`python tools/env_lock.py` writes the files from the live environments that its "
            "`tools.json` names (option `--tools-json`); `python tools/env_lock.py --check` "
            "compares the live environments with the files and exits 1 on a difference. The "
            "stage-2 job template (`tools/stage2_job.sbatch`) runs the check before its first "
            "trial unless `ENV_CHECK=0` is set.",
            bullet=True,
        ),
        *paragraph(
            "A package that was pip-installed inside a conda environment is not in the conda "
            "records and is not listed.",
            bullet=True,
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


# -- the command line ------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_supported_tools_doc.py",
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if docs/supported-tools.md differs",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        text = render(LOCKS)
    except GenerationError as exc:
        print(f"make_supported_tools_doc: {exc}", file=sys.stderr)
        return 2
    if args.check:
        return sibling("make_rules_doc").check_page(text, PAGE, SCRIPT)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
