"""Shared readers for the variant-file rules of the `var` pack (standard library only).

A rule that must look at the data a call reads needs the same few things: the VCF records with
their samples and genotypes, the file names a call names, a sample list file, a BED first line.
They live here so the rules stay short. Everything reads through `Ctx.lines` (plain or gzip text,
bounded), so it works for T1 checkers only.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from biowarden.engine.argv import option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx

VCF_SUFFIXES = (".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")
MAX_VCF_LINES = 40_000
_HEADER_ID = re.compile(r"##(INFO|FORMAT)=<ID=([^,>]+)")


@dataclass(frozen=True)
class Rec:
    """One VCF data line; `samples[i]` holds the colon-separated values of sample i."""

    chrom: str
    pos: int
    id: str
    ref: str
    alts: tuple[str, ...]
    qual: str
    filt: str
    info: dict[str, str]
    fmt: tuple[str, ...]
    samples: tuple[tuple[str, ...], ...]

    def value(self, sample: int, key: str) -> str | None:
        """The FORMAT value of a sample (None when the key is absent or the field is short)."""
        if key not in self.fmt:
            return None
        parts = self.samples[sample]
        i = self.fmt.index(key)
        return parts[i] if i < len(parts) else None

    def genotype(self, sample: int) -> list[int | None]:
        return alleles(self.value(sample, "GT") or ".")


@dataclass
class Vcf:
    path: Path
    samples: list[str] = field(default_factory=list)
    records: list[Rec] = field(default_factory=list)
    truncated: bool = False
    info_ids: set[str] = field(default_factory=set)
    format_ids: set[str] = field(default_factory=set)


def alleles(gt: str) -> list[int | None]:
    """Allele indices of a GT string; None for a missing allele (`./.`, `.`)."""
    out: list[int | None] = []
    for part in re.split(r"[/|]", gt):
        out.append(int(part) if part.isdigit() else None)
    return out


def has_alt(gt: str) -> bool:
    return any(a is not None and a > 0 for a in alleles(gt))


def is_missing_gt(gt: str) -> bool:
    return all(a is None for a in alleles(gt))


def parse_info(text: str) -> dict[str, str]:
    info: dict[str, str] = {}
    if text in (".", ""):
        return info
    for item in text.split(";"):
        key, _, value = item.partition("=")
        info[key] = value
    return info


def read_vcf(ctx: Ctx, path: Path | str, max_records: int = 20_000) -> Vcf | None:
    """Header sample names and the first `max_records` data lines of a VCF (plain, gzip or bgzip).

    None when the file is a BCF, unreadable, or has no #CHROM line (not a VCF).
    """
    name = str(path)
    if name.endswith(".bcf"):
        return None
    vcf = Vcf(ctx.resolve(name))
    seen_header = False
    try:
        lines = ctx.lines(vcf.path, limit=MAX_VCF_LINES)
        try:
            for line in lines:
                if line.startswith("##"):
                    hit = _HEADER_ID.match(line)
                    if hit:
                        (vcf.info_ids if hit.group(1) == "INFO" else vcf.format_ids).add(
                            hit.group(2)
                        )
                    continue
                if line.startswith("#"):
                    vcf.samples = line.split("\t")[9:]
                    seen_header = True
                    continue
                if not seen_header or not line.strip():
                    continue
                if len(vcf.records) >= max_records:
                    vcf.truncated = True
                    break
                f = line.split("\t")
                if len(f) < 8 or not f[1].isdigit():
                    continue
                fmt = tuple(f[8].split(":")) if len(f) > 8 else ()
                samples = tuple(tuple(x.split(":")) for x in f[9:])
                vcf.records.append(
                    Rec(
                        f[0],
                        int(f[1]),
                        f[2],
                        f[3],
                        tuple(f[4].split(",")),
                        f[5],
                        f[6],
                        parse_info(f[7]),
                        fmt,
                        samples,
                    )
                )
        finally:
            lines.close()  # type: ignore[attr-defined]
    except (OSError, EOFError, zlib.error):
        return None
    return vcf if seen_header else None


def existing_vcfs(ctx: Ctx, argv: Sequence[str], skip_after: Iterable[str] = ()) -> list[Path]:
    """Existing VCF/BCF files named as arguments (not the value of an output option)."""
    skip = {"-o", "--output", *skip_after}
    found: list[Path] = []
    items = list(argv)
    for i, token in enumerate(items):
        if i and items[i - 1] in skip:
            continue
        if token.startswith("-") or not token.endswith(VCF_SUFFIXES):
            continue
        path = ctx.resolve(token)
        if path.is_file() and path not in found:
            found.append(path)
    return found


def read_list(ctx: Ctx, name: str, first_field_only: bool = True) -> list[str] | None:
    """Non-empty lines of a sample list; the first whitespace-separated token of each line."""
    try:
        lines = list(ctx.lines(name, limit=100_000))
    except (OSError, EOFError, zlib.error):
        return None
    out = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        out.append(line.split()[0] if first_field_only else line.strip())
    return out


def first_line(ctx: Ctx, name: str) -> str | None:
    """The first line of a text file (None when it is missing, unreadable or empty)."""
    try:
        lines = ctx.lines(name, limit=1)
        try:
            return next(iter(lines), None)
        finally:
            lines.close()  # type: ignore[attr-defined]
    except (OSError, EOFError, zlib.error):
        return None


def is_interval_line(line: str) -> bool:
    """`chrom start end ...` with integer start < end: a BED data line, not a header."""
    f = line.replace(",", "").split()
    return len(f) >= 3 and f[1].isdigit() and f[2].isdigit() and int(f[1]) <= int(f[2])


def values_of(argv: Sequence[str], *names: str) -> list[str]:
    return option_values(argv, *names)


def fail(
    code: str,
    message: str,
    fix: Sequence[str] = (),
    expected: object = None,
    got: object = None,
) -> CheckResult:
    return CheckResult("fail", code, message, expected=expected, got=got, fix=tuple(fix))


def unknown(code: str, message: str, fix: Sequence[str] = ()) -> CheckResult:
    return CheckResult("unknown", code, message, fix=tuple(fix))


# ---- sample subsets: sites that are monomorphic in the selected samples --------------------------

SUBSET_KEY = "subset_sites"


def monomorphic_in(vcf: Vcf, which: Sequence[int], fixed_counts: bool = False) -> list[Rec]:
    """Records at which none of the samples `which` carries an alternate allele (when
    `fixed_counts`, a site where they are all homozygous alternate counts too: the minor allele
    count of PLINK is 0 there)."""
    out: list[Rec] = []
    for rec in vcf.records:
        seen = {a for i in which for a in rec.genotype(i) if a is not None}
        mono = len(seen) <= 1 if fixed_counts else not any(a > 0 for a in seen)
        if mono:
            out.append(rec)
    return out


def subset_verdict(
    call: Call,
    key: str,
    tool: str,
    file_name: str,
    vcf: Vcf,
    which: Sequence[int],
    filtered: bool,
    fix: Sequence[str],
    fixed_counts: bool = False,
) -> CheckResult:
    """The shared decision of the four subset rules (bcftools, vcftools, plink, GATK)."""
    mono = monomorphic_in(vcf, which, fixed_counts)
    if not mono:
        return CheckResult("pass", "ok")
    declared = call.declared.get(key)
    if declared in ("filtered_downstream", "keep_all_sites"):
        return CheckResult("pass", "ok")
    if filtered:
        return CheckResult("pass", "ok")
    first = mono[0]
    example = first.id or f"{first.chrom}:{first.pos}"
    facts = (
        f"{len(mono)} of {len(vcf.records)} sites of {file_name} have no alternate allele in the "
        f"{len(which)} selected samples (for example {example})"
    )
    if declared == "variant_sites_only":
        return fail(
            "declared_variant_sites_only",
            f"you declared {key}=variant_sites_only but {tool} keeps every site when it selects "
            f"samples: {facts}.",
            fix=fix,
            expected="variant_sites_only",
            got="all sites",
        )
    return fail(
        "monomorphic_sites_kept",
        f"selecting samples does not select variants: {tool} writes every site of {file_name}, "
        f"{facts}; a count or list of its records includes the sites where nobody carries an "
        "alternate allele.",
        fix=(
            *fix,
            f"or --declare {key}=keep_all_sites if the monomorphic sites are wanted, "
            f"{key}=filtered_downstream if a later stage drops them",
        ),
        expected=["keep_all_sites", "variant_sites_only", "filtered_downstream"],
    )
