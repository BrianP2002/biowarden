"""Random tiny genomes with variants and regions, boundary-enriched so index-base errors show.

Everything is a pure function of the seed. Variants sit on and next to region edges on purpose: in
the feasibility probe an off-by-one changed the output of only 15-38% of uniformly random
instances, so uniform data would hide the very faults the tasks are about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

DELETION = 7  # bases removed from chr1 in the "other build"
FLANK = 5
CONTIG_BOUNDS = {"chr1": (1500, 2500), "chr2": (1200, 2000)}
# 1-based positions next to a region: the base before it, its first and last, the base after.
EDGE_KINDS = (
    lambda r: r.start0,
    lambda r: r.start0 + 1,
    lambda r: r.end0,
    lambda r: r.end0 + 1,
)


@dataclass(frozen=True)
class Variant:
    id: str
    chrom: str
    pos: int  # 1-based
    ref: str
    alt: str


@dataclass(frozen=True)
class Region:
    chrom: str
    start0: int  # BED convention: 0-based start, end exclusive
    end0: int


@dataclass(frozen=True)
class Instance:
    contigs: dict[str, int]
    ref: dict[str, str]
    ref_b: dict[str, str]  # the same genome in another build: chr1 lacks DELETION bases at cut
    cut: int  # 0-based index in chr1 where ref_b drops DELETION bases
    regions: list[Region]
    variants: list[Variant]

    def in_regions(self) -> list[str]:
        """IDs of the variants inside any region (BED semantics: start0 <= pos-1 < end0)."""
        return [
            v.id
            for v in self.variants
            if any(r.chrom == v.chrom and r.start0 <= v.pos - 1 < r.end0 for r in self.regions)
        ]

    def context(self, v: Variant, genome: dict[str, str] | None = None) -> str:
        seq = (genome or self.ref)[v.chrom]
        i = v.pos - 1
        return f"{seq[i - FLANK : i]}[{seq[i]}]{seq[i + 1 : i + 1 + FLANK]}"

    def contexts(self, genome: dict[str, str] | None = None) -> dict[str, str]:
        return {v.id: self.context(v, genome) for v in self.variants}


def _regions(rnd: random.Random, contigs: dict[str, int]) -> list[Region]:
    regions: list[Region] = []
    for chrom, n in contigs.items():
        taken: list[tuple[int, int]] = []
        while len(taken) < 6:
            start = rnd.randint(30, n - 230)
            end = start + rnd.randint(20, 150)
            if all(end + 8 < s or start > e + 8 for s, e in taken):  # keep edges clear of others
                taken.append((start, end))
        regions += [Region(chrom, s, e) for s, e in sorted(taken)]
    return regions


NO_EDGE_MARGIN = 6


def _positions_no_edge(
    rnd: random.Random, contigs: dict[str, int], regions: list[Region]
) -> list[tuple]:
    """20 variants well inside regions, 20 well outside: only contig naming moves a count."""
    chosen: dict[tuple[str, int], None] = {}
    margin = NO_EDGE_MARGIN
    while len(chosen) < 20:
        region = rnd.choice(regions)
        chosen[(region.chrom, rnd.randint(region.start0 + margin + 1, region.end0 - margin))] = None
    while len(chosen) < 40:
        chrom = rnd.choice(list(contigs))
        pos = rnd.randint(12, contigs[chrom] - 12)
        near = any(
            r.chrom == chrom and r.start0 - margin <= pos - 1 < r.end0 + margin for r in regions
        )
        if not near:
            chosen[(chrom, pos)] = None
    order = {c: i for i, c in enumerate(contigs)}
    return sorted(chosen, key=lambda cp: (order[cp[0]], cp[1]))


def _positions(
    rnd: random.Random, contigs: dict[str, int], regions: list[Region], profile: str = "edge"
) -> list[tuple]:
    """(chrom, pos1) pairs. Profile edge: 3 on each kind of region edge, the rest random."""
    if profile == "no_edge":
        return _positions_no_edge(rnd, contigs, regions)
    chosen: dict[tuple[str, int], None] = {}
    for pick in EDGE_KINDS:
        for region in rnd.sample(regions, 3):
            chosen[(region.chrom, pick(region))] = None
    while len(chosen) < 40:
        chrom = rnd.choice(list(contigs))
        chosen[(chrom, rnd.randint(12, contigs[chrom] - 12))] = None
    order = {c: i for i, c in enumerate(contigs)}
    return sorted(chosen, key=lambda cp: (order[cp[0]], cp[1]))


def make_instance(seed: int, profile: str = "edge") -> Instance:
    rnd = random.Random(f"instance-{seed}")
    contigs = {c: rnd.randint(lo, hi) for c, (lo, hi) in CONTIG_BOUNDS.items()}
    ref = {c: "".join(rnd.choice("ACGT") for _ in range(n)) for c, n in contigs.items()}
    regions = _regions(rnd, contigs)
    variants = []
    for i, (chrom, pos) in enumerate(_positions(rnd, contigs, regions, profile)):
        base = ref[chrom][pos - 1]
        alt = rnd.choice([b for b in "ACGT" if b != base])
        variants.append(Variant(f"v{i:02d}", chrom, pos, base, alt))

    chr1_positions = [v.pos for v in variants if v.chrom == "chr1"]
    k = max(1, len(chr1_positions) // 3)
    cut = max(12, chr1_positions[k - 1] + FLANK + 3 + rnd.randint(0, 4))
    ref_b = {**ref, "chr1": ref["chr1"][:cut] + ref["chr1"][cut + DELETION :]}
    return Instance(contigs, ref, ref_b, cut, regions, variants)
