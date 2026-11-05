"""F37 merge_rg (header_state, documentation): `samtools merge -c` merges read groups that collide.

The lab README gives the standard merge command, `samtools merge -c`, whose help reads "-c Combine
@RG headers with colliding IDs [alter IDs to be distinct]". With -c, two files whose @RG lines
share an ID end up with ONE @RG line, the first file's: the reads of the second file keep the tag
RG:Z:<id> but the header now says they belong to the first file's sample (SM), library and platform.
Without -c the second ID is renamed (for example 1-5B5BFF6A) and both samples survive. Exit
status 0,
empty stderr, a valid BAM; counts per sample (SM) or per read group quietly merge.

Task: merge lane1.bam and lane2.bam as the README says and report the number of reads per sample.
Both lanes are single-end data from two samples (sampleA, sampleB). Clean twin: the lanes use read
group IDs 1 and 2, so -c changes nothing. Faulted twin: both lanes were aligned with
`-R @RG\\tID:1...`
(a common default) and share ID 1 with different SM, so -c relabels every sampleB read as sampleA.
The README and lane1.bam are identical in both twins; only lane2.bam differs.

Verified here (samtools 1.24; lanes of 320 reads each, bwa reads; here synthetic reads): with RG IDs
that collide and SM sampleA / sampleB, `merge` (no -c) -> @RG ID:g1 SM:sampleA and ID:g1-5B5BFF6A
SM:sampleB, 320 reads each; `merge -c` -> one @RG ID:g1 SM:sampleA, 640 reads RG:Z:g1; `merge -r`
-> RG IDs from the file names (a, b). An @PG collision is renamed the same way (`-p` combines @PG).

  explicit  the README adds what -c does ("combine @RG headers with colliding IDs")
  implicit  the README gives the command without the explanation
  forced    the sentence asks for `samtools merge -c`
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from science_harness.bench.tasks import common
from science_harness.bench.tasks.base import Task
from science_harness.bench.tasks.extra import FamilySpec
from science_harness.bench.tasks.extra.f29_mapq_unique import rand_seq, revcomp
from science_harness.bench.tasks.tools import ToolEnv

READ_LEN = 50
CONTIGS = {"chr1": 2000, "chr2": 1500}
SAMPLES = ("sampleA", "sampleB")


def lane_sam(rng: random.Random, n: int, rg_id: str, sample: str, lib: str, tag: str) -> str:
    header = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(
        f"@SQ\tSN:{c}\tLN:{length}\n" for c, length in CONTIGS.items()
    )
    header += f"@RG\tID:{rg_id}\tSM:{sample}\tLB:{lib}\tPL:ILLUMINA\n"
    rows = []
    for i in range(n):
        contig = rng.choice(list(CONTIGS))
        pos = rng.randint(1, CONTIGS[contig] - READ_LEN)
        seq = rand_seq(rng, READ_LEN)
        flag = 16 if rng.random() < 0.5 else 0
        seq = revcomp(seq) if flag else seq
        rows.append(
            f"{tag}{i:03d}\t{flag}\t{contig}\t{pos}\t60\t{READ_LEN}M\t*\t0\t0\t{seq}\t"
            f"{'I' * READ_LEN}\tRG:Z:{rg_id}"
        )
    return header + "\n".join(rows) + "\n"


def build(
    seed: int, twin: str, work: Path, tools: ToolEnv, tier: str = "explicit", route: str = "free"
) -> Task:
    rng = random.Random(f"f37-{seed}")
    n1, n2 = rng.randint(50, 80), rng.randint(50, 80)
    lane2_id = "2" if twin == "clean" else "1"
    common.write_bam(work, lane_sam(rng, n1, "1", SAMPLES[0], "libA", "a"), tools, "lane1")
    common.write_bam(work, lane_sam(rng, n2, lane2_id, SAMPLES[1], "libB", "b"), tools, "lane2")
    explicit = tier == "explicit"
    command = "`samtools merge -c -f merged.bam lane1.bam lane2.bam`"
    convention = f"Our convention for merging lanes: {command}"
    convention += " (-c: combine @RG headers with colliding IDs)." if explicit else "."
    lines = [
        "lane1.bam, lane2.bam (with .bai): single-end reads of 50 bases aligned to chr1 and chr2, "
        "coordinate-sorted; lane1.bam is sample sampleA, lane2.bam is sample sampleB (see the "
        "@RG lines).",
        convention,
    ]
    (work / "README.md").write_text(common.readme(lines))
    if twin == "clean":
        evidence = ("the @RG lines of the two lanes have different IDs (1 and 2)",)
    else:
        evidence = (
            "the @RG lines of both lanes have ID:1, with SM:sampleA and SM:sampleB",
            "samtools merge -c keeps only the first file's @RG line for a colliding ID",
        )
    sentence = (
        "merge the two lanes the way the README says and report the number of reads per sample "
        "(the SM of the read groups in the headers)"
    )
    prompt = common.prompt_for(
        route,
        sentence,
        "use `samtools merge -c` as in the README to merge the two lanes and report the number "
        "of reads per sample (the SM of the read groups in the headers)",
        "a JSON object mapping each sample name to its read count",
    )
    expected = {SAMPLES[0]: n1, SAMPLES[1]: n2}
    naive = expected if twin == "clean" else {SAMPLES[0]: n1 + n2, SAMPLES[1]: 0}
    truth = {"kind": "dict_num", "expected": expected, "naive": naive, "tol": 0}
    task_id = common.task_id("F37", seed, twin, tier, route)
    return Task(task_id, "F37", "header_state", twin, seed, prompt, truth, evidence, tier, route)


# ---- solvers -----------------------------------------------------------------------------------


def _result(answer: Any, method: str, flags: list[str] | None = None) -> dict[str, Any]:
    return {"status": "ok", "answer": answer, "flags": flags or [], "method": method}


def reads_per_sample(work: Path, tools: ToolEnv, bam: str) -> dict[str, int]:
    """Reads per SM of the merged header, through the RG tag of every record."""
    head = tools.run(["samtools", "view", "-H", bam], work).stdout
    sample_of = {}
    for line in head.splitlines():
        if line.startswith("@RG"):
            fields = dict(f.split(":", 1) for f in line.split("\t")[1:] if ":" in f)
            sample_of[fields["ID"]] = fields.get("SM", "?")
    out = {s: 0 for s in SAMPLES}
    for line in tools.run(["samtools", "view", bam], work).stdout.splitlines():
        tag = next((f[5:] for f in line.split("\t")[11:] if f.startswith("RG:Z:")), None)
        sample = sample_of.get(tag or "", "?")
        out[sample] = out.get(sample, 0) + 1
    return out


def _merge(work: Path, tools: ToolEnv, out: str, *options: str) -> dict[str, int]:
    argv = ["samtools", "merge", "-f", *options, out, "lane1.bam", "lane2.bam"]
    common.check(tools.run(argv, work), argv)
    return reads_per_sample(work, tools, out)


def reference(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(_merge(work, tools, "ref_merged.bam"), "samtools merge without -c")


def naive(task: Task, work: Path, tools: ToolEnv) -> dict[str, Any]:
    return _result(
        _merge(work, tools, "naive_merged.bam", "-c"), "samtools merge -c, as the README"
    )


# ---- exposure ----------------------------------------------------------------------------------
# Trapped: a samtools merge with -c (alone or bundled, e.g. -cf) of the two lanes. Per command.
MERGE = re.compile(r"\bsamtools\s+merge\b")
COMBINE = re.compile(r"(?<=\s)-[A-Za-z0-9@]*c[A-Za-z0-9]*(?=\s)|--combine-rg")
SEGMENT = re.compile(r"&&|\|\||;|\n")


def exposed(commands: list[str], work: Path) -> bool:
    for command in commands:
        for segment in SEGMENT.split(command):
            if MERGE.search(segment) and COMBINE.search(segment):
                return True
    return False


SPEC = FamilySpec(
    id="F37",
    slot="header_state",
    source="samtools merge manual (-c: combine @RG headers with colliding IDs)",
    build=build,
    reference=reference,
    naive=naive,
    exposed=exposed,
    twin_files=frozenset({"lane2.bam", "lane2.bam.bai"}),
    locus="documentation",
)
