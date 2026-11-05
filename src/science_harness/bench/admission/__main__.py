"""Calibrate every rule: run its case source over many seeds and write the promotion table.

    python -I -m science_harness.bench.admission --seeds 0-299

A rule is promoted to `block` only if it is admitted and its false-block rate on correct,
declared calls has an exact 95% upper bound of at most 2% over at least 200 calls. The table
records the spec hash, the tool versions and the four confusion cells, so a later spec change makes
the old evidence inapplicable (the engine then falls back to warn).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from science_harness.bench.admission import known_sources, measure
from science_harness.bench.tasks.tools import TOOL_NAMES, ToolEnv
from science_harness.engine.decide import PACKAGED_PROMOTION
from science_harness.engine.spec import packaged_specs


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        seeds += range(int(low), int(high or low) + 1)
    return seeds


def tool_versions(tools: ToolEnv) -> dict[str, str]:
    versions = {}
    for name in TOOL_NAMES:
        proc = tools.run([name, "--version"], Path.cwd())
        versions[name] = (
            (proc.stdout or proc.stderr).splitlines()[0] if proc.returncode == 0 else "?"
        )
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="science_harness.bench.admission", description=__doc__)
    parser.add_argument("--seeds", default="0-299")
    parser.add_argument("--rules", nargs="*", help="rule ids (default: every rule with a source)")
    parser.add_argument("--out", type=Path, default=PACKAGED_PROMOTION)
    args = parser.parse_args(argv)
    tools = ToolEnv.discover(Path.cwd())
    if tools is None:
        print("samtools, bcftools and bedtools not found (see .local/tools.json)", file=sys.stderr)
        return 2
    seeds = parse_seeds(args.seeds)
    specs = packaged_specs()
    hashes = {r.id: s.content_hash() for s in specs for r in s.rules}
    sources = known_sources()
    wanted = args.rules or sorted(r for r in hashes if r in sources)
    table: dict[str, dict] = {}
    versions = tool_versions(tools)
    for rule_id in wanted:
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="tbm-adm-") as scratch:
            stats = measure(rule_id, specs, seeds, tools, Path(scratch))
        entry = stats.promotion_entry()
        entry.update(
            spec_hash=hashes[rule_id],
            tools=versions,
            seeds=f"{seeds[0]}-{seeds[-1]}",
            admitted=stats.admitted(),
            killed=stats.killed(),
            cells={
                f"{label}_{'declared' if declared else 'undeclared'}": {
                    "n": cell.n,
                    "failed": cell.failed,
                    "unknown": cell.unknown,
                    "crashed": cell.crashed,
                }
                for (label, declared), cell in stats.cells.items()
            },
        )
        table[rule_id] = entry
        print(
            f"{rule_id:<34} {entry['mode']:<6} admitted={entry['admitted']} "
            f"false-block {entry['right_declared'][0]}/{entry['right_declared'][1]} "
            f"(upper {entry['false_block_upper']}) recall {entry['recall']} "
            f"[{time.time() - started:.0f}s]",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(args.out.read_text()) if args.out.is_file() else {}
    merged.update(table)  # re-measuring some rules must not drop the others
    args.out.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
