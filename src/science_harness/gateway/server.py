"""FastMCP front end: one tool, ``predict_complex``, backed by the Gateway."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from science_harness.contracts.request import Entity
from science_harness.gateway.core import Gateway

INSTRUCTIONS = (
    "Predict a protein complex. Pass the entities in the order you want the output chains "
    "paired with them. A blocked or flagged result has isError=true: read the structured "
    "violations and the proposed fix instead of retrying the same call."
)


def build_server(gateway: Gateway) -> FastMCP:
    mcp = FastMCP("science-harness", instructions=INSTRUCTIONS)

    @mcp.tool
    def predict_complex(
        entities: list[Entity],
        backend: Literal["tfoldlike", "boltzlike"],
        apply_proposed_fix: bool = False,
    ) -> ToolResult:
        """Predict a complex with a contract-checked backend.

        entities: ordered list of {role, label, sequence}. backend: tfoldlike takes tFold-TCR role
        codes (B, A, D, G, M, N, P); boltzlike ignores roles. apply_proposed_fix: accept the
        validator's proposed reordering (it is never applied unless you ask).
        """
        result = gateway.predict_complex(entities, backend, apply_proposed_fix)
        return ToolResult(
            content=result.text,
            structured_content=result.model_dump(mode="json"),
            is_error=result.is_error,
        )

    return mcp


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="science-harness-gateway")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--ablate-precall", action="store_true", help="benchmark ablation: skip pre-call checks"
    )
    args = parser.parse_args(argv)
    gateway = Gateway(args.work_dir, args.ledger, precall=not args.ablate_precall)
    build_server(gateway).run(show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
