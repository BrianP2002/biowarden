"""Run any command, agent or shell with the checking layer on its PATH.

    biowarden [--mode warn|block|promoted] [--ledger FILE] -- COMMAND [ARGS...]

Starts the engine in this process, puts a directory of shims in front of the real tools on PATH and
runs COMMAND with that PATH, so every samtools, bcftools, bedtools, awk, python3 or perl call the
command makes (and every call its child processes make) is checked before it runs. The exit status
is the command's. This is the shim front end without a sandbox: it sees processes, not command
strings, so a pipeline stage that names no file is not covered (the benchmark runner adds that).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from biowarden.engine.decide import BlockState, Engine, Mode, Promotion
from biowarden.engine.install import install_shims
from biowarden.engine.server import EngineServer, PathMap, serve_in_thread
from biowarden.engine.spec import collect_specs
from biowarden.engine.tools import (
    find_binaries,
    import_python_rules,
    path_dirs,
    watched_tools,
)
from biowarden.ledger import Ledger

SOCKET_NAME = "e.sock"
SOCKET_PATH_MAX = 107  # bytes of an AF_UNIX path


def private_dir() -> Path | None:
    """A new private directory for the shims, the socket and the block counts: under $TMPDIR, or
    under /tmp when $TMPDIR is too long for a unix socket path (None if neither is short enough)."""
    for root in (tempfile.gettempdir(), "/tmp"):
        longest = Path(root) / "biowarden-guard-XXXXXXXX" / SOCKET_NAME
        if len(os.fsencode(longest)) <= SOCKET_PATH_MAX:
            return Path(tempfile.mkdtemp(prefix="biowarden-guard-", dir=root))
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biowarden", description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="promoted", choices=["warn", "block", "promoted"])
    parser.add_argument("--ledger", type=Path, help="append every decision to this ledger")
    parser.add_argument(
        "--specs", type=Path, action="append", default=[], help="directory of extra rule specs"
    )
    parser.add_argument("--no-packaged", action="store_true", help="do not load the shipped rules")
    parser.add_argument(
        "--watch", action="append", default=[], help="also put a shim in front of this program"
    )
    parser.add_argument(
        "--trust-python",
        action="store_true",
        help="import *.py checkers from --specs directories (this runs their code)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give a command after --, for example: biowarden -- bash")
    dirs = path_dirs()
    if args.trust_python:
        import_python_rules(args.specs)
    specs = collect_specs(args.specs, packaged=not args.no_packaged)
    names = watched_tools(specs, args.watch)
    tmp = private_dir()
    if tmp is None:
        parser.error("$TMPDIR is too long for a unix socket path: set it to a shorter directory")
    try:
        shims = tmp / "shims"
        if not install_shims(shims, dirs, names, sys.executable):
            parser.error(f"none of the watched programs ({', '.join(names)}) were found on PATH")
        mode: Mode = args.mode
        cwd = Path.cwd().resolve()
        engine = Engine(
            specs,
            mode=mode,
            promotion=Promotion.packaged(),
            real_bins=find_binaries(names, dirs),
            real_env=dict(os.environ),
            state=BlockState(tmp / "blocks.json"),
            ledger=Ledger(args.ledger) if args.ledger else None,
        )
        server = EngineServer(tmp / SOCKET_NAME, engine, PathMap(cwd, str(cwd)))
        serve_in_thread(server)
        env = dict(os.environ)
        env.update(
            PATH=os.pathsep.join([str(shims), *dirs]),
            BIOWARDEN_SOCK=str(server.socket_path),
            BIOWARDEN_REAL_PATH=os.pathsep.join(dirs),
        )
        try:
            return subprocess.run(command, env=env, check=False).returncode
        except FileNotFoundError:
            print(f"biowarden: command not found: {command[0]}", file=sys.stderr)
            return 127
        finally:
            server.shutdown()
            server.server_close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
