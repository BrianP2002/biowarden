#!/usr/bin/python3
"""Command shim: the executable that stands in for a tool on PATH.

This file is copied verbatim to <shim dir>/tbm-shim and every watched tool name is a symlink to it,
so it must run with nothing but the standard library of the oldest python3 on the box (3.6): no
dataclasses, no `from __future__`, no walrus. It strips `--declare k=v` markers, asks the engine
server on a unix socket whether the call may run, then either refuses (exit 97) or replaces itself
with the real tool. If the server cannot be reached the call runs unchecked and says so on stderr.

Environment: TBM_SOCK (socket path), TBM_REAL_PATH (where the real tools are), TBM_DECLARE,
TBM_WATCH (comma-separated programs the rules watch; any other program is passed straight through).
"""

import json
import os
import socket
import sys

EXIT_BLOCKED = 97
DECLARE_FLAG = "--declare"
TIMEOUT_S = 120


def split_declarations(argv):
    clean, pairs, i = [], [], 0
    while i < len(argv):
        token = argv[i]
        if token == DECLARE_FLAG and i + 1 < len(argv):
            pairs.append(argv[i + 1])
            i += 2
        elif token.startswith(DECLARE_FLAG + "="):
            pairs.append(token[len(DECLARE_FLAG) + 1 :])
            i += 1
        else:
            clean.append(token)
            i += 1
    return clean, pairs


def real_executable(tool, shim_dir):
    here = os.path.realpath(shim_dir)
    for directory in os.environ.get("TBM_REAL_PATH", os.environ.get("PATH", "")).split(":"):
        if not directory or os.path.realpath(directory) == here:
            continue
        candidate = os.path.join(directory, tool)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def ask(payload):
    path = os.environ.get("TBM_SOCK")
    if not path:
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(TIMEOUT_S)
            conn.connect(path)
            conn.sendall((json.dumps(payload) + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        return json.loads(data.decode())
    except (OSError, ValueError):
        return None


def main():
    tool = os.path.basename(sys.argv[0])
    shim_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    clean, pairs = split_declarations(sys.argv[1:])
    real = real_executable(tool, shim_dir)
    if real is None:
        sys.stderr.write(f"{tool}: command not found\n")
        return 127
    watch = os.environ.get("TBM_WATCH")
    if watch is not None and tool not in watch.split(","):
        os.execv(real, [real] + clean)  # a program no rule watches: no round trip
    reply = ask(
        {
            "tool": tool,
            "argv": clean,
            "declare": pairs,
            "env_declare": os.environ.get("TBM_DECLARE", ""),
            "cwd": os.getcwd(),
        }
    )
    if reply is None:
        sys.stderr.write(
            "[tbm] the checking layer could not be reached; this call was not checked\n"
        )
    else:
        text = reply.get("text", "")
        if text:
            sys.stderr.write(text.rstrip("\n") + "\n")
            sys.stderr.flush()
        if reply.get("action") == "block":
            return EXIT_BLOCKED
    sys.stderr.flush()  # python 3.6 block-buffers a piped stderr, and exec would drop it
    os.execv(real, [real] + clean)


if __name__ == "__main__":
    sys.exit(main())
