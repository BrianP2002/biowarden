"""Shared readers for the `struct` rules (protein structure files and the tools around them).

Nothing here is a rule. It holds what several rules need: the file-name patterns, the text of a
command (including the scripts it runs), the interpreters a library rule watches, and small
gemmi-based readers that never raise (a file that cannot be read gives None, and the checker
answers `unknown`). The engine runs on the host, so gemmi is imported inside the functions.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any

from science_harness.engine.call import Call
from science_harness.engine.checks import Ctx

# structure files named anywhere in a command (also inside python -c code or a heredoc)
STRUCT_FILE = r"(?<![\w.+/@~-])[\w.+/@~-]*\.(?:pdb|ent|cif|mmcif)(?:\.gz)?(?![\w-]|\.\w)"
STRUCT_RE = re.compile(STRUCT_FILE)
FASTA_RE = re.compile(r"(?<![\w.+/@~-])[\w.+/@~-]*\.(?:fasta|fa|faa|fas)(?![\w-]|\.\w)")
SCRIPT_RE = re.compile(r"(?<![\w./~+-])[\w./~+-]+\.(?:py|awk|pl|sh|rb|R)(?![\w.])")

# named interpreters of the extended environment plus the system ones
PYTHON_TOOLS = ["python-struct", "python-bio", "python3", "python"]
SCRIPTING = re.compile(
    r"[gmn]?awk|python[\d.-]*\w*|perl[\d.]*|ruby|Rscript|R|julia|node"
    r"|bash|sh|zsh|dash|ksh|env|xargs|parallel|timeout|time|nohup|srun"
)
VIEWERS = frozenset(
    "head tail cat less more zless zmore wc ls file stat du md5sum sha1sum sha256sum sha512sum "
    "cksum zcat gzcat bzcat xzcat gunzip gzip bgzip nl tac rev fold cut paste column sort uniq tr "
    "sed grep egrep fgrep zgrep xxd od hexdump cp mv rm ln diff cmp touch mkdir basename dirname "
    "readlink realpath tree echo printf test find vi vim nano emacs scp rsync tar zip unzip".split()
)
MAX_SCRIPTS = 5


def script_texts(call: Call, ctx: Ctx) -> list[str]:
    """Text of the script files named in the command (at most MAX_SCRIPTS, 2000 lines each)."""
    found: list[str] = []
    seen: set[Path] = set()
    for token in call.argv:
        for hit in SCRIPT_RE.finditer(token):
            path = ctx.resolve(hit.group(0))
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                found.append("\n".join(ctx.lines(path, limit=2000)))
            except (OSError, EOFError, zlib.error):
                continue
            if len(found) >= MAX_SCRIPTS:
                return found
    return found


def command_text(call: Call, ctx: Ctx) -> str:
    """The command as code to look at: tool, arguments, the scripts it runs and earlier words."""
    return "\n".join([call.tool, *call.argv, *script_texts(call, ctx)])


def reads_structure_as_code(tool: str, text: str, markers: re.Pattern[str]) -> bool:
    """Does this command compute on a structure file? Scripting tools only when the code matches
    `markers`; viewers never; any other program is assumed to read it."""
    if SCRIPTING.fullmatch(tool):
        return markers.search(text) is not None
    return tool not in VIEWERS


def is_cif(path: Path | str) -> bool:
    name = str(path).lower()
    return name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz"))


def file_text(ctx: Ctx, path: Path | str) -> str | None:
    try:
        return "\n".join(ctx.lines(path))
    except (OSError, EOFError, zlib.error):
        return None


def cif_block(ctx: Ctx, path: Path | str) -> Any | None:
    """The first data block of an mmCIF file (gemmi.cif.Block), None when unreadable."""
    try:
        import gemmi
    except ImportError:
        return None
    text = file_text(ctx, path)
    if text is None:
        return None
    try:
        return gemmi.cif.read_string(text)[0]
    except Exception:
        return None


def read_structure(ctx: Ctx, path: Path | str) -> Any | None:
    """A gemmi.Structure for a PDB or mmCIF file, None when gemmi is missing or the file bad."""
    try:
        import gemmi
    except ImportError:
        return None
    text = file_text(ctx, path)
    if text is None:
        return None
    try:
        if is_cif(path):
            structure = gemmi.make_structure_from_block(gemmi.cif.read_string(text)[0])
        else:
            structure = gemmi.read_pdb_string(text)
    except Exception:
        return None
    empty = len(structure) == 0 or sum(len(chain) for chain in structure[0]) == 0
    return None if empty else structure


def existing_files(call: Call, ctx: Ctx, pattern: re.Pattern[str]) -> list[Path]:
    """Files matching `pattern` named in the arguments (then in the earlier words of the line)."""
    found: list[Path] = []
    for token in (*call.argv, *call.context):
        for hit in pattern.finditer(token):
            path = ctx.resolve(hit.group(0))
            if path.is_file() and path not in found:
                found.append(path)
    return found


def positionals(argv: tuple[str, ...], skip_values: frozenset[str] = frozenset()) -> list[str]:
    """Arguments that are not options (and not the value of an option in `skip_values`)."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token.startswith("-") and len(token) > 1:
            skip = token in skip_values
            continue
        out.append(token)
    return out


def residue_label(res: Any) -> str:
    return f"{res.name} {res.seqid.num}{res.seqid.icode.strip()}"
