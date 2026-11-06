"""Small, careful readers for command-line options (no tool-specific knowledge here)."""

from __future__ import annotations

from collections.abc import Sequence


def option_values(argv: Sequence[str], *names: str, attached: bool = True) -> list[str]:
    """Values given to any of `names`, in order: `-R f`, `-Rf`, `--regions-file f`, `--x=f`.

    A short name (one letter) also matches when the value is attached (`-Rf`), unless
    `attached=False` (bedtools spells long options with one dash: `-bed` is not `-b ed`). A bare
    flag with no value yields nothing. Everything after a lone `--` is positional and ignored.
    """
    found: list[str] = []
    items = list(argv)
    for i, token in enumerate(items):
        if token == "--":
            break
        for name in names:
            if token == name:
                if i + 1 < len(items):
                    found.append(items[i + 1])
            elif name.startswith("--") and token.startswith(name + "="):
                found.append(token[len(name) + 1 :])
            elif (
                attached and len(name) == 2 and not name.startswith("--") and token.startswith(name)
            ):
                if not token.startswith("--") and len(token) > 2:
                    found.append(token[2:])
    return found


def has_option(argv: Sequence[str], *names: str, attached: bool = True) -> bool:
    """True when any of `names` appears as a flag, with or without a value."""
    for token in argv:
        if token == "--":
            break
        for name in names:
            if token == name or (name.startswith("--") and token.startswith(name + "=")):
                return True
            if attached and len(name) == 2 and not name.startswith("--") and token.startswith(name):
                if not token.startswith("--"):
                    return True
    return False


def first_positional(argv: Sequence[str]) -> str | None:
    """The first token not starting with '-': the subcommand of samtools, bcftools, bedtools."""
    for token in argv:
        if not token.startswith("-"):
            return token
    return None
