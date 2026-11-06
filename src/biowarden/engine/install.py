"""Create a shim directory: one copy of the client plus a symlink per watched tool name."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

CLIENT = Path(__file__).with_name("shim_client.py")
# Tools whose calls carry call rules, and interpreters that may name a data file (fact rules).
SHIM_NAMES = (
    "samtools",
    "bcftools",
    "bedtools",
    "awk",
    "gawk",
    "mawk",
    "python3",
    "python",
    "perl",
    "Rscript",
    "mockfold",
)


def install_shims(
    shim_dir: Path,
    real_dirs: Sequence[str | Path],
    names: Sequence[str] = SHIM_NAMES,
    interpreter: str = "/usr/bin/python3",
) -> list[str]:
    """Write biowarden-shim and a symlink for every name that exists in one of `real_dirs`."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    script = shim_dir / "biowarden-shim"
    source = CLIENT.read_text()
    script.write_text(source.replace("#!/usr/bin/python3", f"#!{interpreter}", 1))
    script.chmod(0o755)
    made: list[str] = []
    for name in names:
        if any((Path(d) / name).is_file() for d in real_dirs):
            link = shim_dir / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to("biowarden-shim")
            made.append(name)
    return made
