"""Every relative link, image and anchor in the repository's Markdown files resolves.

The README links into docs/, the docs link into src/ and envs/, and a link that breaks when a
file moves is the first thing a visitor notices. External links are not fetched.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "docs" / "research"  # git-ignored notes, never linked


def markdown_files() -> list[Path]:
    found = [*ROOT.glob("*.md"), *(ROOT / ".github").rglob("*.md")]
    for folder in ("docs", "examples"):
        found += [p for p in (ROOT / folder).rglob("*.md") if PRIVATE not in p.parents]
    return sorted(found)


def slugs(text: str) -> set[str]:
    """GitHub's anchors of the headings of a Markdown file (repeated headings get -1, -2, ...)."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    seen: dict[str, int] = {}
    found = set()
    for heading in re.findall(r"^#{1,6} +(.+?) *#*$", text, flags=re.M):
        heading = re.sub(r"<[^>]+>", "", heading.replace("`", ""))
        heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        found.add(slug if n == 0 else f"{slug}-{n}")
    return found


def targets(text: str) -> list[str]:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    found = re.findall(r"\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)", text)
    return found + re.findall(r'(?:src|srcset|href)="([^"]+)"', text)


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_and_anchors_resolve(path):
    broken = []
    for target in targets(path.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        name, _, anchor = target.partition("#")
        file = path if not name else (path.parent / name).resolve()
        if not file.exists():
            broken.append(target)
        elif anchor and file.suffix == ".md" and anchor not in slugs(file.read_text()):
            broken.append(target)

    assert not broken, f"{path.relative_to(ROOT)}: {broken}"


def test_the_anchor_rules_follow_githubs():
    text = "# A `code` title\n## Same\n## Same\n### What it catches?\n```\n# not a heading\n```\n"

    assert slugs(text) == {"a-code-title", "same", "same-1", "what-it-catches"}
    assert targets('[a](b.md#c) `[x](y.md)` ![i](p.svg) <img src="q.png">') == [
        "b.md#c",
        "p.svg",
        "q.png",
    ]
