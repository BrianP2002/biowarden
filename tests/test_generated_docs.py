"""docs/supported-tools.md and docs/rules.md are what their generators write, and publishable.

tools/make_supported_tools_doc.py writes the first page from envs/locks/; tools/make_rules_doc.py
writes the second from the rule specs, the promotion table and the family map. When a test here
fails, a page no longer matches its sources: regenerate it with the command in the message and
commit it together with the change that made it differ (a lock file, a rule spec, the promotion
table, a family).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

from biowarden.engine.spec import packaged_specs

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LOCKS = ROOT / "envs" / "locks"
ENGINE_SPECS = ROOT / "src" / "biowarden" / "engine" / "specs"
FAMILY_RULES = ROOT / "src" / "biowarden" / "bench" / "tasks" / "family_rules.json"
PAGES = {"make_supported_tools_doc": "supported-tools.md", "make_rules_doc": "rules.md"}
# a path in published text, as in tools/env_lock.py: '/' after a space, '=' or a quote, then a
# name and another '/'
PATH_LIKE = re.compile(r"(?:^|[\s=\"'(])/(?!/)[^\s/]+/")
SEPARATOR = set("|-: ")


def load(name: str):
    """tools/<name>.py as a module. The generators load each other through sys.modules, so a
    module loaded here is the one they use."""
    path = ROOT / "tools" / f"{name}.py"
    known = sys.modules.get(name)
    if known is not None and Path(known.__file__) == path:
        return known
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    """page name -> the text its generator gives now."""
    return {page: load(name).render() for name, page in PAGES.items()}


def committed(page: str) -> str:
    return (DOCS / page).read_text(encoding="utf-8")


def table_rows(text: str) -> list[list[str]]:
    """The body rows of the markdown tables in `text` (not the header rows), as lists of cells."""
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.startswith("| ") or set(line) <= SEPARATOR:
            continue
        following = lines[i + 1] if i + 1 < len(lines) else ""
        if following.startswith("|") and set(following) <= SEPARATOR:
            continue  # the header row of a table
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def private_paths() -> list[str]:
    """Where this checkout and this user's home are, and the shapes of a home directory: text
    that holds one of them is not publishable."""
    paths = ["/home/", "/Users/", str(ROOT)]
    try:
        paths.append(str(Path.home()))
    except RuntimeError:  # no home directory is known
        pass
    return [path for path in paths if len(path) > 1]


def number(text: str) -> int:
    return int(text.replace(",", ""))


def flat(text: str) -> str:
    """`text` on one line, so that a sentence the generator wrapped can be searched."""
    return " ".join(text.split())


# -- the pages are current ------------------------------------------------------------------------


@pytest.mark.parametrize("name", PAGES)
def test_the_page_is_what_the_generator_writes(name, capsys):
    page = PAGES[name]
    code = load(name).main(["--check"])
    assert code == 0, (
        f"docs/{page} is out of date; regenerate it with: python tools/{name}.py > docs/{page}\n"
        + capsys.readouterr().err
    )


def test_the_families_page_is_what_its_generator_writes(capsys):
    """The domain table of supported-tools.md counts the families of this page's packs."""
    load("make_families_doc").main()
    assert committed("families.md") == capsys.readouterr().out, (
        "docs/families.md is out of date; regenerate it with: "
        "python tools/make_families_doc.py > docs/families.md"
    )


@pytest.mark.parametrize("name", PAGES)
def test_check_fails_with_the_command_to_regenerate_and_writes_nothing(
    name, tmp_path, monkeypatch, capsys
):
    module = load(name)
    page = tmp_path / PAGES[name]
    monkeypatch.setattr(module, "PAGE", page)
    assert module.main(["--check"]) == 1  # the page does not exist
    assert "does not exist" in capsys.readouterr().err
    page.write_text("stale\n")
    assert module.main(["--check"]) == 1  # the page is out of date
    assert f"regenerate it with: python tools/{name}.py > " in capsys.readouterr().err
    assert page.read_text() == "stale\n"
    page.write_text(committed(PAGES[name]))
    assert module.main(["--check"]) == 0


# -- the pages can be published -------------------------------------------------------------------


@pytest.mark.parametrize("page", PAGES.values())
def test_the_page_holds_no_absolute_path(page, generated):
    assert PATH_LIKE.search(" see /usr/lib/x") and PATH_LIKE.search("PATH=/opt/x/bin")  # the probe
    for where, text in (("generated", generated[page]), ("committed", committed(page))):
        for needle in private_paths():
            assert needle not in text, f"{where} docs/{page} holds {needle}"
        hit = PATH_LIKE.search(text)
        assert hit is None, f"{where} docs/{page} holds a path: {hit and hit.group(0)!r}"


@pytest.mark.parametrize("page", PAGES.values())
def test_the_page_prints_no_url_hash_or_non_ascii_text(page, generated):
    for where, text in (("generated", generated[page]), ("committed", committed(page))):
        assert text.isascii(), f"{where} docs/{page} holds a non-ASCII character"
        assert "http://" not in text and "https://" not in text, f"{where} docs/{page}: a URL"
        assert not re.search(r"\b[0-9a-f]{64}\b", text), f"{where} docs/{page}: a sha256"
        assert text.endswith("\n") and not text.endswith("\n\n"), f"{where} docs/{page}"


@pytest.mark.parametrize("page", PAGES.values())
def test_the_links_into_the_repository_resolve(page, generated):
    """Links to files outside docs/ are written by the generators, so they are checked here; the
    links between the pages of docs/ belong to a link check of the whole tree."""
    links = re.findall(r"\]\((\.\./[^)#]+)", generated[page])
    assert links, f"docs/{page} links to nothing in the repository"
    for target in links:
        assert (DOCS / target).exists(), f"docs/{page} links to {target}, which is not there"


# -- rules.md -------------------------------------------------------------------------------------


def rule_rows(text: str) -> dict[str, dict[str, str]]:
    """rule id -> {column: cell} for the rows of the rule tables of docs/rules.md; the Rule cell
    is the link `[`id`](path)`."""
    headers = load("make_rules_doc").HEADERS
    return {
        cells[0].split("`")[1]: dict(zip(headers, cells, strict=True))
        for cells in table_rows(text)
        if len(cells) == len(headers) and cells[0].startswith("[`")
    }


def test_the_rules_page_has_one_row_per_rule_with_the_numbers_of_the_promotion_table():
    table = json.loads((ENGINE_SPECS / "promotion.json").read_text())  # the file, not the loader
    rules = [rule for spec in packaged_specs() for rule in spec.rules]
    rows = rule_rows(committed("rules.md"))
    assert sorted(rows) == sorted(rule.id for rule in rules), "one row per rule is needed"
    for rule in rules:
        cells, entry = rows[rule.id], table[rule.id]
        failed, calls = entry["right_declared"]
        wrong = sum(entry["cells"][c]["n"] for c in ("wrong_declared", "wrong_undeclared"))
        killed = entry["killed"]
        if rule.declare is None:
            # the table holds every call of such a rule in both cells; the page counts it once
            assert entry["cells"]["wrong_declared"] == entry["cells"]["wrong_undeclared"], rule.id
            wrong, killed = wrong // 2, killed // 2
        assert cells["Mode"] == entry["mode"], rule.id
        assert cells["Seeds"] == entry["seeds"], rule.id
        assert cells["False blocks"] == f"{failed:,} of {calls:,}", rule.id
        upper = float(cells["Upper bound"].removesuffix("%")) / 100
        assert upper == pytest.approx(entry["false_block_upper"], abs=1e-9), rule.id
        found = re.fullmatch(r"([\d.]+)% \(([\d,]+) of ([\d,]+)\)", cells["Recall"])
        assert found, f"{rule.id}: recall cell {cells['Recall']!r}"
        assert float(found[1]) / 100 == pytest.approx(entry["recall"], abs=1e-9), rule.id
        assert (number(found[2]), number(found[3])) == (killed, wrong), rule.id
        assert killed / wrong == pytest.approx(entry["recall"], abs=1e-4), rule.id


def test_the_families_column_is_the_family_map_reversed():
    families = json.loads(FAMILY_RULES.read_text())["families"]
    for rule_id, cells in rule_rows(committed("rules.md")).items():
        expected = {
            family
            for family, entry in families.items()
            if rule_id in entry.get("rules", [])
            or any(rule_id in rules for rules in entry.get("route_rules", {}).values())
        }
        assert set(re.findall(r"F\d+", cells["Families"])) == expected, rule_id
        if not expected:
            assert cells["Families"] == "none", rule_id


def test_each_rule_links_to_the_spec_file_that_defines_it():
    for rule_id, cells in rule_rows(committed("rules.md")).items():
        target = re.search(r"\]\(\.\./([^)]+)\)", cells["Rule"])[1]
        spec = ROOT / target
        assert spec.parent == ENGINE_SPECS and spec.suffix == ".yaml", rule_id
        assert f"id: {rule_id}\n" in spec.read_text(), f"{rule_id} is not defined in {spec.name}"


def test_a_call_rule_that_needs_an_option_says_so_in_its_watches_cell():
    limit = load("make_rules_doc").OPTIONS_SHOWN
    rules = [rule for spec in packaged_specs() for rule in spec.rules if rule.kind == "call"]
    rows = rule_rows(committed("rules.md"))
    assert any(rule.applies.options_any for rule in rules)
    for rule in rules:
        options, watches = rule.applies.options_any, rows[rule.id]["Watches"]
        if not options:
            assert ", with " not in watches, rule.id
        elif len(options) > limit:
            assert watches.endswith(f", with any of {len(options)} options"), rule.id
        else:
            shown = re.findall(r"`([^`]+)`", watches.partition(", with ")[2])
            assert shown == options, rule.id


def test_every_fact_rule_has_a_readable_file_pattern():
    rules_doc = load("make_rules_doc")
    facts = [rule for spec in packaged_specs() for rule in spec.rules if rule.kind == "fact"]
    assert facts
    for rule in facts:
        names = rules_doc.file_names(rule.applies.file_patterns)
        assert names and all(name.startswith("*") for name in names), rule.id
    fastq = r"(?<![\w-])[\w-]*\.f(?:ast)?q(?:\.b?gz)?(?![\w])"
    assert rules_doc.file_names([fastq]) == ["*.fq", "*.fastq"]


def test_a_spec_file_that_belongs_to_no_domain_stops_the_generator():
    rules_doc = load("make_rules_doc")
    with pytest.raises(rules_doc.GenerationError, match=r"zzz_new\.yaml belongs to 0 domains"):
        rules_doc.domain_of("zzz_new")


def promotion_without(monkeypatch, rules_doc, change):
    """Make Promotion.packaged() give the shipped table after `change(entries)`."""
    entries = json.loads((ENGINE_SPECS / "promotion.json").read_text())
    change(entries)
    monkeypatch.setattr(rules_doc.Promotion, "packaged", classmethod(lambda cls: cls(entries)))


def test_a_rule_without_a_promotion_entry_is_marked_and_only_warns(monkeypatch):
    rules_doc = load("make_rules_doc")
    gone = "samtools.fixmate_grouping"
    promotion_without(monkeypatch, rules_doc, lambda entries: entries.pop(gone))
    text = rules_doc.render()
    row = next(line for line in text.splitlines() if line.startswith(f"| [`{gone}`]"))
    assert row.endswith("| warn (not calibrated) | - | - | - | - |"), row
    assert f"Rules marked in the Mode column ({gone})" in flat(text)
    assert "107 rules block and 1 only warn" in flat(text)


def test_a_rule_without_a_declaration_with_different_cells_of_wrong_calls_stops_the_generator(
    monkeypatch,
):
    rules_doc = load("make_rules_doc")

    def change(entries):
        entries["samtools.fixmate_grouping"]["cells"]["wrong_undeclared"]["n"] += 1

    promotion_without(monkeypatch, rules_doc, change)
    with pytest.raises(rules_doc.GenerationError, match="fixmate_grouping: no declaration"):
        rules_doc.render()


# -- supported-tools.md ---------------------------------------------------------------------------


def read_tsv(path: Path) -> dict[str, str]:
    """package name -> version of a conda lock file."""
    header, *lines = path.read_text().splitlines()
    columns = header.split("\t")
    records = (dict(zip(columns, line.split("\t"), strict=True)) for line in lines)
    return {record["name"]: record["version"] for record in records}


def test_the_versions_are_those_of_the_lock_files():
    page = committed("supported-tools.md")
    tool_lines = (LOCKS / "tool-versions.txt").read_text().splitlines()
    freeze = {
        normalise(line.partition("==")[0]): line.partition("==")[2]
        for line in (LOCKS / "venv-pip-freeze.txt").read_text().splitlines()
        if "==" in line
    }
    stems = set()
    for section in page.split("\n### ")[1:]:
        title, _, body = section.partition("\n")
        rows = table_rows(body)
        assert rows, title
        lock = re.search(r"Lock file: \[`([\w.-]+)`\]", body)
        if title.startswith("Tool binaries"):
            assert [row[0].split(" ")[0] for row in rows] == [
                "samtools",
                "bcftools",
                "bedtools",
                "bubblewrap",
            ]
            for name, version, *_ in rows:
                key = "bwrap" if name.startswith("bubblewrap") else name
                line = next(x for x in tool_lines if x.startswith(f"{key}: "))
                assert re.search(r"\d+(?:\.\d+)+", line).group(0) == version, name
        elif lock.group(1).endswith(".tsv"):
            stems.add(lock.group(1).removesuffix(".tsv"))
            versions = read_tsv(LOCKS / lock.group(1))
            for name, version, package in rows:
                assert versions[package.strip("`")] == version, f"{title}: {name}"
            facts = re.search(r"(\d+) packages, Python ([\d.]+)\.", flat(body))
            assert (int(facts[1]), facts[2]) == (len(versions), versions["python"]), title
        else:
            for name, version, package in rows:
                assert freeze[normalise(package.strip("`"))] == version, name
            facts = re.search(
                r"(\d+) pinned packages; the interpreter is Python ([\d.]+)\.", flat(body)
            )
            assert int(facts[1]) == len(freeze), title
            assert f"python: Python {facts[2]}" in tool_lines, title
    assert stems == {p.stem for p in LOCKS.glob("*.tsv")}, "a lock file has no section"


def test_the_binaries_are_those_the_calibration_recorded():
    """The promotion table records the tool versions each rule was measured with."""
    table = json.loads((ENGINE_SPECS / "promotion.json").read_text())
    recorded = {tuple(sorted(entry["tools"].items())) for entry in table.values()}
    assert len(recorded) == 1, "the entries of the promotion table saw different tool versions"
    text = flat(committed("supported-tools.md"))
    for name, line in recorded.pop():
        version = re.search(r"\d+(?:\.\d+)+", line).group(0)
        assert f"{name} {version}" in text, name
    assert f"for all {len(table)} rules." in text


def test_the_harness_table_pins_the_dependencies_of_pyproject():
    harness = {normalise(dist) for _, dist in load("make_supported_tools_doc").HARNESS}
    text = (ROOT / "pyproject.toml").read_text()
    wanted = set()
    for pattern in (r"^dependencies = \[(.*?)\]", r"^bench = \[(.*?)\]"):
        block = re.search(pattern, text, re.S | re.M)
        assert block, pattern
        wanted |= {normalise(name) for name in re.findall(r'"([A-Za-z0-9_.-]+)', block[1])}
    assert wanted <= harness, f"pyproject.toml needs {sorted(wanted - harness)} in HARNESS"


def copy_locks(tmp_path: Path) -> Path:
    shutil.copytree(LOCKS, tmp_path / "locks")
    return tmp_path / "locks"


def test_a_mapped_package_missing_from_its_lock_stops_the_generator(tmp_path, monkeypatch, capsys):
    module = load("make_supported_tools_doc")
    stem, _, mapping = module.ENVIRONMENTS[0]
    shown, package = mapping[0]
    locks = copy_locks(tmp_path)
    lock = locks / f"{stem}.tsv"
    lines = lock.read_text().splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(f"{package}\t")]
    assert len(kept) == len(lines) - 1
    lock.write_text("".join(kept))
    message = re.escape(f"{shown}: the conda package '{package}' is not in ")
    with pytest.raises(module.GenerationError, match=message):
        module.render(locks)
    monkeypatch.setattr(module, "LOCKS", locks)
    assert module.main([]) == 2
    assert shown in capsys.readouterr().err


def test_a_lock_file_that_the_mapping_does_not_know_stops_the_generator(tmp_path):
    module = load("make_supported_tools_doc")
    locks = copy_locks(tmp_path)
    (locks / "unlisted-env.tsv").write_text(
        (locks / f"{module.ENVIRONMENTS[0][0]}.tsv").read_text()
    )
    with pytest.raises(
        module.GenerationError, match=r"unlisted-env\.tsv has no entry in ENVIRONMENTS"
    ):
        module.render(locks)


def domains_of_the_rules() -> dict[str, tuple[str, str]]:
    """rule id -> (domain, Families cell), read from the sections of docs/rules.md."""
    found, domain = {}, ""
    for line in committed("rules.md").splitlines():
        if line.startswith("## "):
            domain = line[3:]
        elif line.startswith("| [`"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            found[cells[0].split("`")[1]] = (domain, cells[6])
    return found


def packs_of_the_families() -> dict[int, str]:
    """family number -> pack, read from the sections of docs/families.md."""
    found, pack = {}, ""
    for line in committed("families.md").splitlines():
        if line.startswith("## "):
            pack = line[3:]
        elif match := re.match(r"\| F(\d+) \|", line):
            found[int(match[1])] = pack
    return found


def test_a_domain_with_a_tool_that_no_table_has_stops_the_generator(monkeypatch, capsys):
    module = load("make_supported_tools_doc")
    domains = dict(module.DOMAIN_TOOLS)
    domains[next(iter(domains))] += ", nosuchtool"
    monkeypatch.setattr(module, "DOMAIN_TOOLS", domains)
    with pytest.raises(module.GenerationError, match="'nosuchtool', which no table has"):
        module.render()
    assert module.main([]) == 2
    assert "nosuchtool" in capsys.readouterr().err


def test_a_domain_that_is_not_a_pack_of_the_families_page_stops_the_generator(monkeypatch):
    module = load("make_supported_tools_doc")
    domains = dict(module.DOMAIN_TOOLS)
    domains["Not a pack"] = domains.pop(next(iter(domains)))
    monkeypatch.setattr(module, "DOMAIN_TOOLS", domains)
    with pytest.raises(module.GenerationError, match="must have the same titles"):
        module.render()


def test_the_domain_table_counts_the_rows_of_the_rules_page_and_the_families_page():
    rows = table_rows(committed("supported-tools.md").split("## Versions")[0])
    *domains, total = [row for row in rows if len(row) == 4]
    assert total[0] == "Total"
    rules, packs = domains_of_the_rules(), packs_of_the_families()
    rules_per_domain = Counter(domain for domain, _ in rules.values())
    families_per_pack = Counter(packs.values())
    for title, _tools, n_rules, n_families in domains:
        assert int(n_rules) == rules_per_domain[title], title
        assert int(n_families) == families_per_pack[title], title
    assert {row[0] for row in domains} == set(rules_per_domain) == set(families_per_pack)
    assert (int(total[2]), int(total[3])) == (len(rules), len(packs))


def test_the_note_on_rules_counted_away_from_their_families_names_exactly_those_rules():
    """A rule is counted away when none of its families is in the pack of its domain."""
    rules, packs = domains_of_the_rules(), packs_of_the_families()
    apart = {}
    for rule_id, (domain, families) in rules.items():
        numbers = [int(n) for n in re.findall(r"F(\d+)", families)]
        if numbers and domain not in {packs[n] for n in numbers}:
            apart[rule_id] = sorted(f"F{n}" for n in numbers)
    assert apart, "the note is not tested"
    page = flat(committed("supported-tools.md"))
    note = page.partition("can therefore fall in different rows.")[2].partition(" Models such")[0]
    named = re.findall(r"`([\w.]+)` \(([F\d, ]+)\)", note)
    assert {rule_id: families.split(", ") for rule_id, families in named} == apart


def test_the_notes_the_generators_print_as_plain_text_still_hold():
    rules_doc = load("make_rules_doc")
    families_doc = load("make_families_doc")
    rows = rules_doc.collect()
    # rules.md: the rules of the last domain are fact rules about data formats, written for
    # families of the original pack
    pack = rules_doc.DATA_FORMAT_DOMAIN
    low, high = next((low, high) for title, low, high in families_doc.DOMAINS if title == pack)
    data_formats = [row for row in rows if row.domain == pack]
    assert data_formats
    for row in data_formats:
        assert row.rule.kind == "fact", row.rule.id
        numbers = [int(label.split(" ")[0][1:]) for label in row.families]
        assert numbers and all(low <= n <= high for n in numbers), row.rule.id
    # supported-tools.md: the one rule that watches a structure-prediction program
    predictors = [row.rule.id for row in rows if "mockfold" in row.rule.applies.tools]
    assert predictors == ["structure.chain_order"]
    assert (ROOT / "src" / "biowarden" / "bench" / "fixtures" / "mockfold.py").is_file()
