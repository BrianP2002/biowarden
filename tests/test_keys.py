"""Key discovery: precedence, no value ever printed, and the repo ignores .env."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from biowarden import keys

SECRET = "sk-test-0123456789-not-a-real-key"
ANTHROPIC = "BIOWARDEN_ANTHROPIC_API_KEY"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def write_env(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text)
    path.chmod(mode)
    return path


def test_parse_dotenv_handles_export_quotes_comments_and_junk():
    text = f"""
# a comment
export {ANTHROPIC}="{SECRET}"
OTHER='quoted value'
PLAIN=abc # trailing comment
not a valid line
=novalue
EMPTY=
"""
    assert keys.parse_dotenv(text) == {
        ANTHROPIC: SECRET,
        "OTHER": "quoted value",
        "PLAIN": "abc",
        "EMPTY": "",
    }


def test_get_key_order_is_environment_then_files_and_prefixed_name_first(tmp_path):
    first = write_env(tmp_path / "first.env", f"{ANTHROPIC}=from-first\nANTHROPIC_API_KEY=bare\n")
    second = write_env(tmp_path / "second.env", f"{ANTHROPIC}=from-second\n")

    assert keys.get_key("anthropic", environ={}, files=[first, second]) == "from-first"
    assert keys.get_key("anthropic", environ={}, files=[second, first]) == "from-second"
    assert keys.get_key("anthropic", environ={ANTHROPIC: "from-env"}, files=[first]) == "from-env"
    bare_only = write_env(tmp_path / "bare.env", "ANTHROPIC_API_KEY=bare\n")
    assert keys.get_key("anthropic", environ={}, files=[bare_only]) == "bare"


def test_get_key_treats_an_empty_value_and_a_missing_file_as_unset(tmp_path):
    placeholder = write_env(tmp_path / "copied-from-example.env", f"{ANTHROPIC}=\n")

    assert keys.get_key("anthropic", environ={}, files=[placeholder, tmp_path / "missing"]) is None
    assert keys.get_key("deepseek", environ={}, files=[placeholder]) is None


def test_env_files_search_explicit_then_project_then_home(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    found = keys.env_files(project / "sub", {keys.ENV_FILE_VAR: str(tmp_path / "x.env")})

    assert found == [
        tmp_path / "x.env",
        project.resolve() / ".env",
        tmp_path / "home" / ".config" / "biowarden" / "keys.env",
    ]


def test_audit_and_cli_report_names_and_sources_but_never_the_value(tmp_path, capsys):
    env = write_env(tmp_path / "keys.env", f"{ANTHROPIC}={SECRET}\n")

    statuses, files = keys.audit(environ={}, files=[env])
    code = keys.main(["--env-file", str(env)])
    out = capsys.readouterr().out

    assert SECRET not in repr(statuses) + repr(files) + out
    assert [s.provider for s in statuses if s.variable] == ["anthropic"]
    assert f"{ANTHROPIC}  {env}" in out
    assert "mode 600" in out
    assert code in (0, 1)  # git state of tmp_path is not what this test is about


def test_problems_flag_a_file_open_to_group_and_other(tmp_path):
    loose = write_env(tmp_path / "loose.env", f"{ANTHROPIC}={SECRET}\n", mode=0o644)
    tight = write_env(tmp_path / "tight.env", f"{ANTHROPIC}={SECRET}\n", mode=0o600)

    _, files = keys.audit(environ={}, files=[loose, tight])
    found = keys.problems(files, environ={})

    assert any("loose.env" in p and "644" in p for p in found)
    assert not any("tight.env" in p for p in found)
    assert SECRET not in "\n".join(found)


def test_problems_flag_the_bare_anthropic_name_in_a_file_and_in_the_environment(tmp_path):
    bare = write_env(tmp_path / "bare.env", f"ANTHROPIC_API_KEY={SECRET}\n")

    _, files = keys.audit(environ={}, files=[bare])
    found = keys.problems(files, environ={"ANTHROPIC_API_KEY": SECRET})

    assert any("bare.env" in p and "BIOWARDEN_ANTHROPIC_API_KEY" in p for p in found)
    assert any("exported" in p and "bill the API" in p for p in found)
    assert SECRET not in "\n".join(found)


@needs_git
def test_problems_flag_a_dotenv_that_git_would_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    env = write_env(repo / ".env", f"{ANTHROPIC}={SECRET}\n")

    _, files = keys.audit(environ={}, files=[env])
    assert any("not ignored by git" in p for p in keys.problems(files, environ={}))

    (repo / ".gitignore").write_text(".env\n")
    _, files = keys.audit(environ={}, files=[env])
    assert keys.problems(files, environ={}) == []


@needs_git
def test_this_repo_ignores_dotenv_files_but_keeps_the_example():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")

    def ignored(name: str) -> bool:
        proc = subprocess.run(["git", "check-ignore", "-q", name], cwd=root, check=False)
        return proc.returncode == 0

    assert ignored(".env") and ignored(".env.local") and ignored("docs/research/x.md")
    assert not ignored(".env.example")
