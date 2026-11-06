"""tools/preflight.py: the pin, credit, smoke and probe checks, and the schedule of a job.

No test calls an API: the HTTP GET and the trial runners are replaced by stand-ins.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from biowarden.bench import stage2

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-" + "secret-value-xyz"  # a made-up key: it must never be printed
DSFLASH, GPTOSS = "deepseek/deepseek-v4.1-flash", "openai/gpt-oss-120b"


@pytest.fixture(scope="module")
def pf():
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "tools" / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["preflight"] = module
    spec.loader.exec_module(module)
    return module


def sets_dict(primary=("F2", "F10"), q1=("F3",), q2=("F5",), q3=("F6",), excluded=()):
    families = [*primary, *q1, *q2, *q3]
    return {
        "format": 2,
        "primary": list(primary),
        "q1": list(q1),
        "q2": list(q2),
        "q3": list(q3),
        "excluded": list(excluded),
        "seeds": {f: list(stage2.registry_seeds(f)) for f in families},
        "default_seeds": {f: list(stage2.registry_seeds(f)[:3]) for f in [*families, "F4"]},
        "default_only": ["F4"],
    }


def args(**kw):
    return argparse.Namespace(snapshot=None, **kw)


# -- endpoints ---------------------------------------------------------------------------------


def endpoint(tag, params=("tools", "reasoning"), **more):
    return {"tag": tag, "provider_name": "P", "supported_parameters": list(params), **more}


def test_an_endpoint_with_tools_and_reasoning_passes(pf):
    payload = {
        "data": {"endpoints": [endpoint("other/x"), endpoint("wafer/us", quantization="fp8")]}
    }

    line, problems = pf.endpoint_report(DSFLASH, "wafer/us", payload)

    assert problems == []
    assert "ok" in line and "fp8" in line


def test_a_missing_tag_or_capability_fails(pf):
    payload = {"data": {"endpoints": [endpoint("other/x")]}}
    _, problems = pf.endpoint_report(DSFLASH, "wafer/us", payload)
    assert len(problems) == 1 and "no endpoint tagged 'wafer/us'" in problems[0]
    assert "other/x" in problems[0]  # says what the model does have

    no_tools = {"data": {"endpoints": [endpoint("wafer/us", params=("reasoning",))]}}
    assert pf.endpoint_report(DSFLASH, "wafer/us", no_tools)[1] == [
        f"{DSFLASH} wafer/us: no tool support"
    ]
    no_reasoning = {"data": {"endpoints": [endpoint("wafer/us", params=("tools", "max_tokens"))]}}
    assert pf.endpoint_report(DSFLASH, "wafer/us", no_reasoning)[1] == [
        f"{DSFLASH} wafer/us: no reasoning support"
    ]
    # include_reasoning is the older spelling of the same capability
    old = {"data": {"endpoints": [endpoint("wafer/us", params=("tools", "include_reasoning"))]}}
    assert pf.endpoint_report(DSFLASH, "wafer/us", old)[1] == []


def test_a_reply_that_is_not_an_endpoint_list_fails(pf):
    for payload in ({}, {"data": None}, {"data": {"endpoints": "x"}}, [], None):
        assert pf.endpoint_report(DSFLASH, "wafer/us", payload)[1], payload


def test_the_endpoints_command_checks_every_pinned_model_and_sets_the_status(pf, monkeypatch):
    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: None)
    seen = []

    def fetch(url, key=None):
        seen.append(url)
        model = url.split("/models/")[1].removesuffix("/endpoints")
        tag = {DSFLASH: "wafer/us", GPTOSS: "near-ai/fp4"}[model]
        return {"data": {"endpoints": [endpoint(tag)]}}

    assert pf.cmd_endpoints(args(models=None), fetch) == 0
    assert sorted(seen) == [
        f"https://openrouter.ai/api/v1/models/{DSFLASH}/endpoints",
        f"https://openrouter.ai/api/v1/models/{GPTOSS}/endpoints",
    ]
    assert pf.cmd_endpoints(args(models=["gptoss"]), lambda url, key=None: {"data": {}}) == 1
    with pytest.raises(pf.Refused, match="no provider pin for claude-haiku-5-5"):
        pf.cmd_endpoints(args(models=["haiku"]), fetch)


def test_the_endpoints_command_passes_the_key_on_without_printing_it(pf, monkeypatch, capsys):
    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: KEY)
    keys = []

    def fetch(url, key=None):
        keys.append(key)
        return {"data": {"endpoints": [endpoint("wafer/us")]}}

    assert pf.cmd_endpoints(args(models=["dsflash"]), fetch) == 0
    assert keys == [KEY] and KEY not in capsys.readouterr().out


def test_fetch_json_parses_and_turns_failures_into_refusals(pf):
    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    sent = {}

    def opener(request, timeout):
        sent["auth"] = request.get_header("Authorization")
        return Reply(b'{"data": {"total_credits": 5, "total_usage": 1.5}}')

    assert pf.fetch_json("https://x.test/credits", "k", opener=opener)["data"]["total_credits"] == 5
    assert sent["auth"] == "Bearer k"
    pf.fetch_json("https://x.test/models", opener=opener)
    assert sent["auth"] is None  # the public listing is fetched without a key

    for broken in (lambda r, timeout: Reply(b"<html>"), lambda r, timeout: Reply(b"")):
        with pytest.raises(pf.Refused, match="GET https://x.test"):
            pf.fetch_json("https://x.test/credits", opener=broken)

    def down(request, timeout):
        raise urllib.error.URLError("boom")

    with pytest.raises(pf.Refused, match="boom"):
        pf.fetch_json("https://x.test/credits", opener=down)


# -- credits -----------------------------------------------------------------------------------


def test_remaining_credit_is_total_minus_usage(pf):
    assert pf.remaining_credit(
        {"data": {"total_credits": "30.0", "total_usage": 8.37}}
    ) == pytest.approx(21.63)
    for bad in ({}, {"data": {"total_credits": 1}}, {"data": None}, None):
        with pytest.raises(pf.Refused, match="unexpected reply"):
            pf.remaining_credit(bad)


def test_credits_gate_compares_the_balance_with_the_need(pf, monkeypatch, capsys):
    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: KEY)
    calls = []

    def fetch(url, key=None):
        calls.append((url, key))
        return {"data": {"total_credits": 30, "total_usage": 26}}

    assert pf.cmd_credits(args(model="dsflash", need=3.0), fetch) == 0
    assert pf.cmd_credits(args(model="dsflash", need=5.0), fetch) == 1
    assert calls[0] == ("https://openrouter.ai/api/v1/credits", KEY)
    out = capsys.readouterr().out
    assert "$4.00 left, $3.00 needed: ok" in out and "NOT ENOUGH" in out
    assert KEY not in out and "secret-value" not in out  # a key is never printed


def test_credits_gate_has_nothing_to_check_for_anthropic_and_needs_a_key(pf, monkeypatch, capsys):
    assert pf.cmd_credits(args(model="haiku", need=9.0), lambda *a, **k: 1 / 0) == 0
    assert "no balance endpoint" in capsys.readouterr().out
    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: None)
    with pytest.raises(pf.Refused, match="no OpenRouter key"):
        pf.cmd_credits(args(model="gptoss", need=1.0), lambda *a, **k: 1 / 0)


# -- smoke test ---------------------------------------------------------------------------------


def record(**kw):
    base = {
        "outcome": "pass",
        "turns": 4,
        "usd": 0.002,
        "providers": ["Wafer"],
        "pin_match": True,
        "tokens": {"input_tokens": 900, "output_tokens": 300, "reasoning_tokens": 120},
    }
    return base | kw


def test_a_clean_smoke_run_has_no_problems(pf):
    assert pf.smoke_problems(DSFLASH, [record() for _ in range(20)], 20) == []


def test_smoke_problems_name_what_is_wrong(pf):
    rows = [record() for _ in range(20)]
    assert pf.smoke_problems(DSFLASH, rows[:17], 20) == ["only 17 of 20 trials left a record"]

    two = [*rows[:19], record(providers=["Other"])]
    assert "served by ['Other', 'Wafer'], expected exactly one" in pf.smoke_problems(
        DSFLASH, two, 20
    )
    off = [*rows[:19], record(pin_match=False)]
    assert any("pin_match false" in p for p in pf.smoke_problems(DSFLASH, off, 20))
    nameless = [record(providers=[]) for _ in range(20)]
    assert any("no named provider" in p for p in pf.smoke_problems(DSFLASH, nameless, 20))

    free = [*rows[:19], record(usd=0.0)]
    assert pf.smoke_problems(DSFLASH, free, 20) == ["1 trials answered but cost usd = 0"]
    silent = [record(tokens={"input_tokens": 9, "output_tokens": 1}) for _ in range(20)]
    assert pf.smoke_problems(DSFLASH, silent, 20) == ["no reasoning tokens in any reply"]
    failed = [*rows[:19], record(outcome="infra_error", reason="Timeout: slow", turns=0, usd=0)]
    assert any(
        "1 infra_error trials, first: Timeout: slow" in p
        for p in pf.smoke_problems(DSFLASH, failed, 20)
    )


def test_anthropic_smoke_checks_cost_only(pf):
    rows = [record(providers=[], pin_match=None, tokens={"input_tokens": 9}) for _ in range(20)]
    assert pf.smoke_problems("claude-haiku-5-5", rows, 20) == []
    rows[3]["usd"] = 0
    assert pf.smoke_problems("claude-haiku-5-5", rows, 20) == ["1 trials answered but cost usd = 0"]


def write_records(out: Path, rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        folder = out / "trials" / "m" / f"t{i:03d}"
        folder.mkdir(parents=True)
        (folder / "score.json").write_text(json.dumps(row))


def test_the_smoke_command_runs_the_runner_pinned_and_judges_its_records(
    pf, tmp_path, capsys, monkeypatch
):
    calls = []

    def runner(argv):
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        write_records(out, [record() for _ in range(20)])
        return 0

    ns = args(model="dsflash", n=20, workers=4, max_usd=1.0, out=str(tmp_path), env_locks="locks")
    assert pf.cmd_smoke(ns, runner) == 0
    argv = calls[0]
    assert argv[1:4] == ["-m", "biowarden.bench.runner.cli", "run"]
    assert argv[argv.index("--models") + 1] == DSFLASH
    assert ("--limit", "20") == (argv[argv.index("--limit")], argv[argv.index("--limit") + 1])
    assert "--no-pin" not in argv  # the pin is what is being tested
    from biowarden.bench.runner import cli

    parsed = []
    monkeypatch.setattr(cli, "cmd_run", lambda a: parsed.append(a) or 0)
    assert cli.main(argv[3:]) == 0  # the real command line accepts every flag it was given
    assert parsed[0].limit == 20 and parsed[0].families[:2] == ["F1", "F2"]
    assert argv[argv.index("--arm") + 1] == "bare"
    assert "providers ['Wafer']" in capsys.readouterr().out

    # a runner that leaves nothing behind is a failed smoke test, whatever it returns
    ns = args(model="gptoss", n=20, workers=4, max_usd=1.0, out=str(tmp_path / "b"), env_locks="l")
    assert pf.cmd_smoke(ns, lambda argv: 3) == 1


def test_a_smoke_or_probe_directory_that_already_has_trials_is_refused(pf, tmp_path):
    used = tmp_path / "smoke" / DSFLASH.replace("/", "__") / "trials"
    used.mkdir(parents=True)
    ns = args(model="dsflash", n=20, workers=4, max_usd=1.0, out=str(tmp_path), env_locks="l")
    with pytest.raises(pf.Refused, match="already holds trials; use a fresh --out"):
        pf.cmd_smoke(ns, lambda argv: 1 / 0)
    (tmp_path / "probe" / DSFLASH.replace("/", "__")).mkdir(parents=True)
    (tmp_path / "probe" / DSFLASH.replace("/", "__") / "trials").mkdir()
    ns = args(model="dsflash", n=40, workers=8, max_usd=2.0, max_share=0.05, out=str(tmp_path))
    with pytest.raises(pf.Refused, match="already holds trials"):
        pf.cmd_probe(ns, lambda *a: 1 / 0)


# -- probe --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, kind",
    [
        ("RateLimitError: Error code: 429 - {'error': 'slow down'}", "429"),
        ("InternalServerError: Error code: 503 - upstream", "5xx"),
        ("APIStatusError: status 502 bad gateway", "5xx"),
        ("APITimeoutError: Request timed out.", "timeout"),
        ("APIConnectionError: Connection error.", "connection"),
        ("BadRequestError: Error code: 400 - max_tokens 500 is too large", "other"),
        ("unpriced: 4 tokens, cost 0.0", "other"),
    ],
)
def test_status_kind_reads_the_status_not_a_stray_number(pf, text, kind):
    assert pf.status_kind(text) == kind


def test_probe_summary_counts_failed_requests_against_all_requests(pf):
    ok = [record(turns=5) for _ in range(30)]  # 150 requests
    limited = [
        record(outcome="infra_error", stop="api_error", turns=2, reason="Error code: 429")
        for _ in range(4)
    ]
    down = [record(outcome="infra_error", stop="api_error", turns=0, error="Connection error.")]
    other = [record(outcome="infra_error", stop="crashed", turns=0, reason="crashed: X")]

    summary = pf.probe_summary(ok + limited + down + other)

    assert summary["trials"] == 36
    assert summary["failed"] == 5  # a crash of the trial is not an API failure
    assert summary["requests"] == 150 + 4 * 2 + 5
    assert summary["kinds"] == {"429": 4, "connection": 1}
    assert summary["share"] == pytest.approx(4 / 163)
    assert pf.probe_summary([])["trials"] == 0


@pytest.mark.parametrize(
    "model, pinned",
    [(DSFLASH, ["wafer/us"]), (GPTOSS, ["near-ai/fp4"]), ("claude-haiku-5-5", None)],
)
def test_the_real_probe_runner_has_no_retries_no_breaker_and_the_pin(
    pf, tmp_path, monkeypatch, model, pinned
):
    from biowarden.bench.runner import trial
    from biowarden.bench.tasks.tools import ToolEnv

    seen = {}

    def run_trial(spec, root, clients, tools, cfg, budget, info):
        seen.update(spec=spec, clients=clients, cfg=cfg, budget=budget, info=info)
        return {}

    monkeypatch.setattr(trial, "run_trial", run_trial)
    monkeypatch.setattr(ToolEnv, "discover", classmethod(lambda cls, root=None: cls(bin_dirs=())))
    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: KEY)

    run = pf.probe_runner(model, tmp_path, 2.5)
    run("a spec")

    cfg, clients = seen["cfg"], seen["clients"]
    assert (cfg.max_retries, cfg.breaker.limit) == (0, 0)  # every failed request shows, none trips
    assert (cfg.provider or {}).get("order") == pinned
    assert seen["budget"].max_usd == 2.5 and seen["info"] is None and seen["spec"] == "a spec"
    client = clients.openrouter if "/" in model else clients.anthropic
    assert client.max_retries == 0
    assert (clients.anthropic is None) == ("/" in model)


def test_the_real_probe_runner_needs_a_key_and_the_tools(pf, tmp_path, monkeypatch):
    from biowarden.bench.tasks.tools import ToolEnv

    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: None)
    with pytest.raises(pf.Refused, match="no OpenRouter key"):
        pf.probe_runner(DSFLASH, tmp_path, 1.0)
    with pytest.raises(pf.Refused, match="no Anthropic key"):
        pf.probe_runner("claude-haiku-5-5", tmp_path, 1.0)
    monkeypatch.setattr("biowarden.keys.get_key", lambda provider: KEY)
    monkeypatch.setattr(ToolEnv, "discover", classmethod(lambda cls, root=None: None))
    with pytest.raises(pf.Refused, match="samtools, bcftools and bedtools not found"):
        pf.probe_runner(DSFLASH, tmp_path, 1.0)


def test_the_probe_runs_without_retries_at_the_planned_worker_count(pf, tmp_path, capsys):
    seen = {}

    def make_runner(model, out, max_usd):
        seen["args"] = (model, out, max_usd)

        def run(spec):
            folder = out / "trials" / "m" / f"{spec.family}-{spec.twin}"
            folder.mkdir(parents=True, exist_ok=True)
            bad = spec.family == "F1" and spec.twin == "clean"
            row = record(turns=3, outcome="infra_error", stop="api_error", reason="Error code: 429")
            (folder / "score.json").write_text(json.dumps(row if bad else record(turns=3)))

        return run

    ns = args(model="dsflash", n=40, workers=8, max_usd=2.0, max_share=0.05, out=str(tmp_path))
    assert pf.cmd_probe(ns, make_runner) == 0
    assert seen["args"][0] == DSFLASH and seen["args"][2] == 2.0
    out = capsys.readouterr().out
    assert "8 workers, max_retries=0" in out and "429/5xx share" in out


def test_the_probe_fails_above_the_error_share(pf, tmp_path):
    def make_runner(model, out, max_usd):
        def run(spec):
            folder = out / "trials" / "m" / f"{spec.family}-{spec.twin}"
            folder.mkdir(parents=True, exist_ok=True)
            row = record(turns=1, outcome="infra_error", stop="api_error", reason="Error code: 503")
            (folder / "score.json").write_text(json.dumps(row))

        return run

    ns = args(model="gptoss", n=40, workers=2, max_usd=2.0, max_share=0.05, out=str(tmp_path))
    assert pf.cmd_probe(ns, make_runner) == 1


# -- the snapshot -------------------------------------------------------------------------------


def test_the_snapshot_check_accepts_the_imported_package_and_refuses_another_place(pf, tmp_path):
    import biowarden

    package_root = Path(biowarden.__file__).resolve().parents[1]
    pf.check_snapshot(str(package_root))  # the directory it is imported from
    with pytest.raises(pf.Refused, match="not from the snapshot"):
        pf.check_snapshot(str(tmp_path))


def test_main_turns_a_refusal_into_status_2(pf, tmp_path, capsys):
    status = pf.main(["chunks", "--cell", "primary", "--sets", str(tmp_path / "none.json")])
    assert status == 2
    assert "preflight: --sets" in capsys.readouterr().err
    assert pf.main(["chunks", "--cell", "primary"]) == 2  # no sets file given
    with pytest.raises(SystemExit) as exc:
        pf.main(["blocks", "--cell", "primary", "--arm", "nope", "--chunk", "0"])
    assert exc.value.code == 2


# -- the schedule ---------------------------------------------------------------------------------


def test_blocks_group_the_families_by_seed_block_and_cut_the_seeds_into_chunks(pf):
    sets = sets_dict()

    first = pf.blocks("primary", "bare", 0, 2, sets)
    assert [(b.seeds, b.families) for b in first] == [
        ("1000,1001", ["F2", "F3", "F5", "F6"]),
        ("1020,1021", ["F10"]),
    ]
    assert all(b[:5] == ("bare", "implicit", "forced", "bare", 1) for b in first)
    assert [b.seeds for b in pf.blocks("primary", "bare", 2, 2, sets)] == ["1004", "1024"]
    assert pf.blocks("primary", "bare", 3, 2, sets) == []
    assert pf.chunk_count("primary", 2, sets) == 3 and pf.chunk_count("primary", 1, sets) == 5
    assert pf.chunk_count("primary", 5, sets) == 1 and pf.chunk_count("primary", 9, sets) == 1


def test_the_layer_arms_and_the_replicate_run_on_their_own_families(pf):
    sets = sets_dict()

    warn = pf.blocks("primary", "warn-nonote", 0, 1, sets)
    assert [(b.arm, b.families) for b in warn] == [
        ("warn-nonote", ["F2", "F3"]),  # P and Q1: the bite families
        ("warn-nonote", ["F10"]),
    ]
    block = pf.blocks("primary", "block-nonote", 0, 1, sets)
    assert block[0].families == ["F2", "F3", "F5", "F6"]
    aa = pf.blocks("primary", "aa", 0, 1, sets)
    assert [(b.arm, b.reps, b.families) for b in aa] == [("bare", 2, ["F2"]), ("bare", 2, ["F10"])]
    assert pf.blocks("default", "aa", 0, 1, sets) == []  # the replicate is a primary-cell thing
    assert pf.blocks("default", "warn-nonote", 0, 1, sets) == []  # and so is warn-nonote


def test_the_default_cell_has_three_seeds_and_f4(pf):
    sets = sets_dict()

    chunks = pf.chunk_count("default", 1, sets)
    seen = {b.seeds for c in range(chunks) for b in pf.blocks("default", "bare", c, 1, sets)}

    assert chunks == 3
    assert seen == {"1000", "1001", "1002", "1020", "1021", "1022"}
    fresh = pf.blocks("default", "bare", 0, 1, sets)[0]
    assert fresh[:3] == ("full", "explicit", "free")
    assert "F4" in fresh.families


def test_an_excluded_family_is_in_no_block(pf):
    sets = sets_dict(primary=("F2", "F10"), q3=("F6",), excluded=("F34",))
    every = {
        f
        for c in range(5)
        for a in ("bare", "block-nonote")
        for b in pf.blocks("primary", a, c, 1, sets)
        for f in b.families
    }
    assert "F34" not in every and every == {"F2", "F3", "F5", "F6", "F10"}


def test_the_pilot_runs_every_family_on_seeds_0_to_4_in_one_call(pf):
    only = pf.blocks("pilot", "bare", 0, 1, None)

    assert [b.line() for b in only] == ["bare\timplicit\tforced\tbare\t1\t0-4\t-"]
    assert pf.blocks("pilot", "block-nonote", 0, 1, None) == []
    assert pf.blocks("pilot", "bare", 1, 1, None) == []  # chunk 0 is the only one
    assert pf.chunk_count("pilot", 1, None) == 1
    assert pf.load_sets(None, "pilot") is None
    with pytest.raises(pf.Refused, match="--sets is needed"):
        pf.load_sets(None, "primary")


def test_blocks_lines_are_tab_separated_with_space_separated_families(pf):
    block = pf.Block("bare", "implicit", "forced", "block-nonote", 1, "1000,1001", ["F2", "F3"])
    assert block.line() == "bare\timplicit\tforced\tblock-nonote\t1\t1000,1001\tF2 F3"


# -- plan manifest and verify ----------------------------------------------------------------------


def test_the_plan_manifest_has_one_digest_per_model_arm_and_cell(pf, tmp_path):
    raw = json.dumps(sets_dict()).encode()
    sets = json.loads(raw)

    manifest = pf.plan_manifest(sets, raw)
    again = pf.plan_manifest(sets, raw)

    assert manifest == again  # deterministic
    entries = {(e["model"], e["arm"], e["cell"]): e for e in manifest["entries"]}
    assert len(entries) == 15 and manifest["total"] == sum(e["n"] for e in entries.values())
    haiku = "claude-haiku-5-5"
    assert entries[(haiku, "bare", "bare/implicit/forced")]["n"] == 5 * 2 * 5  # 5 families, 5 seeds
    assert entries[(haiku, "warn-nonote", "bare/implicit/forced")]["n"] == 3 * 2 * 5
    assert entries[(haiku, "bare", "full/explicit/free")]["n"] == 6 * 2 * 3  # with F4
    other = json.loads(raw)
    other["q3"] = []
    changed = pf.plan_manifest(other, json.dumps(other).encode())
    assert changed["sets_sha256"] != manifest["sets_sha256"]
    assert changed["entries"][0]["sha256"] != manifest["entries"][0]["sha256"]


def test_verify_job_compares_the_plan_with_plan_json_and_the_records(pf, tmp_path):
    wanted = {"bare": ["m/a", "m/b", "m/c", "m/d", "m/e"], "aa": ["m/r"]}
    out = tmp_path
    (out / "plan.json").write_text(json.dumps({"ids": ["m/a", "m/b", "m/c", "m/d", "m/r"]}))

    def put(trial, **row):
        folder = out / "trials" / trial
        folder.mkdir(parents=True)
        (folder / "score.json").write_text(row.pop("raw", None) or json.dumps(row))

    put("m/a", outcome="pass", attempt=1)
    put("m/b", outcome="infra_error", attempt=3)  # tried three times: final, excluded
    put("m/c", outcome="infra_error", attempt=2)  # a try left
    put("m/d", raw='{"outcome": "pa')  # truncated
    # m/e is in no plan.json and m/r has no record

    rows, problems = pf.verify_job(out, wanted)

    assert problems == [
        "bare: 1 infra_error, attempts left",
        "bare: 1 unreadable",
        "bare: 1 not in plan.json",
        "aa: 1 absent",
    ]
    assert len(rows) == 2 and rows[0].startswith("bare")
    put("m/r", outcome="pass")
    (out / "trials" / "m" / "d" / "score.json").write_text('{"outcome": "pass"}')
    (out / "trials" / "m" / "c" / "score.json").write_text('{"outcome": "pass"}')
    (out / "plan.json").write_text(json.dumps({"ids": [*wanted["bare"], "m/r"]}))
    put("m/e", outcome="pass")
    assert pf.verify_job(out, wanted)[1] == []
    assert pf.verify_job(tmp_path / "nowhere", wanted)[1] == [
        f"{tmp_path / 'nowhere'}/plan.json is missing or unreadable"
    ]


def test_an_empty_plan_is_not_a_finished_job(pf, tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"ids": []}))

    assert pf.verify_job(tmp_path, {"bare": [], "block-nonote": []})[1] == [
        "the frozen plan has no trial for this job"
    ]
