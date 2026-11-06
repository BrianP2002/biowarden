"""The pilot runner: sandbox isolation, the tool loop (scripted client, no API calls), resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from biowarden.bench.runner import sandbox
from biowarden.bench.runner.agent import AgentConfig, run_agent, system_prompt
from biowarden.bench.runner.agent_openai import provider_routing, run_agent_openai
from biowarden.bench.runner.cli import parse_seeds, plan, report, rescore
from biowarden.bench.runner.exposure import commands_of, exposure
from biowarden.bench.runner.trial import Budget, Clients, TrialSpec, run_trial
from biowarden.bench.tasks import ToolEnv, build_task

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-haiku-5-5"
SECRET = "sk-test-DO-NOT-LEAK-0123456789"

# the sandbox binds /usr and its scripted commands run the host's python3.12
needs_python312 = pytest.mark.skipif(
    not os.access("/usr/bin/python3.12", os.X_OK),
    reason="/usr/bin/python3.12 is missing (the sandbox runs it)",
)


@pytest.fixture(scope="module")
def tools():
    env = ToolEnv.discover(ROOT)
    if env is None or not sandbox.available():
        pytest.skip("needs samtools, bcftools, bedtools and bubblewrap")
    return env


@pytest.fixture
def work(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return path


# --- sandbox -------------------------------------------------------------------------------------


@needs_python312
def test_the_sandbox_runs_the_real_tools_and_both_pythons(tools, work):
    out = sandbox.run_command(
        work,
        tools,
        "samtools --version | head -1; bcftools --version | head -1; bedtools --version; "
        "python3.12 -c 'print(6*7)'",
    )

    # the programs of this machine, whatever their version (CI installs the distribution's)
    banner = {
        name: tools.run([name, "--version"], work).stdout.splitlines()[0]
        for name in ("samtools", "bcftools")
    }
    assert out.exit_code == 0
    assert banner["samtools"] in out.stdout and banner["bcftools"] in out.stdout
    assert "42" in out.stdout
    assert "no version information" not in out.stderr


def test_the_sandbox_hides_the_project_the_home_directories_and_the_network(
    tools, work, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("BIOWARDEN_ANTHROPIC_API_KEY", SECRET)
    env_file = ROOT / ".env"
    probe = (
        f"cat {env_file} 2>&1; ls /gpfs /home /projects 2>&1; pwd; env; "
        "(getent hosts api.anthropic.com || echo no-dns) 2>&1; touch /usr/x 2>&1; echo ok > /work/w"
    )

    out = sandbox.run_command(work, tools, probe)

    text = out.stdout + out.stderr
    assert SECRET not in text and "ANTHROPIC" not in text
    assert "No such file or directory" in text and "no-dns" in text
    assert "Read-only file system" in text
    assert "/work" in out.stdout.splitlines()  # pwd
    assert (work / "w").read_text() == "ok\n"


def test_a_symlink_to_a_host_file_does_not_resolve_inside_the_sandbox(tools, work, tmp_path):
    secret_file = tmp_path / "outside.txt"
    secret_file.write_text(SECRET)
    (work / "link").symlink_to(secret_file)

    out = sandbox.run_command(work, tools, "cat link; ls -l link")

    assert SECRET not in out.stdout + out.stderr


def test_timeouts_and_huge_outputs_are_contained(tools, work):
    slow = sandbox.run_command(work, tools, "sleep 30", timeout=1)
    big = sandbox.run_command(work, tools, "yes x | head -c 200000", max_chars=1000)

    assert slow.timed_out and "timed out after 1s" in slow.render()
    assert big.truncated and len(big.stdout) < 1200 and "characters omitted" in big.stdout


def test_render_shows_exit_code_stdout_and_stderr(tools, work):
    out = sandbox.run_command(work, tools, "echo hi; echo oops >&2; exit 3")

    assert out.render() == "[exit code 3]\n--- stdout ---\nhi\n--- stderr ---\noops"


# --- the tool loop with a scripted client ---------------------------------------------------------


def block(kind, **fields):
    return SimpleNamespace(
        type=kind, model_dump=lambda mode="json": {"type": kind, **fields}, **fields
    )


def response(stop, *blocks, in_tokens=1000, out_tokens=200, category=None):
    usage = SimpleNamespace(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    details = SimpleNamespace(category=category) if category else None
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, usage=usage, model=MODEL, stop_details=details
    )


def bash_call(command, call_id="toolu_1"):
    return block("tool_use", id=call_id, name="bash", input={"command": command})


class ScriptedClient:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs | {"messages": list(kwargs["messages"])})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def write_json_command(payload):
    return "python3.12 -c " + json.dumps(
        f"import json; json.dump({payload!r}, open('result.json','w'))"
    )


def solved(truth_answer, flags=()):
    return {"status": "ok", "answer": truth_answer, "flags": list(flags), "method": "scripted"}


@needs_python312
def test_a_scripted_agent_that_writes_the_right_answer_passes_and_the_trial_is_resumable(
    tools, tmp_path
):
    spec = TrialSpec(MODEL, "F1", 0, "faulted")
    truth = build_task("F1", 0, "faulted", tmp_path / "peek", tools).truth["expected"]
    client = ScriptedClient(
        response(
            "tool_use",
            block("thinking", thinking="", signature="sig"),
            bash_call(write_json_command(solved(truth))),
        ),
        response("end_turn", block("text", text="done")),
    )
    cfg = AgentConfig(MODEL)

    record = run_trial(spec, tmp_path / "run", client, tools, cfg, Budget(5))

    assert record["outcome"] == "pass" and record["turns"] == 2 and record["commands"] == 1
    trial = tmp_path / "run" / "trials" / spec.trial_id
    assert {p.name for p in trial.iterdir()} == {
        "task.json",
        "transcript.json",
        "score.json",
        "work",
    }
    assert (
        "expected" in (trial / "task.json").read_text()
    )  # the truth is stored, outside the sandbox
    assert "expected" not in "".join(
        p.read_text(errors="ignore") for p in (trial / "work").iterdir() if p.suffix == ".json"
    )
    assert (
        run_trial(spec, tmp_path / "run", ScriptedClient(), tools, cfg, Budget(5)) == record
    )  # no new API calls
    assert report(tmp_path / "run", out=open("/dev/null", "w"))[
        f"full/explicit/free|faulted|{MODEL}"
    ]["counts"] == {"pass": 1}


@needs_python312
def test_the_naive_bcftools_call_inside_the_sandbox_is_scored_silent_wrong(tools, tmp_path):
    command = (
        "bcftools view -H -R regions.tsv variants.vcf.gz | cut -f3 | python3.12 -c "
        '"import sys,json; ids=sorted(l.strip() for l in sys.stdin); '
        "json.dump({'status':'ok','answer':ids,'flags':[],'method':'bcftools -R'}, "
        "open('result.json','w'))\""
    )
    client = ScriptedClient(response("tool_use", bash_call(command)), response("end_turn"))

    record = run_trial(
        TrialSpec(MODEL, "F1", 1, "faulted"), tmp_path, client, tools, AgentConfig(MODEL), Budget(5)
    )

    assert record["outcome"] == "silent_wrong" and not record["flagged"]


def test_requests_carry_no_sampling_parameters_no_forced_tool_and_no_fallbacks(tools, work):
    client = ScriptedClient(response("tool_use", bash_call("echo hi")), response("end_turn"))

    run_agent(client, AgentConfig(MODEL, effort="medium"), "task", work, tools)

    first = client.calls[0]
    assert first["output_config"] == {"effort": "medium"} and first["cache_control"] == {
        "type": "ephemeral"
    }
    assert not {
        "temperature",
        "top_p",
        "top_k",
        "tool_choice",
        "fallbacks",
        "betas",
        "thinking",
    } & set(first)
    assert "Only result.json is read" in first["system"] and "faulted" not in first["system"]


def test_assistant_content_is_passed_back_unchanged_with_a_matching_tool_result(tools, work):
    thinking = block("thinking", thinking="", signature="sig")
    call = bash_call("echo hi", call_id="toolu_9")
    first = response("tool_use", thinking, call)
    client = ScriptedClient(first, response("end_turn"))

    run_agent(client, AgentConfig(MODEL), "task", work, tools)

    sent = client.calls[1]["messages"]
    assert sent[1] == {"role": "assistant", "content": first.content}
    assert sent[1]["content"][0] is thinking
    assert (
        sent[2]["content"][0]["tool_use_id"] == "toolu_9"
        and sent[2]["content"][0]["is_error"] is False
    )


def test_a_refusal_is_recorded_as_an_outcome_and_never_retried(tools, tmp_path):
    client = ScriptedClient(response("refusal", category="bio"))

    record = run_trial(
        TrialSpec(MODEL, "F2", 0, "clean"), tmp_path, client, tools, AgentConfig(MODEL), Budget(5)
    )

    assert (
        record["outcome"] == "refused"
        and record["refusal_category"] == "bio"
        and len(client.calls) == 1
    )


def test_api_errors_are_infrastructure_errors_not_model_failures(tools, tmp_path):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    client = ScriptedClient(error)

    record = run_trial(
        TrialSpec(MODEL, "F3", 0, "clean"), tmp_path, client, tools, AgentConfig(MODEL), Budget(5)
    )

    assert record["outcome"] == "infra_error" and record["stop"] == "api_error"


def test_turn_cap_and_spend_cap_end_a_trial_without_a_result(tools, tmp_path):
    loop = [response("tool_use", bash_call("echo hi", f"t{i}")) for i in range(5)]
    capped = run_trial(
        TrialSpec(MODEL, "F1", 2, "clean"),
        tmp_path / "a",
        ScriptedClient(*loop),
        tools,
        AgentConfig(MODEL, max_turns=3),
        Budget(5),
    )
    pricey = [
        response("tool_use", bash_call("echo hi", f"p{i}"), in_tokens=2_000_000) for i in range(3)
    ]
    broke = run_trial(
        TrialSpec(MODEL, "F1", 3, "clean"),
        tmp_path / "b",
        ScriptedClient(*pricey),
        tools,
        AgentConfig(MODEL, max_usd=0.05),
        Budget(5),
    )

    assert (
        capped["outcome"] == "protocol_fail"
        and capped["stop"] == "max_turns"
        and capped["turns"] == 3
    )
    assert broke["outcome"] == "protocol_fail" and broke["stop"] == "budget" and broke["turns"] == 1


def test_an_unknown_tool_gets_an_error_result_instead_of_crashing(tools, work):
    odd = block("tool_use", id="x1", name="python", input={"code": "1"})
    client = ScriptedClient(response("tool_use", odd), response("end_turn"))

    run_agent(client, AgentConfig(MODEL), "task", work, tools)

    result = client.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True and "unknown tool" in result["content"]


def test_an_exhausted_budget_skips_trials_without_calling_the_api(tools, tmp_path):
    budget = Budget(0.01)
    budget.charge(0.02)

    record = run_trial(
        TrialSpec(MODEL, "F1", 0, "clean"),
        tmp_path,
        ScriptedClient(),
        tools,
        AgentConfig(MODEL),
        budget,
    )

    assert record["outcome"] == "skipped_budget"
    assert not (tmp_path / "trials").exists() or not list((tmp_path / "trials").rglob("score.json"))


# --- planning ------------------------------------------------------------------------------------


def test_parse_seeds_and_plan_order():
    assert parse_seeds("0-2,5") == [0, 1, 2, 5]
    specs = list(plan(["m1", "m2"], ["F1", "F2"], [0, 1], reps=1))

    assert len(specs) == 2 * 2 * 2 * 2
    first = specs[0].family  # inside a seed the families come in a fixed pseudo-random order
    assert [(s.model, s.family, s.twin) for s in specs[:4]] == [
        ("m1", first, "clean"),
        ("m2", first, "clean"),
        ("m1", first, "faulted"),
        ("m2", first, "faulted"),
    ]
    assert {s.seed for s in specs[:8]} == {0}  # seed-major: a cut-short run stays balanced


@needs_python312
def test_rescore_applies_the_current_rules_to_finished_trials(tools, tmp_path):
    spec = TrialSpec(MODEL, "F1", 0, "faulted")
    truth = build_task("F1", 0, "faulted", tmp_path / "peek", tools).truth["expected"]
    client = ScriptedClient(
        response("tool_use", bash_call(write_json_command(solved(truth)))), response("end_turn")
    )
    record = run_trial(spec, tmp_path / "run", client, tools, AgentConfig(MODEL), Budget(5))
    done = tmp_path / "run" / "trials" / spec.trial_id / "score.json"
    done.write_text(json.dumps({**record, "outcome": "silent_wrong", "scorer": 1}))

    assert rescore(tmp_path / "run") == 1
    assert json.loads(done.read_text())["outcome"] == "pass"
    assert rescore(tmp_path / "run") == 0


def test_rescore_judges_exposure_on_the_commands_that_ran_not_the_ones_the_layer_refused(
    tmp_path,
):
    # regression for A-5: rescoring a layer-arm record used to count the refused command as
    # exposure (commands_of), flipping `exposed` in about half of the faulted block-nonote records
    from dataclasses import asdict

    from biowarden.bench import score as scoring

    spec = TrialSpec(MODEL, "F1", 0, "faulted", arm="block-nonote")
    trial = tmp_path / "run" / "trials" / spec.trial_id
    (trial / "work").mkdir(parents=True)
    (trial / "work" / "result.json").write_text(
        json.dumps({"status": "ok", "answer": 3, "flags": [], "method": "scripted"})
    )
    truth = {"kind": "int", "expected": 3, "naive": 5}
    (trial / "task.json").write_text(
        json.dumps({"family": "F1", "seed": 0, "truth": truth, "spec": asdict(spec)})
    )
    refused = "BLOCKED, the command did not run. FIX: declare the region semantics"
    transcript = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "bash",
                    "input": {"command": "bcftools view -H -R regions.tsv variants.vcf.gz"},
                },
                {"type": "tool_use", "id": "b", "name": "bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": refused},
                {"type": "tool_result", "tool_use_id": "b", "content": "[exit code 0]"},
            ],
        },
    ]
    (trial / "transcript.json").write_text(json.dumps(transcript))
    score = scoring.score_result(truth, {"status": "ok", "answer": 3, "flags": []}, "", "full")
    record = {
        "trial": spec.trial_id,
        **asdict(spec),
        "family": "F1",
        **score.to_json(),
        "exposed": False,  # what the runner wrote: the refused command did not run
        "exposed_issued": True,
        "n_commands": 1,
        "scorer": scoring.SCORER_VERSION,
        "stop": "end_turn",
    }
    (trial / "score.json").write_text(json.dumps(record))

    assert rescore(tmp_path / "run") == 0

    kept = json.loads((trial / "score.json").read_text())
    assert kept["exposed"] is False and kept["exposed_issued"] is True


def test_trial_ids_encode_only_the_knobs_that_differ_from_the_default():
    assert TrialSpec("m", "F1", 3, "faulted").trial_id == "m/F1-s03-faulted-r0"
    spec = TrialSpec("m", "F1", 3, "faulted", 1, "implicit", "forced", "bare")
    assert spec.trial_id == "m/F1-s03-faulted-implicit-forced-bare-r1"


def test_plan_schedules_the_must_detect_family_only_in_the_default_cell():
    default = list(plan(["m"], ["F1", "F4"], [0], 1))
    forced = list(plan(["m"], ["F1", "F4"], [0], 1, "implicit", "forced"))
    bare = list(plan(["m"], ["F1", "F4"], [0], 1, contract="bare"))

    assert {s.family for s in default} == {"F1", "F4"}
    assert {s.family for s in forced} == {"F1"} and {s.family for s in bare} == {"F1"}
    assert all(s.route == "forced" and s.tier == "implicit" for s in forced)
    assert all(s.contract == "bare" for s in bare)


def test_the_bare_system_prompt_offers_no_warning_channel_and_the_full_one_does():
    bare, full = system_prompt("bare"), system_prompt("full")

    assert "flags" not in bare and "needs_input" not in bare and '{"answer"' in bare
    assert "flags" in full and "needs_input" in full
    assert "Only result.json is read" in bare and "faulted" not in bare + full


def call(command):
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "name": "bash", "input": {"command": command}}],
    }


def test_exposure_f1_means_a_bcftools_region_file_option_was_executed(tmp_path):
    ran = [call("bcftools view -H -R regions.tsv variants.vcf.gz | cut -f3")]
    renamed = [call("cp regions.tsv r.bed && bcftools view -H -T r.bed variants.vcf.gz")]
    other = [
        call("bedtools intersect -a variants.vcf.gz -b regions.tsv -u"),
        call("awk '{print $1}' regions.tsv"),
    ]

    assert exposure("F1", commands_of(ran), tmp_path)["exposed"] is True
    assert exposure("F1", commands_of(renamed), tmp_path)["exposed"] is True  # the route was taken
    assert exposure("F1", commands_of(other), tmp_path)["exposed"] is False


def test_exposure_f2_means_faidx_was_run_on_the_region_strings(tmp_path):
    (tmp_path / "regions.txt").write_text("chr1:100-200\nchr2:50-90\n")
    pasted = [call("samtools faidx ref.fa chr1:100-200 chr2:50-90")]
    from_file = [call("samtools faidx ref.fa $(cat regions.txt)")]
    converted = [call("samtools faidx ref.fa chr1:101-200")]
    other = [call("bedtools getfasta -fi ref.fa -bed regions.bed")]

    assert exposure("F2", commands_of(pasted), tmp_path)["exposed"] is True
    assert exposure("F2", commands_of(from_file), tmp_path)["exposed"] is True
    assert exposure("F2", commands_of(converted), tmp_path)["exposed"] is False
    assert exposure("F2", commands_of(other), tmp_path)["exposed"] is False


def test_exposure_f3_means_a_region_tool_ran_on_the_original_regions_file(tmp_path):
    original = [
        call("cat regions.bed"),
        call("bedtools intersect -a variants.vcf.gz -b regions.bed -u | wc -l"),
    ]
    fixed = [
        call("cat regions.bed"),
        call(
            "sed 's/^/chr/' regions.bed > r2.bed; "
            "bedtools intersect -a variants.vcf.gz -b r2.bed -u"
        ),
    ]

    assert exposure("F3", commands_of(original), tmp_path)["exposed"] is True
    assert exposure("F3", commands_of(fixed), tmp_path)["exposed"] is False


def test_exposure_f4_is_true_whenever_the_reference_is_read(tmp_path):
    assert exposure("F4", commands_of([call("samtools faidx ref.fa chr1:1-10")]), tmp_path)[
        "exposed"
    ]
    assert not exposure("F4", commands_of([call("ls")]), tmp_path)["exposed"]


@needs_python312
def test_scripted_trial_records_exposure_route_and_contract(tools, tmp_path):
    spec = TrialSpec(MODEL, "F1", 0, "faulted", route="forced", contract="bare")
    truth = build_task("F1", 0, "faulted", tmp_path / "peek", tools, "explicit", "forced").truth
    client = ScriptedClient(
        response(
            "tool_use", bash_call("bcftools view -H -R regions.tsv variants.vcf.gz | head -1")
        ),
        response(
            "tool_use",
            bash_call(
                "python3.12 -c "
                + json.dumps(
                    f"import json; json.dump({{'answer': {truth['naive']!r}}}, "
                    "open('result.json','w'))"
                ),
                "toolu_2",
            ),
        ),
        response("end_turn"),
    )

    record = run_trial(spec, tmp_path / "run", client, tools, AgentConfig(MODEL), Budget(5))

    assert record["outcome"] == "silent_wrong" and record["signature"] is True
    assert (
        record["exposed"] is True and record["route"] == "forced" and record["contract"] == "bare"
    )
    assert "flags" not in client.calls[0]["system"] and spec.trial_id.endswith("-forced-bare-r0")


# --- the OpenAI-compatible loop (open-weight models through a gateway) --------------------


OPEN_MODEL = "deepseek/deepseek-v4.1-flash"


def oa_call(command, call_id="call_1", name="bash", raw=None):
    arguments = raw if raw is not None else json.dumps({"command": command})
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments)
    )


def oa_message(content=None, calls=()):
    dump = {"role": "assistant", "content": content}
    if calls:
        dump["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in calls
        ]
    return SimpleNamespace(
        content=content,
        tool_calls=list(calls),
        model_dump=lambda exclude_none=True: {k: v for k, v in dump.items() if v is not None},
    )


def oa_response(message, finish="stop", cost=0.001, provider="TestProvider"):
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=100, model_extra={"cost": cost})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=usage,
        model=OPEN_MODEL,
        model_extra={"provider": provider},
    )


class ScriptedOpenAI:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs | {"messages": list(kwargs["messages"])})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@needs_python312
def test_openai_loop_runs_commands_scores_and_records_cost_and_provider(tools, tmp_path):
    spec = TrialSpec(OPEN_MODEL, "F1", 0, "faulted")
    truth = build_task("F1", 0, "faulted", tmp_path / "peek", tools).truth["expected"]
    command = write_json_command(solved(truth))
    client = ScriptedOpenAI(
        oa_response(oa_message(calls=[oa_call(command)]), "tool_calls"),
        oa_response(oa_message("done"), "stop"),
    )

    record = run_trial(
        spec,
        tmp_path / "run",
        Clients(openrouter=client),
        tools,
        AgentConfig(OPEN_MODEL),
        Budget(5),
    )

    assert record["outcome"] == "pass" and record["turns"] == 2 and record["commands"] == 1
    assert record["usd"] == 0.002 and record["served_by"] == f"{OPEN_MODEL}@TestProvider"
    first = client.calls[0]
    assert (
        first["messages"][0]["role"] == "system" and first["tools"][0]["function"]["name"] == "bash"
    )
    assert first["extra_body"] == {"usage": {"include": True}, "reasoning": {"effort": "medium"}}
    tool_turn = client.calls[1]["messages"][-1]
    assert tool_turn["role"] == "tool" and tool_turn["tool_call_id"] == "call_1"
    assert spec.trial_id.startswith("deepseek__deepseek-v4.1-flash/")  # two directory levels only
    assert (tmp_path / "run" / "trials" / spec.trial_id / "score.json").exists()


@needs_python312
def test_a_pinned_provider_is_sent_with_every_request_and_recorded(tools, tmp_path):
    spec = TrialSpec(OPEN_MODEL, "F1", 0, "faulted")
    truth = build_task("F1", 0, "faulted", tmp_path / "peek", tools).truth["expected"]
    client = ScriptedOpenAI(
        oa_response(oa_message(calls=[oa_call(write_json_command(solved(truth)))]), "tool_calls"),
        oa_response(oa_message("done"), "stop", provider="Wafer"),
    )
    pin = provider_routing(OPEN_MODEL)
    assert pin == {"order": ["wafer/us"], "allow_fallbacks": False}

    record = run_trial(
        spec,
        tmp_path / "run",
        Clients(openrouter=client),
        tools,
        AgentConfig(OPEN_MODEL, provider=pin),
        Budget(5),
    )

    assert all(call["extra_body"]["provider"] == pin for call in client.calls)
    assert record["provider_pin"] == ["wafer/us"]
    assert record["providers"] == ["TestProvider", "Wafer"]  # a switch inside a trial is visible


def test_providers_are_pinned_only_for_models_that_have_a_pin():
    assert provider_routing("openai/gpt-oss-120b") == {
        "order": ["near-ai/fp4"],
        "allow_fallbacks": False,
    }
    assert provider_routing(OPEN_MODEL, pin=False) is None
    assert provider_routing("z-ai/glm-5.3-flash") is None


def test_invalid_tool_arguments_get_an_error_result_and_are_counted(tools, work):
    bad = oa_call("", raw='{"command": "ls"')  # truncated JSON, as a broken parser would emit
    client = ScriptedOpenAI(
        oa_response(oa_message(calls=[bad]), "tool_calls"),
        oa_response(oa_message(calls=[oa_call("echo hi", "call_2")]), "tool_calls"),
        oa_response(oa_message("ok"), "stop"),
    )

    traj = run_agent_openai(client, AgentConfig(OPEN_MODEL), "task", work, tools)

    assert traj.tool_arg_errors == 1 and traj.commands == 1 and traj.stop == "end_turn"
    error_turn = client.calls[1]["messages"][-1]
    assert error_turn["role"] == "tool" and "invalid tool arguments" in error_turn["content"]


def test_unknown_tools_length_and_content_filter_stops_are_handled(tools, work):
    odd = ScriptedOpenAI(
        oa_response(oa_message(calls=[oa_call("1", name="python")]), "tool_calls"),
        oa_response(oa_message("x"), "length"),
    )
    traj = run_agent_openai(odd, AgentConfig(OPEN_MODEL), "task", work, tools)
    assert traj.stop == "max_tokens" and traj.commands == 0
    assert "unknown tool" in odd.calls[1]["messages"][-1]["content"]

    filtered = ScriptedOpenAI(oa_response(oa_message(None), "content_filter"))
    traj = run_agent_openai(filtered, AgentConfig(OPEN_MODEL), "task", work, tools)
    assert traj.stop == "refusal" and traj.refusal_category == "content_filter"


def test_a_gateway_failure_is_an_infrastructure_error(tools, tmp_path):
    import openai

    error = openai.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    client = ScriptedOpenAI(error)

    record = run_trial(
        TrialSpec(OPEN_MODEL, "F2", 0, "clean"),
        tmp_path,
        Clients(openrouter=client),
        tools,
        AgentConfig(OPEN_MODEL),
        Budget(5),
    )

    assert record["outcome"] == "infra_error" and record["stop"] == "api_error"


def test_commands_are_recovered_from_both_transcript_formats():
    openai_style = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls -l"}'}}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "bash", "arguments": "{not json"}}],
        },
    ]
    anthropic_style = [call("pwd")]

    assert commands_of(openai_style) == ["ls -l"] and commands_of(anthropic_style) == ["pwd"]


def test_a_cached_infrastructure_error_is_retried_but_a_real_outcome_is_not(tools, tmp_path):
    spec = TrialSpec(MODEL, "F2", 0, "clean")
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    first = run_trial(spec, tmp_path, ScriptedClient(error), tools, AgentConfig(MODEL), Budget(5))
    assert first["outcome"] == "infra_error"

    retry = ScriptedClient(response("end_turn"))
    second = run_trial(spec, tmp_path, retry, tools, AgentConfig(MODEL), Budget(5))

    assert len(retry.calls) == 1 and second["outcome"] == "protocol_fail"  # it ran again
    assert (
        run_trial(spec, tmp_path, ScriptedClient(), tools, AgentConfig(MODEL), Budget(5)) == second
    )


def test_the_sandbox_sets_no_process_limit_that_other_trials_could_exhaust(tools, work):
    assert "-u" not in sandbox.LIMITS.split()
    out = sandbox.run_command(
        work, tools, "for i in 1 2 3 4 5 6; do true & done; wait; echo forked; ulimit -u"
    )

    assert out.exit_code == 0
    assert out.stdout.split() == ["forked", out.stdout.split()[-1]]
    assert out.stdout.split()[-1] != "256"


def test_refused_commands_are_not_exposure():
    blocked = "BLOCKED, the command did not run. FIX: x"
    anthropic = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "bash",
                    "input": {"command": "bcftools view -T r.bed v.vcf"},
                },
                {"type": "tool_use", "id": "b", "name": "bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": blocked},
                {"type": "tool_result", "tool_use_id": "b", "content": "[exit code 0]"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "c",
                    "name": "run_tool",
                    "input": {"tool": "bcftools", "args": ["view", "-R", "r.bed"]},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c", "content": "[exit code 0]"}],
        },
    ]
    openai = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "samtools depth a.bam"}',
                    },
                },
                {"id": "y", "function": {"name": "bash", "arguments": '{"command": "echo hi"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "x", "content": blocked},
        {"role": "tool", "tool_call_id": "y", "content": "[exit code 0]"},
    ]

    from biowarden.bench.runner.exposure import executed_commands_of

    assert executed_commands_of(anthropic) == ["ls", "bcftools view -R r.bed"]
    assert executed_commands_of(openai) == ["echo hi"]


def test_the_note_arm_adds_the_layer_sentence_and_nothing_else():
    from biowarden.bench.runner.agent import LAYER_NOTE, system_prompt
    from biowarden.bench.runner.layer import ARMS

    assert ARMS["note"] is None
    assert LAYER_NOTE in system_prompt("bare", True)
    assert LAYER_NOTE not in system_prompt("bare", False)
    assert system_prompt("bare", True).replace("\n\n" + LAYER_NOTE, "") == system_prompt(
        "bare", False
    )


def test_every_layer_arm_keeps_the_prompt_sentence_except_the_nonote_variants():
    from biowarden.bench.runner.layer import ARMS

    with_note = {a for a, spec in ARMS.items() if spec is not None and spec.note}
    without = {a for a, spec in ARMS.items() if spec is not None and not spec.note}

    assert without == {"warn-nonote", "block-nonote", "warn-generic", "block-generic"}
    assert {"warn", "block", "block-bypass", "block-gateway", "blockall"} <= with_note


def test_the_careful_arm_adds_one_generic_sentence_and_no_layer():
    from biowarden.bench.runner.agent import CAREFUL_NOTE, LAYER_NOTE, system_prompt
    from biowarden.bench.runner.layer import ARMS

    assert ARMS["careful"] is None
    text = system_prompt("bare", False, True)
    assert CAREFUL_NOTE in text and LAYER_NOTE not in text
    for hint in ("bcftools", "BED", "Phred", "log2", "strand", "declare", "convention"):
        assert hint not in CAREFUL_NOTE


def test_generic_arms_have_no_prompt_sentence_and_the_generic_style():
    from biowarden.bench.runner.layer import ARMS

    for arm in ("warn-generic", "block-generic"):
        assert ARMS[arm].style == "generic" and ARMS[arm].note is False


def test_a_broken_family_builder_costs_one_trial_not_the_run(tmp_path, monkeypatch):
    from biowarden.bench.runner import trial as trial_module
    from biowarden.bench.runner.agent import AgentConfig
    from biowarden.bench.runner.trial import Budget, TrialSpec, run_trial

    def boom(*args, **kwargs):
        raise FileNotFoundError("builder used a relative path")

    monkeypatch.setattr(trial_module, "build_task", boom)

    record = run_trial(
        TrialSpec("m", "F1", 0, "clean"), tmp_path, None, None, AgentConfig("m"), Budget(1)
    )

    assert record["outcome"] == "infra_error" and "build failed" in record["reason"]


def test_trial_directories_are_absolute_even_for_a_relative_output_root(tmp_path, monkeypatch):
    from biowarden.bench.runner import trial as trial_module
    from biowarden.bench.runner.agent import AgentConfig
    from biowarden.bench.runner.trial import Budget, TrialSpec, run_trial

    seen = []

    def spy(spec_family, seed, twin, work, *args):
        seen.append(work)
        raise RuntimeError("stop")

    monkeypatch.setattr(trial_module, "build_task", spy)
    monkeypatch.chdir(tmp_path)

    run_trial(
        TrialSpec("m", "F1", 0, "clean"), Path("runs/x"), None, None, AgentConfig("m"), Budget(1)
    )

    assert seen and seen[0].is_absolute()
