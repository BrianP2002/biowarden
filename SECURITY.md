# Security policy

## Supported versions

Research software: only the latest 0.0.x release receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting: open the **Security** tab of this repository
and choose **Report a vulnerability**. Give the version (`pip show biowarden`), the front end and the command or
file that triggers the problem. One person maintains the project; reports are answered on a best-effort basis.

## In scope

A guard that decides which shell commands run is security-relevant:

- a bypass: a call that a front end is documented to check ([front ends](docs/architecture.md#front-ends)) runs unchecked, or
  is allowed although a rule blocks it;
- path confinement: where a sandbox sits between the agent and the host (the benchmark runner), a file-reading check that
  opens a file outside the trial directory, or a way out of the sandbox through the engine socket or the shims;
- untrusted data files: a crafted input that makes a check run code, write outside its temporary directory or exceed its
  limits (4,000,000 bytes of text per file; 30 s and 6 GiB of address space per dry run of a tool; 20 s per call);
- an API key value in a ledger, a trial record, a log or a message.

Not in scope: the limits below, flaws in the bioinformatics tools or in dependencies (report those upstream), and silent
errors the layer misses (use the issue forms).

## What the guard is not

It is a pre-check and a set of shims, not a sandbox.

- `biowarden -- COMMAND` runs COMMAND as an ordinary child process with a directory of shims first on `PATH`. Nothing limits
  what it reads, writes or sends, and a watched program called by its absolute path, or after `PATH` is reset, is not checked.
- It fails open when the layer itself fails: an engine that cannot be reached (the shim says so on stderr), a name the host
  cannot stat or an evaluation longer than 20 s lets the call run, with a warning at most. `--mode warn` lets every call run.
- The pre-check splits bash strings and is not a shell parser: globs, brace expansion and variables not set by a literal
  assignment stay unexpanded, and only the first 200 simple commands of a string are checked.
- `biowarden-mcp` runs the served programs with the model's arguments, as the user who started it. It takes their directories from
  `BIOWARDEN_TOOLS`, else `<cwd>/.local/tools.json`, else `PATH`: a `--cwd` you do not trust can choose which program
  runs, so set `BIOWARDEN_TOOLS` or use a trusted directory.
- The checks read the files a call names with the user's permissions; they are confined to the trial directory only where a
  sandbox sits between the agent and the host, as in the benchmark runner. `--trust-python` runs the Python files of the
  `--specs` directories; YAML specs cannot run code.
- The ledger records the arguments of every checked call, and its hash chain is an integrity aid, not a security boundary
  ([ledger](docs/architecture.md#ledger)).
- The runner executes each trial's commands in bubblewrap: no network, a read-only system, a private `/tmp`, only the work
  directory persisting, an emptied environment, 120 s of CPU, 6 GiB of address space and 256 MiB per file, no seccomp filter
  and no process limit. It is built for the benchmark's own trials, not to contain hostile code.

## Telemetry

None. The code in `src/` and `tools/` makes no network request other than the benchmark runner's calls to the Anthropic API
and OpenRouter, the OpenRouter credit and endpoint checks of `tools/preflight.py`, and the downloads of the bedtools, samtools
and bcftools releases from github.com by `tools/real_code_scan.py fetch`, which runs only when you call it. The engine uses
only local Unix sockets and, for `biowarden-mcp`, MCP on stdio. This statement does not cover the dependencies.
