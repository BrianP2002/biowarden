# biowarden

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-246893?labelColor=0C1F4B)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-246893?labelColor=0C1F4B)](pyproject.toml)

**Catch silent errors in bioinformatics tool calls before they run, and measure how often LLM agents make them.**

A *silent error* is a tool call that exits 0 and returns a plausible but scientifically wrong result: a BED file read as 1-based,
a Phred+64 FASTQ read as Phred+33, a log-scaled matrix treated as raw counts. Nothing in the output tells an agent, or a tired
human, that it is wrong. This project has two parts:

- a **checking layer**: 108 rules, each for one known trap, that block the call before it runs and print a one-line fix. Use
  it around any shell or agent (`biowarden`) or as an MCP (Model Context Protocol) server (`biowarden-mcp`); it runs locally
  and needs no API key.
- a **benchmark**: 105 task families, each a clean and a faulted twin (the faulted one holds the trap), that measure how often
  LLM agents end with a wrong answer.

In a study registered in advance, the layer cut silent wrong answers from 38%, 55% and 73% to 9%, 1% and 15% for three agents,
on the 47 task families where agents fail and a rule exists, in the hardest task wording ([Results](#results)). Version 0.0.1.

```console
$ cd examples/bed-base && biowarden -- bcftools view -H -R regions.tsv variants.vcf.gz
FIX: cp regions.tsv regions.tsv.bed && bcftools view -H -R regions.tsv.bed variants.vcf.gz (if
  regions.tsv already holds 1-based inclusive coordinates, re-run the original command with
  --declare region_file_base=one_based_closed instead)
BLOCKED, the command did not run. bcftools.regions_file_base [index_base/T1]: bcftools reads
  regions.tsv as 1-based inclusive CHROM POS END because its name does not end in .bed or .bed.gz,
  but its 12 data lines look like BED intervals (0-based, half-open), so each interval would start
  one base early; read as 1-based it selects 17 records of variants.vcf.gz, read as BED 14
Also: bcftools.region_semantics [interval_semantics/T0]: state which interval_semantics convention you intend …
```

Following that FIX meets the `Also:` rule, which wants the convention stated with `--declare`
([walk-through](examples/bed-base/README.md)). A tag reads `[convention/tier]`: tier T0 judges the
command line alone, tier T1 also reads the files it names.

[Get started](#get-started) | [How it works](#how-it-works) | [Results](#results) | [What it catches](#what-it-catches) | [Cite](#citing-and-contributing)

## Get started

```bash
python3 -m venv .venv && . .venv/bin/activate    # Linux, Python 3.10+
pip install -e .                                 # from a clone; samtools, bcftools, ... must be on PATH
biowarden -- bash                                # any shell or agent: calls to supported tools are checked
claude mcp add biowarden -- "$PWD/.venv/bin/biowarden-mcp" --cwd /abs/path/to/data    # or as an MCP server
```

**Notes.**
- `biowarden` is the PATH shims alone: it skips a tool called by absolute path or put first on `PATH` later (`module load`,
  `conda activate`, `~/.bashrc`) and says nothing; inside it, `type -P samtools` should end in `/shims/samtools`. `biowarden-mcp`
  checks only calls made through `run_tool`, not the agent's own Bash. The results below were measured with PATH shims plus a
  whole-command pre-check, in a sandbox.
- State a convention with `--declare key=value` or `BIOWARDEN_DECLARE`. 9 of the 108 rules ask for one on every call in their
  scope and block until it is given, even a correct call (any bcftools call with `-r`, `-R`, `-t` or `-T`). A blocked shell
  call exits 97 (use `set -o pipefail`); `--mode warn` lets calls run.
- Calibrated on samtools 1.24, bcftools 1.24 and bedtools 2.31.1 (the layer does not check your version); Linux only (CI: Ubuntu).

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/layers-dark.svg">
  <img alt="Where the layer sits. An LLM agent or user reaches the real tools through three front ends of the checking layer: biowarden, whose PATH shims check each call, by name, to a watched tool (child processes too, not command strings); a whole-command pre-check, which checks each command of a bash string and is used only in the benchmark sandbox; and biowarden-mcp, which sees only calls made through run_tool. All three hand a Call to one engine, which uses 108 rules and a promotion table and may write a hash-chained ledger; the real tools run only after the engine allows or warns. A dashed brown path shows the calls that skip the layer: with biowarden alone, an absolute path or a reset PATH; with biowarden-mcp alone, the agent's own Bash." src="docs/figures/layers.svg" width="830">
</picture>

*Three front ends share one engine, so a rule is written once. A shim is a stand-in program, first on `PATH`, that asks the engine.*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/decision-dark.svg">
  <img alt="How one call is decided, in five steps. 1: a call arrives, as a command in a process, a bash string (pre-check) or a run_tool call. 2: a front end builds a Call. 3: the rules that apply are matched, call rules by tool, subcommand and options, fact rules by the files the call names. 4: each rule checks its declaration, then runs its checker; T0 reads the arguments only, T1 reads files and may dry-run the tool. 5: the most severe effect wins, from the verdict, tier, mode and promotion table; unknown only warns. The outcome is allow or warn (the real tool runs, warn also prints a warning) or block (exit 97 and the fix text for the shim and the pre-check; biowarden-mcp returns an error result carrying the fix); the same call blocked a 3rd time ends in a STOP. Every evaluation also gets a ledger record if one is configured. A crashing checker blocks the call (warn mode only warns); a fault of the layer itself, or a shim that cannot reach the engine, lets the call run." src="docs/figures/decision.svg" width="830">
</picture>

*One call, five steps. Modes: `off`, `warn`, `block` and `promoted`, the default; all 108 shipped rules are promoted, so it blocks.
The engine never rewrites a call. More in [`docs/architecture.md`](docs/architecture.md).*

## Results

Setting: generated tasks on seeds not used while writing the rules; three agents in a minimal loop with one bash tool; the hardest
wording (the task names the tool whose default is the trap, and the agent writes only a result file, so any wrong answer is
silent). Bars: no layer, the layer in `--mode warn`, the layer in its default mode; whiskers are 95% intervals over families.
The right panel and the third column use 103 of the 105 families (F4 and F34 do not run in this setting).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/stage2_silent_errors-dark.svg">
  <img alt="Stage 2, share of faulted tasks ending in a silent wrong answer, held-out seeds. Left, the 47 families where a trap bites and a rule exists: Claude Haiku 5.5 38% with no layer, 12% with a warning, 9% with a block; DeepSeek V4.1 Flash 55%, 13%, 1%; gpt-oss-120b 73%, 42%, 15%. Right, all 103 gated families, no layer against block: 22% and 8%, 34% and 7%, 50% and 18%." src="docs/figures/stage2_silent_errors.svg" width="830">
</picture>

| Model | 47 families: no layer → block | All 103 families | Registered bars |
|---|---|---|---|
| Claude Haiku 5.5 | 38% → 9% | 22% → 8% | not met: 7.2 points more tasks ended with no answer (bar: 5) |
| DeepSeek V4.1 Flash | 55% → 1% | 34% → 7% | all five met |
| gpt-oss-120b | 73% → 15% | 50% → 18% | all five met |

> [!NOTE]
> The rules were written for these families, so this is the layer's effect where a rule applies, not a coverage claim. The layer
> counted as effective if at least two of the three agents met all five registered bars. The bars and the registration date
> (2026-11-01, a SHA-256 manifest; no external registry) are in [the registration](docs/preregistration/stage2-draft.md).
> Intervals, the other families and the stage-1 comparison with prompt-only baselines: [`docs/results`](docs/results/README.md).
> Two later checks, not pre-registered: [cases written without sight of the rules](docs/blind-coverage.md) and
> [the test suites of three tools](docs/real-code-scan.md).

## What it catches

Each pair is a real command in a faulted benchmark workspace and the first line of the block message:

```console
$ featureCounts -a genes.gtf -o counts.txt reads.bam     # dUTP library, default -s 0
FIX: re-run with -s 2 (declare library_strandedness=reverse)
$ plink --vcf cohort.vcf.gz --geno 0.1 --make-bed --out x     # multiallelic sites
FIX: run plink2 for the same options (it keeps every allele: plink2 --vcf ... --geno 0.1) or skip …
$ blastp -query queries.fasta -db proteins.fasta -outfmt 6 -max_target_seqs 1     # stale database
FIX: rebuild it first: makeblastdb -in proteins.fasta -dbtype prot -out proteins.fasta or, if the old …
```

| Domain | Tools | Rules | Families |
|---|---|---|---|
| Read alignment | bwa, Bowtie 2, minimap2, HISAT2, STAR, samtools | 11 | 11 |
| RNA-seq quantification | featureCounts, HTSeq, Salmon, kallisto, RSEM, StringTie | 10 | 6 |
| Variants and genotypes | bcftools, VCFtools, plink, plink2, tabix, GATK, Picard | 19 | 12 |
| Intervals, peaks, liftover | bedtools, BEDOPS, liftOver, deepTools, MACS3 | 26 | 12 |
| Sequence processing and search | seqtk, seqkit, fastp, cutadapt, Trimmomatic, BLAST+, DIAMOND, MAFFT, … | 12 | 12 |
| Protein structure | gemmi, Biopython, ProDy, MDAnalysis, mdtraj, RDKit, pdb-tools, TM-align, … | 13 | 12 |
| Single-cell | scanpy, anndata, mudata, loompy | 14 | 12 |
| Original families | samtools, bcftools, bedtools | 3 | 28 |

The 3 rules of the last row are about data formats and apply to any tool.
Catalogues: [`rules.md`](docs/rules.md), [`supported-tools.md`](docs/supported-tools.md), [`families.md`](docs/families.md).

## Citing and contributing

Use GitHub's "Cite this repository" ([`CITATION.cff`](CITATION.cff)), and please also cite the tools you run. Missed a silent
error, or hit a wrong block? Open an issue with the command, a small input file and the expected result. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md); tests and studies: [`docs/reproduce.md`](docs/reproduce.md).

## License

Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Figure colours: MoMAColors by Blake R. Mills (MIT; palettes VanGogh, after
*The Starry Night*, and Picabia, after *The Spring*) and MetBrewer (CC0; Hokusai3 and Hiroshige).
