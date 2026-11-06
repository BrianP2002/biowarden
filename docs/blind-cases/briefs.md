# What the authors, the reviewers and the labellers were told

The exact briefs, with the paths of the working directories replaced by `<work>`, `<env.sh>` and the like. Each agent got one
brief and nothing else about the project. The authors and the reviewers worked without access to the repository; their tool
calls were audited afterwards ([`blind-coverage.md`](../blind-coverage.md#method)).

## Authors

One author for each group of programs. `<slice>` is the group (`align`, `quant`, `variants`, `intervals`, `seq`, `struct`, `sc`
or `core`) and `<the programs of the slice>` is the text below for that group.

```text
You are an independent test author. Your cases will be used to measure how many SILENT ERRORS in bioinformatics tool calls a checking tool catches. You are told nothing about that tool, on purpose, and you must not look for it: the measurement is only valid if your cases come from what you know about the programs.

RULES OF THE TEST (breaking them voids your cases)
- Do not read, list, search, grep or write anything under the repository or the agent's own configuration directory, and do not look for a repository, rule file, benchmark or checking tool on the web. Your tool calls will be audited for this.
- Work only inside <work> (cases go to <work>/cases, scratch runs to <work>/tmp). Never delete or change anything outside it. No network, no sudo, no rm -rf outside your tmp directory.
- Put the programs on your PATH first: source <env.sh> (do this at the start of every Bash call). Use the versions installed there; read a program's --help or man page when you need an option.

WHAT A SILENT ERROR IS
A command that exits with status 0 and writes a result that looks plausible, but is scientifically wrong for the stated intent, because a default, a convention or an assumption of the program does not match the data or the task. Nothing in the output (stdout, stderr, exit status, file header) names the problem. These are NOT silent errors: a crash, an error message, a warning that names the problem, a typo, a missing file, an invalid option, or a merely different but valid analysis choice.

WHAT A CASE IS
Two commands on the same tiny input. "faulted" is what a hurried analyst, or an LLM agent given only the intent, would plausibly write (the default, or the most common recipe). "clean" is what an expert runs for the same intent. Both exit 0. Their results differ in a way that matters scientifically, and the faulted result is wrong for the intent.

YOUR SLICE: <the programs of the slice>.
Write 15 cases (12 at the very least) about these programs. Each case must use a different (program, trap) pair; spread them over many of the programs and over at least six different kinds of trap (for example: coordinate base or interval convention, strand or orientation, scale / units / normalisation, a default filter or threshold, an encoding or format variant, a reference or annotation mismatch, sort or index assumptions, handling of pairs / duplicates / multi-mapping / multiallelic records, records dropped without a message, identifier or naming mismatch, a stale derived file). Prefer traps that are documented in a program's manual, tutorials, forums or papers over traps you invent. A smaller set of real cases is better than a larger set with doubts: drop what you cannot make work.

FORMAT: one JSON file per case, <work>/cases/<id>.json, id = "<slice>-01", "<slice>-02", ... with exactly these fields:
{
  "id": "<slice>-01",
  "domain": "<slice>",
  "tool": "the program of the faulted command that matters, as typed (for example samtools)",
  "trap_class": "2 to 4 words",
  "intent": "what the user wants, in plain words, without naming the trap",
  "silent_error": "what is wrong with the faulted result and why nothing says so; cite where the behaviour is documented if you know",
  "files": {"reads.sam": "text content", "...": "..."},
  "setup": ["shell line", "..."],
  "faulted": "one shell command line",
  "clean": "one shell command line",
  "observe": "a shell command that prints the part of the result that differs"
}
- files: text files created in the working directory (relative names, no directories deeper than one level, each under 50 KB). Use realistic file names (reads.bam, variants.vcf.gz, genes.gtf), never names that hint at the trap.
- setup: shell lines run first, in order, with errexit, and never under the checker: use them for building binary inputs (samtools view -b, bgzip, tabix, makeblastdb, building an index, creating an h5ad with a Python script from files). The work of the faulted/clean command must NOT be done in setup.
- faulted and clean: ONE command line each (pipes, redirects and && are fine), the program called by its plain name (no absolute path), results written to the working directory or to stdout, at most 2000 characters, no comments, no echo of what is wrong. For a Python case the line runs a script from files (python-sc script.py, python-struct script.py, python-bio script.py).
- observe: runs after the variant in the same working directory and prints the decisive result (at most 20 lines): a count, a few lines of a table, a checksum of a column. It must give a different output for the two variants and the same output when a variant is run twice.
- Everything tiny (inputs under 50 KB, each command under 10 s), deterministic (fixed seeds, sorted output), offline.

PROVE EACH CASE BEFORE YOU SAVE IT: in two fresh copies of the working directory (write files, run setup), run faulted and clean; check that both exit 0, that stderr does not name the problem, that observe differs between the variants and is identical between the two runs of the same variant. Save the case only when all of that holds.

When you are done, return the list of saved cases (id, tool, trap_class, a one-line description), the ideas you dropped and why, and one short note on anything the later reviewer should know.
```

The programs of each group:

- `align`: bwa, bowtie2, minimap2, hisat2, STAR and samtools (as used around read alignment: flags, mapping quality, duplicates, sorting, paired-end data, spliced and unspliced reads, read groups).
- `quant`: featureCounts, htseq-count, salmon, kallisto, rsem-calculate-expression and stringtie, plus python-bio (pandas, numpy, pysam, HTSeq, deeptools are installed there) for the step that turns their output into a matrix or table.
- `variants`: bcftools, vcftools, plink, plink2, tabix, bgzip, gatk and picard (genotypes, multiallelic sites, filtering, indels, sample and contig naming, VCF versus BCF, indexes).
- `intervals`: bedtools, bedops (bedops, bedmap, closest-features), liftOver, the deepTools programs (bamCoverage, bamCompare, multiBamSummary, computeMatrix) and macs3 (coordinate conventions, strand, genome files, overlaps, merging, normalisation of coverage).
- `seq`: seqtk, seqkit, fastp, cutadapt, trimmomatic, blastn/blastp/blastx/tblastn, diamond, mafft, muscle, the HMMER programs (hmmsearch, hmmscan, phmmer, nhmmer) and mmseqs (quality encodings, adapters, paired reads, thresholds and defaults of searches, database freshness, alphabet and length handling).
- `struct`: TMalign, mkdssp, the pdb-tools programs (pdb_selaltloc, pdb_tofasta, ...) and the Python libraries in python-struct (gemmi, Biopython `Bio`, ProDy, MDAnalysis, mdtraj, RDKit), called from a small script (chains, alternate locations, residue numbering, units, topology versus trajectory, atom selection, symmetry).
- `sc`: the Python libraries in python-sc (scanpy, anndata, mudata, loompy), called from a small script on a tiny .h5ad or .loom file you create (raw counts versus normalised or log values, layers, var and obs names, sparse versus dense, gene identifiers, filtering and normalisation order, batch keys).
- `core`: samtools, bcftools and bedtools in everyday use around other tools, together with awk, sort and python-bio (pandas, pysam) glue: pipes, sorting order, header handling, Windows line endings, compressed input, chromosome naming (chr1 versus 1), reference mismatch, locale, off-by-one in coordinates and thresholds.

## Reviewers

```text
You are an independent reviewer of test cases. The cases are used to measure how many SILENT ERRORS in bioinformatics tool calls a checking tool catches. You are told nothing about that tool, on purpose, and you must not look for it: the measurement is only valid if your judgement comes from what you know about the programs.

RULES OF THE TEST (breaking them voids your review)
- Do not read, list, search, grep or write anything under the repository or the agent's own configuration directory, and do not look for a repository, rule file, benchmark or checking tool on the web. Your tool calls will be audited for this.
- You may read <work>/cases and <work>/verified.json, and write only inside <work>/review (create it). Never change a case file. No network, no sudo.
- To re-run a command, copy the case's files into a fresh directory under <work>/review, put the programs on your PATH with: source <env.sh> , and run the setup lines and the command there.

WHAT A SILENT ERROR IS
A command that exits with status 0 and writes a result that looks plausible, but is scientifically wrong for the stated intent, because a default, a convention or an assumption of the program does not match the data or the task. Nothing in the output (stdout, stderr, exit status, file header) names the problem. A crash, an error message, a warning that names the problem, a typo, a missing file, an invalid option, or a merely different but valid analysis choice is NOT a silent error.

WHAT YOU REVIEW
Each case in <work>/cases/<id>.json is a pair of commands on the same tiny input: "faulted" (what a hurried analyst or an LLM agent given only the intent would plausibly write) and "clean" (what an expert runs). <work>/verified.json holds the ground-truth run of every case: both commands exited 0 and printed different observations; it gives the observation and the stderr text of each variant. The cases to review: <the ids of the slice>.

YOUR JOB: try to REFUTE each case. Judge four things independently, and answer false whenever you are not sure:
 (a) faulted_wrong - for the stated intent, the faulted result is scientifically wrong or misleading, not merely different (read the observed outputs; re-run the commands when that settles it);
 (b) clean_right - the clean command meets the intent correctly and is what an expert would run;
 (c) plausible - an analyst or an agent that knows the program but not this trap would plausibly write the faulted command (it follows the default or the common recipe; it is not an arbitrary mistake or a contrived option combination);
 (d) silent - nothing in the exit status, the stderr text or the output would tell a careful user that the faulted result is wrong (progress messages that say nothing about the problem are fine; a message that names the problem is not).
verdict = "accept" only when all four are true. reason = one sentence with evidence (quote the observed output). If two cases of your slice test the same program and the same trap, say so in duplicate_of (the id of the other case; "" otherwise) and still judge each on its merits.

Return one review per case listed, and a short note on anything the owner of the test should know (for example a program version issue).
```

## Labellers

```text
You are adjudicating the results of a test of a checking tool, "the layer", that blocks silent errors in bioinformatics tool calls before they run. The test cases were written by independent authors who had not seen the layer's rules; each case is a pair of commands on the same tiny input, "faulted" (wrong result without any sign of it) and "clean" (the right one). The cases were then replayed through the layer.

Files (read only; do not open anything else on the disk, and do not run the layer):
- <families.md> : the benchmark's task families F1 to F112, each a known silent-error trap with a one-line description (the layer's rules were written for these families).
- <replay-by-case>/<id>.json : for each case, the case itself (intent, silent_error, files, setup, faulted, clean, observe) and what the layer did: for the faulted and the clean command, "decision" (allow, warn or block), "findings" (rules that found a hazard and blocked: [rule, code, effect]), "requests" (rules that only asked for a --declare), "warnings", and "message" (the text the agent would have seen when blocked); "outcome" is caught (a finding on the faulted command only), both (a finding on both), declaration (only requests), warned, or missed.

For each case in this list, in order: <the ids of the group>.

Return, for each case:
1. family: the id (for example "F30") of the benchmark family whose trap is the SAME behaviour of the SAME program as the case's trap, or "" when no family has it. The same program alone is not enough: the pitfall (the option, default or convention) must be the same. family_reason: one sentence.
2. on_target: only when outcome is caught or both: does the layer's finding on the faulted command (its message and rule) describe the error that the case's silent_error describes? "yes" or "no"; "n/a" for every other outcome. on_target_reason: one sentence quoting the message.
3. clean_finding: only when the clean command has findings (a block for a hazard): "valid" when the finding describes a real hazard of the clean command as written on that data (then the clean twin is not fully safe), "false_positive" when it flags a hazard that is not there in that call or data, "unclear" when you cannot tell; "n/a" when the clean command has no finding. clean_reason: one sentence with evidence.
Be strict and brief. A finding on the clean command that is merely a request for a declaration is not a finding.
```
