# Cases written without sight of the rules

How often does the layer catch a silent error that nobody wrote a rule for? The 105 task families were written together with the
rules, so the stage-2 result is the layer's effect where a rule applies. This page asks the other question. Eight authors who never
saw the rules wrote 123 pairs of commands on tiny inputs, each a silent error and its right twin; reviewers tried to refute them
and accepted 123; the layer then ran on each pair in its default mode. It blocked 22 of the 123 faulted commands (18%): 21 of the 43
that repeat the trap of a benchmark family, and 1 of the 80 that do not.

The check was not pre-registered. It ran on 2026-11-04 and 2026-11-05 against the frozen rules (108, all promoted, so every failing
rule blocks). The replay called no model. The authors, the reviewers and the labellers were Claude agents (Sonnet 5.5). The cases, the
decisions and the results are in [`blind-cases/`](blind-cases/README.md) and [`results/blind-coverage.json`](results/blind-coverage.json).

## Method

1. **Authors.** Eight agents, one for each group of programs: read alignment, RNA-seq quantification, variants, intervals, sequence
   tools, protein structure, single-cell, and everyday samtools, bcftools and bedtools use with awk, sort and pandas. Each was told
   what a silent error is, which programs to use and the case format. It was not shown a rule, a task family or the layer, and was told
   not to read the repository. Their tool calls were audited afterwards: 868 calls of the authors and 370 of the reviewers, and none
   touched the repository ([the briefs](blind-cases/briefs.md)). Between them the authors also dropped 121 ideas that they could not
   make silent, clearly wrong or clearly right on the other side.
2. **A case** is a `faulted` command that an analyst or an agent would plausibly write and a `clean` one that an expert would, on the
   same tiny input, with the intent in plain words and an `observe` line that prints the result that differs.
3. **Verification.** `blind_coverage.py verify` ran both commands of every case twice in a sandbox, without the layer. A case stays
   when both exit 0 and the two observations differ and repeat: 123 of 123 did.
4. **Review.** One reviewer agent for each group, told to refute, answered four questions for every case: is the faulted result wrong
   for the intent, is the clean one right, would a hurried analyst plausibly write the faulted command, is it silent. All
   123 got four yes: none was rejected and none was called a duplicate. Their reasons and notes are in
   [`reviews.json`](blind-cases/reviews.json), with the softest points: slips in an author's wording and cases that depend on a
   program version.
5. **Replay.** Each faulted and each clean command ran in a fresh copy of its workspace under `biowarden`, in its default mode.
6. **Labels.** After the replay, agents that could see it labelled each case: does it repeat the trap of a benchmark family (the same
   program and the same pitfall, not the program alone), does a block name the error of the case, and is a block on a clean command
   about a hazard that is there ([`labels.json`](blind-cases/labels.json)).

The layer's answer to a faulted command is *caught* (a finding on it and none on the clean command), *both twins* (a finding on both),
*declaration* (only requests for a `--declare`), *warned* or *missed*. A finding is a rule that read the call or its files and found the
error it knows; a request is a rule that asks for a declaration on every call in its scope ([the two kinds](real-code-scan.md#what-blocked)).

## Results

| Domain | Cases | Caught | Both twins | Declaration | Warned | Missed | False blocks | Clean asked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| align | 18 | 2/18 (11%, 3-33%) | 0 | 1 | 5 | 10 | 0 | 1 |
| core | 15 | 3/15 (20%, 7-45%) | 1 | 0 | 0 | 11 | 1 | 1 |
| intervals | 15 | 6/15 (40%, 20-64%) | 0 | 1 | 0 | 8 | 1 | 1 |
| quant | 15 | 1/15 (7%, 1-30%) | 0 | 0 | 2 | 12 | 0 | 0 |
| sc | 15 | 0/15 (0%, 0-20%) | 0 | 0 | 0 | 15 | 0 | 0 |
| seq | 15 | 5/15 (33%, 15-58%) | 0 | 0 | 2 | 8 | 0 | 0 |
| struct | 15 | 3/15 (20%, 7-45%) | 1 | 0 | 0 | 11 | 1 | 0 |
| variants | 15 | 2/15 (13%, 4-38%) | 1 | 1 | 0 | 11 | 2 | 2 |
| all | 123 | 22/123 (18%, 12-26%) | 3 | 3 | 9 | 86 | 5 | 5 |
| call seen by a shim | 118 | 22/118 (19%, 13-27%) | 3 | 3 | 9 | 81 | 5 | 5 |
| repeats a family trap | 43 | 21/43 (49%, 35-63%) | 1 | 2 | 0 | 19 | 2 | 4 |
| new trap | 80 | 1/80 (1%, 0-7%) | 2 | 1 | 9 | 67 | 3 | 1 |

Caught for the reason of the case: 22 of 22.
Findings on the clean command, judged (5): 0 valid, 5 false positive, 0 unclear.

Terms of the table: *both twins* is a finding on the faulted and on the clean command; *false blocks* counts the cases whose clean
command got a finding, *clean asked* those whose clean command only got a request; *call seen by a shim* counts the cases whose
faulted command ran a program that has a shim.

**What it catches.** The layer catches what its rules were written for: 21 of 43 cases that repeat a family trap, and 1 of 80 that
do not. The other 22 repeats were not caught: 2 only asked for a declaration and 1 blocked both twins; 5 were Python scripts that open
their data themselves, and the single-cell rules look for the data files a call names; the rest used a subcommand, an option or a data
layout that the family's rule did not recognise, for example `samtools view -c -F 4` (F8), `bcftools annotate -a` with other contig
names (F3) or `bedtools flank` (F16). The 9 cases that only got a warning all got the same one, from `fastq.phred_offset`, which could not
decide the encoding of a tiny FASTQ and says nothing about the error of the case. The one new trap that was caught (`intervals-06`,
the default 100% overlap of `bedops --element-of`) belongs to `bedops.element_of_amount`, a rule that no family lists as its own. Of the faulted commands that were not caught, 5 ran a
program the layer has no shim for (STAR three times, `pdb_selatom` and `pdb_reres`), and the rest ran under a shim that no rule answered.

**What it asks.** 5 correct commands were blocked by a finding. In all 5 the labellers found the hazard absent from the command as
written or latent on its data: a rule saw one stage of a pipeline, or could not tell which column or which coordinate base was meant.
The four rules involved (one of them twice) all take a `--declare` that lifts the block. 5 more were held up by a request. 10 of the
123 correct commands (8%) were stopped in all.

**What it changes.** Of the 208 commands that were allowed, all 208 gave the result they give without the layer: the shims were
transparent.

### What was caught

| Rule | Cases |
|---|---|
| `bcftools.regions_file_base` | core-05, variants-02 |
| `bedtools.spliced_without_split` | core-09, intervals-04 |
| `fastq.phred_offset` | align-09, seq-05 |
| `bcftools.format_filter_scope` | core-08 |
| `bedops.bedmap_score_column` | intervals-08 |
| `bedops.element_of_amount` | intervals-06 |
| `bedops.sorted_input` | intervals-07 |
| `bedtools.subtract_trims` | intervals-03 |
| `bwa.mem_min_score` | align-06 |
| `deeptools.matrix_missing_data` | intervals-14 |
| `diamond.hit_cap` | seq-11 |
| `hmmer.split_database` | seq-14 |
| `htseq.strandedness` | quant-01 |
| `mmseqs.cluster_coverage` | seq-15 |
| `pdbtools.modified_residues` | struct-09 |
| `plink.allele_order` | variants-09 |
| `prody.range_end` | struct-10 |
| `rdkit.pdb_bond_orders` | struct-14 |
| `seqkit.rmdup_strands` | seq-04 |

## Reading it

- A case is a silent error that a model recalls from documentation, forums and papers. It is not a random draw of the errors that
  happen in practice, and its authors and reviewers are models of one family.
- About fifteen cases for each group of programs are enough to see a pattern and not to compare groups: the intervals of the rows
  are wide.
- The rules were frozen before any case was written, and no result changed a rule. What the page measures is what the families left
  uncovered; each missed case names a trap, a program and an option that a rule could take up.
- The labels are judgements. The match with a family needs the same program and the same pitfall, so a trap that shares a convention
  with a family but sits in another program counts as new.

## Reproduce

```bash
python tools/blind_coverage.py verify docs/blind-cases --out verified.json
python tools/blind_coverage.py replay docs/blind-cases --verified verified.json --out replay.json
python tools/blind_coverage.py summary replay.json --accepted docs/blind-cases/accepted.json --labels docs/blind-cases/labels.json
```

The programs of the cases must be on `PATH`, among them the Python environments of the structure and the single-cell cases
([Run and reproduce](reproduce.md#two-later-checks)). A case whose program is missing fails `verify` and drops out.
