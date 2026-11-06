# The test suites of three tools, with and without the layer

How often does the layer stop a call that is correct? This page runs the regression tests of bedtools, samtools and bcftools,
which are the tools' own lists of calls with known answers, once without the layer and once under `biowarden` in its default
mode, and counts what changed. Of the 6,421 calls the layer checked, 95.5% passed untouched, 1.9% got a warning and 2.6% were
blocked. 161 of the 3,594 tests that pass without the layer (4.5%) did not pass with it.

The check was not pre-registered. It ran against the frozen rules (108, all promoted, so every failing rule blocks), without a
model call, before and after the project was renamed, with the same counts. The numbers are those of the second run, and are in
[`results/real-code-scan.json`](results/real-code-scan.json).

## Method

- **Suites.** bedtools 2.31.1 (31 shell scripts, `test/<subcommand>/test-<subcommand>.sh`), samtools 1.24 and bcftools 1.24
  (`test/test.pl`). `tools/real_code_scan.py fetch` downloads the tagged releases and compares the digest of each unpacked
  `test` tree with the one written in the script.
- **Runs.** A suite runs in a scratch directory under bubblewrap (the file system read-only but that directory, no network):
  twice as it is, and once under `biowarden`, which checks every call to a watched program on `PATH` by name (samtools, bcftools,
  bedtools, awk, sort, perl and the rest). Two runs without the layer tell a flaky test from a disturbed one; none was flaky.
- **Disturbed** is a test that passes in both runs without the layer and does not pass with it. A bedtools test is one
  `label...ok` line. For samtools and bcftools the runs are compared by the totals `test.pl` prints.
- **Two changes to the suites**, the same in every run. The bedtools scripts lose their `set -e` line, so that a blocked call
  fails its own test instead of ending the script. `cmd` in `test.pl`, which builds inputs and expected outputs and ends the
  script at the first failure, prints the failure and goes on.

## Results

| Suite | Tests that pass without the layer | Disturbed | Calls checked | Allowed | Warned | Blocked |
|---|---:|---:|---:|---:|---:|---:|
| bedtools 2.31.1 | 663 of 715 | 15 | 818 | 714 | 90 | 14 |
| samtools 1.24 | 952 of 1,032 | 0 | 1,354 | 1,336 | 18 | 0 |
| bcftools 1.24 | 1,979 of 1,981 | 146 | 4,249 | 4,084 | 13 | 152 |
| all | 3,594 | 161 | 6,421 | 6,134 | 121 | 166 |

The tests that fail without the layer too (for example a helper program the suite builds, or a library warning that this machine
adds to stderr) say nothing and are left out. Every disturbed bedtools test is a blocked call (14) or a warning (1, below). For samtools and bcftools the disturbed
count is the drop in the passing total, not a count of blocked calls: bcftools had 152 blocked calls and 146 disturbed tests.

### What blocked

A *request* is a rule that asks for a `--declare` on every call in its scope and looks at nothing else. A *finding* is a rule
that read the call and its files and found the situation it guards. The layer's answer to both is the same: the fix text, and
one re-run with `--declare key=value`.

| Rule | Kind | Blocked calls | What it guards |
|---|---|---:|---|
| `bcftools.region_semantics` | request | 101 | `-r`, `-R`, `-t` and `-T` do not agree on whether a record that begins before the region is kept |
| `bcftools.merge_absent_sites` | finding | 26 | `merge` writes `./.` for the samples of an input that lacks a site |
| `bcftools.format_filter_scope` | finding | 11 | a FORMAT test in `-i`/`-e` is true when any sample satisfies it |
| `bcftools.missing_value_filter` | finding | 10 | a comparison on a missing value is false for every operator |
| `bcftools.subset_monomorphic` | finding | 4 | `view -s` keeps the sites at which no selected sample carries an alternate allele |
| `bedtools.subtract_trims` | finding | 11 | `subtract` trims a partly covered feature, it does not remove it |
| `bedtools.spliced_without_split` | finding | 2 | without `-split` a spliced read covers the introns it skips |
| `bedtools.prec_significant` | request | 1 | `-prec N` keeps N significant digits, not N decimals |

102 of the 166 blocks are requests and 64 are findings. In a suite that tests the defaults on purpose, a finding is often the
point of the test: `bedtools subtract` is tested on intervals it trims and `bcftools merge` on inputs that lack sites. The layer
is right about the default and cannot know the intent. samtools had no block.

### Warnings

121 calls got a warning and ran. A warning here is a rule whose verdict was *unknown* (it could not tell, for example, the unit
of a distance): unknown only warns, in the modes warn, block and promoted, also for a promoted rule. A warning is written to the tool's stderr, so a script that compares stderr sees it: one
bedtools test (`merge.t23b`) compares the stderr of the call before it, and failed for that reason. The warnings came from
`bedtools.closest_distance_unit` (71), `bedtools.merge_bookended` (19), `bcftools.isec_representation` (11),
`samtools.mapq_unique_cutoff` (10), `bcftools.format_filter_scope` (6), `samtools.merge_combine_rg` (6) and
`fastq.phred_offset` (2); a call can carry two.

### Time

The layer added about 133 ms to each call it checked for bedtools, 164 ms for samtools and 320 ms for bcftools (the time of
the guarded run minus the time of the plain run, divided by the calls checked). They were measured on a shared cluster node with
the code on a network file system; another machine gives other times.

## Reproduce

```bash
python tools/real_code_scan.py fetch sources
python tools/real_code_scan.py run sources --out scan.json --compact docs/results/real-code-scan.json
python tools/real_code_scan.py summary scan.json
```

It needs bedtools 2.31.1, samtools 1.24, bcftools 1.24, `perl` and `bwrap` (see [Run and reproduce](reproduce.md#two-later-checks)).
The counts repeat; the times do not.
