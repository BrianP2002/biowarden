# bed-base: a regions file whose name hides its coordinate base

`variants.vcf.gz` (with its index) and `regions.tsv`, which holds BED intervals (0-based, half-open)
under a name that does not end in `.bed`. bcftools reads such a file as 1-based inclusive, so every
interval starts one base early. The exit status is 0 and the output looks fine.

Run from this directory, with `bcftools` on the `PATH`:

```console
$ bcftools view -H -R regions.tsv variants.vcf.gz | wc -l       # no layer: exits 0
17
$ cp regions.tsv regions.tsv.bed && bcftools view -H -R regions.tsv.bed variants.vcf.gz | wc -l
14
$ biowarden -- bcftools view -H -R regions.tsv variants.vcf.gz    # blocked, with the fix
```

The blocked output is the demo in the top-level README (`tests/test_examples.py` checks that the two
agree). Its tag reads `rule id [convention/tier]`: a T0 rule decides from the arguments alone, a T1
rule also reads the files the call names and may run the real tool for a dry run (30 s cap). The
output ends with `Also: bcftools.region_semantics ...`, a second rule.

Following the fix meets that second rule, which asks which convention you mean. Say it on the tool's
own command line (the shim removes `--declare` before bcftools runs, and `BIOWARDEN_DECLARE` does the same):

```bash
cp regions.tsv regions.tsv.bed
biowarden -- bcftools view -H -R regions.tsv.bed variants.vcf.gz --declare region_semantics=overlap     # the 14 records
```

The files are generated data from the benchmark's family `F1` (seed 0, faulted twin).
