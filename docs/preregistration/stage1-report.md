# Stage 1 report (aggregated; raw trial logs are not published)

Pre-registration: `stage1.md`. Code digests: `freeze.txt`, `freeze-rerun.txt` (the other freeze files and a note on them are at the end).
Analysis: `python -I -m science_harness.bench.stage1`.
6 models, 5 families (F1, F7, F10, F13, F15), seeds 1000-1019 held out from rule development, 2 twins, hardest cell
(bare reporting / implicit / forced), 7800 trials in total including a 3-arm safety cell. Zero infrastructure errors remain
(Deviation 1 in the pre-registration lists the one rerun).

## Primary result: faulted-twin silent_wrong over all planned trials (100 per model and arm)

| model | bare | warn | block | block-bypass | block-gateway |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 60% | 0% (p<0.001) | 0% (p<0.001) | 1% (p<0.001) | 11% (p<0.001) |
| deepseek-v4.1-flash | 94% | 0% (p<0.001) | 1% (p<0.001) | 2% (p<0.001) | 24% (p<0.001) |
| gpt-oss-120b | 85% | 9% (p<0.001) | 8% (p<0.001) | 1% (p<0.001) | 62% (p<0.001) |
| qwen3.8-flash | 60% | 0% (p<0.001) | 0% (p<0.001) | 1% (p<0.001) | 4% (p<0.001) |
| mimo-v2.6-flash | 66% | 0% (p<0.001) | 0% (p<0.001) | 0% (p<0.001) | 46% (p_holm 0.006) |
| glm-5.3-flash | 91% | 6% (p<0.001) | 5% (p<0.001) | 7% (p<0.001) | 74% (p_holm 0.003) |

Pooled over the five open-weight models (cluster bootstrap, exploratory):

- warn: +76.2 points, 95% CI [+72.0, +80.0] (100 family-seed clusters)
- block: +76.4 points, 95% CI [+72.2, +80.0] (100 family-seed clusters)
- block-bypass: +77.0 points, 95% CI [+73.0, +80.6] (100 family-seed clusters)
- block-gateway: +37.2 points, 95% CI [+31.4, +42.8] (100 family-seed clusters)

Pre-registered criterion (block against bare, open-weight models only): **effective = True**. Models with a drop of at least 10
points and Holm p < 0.05: 5 of 5 (need 3). Pooled clean-twin pass-rate loss -0.4 points,
pooled delivered-share loss +0.6 points (each had to be at most 5).

## Side effects (both twins, hardest cell)

| model | arm | clean pass | clean trials with a block | delivered | median turns |
|---|---|---|---|---|---|
| claude-haiku-5-5 | bare | 94/100 | 0/100 | 91% | 4 |
| claude-haiku-5-5 | warn | 90/100 | 0/100 | 90% | 5 |
| claude-haiku-5-5 | block | 91/100 | 40/100 | 90% | 5 |
| claude-haiku-5-5 | block-bypass | 91/100 | 40/100 | 90% | 5 |
| claude-haiku-5-5 | block-gateway | 83/100 | 20/100 | 88% | 5 |
| deepseek-v4.1-flash | bare | 98/100 | 0/100 | 100% | 6 |
| deepseek-v4.1-flash | warn | 97/100 | 0/100 | 98% | 6 |
| deepseek-v4.1-flash | block | 95/100 | 40/100 | 97% | 7 |
| deepseek-v4.1-flash | block-bypass | 95/100 | 40/100 | 96% | 7 |
| deepseek-v4.1-flash | block-gateway | 92/100 | 26/100 | 100% | 6 |
| gpt-oss-120b | bare | 87/100 | 0/100 | 86% | 8 |
| gpt-oss-120b | warn | 89/100 | 0/100 | 90% | 9 |
| gpt-oss-120b | block | 92/100 | 40/100 | 88% | 9 |
| gpt-oss-120b | block-bypass | 89/100 | 39/100 | 88% | 10 |
| gpt-oss-120b | block-gateway | 88/100 | 30/100 | 91% | 9 |
| qwen3.8-flash | bare | 99/100 | 0/100 | 100% | 7 |
| qwen3.8-flash | warn | 100/100 | 0/100 | 100% | 8 |
| qwen3.8-flash | block | 100/100 | 40/100 | 99% | 9 |
| qwen3.8-flash | block-bypass | 100/100 | 40/100 | 100% | 9 |
| qwen3.8-flash | block-gateway | 100/100 | 20/100 | 100% | 8 |
| mimo-v2.6-flash | bare | 96/100 | 0/100 | 93% | 6 |
| mimo-v2.6-flash | warn | 98/100 | 0/100 | 95% | 6 |
| mimo-v2.6-flash | block | 94/100 | 40/100 | 91% | 7 |
| mimo-v2.6-flash | block-bypass | 93/100 | 40/100 | 90% | 7 |
| mimo-v2.6-flash | block-gateway | 96/100 | 7/100 | 94% | 6 |
| glm-5.3-flash | bare | 91/100 | 0/100 | 96% | 4 |
| glm-5.3-flash | warn | 97/100 | 0/100 | 98% | 5 |
| glm-5.3-flash | block | 92/100 | 40/100 | 98% | 5 |
| glm-5.3-flash | block-bypass | 93/100 | 41/100 | 100% | 5 |
| glm-5.3-flash | block-gateway | 88/100 | 6/100 | 94% | 5 |

About 40% of clean trials were blocked once: the `region_semantics` rule deliberately requires a declaration on every bcftools
region call (F1 and F7 are 2 of the 5 families). It costs about one turn and no accuracy.

## How to read it

- `warn` is as good as `block` for these models: the call runs, the warning carries the fix, and the model follows it. What helps is
  delivering the missing convention at the moment of need; blocking was not necessary here.
- `block-gateway` helps much less because the model keeps bash and often routes around the typed tool (it used `run_tool` in
  386 of 1000 trials). The shim catches the call whichever way the model makes it.
- The rules were written for these five families; the seeds were held out, but the trap types were known. The result shows that
  the mechanism works when a rule exists. It does not measure coverage of traps for which no rule exists.
- One harness (a minimal bash loop), one prompt sentence describing the layer (see the ablation below), cheap open-weight models and one
  reference model, single-step tasks.

## Post-hoc ladder (not pre-registered; complete)

Added after the primary result to separate what the layer does from what the prompt does. Same families, seeds and cell as the primary
analysis; faulted-twin silent_wrong over all planned trials (100 per model and arm; 500 pooled).

| arm | deepseek-v4.1-flash | gpt-oss-120b | qwen3.8-flash | mimo-v2.6-flash | glm-5.3-flash | claude-haiku-5-5 | five open-weight pooled |
|---|---|---|---|---|---|---|---|
| no layer | 94% | 85% | 60% | 66% | 91% | 60% | 79% |
| prompt: check formats | 79% | 88% | 51% | 67% | 81% | 39% | 73% |
| prompt: layer sentence only | 84% | 81% | 52% | 62% | 84% | 47% | 73% |
| layer, generic warning | 75% | 88% | 33% | 72% | 83% | 46% | 70% |
| layer, generic block | 42% | 66% | 38% | 68% | 68% | 38% | 56% |
| typed tool + bash | 24% | 62% | 4% | 46% | 74% | 11% | 42% |
| layer, full warning, no prompt sentence | 8% | 31% | 1% | 4% | 5% | 1% | 10% |
| layer, full block, no prompt sentence | 0% | 6% | 0% | 1% | 6% | 2% | 3% |
| layer, full block, prompt sentence | 1% | 8% | 0% | 0% | 5% | 0% | 3% |

Reading the ladder (pooled over the five open-weight models, McNemar on paired (family, seed, model) trials against no layer):

- A generic carefulness sentence in the prompt barely helps (79% to 73%, p = 0.014); the layer's own one-sentence note, with no layer, is
  the same (73%, p = 0.007).
- The layer that only says "a convention may be misread, check formats and meaning" helps modestly as a warning (70%, p < 0.001) and more as
  a one-time speed bump (56%, p < 1e-15). Detection alone is not enough.
- The layer with the specific message is far better: 10% as a warning and 3% as a block, with no prompt sentence at all. The effect is
  therefore the layer, not the prompt wording.
- For gpt-oss-120b the warning without the prompt sentence is much weaker (31%) than blocking (6%): that model needs the block.
- A typed tool next to an unchecked bash (42%) is limited by models routing around it; the shim catches the call whichever way it is made.

## Negative control (post hoc; rewritten on 2026-11-01)

F17-F28 were run as a check that the layer does nothing where it has no rule. The first version of this paragraph called them "families
without a rule"; that was wrong, and this text replaces it. The control was not registered: it was added after the primary result, and the
strata below were defined after the results had been read, from the layer's own record (which rules blocked in the block-arm trials) and
not from the outcomes. Seeds 1000-1009, hardest cell, three arms (no layer, the layer's prompt sentence only, the full block), six models,
4,319 trials (one planned trial has no record). Cells are faulted-twin silent_wrong / planned trials.

| stratum | families | what the layer did | five open-weight models: none, prompt sentence, block | six models: none, block |
|---|---|---|---|---|
| no block in any trial | F17 F19 F21 F23 F25 F26 F27 | nothing (F26: one block in 60 faulted trials) | 188/350, 202/350, 193/350 | 224/420, 224/420 |
| a rule written for another family blocks | F18 F20 F24 F28 | `bcftools.region_semantics` blocked every trial of F18 and F28 (both twins) and every faulted trial of F20; `matrix.log_scale_state` blocked every trial of F24 | 100/200, 99/200, 97/200 | 110/240, 106/240 |
| the family's own rule | F22 | `fastq.phred_offset` blocked 27 of 60 faulted trials, all on even seeds | 31/50, 32/50, 19/50 | 40/60, 22/60 |

F22 is not a negative control. Its sentence states Phred+33 on even seeds and Phred+64 on odd seeds, and the faulted file has the other
offset. On even seeds the data are Phred+64 and `fastq.phred_offset` blocks (27 of the 30 faulted block-arm trials): silent_wrong fell from
27/30 to 7/30 over six models (22/25 to 5/25 for the open-weight models). On odd seeds no rule applies and nothing changed (13/30 to 15/30).
Without F22 the pooled movement of the five open-weight models is 288/550 (no layer) to 290/550 (block), exact McNemar p = 0.92 (43 and
45 discordant trials); with F22 it is 319/600 to 309/600, p = 0.40 (61 and 51). The p = 0.069 that this report gave for "block versus none"
belongs to the six-model pool (78 and 56 discordant trials), not to the five open-weight models.

| model | none | prompt sentence | block | none, without F22 | block, without F22 |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 55/120 | 44/120 | 43/120 | 46/110 | 40/110 |
| deepseek-v4.1-flash | 66/120 | 69/120 | 61/120 | 60/110 | 59/110 |
| gpt-oss-120b | 92/120 | 91/120 | 78/120 | 82/110 | 72/110 |
| qwen3.8-flash | 25/120 | 29/120 | 21/120 | 22/110 | 21/110 |
| mimo-v2.6-flash | 54/120 | 54/120 | 54/120 | 47/110 | 49/110 |
| glm-5.3-flash | 82/120 | 90/120 | 95/120 | 77/110 | 89/110 |
| five open-weight models | 319/600 | 333/600 | 309/600 | 288/550 | 290/550 |
| six models | 374/720 | 377/720 | 352/720 | 334/660 | 330/660 |

The pooled null hides movement in single models. Without F22, Haiku fell from 46 to 40 of 110, gpt-oss from 82 to 72 and GLM rose from 77 to 89
(-5.5, -9.1 and +10.9 points); the other three are within 2 trials. Even where the layer blocked nothing, the arms of one model differ by
up to 9 of 70 trials (GLM, 50 to 59). That is either the run-to-run noise of one run per arm or a non-specific effect of the shims and the
pre-check; this control cannot tell them apart, and stage 2 adds a second bare run on the families of its primary set to measure it.

## Notes added on 2026-11-01

Freeze files. `freeze.txt` was taken from the launch snapshot, except its `stage1` line (the analysis script), which was recorded after the jobs
were submitted and before any stage-1 result was inspected. In `freeze-rerun.txt`, `freeze-ablation.txt`, `freeze-nonote.txt` and
`freeze-generic.txt` the `stage1` line is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, the SHA-256 of empty input:
the snapshots those four files describe were taken without `bench/stage1.py`, so that group had no file. The analysis script is frozen
by the `stage1` line of `freeze.txt` and of `freeze-negctl.txt` (both `115999c7...8816`), and `stage1.py` still has that digest (checked on
2026-11-01). The other lines of the four files are digests of real files. The files are left as they were.

Sandbox. The stage-2 review found that an agent could read the process table of the sandbox, which printed the host path of its trial
directory and so the family, seed, twin, arm and model. Of the 19,319 trials of stage 1 and of the negative control, 11 read the process
table (5 and 6) and 11 the repository path (6 and 5). None looked at the project's private directory: it entered the sandbox's tool
configuration on 2026-10-30, after stage 1 had run. Leaving out the trials that read the process table or the repository path moves none of
six arms of the hardest cell (no layer, prompt check, generic block, warn and block without the prompt sentence, block) by more than 0.2 points
of faulted-twin silent/planned (six models pooled, 600 trials per arm; no layer stays at 76.0% and block at 2.3%). The stage-2 sandbox hides
these paths.
