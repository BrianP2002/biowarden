# Stage 2 report (aggregated; raw trial logs are not published)

Registration: `stage2-draft.md` (frozen at `freeze-stage2.txt`, root `026780b7217957bdfd79a58f734b5468ef8ba794037fa2f7ec33d2b71ac8f36c`). Analysis:
`python -m science_harness.bench.stage2 analyze`, run once from the frozen code (digest `d84525f183e8fdf71eb9da3eaaf51a8de1324104de107b025b88dc68069883f0`)
after the last job had ended; all numbers below are its output, in `docs/results/stage2-results.json`. Differences from the registered procedure are in
`stage2-deviations.md`.

## What was run
Three agents (Claude Haiku 5.5; DeepSeek V4.1 Flash on the Wafer endpoint; gpt-oss-120b on the Near AI endpoint), 103 gated task families plus F4 (default
cell only), held-out seeds (five per family, a registry in the sets file), 13,104 trials: 11,694 planned trials and 1,410 for the A/A replicate (a second
bare run of every family of the primary set). Primary cell: answer reporting `bare`, tier `implicit`, route `forced` (the task sentence names the tool whose
default is the trap). Arms: `bare`, `warn-nonote`, `block-nonote` (the layer with no sentence about it in the system prompt). Default cell: full
reporting, explicit evidence, free route, three seeds, bare and block-nonote. Every planned trial has a final record (none lost, none skipped for the
budget, no refusal); every OpenRouter record carries the registered pin and the matching provider. Spend of the held-out jobs: USD 6.32 (Haiku), 7.47
(DeepSeek), 2.31 (gpt-oss).

The primary set P has 47 families: the traps that bit at least one agent in a development-seed pilot (at least two silent errors among at least three
trials that took the trapped route) and for which one of the layer's own rules would block the call. It was fixed before any held-out trial. The other
families of the 103 are Q1 (12: a bite, no own rule would block), Q2 (26: an own rule, no bite) and Q3 (18: neither).

## Registered result: the layer is effective (2 of 3 models)
`block-nonote` against `bare`, primary cell, P, faulted twin, 235 pairs per model (47 families x 5 seeds). Silent = a wrong answer in `result.json` with no
machine-readable flag (the chat reply is not scored). Points; a family counts once; whiskers are family-bootstrap 95% intervals.

| Model | Silent, no layer | Silent, block | Silent drop [95% CI] | Correct answers gained | Delivered answers lost | Clean-twin pass lost, P | Clean-twin pass lost, 103 families | Holm p | Registered rule |
|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 38.3% | 8.9% | +29.4 [+20.0, +39.6] | +22.1 | +7.2 | +1.3 | +1.4 | 5e-5 | not met: delivered loss above 5 |
| DeepSeek V4.1 Flash | 55.3% | 0.9% | +54.5 [+43.0, +66.0] | +55.3 | -0.9 | -1.3 | -1.6 | 4e-11 | met |
| gpt-oss-120b | 72.8% | 15.3% | +57.4 [+44.7, +69.4] | +55.7 | +1.7 | +3.4 | +0.8 | 2e-9 | met |

The rule: for at least two of three models, a silent drop and a gain in correct answers of at least 10 points with Holm-adjusted p < 0.05 (the larger of
the two one-sided exact sign-flip p-values), at most 5 points lost in delivered answers on P, and at most 5 points of clean-twin pass lost on P and on all
gated families. DeepSeek and gpt-oss meet all five conditions; Haiku meets four and misses the delivered-share bar, because a blocked agent sometimes ends
without an answer. Pooled over the three models the silent drop is +47.1 [+39.0, +55.2] on 47 families (descriptive). No trial was lost, so the
worst-case analysis for lost trials has nothing to add: the verdict stands.

Where the bare silent trials of P went under `block-nonote` (same family and seed):

| Model | Bare silent | Now correct | Flagged | Undelivered | Still silent | Silent only with the layer |
|---|---|---|---|---|---|---|
| Haiku | 90 | 63 | 0 | 14 (no file 8, wrong type 6) | 13 | 8 |
| DeepSeek | 130 | 129 | 0 | 1 | 0 | 2 |
| gpt-oss | 171 | 127 | 0 | 23 (no file 17, not an object 3, invalid JSON 1, turn cap 1, wrong type 1) | 21 | 15 |

## Where the effect is, and where it is not
`block-nonote` against `bare`, silent drop in points, two-sided exact sign-flip p (not adjusted), family bootstrap intervals. Q is not a placebo: only Q1
(a bite, no rule of the layer for that trap) is a check that the layer's presence alone does little.

| Set (families) | Haiku | DeepSeek | gpt-oss |
|---|---|---|---|
| P, bite and an own rule (47) | +29.4 [+20.0, +39.6] | +54.5 [+43.0, +66.0] | +57.4 [+44.7, +69.4] |
| Q1, bite, no own rule (12) | +6.7 [-8.3, +28.3] (p 0.75) | +1.7 [-10.0, +13.3] (p 1.0) | +15.0 [0.0, +31.7] (p 0.17) |
| Q2, an own rule, no bite (26) | +1.5 [0.0, +3.8] | +4.6 [0.0, +10.8] | +14.6 [+6.2, +22.3] (p 0.004) |
| Q3, neither (18) | -1.1 [-3.3, 0.0] | +2.2 [0.0, +5.6] | -1.1 [-8.9, +6.7] |
| all bite families (59) | +24.7 [+16.3, +34.2] | +43.7 [+32.9, +55.3] | +48.8 [+37.6, +60.0] |
| all gated families (103) | +14.4 [+8.9, +20.2] | +26.6 [+19.4, +34.2] | +31.5 [+23.3, +39.2] |
| the five stage-1 families (replication) | +44.0 [+32.0, +56.0] | +88.0 [+72.0, +100.0] | +80.0 [+64.0, +96.0] |

P minus Q1: +22.7 [+0.1, +41.9] (Haiku), +52.8 [+37.2, +68.7] (DeepSeek), +42.4 [+21.4, +62.2] (gpt-oss). The effect is large where a trap bites and a rule
was written for it, near zero for Haiku and DeepSeek where no rule applies (Q1) or nothing bites (Q3), and diluted to +14 to +32 points over every gated
family. For gpt-oss, Q1 and Q2 show some movement that the registration cannot attribute (a rule written for another family firing, or the interruption
itself). The warning-only arm (`warn-nonote`) also meets the registered bars on P for all three models: silent drop +26.0 [+17.4, +35.3] (Haiku), +42.6
[+31.5, +53.2] (DeepSeek), +31.1 [+19.1, +42.6] (gpt-oss) with no loss of delivered answers (secondary, no claim).

By pack (silent drop, P, number of families; no tests): the original, quantification, sequence and single-cell packs move all three models (+20 to +89);
for Haiku the drop is near zero in the alignment (+0.0, 3 families), intervals (-3.3, 6), variants (+6.7, 6) and structure (+4.0, 5) packs, and for gpt-oss
in the structure pack (-8.0, 5). By locus: data +37.3 / +70.7 / +72.0 (15 families), tool default +26.7 / +47.3 / +51.3 (30 families; Haiku / DeepSeek /
gpt-oss).

## Robustness
- The registered decision is "effective" in every registered sensitivity: without the five stage-1 families (42 families), leaving each of the nine packs
  out in turn (38 to 46 families), without the one, two or three most influential families, with weak bites added (54), with each model's own bite set,
  with every bite family whether or not a rule would block (59), without the 851 layer trials that had a layer error or touched the layer, and with a
  lenient reading of `result.json` (no trial had to be re-read). Two models pass in each, three in some.
- A/A noise. The two bare runs of P differ by +1.7 [-3.8, +7.2] (Haiku), +1.3 [-3.4, +6.0] (DeepSeek), -3.4 [-9.8, +3.4] (gpt-oss) points: the effects
  above are many times the test-retest noise.
- Harm and friction, default cell (full reporting, explicit evidence, free route; three seeds per family): faulted silent drop +2.3 / +1.0 / +5.2 points,
  clean-twin pass lost +1.9 / -0.3 / -3.6, delivered answers lost 0.0 / -0.5 / -1.1 (Haiku / DeepSeek / gpt-oss); no model has a harm flag. Where the
  agents rarely err, the layer changes little.
- Friction, primary cell: 9.3% / 11.8% / 10.5% of the clean twins of the gated families were blocked at least once (77.9% / 91.9% / 82.6% of the faulted
  twins of P). Regret (a pair right without the layer and wrong with it): faulted 8/139, 2/102, 6/31 pairs, against 21/139, 21/102, 17/31 between the two
  bare runs.
- Bypass. In the faulted layer trials of P, the trapped route ran in 57 of 235 (Haiku), 77 (DeepSeek) and 86 (gpt-oss) trials, 37, 70 and 69 of them after
  a rule had fired; of the trials in which it ran, 15, 2 and 25 ended silent.

## What the result does not show
- The rules were written for these families. The seeds were held out, the trap types were not; P is chosen where agents already fail and a rule exists,
  so +29 to +57 points is the effect there, not on bioinformatics tool errors in general (+14 to +32 over every gated family, which includes traps that
  never bit). Whether the rules cover traps nobody wrote a family for is a separate coverage study that has not been run.
- Silent is a property of `result.json`. In the bare answers of P that were silent, the final chat message gave the exact expected value in 21 of 90
  (Haiku), 17 of 130 (DeepSeek) and 1 of 171 (gpt-oss) cases, and a phrase of unreliability in 1, 0 and 0; the rest said nothing the regex finds (a hand audit of 20 of these
  trials, 12 gpt-oss, 6 DeepSeek and 2 Haiku, is listed in the JSON).
- The layer is discoverable. Commands that name its path (`/opt/tbm`) occur in 278 of 1,030 `block-nonote` trials of Haiku, 119 of DeepSeek and 69 of
  gpt-oss; the sensitivity row above drops them. The agents were not told about the layer in the prompt.
- One harness, generated data, three agents on one endpoint each, one run per (family, seed, arm). The default cell has three seeds per family and no test.
- gpt-oss ends without a valid `result.json` in about 15% of its bare trials; they count as not silent and stay in the denominators.
- Haiku declines more often when blocked (delivered loss +7.2 points), which the registered rule counts against it. The 14 undelivered trials are listed
  by kind above.

## Reproduce
`docs/preregistration/stage2-draft.md` (Procedure) lists every command; the sets file, the plan manifest and the two freeze files are in this directory,
`docs/results/stage2-results.json` holds every number of the analysis including the per-family differences, and `docs/figures/stage2_silent_errors.svg`
is drawn by `tools/make_figures.py --stage2`.
