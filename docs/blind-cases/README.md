# Cases written without sight of the rules

The cases of [`blind-coverage.md`](../blind-coverage.md): pairs of commands on a tiny input, the first with a silent error and the
second the right way to do the same thing. Eight authors, each given a list of programs and none of the rules, wrote them; a
reviewer per group tried to refute each one; the layer then ran them. The page says what came out. This folder holds the data.

| File | What it is |
|---|---|
| `<group>-<nn>.json` | one case: `intent`, `silent_error`, the input `files`, the `setup` lines, the `faulted` and the `clean` command, and the `observe` line that prints the result that differs |
| [`verified.json`](verified.json) | both commands of every case, run twice without the layer: their exit status, the observation of each, and what they printed on stderr |
| [`reviews.json`](reviews.json) | the reviewers' answers: is the faulted result wrong, is the clean one right, would an analyst plausibly write the faulted command, is it silent, and whether the case duplicates another |
| [`accepted.json`](accepted.json) | the cases counted: every answer yes, one per group of duplicates (`python tools/blind_coverage.py accept reviews.json --out accepted.json`) |
| [`labels.json`](labels.json) | the labels given after the replay, by agents that could see it: the benchmark family whose trap the case repeats (empty for a new trap), whether a block named the error of the case, and whether a block on the clean command flagged a hazard that is there |
| [`briefs.md`](briefs.md) | what the authors, the reviewers and the labellers were told |

A case is only as good as its author: the descriptions are the authors' words, and the reviewers noted small slips in some of them
(for example `seq-10` and `seq-12`: the log line and a shift they describe differ from what the program prints). The commands and
the observations were run, not described.

To run the cases against the layer, with the programs of the cases on `PATH` (see [Run and reproduce](../reproduce.md#two-later-checks)):

```bash
python tools/blind_coverage.py verify docs/blind-cases --out verified.json
python tools/blind_coverage.py replay docs/blind-cases --verified verified.json --out replay.json
python tools/blind_coverage.py summary replay.json --accepted docs/blind-cases/accepted.json --labels docs/blind-cases/labels.json
```
