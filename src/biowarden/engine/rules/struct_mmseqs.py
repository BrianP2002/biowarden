"""Rule mmseqs.cluster_coverage (slot coverage, family F94): `mmseqs easy-cluster` needs 80%
coverage of both sequences by default, so partial sequences never join their parent's cluster.

The trap (MMseqs2 defaults: `-c 0.8 --cov-mode 0` for clustering, `--min-seq-id 0.0`). Verified
here (MMseqs2 18.8cc5c, `--threads 1 -v 1`, exit 0, empty stderr) on generated sets of 10-14 protein
families (2-4 members at 98% identity) plus 7-15 fragments, each an exact piece of 35-45% of a
member:
  easy-cluster --min-seq-id 0.9                      families + fragments clusters (all 30 sets)
  easy-cluster --min-seq-id 0.9 --cov-mode 1         the number of families (all 30 sets)
  easy-cluster --min-seq-id 0.9 -c 0.3 (or 0.1, 0.0) a fragment can become the representative of a
                                                     family, and a second fragment from another
                                                     region of the same protein cannot join it: two
                                                     clusters too many (-c 0.0: 25 of 30 sets)
  the same sets without fragments: families clusters with any of the settings.
`--min-seq-id 0.9` alone therefore does not mean "clusters at 90% identity" for a set that holds
partial sequences: identity is measured over the alignment, the coverage rule sits on top.

What the rule does. A CALL rule on `mmseqs` (subcommands easy-cluster, cluster, easy-linclust,
linclust), tier T1. It reads the FASTA file named by the call (the first non-option argument after
the subcommand; a database prefix instead of a FASTA file is not analysed and passes) and looks for
fragments: a sequence at least 10% shorter than another whose 5-mers are at least 50% contained in
it. With no fragment the rule is silent. With fragments:
  `--cov-mode 1` or `--cov-mode 2`        -> pass (coverage of the member or of the representative)
  declared fragment_policy=separate        -> pass (the caller wants fragments apart)
  declared with_parent with another mode   -> fail (declared_mismatch)
  no declaration                           -> fail (fragment_policy_undeclared)
Declaration `fragment_policy` (optional): with_parent | separate. `unknown`: an unreadable FASTA.

Not seen: the same sets given as an mmseqs database (createdb output), protein sets over 2000
sequences (only the first 2000 are compared), nucleotide input, and cd-hit / vsearch.
"""

from __future__ import annotations

from collections import Counter

from biowarden.engine.argv import option_values
from biowarden.engine.call import Call
from biowarden.engine.checks import CheckResult, Ctx, Match, checker, passed
from biowarden.engine.rules.struct_common import file_text, positionals

K = 5
MAX_SEQS = 2000
CONTAINED = 0.5
SHORTER = 0.9
VALUE_OPTIONS = frozenset(
    "-c -e -s --min-seq-id --cov-mode --cluster-mode --threads --split-memory-limit "
    "--max-seqs --cluster-steps -v --kmer-per-seq --mask --similarity-type".split()
)


def read_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith(">"):
            records.append((line[1:].split()[0] if line[1:].split() else "", []))
        elif records:
            records[-1][1].append(line.strip())
    return [(name, "".join(parts).upper()) for name, parts in records][:MAX_SEQS]


def fragments(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(fragment, parent) pairs: a sequence much shorter than another that contains its 5-mers."""
    kmers = [{s[i : i + K] for i in range(len(s) - K + 1)} for _, s in records]
    postings: dict[str, list[int]] = {}
    for idx, ks in enumerate(kmers):
        for k in ks:
            postings.setdefault(k, []).append(idx)
    found: list[tuple[str, str]] = []
    for i, (name, seq) in enumerate(records):
        if len(kmers[i]) == 0:
            continue
        shared: Counter[int] = Counter()
        for k in kmers[i]:
            shared.update(j for j in postings[k] if j != i)
        for j, n in shared.most_common():
            if n / len(kmers[i]) < CONTAINED:
                break
            if len(seq) <= SHORTER * len(records[j][1]):
                found.append((name, records[j][0]))
                break
    return found


@checker("mmseqs_cluster_coverage")
def mmseqs_cluster_coverage(call: Call, match: Match, ctx: Ctx) -> CheckResult:
    rule = match.rule
    assert rule.declare is not None
    key = rule.declare.key
    argv = call.argv[1:] if call.argv else ()
    modes = option_values(argv, "--cov-mode")
    if modes and modes[-1] in ("1", "2"):
        return passed()
    declared = call.declared.get(key)
    if declared == "separate":
        return passed()
    args = positionals(argv, VALUE_OPTIONS)
    if not args:
        return passed()
    path = ctx.resolve(args[0])
    if not path.is_file():
        return passed()
    text = file_text(ctx, path)
    if text is None:
        return CheckResult("unknown", "unreadable_fasta", f"could not read {args[0]}.")
    if not text.lstrip().startswith(">"):
        return passed()  # a database, not a FASTA file
    found = fragments(read_fasta(text))
    if not found:
        return passed()
    child, parent = found[0]
    mode = modes[-1] if modes else "0 (the default)"
    facts = (
        f"{args[0]} holds {len(found)} fragment(s) such as {child}, a piece of {parent} (at least "
        f"10% shorter, at least {int(CONTAINED * 100)}% of its {K}-mers inside it), and mmseqs "
        f"cluster demands that the coverage rule (-c, default 0.8, --cov-mode {mode}) holds for "
        "both sequences, so a fragment stays out of its parent's cluster."
    )
    if declared == "with_parent":
        return CheckResult(
            "fail",
            "declared_mismatch",
            f"you declared {key}=with_parent but --cov-mode {mode} keeps fragments apart. " + facts,
            expected="with_parent",
            got="separate",
            fix=("add --cov-mode 1 (coverage of the member, so a fragment can join its parent)",),
        )
    return CheckResult(
        "fail",
        "fragment_policy_undeclared",
        facts,
        expected=list(rule.declare.values),
        got=None,
        fix=(
            f"add --cov-mode 1 and --declare {key}=with_parent so fragments join their parent",
            f"or --declare {key}=separate if each fragment should be a cluster of its own",
        ),
    )
