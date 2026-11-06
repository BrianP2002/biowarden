"""Ways an agent types one command, for the exposure tests of F53-F76 (a helper, not a test module).

The exposure checks of the variant and interval packs read text. `typed_forms` rewrites a command
into the forms that used to hide it from them: one option per line joined with a backslash, its
files addressed under /work (the workspace is the working directory in the sandbox), its file names
quoted, and, for a plain command, a Python `subprocess.run` argument list. A trapped command must
stay trapped and a safe one safe in every form.
"""

from __future__ import annotations

import re
import shlex

FILE_NAME = re.compile(
    r"(?<=\s)([A-Za-z][\w.-]*\.(?:vcf\.gz|vcf|bed|bam|tsv|txt|narrowPeak|chain|bw|bedgraph|fa))"
    r"(?=\s|$)"
)


def typed_forms(command: str) -> list[str]:
    forms = [
        command.replace(" -", " \\\n  -"),
        FILE_NAME.sub(r"/work/\1", command),
        FILE_NAME.sub(r"'\1'", command),
    ]
    if not re.search(r"[|;&<>$()]", command):
        argv = shlex.split(command)
        if '"' not in repr(argv):
            forms.append(f'python3 -c "import subprocess; subprocess.run({argv!r})"')
    return forms
