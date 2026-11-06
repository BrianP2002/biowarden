"""Split a bash command string, as an LLM agent writes it, into simple commands.

Used offline to replay agent sessions through the engine: one `bash` tool call such as
``cd /work && head -3 expr.tsv | awk -F'\\t' '{print $2}' > out.txt`` becomes the individual tool
invocations the engine would have seen (``cd``, ``head``, ``awk``), each with its tool basename, its
argv with quotes removed (an awk program or a ``python3 -c`` string stays ONE argument) and the
directory in effect.

This is a tolerant reader, not a bash implementation. It understands:
  quoting       '..', "..", \\x, $'..', $"..", line continuations, # comments
  separators    |  |&  &&  ||  ;  &  newline, (subshells) and { groups }
  substitution  $(..), `..`, <(..), >(..) (the inner commands are returned too, ``nested=True``;
                the word itself stays in the argument as written)
  here-documents  <<EOF, <<-EOF, <<'EOF', here-strings <<<; the body is returned apart from argv
  redirections  returned as ``Redirect`` records, never mixed into argv
  cd tracking   cd, pushd, popd, ``cd -``; changes inside ( ), $( ), a pipeline or ``&`` do not leak
  variables     NAME=value and ``for NAME in a b c; do ...; done`` over literal words (the loop body
                is returned once per word, up to MAX_LOOP_ITEMS); $NAME / ${NAME} of a known value
                are substituted outside single quotes
  wrappers      time, nohup, env, sudo, timeout, nice, ionice, stdbuf, setsid, command, exec,
                xargs are removed so ``tool`` is the program that really runs; ``bash -c '..'``,
                ``sh -c`` and ``eval`` are kept AND their text is split again
  control flow  if/while/until/case/functions are flattened: every command in every branch is
                returned, as if all of them ran (an over-approximation)

Known limits (see ``Unparsed`` for what is flagged, the rest is silent):
  * Globs, brace expansion ({a,b}), ~ and arithmetic ($((..))) are not expanded: they stay literal.
  * Variables set by anything other than a plain NAME=literal (read, command substitution, arrays,
    ${x:-d}, export of an unknown value) stay unexpanded, and the cwd becomes ``None`` after a cd to
    such a path; a later absolute cd restores it.
  * A for loop over a glob or a $(..) result is returned once, with the variable unexpanded.
  * Functions are flattened, not called; aliases and sourced files are invisible.
  * ``find -exec``, ``parallel``, ``ssh host cmd`` and ``docker run img cmd`` are not unwrapped.
  * The body of an unquoted here-document is not scanned for $(..); `` `..` `` inside it is ignored.
  * Anything that confuses the reader (an unterminated quote or $(, a stray ')', a here-document
    without terminator, a for loop without done, nesting deeper than MAX_DEPTH, more than
    MAX_COMMANDS commands) is reported in ``Split.unparsed`` and the rest is still returned.

``split_command`` never raises: an internal error is reported as ``Unparsed`` and what was
collected so far is returned.
"""

from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field

MAX_DEPTH = 6
MAX_COMMANDS = 2000
MAX_LOOP_ITEMS = 24
MAX_TOKENS = 200_000


@dataclass(frozen=True)
class Redirect:
    """One redirection: op is `<`, `>`, `>>`, `<<`, `<<<`, `>&`, `&>` ...; fd the leading number."""

    op: str
    target: str
    fd: str = ""


@dataclass(frozen=True)
class Unparsed:
    reason: str
    text: str


@dataclass(frozen=True)
class SimpleCommand:
    tool: str  # basename of the program that really runs (wrappers removed)
    argv: tuple[str, ...]  # arguments after the tool, quotes removed
    cwd: str | None  # directory in effect when the command starts; None when it cannot be known
    raw: str  # the source text of the command
    redirects: tuple[Redirect, ...] = ()
    heredoc: str | None = None  # body of the first here-document attached to the command
    env: tuple[tuple[str, str], ...] = ()  # NAME=value prefixes (also those of env/sudo)
    wrappers: tuple[str, ...] = ()  # wrappers removed, outermost first
    nested: bool = False  # found inside $(..), `..`, <(..), bash -c or eval
    program: str = ""  # the program word as written, wrappers removed (a path stays a path)


@dataclass(frozen=True)
class Split:
    commands: tuple[SimpleCommand, ...]
    unparsed: tuple[Unparsed, ...]
    cwd: str | None  # directory in effect after the last command


# -- lexer ---------------------------------------------------------------------------------------

_OPS = (
    "<<<",
    "<<-",
    "&>>",
    ";;&",
    "<<",
    ">>",
    "&&",
    "||",
    "|&",
    ";;",
    ";&",
    ">&",
    "<&",
    "<>",
    ">|",
    "&>",
    "|",
    "&",
    ";",
    "(",
    ")",
    "<",
    ">",
)
_REDIR_OPS = frozenset(
    {"<", ">", ">>", "<<", "<<-", "<<<", ">&", "<&", "<>", ">|", "&>", "&>>"},
)
_HEREDOC_OPS = frozenset({"<<", "<<-"})
_WORD_SPECIAL = re.compile(r"""[ \t\n;&|()<>'"\\`$]""")
_DQ_SPECIAL = re.compile(r'["\\`$]')
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "v": "\v",
}


@dataclass
class _Tok:
    kind: str  # "word" | "op" | "redir" | "arith"
    text: str  # raw source text of the token
    start: int
    end: int
    segs: list[tuple[str, bool]] = field(default_factory=list)  # (text, may take $VAR)
    subs: list[str] = field(default_factory=list)  # inner command text found in the word
    quoted: bool = False
    fd: str = ""
    delim: str = ""
    body: str | None = None


def _ansi_c(text: str) -> str:
    """Decode the body of $'..'."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\" or i + 1 >= len(text):
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1]
        if nxt in _ESCAPES:
            out.append(_ESCAPES[nxt])
            i += 2
        elif nxt == "x" and re.match(r"[0-9a-fA-F]{1,2}", text[i + 2 : i + 4]):
            digits = re.match(r"[0-9a-fA-F]{1,2}", text[i + 2 : i + 4]).group(0)  # type: ignore[union-attr]
            out.append(chr(int(digits, 16)))
            i += 2 + len(digits)
        elif nxt in "01234567":
            digits = re.match(r"[0-7]{1,3}", text[i + 1 : i + 4]).group(0)  # type: ignore[union-attr]
            out.append(chr(int(digits, 8)))
            i += 1 + len(digits)
        elif nxt in "uU" and re.match(r"[0-9a-fA-F]{1,8}", text[i + 2 : i + 10]):
            digits = re.match(r"[0-9a-fA-F]{1,8}", text[i + 2 : i + 10]).group(0)  # type: ignore[union-attr]
            try:
                out.append(chr(int(digits, 16)))
            except (ValueError, OverflowError):
                out.append(text[i : i + 2 + len(digits)])
            i += 2 + len(digits)
        else:
            out.append(ch)
            i += 1
    return "".join(out)


class _Lexer:
    def __init__(self, text: str) -> None:
        self.s = text
        self.n = len(text)
        self.i = 0
        self.toks: list[_Tok] = []
        self.unparsed: list[Unparsed] = []
        self.pending: list[tuple[_Tok, bool]] = []  # here-documents waiting for their body
        self.fd = ""

    # -- driver ------------------------------------------------------------------------------

    def run(self) -> list[_Tok]:
        s, n = self.s, self.n
        while self.i < n and len(self.toks) < MAX_TOKENS:
            c = s[self.i]
            if c in " \t":
                self.i += 1
            elif c == "\\" and s.startswith("\n", self.i + 1):
                self.i += 2
            elif c == "\n":
                self.toks.append(_Tok("op", "\n", self.i, self.i + 1))
                self.i += 1
                self._heredoc_bodies()
            elif c == "#":
                j = s.find("\n", self.i)
                self.i = n if j < 0 else j
            elif c in "|&;()<>":
                self._operator()
            else:
                self._word()
        if self.i < n:
            self.unparsed.append(Unparsed("too many tokens", s[self.i : self.i + 80]))
        return self.toks

    # -- operators and here-documents --------------------------------------------------------

    def _operator(self) -> None:
        s, i = self.s, self.i
        c = s[i]
        if c in "<>" and s.startswith("(", i + 1):
            self._word()  # process substitution is part of a word
            return
        if c == "(" and s.startswith("(", i + 1):
            end = self._arith_end(i + 2)
            if end is not None:
                self.toks.append(_Tok("arith", s[i:end], i, end))
                self.i = end
                return
        op = next((o for o in _OPS if s.startswith(o, i)), c)
        self.i = i + len(op)
        kind = "redir" if op in _REDIR_OPS else "op"
        tok = _Tok(kind, op, i, self.i, fd=self.fd if kind == "redir" else "")
        self.fd = ""
        self.toks.append(tok)
        if op in _HEREDOC_OPS:
            self._heredoc_delimiter(tok, op == "<<-")

    def _heredoc_delimiter(self, tok: _Tok, strip_tabs: bool) -> None:
        s, n = self.s, self.n
        while self.i < n and s[self.i] in " \t":
            self.i += 1
        before = len(self.toks)
        if self.i < n and s[self.i] not in "\n;&|()<>":
            self._word()
        if len(self.toks) > before and self.toks[-1].kind == "word":
            word = self.toks.pop()
            tok.delim = "".join(text for text, _ in word.segs)
            tok.end = word.end
            self.pending.append((tok, strip_tabs))
        else:
            self.unparsed.append(
                Unparsed("here-document without delimiter", s[tok.start : tok.end])
            )

    def _heredoc_bodies(self) -> None:
        s, n = self.s, self.n
        for tok, strip_tabs in self.pending:
            lines: list[str] = []
            found = False
            while self.i < n:
                j = s.find("\n", self.i)
                line = s[self.i : n if j < 0 else j]
                self.i = n if j < 0 else j + 1
                shown = line.lstrip("\t") if strip_tabs else line
                if shown == tok.delim:
                    found = True
                    break
                lines.append(shown)
            if not found:
                self.unparsed.append(Unparsed("here-document not terminated", tok.delim))
            tok.body = "\n".join(lines) + ("\n" if lines else "")
        self.pending = []

    # -- words -------------------------------------------------------------------------------

    def _word(self) -> None:
        s, n = self.s, self.n
        start = self.i
        segs: list[tuple[str, bool]] = []
        subs: list[str] = []
        quoted = False
        while self.i < n:
            m = _WORD_SPECIAL.search(s, self.i)
            end = m.start() if m else n
            if end > self.i:
                segs.append((s[self.i : end], True))
                self.i = end
            if self.i >= n:
                break
            c = s[self.i]
            if c in " \t\n;&|()":
                break
            if c in "<>":
                if not s.startswith("(", self.i + 1):
                    break
                close = self._match_paren(self.i + 1)
                if close is None:
                    self.unparsed.append(
                        Unparsed("unterminated process substitution", s[self.i :][:80])
                    )
                    segs.append((s[self.i :], False))
                    self.i = n
                else:
                    segs.append((s[self.i : close + 1], False))
                    subs.append(s[self.i + 2 : close])
                    self.i = close + 1
            elif c == "\\":
                nxt = s[self.i + 1] if self.i + 1 < n else ""
                if nxt == "\n":
                    self.i += 2
                elif nxt:
                    segs.append((nxt, False))
                    quoted = True
                    self.i += 2
                else:
                    segs.append(("\\", False))
                    self.i += 1
            elif c == "'":
                quoted = True
                close = s.find("'", self.i + 1)
                if close < 0:
                    self.unparsed.append(Unparsed("unterminated single quote", s[self.i :][:80]))
                    segs.append((s[self.i + 1 :], False))
                    self.i = n
                else:
                    segs.append((s[self.i + 1 : close], False))
                    self.i = close + 1
            elif c == '"':
                quoted = True
                self._dquote(segs, subs)
            elif c == "`":
                self._backtick(segs, subs)
            else:  # "$"
                quoted = self._dollar(segs, subs, dq=False) or quoted
        merged: list[tuple[str, bool]] = []
        for text, flag in segs:
            if merged and merged[-1][1] == flag:
                merged[-1] = (merged[-1][0] + text, flag)
            else:
                merged.append((text, flag))
        raw = s[start : self.i]
        if (
            not quoted
            and raw.isdigit()
            and self.i < n
            and s[self.i] in "<>"
            and not s.startswith("(", self.i + 1)
        ):
            self.fd = raw  # `2>&1`: the number belongs to the redirection that follows
            return
        self.toks.append(_Tok("word", raw, start, self.i, merged, subs, quoted))

    def _dquote(self, segs: list[tuple[str, bool]], subs: list[str]) -> None:
        s, n = self.s, self.n
        self.i += 1
        while self.i < n:
            m = _DQ_SPECIAL.search(s, self.i)
            end = m.start() if m else n
            if end > self.i:
                segs.append((s[self.i : end], True))
                self.i = end
            if self.i >= n:
                break
            c = s[self.i]
            if c == '"':
                self.i += 1
                return
            if c == "\\":
                nxt = s[self.i + 1] if self.i + 1 < n else ""
                if nxt in ("$", "`", '"', "\\"):
                    segs.append((nxt, False))
                    self.i += 2
                elif nxt == "\n":
                    self.i += 2
                else:
                    segs.append(("\\", False))
                    self.i += 1
            elif c == "`":
                self._backtick(segs, subs)
            else:
                self._dollar(segs, subs, dq=True)
        self.unparsed.append(Unparsed("unterminated double quote", s[-80:]))

    def _backtick(self, segs: list[tuple[str, bool]], subs: list[str]) -> None:
        s, n = self.s, self.n
        j = self.i + 1
        inner: list[str] = []
        while j < n and s[j] != "`":
            if s[j] == "\\" and j + 1 < n and s[j + 1] in "$`\\":
                inner.append(s[j + 1])
                j += 2
            else:
                inner.append(s[j])
                j += 1
        if j >= n:
            self.unparsed.append(Unparsed("unterminated backtick", s[self.i :][:80]))
            segs.append((s[self.i :], False))
            self.i = n
            return
        segs.append((s[self.i : j + 1], False))
        subs.append("".join(inner))
        self.i = j + 1

    def _dollar(self, segs: list[tuple[str, bool]], subs: list[str], dq: bool) -> bool:
        """Handle `$` at self.i; True when the construct is a quoting form ($'..' or $"..")."""
        s, n, i = self.s, self.n, self.i
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt == "(":
            if s.startswith("(", i + 2):
                end = self._arith_end(i + 3)
                if end is not None:
                    segs.append((s[i:end], False))
                    self.i = end
                    return False
            close = self._match_paren(i + 1)
            if close is None:
                self.unparsed.append(Unparsed("unterminated $(", s[i : i + 80]))
                segs.append((s[i:], False))
                self.i = n
                return False
            segs.append((s[i : close + 1], False))
            subs.append(s[i + 2 : close])
            self.i = close + 1
            return False
        if nxt == "{":
            close = self._match_brace(i + 1)
            if close is None:
                self.unparsed.append(Unparsed("unterminated ${", s[i : i + 80]))
                segs.append((s[i:], False))
                self.i = n
                return False
            segs.append((s[i : close + 1], True))
            self.i = close + 1
            return False
        if nxt == "'" and not dq:
            j = i + 2
            while j < n and s[j] != "'":
                j += 2 if s[j] == "\\" else 1
            if j >= n:
                self.unparsed.append(Unparsed("unterminated $'", s[i : i + 80]))
                segs.append((_ansi_c(s[i + 2 :]), False))
                self.i = n
            else:
                segs.append((_ansi_c(s[i + 2 : j]), False))
                self.i = j + 1
            return True
        if nxt == '"' and not dq:
            self.i += 1  # `$"..."` is a double-quoted string; the main loop reads it
            return False
        segs.append(("$", True))
        self.i += 1
        return False

    # -- helpers -----------------------------------------------------------------------------

    def _arith_end(self, start: int) -> int | None:
        """Index after the `))` that closes an arithmetic expression whose `((` ends at start."""
        s, n = self.s, self.n
        depth, k = 2, start
        while k < n:
            ch = s[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return k + 1 if s[k - 1] == ")" else None
            k += 1
        return None

    def _skip_dq(self, k: int) -> int:
        """Index after the double-quoted string that starts at k, or -1."""
        s, n = self.s, self.n
        k += 1
        while k < n:
            ch = s[k]
            if ch == "\\":
                k += 2
            elif ch == '"':
                return k + 1
            elif ch == "$" and s.startswith("(", k + 1):
                close = self._match_paren(k + 1)
                if close is None:
                    return -1
                k = close + 1
            else:
                k += 1
        return -1

    def _match_paren(self, j: int) -> int | None:
        """Index of the `)` matching the `(` at s[j], skipping quotes, or None."""
        s, n = self.s, self.n
        depth, k = 0, j
        while k < n:
            ch = s[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "'":
                end = s.find("'", k + 1)
                if end < 0:
                    return None
                k = end + 1
                continue
            if ch == '"':
                k = self._skip_dq(k)
                if k < 0:
                    return None
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return k
            k += 1
        return None

    def _match_brace(self, j: int) -> int | None:
        s, n = self.s, self.n
        depth, k = 0, j
        while k < n:
            ch = s[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "'":
                end = s.find("'", k + 1)
                if end < 0:
                    return None
                k = end + 1
                continue
            if ch == '"':
                k = self._skip_dq(k)
                if k < 0:
                    return None
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return k
            k += 1
        return None


# -- shell state ---------------------------------------------------------------------------------


class _Sh:
    """The part of shell state that matters here: directory and plain variables."""

    __slots__ = ("cwd", "prev", "env")

    def __init__(self, cwd: str | None, prev: str | None = None, env: dict[str, str] | None = None):
        self.cwd = cwd
        self.prev = prev
        self.env = {} if env is None else env

    def copy(self) -> _Sh:
        return _Sh(self.cwd, self.prev, dict(self.env))


_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\+?)=")
_NAME_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)


def _expand(tok: _Tok, env: dict[str, str]) -> str:
    out: list[str] = []
    for text, expandable in tok.segs:
        if expandable and env and "$" in text:
            text = _VAR.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), text)
        out.append(text)
    return "".join(out)


def _cd(sh: _Sh, argv: list[str]) -> None:
    args = [a for a in argv if a == "-" or not a.startswith("-")]
    target = args[0] if args else "~"
    here = sh.cwd
    if target == "-":
        new = sh.prev
    elif any(ch in target for ch in "$`*?[{"):
        new = None
    elif target.startswith("/") or target.startswith("~"):
        new = posixpath.normpath(target) if target.startswith("/") else target
    elif here is None:
        new = None
    else:
        new = posixpath.normpath(posixpath.join(here, target))
    sh.prev, sh.cwd = here, new


# -- wrappers ------------------------------------------------------------------------------------


def _skip_options(items: list[str], with_value: frozenset[str] = frozenset()) -> list[str]:
    """Drop leading -x / --long options (and the values of those in with_value)."""
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "--":
            return items[i + 1 :]
        if not tok.startswith("-") or tok == "-":
            break
        i += 2 if tok in with_value else 1
    return items[i:]


def _unwrap_env(items: list[str], env: list[tuple[str, str]]) -> list[str] | None:
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "--":
            i += 1
            break
        if tok in ("-u", "-C", "-S", "--unset", "--chdir"):
            i += 2
        elif tok.startswith("-") and tok != "-":
            i += 1
        elif _NAME_ASSIGN.match(tok):
            m = _NAME_ASSIGN.match(tok)
            assert m is not None
            env.append((m.group(1), m.group(2)))
            i += 1
        else:
            break
    return items[i:]


def _unwrap_timeout(items: list[str], env: list[tuple[str, str]]) -> list[str] | None:
    rest = _skip_options(items, frozenset({"-k", "-s", "--kill-after", "--signal"}))
    return rest[1:] if rest else None  # the first positional is the duration


def _unwrap_xargs(items: list[str], env: list[tuple[str, str]]) -> list[str] | None:
    valued = frozenset({"-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a", "-J", "-R", "-S", "-l"})
    rest = _skip_options(items, valued)
    return rest or None


def _unwrap_sudo(items: list[str], env: list[tuple[str, str]]) -> list[str] | None:
    valued = frozenset({"-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-U"})
    rest = _skip_options(items, valued)
    while rest and _NAME_ASSIGN.match(rest[0]):
        m = _NAME_ASSIGN.match(rest[0])
        assert m is not None
        env.append((m.group(1), m.group(2)))
        rest = rest[1:]
    return rest or None


def _unwrap_command(items: list[str], env: list[tuple[str, str]]) -> list[str] | None:
    if items and items[0] in ("-v", "-V"):
        return None  # a lookup, not an execution
    return _skip_options(items) or None


def _unwrap_plain(valued: frozenset[str] = frozenset()):  # type: ignore[no-untyped-def]
    def unwrap(items: list[str], env: list[tuple[str, str]]) -> list[str] | None:
        return _skip_options(items, valued) or None

    return unwrap


_WRAPPERS = {
    "time": _unwrap_plain(frozenset({"-f", "-o", "--format", "--output"})),
    "nohup": _unwrap_plain(),
    "exec": _unwrap_plain(frozenset({"-a"})),
    "builtin": _unwrap_plain(),
    "setsid": _unwrap_plain(),
    "stdbuf": _unwrap_plain(frozenset({"-i", "-o", "-e"})),
    "nice": _unwrap_plain(frozenset({"-n", "--adjustment"})),
    "ionice": _unwrap_plain(frozenset({"-c", "-n", "-p", "-P", "-u"})),
    "command": _unwrap_command,
    "env": _unwrap_env,
    "sudo": _unwrap_sudo,
    "timeout": _unwrap_timeout,
    "xargs": _unwrap_xargs,
}
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
_STATE_BUILTINS = frozenset({"export", "declare", "typeset", "local", "readonly"})
_KEYWORDS = frozenset(
    {"if", "then", "elif", "else", "fi", "while", "until", "select", "do", "done", "!", "{", "}"}
)
_SEPARATORS = frozenset({"\n", ";", "&&", "||", "|", "|&", "&", ";;", ";&", ";;&"})
_CMD_START_PREV = frozenset({"\n", ";", "&&", "||", "|", "|&", "&", "(", ";;"})
_LOOP_OPENERS = frozenset({"for", "while", "until", "select"})


def _unwrap(argv: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    wrappers: list[str] = []
    env: list[tuple[str, str]] = []
    items = list(argv)
    while items:
        name = posixpath.basename(items[0])
        rule = _WRAPPERS.get(name)
        if rule is None:
            break
        rest = rule(items[1:], env)
        if not rest:
            break
        wrappers.append(name)
        items = rest
    return items, wrappers, env


# -- parser --------------------------------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.commands: list[SimpleCommand] = []
        self.unparsed: list[Unparsed] = []

    @property
    def full(self) -> bool:
        return len(self.commands) >= MAX_COMMANDS


@dataclass
class _Cur:
    """The simple command being collected."""

    words: list[_Tok] = field(default_factory=list)
    assigns: list[_Tok] = field(default_factory=list)
    redirs: list[tuple[_Tok, _Tok | None]] = field(default_factory=list)
    start: int | None = None
    end: int = 0

    def touch(self, tok: _Tok) -> None:
        if self.start is None:
            self.start = tok.start
        self.end = max(self.end, tok.end)

    @property
    def empty(self) -> bool:
        return not (self.words or self.assigns or self.redirs)


class _Parser:
    def __init__(self, ctx: _Ctx, src: str, depth: int, nested: bool) -> None:
        self.ctx = ctx
        self.src = src
        self.depth = depth
        self.nested = nested

    # -- driver ------------------------------------------------------------------------------

    def run(self, toks: list[_Tok], sh: _Sh) -> _Sh:
        ctx = self.ctx
        stack: list[_Sh] = []
        cases: list[bool] = []  # True while the patterns of that case are being read
        cur = _Cur()
        piped = False  # the previous separator was a pipe
        i, n = 0, len(toks)
        while i < n and not ctx.full:
            t = toks[i]
            if cases and cases[-1]:  # case pattern: skip up to the `)` that ends it
                if t.kind == "op" and t.text == ")":
                    cases[-1] = False
                elif t.kind == "word" and t.text == "esac" and not t.quoted:
                    cases.pop()
                i += 1
                continue
            if t.kind == "arith":
                i += 1
                continue
            if t.kind == "op":
                if t.text in _SEPARATORS:
                    self._flush(cur, sh, scoped=piped or t.text in ("|", "|&", "&"))
                    cur = _Cur()
                    piped = t.text in ("|", "|&")
                    if cases and t.text in (";;", ";&", ";;&"):
                        cases[-1] = True
                elif t.text == "(":
                    self._flush(cur, sh, scoped=piped)
                    cur = _Cur()
                    stack.append(sh)
                    sh = sh.copy()
                elif t.text == ")":
                    self._flush(cur, sh, scoped=piped)
                    cur = _Cur()
                    if stack:
                        sh = stack.pop()
                    else:
                        ctx.unparsed.append(Unparsed("unmatched )", self.src[t.start :][:60]))
                i += 1
                continue
            if t.kind == "redir":
                cur.touch(t)
                if t.text in _HEREDOC_OPS:
                    cur.redirs.append((t, None))
                    i += 1
                elif i + 1 < n and toks[i + 1].kind == "word":
                    cur.touch(toks[i + 1])
                    cur.redirs.append((t, toks[i + 1]))
                    i += 2
                else:
                    ctx.unparsed.append(Unparsed("redirection without a target", t.text))
                    i += 1
                continue
            # a word
            if cur.empty and not t.quoted:
                skipped = self._keyword(toks, i, sh, cases)
                if skipped is not None:
                    i = skipped
                    continue
            if cur.empty and i + 2 < n and self._is_function_def(toks, i):
                i += 3
                continue
            cur.touch(t)
            if not cur.words and _ASSIGN.match(t.text) and t.segs and t.segs[0][1]:
                cur.assigns.append(t)
            else:
                cur.words.append(t)
            i += 1
        self._flush(cur, sh, scoped=piped)
        if stack:  # unclosed ( ... keep the outer state
            sh = stack[0]
            ctx.unparsed.append(Unparsed("unclosed (", f"{len(stack)} open"))
        return sh

    # -- keywords and compound commands ------------------------------------------------------

    @staticmethod
    def _is_function_def(toks: list[_Tok], i: int) -> bool:
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        return (
            a.kind == "word"
            and not a.quoted
            and b.kind == "op"
            and b.text == "("
            and c.kind == "op"
            and c.text == ")"
        )

    def _keyword(self, toks: list[_Tok], i: int, sh: _Sh, cases: list[bool]) -> int | None:
        """Index to continue at when toks[i] is a reserved word handled here, else None."""
        word = toks[i].text
        n = len(toks)
        if word in _KEYWORDS:
            return i + 1
        if word == "for":
            return self._for_loop(toks, i, sh)
        if word == "case":
            j = i + 1
            while j < n and not (toks[j].kind == "word" and toks[j].text == "in"):
                self._emit_subs(toks[j], sh)
                j += 1
            cases.append(True)
            return min(j + 1, n)
        if word == "esac":
            if cases:
                cases.pop()
            return i + 1
        if word == "function":
            j = i + 2
            if j + 1 < n and toks[j].text == "(" and toks[j + 1].text == ")":
                j += 2
            return min(j, n)
        if word == "[[":
            j = i + 1
            while j < n and not (toks[j].kind == "word" and toks[j].text == "]]"):
                self._emit_subs(toks[j], sh)
                j += 1
            return min(j + 1, n)
        return None

    def _emit_subs(self, tok: _Tok, sh: _Sh) -> None:
        for inner in tok.subs:
            _split(inner, sh.copy(), self.ctx, self.depth + 1, True)

    @staticmethod
    def _find_done(toks: list[_Tok], start: int) -> int | None:
        depth = 1
        for k in range(start, len(toks)):
            t = toks[k]
            if t.kind != "word" or t.quoted:
                continue
            prev = toks[k - 1] if k > start else None
            at_start = (
                prev is None
                or (prev.kind == "op" and prev.text in _CMD_START_PREV)
                or (prev.kind == "word" and prev.text in ("do", "then", "else", "{", "!"))
            )
            if not at_start:
                continue
            if t.text in _LOOP_OPENERS:
                depth += 1
            elif t.text == "done":
                depth -= 1
                if depth == 0:
                    return k
        return None

    def _for_loop(self, toks: list[_Tok], i: int, sh: _Sh) -> int:
        n = len(toks)
        j = i + 1
        name: str | None = None
        values: list[str | None] = [None]
        if j < n and toks[j].kind == "arith":
            j += 1
        elif j < n and toks[j].kind == "word":
            name = toks[j].text
            j += 1
            if j < n and toks[j].kind == "word" and toks[j].text == "in":
                j += 1
                items: list[_Tok] = []
                while j < n and not (toks[j].kind == "op" and toks[j].text in (";", "\n")):
                    if toks[j].kind == "word":
                        items.append(toks[j])
                    j += 1
                for item in items:
                    self._emit_subs(item, sh)
                if items and not any(item.subs for item in items):
                    values = [_expand(item, sh.env) for item in items][:MAX_LOOP_ITEMS]
                    if len(items) > MAX_LOOP_ITEMS:
                        self.ctx.unparsed.append(
                            Unparsed("for list truncated", f"{len(items)} items")
                        )
        while j < n and toks[j].kind == "op" and toks[j].text in (";", "\n"):
            j += 1
        if not (j < n and toks[j].kind == "word" and toks[j].text == "do"):
            self.ctx.unparsed.append(Unparsed("for without do", self.src[toks[i].start :][:60]))
            return min(j, n)
        body_start = j + 1
        done = self._find_done(toks, body_start)
        if done is None:
            self.ctx.unparsed.append(Unparsed("for without done", self.src[toks[i].start :][:60]))
            return body_start  # keep reading the body as plain commands
        body = toks[body_start:done]
        for value in values:
            if self.ctx.full:
                break
            if name is not None:
                if value is None:
                    sh.env.pop(name, None)
                else:
                    sh.env[name] = value
            self.run(body, sh)
        return done + 1

    # -- one simple command ------------------------------------------------------------------

    def _flush(self, cur: _Cur, sh: _Sh, scoped: bool) -> None:
        if cur.empty or self.ctx.full:
            return
        for tok in [*cur.assigns, *cur.words, *(t for _, t in cur.redirs if t is not None)]:
            self._emit_subs(tok, sh)
        argv_all = [_expand(w, sh.env) for w in cur.words]
        if not argv_all:
            if not scoped:
                for tok in cur.assigns:
                    self._assign(tok, sh)
            return
        tool_argv, wrappers, wrap_env = _unwrap(argv_all)
        tool = posixpath.basename(tool_argv[0]) if tool_argv else ""
        if not tool:
            return
        argv = tool_argv[1:]
        env = [self._pair(tok, sh) for tok in cur.assigns] + wrap_env
        redirects: list[Redirect] = []
        heredoc: str | None = None
        for op_tok, target in cur.redirs:
            if op_tok.text in _HEREDOC_OPS:
                redirects.append(Redirect(op_tok.text, op_tok.delim, op_tok.fd))
                if heredoc is None:
                    heredoc = op_tok.body
            else:
                redirects.append(
                    Redirect(op_tok.text, _expand(target, sh.env) if target else "", op_tok.fd)
                )
        start = cur.start if cur.start is not None else 0
        command = SimpleCommand(
            tool,
            tuple(argv),
            sh.cwd,
            self.src[start : cur.end].strip(),
            tuple(redirects),
            heredoc,
            tuple(env),
            tuple(wrappers),
            self.nested,
            tool_argv[0],
        )
        self.ctx.commands.append(command)
        self._effects(tool, argv, sh, scoped)

    @staticmethod
    def _pair(tok: _Tok, sh: _Sh) -> tuple[str, str]:
        text = _expand(tok, sh.env)
        m = _NAME_ASSIGN.match(text)
        return (m.group(1), m.group(2)) if m else (text, "")

    @staticmethod
    def _assign(tok: _Tok, sh: _Sh) -> None:
        name, value = _Parser._pair(tok, sh)
        if "$" in value or "`" in value:
            sh.env.pop(name, None)  # the value is computed: unknown
        else:
            sh.env[name] = value

    def _effects(self, tool: str, argv: list[str], sh: _Sh, scoped: bool) -> None:
        if tool in _SHELLS:
            for k, arg in enumerate(argv):
                if arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:]:
                    if k + 1 < len(argv):
                        _split(argv[k + 1], sh.copy(), self.ctx, self.depth + 1, True)
                    return
                if not arg.startswith("-"):
                    return
            return
        if tool == "eval":
            _split(" ".join(argv), sh.copy(), self.ctx, self.depth + 1, True)
            return
        if scoped:
            return
        if tool in ("cd", "pushd"):
            _cd(sh, argv)
        elif tool == "popd":
            sh.prev, sh.cwd = sh.cwd, None
        elif tool in _STATE_BUILTINS:
            for arg in argv:
                m = _NAME_ASSIGN.match(arg)
                if m:
                    if "$" in m.group(2) or "`" in m.group(2):
                        sh.env.pop(m.group(1), None)
                    else:
                        sh.env[m.group(1)] = m.group(2)


def _split(text: str, sh: _Sh, ctx: _Ctx, depth: int, nested: bool) -> None:
    if depth > MAX_DEPTH:
        ctx.unparsed.append(Unparsed("nesting too deep", text[:80]))
        return
    lexer = _Lexer(text)
    toks = lexer.run()
    ctx.unparsed.extend(lexer.unparsed)
    _Parser(ctx, text, depth, nested).run(toks, sh)


def split_command(command: str, cwd: str | None = ".") -> Split:
    """Split a bash command string into simple commands. Never raises."""
    ctx = _Ctx()
    sh = _Sh(cwd)
    text = command if isinstance(command, str) else str(command)
    text = text.replace("\r\n", "\n")
    try:
        _split(text, sh, ctx, 0, False)
    except Exception as exc:  # the reader must never take the replay down
        ctx.unparsed.append(Unparsed(f"internal error {type(exc).__name__}", text[:200]))
    if ctx.full:
        ctx.unparsed.append(Unparsed("too many commands", f"stopped after {MAX_COMMANDS}"))
    return Split(tuple(ctx.commands), _compact(ctx.unparsed), sh.cwd)


def _compact(items: list[Unparsed], keep: int = 3) -> tuple[Unparsed, ...]:
    """At most `keep` reports per reason, then one line with the number of the others."""
    seen: Counter[str] = Counter()
    out: list[Unparsed] = []
    for item in items:
        seen[item.reason] += 1
        if seen[item.reason] <= keep:
            out.append(item)
    out += [Unparsed(f"{r} (+{n - keep} more)", "") for r, n in seen.items() if n > keep]
    return tuple(out)
