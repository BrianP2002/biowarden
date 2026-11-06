"""Append-only JSONL ledger with a hash chain.

Each record stores ``prev_hash`` (the previous record's hash) and its own ``hash`` over the
canonical JSON of everything else. A sidecar ``<ledger>.head`` anchors the latest ``seq`` and
``hash`` so that cutting records off the end is detectable; it is an integrity aid, not a security
boundary, since anyone who can edit the ledger can edit the sidecar as well.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biowarden.contracts.request import canonical_json, sha256_hex

GENESIS = "0" * 64


class LedgerCorruptError(Exception):
    """The ledger fails verification, so nothing more is appended to it."""


@dataclass
class LedgerReport:
    ok: bool
    n_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _head_path(path: Path) -> Path:
    return path.with_name(path.name + ".head")


def _record_hash(record: dict[str, Any]) -> str:
    return sha256_hex(canonical_json({k: v for k, v in record.items() if k != "hash"}))


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Add one record; ``seq``, ``ts``, ``prev_hash`` and ``hash`` are assigned here."""
        reserved = {"seq", "ts", "prev_hash", "hash"} & fields.keys()
        if reserved:
            raise ValueError(f"reserved ledger field(s): {sorted(reserved)}")
        fields = _scrub(fields)
        with self._lock, open(self.path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            report = verify_ledger(self.path)
            if not report.ok:
                raise LedgerCorruptError("; ".join(report.errors))
            last = _last_record(handle)
            record = {
                "seq": 0 if last is None else last["seq"] + 1,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                **fields,
                "prev_hash": GENESIS if last is None else last["hash"],
            }
            record["hash"] = _record_hash(record)
            handle.write((canonical_json(record) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            _write_head(self.path, record)
        return record


def _scrub(value: Any) -> Any:
    """Replace lone surrogates (model output can carry them) so the record encodes as UTF-8."""
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {_scrub(k): _scrub(v) for k, v in value.items()}
    return value


def _last_record(handle: Any) -> dict[str, Any] | None:
    handle.seek(0)
    lines = [ln for ln in handle.read().decode("utf-8").split("\n") if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def _write_head(path: Path, record: dict[str, Any]) -> None:
    head = _head_path(path)
    tmp = head.with_name(head.name + ".tmp")
    tmp.write_text(json.dumps({"seq": record["seq"], "hash": record["hash"]}))
    os.replace(tmp, head)


def verify_ledger(path: str | Path, *, require_head: bool = False) -> LedgerReport:
    """Check hashes, links, sequence numbers and the head anchor."""
    path = Path(path)
    report = LedgerReport(ok=True, n_records=0)
    if not path.exists():
        return report
    prev_hash, last = GENESIS, None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            report.errors.append(f"line {lineno}: invalid JSON (truncated or edited)")
            break
        if record.get("seq") != report.n_records:
            report.errors.append(
                f"line {lineno}: seq {record.get('seq')} where {report.n_records} was expected"
            )
        if record.get("prev_hash") != prev_hash:
            report.errors.append(f"line {lineno}: prev_hash does not match the previous record")
        if record.get("hash") != _record_hash(record):
            report.errors.append(f"line {lineno}: hash mismatch (record was modified)")
        prev_hash = record.get("hash", "")
        last = record
        report.n_records += 1

    head = _head_path(path)
    if head.exists():
        try:
            anchor = json.loads(head.read_text())
        except json.JSONDecodeError:
            report.errors.append("head anchor is unreadable")
        else:
            expected = None if last is None else {"seq": last["seq"], "hash": last["hash"]}
            if anchor != expected and not report.errors:
                report.errors.append(
                    f"head anchor says seq {anchor.get('seq')} but the ledger ends at "
                    f"{'nothing' if last is None else last['seq']} (records cut off or added)"
                )
    elif report.n_records:
        (report.errors if require_head else report.warnings).append("head anchor is missing")
    report.ok = not report.errors
    return report
