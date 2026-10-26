"""Stable fingerprints for source and derived EventRecord payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from recallvision.events.schema import EventRecord


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def event_fingerprint(event: EventRecord) -> str:
    return hashlib.sha256(canonical_json_bytes(event.to_record())).hexdigest()
