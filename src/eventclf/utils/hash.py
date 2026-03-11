from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

def stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON string for hashing/comparisons.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()