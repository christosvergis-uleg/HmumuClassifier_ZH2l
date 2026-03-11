from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]

def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: PathLike, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

def load_json(path: PathLike) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())