from eventclf.utils import stable_json_dumps, file_sha256
from pathlib import Path

def test_stable_json_dumps_deterministic():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert stable_json_dumps(a) == stable_json_dumps(b)

def test_file_sha256(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1 == h2