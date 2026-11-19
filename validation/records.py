from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

from provablerag_native.util import canonical, write


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_sources():
    import provablerag
    base = Path(__file__).resolve().parents[1]
    native = {p.relative_to(base).as_posix():sha(p) for p in (base/"provablerag_native").glob("*.py")}
    native["run.py"] = sha(base/"run.py")
    native["validation/records.py"] = sha(Path(__file__))
    common = Path(provablerag.__file__).resolve().parent
    for p in (base/"runtime_common/provablerag").glob("*.py"):
        native["common/"+p.name] = sha(common/p.name)
    return native


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable record differs at {path}")
    else:
        write(path, value)


def lines(path, records):
    raw = "".join(canonical(value)+"\n" for value in records).encode()
    if path.exists():
        if path.read_bytes()!=raw:
            raise ValueError(f"immutable lines differ at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


@contextmanager
def locked_writer(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/".writer.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
