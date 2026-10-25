"""Local run records and Linux process identity checks."""
import json
import os
from pathlib import Path
import time

SCHEMA = "rosbridge/run/1"


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start": fields[19],
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def alive(record):
    owner = record.get("owner")
    return bool(owner and identity(owner["pid"]) == owner)


def read(path):
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("uid") != os.getuid():
        raise ValueError("Not a rosbridge run owned by this user")
    return value


def directory(root):
    return root / ".cache/rosbridge/runs"


def select(root, run=None):
    if run:
        candidate = Path(run).expanduser()
        if candidate.is_dir():
            candidate /= "run.json"
        if not candidate.is_file():
            if Path(run).name != run:
                raise ValueError("Select a run ID or existing run.json")
            candidate = directory(root) / run / "run.json"
        if not candidate.is_file():
            raise ValueError(f"Unknown run: {run}")
        return candidate, read(candidate)
    matches = []
    for path in directory(root).glob("*/run.json"):
        record = read(path)
        if alive(record) and record["state"] in ("starting", "ready", "stopping"):
            matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} live runs; select an explicit RUN_ID")
    return matches[0]


def wait_stopped(path, timeout=45.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = read(path)
        if not alive(record):
            return record
        time.sleep(.05)
    raise TimeoutError("Run did not stop; inspect its log and process identity")
