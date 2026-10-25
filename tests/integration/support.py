"""Shared helpers for end-to-end CLI tests."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CLI = [str(ROOT / ".venv/bin/python"), "-m", "rosbridge"]


def invoke(*args, expected=0):
    result = subprocess.run([*CLI, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def start(runs, *args, selection="abc"):
    record = json.loads(invoke("launch", selection, "--execution-mode", "stepped", "--detach", "--json", *args).stdout)
    runs.append(record)
    return record

