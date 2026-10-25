"""Disposable job directories and failure-safe atomic metadata."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat

import pytest

from rosbridge.storage import local_environment, run_environment, write_json


def test_atomic_json_failure_preserves_previous_file(tmp_path):
    path = tmp_path / 'run.json'
    write_json(path, {'state': 'ready'})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(TypeError):
        write_json(path, {'invalid': object()})
    assert json.loads(path.read_text()) == {'state': 'ready'}
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_json_writes_publish_complete_files(tmp_path):
    path = tmp_path / 'run.json'
    def write(index):
        write_json(path, {'index': index, 'data': [index] * 1000})
        saved = json.loads(path.read_text())
        assert saved['data'] == [saved['index']] * 1000
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(write, range(40)))
    assert list(tmp_path.iterdir()) == [path]


def test_run_environment_isolated_without_copying_dependencies(tmp_path):
    base = local_environment(tmp_path, {})
    first = dict(base, **run_environment(tmp_path / '.cache/rosbridge/runs/one'))
    second = dict(base, **run_environment(tmp_path / '.cache/rosbridge/runs/two'))
    for key in ('ROS_HOME', 'ROS_LOG_DIR', 'TMPDIR'):
        assert first[key] != second[key]
        assert Path(first[key]).is_dir() and Path(second[key]).is_dir()
    for key in base.keys() - {'ROS_HOME', 'ROS_LOG_DIR', 'TMPDIR'}:
        assert first[key] == second[key]
    assert Path(first['UV_PYTHON_INSTALL_DIR']).is_relative_to(tmp_path / '.runtime')
