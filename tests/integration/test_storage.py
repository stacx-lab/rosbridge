"""Parallel launches share assets while keeping all job files together."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from tests.integration.support import ROOT, invoke, start


@pytest.mark.parametrize('simulators', [('abc', 'abc'), ('libero', 'libero'), ('abc', 'libero')])
def test_parallel_runs_isolate_logs_and_temporary_files(runs, simulators):
    assets = ROOT / '.runtime/assets'
    before = {p.relative_to(assets): (p.stat().st_ino, p.stat().st_size)
              for p in assets.rglob('*') if p.is_file()}
    def launch(index):
        return start(runs, selection=simulators[index])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(launch, range(2)))
    assert first['domain_id'] != second['domain_id']
    code = '''import json, os, tempfile
from pathlib import Path
with tempfile.NamedTemporaryFile(delete=False) as stream:
    stream.write(b'job temporary file')
    temporary = stream.name
print(json.dumps({'temporary': temporary, 'environment':
                 {key: os.environ[key] for key in ('ROS_HOME', 'ROS_LOG_DIR', 'TMPDIR')}}))
'''
    for record in (first, second):
        folder = Path(record['run_file']).parent
        details = json.loads(invoke('exec', record['id'], '--', 'python', '-c', code).stdout)
        for key, value in details['environment'].items():
            assert Path(value).is_relative_to(folder)
            assert value == record['ros_environment'][key]
        assert Path(details['temporary']).is_relative_to(folder / 'tmp')
        assert list((folder / 'ros/log').rglob('*.log'))
        assert (folder / 'graph.log').exists()
    invoke('stop', first['id'], '--json')
    assert json.loads(invoke('status', second['id'], '--json').stdout)['ready']
    assert Path(first['run_file']).exists()  # History is retained, never pruned.
    invoke('stop', second['id'], '--json')
    after = {p.relative_to(assets): (p.stat().st_ino, p.stat().st_size)
             for p in assets.rglob('*') if p.is_file()}
    assert before == after
