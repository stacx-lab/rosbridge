"""Pinned, bundled assets and noninteractive LIBERO configuration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

from rosbridge.assets import link_asset
from rosbridge.catalog import resolve
from rosbridge.project import project_root
from rosbridge.storage import write_json

PRIVATE_MACROS = '''# Managed by rosbridge prepare libero. Keep logs in the managed run.
import robosuite.macros as macros
macros.FILE_LOGGING_LEVEL = None
'''


def _logging_config():
    spec = importlib.util.find_spec('robosuite')
    return Path(spec.origin).parent / 'macros_private.py'


def task_spec(task):
    return resolve('libero', task=task)[2]


def task_files(runtime, task):
    spec = task_spec(task)
    package = Path(runtime) / 'libero/libero'
    name, suite = spec['task_name'], spec['suite']
    return (package / 'bddl_files' / suite / f'{name}.bddl',
            package / 'init_files' / suite / f'{name}.pruned_init')


def prepare(runtime, task):
    runtime = Path(runtime)
    config = _logging_config()
    if not config.exists():
        config.write_text(PRIVATE_MACROS)
    elif config.read_text() != PRIVATE_MACROS:
        raise ValueError(f'Unexpected robosuite logging configuration: {config}')
    # The upstream bundles these assets. Persist a single copy independently of
    # generated source and disposable caches; preparation holds the source lock.
    source = runtime / 'libero/libero/assets'
    destination = runtime.parent / 'assets/libero/assets'
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.assets-') as tmp:
            staging = Path(tmp) / 'assets'
            shutil.copytree(source, staging)
            staging.rename(destination)
    link_asset(destination, source)
    files = task_files(runtime, task)
    marker = runtime / '.asset_tasks.json'
    receipts = json.loads(marker.read_text()) if marker.exists() else {}
    receipts[task] = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in files}
    write_json(marker, receipts)
    check_task(runtime, task)


def check_task(runtime, task):
    runtime = Path(runtime)
    try:
        receipt = json.loads((runtime / '.asset_tasks.json').read_text())[task]
        if _logging_config().read_text() != PRIVATE_MACROS:
            raise ValueError('unmanaged robosuite logging configuration')
        if not (runtime / 'libero/libero/assets').is_dir():
            raise ValueError('missing model assets')
        for path in task_files(runtime, task):
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt[str(path.relative_to(runtime))]:
                raise ValueError('asset checksum mismatch')
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f'LIBERO assets unavailable; run rosbridge prepare libero --task {task}') from error


def configure():
    """Write configuration before importing LIBERO (which otherwise prompts)."""
    root = project_root()
    runtime = root / '.runtime/libero'
    package = runtime / 'libero/libero'
    directory = Path(os.environ.get('ROS_HOME', root / '.cache/rosbridge/ros')) / 'libero'
    directory.mkdir(parents=True, exist_ok=True)
    datasets = directory / 'datasets'
    datasets.mkdir(exist_ok=True)
    paths = {key: str(package / suffix) for key, suffix in {
        'benchmark_root': '.', 'bddl_files': 'bddl_files', 'init_states': 'init_files',
        'assets': 'assets'}.items()}
    paths['datasets'] = str(datasets)
    # JSON is valid YAML; reuse atomic, private publication with unique temp files.
    target = directory / 'config.yaml'
    if not target.exists() or json.loads(target.read_text()) != paths:
        write_json(target, paths)
    os.environ['LIBERO_CONFIG_PATH'] = str(directory)
    return runtime
