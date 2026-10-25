"""Managed spectator recording: destination ownership and lifecycle metadata; no ROS imports."""
import json
from pathlib import Path

from rosbridge.storage import write_json


def destination(value):
    path = Path(value).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'recording directory must be new or empty: {path}')
    return path


def reserve(path, run_id, task):
    """Exclusive marker prevents two launches from writing the same directory."""
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f'recording directory must be new or empty: {path}')
    metadata = {'schema': 'rosbridge/recording/1', 'run_id': run_id,
                'task': task, 'state': 'starting', 'views': {}}
    with (path / 'recording.json').open('x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    return metadata


def finalize(record, returncode):
    """Validate recorder completion, retaining the original run failure if present."""
    path = Path(record['recording_dir']) / 'recording.json'
    error = record.get('error')
    try:
        metadata = json.loads(path.read_text())
        if (not isinstance(metadata, dict) or not isinstance(metadata.get('state'), str)
                or not isinstance(metadata.get('error'), (str, type(None)))):
            raise ValueError('invalid recording state or error')
    except (OSError, ValueError) as failure:
        metadata = {}
        error = error or f'Cannot read recording metadata: {failure}'
    detail = metadata.get('error')
    if error and detail and detail not in error:
        error = f'{error}; recording: {detail}'
    if error or returncode != 0 or metadata['state'] != 'complete':
        error = error or metadata.get('error') or 'recording did not finalize cleanly'
        metadata.update(state='failed', error=error)
        record['error'] = error
    record['recording_state'] = metadata['state']
    if error:
        write_json(path, metadata)
