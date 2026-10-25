import hashlib
import json
import re

import pytest

from rosbridge.assets import AssetStore
from rosbridge.catalog import resolve, tasks
from rosbridge_robocasa import assets

SLUG = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def test_catalog_holds_robocasa365_with_the_evaluation_split():
    entries = list(tasks(name='robocasa').values())
    assert len(entries) == 365
    assert sum(t.get('split') == 'target' for t in entries) == 50
    assert {t.get('split') for t in entries} == {None, 'target'}
    assert {t['robot'] for t in entries} == {'robocasa/robocasa-panda-omron'}
    for task in entries:
        assert task['id'] == 'robocasa/' + SLUG.sub('-', task['native_task']).lower()
        assert resolve(task['id'])[2] == task
    assert resolve('robocasa')[2]['id'] == 'robocasa/pick-place-counter-to-cabinet'
    assert resolve('PrepareVeggieDip')[2]['id'] == 'robocasa/prepare-veggie-dip'


def test_seeded_prepare_adopts_verified_archives_and_overlays_the_upstream_tree(tmp_path,
                                                                                monkeypatch):
    seed = tmp_path / 'seed/robocasa'
    for name, (_, root, _) in assets.ARCHIVES.items():
        files = {f'{root}/a/{name}.xml': name.encode()}
        if root == 'fixtures':  # upstream tracks fixtures/cabinets and repeats one file
            files.update({f'{root}/cabinets/panel.xml': b'tracked',
                          f'{root}/cabinets/door/model.xml': b'door'})
        for path, data in files.items():
            (seed / path).parent.mkdir(parents=True, exist_ok=True)
            (seed / path).write_bytes(data)
        (seed / f'{name}.json').write_text(json.dumps({'url': assets.url(name), 'files': [
            {'path': path, 'sha256': hashlib.sha256(data).hexdigest()} for path, data in files.items()]}))
    runtime = tmp_path / 'runtime'
    tree = runtime / 'robocasa/models/assets'
    (tree / 'fixtures/cabinets').mkdir(parents=True)
    (tree / 'fixtures/cabinets/panel.xml').write_bytes(b'tracked')
    (runtime / 'robosuite/robosuite').mkdir(parents=True)
    monkeypatch.setenv('ROSBRIDGE_ASSET_SEED', str(tmp_path / 'seed'))
    store = AssetStore(tmp_path / 'store')
    with pytest.raises(RuntimeError, match='not prepared'):
        assets.check_task(runtime, 'Task')
    (seed / 'textures/a/textures.xml').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differs from its receipt'):
        assets.prepare(runtime, 'Task', store)
    assert (seed / 'textures').is_dir()  # a rejected seed stays where it was
    (seed / 'textures/a/textures.xml').write_bytes(b'textures')
    assets.prepare(runtime, 'Task', store)
    assert not (seed / 'textures').exists() and (seed / 'textures.json').exists()
    assert not (tree / 'fixtures').is_symlink() and (tree / 'fixtures/cabinets/door').is_symlink()
    assert not (tree / 'fixtures/cabinets/panel.xml').is_symlink()
    assert (tree / 'objects/lightwheel/a/objects_lightwheel.xml').read_bytes() == b'objects_lightwheel'
    assert (tree / 'textures').is_symlink() and (runtime / 'robocasa/macros_private.py').exists()
    assets.prepare(runtime, 'Other', store)  # one preparation serves every task
    assets.check_task(runtime, 'Other')
