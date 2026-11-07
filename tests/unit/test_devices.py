import pytest
from rosbridge.core.devices import DeviceSpec, SampleSchedule, SpectatorView, device_inventory, spectator_views
from rosbridge.catalog import resolve
from tests.toy_backend import ToyBackend


def test_inventory_is_backend_shaped_and_collision_checked():
    inventory = device_inventory(ToyBackend.layout, {'camera_overview': 'top'})
    assert [(d.name, d.kind, d.source) for d in inventory] == [
        ('axes', 'joint', 'axes'), ('finger', 'gripper', 'finger'), ('camera_overview', 'rgb', 'top')]
    with pytest.raises(ValueError, match='unique'):
        device_inventory(ToyBackend.layout, {'axes': 'top'})
    for name in ('invalid/name', 'world'):
        with pytest.raises(ValueError, match='name'):
            DeviceSpec(name, 'rgb', 'top')
    with pytest.raises(ValueError, match='kind'):
        DeviceSpec('overview', 'spectator', 'top')
    with pytest.raises(ValueError, match='dimensions'):
        SpectatorView('overview', 'top', 0, 720)


def test_spectator_views_keep_only_the_selected_ones():
    pairs = (('front', 'base'), ('head', 'head_camera'))
    assert spectator_views(pairs, 640, 480) == (SpectatorView('front', 'base', 640, 480),
                                                SpectatorView('head', 'head_camera', 640, 480))
    assert spectator_views(pairs, 640, 480, ['head', 'missing']) == (SpectatorView('head', 'head_camera', 640, 480),)


def test_sampling_uses_sim_time_skips_gaps_and_resets():
    schedule = SampleSchedule(10)
    assert schedule.due(0, reset=True)
    assert not schedule.due(0)
    assert not schedule.due(.034)
    assert schedule.due(.102)
    assert not schedule.due(.136)
    assert schedule.due(3.04)
    assert not schedule.due(3.05)
    assert schedule.due(0, reset=True)
    assert not SampleSchedule(0).due(0, reset=True)
    for rate in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            SampleSchedule(rate)


def test_bare_native_task_resolves():
    assert resolve('put_plastic_bottles_in_bin')[2]['id'] == 'abc/put-plastic-bottles-in-bin'
    assert resolve('put-plastic-bottles-in-bin')[2]['id'] == 'abc/put-plastic-bottles-in-bin'
