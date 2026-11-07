"""When the BEHAVIOR backend renders: evaluator rendering for the R1Pro embodiments, on-demand
DLAA captures for the G1. OmniGibson and Kit are faked; renders are recorded as
(deliveries, dlaa)."""
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge_behavior import backend
from rosbridge_behavior.backend import BehaviorSim


class FakeEpisode:
    lights = None

    def __init__(self, task, embodiment, max_steps):
        sensor = SimpleNamespace(image_height=2, image_width=2, get_obs=lambda: (
            {"rgb": np.zeros((2, 2, 4), np.uint8), "depth_linear": np.ones((2, 2))}, {}))
        self.robot = SimpleNamespace(sensors={"head": sensor})
        self.env = SimpleNamespace(task=SimpleNamespace(success=[False], get_goal_option_satisfaction=lambda env: []))
        self.max_steps = 100

    def load(self, seed):
        pass

    def base_pose(self):
        return np.zeros(3), np.eye(3)

    def step(self, action):
        return False


@pytest.fixture
def renders(monkeypatch):
    calls = []

    def render_for_capture(deliveries, dlaa=True):
        calls.append((deliveries, dlaa))
        return deliveries
    monkeypatch.setattr(backend, "render_for_capture", render_for_capture)
    monkeypatch.setattr(backend, "boot", lambda simulator: None)
    monkeypatch.setattr(backend, "Episode", FakeEpisode)
    return calls


def make(rendering, lights=False, spectators=False):
    embodiment = SimpleNamespace(rendering=rendering, frequencies={"action_frequency": 30},
                                 cameras={"head_camera": "head"}, layout=lambda robot: None,
                                 reset=lambda episode: None, action=lambda episode, command: command)
    sim = BehaviorSim(embodiment)
    sim.check_task = lambda runtime, task: None
    FakeEpisode.lights = object() if lights else None
    if spectators:
        sim.configure_spectators()
        placed = sim.placed = []
        rgb = SimpleNamespace(get_data=lambda: np.zeros((2, 2, 4), np.uint8))
        sim._open_spectators = lambda: sim._spectators.update(
            {view.name: (view, None, None, rgb) for view in sim.spectators})
        sim._place = lambda names: placed.append(list(names))
    sim.reset("turning_on_radio", 0)
    return sim


def test_evaluator_rendering_reads_render_once_after_a_step_and_never_unread(renders):
    sim = make("evaluator")
    sim.render("head_camera")
    assert renders == []  # env.reset rendered the reset state
    sim.step([0.])
    sim.render("head_camera"), sim.render_depth("head_camera"), sim.render("head_camera")
    assert renders == [(1, False)]  # one render after the step serves every read
    sim.step([0.]), sim.step([0.])
    assert renders == [(1, False)]  # unread steps do not render
    sim.render_depth("head_camera")
    assert renders[-1] == (3, False)  # no render at the previous step: no recent history
    sim.step([0.])
    sim.render("head_camera")
    assert renders[-1] == (1, False) and sim.render_count() == 5


def test_light_tasks_rerender_after_the_sync_as_the_evaluator(renders):
    sim = make("evaluator", lights=True)
    assert renders == [(3, False)]  # after the reset's light sync
    sim.render("head_camera")
    assert len(renders) == 1
    sim.step([0.])
    sim.render("head_camera")
    assert renders[-1] == (4, False)


def test_recording_renders_every_step_and_one_render_serves_every_spectator(renders):
    sim = make("evaluator", spectators=True)
    views = ["front_left", "rear_left", "head"]
    assert renders == [(16, False)] and sim.placed == [views]  # the spectators cut to the new episode
    sim.step([0.]), sim.step([0.])
    assert renders[1:] == [(1, False)] * 2
    frames = [sim.render_spectator(view.name) for view in sim.spectators]
    assert sim.placed == [views] * 2 and renders[-1] == (1, False)
    assert len(renders) == 4 and all(frame.shape == (2, 2, 3) for frame in frames)
    sim.render("head_camera")
    assert len(renders) == 4  # the spectator render holds the current state


def test_on_demand_rendering_keeps_dlaa_captures(renders):
    sim = make("on_demand", spectators=True)
    sim.render("head_camera")
    assert renders == [(16, True)]
    sim.step([0.])
    assert len(renders) == 1  # steps never render
    sim.render("head_camera")
    sim.render_spectator("head"), sim.render_spectator("front_left")
    assert renders[1:] == [(2, True)] * 3 and sim.placed == [["head"], ["front_left"]]
