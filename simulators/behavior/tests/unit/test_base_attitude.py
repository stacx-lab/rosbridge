"""A tilting base's odometry in full (environment.Episode.attitude), without OmniGibson: the R1Pro's z, roll and pitch
joints are free, so the planar odometry showed a tipped robot upright."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge.core.worker import DYNAMIC
from rosbridge_behavior import backend, environment
from rosbridge_behavior.environment import rotation, turn


def quaternion(matrix):
    """(x, y, z, w) of a rotation matrix (Shepperd's method, robust at every angle)."""
    m = np.asarray(matrix, float)
    trace = np.trace(m)
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[i], q[j], q[k], q[3] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s, (m[k, j] - m[j, k]) / s
    return q


def zyx(roll, pitch, yaw):
    return turn("z", yaw) @ turn("y", pitch) @ turn("x", roll)


class Link:
    """A base link as OmniGibson reads it: its world pose, and its velocities in the world frame."""
    def __init__(self, position, matrix, linear=(0., 0., 0.), angular=(0., 0., 0.)):
        self.position, self.orientation = np.asarray(position, float), quaternion(matrix)
        self.linear, self.angular = np.asarray(linear, float), np.asarray(angular, float)

    def get_position_orientation(self):
        return self.position, self.orientation

    def get_linear_velocity(self):
        return self.linear

    def get_angular_velocity(self):
        return self.angular


def episode(link, start_position=(2., -1., .03), start=None):
    """An Episode that started at start_position with rotation start (upright, heading 0.5 rad, by default), its base
    now at link."""
    value = environment.Episode.__new__(environment.Episode)
    value.robot = SimpleNamespace(base_footprint_link=link)
    value.origin = np.asarray(start_position, float), turn("z", .5) if start is None else start
    return value


def published(run):
    """The rotation a client rebuilds from the odometry's yaw and the attitude's roll and pitch (ZYX), relative to
    the episode start."""
    attitude, (pose, _) = run.attitude(), run.odometry()
    return zyx(attitude["roll"], attitude["pitch"], pose[2]), attitude


def test_an_upright_base_has_zero_attitude_and_its_planar_odometry_as_before():
    link = Link((3., 0., .03), turn("z", 1.2), linear=(.3, .1, 0.), angular=(0., 0., .4))
    run = episode(link)
    attitude = run.attitude()
    assert attitude == pytest.approx({"z": 0., "roll": 0., "pitch": 0., "vz": 0., "wx": 0., "wy": 0., "tilt": 0.},
                                     abs=1e-12)
    # The planar odometry is computed exactly as before the attitude was added.
    (start, start_rotation), world = run.origin, rotation(link.orientation)
    offset, relative = start_rotation.T @ (link.position - start), start_rotation.T @ world
    pose, twist = run.odometry()
    assert pose == (offset[0], offset[1], np.arctan2(relative[1, 0], relative[0, 0]))
    assert twist == ((world.T @ link.linear)[0], (world.T @ link.linear)[1], (world.T @ link.angular)[2])
    assert pose[2] == pytest.approx(.7)


def test_a_tilted_base_reports_its_roll_pitch_height_and_rates_and_its_tilt_from_vertical():
    roll, pitch, yaw = .4, -.25, 1.9
    world = zyx(roll, pitch, yaw)
    linear, angular = np.array([.1, -.2, .05]), np.array([.6, -.3, .2])  # in the base's own frame
    run = episode(Link((2.5, -1.2, .21), world, linear=world @ linear, angular=world @ angular))
    attitude = run.attitude()
    assert (attitude["roll"], attitude["pitch"]) == pytest.approx((roll, pitch))
    assert attitude["z"] == pytest.approx(.18)
    assert (attitude["vz"], attitude["wx"], attitude["wy"]) == pytest.approx((.05, .6, -.3))
    assert attitude["tilt"] == pytest.approx(math.acos(math.cos(roll) * math.cos(pitch)))
    pose, twist = run.odometry()
    assert pose[2] == pytest.approx(yaw - .5) and twist == pytest.approx((.1, -.2, .2))
    assert published(run)[0] == pytest.approx(turn("z", .5).T @ world)


def test_random_orientations_round_trip_and_tilt_is_the_angle_from_vertical():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        relative = zyx(rng.uniform(-math.pi, math.pi), rng.uniform(-1.5, 1.5), rng.uniform(-math.pi, math.pi))
        run = episode(Link((0., 0., 0.), turn("z", .5) @ relative))
        rebuilt, attitude = published(run)
        assert np.abs(rebuilt - relative).max() < 1e-9
        assert attitude["tilt"] == pytest.approx(math.acos(np.clip((turn("z", .5) @ relative)[2, 2], -1, 1)),
                                                 abs=1e-9)


@pytest.mark.parametrize("pitch_deg", [89., 89.999, 90., -90., -89.999])
@pytest.mark.parametrize("roll, yaw", [(.3, .7), (-2.5, -1.), (0., 0.)])
def test_a_base_pitched_onto_its_nose_or_tail_is_rebuilt_exactly(pitch_deg, roll, yaw):
    """At exactly +-90 degrees of pitch yaw and roll are not separately defined: the odometry's yaw is whatever the
    rounding gives, and roll makes up the rest (an earlier formula rebuilt it up to 15 degrees off there)."""
    relative = zyx(roll, math.radians(pitch_deg), yaw)
    run = episode(Link((0., 0., .4), turn("z", .5) @ relative))
    rebuilt, attitude = published(run)
    assert np.abs(rebuilt - relative).max() < 1e-9
    assert math.degrees(attitude["pitch"]) == pytest.approx(pitch_deg, abs=1e-6)
    tilt = math.acos(math.cos(roll) * math.cos(math.radians(pitch_deg)))
    assert attitude["tilt"] == pytest.approx(tilt, abs=1e-9)


def test_roll_and_pitch_are_relative_to_the_start_and_tilt_to_gravity():
    start = turn("x", math.radians(4.))  # the base settled 4 degrees off level before the episode started
    attitude = episode(Link((0., 0., 0.), start), start=start).attitude()
    assert math.degrees(attitude["roll"]) == pytest.approx(0., abs=1e-9)
    assert math.degrees(attitude["tilt"]) == pytest.approx(4.)


def test_a_base_lying_on_its_side_reads_a_tilt_near_ninety_degrees():
    link = Link((2., -1., .35), zyx(math.radians(88.), 0., .5))
    assert math.degrees(episode(link).attitude()["tilt"]) == pytest.approx(88.)


def test_both_r1pro_embodiments_report_their_attitude_and_the_worker_carries_it():
    from rosbridge_behavior.g1_sharpa import G1Sharpa
    from rosbridge_behavior.r1pro import R1Pro
    from rosbridge_behavior.r1pro_sharpa import R1ProSharpa

    assert R1Pro.attitude is True and R1ProSharpa.attitude is True  # the evaluator's robot has the same base
    assert not getattr(G1Sharpa, "attitude", False)
    assert "base_attitude" in DYNAMIC  # snapshot after every worker call, so the world reads it each tick


class FakeEpisode:
    lights = None

    def __init__(self, task, embodiment, max_steps):
        self.robot = SimpleNamespace(sensors={})
        self.env = SimpleNamespace(task=SimpleNamespace(success=[False], get_goal_option_satisfaction=lambda env: []))
        self.max_steps, self.tilt = 100, 0.

    def load(self, seed):
        pass

    def base_pose(self):
        return np.zeros(3), np.eye(3)

    def attitude(self):
        return {"z": 0., "roll": self.tilt, "pitch": 0., "vz": 0., "wx": 0., "wy": 0., "tilt": self.tilt}

    def step(self, action):
        self.tilt += .1
        return False


@pytest.mark.parametrize("attitude", [True, False])
def test_the_backend_keeps_the_attitude_after_every_reset_and_step(monkeypatch, attitude):
    monkeypatch.setattr(backend, "render_for_capture", lambda deliveries, dlaa=True: deliveries)
    monkeypatch.setattr(backend, "boot", lambda simulator: None)
    monkeypatch.setattr(backend, "Episode", FakeEpisode)
    embodiment = SimpleNamespace(rendering="evaluator", frequencies={"action_frequency": 30}, cameras={},
                                 layout=lambda robot: None, reset=lambda episode: None,
                                 action=lambda episode, command: command, attitude=attitude)
    sim = backend.BehaviorSim(embodiment)
    sim.check_task = lambda runtime, task: None
    sim.reset("turning_on_radio", 0)
    assert (sim.base_attitude or {}).get("tilt") == (0. if attitude else None)
    sim.step([0.])
    sim.step([0.])
    assert (sim.base_attitude or {}).get("tilt") == (pytest.approx(.2) if attitude else None)
