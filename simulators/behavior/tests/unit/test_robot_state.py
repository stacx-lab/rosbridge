"""The BEHAVIOR backend checks its robot's state after every step (BehaviorSim._check_state), at two levels: a joint
more than JOINT_NOTICE beyond its range is stated (state_notice) and the episode goes on; a value that is not finite, a
revolute joint more than REVOLUTE_END or a prismatic one more than its range width beyond its range ends the episode
with the reason (fault), also when OmniGibson's own step raises on it. OmniGibson and Kit are faked."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge.core.worker import DYNAMIC
from rosbridge_behavior import backend
from rosbridge_behavior.backend import JOINT_NOTICE, REVOLUTE_END, BehaviorSim

TORSO3, ARM5, FINGER, BASE_Z, BASE_RZ = range(5)


class Robot:
    """torso_joint3 [-1.8326, 1.5708] and left_arm_joint5 [-2.3562, 2.3562] (revolute), a gripper finger
    [0, 0.05] m (prismatic), and two of a holonomic base's virtual joints [-1000, 1000]; a camera."""
    dof_names_ordered = ["torso_joint3", "left_arm_joint5", "left_gripper_finger_joint1", "base_footprint_z_joint",
                         "base_footprint_rz_joint"]
    joint_lower_limits = [-1.8326, -2.3562, 0., -1000., -1000.]
    joint_upper_limits = [1.5708, 2.3562, .05, 1000., 1000.]
    joints = {name: SimpleNamespace(joint_type=kind) for name, kind in zip(dof_names_ordered, (
        "RevoluteJoint", "RevoluteJoint", "PrismaticJoint", "PrismaticJoint", "RevoluteJoint"))}
    is_holonomic_base, base_idx = True, [BASE_Z, BASE_RZ]

    def __init__(self):
        self.q, self.dq = np.zeros(5), np.zeros(5)
        self.frame = 7
        self.sensors = {"head": SimpleNamespace(image_height=2, image_width=3, get_obs=lambda: (
            {"rgb": np.full((2, 3, 4), self.frame, np.uint8), "depth_linear": np.ones((2, 3))}, {}))}

    def get_joint_positions(self):
        return self.q.copy()

    def get_joint_velocities(self):
        return self.dq.copy()


class FakeEpisode:
    lights = None

    def __init__(self, task, embodiment, max_steps):
        self.robot = Robot()
        self.env = SimpleNamespace(task=SimpleNamespace(success=[False], get_goal_option_satisfaction=lambda env: []))
        self.max_steps = 100
        self.next = None  # what the next step does to the robot: a function, or an exception to raise

    def load(self, seed):
        pass

    def base_pose(self):
        if not np.isfinite(self.robot.q).all():
            # OmniGibson's RigidDynamicPrim.get_position_orientation on a NaN orientation
            raise AssertionError("base_link orientation tensor([nan, nan, nan, nan]) is not a unit quaternion.")
        return np.zeros(3), np.eye(3)

    def step(self, action):
        change, self.next = self.next, None
        if isinstance(change, Exception):
            self.robot.q[:] = np.nan
            raise change
        if change:
            change(self.robot)
        return False


@pytest.fixture
def sim(monkeypatch):
    monkeypatch.setattr(backend, "render_for_capture", lambda deliveries, dlaa=True: deliveries)
    monkeypatch.setattr(backend, "boot", lambda simulator: None)
    monkeypatch.setattr(backend, "Episode", FakeEpisode)
    embodiment = SimpleNamespace(rendering="evaluator", frequencies={"action_frequency": 30},
                                 cameras={"head_camera": "head"}, layout=lambda robot: None,
                                 reset=lambda episode: None, action=lambda episode, command: command,
                                 state=lambda episode: (episode.robot.q.copy(), episode.robot.dq.copy()))
    value = BehaviorSim(embodiment)
    value.check_task = lambda runtime, task: None
    value.reset("turning_on_radio", 0)
    return value


def then(sim, joint=None, position=None):
    """One step that leaves the joint at the position (or raises, for an exception)."""
    if isinstance(joint, Exception):
        sim.episode.next = joint
    elif joint is not None:
        sim.episode.next = lambda robot: robot.q.__setitem__(joint, position)
    sim.step(np.zeros(5))


def test_the_notice_and_end_levels():
    assert (JOINT_NOTICE, REVOLUTE_END) == (.5, 2 * math.pi)


def test_a_valid_step_and_a_joint_within_the_notice_margin_go_on_unremarked(sim):
    then(sim)
    then(sim, ARM5, 2.3562 + JOINT_NOTICE - .01)  # 0.49 rad beyond: contact
    assert (sim.fault, sim.state_notice, sim.done, sim.steps) == (None, None, False, 2)


@pytest.mark.parametrize("excess", [JOINT_NOTICE + .01, .788, 3.17, 4.97])
def test_a_revolute_joint_beyond_the_notice_margin_is_stated_and_the_episode_goes_on(sim, excess):
    """0.788 rad ended an episode when the end level was 0.5; 4.97 rad is the largest measured in a fall."""
    then(sim, ARM5, 2.3562 + excess)
    assert sim.fault is None and not sim.done
    assert sim.state_notice == (f"left_arm_joint5 is at {2.3562 + excess:.3f}, {excess:.3f} beyond its range "
                                "[-2.356, 2.356]")
    then(sim, ARM5, 2.3)  # back within its range: no longer stated
    assert sim.state_notice is None and not sim.done


def test_several_joints_beyond_the_notice_margin_are_counted(sim):
    def bend(robot):
        robot.q[ARM5], robot.q[TORSO3] = 3.5, -2.6
    sim.episode.next = bend
    sim.step(np.zeros(5))
    assert sim.state_notice == ("left_arm_joint5 is at 3.500, 1.144 beyond its range [-2.356, 2.356] "
                                "(2 joints are more than 0.5 beyond their ranges)")


def test_a_revolute_joint_more_than_the_end_level_beyond_its_range_ends_the_episode_with_the_reason(sim):
    then(sim, TORSO3, 1.5708 + REVOLUTE_END - .01)
    assert not sim.done and sim.state_notice.startswith("torso_joint3")
    then(sim, TORSO3, 24.036)  # torso_joint2's reading in a native blow-up, on torso_joint3's range
    assert sim.done and sim.steps == 2 and sim.state_notice is None
    assert sim.fault == "torso_joint3 is at 24.036, 22.465 beyond its range [-1.833, 1.571]"
    assert sim.state()[0][TORSO3] == 24.036  # still readable: observations stay live
    assert sim.render("head_camera")[0, 0, 0] == 7
    with pytest.raises(RuntimeError, match="episode is done"):
        sim.step(np.zeros(5))
    assert {"fault", "state_notice", "base_attitude"} <= set(DYNAMIC)  # the world reads them from the worker


def test_a_prismatic_joint_ends_the_episode_beyond_its_own_range_width(sim):
    """The gripper finger's range is 0.05 m wide: 0.04 m beyond goes on (the notice margin, 0.5, is never reached
    before the end for it), 0.06 m beyond ends."""
    then(sim, FINGER, .05 + .04)
    assert (sim.fault, sim.state_notice, sim.done) == (None, None, False)
    then(sim, FINGER, .05 + .06)
    assert sim.done and sim.fault == "left_gripper_finger_joint1 is at 0.110, 0.060 beyond its range [0.000, 0.050]"


@pytest.mark.parametrize("joint, position", [(BASE_Z, -1001.59), (BASE_RZ, 5000.)])
def test_a_virtual_base_joint_is_checked_for_finite_values_only(sim, joint, position):
    """Their [-1000, 1000] limits bound no motion of the robot."""
    then(sim, joint, position)
    assert (sim.fault, sim.state_notice, sim.done) == (None, None, False)
    then(sim, joint, math.inf)
    assert sim.done and sim.fault.startswith("the state of ") and sim.fault.endswith(" is not finite")


def test_a_state_that_is_not_finite_ends_the_episode_and_freezes_the_observations(sim):
    first = sim.state()
    camera = sim.render("head_camera").copy()
    sim.episode.next = lambda robot: robot.dq.__setitem__(ARM5, np.nan)
    sim.step(np.zeros(5))
    assert sim.done and sim.fault == "the state of left_arm_joint5 is not finite"
    sim.episode.robot.frame = 99
    assert sim.state() is first  # the last valid observation
    assert (sim.render("head_camera") == camera).all() and sim.render_depth("head_camera").shape == (2, 3)


def test_the_physics_failing_inside_omnigibsons_step_ends_the_episode_not_the_world(sim):
    """PhysX's Illegal BroadPhaseUpdateData left the base's orientation NaN and OmniGibson's own step raised on it."""
    before = sim.state()  # as the world reads it after every step
    then(sim, AssertionError("base_link orientation tensor([nan, nan, nan, nan]) is not a unit quaternion."))
    assert sim.done and sim.steps == 1
    assert sim.fault == ("the state of torso_joint3, left_arm_joint5, left_gripper_finger_joint1 and others is not "
                         "finite; the physics step failed (AssertionError)")
    assert sim.state() is before and np.isfinite(before[0]).all()  # the last valid observation, not NaN
    frame = sim.render("head_camera")  # no frame rendered before: a blank one, nothing rendered
    assert frame.shape == (2, 3, 3) and not frame.any()


def test_any_other_error_in_a_step_is_raised_as_before(sim):
    sim.episode.next = RuntimeError("something else")
    sim.episode.base_pose = lambda: (np.zeros(3), np.eye(3))
    sim.episode.robot.get_joint_positions = lambda: np.zeros(5)  # the robot stays valid
    with pytest.raises(RuntimeError, match="something else"):
        sim.step(np.zeros(5))
    assert sim.fault is None


def test_a_reset_clears_the_fault_and_the_notice(sim):
    then(sim, ARM5, 3.)
    assert sim.state_notice
    then(sim, ARM5, 9.)
    assert sim.fault
    sim.episode.robot.q[:] = 0.
    sim.reset("turning_on_radio", 0)
    assert (sim.fault, sim.state_notice, sim.done, sim._frozen) == (None, None, False, False)
