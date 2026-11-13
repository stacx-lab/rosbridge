"""HandsOn's trial scoring, optional partial credit and native commands, without Isaac: a native step's info and adapter
reason in, its EEF52 command out."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge_handson.backend import HandsOnSim, judge, limits_palm, pose_matrix_wxyz, reach, rotation_vector


@pytest.mark.parametrize("failure, info, reason, expected", [
    (None, {"succeed": False}, None, (None, False)),
    (None, {"succeed": True}, "native_terminated", (None, True)),
    (None, {"succeed": False, "failed": True}, "native_terminated", ("native_failed", False)),
    # t11's adapter names its contact rule before the native flags assert termination.
    (None, {"succeed": False}, "native_c12_direct_hand_contact", ("native_c12_direct_hand_contact", False)),
    (None, {"succeed": False}, "native_truncated", (None, False)),
    # A step that succeeds and fails at once is a success, as upstream scores the step that ends its trial.
    (None, {"succeed": True, "failed": True}, "native_terminated", (None, True)),
    # A latched failure stays, and nothing after it succeeds.
    ("native_failed", {"succeed": True}, "native_terminated", ("native_failed", False)),
])
def test_a_failure_latches_unless_its_step_succeeds(failure, info, reason, expected):
    assert judge(failure, info, reason) == expected


class Checklist:
    """A native environment's success checklist, upstream's teleoperation panel rows."""

    def __init__(self):
        self.rows, self.reads = {"bowl_upright": True, "ball_in_bowl": False, "ball_still": False}, 0

    def _success_condition_rows(self):
        self.reads += 1
        return [dict(key=key, label=key, ok=ok) for key, ok in self.rows.items()]


def episode(steps, **options):
    """Settle `steps` native steps, each an (info, reason, checklist changes) triple."""
    sim = HandsOnSim(max_steps=10, **options)
    sim.env, sim.steps = Checklist(), 0
    for info, reason, changes in steps:
        sim.env.rows.update(changes)
        sim.steps += 1
        sim._settle(info, reason)
    return sim


PLAIN = ({"succeed": False}, None, {})
FAILED = ({"succeed": False, "failed": True}, "native_terminated", {})


def test_a_run_created_to_stop_on_success_also_ends_at_a_failure():
    stopping = episode([PLAIN, FAILED], terminate_on_success=True)
    assert stopping.done and stopping.end_reason() == ("failure", "native_failed") and not stopping.success
    # Otherwise the failure latches and the episode goes on to its step limit.
    going_on = episode([PLAIN, FAILED])
    assert not going_on.done and going_on.end_reason() is None and going_on.failure == "native_failed"
    assert episode([PLAIN, PLAIN], terminate_on_success=True).end_reason() is None


def test_a_run_created_to_stop_on_failure_ends_at_a_failure_but_not_at_a_success(monkeypatch):
    monkeypatch.setenv("ROSBRIDGE_HANDSON_STOP_ON_FAILURE", "1")
    stopping = episode([PLAIN, PLAIN, FAILED])
    assert stopping.done and stopping.end_reason() == ("failure", "native_failed") and stopping.failure_step == 3
    assert stopping.measures() == {"failure": "native_failed", "failure_detail": {"failed": True}, "failure_step": 3}
    succeeding = episode([PLAIN, ({"succeed": True}, "native_terminated", {}), PLAIN])
    assert not succeeding.done and succeeding.end_reason() is None
    monkeypatch.delenv("ROSBRIDGE_HANDSON_STOP_ON_FAILURE")
    assert not episode([PLAIN, FAILED]).done  # off by default


def test_a_step_that_succeeds_and_fails_at_once_ends_a_stop_on_success_run_as_a_success():
    sim = episode([PLAIN, ({"succeed": True, "failed": True}, "native_terminated", {})], terminate_on_success=True)
    assert sim.done and sim.success and sim.reward == 1. and sim.failure is None and sim.end_reason() is None


def test_a_failure_records_the_native_info_that_explains_it():
    failing = ({"succeed": False, "failed": True, "ycb01_failure_reason": "ball_or_hand_touched_gate",
                "ycb01_gate_contact_seen": True, "c05_failed_early_click": False, "truncated_reason": "",
                "failure_steps": 0, "ycb01_hand_gate_contact_force": 1.06}, "native_terminated", {})
    later = ({"succeed": False, "failed": True, "ycb01_failure_reason": "something later"}, "native_terminated", {})
    sim = episode([PLAIN, failing, later])
    # Set scalar entries named for a failure or reason, from the step the failure latched; never in replies.
    assert sim.measures()["failure_detail"] == {"failed": True, "ycb01_failure_reason": "ball_or_hand_touched_gate"}
    assert "failure_detail" not in episode([PLAIN, PLAIN]).measures()


def test_partial_credit_is_off_by_default_and_then_reads_nothing(monkeypatch):
    monkeypatch.delenv("ROSBRIDGE_HANDSON_PARTIAL_CREDIT", raising=False)
    sim = episode([PLAIN, ({"succeed": False}, None, {"ball_in_bowl": True}), FAILED])
    assert sim.measures() == {"failure": "native_failed", "failure_detail": {"failed": True}, "failure_step": 3}
    assert sim.env.reads == 0


def test_partial_credit_is_behaviors_q_score_over_the_checklist(monkeypatch):
    monkeypatch.setenv("ROSBRIDGE_HANDSON_PARTIAL_CREDIT", "1")
    sim = episode([PLAIN, ({"succeed": False}, None, {"ball_in_bowl": True}), PLAIN])
    measures = sim.measures()
    assert measures["conditions_initial"] == {"bowl_upright": True, "ball_in_bowl": False, "ball_still": False}
    assert measures["conditions"] == {"bowl_upright": True, "ball_in_bowl": True, "ball_still": False}
    # BEHAVIOR's rule: conditions that hold at the end and did not at the start, over all of them (bowl_upright
    # held from the start, so it never counts).
    assert measures["q_score"] == pytest.approx(1 / 3)
    assert sim.env.reads == 2  # after the first step and at the end, never per step


def test_partial_credit_freezes_at_the_failure_and_is_one_on_success():
    # Upstream's trial ends at the failure: a condition that holds only later earns nothing.
    failed = episode([PLAIN, FAILED, ({"succeed": False}, None, {"ball_in_bowl": True})], partial_credit=True)
    measures = failed.measures()
    assert measures["conditions"]["ball_in_bowl"] is False and measures["q_score"] == 0.
    succeeded = episode([PLAIN, ({"succeed": True}, "native_terminated", {"ball_in_bowl": True})], partial_credit=True)
    assert succeeded.measures()["q_score"] == 1.


def test_conditions_true_from_the_start_score_nothing():
    sim = episode([({"succeed": False}, None, {"ball_in_bowl": True, "ball_still": True})], partial_credit=True)
    assert sim.measures()["q_score"] == 0.


class Robot:
    """A native arm's joint buffers, as the backend reads their device and dtype."""
    device, data = "cpu", SimpleNamespace(joint_pos=np.zeros((1, 27)))


class Native:
    """A native environment that keeps the EEF52 commands it is stepped with."""

    def __init__(self):
        self._robots, self.commands = {"right": Robot(), "left": Robot()}, []

    def step(self, command):
        self.commands.append(command)
        return None, 0., False, False, {"succeed": False}


def test_each_step_commands_the_palm_pose_its_arm_target_puts_the_palm_at(monkeypatch):
    # The arm follows its joint target in place of the palm-pose IK; native logic that reads the commanded palm (t16's
    # hinge opens only while it rises) gets the palm pose of that target, not a zero pose that leaves it at its reset.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda data, device, dtype: np.array(data)))
    sim = HandsOnSim(max_steps=10, partial_credit=False)
    sim.env, sim.steps, sim.done, sim._arm = Native(), 0, False, {}
    sim._adapter = SimpleNamespace(terminal_reason=lambda result, env: None)
    sim._slices = {"right": (slice(0, 7), slice(7, 27)), "left": (slice(27, 34), slice(34, 54))}
    # Stand-ins for the forward kinematics of each arm: xyz from the first joints, a fixed wxyz per side.
    sim._palm = {"right": lambda joints: np.r_[joints[:3], 0., 1., 0., 0.],
                 "left": lambda joints: np.r_[-joints[:3], 1., 0., 0., 0.]}
    for lift in (0., .1):
        action = np.linspace(-1., 1., 54)
        action[[0, 27]] += lift
        sim.step(action)
        command = sim.env.commands[-1]
        for side, (arm, hand) in sim._slices.items():
            np.testing.assert_array_equal(command[side], np.r_[sim._palm[side](action[arm]), action[hand]])
            np.testing.assert_array_equal(sim._arm[side], action[arm][None])  # what replaces the IK's target
    assert sim.env.commands[1]["right"][0] == pytest.approx(sim.env.commands[0]["right"][0] + .1)


def chain(joints):
    """A 7-joint stand-in arm: its palm at the end of revolute links about alternating axes."""
    pose = np.eye(4)
    for i, angle in enumerate(joints):
        c, s = np.cos(angle), np.sin(angle)
        turn = np.eye(4)
        turn[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]] if i % 2 == 0 else [[1, 0, 0], [0, c, -s], [0, s, c]]
        link = np.eye(4)
        link[2, 3] = .12
        pose = pose @ turn @ link
    return pose


LOWER, UPPER = np.full(7, -2.5), np.full(7, 2.5)
HOME = np.array([.3, .8, -.2, 1.1, .4, -.6, .2])


def test_reach_finds_the_joints_near_its_seed_that_put_the_palm_at_a_pose():
    goal = HOME + [.05, -.04, .03, .06, -.05, .04, -.03]
    joints = reach(chain, chain(goal), [HOME], LOWER, UPPER)
    error = np.linalg.inv(chain(joints)) @ chain(goal)
    assert np.linalg.norm(error[:3, 3]) < 1e-6 and np.linalg.norm(rotation_vector(error[:3, :3])) < 1e-6
    assert np.abs(joints - HOME).max() < .3  # the redundant arm stays near its seed
    seed = HOME.copy()
    assert reach(chain, chain(HOME), [seed], LOWER, UPPER) is seed  # a seed already there is kept as it is
    far = chain(HOME)
    far[:3, 3] = [3., 0., 0.]  # beyond the arm's reach: as near as it gets, within the limits
    joints = reach(chain, far, [HOME], LOWER, UPPER)
    assert np.linalg.norm(chain(joints)[:3, 3] - far[:3, 3]) < np.linalg.norm(chain(HOME)[:3, 3] - far[:3, 3]) - .3
    assert np.all((LOWER <= joints) & (joints <= UPPER))


def test_a_rotation_vector_is_the_axis_times_the_angle():
    for vector in ([0., 0., 0.], [.3, -.2, .1], [0., 2.5, 0.]):
        angle = np.linalg.norm(vector)
        axis = np.asarray(vector) / angle if angle else np.zeros(3)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        matrix = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
        np.testing.assert_allclose(rotation_vector(matrix), vector, atol=1e-12)
    np.testing.assert_allclose(pose_matrix_wxyz([1., 2., 3., np.cos(.4), 0., 0., np.sin(.4)])[:3, :],
                               [[np.cos(.8), -np.sin(.8), 0, 1], [np.sin(.8), np.cos(.8), 0, 2], [0, 0, 1, 3]],
                               atol=1e-12)


def quaternion(matrix):
    """A rotation matrix's wxyz quaternion (for rotations of less than 180 degrees)."""
    w = np.sqrt(1 + np.trace(matrix)) / 2
    return np.r_[w, (matrix[2, 1] - matrix[1, 2]) / (4 * w), (matrix[0, 2] - matrix[2, 0]) / (4 * w),
                 (matrix[1, 0] - matrix[0, 1]) / (4 * w)]


class ChainPalm:
    """Backend Palm stand-in: the chain's palm pose in EEF52's form, and its IK."""

    def __call__(self, joints):
        matrix = chain(joints)
        return np.r_[matrix[:3, 3], quaternion(matrix[:3, :3])]

    def reach(self, pose, seeds):
        return reach(chain, pose_matrix_wxyz(pose), seeds, LOWER, UPPER)


class Arms:
    """A native environment's arm state and palm limiter, as the backend's arm target reads them."""

    def __init__(self, backend="dls", contact=None, limit=None):
        self._arm_ik_backend, self._robots = backend, {"right": Robot(), "left": Robot()}
        if contact is not None:
            self._physx_contact_arm_target_limits_enabled = lambda: contact
        self._arm_joint_ids = {side: list(range(7)) for side in ("right", "left")}
        self._joint_targets = {side: np.r_[HOME, np.zeros(20)][None] for side in ("right", "left")}
        self.limit, self.limited = limit, []

    def _limit_arm_pose_target(self, side, robot, pose):
        self.limited.append(side)
        return self.limit(np.asarray(pose, dtype=float).copy())


def arms(monkeypatch, env, goal=HOME):
    """A backend after a step that commanded the right arm to goal."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda data, device, dtype: np.array(data)))
    sim = HandsOnSim(partial_credit=False)
    sim.env, sim.steps, sim._palm = env, 4, {"right": ChainPalm(), "left": ChainPalm()}
    sim._limits_palm = limits_palm(env)
    sim._arm = {"right": np.asarray(goal, dtype=float)[None], "left": HOME[None]}
    sim._commanded = {"right": sim._palm["right"](goal), "left": sim._palm["left"](HOME)}
    return sim


def floor(z):
    """A palm limiter that lifts the palm to a height, as eval_flex02's keeps the hand above the cloth."""
    def limit(pose):
        pose[2] = max(pose[2], z)
        return pose
    return limit


@pytest.mark.parametrize("env, expected", [
    (Arms(), False), (Arms(contact=False), False),  # the frozen Core classes have no contact limits
    (Arms(contact=True), True), (Arms("haorui"), True), (Arms("haorui", contact=False), True),
])
def test_upstream_limits_the_palm_with_haorui_ik_or_contact_limits(env, expected):
    assert limits_palm(env) is expected


def test_an_arm_takes_its_target_as_it_is_where_upstream_would_command_the_same_palm(monkeypatch):
    sim = arms(monkeypatch, Arms())
    sim._palm["right"].reach = None  # no IK runs
    assert sim._arm_target("right", Robot(), sim._commanded["right"].copy(), True) is sim._arm["right"]
    # A limiter that leaves the palm where the target puts it leaves the target too, bit for bit.
    sim = arms(monkeypatch, Arms(contact=True, limit=lambda pose: pose))
    assert sim._arm_target("right", Robot(), sim._commanded["right"].copy(), True) is sim._arm["right"]
    assert sim.env.limited == ["right"] and sim.guarded == {"right": 0, "left": 0}
    assert "arm_guard_steps" not in sim.measures()
    # Before the first step, or for an arm upstream would not command, the target stays.
    sim._commanded["right"] = None
    assert sim._arm_target("right", Robot(), np.zeros(7), True) is sim._arm["right"]
    assert sim._arm_target("left", Robot(), np.zeros(7), False) is sim._arm["left"]


def test_an_arm_takes_the_joints_that_put_the_palm_where_upstreams_palm_limits_put_it(monkeypatch):
    goal = HOME + [0., .25, 0., .3, 0., 0., 0.]  # lowers the palm
    height = ChainPalm()(HOME)[2]
    sim = arms(monkeypatch, Arms(contact=True, limit=floor(height)), goal)
    assert sim._commanded["right"][2] < height - .01
    for _ in range(2):  # a native guard may solve the step's targets twice; it counts once
        joints = sim._arm_target("right", Robot(), sim._commanded["right"].copy(), True)[0]
    palm = ChainPalm()(joints)
    assert palm[2] == pytest.approx(height, abs=1e-6)
    np.testing.assert_allclose(palm[:2], sim._commanded["right"][:2], atol=1e-6)
    np.testing.assert_allclose(np.abs(palm[3:] @ sim._commanded["right"][3:]), 1., atol=1e-9)
    assert sim.guarded == {"right": 1, "left": 0} and sim.measures()["arm_guard_steps"] == {"right": 1, "left": 0}


def test_an_arm_follows_a_native_guard_that_rewrote_its_commanded_palm(monkeypatch):
    # t10 pins the commanded palms' height until both hands grip the carton; no palm limiter runs.
    goal = HOME + [.1, -.2, 0., -.2, 0., 0., 0.]  # raises the palm
    sim = arms(monkeypatch, Arms(limit=None), goal)
    pinned = sim._commanded["right"].copy()
    pinned[2] = ChainPalm()(HOME)[2]
    joints = sim._arm_target("right", Robot(), pinned, True)[0]
    np.testing.assert_allclose(ChainPalm()(joints)[:3], pinned[:3], atol=1e-6)
    assert sim.env.limited == [] and sim.guarded["right"] == 1


def test_an_arm_comes_as_near_as_it_can_to_a_palm_no_joints_reach(monkeypatch):
    sim = arms(monkeypatch, Arms(contact=True, limit=floor(5.)), HOME)  # 5 m up: out of reach
    joints = sim._arm_target("right", Robot(), sim._commanded["right"].copy(), True)[0]
    assert ChainPalm()(joints)[2] > sim._commanded["right"][2] + .1 and sim.guarded["right"] == 1

