"""HandsOn's trial scoring, optional partial credit and native commands, without Isaac: a native step's info and adapter
reason in, its EEF52 command out."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge_handson.backend import HandsOnSim, judge


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

