"""HandsOn's trial scoring without Isaac: a native step's info and adapter reason in."""
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


def episode(steps, **options):
    """Settle `steps` native steps, each an (info, reason) pair."""
    sim = HandsOnSim(max_steps=10, **options)
    sim.steps = 0
    for info, reason, _ in steps:
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


def test_a_step_that_succeeds_and_fails_at_once_ends_a_stop_on_success_run_as_a_success():
    sim = episode([PLAIN, ({"succeed": True, "failed": True}, "native_terminated", {})], terminate_on_success=True)
    assert sim.done and sim.success and sim.reward == 1. and sim.failure is None and sim.end_reason() is None
