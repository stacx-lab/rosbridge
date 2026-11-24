"""The arms' move action without Isaac: how a goal paces its steps and ends, and the fields it reads and gives."""
import numpy as np
import pytest

from rosbridge_robodojo import move

UPRIGHT = (0., 0., 0., 1.)


def pose(x, y, quaternion=UPRIGHT):
    return np.array([x, y, 0., *quaternion])


def goal(*poses, times=None, **fields):
    """The fields of a move Goal through poses, as rosidl gives them."""
    points = [{"transforms": [{"translation": dict(zip("xyz", map(float, p[:3]))),
                               "rotation": dict(zip("xyzw", map(float, p[3:])))}],
               "time_from_start": {"sec": int(t), "nanosec": round(t % 1 * 1e9)}}
              for p, t in zip(poses, times or [0.] * len(poses))]
    return {"multi_dof_trajectory": {"header": {"frame_id": ""}, "joint_names": ["flange"], "points": points},
            "component_goal_tolerance": [], "goal_time_tolerance": {"sec": 0, "nanosec": 0}, **fields}


def test_a_motion_steps_from_the_measured_pose_at_its_pace():
    half_turn = (0., 0., 1., 0.)
    motion = move.Motion(pose(0., 0.), 0., .1, [pose(1., 0.), pose(1., 0., half_turn)], [0., 0.])
    assert motion.reference() == pytest.approx(pose(.1, 0.))  # 1 m/s for a 0.1 s step
    assert motion.observe(pose(.998, 0.), .1) is None and motion.index == 1  # within tolerance: on to the next pose
    turned = motion.reference()
    assert turned[:3] == pytest.approx([1., 0., 0.]) and 2 * np.arccos(turned[6]) == pytest.approx(.33)
    slow = move.Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [4.])
    assert slow.reference() == pytest.approx(pose(.025, 0.))  # a quarter of the pace for four times the time
    assert move.Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [.5]).reference() == pytest.approx(pose(.1, 0.))


def test_a_motion_ends_reached_late_or_stalled():
    arrive = move.Motion(pose(0., 0.), 0., .1, [pose(.2, 0.)], [0.])
    assert arrive.observe(pose(.1, 0.), .1) is None
    assert arrive.observe(pose(.197, 0.), .2) == (move.SUCCESSFUL, "reached")  # within 5 mm
    # A tip that lags is late settle_s after the goal's nominal end, the 1 s its pace needs.
    late = move.Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [0.], settle_s=.2)
    outcomes = [late.observe(pose(.05 * step, 0.), .1 * step) for step in range(1, 13)]
    assert outcomes[:11] == [None] * 11 and outcomes[11] == (move.GOAL_TOLERANCE_VIOLATED, "not reached in time")
    # One that gains nothing over four steps has stalled.
    blocked = move.Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [0.])
    outcomes = [blocked.observe(pose(min(.1 * step, .2), 0.), .1 * step) for step in range(1, 6)]
    assert outcomes == [None] * 4 + [(move.PATH_TOLERANCE_VIOLATED, "stalled")]
    assert blocked.report("stalled", .5) == "stalled: 0.800 m and 0.000 rad from the goal after 0.50 s"


def test_a_goal_gives_its_poses_times_and_tolerances_or_says_why_it_is_none():
    start = pose(0., 0.)
    motion = move.motion(goal(pose(.5, 0.), pose(1., 0.), times=[0., 2.]), "flange", "left_arm/base", start, 3., .1)
    assert [point[0] for point in motion.points] == [.5, 1.] and motion.tolerance == pytest.approx(move.TOLERANCE)
    assert motion.steps[0] == pytest.approx(move.SPEED * .1) and motion.steps[1][0] == pytest.approx(.025)
    assert motion.deadline_s == pytest.approx(3. + .5 + 2. + move.SETTLE_S)
    loose = move.motion(goal(pose(1., 0.), component_goal_tolerance=[
        {"joint_name": "flange", "component": move.TRANSLATION, "position": .1}],
        goal_time_tolerance={"sec": 2, "nanosec": 500_000_000}), "flange", "left_arm/base", start, 0., .1)
    assert loose.tolerance == pytest.approx([.1, move.TOLERANCE[1]]) and loose.deadline_s == pytest.approx(3.5)
    for bad, reason in ((goal(), "a goal is a multi_dof_trajectory of flange poses in left_arm/base"),
                        (goal(pose(1., 0., (0., 0., 0., 0.))), "goal pose needs a unit quaternion")):
        with pytest.raises(ValueError, match=reason):
            move.motion(bad, "flange", "left_arm/base", start, 0., .1)
    named = goal(pose(1., 0.))
    named["multi_dof_trajectory"]["joint_names"] = ["palm"]
    with pytest.raises(ValueError, match="a goal is a multi_dof_trajectory of flange poses"):
        move.motion(named, "flange", "left_arm/base", start, 0., .1)


def test_feedback_and_results_are_the_actions_fields():
    sample = move.feedback("flange", "left_arm/base", 1.5, pose(.1, 0.), pose(0., 0.))
    assert sample["header"] == {"stamp": {"sec": 1, "nanosec": 500_000_000}, "frame_id": "left_arm/base"}
    assert sample["multi_dof_desired"]["transforms"][0]["translation"] == {"x": .1, "y": 0., "z": 0.}
    assert sample["multi_dof_actual"]["transforms"][0]["rotation"]["w"] == 1.
    assert move.ended(move.SUCCESSFUL, "reached", {"left_arm": [0.] * 6}) == {
        "status": "succeeded", "feedback": None, "commands": {"left_arm": [0.] * 6},
        "result": {"error_code": 0, "error_string": "reached"}}
    assert move.ended(move.PATH_TOLERANCE_VIOLATED, "stalled", {})["status"] == "aborted"
