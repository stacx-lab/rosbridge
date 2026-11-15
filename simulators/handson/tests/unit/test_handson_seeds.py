"""HandsOn's screened seeds, without Isaac: a launch's seed names a screened native seed, and the screen keeps
each task's first accepted seeds in order."""
import pytest

from rosbridge_handson.seeds import Screen, screened_seed

GATED = {"1": "settle_gate", "3": "settle_gate", "4": "settle_gate"}
TABLE = {"count": 3, "tasks": {"t24_place_ball_in_bowl": {"seeds": [0, 2, 5], "rejected": GATED},
                               "t09_insert_peg": {"seeds": [1], "rejected": {"0": "scene: tray moved 0.061 m"}}}}
ACCEPTED, GATE = dict(accepted=True), dict(accepted=False, rejected="settle_gate")
ERROR = dict(accepted=None, error="RuntimeError: Kit crashed")


def test_a_launch_seed_names_the_tasks_screened_native_seed():
    assert [screened_seed("t24_place_ball_in_bowl", seed, TABLE) for seed in range(3)] == [0, 2, 5]
    for seed in (3, -1, 1.5, True):
        with pytest.raises(ValueError, match=r"seeds 0-2"):
            screened_seed("t24_place_ball_in_bowl", seed, TABLE)
    with pytest.raises(ValueError, match="1 of 3 screened seeds"):  # an incomplete screen names no seeds
        screened_seed("t09_insert_peg", 0, TABLE)
    with pytest.raises(ValueError, match="0 of 3 screened seeds"):
        screened_seed("t27_place_largest_fitting_ball", 0, TABLE)


def test_the_screen_keeps_the_first_accepted_seeds_however_runs_finish():
    screen = Screen(["t24"], count=3)
    assert [screen.next() for _ in range(4)] == [("t24", 0), ("t24", 1), ("t24", 2), None]
    screen.record("t24", 2, ACCEPTED)
    screen.record("t24", 1, dict(accepted=False, rejected="scene", moved={"bowl": {"moved_m": .5}}))
    assert screen.next() == ("t24", 3)  # 2 accepted or running and 3 is the next candidate
    screen.record("t24", 0, ACCEPTED)
    screen.record("t24", 3, ACCEPTED)
    assert screen.next() is None and screen.done()
    assert screen.table()["tasks"]["t24"] == {"seeds": [0, 2, 3], "rejected": {"1": "scene: bowl moved 0.500 m"}}


def test_an_error_runs_once_more_and_a_second_leaves_the_task_incomplete():
    screen = Screen(["t12"], count=1)
    assert screen.next() == ("t12", 0)
    screen.record("t12", 0, ERROR)
    assert screen.next() == ("t12", 0)
    screen.record("t12", 0, ERROR)
    assert screen.next() is None and screen.done()
    assert screen.incomplete == {"t12": "seed 0: RuntimeError: Kit crashed"}


def test_a_task_without_enough_accepted_seeds_among_its_candidates_is_incomplete():
    screen = Screen(["t27"], count=2, max_candidates=3)
    while (job := screen.next()) is not None:
        screen.record(*job, ACCEPTED if job[1] == 1 else GATE)
    assert screen.done() and screen.incomplete == {"t27": "1 of 2 seeds accepted among 0-2"}
    assert screen.table()["tasks"]["t27"] == {"seeds": [1], "rejected": {"0": "settle_gate", "2": "settle_gate"}}


def test_a_rerun_resumes_and_a_larger_count_extends_the_table():
    screen = Screen(["t24_place_ball_in_bowl"], count=4, table=TABLE)
    assert screen.next() == ("t24_place_ball_in_bowl", 6)
    screen.record("t24_place_ball_in_bowl", 6, ACCEPTED)
    assert screen.table()["tasks"]["t24_place_ball_in_bowl"] == {"seeds": [0, 2, 5, 6], "rejected": GATED}
    smaller = Screen(["t24_place_ball_in_bowl"], count=2, table=TABLE)
    assert smaller.next() is None
    # Rejections after the last kept seed leave the table.
    assert smaller.table()["tasks"]["t24_place_ball_in_bowl"] == {"seeds": [0, 2], "rejected": {"1": "settle_gate"}}


def test_the_table_names_ten_screened_seeds_for_every_task():
    from rosbridge.catalog import tasks
    from rosbridge_handson.seeds import load
    table = load()
    assert table["count"] == 10 and table["drift_m"] == .02
    assert set(table["tasks"]) == {task["native_task"] for task in tasks(name="handson").values()}
    for task, entry in table["tasks"].items():
        seeds, rejected = entry["seeds"], {int(seed) for seed in entry["rejected"]}
        # The first ten accepted native seeds, each one before the last kept seed accepted or rejected.
        assert len(seeds) == 10 and seeds == sorted(set(seeds)), task
        assert set(seeds) | rejected == set(range(seeds[-1] + 1)) and not set(seeds) & rejected, task
