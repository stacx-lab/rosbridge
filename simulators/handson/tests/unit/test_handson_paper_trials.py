"""HandsOn's optional archived paper trials, without Isaac: the seed names upstream's trial of a task and model."""
from pathlib import Path

import pytest

from rosbridge_handson.backend import paper_trial

# Upstream's evaluation package, once rosbridge prepare handson has exported it; it imports without Isaac.
UPSTREAM = Path(__file__).resolve().parents[4] / ".runtime/handson/handson"


@pytest.fixture
def upstream(monkeypatch):
    if not (UPSTREAM / "handson/evaluation/paper_trials.py").is_file():
        pytest.skip("HandsOn is not prepared")
    monkeypatch.syspath_prepend(str(UPSTREAM))


def test_the_seed_names_an_archived_paper_trial_only_with_the_option(upstream, monkeypatch):
    monkeypatch.delenv("ROSBRIDGE_HANDSON_PAPER_TRIALS", raising=False)
    assert paper_trial("t24_place_ball_in_bowl", 3) is None
    monkeypatch.setenv("ROSBRIDGE_HANDSON_PAPER_TRIALS", "pi05")
    trials = [paper_trial("t24_place_ball_in_bowl", index) for index in range(100)]
    assert [t["index"] for t in trials] == list(range(100)) and all(t["model"] == "pi05" for t in trials)
    assert {t["goal"]["ball"] for t in trials} == {"golf ball", "mini soccer ball", "tennis ball"}
    for task, index, message in (("eval_gen03_orient_bottle_upright", 0, "Core-50 tasks only"),
                                 ("t24_place_ball_in_bowl", 100, "0-99")):
        with pytest.raises(ValueError, match=message):
            paper_trial(task, index)
    monkeypatch.setenv("ROSBRIDGE_HANDSON_PAPER_TRIALS", "no_such_model")
    with pytest.raises(ValueError, match="no archived model"):
        paper_trial("t24_place_ball_in_bowl", 0)
