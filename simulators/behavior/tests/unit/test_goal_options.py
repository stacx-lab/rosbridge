import itertools
import random

import pytest

from rosbridge_behavior.backend import GoalOptions


class Literal:
    """A grounded BDDL literal as bddl compiles it: its tokens in body, evaluated through evaluate_fn."""

    def __init__(self, body):
        self.body = body

    def evaluate(self, evaluate_fn):
        if self.body[0] == "not":
            return not evaluate_fn(*self.body[1])
        return evaluate_fn(*self.body)


def omnigibson_masks(options, evaluate_fn):
    """OmniGibson's get_goal_option_satisfaction: every literal of every option."""
    return [[bool(node.evaluate(evaluate_fn)) for node in option] for option in options]


def omnigibson_q_score(success, now, initial):
    """omnigibson.metrics.task_metric.compute_q_score."""
    if success:
        return 1.0
    if not now:
        return 0.0
    scores = [sum(int((not i) and n) for n, i in zip(now_opt, init_opt)) / len(now_opt) if now_opt else 0.0
              for now_opt, init_opt in zip(now, initial)]
    return max(scores)


def quantified_options():
    """forall bacon cooked, not fridge open, and an exists over two baskets per egg: options recompiled per option,
    so equal literals are distinct objects, as bddl's get_ground_state_options makes them."""
    eggs, baskets = ("egg1", "egg2", "egg3"), ("basket1", "basket2")
    options = []
    for choice in itertools.product(baskets, repeat=len(eggs)):
        option = [Literal(["cooked", "bacon1"]), Literal(["cooked", "bacon2"]), Literal(["not", ["open", "fridge"]])]
        option += [Literal(["inside", egg, basket]) for egg, basket in zip(eggs, choice)]
        options.append(option)
    return options


@pytest.mark.parametrize("seed", range(5))
def test_masks_and_q_score_equal_omnigibsons(seed):
    rng = random.Random(seed)
    options = quantified_options()
    goal = GoalOptions(options)
    assert len(goal.literals) == 3 + 3 * 2 and goal.rows.shape == (8, 6)
    calls = []

    def state(values):
        def evaluate(name, *entities):
            calls.append((name, *entities))
            return values[(name, *entities)]
        return evaluate

    keys = {(n.body[1][0], *n.body[1][1:]) if n.body[0] == "not" else tuple(n.body) for o in options for n in o}
    start = {k: rng.random() < .3 for k in keys}
    later = {k: rng.random() < .6 for k in keys}
    for values in (start, later):
        calls.clear()
        truth = goal.truth(state(values))
        assert len(calls) == len(set(calls)) == len(keys)  # each predicate once per call
        assert goal.masks(truth) == omnigibson_masks(options, state(values))
    initial, now = goal.truth(state(start)), goal.truth(state(later))
    for success in (False, True):
        assert goal.q_score(success, now, initial) == pytest.approx(omnigibson_q_score(
            success, omnigibson_masks(options, state(later)), omnigibson_masks(options, state(start))))


def test_options_of_different_lengths_and_none():
    options = [[Literal(["toggled_on", "radio"])], [Literal(["open", "a"]), Literal(["open", "b"])], []]
    goal = GoalOptions(options)
    values = {("toggled_on", "radio"): False, ("open", "a"): True, ("open", "b"): False}
    truth = goal.truth(lambda *key: values[key])
    assert goal.masks(truth) == [[False], [True, False], []]
    assert goal.q_score(False, truth, [False, False, False]) == .5
    assert GoalOptions([]).q_score(False, [], []) == 0.0
