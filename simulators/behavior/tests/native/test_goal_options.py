"""OmniGibson's goal option satisfaction, the evaluator's q_score input, as patches/behavior/0003
makes it: bddl's per-option evaluation, with each distinct grounded literal evaluated once per
call over an index built at the first. Needs the generated tree's bddl and OmniGibson, not an
Isaac app."""
import random
from types import SimpleNamespace

import pytest
from bddl.condition_evaluation import evaluate_state
from bddl.knowledge_base import KnowledgeBase
from omnigibson.tasks.behavior_task import BehaviorTask


@pytest.fixture(scope="module")
def knowledge():
    return KnowledgeBase(verbose=False)


# One option of seven literals; 46,656 options repeating 36 literals (quantifiers over six logs).
@pytest.mark.parametrize("activity", ["cook_bacon", "stacking_wood"])
def test_each_distinct_literal_is_evaluated_once_per_call_with_bddls_values(knowledge, activity):
    options = knowledge.get_task(f"{activity}-0").compile(scene_layout={}).ground_goal_state_options
    distinct = len({repr(literal.body) for option in options for literal in option})
    rng, truth, calls = random.Random(0), {}, []

    def evaluate_predicate(env_idx, name, *entities):
        calls.append(env_idx)
        return truth.setdefault((name, *entities), rng.random() < .5)

    task = SimpleNamespace(ground_goal_state_options=options, _evaluate_predicate=evaluate_predicate,
                           _goal_option_index=None)
    for call in range(2):  # the second reuses the first's index, on a changed state
        calls.clear()
        masks = BehaviorTask.get_goal_option_satisfaction(task, 0)
        if call == 0:
            index = task._goal_option_index
        assert task._goal_option_index is index
        assert len(calls) == distinct and set(calls) == {0}
        reference = []
        for option in options:
            _, results = evaluate_state(option, lambda name, *entities: truth[(name, *entities)])
            reference.append([i in results["satisfied"] for i in range(len(option))])
        assert masks == reference and any(map(any, masks)) and not all(map(all, masks))
        truth = {key: not value for key, value in truth.items()}
