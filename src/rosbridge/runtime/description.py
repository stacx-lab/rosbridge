"""An episode's task description: its native task and seed, and the instructions upstream's evaluator gives its
policy, as the backend states them after its reset (its `description`).

The world node states it in the run log as one line when the episode starts, and `rosbridge describe` reads that
line back from the live run, so describing a run never opens its episode again.
"""
import json

PREFIX = "task description: "


def task_description(task, sim):
    return {"task": task or sim.default_task, "seed": sim.seed, **sim.description}


def logged(description):
    """The run log's line: the prefix, then the description as one line of JSON."""
    return PREFIX + json.dumps(description, ensure_ascii=False)


def stated(log):
    """The task description a run log states, or None before its episode has started."""
    with open(log, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            _, found, rest = line.partition(PREFIX)
            if found:
                return json.loads(rest)
    return None
