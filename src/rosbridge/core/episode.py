"""One episode of a simulator's backend, opened as a run opens it, and the task description it gives the agent.

The world node opens a run's episode and prints its task description to the run log; `rosbridge describe` opens
the same episode once and prints its description (rosbridge.runs.describe). Both open it here, so the agent reads
the description of the episode it then runs.
"""
from .registry import load_backend


def open_episode(simulator, task, seed, *, configuration=None, scene=None, max_steps=None, early_stop=False,
                 randomize=True, before_reset=None):
    """The simulator's backend, created and reset for one episode: with configuration only when one is selected,
    and the scene options as reset's keyword arguments. before_reset(sim) runs in between."""
    options = {"configuration": configuration} if configuration else {}
    sim = load_backend(simulator)(max_steps=max_steps, early_stop=early_stop, **options)
    try:
        if before_reset is not None:
            before_reset(sim)
        sim.reset(task or sim.default_task, seed, randomize, **(scene or {}))
    except BaseException:
        sim.close()
        raise
    return sim


def task_description(task, sim):
    """The episode's native task and seed, and the instructions upstream's evaluator gives its policy, as the
    backend states them after its reset (its `description`)."""
    return {"task": task or sim.default_task, "seed": sim.seed, **sim.description}
