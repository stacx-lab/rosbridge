"""HandsOn's screened seeds, without Isaac: the table a launch's seed reads and the screen that writes it.

A pre-screen (tools/handson_seeds.py) tries each task's native seeds in the order 0, 1, 2, ... and keeps the first
`count` that upstream's settle gate accepts with every task body left in place. A launch's seed k names the task's
k-th kept seed.
"""
import json
from pathlib import Path

SEEDS = Path(__file__).resolve().parents[1] / "seeds.json"
SCHEMA = 1


def load(path=SEEDS):
    if not Path(path).is_file():
        raise ValueError(f"HandsOn has no screened seed table {path}; run tools/handson_seeds.py")
    return json.loads(Path(path).read_text())


def screened_seeds(task, table=None):
    """The native seeds a pre-screen accepted for a task, in order: launch seeds 0, 1, ... name them."""
    table = load() if table is None else table
    seeds = table["tasks"].get(task, {}).get("seeds", [])
    if len(seeds) < table["count"]:
        raise ValueError(f"{task} has {len(seeds)} of {table['count']} screened seeds; run tools/handson_seeds.py")
    return seeds


def screened_seed(task, seed, table=None):
    """The native seed a launch's seed names: the task's seed-th screened seed."""
    seeds = screened_seeds(task, table)
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < len(seeds):
        raise ValueError(f"{task} has seeds 0-{len(seeds) - 1}, the native seeds a pre-screen accepted; got {seed}")
    return seeds[int(seed)]


def reason(result):
    """Why a screened candidate was rejected: the settle gate, or the bodies the gate moved out of place."""
    if result["rejected"] == "scene":
        return "scene: " + ", ".join(f"{name} moved {body['moved_m']:.3f} m"
                                     for name, body in sorted(result["moved"].items()))
    return result["rejected"]


class Screen:
    """Which native seed each task screens next, and the verdicts so far.

    A task takes candidates in the order 0, 1, 2, ... while its accepted and running candidates are fewer than
    count, so its accepted seeds are the first count in that order however the runs finish. A candidate whose
    reset fails for another reason (accepted None) runs once more; a second error, or max_candidates seeds
    without count accepted, leaves the task incomplete.
    """

    def __init__(self, tasks, count, max_candidates=100, table=None):
        self.count, self.max_candidates = count, max_candidates
        # task -> {seed: None if accepted, else why it was rejected}
        self.verdicts = {task: {} for task in tasks}
        for task, entry in ((table or {}).get("tasks") or {}).items():
            if task in self.verdicts:
                self.verdicts[task].update({int(seed): why for seed, why in entry.get("rejected", {}).items()})
                self.verdicts[task].update(dict.fromkeys(entry.get("seeds", [])))
        self.running = {task: set() for task in tasks}
        self.errors = {task: {} for task in tasks}
        self.incomplete = {}  # task -> why

    def accepted(self, task):
        return sorted(seed for seed, why in self.verdicts[task].items() if why is None)

    def complete(self, task):
        return len(self.accepted(task)) >= self.count

    def next(self):
        """A (task, seed) to screen now, or None while every task is complete, incomplete or waits on its runs."""
        for task in self.verdicts:
            if task in self.incomplete or len(self.accepted(task)) + len(self.running[task]) >= self.count:
                continue
            seed = next((seed for seed in range(self.max_candidates)
                         if seed not in self.verdicts[task] and seed not in self.running[task]), self.max_candidates)
            if seed >= self.max_candidates:
                if not self.running[task]:
                    self.incomplete[task] = (f"{len(self.accepted(task))} of {self.count} seeds accepted among "
                                             f"0-{self.max_candidates - 1}")
                continue
            self.running[task].add(seed)
            return task, seed
        return None

    def record(self, task, seed, result):
        """A candidate's result: accepted True, False with rejected and its reason, or None with an error."""
        self.running[task].discard(seed)
        if result["accepted"] is None:
            self.errors[task][seed] = self.errors[task].get(seed, 0) + 1
            if self.errors[task][seed] > 1:
                self.incomplete.setdefault(task, f"seed {seed}: {result.get('error', 'no result')}")
            return
        self.verdicts[task][seed] = None if result["accepted"] else reason(result)

    def done(self):
        return not any(self.running.values()) and all(
            self.complete(task) or task in self.incomplete for task in self.verdicts)

    def table(self, **facts):
        """The table: each complete task's first count accepted seeds and the seeds rejected before the last of them.

        An incomplete task keeps its verdicts so far, so a rerun resumes it.
        """
        tasks = {}
        for task, verdicts in self.verdicts.items():
            accepted = self.accepted(task)
            kept = accepted[:self.count]
            last = kept[-1] if len(kept) == self.count else max(verdicts, default=-1)
            tasks[task] = dict(seeds=kept, rejected={str(seed): why for seed, why in sorted(verdicts.items())
                                                     if why is not None and seed <= last})
        return dict(schema=SCHEMA, count=self.count, candidates="0, 1, 2, ... in order", **facts, tasks=tasks)
