"""Backend conformance and shared timing without ROS, MuJoCo or ABC."""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
import threading
import time

import pytest

from rosbridge.core import CommandError
from tests.toy_backend import ToyBackend
from rosbridge.runtime.world import WorldRuntime



def world(mode="stepped", max_steps=None, **kwargs):
    published = []
    sim = ToyBackend(max_steps)
    runtime = WorldRuntime(sim, mode, lambda reset: published.append((sim.steps, reset)), **kwargs)
    return runtime, published


def wait(runtime, duration):
    """Admit a world wait on the owner thread; the future completes with its result."""
    result = Future()
    runtime.submit(lambda: runtime.waits.admit(duration, runtime.time_s, runtime.steps,
                                               runtime.sim.period, runtime.sim.done, result.set_result))
    return result


def reset(runtime, task, seed=0, randomize=False):
    return runtime.submit(lambda: runtime.reset(task, seed, randomize))


def finish(runtime):
    runtime.drain()
    while runtime.tick():
        pass


@pytest.mark.parametrize("duration", [-1., float("nan"), float("inf"), -float("inf"), 1e308])
def test_bad_wait_does_not_step(duration):
    runtime, _ = world()
    future = wait(runtime, duration)
    finish(runtime)
    assert not future.result().success
    assert runtime.steps == 0


def test_zero_rounding_and_shared_deadlines():
    runtime, published = world()
    zero = wait(runtime, 0.)
    first = wait(runtime, 0.21)
    same = wait(runtime, 0.21)
    longer = wait(runtime, 0.41)
    runtime.drain()
    assert zero.result().waited_s == 0
    assert runtime.steps == 0
    runtime.tick()
    later = wait(runtime, 0.2)
    runtime.drain()
    runtime.tick()
    assert not first.done()
    runtime.tick()
    assert first.result() == same.result()
    assert first.result().waited_s == pytest.approx(.3)
    assert later.result().waited_s == pytest.approx(.2)
    assert published[-1][0] == 3  # completion follows observation publication
    assert not longer.done()
    finish(runtime)
    assert longer.result().waited_s == pytest.approx(.5)
    assert runtime.steps == 5
    assert not runtime.tick()


def test_wait_completion_after_publication():
    runtime, published = world()
    future = wait(runtime, .1)
    future.add_done_callback(lambda _: published.append(("completed", False)))
    finish(runtime)
    assert published == [(1, False), ("completed", False)]


def test_reset_interrupts_with_old_time_and_discards_commands():
    runtime, _ = world()
    runtime.commands.set_group("finger", [-.5])
    future = wait(runtime, 10.)
    runtime.drain()
    runtime.tick()
    restarted = reset(runtime, "toy", 3, True)
    runtime.drain()
    assert restarted.result().success
    assert not future.result().success
    assert future.result().time_s == pytest.approx(.1)
    assert future.result().waited_s == pytest.approx(.1)
    assert runtime.time_s == 0 and runtime.sim.seed == 3
    assert runtime.commands.action(0).tolist() == [0., 0., 0.]
    assert not runtime.tick()


def test_failed_reset_preserves_episode_and_held_target():
    runtime, _ = world()
    runtime.commands.set_group("finger", [-.5])
    future = wait(runtime, 10.)
    runtime.drain()
    runtime.tick()
    restarted = reset(runtime, "missing", 3, False)
    runtime.drain()
    assert not restarted.result().success
    assert not future.result().success
    assert runtime.time_s == .1
    assert runtime.commands.action(10)[1] == -.5
    next_wait = wait(runtime, .1)
    finish(runtime)
    assert next_wait.result().success


def test_terminal_tick_completes_due_waits_then_fails_others():
    runtime, _ = world(max_steps=2)
    due = wait(runtime, .2)
    late = wait(runtime, .3)
    finish(runtime)
    assert due.result().success
    assert not late.result().success and late.result().waited_s == pytest.approx(.2)
    assert not runtime.tick()
    with pytest.raises(CommandError):
        runtime.commands.set_group("finger", [0.])
    zero = wait(runtime, 0.)
    future = wait(runtime, .1)
    runtime.drain()
    assert zero.result().success and not future.result().success


def test_every_wait_at_or_after_the_end_states_the_world_time_of_the_end():
    """A worker tells an early end (the goal held) from its time limit by the world time of the end; the reply never
    states the outcome. Pending waits keep the 'episode ended during wait' words workers' code looks for."""
    runtime, _ = world(max_steps=3)
    early = wait(runtime, .1)
    runtime.drain()
    runtime.tick()
    assert early.result().message == "ok" and early.result().success
    due, late = wait(runtime, .2), wait(runtime, 1.)
    finish(runtime)
    assert due.result().message == "ok; the episode ended at world time 0.30 s" and due.result().success
    assert late.result().message == "episode ended during wait; the episode ended at world time 0.30 s"
    assert not late.result().success and late.result().time_s == pytest.approx(.3)
    after, zero = wait(runtime, .5), wait(runtime, 0.)
    runtime.drain()
    assert after.result().message == "the episode ended at world time 0.30 s" and not after.result().success
    assert after.result().time_s == pytest.approx(.3) and after.result().waited_s == 0.
    assert zero.result().message == "ok; the episode ended at world time 0.30 s" and zero.result().success
    for result in (due, late, after, zero):
        assert "success" not in result.result().message and "fail" not in result.result().message


def test_realtime_runs_without_waits_and_wait_does_not_add_extra_ticks():
    now = [0.]
    runtime, _ = world("realtime", clock=lambda: now[0])
    assert runtime.tick() and not runtime.tick()
    future = wait(runtime, .2)
    runtime.drain()
    now[0] = .2
    assert runtime.tick() and runtime.tick() and not runtime.tick()
    assert future.result().waited_s == pytest.approx(.2)
    now[0] = .4
    assert runtime.tick()


def test_cross_thread_requests_and_owner_enforcement():
    runtime, _ = world()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = list(pool.map(lambda _: wait(runtime, .3), range(40)))
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(runtime.tick).result()
    finish(runtime)
    assert all(f.result().success for f in futures)
    assert runtime.steps == 3
    runtime.close()
    assert runtime.sim.closed
    with pytest.raises(RuntimeError, match="closed"):
        runtime.submit(lambda: None).result()


def test_shutdown_resolves_pending_and_fails_queued_requests():
    runtime, _ = world()
    admitted = wait(runtime, 1.)
    runtime.drain()
    runtime.tick()
    queued = runtime.submit(lambda: None)
    runtime.close()
    assert not admitted.result().success and admitted.result().waited_s == .1
    with pytest.raises(RuntimeError, match="stopped"):
        queued.result()


def test_terminal_result_is_persisted_once_per_episode(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR',str(tmp_path))
    sim = ToyBackend(max_steps=1)
    runtime = WorldRuntime(sim, 'stepped', lambda reset: None)
    for episode in range(2):
        pending = wait(runtime, 1.)
        runtime.drain()
        assert runtime.tick()
        assert not pending.result().success
        assert not runtime.tick()
        result = [json.loads(line) for line in (tmp_path/'episode-results.jsonl').read_text().splitlines()]
        assert len(result)==episode+1
        assert result[-1]['episode']==episode
        assert result[-1]['done'] and not result[-1]['success']
        assert result[-1]['step']==1
        restarted = reset(runtime, 'toy',0,False)
        runtime.drain()
        assert restarted.result().success
    runtime.close()


def test_judge_stop_reset_and_failed_reset(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world()
    wait(runtime, .2)
    finish(runtime)
    bad = reset(runtime, 'missing', 0, False)
    runtime.drain()
    assert not bad.result().success
    assert not (tmp_path / 'episode-results.jsonl').exists()
    restarted = reset(runtime, 'toy', 3, False)
    runtime.drain()
    assert restarted.result().success
    runtime.close()
    outcomes = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert [item['reason'] for item in outcomes] == ['reset', 'stopped']
    assert [item['step'] for item in outcomes] == [2, 0]
    assert [item['seed'] for item in outcomes] == [0, 3]
    assert all(not item['done'] for item in outcomes)


def test_failed_reset_that_loses_the_episode_records_its_outcome(tmp_path, monkeypatch):
    """A backend that reloads in place (BEHAVIOR) has no episode left after a failed reset: the episode it ended is
    recorded with the outcome measured before the attempt, and the world stays done until a reset succeeds."""
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world()
    wait(runtime, .2)
    finish(runtime)
    sim = runtime.sim

    def lose(task, seed, randomize):
        sim.done = True
        raise RuntimeError("reload failed")
    sim.reset = lose
    bad = reset(runtime, 'toy', 0, False)
    runtime.drain()
    assert not bad.result().success
    outcomes = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert [(item['episode'], item['reason'], item['step']) for item in outcomes] == [(0, 'reset_failed', 2)]
    with pytest.raises(CommandError):
        runtime.commands.set_group("finger", [0.])
    late = wait(runtime, .1)
    runtime.drain()
    assert not late.result().success and not runtime.tick()
    del sim.reset
    restarted = reset(runtime, 'toy', 0, False)
    runtime.drain()
    assert restarted.result().success
    runtime.close()
    outcomes = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert [(item['episode'], item['reason']) for item in outcomes] == [(0, 'reset_failed'), (1, 'stopped')]


def test_operator_judge_reads_results_without_ros(tmp_path, monkeypatch):
    import json
    import os
    from rosbridge.cli.main import main
    from rosbridge.runs.api import judge
    from rosbridge.runs.records import SCHEMA
    run = tmp_path / 'run.json'
    run.write_text(json.dumps({'schema': SCHEMA, 'uid': os.getuid(), 'id': 'test', 'state': 'stopped'}))
    assert judge(str(run))['state'] == 'pending'
    outcome = {'success': True, 'reward': 1., 'step': 2}
    (tmp_path / 'episode-results.jsonl').write_text(json.dumps(outcome) + '\n')
    assert judge(str(run))['episodes'] == [outcome]
    assert main(['judge', str(run), '--json']) == 0


def test_episode_records_count_clipped_and_refused_commands(tmp_path, monkeypatch):
    """Clipped and refused commands are silent to a topic publisher; each episode's record counts them."""
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world()
    for name in ("commands_clipped", "commands_clipped", "commands_refused"):
        runtime.submit(lambda name=name: runtime.count(name))
    runtime.drain()
    reset(runtime, 'toy')
    runtime.drain()
    runtime.close()
    outcomes = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert [(o['reason'], o['commands_clipped'], o['commands_refused']) for o in outcomes] == [
        ('reset', 2, 1), ('stopped', 0, 0)]
    with ThreadPoolExecutor(1) as pool, pytest.raises(RuntimeError, match="owner thread"):
        pool.submit(runtime.count, "commands_clipped").result()


def test_a_world_stopped_by_an_error_records_it(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world()
    pending = wait(runtime, 1.)
    runtime.drain()
    runtime.close(error="WorkerError: native worker exited (exit -9)")
    assert pending.result().message == "runtime stopped"  # all a worker sees
    outcome, = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert (outcome['reason'], outcome['error']) == ('stopped', "WorkerError: native worker exited (exit -9)")


class SlowTerminal(ToyBackend):
    """A terminal step that blocks its thread until released, as a cooking transition did (~130 s of wall
    time in one step while 268 macro particles were replaced one at a time, each rebuilding OmniGibson's physics views)."""
    def __init__(self, terminal, release):
        self.terminal, self.release, self.entered = terminal, release, threading.Event()
        super().__init__()

    def step(self, action):
        if self.steps + 1 == self.terminal:
            self.entered.set()
            self.release.wait()
        super().step(action)
        self.done = self.success = self.steps >= self.terminal


def test_a_wait_queued_behind_a_slow_terminal_step_gets_the_end_time():
    """A client that gave up on its wait during a long step finds out on its next wait when the episode ended."""
    release, stop, ready, holder = threading.Event(), threading.Event(), threading.Event(), {}

    def owner():  # the world node's main loop
        holder["runtime"] = runtime = WorldRuntime(SlowTerminal(3, release), "stepped", lambda reset: None)
        ready.set()
        while not stop.is_set():
            runtime.drain()
            if not runtime.tick():
                runtime.drain(runtime.idle_s())

    thread = threading.Thread(target=owner, daemon=True)
    thread.start()
    assert ready.wait(5)
    runtime = holder["runtime"]
    try:
        first = wait(runtime, 1.)
        assert runtime.sim.entered.wait(5)
        with pytest.raises(FutureTimeout):
            first.result(timeout=.2)  # the client's own timeout fires while the step runs
        second = wait(runtime, .1)
        time.sleep(.1)
        assert not second.done()
        release.set()
        assert first.result(timeout=5).message == "episode ended during wait; the episode ended at world time 0.30 s"
        assert second.result(timeout=5).message == "the episode ended at world time 0.30 s"
    finally:
        release.set()
        stop.set()
        thread.join(5)
