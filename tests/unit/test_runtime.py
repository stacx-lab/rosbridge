"""Backend conformance and shared timing without ROS, MuJoCo or ABC."""
from concurrent.futures import Future, ThreadPoolExecutor

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
