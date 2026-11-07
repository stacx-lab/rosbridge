"""Backend conformance and shared timing without ROS, MuJoCo or ABC."""
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from rosbridge.core import CommandError
from tests.toy_backend import ToyBackend
from rosbridge.runtime.world import WorldRuntime



def world(mode="stepped", max_steps=None, backend=ToyBackend, **kwargs):
    published = []
    sim = backend(max_steps)
    runtime = WorldRuntime(sim, mode, lambda: published.append(sim.steps), **kwargs)
    return runtime, published


def wait(runtime, duration):
    """Admit a world wait on the owner thread; the future completes with its result."""
    result = Future()
    runtime.submit(lambda: runtime.waits.admit(duration, runtime.time_s, runtime.steps,
                                               runtime.sim.period, runtime.sim.done, result.set_result))
    return result


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
    assert published[-1] == 3  # completion follows observation publication
    assert not longer.done()
    finish(runtime)
    assert longer.result().waited_s == pytest.approx(.5)
    assert runtime.steps == 5
    assert not runtime.tick()


def test_wait_completion_after_publication():
    runtime, published = world()
    future = wait(runtime, .1)
    future.add_done_callback(lambda _: published.append("completed"))
    finish(runtime)
    assert published == [1, "completed"]


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


def test_the_episode_is_recorded_once_when_it_ends(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=1)
    pending = wait(runtime, 1.)
    runtime.drain()
    assert runtime.tick()
    assert not pending.result().success
    assert not runtime.tick()
    runtime.close()  # stopping a run whose episode has ended adds no second record
    outcome, = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert outcome['done'] and not outcome['success']
    assert (outcome['reason'], outcome['step']) == ('max_steps', 1)


def test_stopping_records_the_open_episode(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world()
    wait(runtime, .2)
    finish(runtime)
    assert not (tmp_path / 'episode-results.jsonl').exists()
    runtime.close()
    outcome, = [json.loads(line) for line in (tmp_path / 'episode-results.jsonl').read_text().splitlines()]
    assert (outcome['reason'], outcome['step'], outcome['seed'], outcome['done']) == ('stopped', 2, 0, False)


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


class MeasuredToy(ToyBackend):
    """A backend with its own measure of the episode, which fails once the position goes negative."""
    measured = 0

    def measures(self):
        self.measured += 1
        if self.position.min() < 0:
            raise RuntimeError("cannot measure a negative position")
        return {"progress": self.steps / 4}


def records(folder):
    import json
    return [json.loads(line) for line in (folder / 'episode-results.jsonl').read_text().splitlines()]


def test_records_add_a_backends_measures(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=1, backend=MeasuredToy)
    wait(runtime, .1)
    finish(runtime)
    runtime.close()
    outcome, = records(tmp_path)
    assert (outcome['reason'], outcome['progress'], runtime.sim.measured) == ('max_steps', .25, 1)


def test_a_world_stopped_by_an_error_records_no_measures(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(backend=MeasuredToy)
    runtime.close(error="WorkerError: native worker exited (exit -9)")
    outcome, = records(tmp_path)
    assert 'progress' not in outcome and runtime.sim.measured == 0  # the error may have broken the backend
    assert outcome['error'] == "WorkerError: native worker exited (exit -9)"


def test_a_measure_that_fails_is_the_records_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(backend=MeasuredToy)
    runtime.sim.position[:] = -1.
    runtime.close()
    outcome, = records(tmp_path)
    assert outcome['reason'] == 'stopped' and 'progress' not in outcome
    assert outcome['error'] == "measures: RuntimeError: cannot measure a negative position"
    assert "Traceback" in capsys.readouterr().err and runtime.sim.closed
