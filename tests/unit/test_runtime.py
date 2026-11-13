"""Backend conformance and shared timing without ROS, MuJoCo or ABC."""
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from rosbridge.core import CommandError
from rosbridge.core.worker import WorkerError
from tests.toy_backend import StumblingToy, ToyBackend
from rosbridge.runtime.world import WorldRuntime



TIME_LIMIT = "the episode ended because the time limit was reached"


def world(mode="stepped", max_steps=None, backend=ToyBackend, task="toy", stop_on_success=False, stop_on_failure=False,
          **kwargs):
    published = []
    sim = backend(max_steps, stop_on_success)
    if stop_on_failure:
        sim.terminate_on_failure = True
    sim.reset(task, 0, False)
    runtime = WorldRuntime(sim, mode, lambda: published.append(sim.steps), stop_on_success, **kwargs)
    return runtime, published


def wait(runtime, duration):
    """Admit a world wait on the owner thread; the future completes with its result."""
    result = Future()
    runtime.submit(lambda: runtime.waits.admit(duration, runtime.time_s, runtime.steps,
                                               runtime.sim.period, runtime.ended, result.set_result))
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
    early = wait(runtime, .1)
    due = wait(runtime, .2)
    late = wait(runtime, .3)
    finish(runtime)
    assert early.result().message == "ok"
    assert due.result().success and due.result().message == f"ok; {TIME_LIMIT}"
    assert not late.result().success and late.result().waited_s == pytest.approx(.2)
    assert late.result().message == f"episode ended during wait; {TIME_LIMIT}"
    assert not runtime.tick()
    with pytest.raises(CommandError):
        runtime.commands.set_group("finger", [0.])
    zero = wait(runtime, 0.)
    future = wait(runtime, .1)
    runtime.drain()
    assert zero.result().success and zero.result().message == f"ok; {TIME_LIMIT}"
    assert not future.result().success and future.result().message == f"episode is done; {TIME_LIMIT}"


@pytest.mark.parametrize("stop_on_success, max_steps, end_step, cause", [
    (True, 5, 2, "the task succeeded"),
    (True, 2, 2, "the task succeeded"),  # its first success on the limit's own step
    (False, 3, 3, "the time limit was reached"),  # the goal holds at the limit of a run that goes on after success
])
def test_a_wait_reply_says_why_the_episode_ended(stop_on_success, max_steps, end_step, cause):
    runtime, _ = world(max_steps=max_steps, task="succeeds", stop_on_success=stop_on_success)
    pending = wait(runtime, 1.)
    finish(runtime)
    assert runtime.steps == end_step and runtime.sim.success
    assert pending.result().message == f"episode ended during wait; the episode ended because {cause}"


def replied(runtime, duration):
    """A world wait as the world node admits it: its reply names the commands refused since the previous one."""
    result = Future()
    runtime.submit(lambda: runtime.waits.admit(duration, runtime.time_s, runtime.steps, runtime.sim.period,
                                               runtime.ended, lambda done: result.set_result(runtime.reply(done))))
    return result


def refuse(runtime, endpoint, reason, times=1):
    for _ in range(times):
        runtime.submit(lambda: runtime.refuse(endpoint, CommandError(reason)))


def test_a_wait_reply_names_the_commands_refused_since_the_previous_reply():
    runtime, _ = world()
    plain = replied(runtime, 0.)
    refuse(runtime, "/arm/joint_target", "arm_joint1=9.0000 outside [-1.0000, 1.0000]", times=2)
    refuse(runtime, "/finger/target", "finger target has non-finite values")
    named = replied(runtime, .1)
    finish(runtime)
    assert plain.result().message == "ok"
    assert named.result().success and named.result().message == (
        "ok; refused since the previous wait reply: /arm/joint_target: arm_joint1=9.0000 outside [-1.0000, 1.0000] "
        "(2 commands); /finger/target: finger target has non-finite values")
    again = replied(runtime, 0.)
    runtime.drain()
    assert again.result().message == "ok"  # named once
    for kind in range(7):
        refuse(runtime, f"/device{kind}/target", "device target has non-finite values", times=kind + 1)
    many = replied(runtime, 0.)
    runtime.drain()
    message = many.result().message
    assert "/device4/target" in message and "/device5/target" not in message
    assert message.endswith("(5 commands); and 13 more")  # devices 5 and 6: 6 + 7 commands


def test_a_refusal_reaches_a_wait_that_the_end_or_stop_completes():
    runtime, _ = world(max_steps=2)
    pending = replied(runtime, 1.)
    runtime.drain()
    runtime.tick()
    refuse(runtime, "/arm/joint_target", "trajectory start is in the past")
    finish(runtime)
    assert not pending.result().success
    assert pending.result().message == (f"episode ended during wait; {TIME_LIMIT}; refused since the previous "
                                        "wait reply: /arm/joint_target: trajectory start is in the past")
    runtime, _ = world()
    stopped = replied(runtime, 1.)
    runtime.drain()
    runtime.refuse("/finger/target", "finger target has non-finite values")
    runtime.close()
    assert stopped.result().message == ("runtime stopped; refused since the previous wait reply: /finger/target: "
                                        "finger target has non-finite values")


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


FAILED = "the episode ended because the simulation failed"


@pytest.mark.parametrize("stop_on_success, stop_on_failure, max_steps, end_step, reason", [
    (True, False, 5, 2, "failure"),  # a run launched to stop on success ends at its first outcome, here a failure
    (True, False, 2, 2, "failure"),  # a failure on the limit's own step is the failure
    (False, True, 5, 2, "failure"),  # a backend created to stop on failure ends there too, without stop_on_success
    (False, False, 4, 4, "max_steps"),  # otherwise the failure latches unannounced and the episode goes on, as before
])
def test_a_backend_can_end_the_episode_at_an_outcome_of_its_own(tmp_path, monkeypatch, stop_on_success, stop_on_failure,
                                                                 max_steps, end_step, reason):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, published = world(max_steps=max_steps, backend=StumblingToy, task="stumbles",
                               stop_on_success=stop_on_success, stop_on_failure=stop_on_failure)
    pending = wait(runtime, 1.)
    finish(runtime)
    cause = "the task failed" if reason == "failure" else "the time limit was reached"
    assert runtime.steps == end_step and published[-1] == end_step and runtime.sim.failure == "toy_stumbled"
    assert pending.result().message == f"episode ended during wait; the episode ended because {cause}"
    zero = wait(runtime, 0.)
    runtime.drain()
    assert zero.result().message == f"ok; the episode ended because {cause}"
    runtime.close()
    outcome, = records(tmp_path)
    assert (outcome['reason'], outcome['done'], outcome['step'], outcome['success']) == (reason, True, end_step, False)
    # The native cause goes only to the record, and only for the backend's own end.
    assert outcome.get('cause') == ("toy_stumbled" if reason == "failure" else None)


def test_a_backend_without_its_own_ends_ends_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    assert not hasattr(ToyBackend, "end_reason")
    runtime, _ = world(max_steps=5, task="succeeds", stop_on_success=True)
    wait(runtime, 1.)
    finish(runtime)
    runtime.close()
    outcome, = records(tmp_path)
    assert (outcome['reason'], outcome['step']) == ('success', 2) and 'cause' not in outcome


def test_a_step_that_fails_ends_the_episode_and_the_world_goes_on_answering(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, published = world(backend=MeasuredToy, task="fails")
    pending = wait(runtime, 1.)
    finish(runtime)
    assert runtime.steps == 1 and published == [1]  # the failed step publishes nothing
    assert not pending.result().success and pending.result().waited_s == pytest.approx(.1)
    assert pending.result().message == f"episode ended during wait; {FAILED} (FloatingPointError)"
    zero, later = wait(runtime, 0.), wait(runtime, .1)
    runtime.drain()
    assert zero.result().message == f"ok; {FAILED} (FloatingPointError)"
    assert later.result().message == f"episode is done; {FAILED} (FloatingPointError)"
    with pytest.raises(CommandError):
        runtime.commands.set_group("finger", [0.])
    runtime.close()  # stopping adds no second record
    outcome, = records(tmp_path)
    assert (outcome['reason'], outcome['done'], outcome['step']) == ('error', True, 1)
    assert outcome['error'] == "FloatingPointError: the toy's physics failed"
    assert 'progress' not in outcome and runtime.sim.measured == 0  # the failure may have broken the backend
    assert "Traceback" in capsys.readouterr().err


def test_a_failed_step_in_a_worker_names_the_native_exception():
    class Native(ToyBackend):
        def step(self, action):
            error = WorkerError("native AssertionError: the base's orientation is not a unit quaternion")
            error.kind = "AssertionError"
            raise error
    runtime, _ = world(backend=Native)
    pending = wait(runtime, .1)
    finish(runtime)
    assert pending.result().message == f"episode ended during wait; {FAILED} (AssertionError)"


def test_a_measure_that_fails_is_the_records_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(backend=MeasuredToy)
    runtime.sim.position[:] = -1.
    runtime.close()
    outcome, = records(tmp_path)
    assert outcome['reason'] == 'stopped' and 'progress' not in outcome
    assert outcome['error'] == "measures: RuntimeError: cannot measure a negative position"
    assert "Traceback" in capsys.readouterr().err and runtime.sim.closed


class FleetingToy(ToyBackend):
    """The toy whose "succeeds" goal holds at its second step only, as a lifted object that is then dropped."""

    def step(self, action):
        super().step(action)
        self.success = self.task == "succeeds" and self.steps == 2


@pytest.mark.parametrize("backend, success, first", [
    (ToyBackend, True, 2),  # the goal holds from the second step to the end
    (FleetingToy, False, 2),  # it held at the second step only: a stop-on-success run would have ended there, a success
])
def test_a_record_names_the_step_of_the_first_success(tmp_path, monkeypatch, backend, success, first):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=4, backend=backend, task="succeeds")
    wait(runtime, 1.)
    finish(runtime)
    runtime.close()
    outcome, = records(tmp_path)
    assert (outcome['step'], outcome['success'], outcome['first_success_step']) == (4, success, first)


def test_a_record_without_a_success_names_no_step(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=3)
    wait(runtime, 1.)
    finish(runtime)
    runtime.close()
    outcome, = records(tmp_path)
    assert outcome['first_success_step'] is None and not outcome['success']


def test_a_run_launched_to_stop_on_failure_goes_on_after_a_success(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, published = world(max_steps=4, backend=StumblingToy, task="succeeds", stop_on_failure=True)
    pending = wait(runtime, 1.)
    finish(runtime)
    assert runtime.steps == 4 and runtime.sim.success
    assert pending.result().message == "episode ended during wait; the episode ended because the time limit was reached"
    runtime.close()
    outcome, = records(tmp_path)
    # judged on its final state, as a run without either stop is
    assert (outcome['reason'], outcome['step'], outcome['success']) == ("success", 4, True)
    assert outcome['first_success_step'] == 2
