"""Backend conformance and shared timing without ROS, MuJoCo or ABC."""
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from rosbridge.core import CommandError
from tests.toy_backend import GoalToyBackend, ToyBackend
from rosbridge.runtime.world import END_CAUSES, WorldRuntime



def world(mode="stepped", max_steps=None, backend=ToyBackend, stop_on_success=False, **kwargs):
    """A runtime over backend; stop_on_success launches both to end at the first success, as the world node does."""
    published = []
    sim = backend(max_steps, terminate_on_success=stop_on_success)
    runtime = WorldRuntime(sim, mode, lambda: published.append(sim.steps), stop_on_success=stop_on_success,
                           **kwargs)
    return runtime, published


def wait(runtime, duration):
    """Admit a world wait on the owner thread as the world node does; the future completes with its result."""
    result = Future()
    runtime.submit(lambda: runtime.admit_wait(duration, result.set_result))
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


def replied(runtime, duration):
    """A world wait as the world node admits it: its reply names the commands refused since the previous one."""
    result = Future()
    runtime.submit(lambda: runtime.admit_wait(duration, lambda done: result.set_result(runtime.reply(done))))
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
    assert pending.result().message == ("episode ended during wait; the episode ended because the time limit was "
                                        "reached; refused since the previous wait reply: "
                                        "/arm/joint_target: trajectory start is in the past")
    runtime, _ = world()
    stopped = replied(runtime, 1.)
    runtime.drain()
    runtime.refuse("/finger/target", "finger target has non-finite values")
    runtime.close()
    assert stopped.result().message == ("runtime stopped; refused since the previous wait reply: /finger/target: "
                                        "finger target has non-finite values")


# The end notice: every wait reply at or after the end says that the episode ended, and what ended it ---------------

LIMIT, GOAL = "the time limit was reached", "the goal was reached"


def goal(first, last=None):
    """A backend whose goal holds from step `first` until before step `last`."""
    return type("GoalAt", (GoalToyBackend,), {"first": first, "last": last})


class Native(ToyBackend):
    """Ends by its own rule at step 3, with no step limit the world knows: ABC's terminated or truncated without
    max_steps, RoboDojo's end_flag."""
    def step(self, action):
        super().step(action)
        self.done = self.steps >= 3


def notice(cause):
    return "the episode ended" + (f" because {cause}" if cause else "")


def test_every_wait_at_or_after_the_end_says_that_the_episode_ended():
    """At the time limit. Nothing is said before the end; the first words stay first (clients match them), the reply's
    time_s is the end, and a zero and a positive wait after the end say it too."""
    runtime, _ = world(max_steps=3)
    early = replied(runtime, .1)
    runtime.drain()
    runtime.tick()
    assert early.result().message == "ok" and early.result().success and runtime.end_cause() is None
    due, late = replied(runtime, .2), replied(runtime, 1.)
    finish(runtime)
    end = notice(LIMIT)
    assert (due.result().success, due.result().message) == (True, f"ok; {end}")  # its last step was the terminal one
    assert (late.result().success, late.result().message) == (False, f"episode ended during wait; {end}")
    assert due.result().time_s == late.result().time_s == pytest.approx(.3)
    after, zero = replied(runtime, .5), replied(runtime, 0.)  # admitted together after the end
    runtime.drain()
    assert (after.result().success, after.result().message) == (False, f"episode is done; {end}")
    assert (zero.result().success, zero.result().message) == (True, f"ok; {end}")
    assert after.result().time_s == zero.result().time_s == pytest.approx(.3) and after.result().waited_s == 0.


@pytest.mark.parametrize("backend, limit, stop, cause, reason, step", [
    (goal(2), 6, True, GOAL, "success", 2),  # stopped on success
    (goal(3), 3, True, GOAL, "success", 3),  # the goal on the limit's own step, stopping on success
    (goal(2), 6, False, LIMIT, "success", 6),  # running on after success: the goal holds at the limit
    (goal(2, 5), 6, False, LIMIT, "max_steps", 6),  # the goal held, then was lost before the limit
    (goal(99), 4, True, LIMIT, "max_steps", 4),  # stopping on success that never came
    (Native, None, False, "the simulator ended it", "episode_done", 3),
])
def test_the_cause_is_what_ended_the_episode_and_the_record_keeps_its_reason(
        tmp_path, monkeypatch, backend, limit, stop, cause, reason, step):
    """Every kind of reply at or after the end says the same; the record's reason is the evaluator's (success first)
    and differs from the cause where the goal holds at the limit of a run that goes on after success."""
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=limit, backend=backend, stop_on_success=stop)
    pending = replied(runtime, 50.)
    finish(runtime)
    replies = [pending.result()]
    for duration in (0., .1, 1e308):
        later = replied(runtime, duration)
        runtime.drain()
        replies.append(later.result())
    outcome, = records(tmp_path)
    assert (outcome['reason'], outcome['step']) == (reason, step)
    assert [r.message for r in replies] == [f"{first}; {notice(cause)}" for first in (
        "episode ended during wait", "ok", "episode is done", "episode is done")]
    assert [r.success for r in replies] == [False, True, False, False]
    assert all(r.time_s == outcome['time_s'] for r in replies)
    runtime.close()  # stopping the run afterwards changes neither
    assert records(tmp_path) == [outcome] and runtime.end_cause() == cause


def test_a_wait_due_on_the_tick_that_reaches_the_goal_succeeds_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=10, backend=goal(2), stop_on_success=True)
    due = replied(runtime, .2)
    finish(runtime)
    assert (due.result().success, due.result().message) == (True, f"ok; {notice(GOAL)}")
    assert due.result().waited_s == pytest.approx(.2) and records(tmp_path)[0]['reason'] == 'success'


def test_nothing_is_said_before_the_end_even_while_the_goal_holds():
    runtime, _ = world(max_steps=50, backend=goal(1))
    messages = []
    for duration in (.1, 0., .3, 0.):
        reply = replied(runtime, duration)
        finish(runtime)
        messages.append(reply.result().message)
    assert runtime.sim.success and not runtime.sim.done and messages == ["ok"] * 4


def test_the_cause_is_fixed_when_the_episode_ends():
    """Whatever the backend says afterwards (its success or step limit), later replies state the end it had."""
    runtime, _ = world(max_steps=6, backend=goal(2, 5))
    pending = replied(runtime, 50.)
    finish(runtime)
    assert pending.result().message.endswith(LIMIT)
    runtime.sim.success, runtime.sim.max_steps = True, None
    later = replied(runtime, 0.)
    runtime.drain()
    assert later.result().message == f"ok; {notice(LIMIT)}" and runtime.end_cause() == LIMIT


def test_a_realtime_episode_that_ended_with_no_wait_pending_says_so_in_the_next_replies():
    now = [0.]
    runtime, _ = world("realtime", max_steps=2, clock=lambda: now[0])
    now[0] = 5.
    while runtime.tick():
        pass
    assert runtime.sim.done and runtime.steps == 2
    now[0] = 500.  # wall time goes on; the world's clock stays at the end
    zero, positive = replied(runtime, 0.), replied(runtime, 1.)
    runtime.drain()
    assert (zero.result().success, zero.result().message) == (True, f"ok; {notice(LIMIT)}")
    assert (positive.result().success, positive.result().message) == (False, f"episode is done; {notice(LIMIT)}")
    assert zero.result().time_s == positive.result().time_s == pytest.approx(.2)


@pytest.mark.parametrize("duration", [-1., float("nan"), float("inf")])
def test_a_bad_wait_after_the_end_is_refused_as_before(duration):
    runtime, _ = world(max_steps=1)
    finish(runtime)
    bad = replied(runtime, duration)
    runtime.drain()
    assert (bad.result().success, bad.result().message) == (False, "duration_s must be finite and nonnegative")


def test_the_end_follows_the_first_words_and_precedes_the_refusals_also_of_commands_sent_after_it():
    runtime, _ = world(max_steps=1)
    pending = replied(runtime, .1)
    refuse(runtime, "/arm/joint_target", "trajectory start is in the past")
    finish(runtime)
    assert pending.result().message == (f"ok; {notice(LIMIT)}; refused since the previous wait reply: "
                                        "/arm/joint_target: trajectory start is in the past")

    def command():  # as a device admits a command after the end
        try:
            runtime.commands.set_group("finger", [0.])
        except CommandError as error:
            runtime.refuse("/finger/target", error)
    runtime.submit(command)
    after = replied(runtime, 0.)
    runtime.drain()
    assert after.result().message == (f"ok; {notice(LIMIT)}; refused since the previous wait reply: "
                                      "/finger/target: episode is done")


@pytest.mark.parametrize("error", [None, "WorkerError: native worker exited (exit -9)"])
def test_a_stopped_run_states_no_end(tmp_path, monkeypatch, error):
    """Stopping a run, by the operator or an error, is not an end of its episode: the pending wait answers "runtime
    stopped", also while the goal holds, and the record says stopped."""
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    runtime, _ = world(max_steps=50, backend=goal(1))
    pending = replied(runtime, 1.)
    runtime.drain()
    runtime.tick()
    runtime.close(error=error)
    assert pending.result().message == "runtime stopped" and runtime.end_cause() is None
    outcome, = records(tmp_path)
    assert (outcome['reason'], outcome['success'], outcome['done']) == ('stopped', True, False)
    assert outcome.get('error') == error


def test_the_end_cause_is_read_on_the_owner_thread_only():
    runtime, _ = world(max_steps=1)
    finish(runtime)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(runtime.end_cause).result()
    assert END_CAUSES == {"success": GOAL, "max_steps": LIMIT, "episode_done": "the simulator ended it"}


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
