"""Kinex's ROS communication on a rosbridge overlay: fan-out, bounded queues, service failures and teardown."""
import asyncio
import uuid

import pytest

pytest.importorskip('kinex.communication')
from kinex.communication import communication


def test_fanout_precision_bounded_queues_and_teardown():
    async def run():
        endpoint = '/test_' + uuid.uuid4().hex
        operation = {'transport': 'topic', 'direction': 'output', 'endpoint': endpoint,
                     'type': 'std_msgs/msg/Float64MultiArray', 'qos': {'depth': 2}}
        catalog = {'test.output': operation, 'test.input': {**operation, 'direction': 'input'}}
        io = communication(catalog)
        async with io:
            streams = [io.subscribe('test.output', depth=2) for _ in range(10)]
            captures = 0
            for index in range(30):
                captures += 1
                await io.publish('test.input', {'data': [1.2345678901234567, float(index)]})
                await asyncio.sleep(.02)  # Test producer pacing, not a workflow clock.
            await asyncio.sleep(.1)
            assert captures == 30
            assert all(len(stream.buffer) <= 2 and len(stream.inbox) <= 2 for stream in streams)
            assert all(stream.latest().data[0] == 1.2345678901234567 for stream in streams)
            assert all(stream.dropped > 0 for stream in streams)
            streams[0].close()
            streams[0].close()
        assert not io.thread.is_alive()
        assert not io.context.ok()
    asyncio.run(run())


def test_async_service_failure_deadline_and_temporary_cleanup():
    async def run():
        async with communication({}) as io:
            spec = {'transport': 'service', 'direction': 'input', 'type': 'std_srvs/srv/SetBool',
                    'endpoint': f'{io.namespace}/echo'}
            io.catalog['test.echo'] = spec
            async def handler(request):
                await asyncio.sleep(.02)
                return {'success': request.data, 'message': 'expected rejection'}
            io.serve(spec, handler)
            assert (await io.request('test.echo', data=True)).success
            with pytest.raises(RuntimeError, match='expected rejection'):
                await io.request('test.echo', data=False)
            io.catalog['test.absent'] = {**spec, 'endpoint': spec['endpoint'] + '_absent'}
            with pytest.raises(TimeoutError):
                await io.request('test.absent', data=True, timeout_s=.03)
        assert not io.thread.is_alive()
    asyncio.run(run())


SERVICE = {'transport': 'service', 'direction': 'input'}


def arms():
    """The pair toy's two arms as Kinex operations: move goals and stops."""
    catalog = {}
    for arm in ('left', 'right'):
        catalog[f'{arm}.move'] = {'transport': 'action', 'direction': 'input', 'endpoint': f'/{arm}/move',
                                  'type': 'control_msgs/action/FollowJointTrajectory'}
        catalog[f'{arm}.stop'] = {**SERVICE, 'endpoint': f'/{arm}/stop', 'type': 'std_srvs/srv/Trigger'}
    return catalog


def goal(x, y):
    return {'multi_dof_trajectory': {'joint_names': ['tip'], 'points': [{'transforms': [[x, y, 0, 0, 0, 0, 1]]}]}}


def test_goals_run_in_parallel_and_a_world_wait_awaits_them(tmp_path):
    """Kinex sends two arms' move goals and waits for both on the world, the only clock: io.act never steps it."""
    from tests.ros.support import Graph
    graph = Graph(tmp_path, backend='pair_pose_toy')  # tips at (0.5, -0.5), 0.1 m a 0.1 s step, a wall at x = 1.2
    catalog = {**arms(), 'world.wait': {**SERVICE, 'endpoint': '/world/wait', 'type': 'custom_interfaces/srv/Wait'},
               'world.stop': {**SERVICE, 'endpoint': '/world/stop', 'type': 'std_srvs/srv/Trigger'}}

    async def run():
        async with communication(catalog, domain_id=graph.lease.domain) as io:
            near, far = await io.act('left.move', goal(.7, -.5)), await io.act('right.move', goal(1., -.5))
            assert (near.status, near.done, near.result) == ('executing', False, None)
            cut = await io.request('world.wait', goals=[near, far], goal_timeout_s=.3)  # far needs 0.5 s
            assert list(cut.finished) == [True, True] and cut.waited_s == pytest.approx(.3)
            assert (near.status, near.result.error_code) == ('succeeded', 0)
            assert (far.status, far.result.error_code) == ('aborted', -5)
            far = await io.act('right.move', goal(1., -.5))
            both = await io.request('world.wait', duration_s=.5, goals=[near, far])  # its duration and its goals
            assert list(both.finished) == [True, True] and both.waited_s == pytest.approx(.5)
            assert far.result.error_string == 'reached: 0.000 m and 0.000 rad from the goal after 0.20 s'
            assert far.feedback.multi_dof_desired.transforms[0].translation.x == pytest.approx(1.)
            assert (await io.request('world.wait', duration_s=.5)).waited_s == pytest.approx(.5)  # a plain wait
            blocked = await io.act('right.move', goal(1.5, -.5))
            await io.request('world.wait', goals=[blocked])
            assert (blocked.status, blocked.result.error_code) == ('aborted', -4)
            cancelled = await io.act('left.move', goal(1., -.5))
            await io.request('world.wait', duration_s=.1)
            await cancelled.cancel()
            assert (await cancelled.wait()).error_string.startswith('canceled: ') and cancelled.status == 'canceled'
            with pytest.raises(RuntimeError, match='a goal is a multi_dof_trajectory of tip poses in left/base'):
                refused = await io.act('left.move', {'multi_dof_trajectory': {'joint_names': ['palm']}})
                raise RuntimeError((await refused.wait()).error_string)
    try:
        asyncio.run(run())
    finally:
        graph.close()


def test_kinex_awaits_the_goals_of_a_world_that_does_not_observe_them(tmp_path):
    """Where time passes by itself, a world's wait may only wait, as Kinex's template world does: its reply says
    nothing of the goals, and Kinex awaits them and cancels those still executing at their timeout."""
    from tests.ros.support import Graph
    graph = Graph(tmp_path, mode='realtime', backend='pair_pose_toy')

    async def run():
        async with communication(arms(), domain_id=graph.lease.domain) as io:
            io.catalog['world.wait'] = {**SERVICE, 'type': 'custom_interfaces/srv/Wait', 'endpoint': f'{io.namespace}/wait'}

            async def wait(request):
                await asyncio.sleep(request.duration_s)
                return {'success': True, 'waited_s': request.duration_s}
            io.serve(io.catalog['world.wait'], wait)
            near = await io.act('left.move', goal(.7, -.5))  # 0.2 s away
            reply = await io.request('world.wait', goals=[near])
            assert list(reply.finished) == [True] and near.status == 'succeeded' and reply.waited_s > 0
            far = await io.act('left.move', goal(1.1, -.5))  # 0.4 s away
            reply = await io.request('world.wait', duration_s=.1, goals=[far], goal_timeout_s=.2)
            assert list(reply.finished) == [True] and far.status == 'canceled' and .2 <= reply.waited_s < .4
    try:
        asyncio.run(run())
    finally:
        graph.close()
