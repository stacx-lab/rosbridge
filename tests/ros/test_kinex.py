"""Native protocol helper tests. Set KINEX_PYTHON to core/assets/python."""
import asyncio
import os
import sys
import uuid

import pytest

if os.environ.get('KINEX_PYTHON'):
    sys.path.insert(0, os.environ['KINEX_PYTHON'])
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
