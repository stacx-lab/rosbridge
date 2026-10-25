"""Managed recording formats, startup failures, and recorder supervision."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest
from rosbridge.runs.records import alive, read
from tests.integration.support import invoke, start


def metadata(directory):
    return json.loads((directory / 'recording.json').read_text())


def video_info(path):
    output = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                                      '-show_entries', 'stream=nb_read_frames,width,height',
                                      '-select_streams', 'v:0', '-of', 'json', str(path)], text=True)
    return json.loads(output)['streams'][0]


@pytest.mark.parametrize('kind', ['mp4', 'bag', 'both'])
@pytest.mark.parametrize('simulator', ['abc', 'libero'])
def test_record_formats_and_initial_frames(runs, tmp_path, kind, simulator):
    output = tmp_path / f'{kind} with spaces'
    record = start(runs, '--record', output, '--record-type', kind, selection=simulator)
    assert record['recording_dir'] == str(output)
    before = metadata(output)
    assert before['state'] == 'recording'
    expected = {'third_person', 'wrist', 'right_wrist'} if simulator == 'abc' else {'third_person', 'wrist'}
    assert set(before['views']) == expected
    assert all(v['received'] >= 1 for v in before['views'].values())
    modified = (output / 'recording.json').stat().st_mtime_ns
    # Startup did not advance a stepped world while waiting for the recorder.
    assert json.loads(invoke('status', record['id'], '--json').stdout)['time_s'] == 0
    time.sleep(1.2)
    assert (output / 'recording.json').stat().st_mtime_ns == modified
    controller = '''import rclpy
from rclpy.node import Node
from rosbridge.clients import WorldClockClient
rclpy.init()
n=Node('recording_test')
r=WorldClockClient(n).wait(.34,timeout_s=15)
assert r.success
n.destroy_node()
rclpy.shutdown()
'''
    invoke('exec', record['id'], '--', 'python', '-c', controller)
    stopped = json.loads(invoke('stop', record['id'], '--json').stdout)
    assert stopped['recording_state'] == 'complete'
    assert not alive({'owner': stopped['recorder']})
    result = metadata(output)
    assert result['state'] == 'complete'
    if kind in ('mp4', 'both'):
        for view in result['views']:
            video = video_info(output / f'{view}.mp4')
            assert (video['width'], video['height']) == (960, 720)
            assert int(video['nb_read_frames']) >= 3
            assert (output / f'{view}.camera.json').exists()
            assert result['views'][view]['dropped'] == 0
    else:
        assert not list(output.glob('*.mp4'))
    if kind in ('bag', 'both'):
        assert (output / 'bag/metadata.yaml').exists()
        assert list((output / 'bag').glob('*.mcap'))
        assert result['bag_messages'] >= 7
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(output / 'bag'), storage_id='mcap'),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        count, image_stamps = 0, []
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            count += 1
            if topic.endswith('/image_raw'):
                image = deserialize_message(data, Image)
                assert len(image.data) == 960 * 720 * 3
                assert timestamp == image.header.stamp.sec * 1000000000 + image.header.stamp.nanosec
                image_stamps.append(timestamp)
        assert count == result['bag_messages']
        assert min(image_stamps) == 0 and max(image_stamps) > 0
    else:
        assert not (output / 'bag').exists()
    error = invoke('launch', 'abc', '--record', output, '--detach', '--json', expected=1)
    assert 'new or empty' in error.stdout
    assert metadata(output) == result


@pytest.mark.parametrize('manifest', ['intact', 'missing', 'malformed', 'unwritable'])
def test_recorder_failure_stops_simulation(runs, tmp_path, manifest):
    output = tmp_path / 'failed'
    record = start(runs, '--record', output)
    os.kill(record['recorder']['pid'], signal.SIGSTOP)
    path = output / 'recording.json'
    if manifest == 'missing':
        path.unlink()
    elif manifest == 'malformed':
        path.write_text('{broken')
    elif manifest == 'unwritable':
        path.unlink()
        path.mkdir()
    os.kill(record['recorder']['pid'], signal.SIGKILL)
    path = Path(record['run_file'])
    deadline = time.monotonic() + 20.
    while time.monotonic() < deadline and alive(read(path)):
        time.sleep(.1)
    final = read(path)
    assert not alive(final) and not alive({'owner': final['graph']})
    assert final['state'] == 'failed'
    assert final['recording_state'] == 'failed'
    if manifest != 'unwritable':
        assert metadata(output)['state'] == 'failed'
    # The preparation lock and domain lease must also be released on cleanup errors.
    from rosbridge.provisioning.setup import workspace_lock
    from rosbridge.runs.domains import DomainLease
    with workspace_lock(Path(record['project']), 'abc'):
        lease = DomainLease(record['domain_id'])
        lease.close()


def test_realtime_recording_includes_initial_frame(runs, tmp_path):
    output = tmp_path / 'realtime'
    record = start(runs, '--execution-mode', 'realtime', '--record', output)
    status = json.loads(invoke('status', record['id'], '--json').stdout)
    assert status['time_s'] > 0
    invoke('stop', record['id'], '--json')
    result = metadata(output)
    assert result['type'] == 'mp4' and result['state'] == 'complete'
    for name, view in result['views'].items():
        assert view['segments'][0]['first_stamp_ns'] == 0
        assert view['segments'][0]['last_stamp_ns'] > 0
        assert int(video_info(output / f'{name}.mp4')['nb_read_frames']) > 1
