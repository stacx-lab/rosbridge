import json
import subprocess
import time

import pytest

from rosbridge.cli.main import main, parser
from rosbridge.recording import destination, reserve
from rosbridge.recording.video import VideoStream, encoder_path
from rosbridge.runs.api import recording_settings


def test_record_cli_and_empty_destination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parser().parse_args(['launch', 'abc', '--record', 'my videos', '--record-type', 'both'])
    path = destination(args.record)
    assert path == tmp_path / 'my videos'
    reserve(path, 'run-id', 'abc/pouring')
    assert json.loads((path / 'recording.json').read_text())['state'] == 'starting'
    with pytest.raises(ValueError, match='empty'):
        destination(path)
    with pytest.raises(ValueError, match='empty'):
        reserve(path, 'another-run', 'abc')
    assert main(['launch', 'abc', '--record-type', 'bag']) == 1
    for flag, value in (('--record-rate', '5'), ('--record-size', '640x480'), ('--record-views', 'wrist'),
                        ('--record-quality', '28')):
        assert main(['launch', 'abc', flag, value]) == 1  # each requires --record


def test_recording_settings_default_and_check_their_ranges():
    assert recording_settings({}) == (10., 960, 720, [], 20)
    assert recording_settings({'record_rate': 5, 'record_size': '640X480', 'record_views': 'head,,front_left',
                               'record_quality': 28}) == (5., 640, 480, ['head', 'front_left'], 28)
    assert recording_settings({'record_quality': 0})[4] == 0  # lossless, not the default
    for options in ({'record_rate': 60}, {'record_rate': 0}, {'record_size': '8x8'}, {'record_quality': 52}):
        with pytest.raises(ValueError):
            recording_settings(options)


def frames(path):
    result = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                                      '-select_streams', 'v:0', '-show_entries',
                                      'stream=nb_read_frames,width,height', '-of', 'json', str(path)], text=True)
    return json.loads(result)['streams'][0]


def test_video_simulation_timing_and_decoding(tmp_path):
    stream = VideoStream(tmp_path, 'overview', 10., encoder_path())
    pixels = bytes([180, 40, 20]) * (31 * 23)  # Odd dimensions pad for yuv420p.
    for timestamp in (0, 100_000_000, 400_000_000, 400_000_000):
        stream.submit(timestamp, 31, 23, pixels)
    stream.close()
    video = frames(tmp_path / 'overview.mp4')
    assert (video['width'], video['height'], int(video['nb_read_frames'])) == (32, 24, 5)
    rows = [json.loads(line) for line in (tmp_path / 'overview.frames.jsonl').read_text().splitlines()]
    assert [r['frame_index'] for r in rows] == [0, 1, 4]
    assert rows[-1]['held_frames_before'] == 2
    assert stream.snapshot()['dropped'] == 0


def test_world_time_going_backwards_fails_the_recording(tmp_path):
    stream = VideoStream(tmp_path, 'overview', 10., encoder_path())
    for timestamp in (100_000_000, 0):
        stream.submit(timestamp, 16, 16, bytes(16 * 16 * 3))
    with pytest.raises(RuntimeError, match="backwards"):
        stream.close()


def test_an_open_segment_is_readable_up_to_the_frame_before_the_latest(tmp_path):
    """A stepped world sends nothing after its episode ends, and a harness may copy a step's artifacts before the
    run stops and closes the file: the open file must already hold the episode (it lacked 4-8 s at 5 Hz)."""
    import shutil
    stream = VideoStream(tmp_path, 'overview', 5., encoder_path())
    pixels = [bytes([20 * i % 256, 40, 200 - 5 * i]) * (64 * 48) for i in range(30)]
    for i, frame in enumerate(pixels):
        stream.submit(i * 200_000_000, 64, 48, frame)
        time.sleep(.02)  # within the 16-frame queue
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and sum(s['frames'] for s in stream.snapshot()['segments']) < 30:
        time.sleep(.05)
    time.sleep(.5)
    shutil.copy(tmp_path / 'overview.mp4', tmp_path / 'copy.mp4')
    assert int(frames(tmp_path / 'copy.mp4')['nb_read_frames']) >= 29
    stream.close()
    assert int(frames(tmp_path / 'overview.mp4')['nb_read_frames']) == 30


def test_encoder_failure_is_reported(tmp_path):
    stream = VideoStream(tmp_path, 'overview', 10., '/bin/false')
    stream.submit(0, 16, 16, bytes(16 * 16 * 3))
    with pytest.raises(RuntimeError):
        stream.close()


def test_encoder_queue_is_bounded(tmp_path, monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    seen = []
    def frame(self, *args):
        entered.set()
        assert release.wait(5.)
        seen.append(args[0])
    monkeypatch.setattr(VideoStream, '_frame', frame)
    stream = VideoStream(tmp_path, 'overview', 10., '/bin/false')
    try:
        stream.submit(0, 1, 1, bytes(3))
        assert entered.wait(1.)
        start = time.monotonic()
        for i in range(1, 101):
            stream.submit(i, 1, 1, bytes(3))
        assert time.monotonic() - start < 1.
        assert stream.dropped == 84
    finally:
        release.set()
        stream.close()
    assert len(seen) == 17


def test_stalled_encoders_share_shutdown_deadline(tmp_path, monkeypatch):
    import sys
    from rosbridge.recording import video
    executable = tmp_path / 'stalled-encoder'
    executable.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(60)\n')
    executable.chmod(0o700)
    processes = []
    original = video.subprocess.Popen
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(video.subprocess, 'Popen', spawn)
    streams = [VideoStream(tmp_path, f'view{i}', 10., str(executable)) for i in range(3)]
    try:
        for stream in streams:
            stream.submit(0, 512, 512, bytes(512 * 512 * 3))
        deadline = time.monotonic() + 5.
        while len(processes) < 3:
            assert time.monotonic() < deadline
            time.sleep(.01)
        # Blocked pipe writes must never hold the statistics lock.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as pool:
            for stream in streams:
                assert pool.submit(stream.snapshot).result(timeout=.5)['received'] == 1
        start = time.monotonic()
        with pytest.raises(RuntimeError, match='drain timed out'):
            video.close_streams(streams, timeout=.2, kill_timeout=1.)
        assert time.monotonic() - start < 1.5
        assert all(process.poll() is not None for process in processes)
        assert all(not stream._thread.is_alive() for stream in streams)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for stream in streams:
            try:
                stream.close()
            except RuntimeError:
                pass


@pytest.mark.parametrize('contents', [None, '{broken', '[]', '{"state":"failed","error":["bad"]}'])
def test_invalid_recording_metadata_retains_original_failure(tmp_path, contents):
    from rosbridge.recording import finalize
    path = tmp_path / 'recording.json'
    if contents is not None:
        path.write_text(contents)
    record = {'recording_dir': str(tmp_path), 'error': 'graph failed first'}
    finalize(record, 1)
    assert record['recording_state'] == 'failed'
    assert record['error'] == 'graph failed first'
    assert json.loads(path.read_text())['error'] == 'graph failed first'


def test_recorder_failure_detail_is_retained(tmp_path):
    from rosbridge.recording import finalize
    path = tmp_path / 'recording.json'
    path.write_text(json.dumps({'state': 'failed', 'error': 'encoder failed'}))
    record = {'recording_dir': str(tmp_path), 'error': 'Recorder exited unexpectedly'}
    finalize(record, 1)
    assert record['error'] == 'Recorder exited unexpectedly; recording: encoder failed'
    assert record['recording_state'] == 'failed'
