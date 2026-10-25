"""Bounded asynchronous MP4 encoding using simulation timestamps."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


def encoder_path():
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('ffmpeg is missing; run rosbridge prepare for this simulator')
    result = subprocess.run([executable, '-hide_banner', '-encoders'], capture_output=True,
                            text=True, timeout=15, check=True)
    if 'libx264 ' not in result.stdout:
        raise RuntimeError('ffmpeg must include the libx264 encoder')
    return executable


def close_streams(streams, timeout=10., kill_timeout=2.):
    """Drain every encoder concurrently under one deadline, then kill stalled ones."""
    streams = list(streams)
    for stream in streams:
        stream.stopping.set()
    deadline = time.monotonic() + timeout
    for stream in streams:
        stream._thread.join(max(0., deadline - time.monotonic()))
    for stream in streams:
        if stream._thread.is_alive():
            stream._abort()
    deadline = time.monotonic() + kill_timeout
    for stream in streams:
        stream._thread.join(max(0., deadline - time.monotonic()))
    errors = [f'{s.name}: {s.error}' for s in streams if s.error]
    if errors:
        raise RuntimeError('; '.join(errors))


class VideoStream:
    """One camera, at most sixteen pending frames, and a file per reset segment."""
    def __init__(self, directory, name, fps, executable):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('recording requires a positive, finite spectator rate')
        self.directory, self.name, self.fps = Path(directory), name, fps
        self.executable = executable
        self.queue = queue.Queue(maxsize=16)
        self.received = self.dropped = 0
        self.segments = []
        self._error = None
        self._lock = threading.Lock()
        self.initial_written = threading.Event()
        self.stopping = threading.Event()
        self._aborted = threading.Event()
        self._process = None
        self._files = ExitStack()
        self._journal_dirty = False
        self._last_flush = time.monotonic()
        self.last_pixels = None
        self._thread = threading.Thread(target=self._run, name=f'encode-{name}', daemon=True)
        self._thread.start()

    @property
    def error(self):
        with self._lock:
            return self._error

    def _fail(self, error):
        with self._lock:
            self._error = self._error or str(error)

    def check(self):
        with self._lock:
            if self._process is not None and self._process.poll() not in (None, 0):
                self._error = self._error or 'encoder exited unexpectedly'
            error = self._error
        if error:
            raise RuntimeError(f'{self.name}: {error}')

    def submit(self, stamp_ns, width, height, pixels):
        if width <= 0 or height <= 0 or len(pixels) != width * height * 3:
            raise ValueError(f'invalid RGB frame for {self.name}')
        with self._lock:
            self.received += 1
            try:
                self.queue.put_nowait((stamp_ns, width, height, pixels))
            except queue.Full:
                self.dropped += 1

    def snapshot(self):
        with self._lock:
            return {'received': self.received, 'dropped': self.dropped,
                    'segments': [dict(segment) for segment in self.segments], 'error': self._error}

    def _open(self, stamp_ns, width, height):
        index = len(self.segments)
        stem = self.name if index == 0 else f'{self.name}_episode{index:03d}'
        path = self.directory / f'{stem}.mp4'
        self.segment = {'file': path.name, 'frames': 0, 'width': width, 'height': height,
                        'fps': self.fps, 'first_stamp_ns': stamp_ns, 'last_stamp_ns': stamp_ns}
        with self._lock:
            self.segments.append(self.segment)
        log = self._files.enter_context((self.directory / f'{stem}.ffmpeg.log').open('wb'))
        self.journal = self._files.enter_context((self.directory / f'{stem}.frames.jsonl').open('x'))
        command = [self.executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                   '-framerate', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-preset', 'veryfast', '-crf', '20', '-threads', '2',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                   '-g', str(max(1, round(self.fps * 2))),
                   '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(path)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=log)
        with self._lock:
            self._process = process
            if self._aborted.is_set():
                process.kill()

    def _write(self, pixels):
        if self._aborted.is_set():
            raise RuntimeError('encoder drain timed out')
        self._process.stdin.write(pixels)
        self._process.stdin.flush()
        with self._lock:
            self.segment['frames'] += 1

    def _frame(self, stamp_ns, width, height, pixels):
        if self._process is not None and stamp_ns < self.segment['last_stamp_ns']:
            self._finish_segment()
        if self._process is None:
            self._open(stamp_ns, width, height)
        elif (width, height) != (self.segment['width'], self.segment['height']):
            raise ValueError('camera dimensions changed during recording')
        elif stamp_ns == self.segment['last_stamp_ns'] and self.segment['frames']:
            return
        target = max(0, math.floor((stamp_ns - self.segment['first_stamp_ns']) * self.fps / 1e9 + 1e-6))
        held = max(0, target - self.segment['frames'])
        for _ in range(held):
            self._write(self.last_pixels)
        index = self.segment['frames']
        self._write(pixels)
        self.last_pixels = pixels
        with self._lock:
            self.segment['last_stamp_ns'] = stamp_ns
        self.journal.write(json.dumps({'stamp_ns': stamp_ns, 'frame_index': index,
                                       'held_frames_before': held}) + '\n')
        self._journal_dirty = True
        self.initial_written.set()

    def _finish_segment(self):
        process = self._process
        try:
            if process is not None:
                process.stdin.close()
                try:
                    code = process.wait(timeout=8.)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('encoder finalization timed out') from None
                if code:
                    raise RuntimeError(f'ffmpeg exited {code}; see {self.segment["file"][:-4]}.ffmpeg.log')
        finally:
            try:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            finally:
                with self._lock:
                    self._process = None
                self._files.close()
                self._journal_dirty = False

    def _run(self):
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                if self._aborted.is_set():
                    break
                try:
                    frame = self.queue.get(timeout=.05)
                except queue.Empty:
                    pass
                else:
                    self._frame(*frame)
                if self._journal_dirty and time.monotonic() - self._last_flush >= 1.:
                    self.journal.flush()
                    self._journal_dirty = False
                    self._last_flush = time.monotonic()
        except Exception as error:
            self._fail(error)
        finally:
            try:
                self._finish_segment()
            except Exception as error:
                self._fail(error)

    def _abort(self):
        self._aborted.set()
        with self._lock:
            self._error = self._error or 'encoder drain timed out'
            if self._process is not None and self._process.poll() is None:
                self._process.kill()

    def close(self):
        close_streams([self])
