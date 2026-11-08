"""Recorder process owned by the run supervisor; never accesses native simulation."""
import json
import math
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.signals import SignalHandlerOptions
from rcl_interfaces.srv import GetParameters
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

from rosbridge.storage import write_json
from rosbridge.recording import finalize
from rosbridge_runtime.devices import STREAM_QOS
from .video import VideoStream, close_streams, encoder_path


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Recorder(Node):
    def __init__(self, record):
        super().__init__('rosbridge_recorder', use_global_arguments=False)
        self.record = record
        self.directory = Path(record['recording_dir'])
        self.metadata = json.loads((self.directory / 'recording.json').read_text())
        self.metadata['type'] = record['recording_type']
        self.streams = {}
        self.seen_info = set()
        self.received = {}
        self.clock_seen = False
        self.views = []
        self.bag = None
        self.bag_topics = set()
        self.bag_messages = 0
        self.ready = False
        self.error = None
        self.configured = False
        self.executable = encoder_path() if record['recording_type'] in ('mp4', 'both') else None
        self.parameters = self.create_client(GetParameters, 'world/get_parameters')
        self.parameters_future = None
        self.last_report = 0.
        self.last_metadata = None

    def _open_bag(self):
        if self.record['recording_type'] not in ('bag', 'both'):
            return
        import rosbag2_py
        self.bag = rosbag2_py.SequentialWriter()
        self.bag.open(rosbag2_py.StorageOptions(uri=str(self.directory / 'bag'), storage_id='mcap'),
                      rosbag2_py.ConverterOptions('cdr', 'cdr'))

    def _topic(self, topic, kind):
        topic = self.resolve_topic_name(topic)
        if self.bag is not None:
            import rosbag2_py
            self.bag.create_topic(rosbag2_py.TopicMetadata(
                id=len(self.bag_topics), name=topic, type=kind, serialization_format='cdr'))
            self.bag_topics.add(topic)
        return topic

    def _write_bag(self, topic, msg, timestamp):
        if self.bag is not None:
            self.bag.write(topic, serialize_message(msg), timestamp)
            self.bag_messages += 1

    def configure(self, values):
        self.views = list(values[0].string_array_value)
        fps = values[1].double_value
        if not self.views:
            raise ValueError('world must expose spectator views to record')
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('--record requires a positive spectator_rate_hz')
        self._open_bag()
        self.received = dict.fromkeys(self.views, 0)
        self.metadata.update(spectator_views=self.views)
        for view in self.views:
            if self.executable is not None:
                self.streams[view] = VideoStream(self.directory, view, fps, self.executable,
                                                 crf=self.record['recording_crf'])
            image_topic = self._topic(f'world/spectator/{view}/image_raw', 'sensor_msgs/msg/Image')
            info_topic = self._topic(f'world/spectator/{view}/camera_info', 'sensor_msgs/msg/CameraInfo')
            self.create_subscription(Image, image_topic,
                                     lambda msg, v=view, t=image_topic: self.image(v, t, msg), STREAM_QOS)
            self.create_subscription(CameraInfo, info_topic,
                                     lambda msg, v=view, t=info_topic: self.info(v, t, msg), STREAM_QOS)
        topic = self._topic('/clock', 'rosgraph_msgs/msg/Clock')
        self.create_subscription(Clock, topic, lambda msg: self.clock(topic, msg), STREAM_QOS)
        self.metadata['fps'] = fps
        self.metadata['topics'] = sorted(self.bag_topics)
        self.configured = True

    def image(self, view, topic, msg):
        if msg.encoding != 'rgb8' or msg.step != msg.width * 3:
            raise ValueError(f'{view}: recording expects packed rgb8 images')
        timestamp = stamp_ns(msg.header.stamp)
        self._write_bag(topic, msg, timestamp)
        if view in self.streams:
            self.streams[view].submit(timestamp, msg.width, msg.height, bytes(msg.data))
        self.received[view] += 1

    def info(self, view, topic, msg):
        self._write_bag(topic, msg, stamp_ns(msg.header.stamp))
        # Preserve calibration alongside MP4, even when no bag was requested.
        if view not in self.seen_info:
            write_json(self.directory / f'{view}.camera.json', {
                'frame_id': msg.header.frame_id, 'width': msg.width, 'height': msg.height,
                'k': list(msg.k), 'd': list(msg.d), 'r': list(msg.r), 'p': list(msg.p)})
            self.seen_info.add(view)

    def clock(self, topic, msg):
        self._write_bag(topic, msg, stamp_ns(msg.clock))
        self.clock_seen = True

    def progress(self):
        if not self.configured:
            if self.parameters_future is None and self.parameters.service_is_ready():
                self.parameters_future = self.parameters.call_async(
                    GetParameters.Request(names=['spectator_views', 'spectator_rate_hz']))
            elif self.parameters_future is not None and self.parameters_future.done():
                self.configure(self.parameters_future.result().values)
        for stream in self.streams.values():
            stream.check()
        if (not self.ready and self.configured and self.clock_seen
                and all(self.received.values()) and self.seen_info == set(self.views)
                and all(s.initial_written.is_set() for s in self.streams.values())):
            self.ready = True
            self.report('recording')
            write_json(Path(self.record['run_file']).parent / 'recording.ready', {'ready': True})
        if self.ready and time.monotonic() - self.last_report >= 1.:
            self.report('recording')

    def report(self, state):
        metadata = dict(self.metadata, state=state, error=self.error, bag_messages=self.bag_messages,
                        views={name: self.streams[name].snapshot() if name in self.streams
                               else {"received": count} for name, count in self.received.items()})
        if metadata != self.last_metadata:
            write_json(self.directory / 'recording.json', metadata)
            self.last_metadata = metadata
        self.last_report = time.monotonic()

    def finish(self):
        try:
            close_streams(self.streams.values())
        except Exception as error:
            self.error = self.error or str(error)
        if self.bag is not None:
            try:
                self.bag.close()
            except Exception as error:
                self.error = self.error or str(error)
        self.report('failed' if self.error else ('complete' if self.ready else 'incomplete'))


def main():
    record = json.loads(Path(sys.argv[1]).read_text())
    stop = [False]
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = None
    error = None
    try:
        node = Recorder(record)
        while not stop[0]:
            rclpy.spin_once(node, timeout_sec=.05)
            node.progress()
        # The producer has stopped first; drain messages already in DDS.
        deadline = time.monotonic() + .5
        while node.ready and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            node.progress()
    except Exception as failure:
        error = str(failure)
        print(error, file=sys.stderr, flush=True)
    finally:
        try:
            if node is not None:
                node.error = error
                node.finish()
                error = node.error
            else:
                record['error'] = error
                finalize(record, 1)
        except Exception as failure:
            error = error or (node.error if node is not None else None) or str(failure)
            print(f'Recorder cleanup failed: {failure}', file=sys.stderr, flush=True)
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                rclpy.try_shutdown()
    raise SystemExit(1 if error else 0)


if __name__ == '__main__':
    main()
