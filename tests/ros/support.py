"""Subprocess ROS graphs, each on its own leased domain, with explicit cleanup."""
import os
import signal
import subprocess
import sys
import time

import rclpy
from kinex_interfaces.srv import Wait
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger

from rosbridge.runs.domains import DomainLease
from rosbridge_runtime.devices import STREAM_QOS


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Graph:
    def __init__(self, tmp_path, mode="stepped", backend="toy", max_steps=0, spectators=False,
                 startup_s=60., task="toy", stop_on_success=False):
        self.lease = DomainLease()
        self.env = dict(os.environ, ROS_DOMAIN_ID=str(self.lease.domain))
        self.processes, self.logs = [], []
        self.context = Context()
        rclpy.init(context=self.context, domain_id=self.lease.domain)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.node = Node("probe", context=self.context)
        self.last = {}
        self.node.create_subscription(Clock, "/clock", lambda msg: self.last.update(clock=msg), STREAM_QOS)
        self.wait_client = self.node.create_client(Wait, "world/wait")
        self.stop_client = self.node.create_client(Trigger, "world/stop")
        self.tmp_path = tmp_path
        try:
            if backend == "toy":
                self.owner = self.start([sys.executable, "-m", "tests.ros.toy_server", "--ros-args",
                                         "-p", f"mode:={mode}", "-p", f"max_steps:={max_steps}",
                                         "-p", f"task:={task}",
                                         "-p", f"terminate_on_success:={str(stop_on_success).lower()}"])
            else:
                self.owner = self.start(["ros2", "launch", "rosbridge_runtime", "sim.launch.py",
                                         f"backend:={backend}", f"mode:={mode}",
                                         *(["spectator_rate_hz:=10.0"] if spectators else [])])
            # Some backends build scenes or start renderers for minutes before serving.
            self.until(self.wait_client.service_is_ready, startup_s)
            # Confirm that the owner loop is servicing requests after discovery.
            deadline = time.monotonic() + 5.
            while not self.wait(0.).success:
                assert time.monotonic() < deadline, "world never became responsive"
                self.spin(.05)
        except BaseException:
            self.close()
            raise

    def start(self, argv):
        log = open(self.tmp_path / f"process_{len(self.processes)}.log", "w+")
        self.logs.append(log)
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=self.env)
        self.processes.append(process)
        return process

    def until(self, predicate, timeout=15.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            rclpy.spin_once(self.node, executor=self.executor, timeout_sec=.01)
        for log in self.logs:
            log.flush()
            log.seek(0)
            print(log.read())
        raise AssertionError("ROS graph timed out")

    def spin(self, duration=.2):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            rclpy.spin_once(self.node, executor=self.executor, timeout_sec=.01)

    def call(self, client, request, timeout=15.):
        future = client.call_async(request)
        self.until(future.done, timeout)
        return future.result()

    def frame(self, camera, depth=False, timeout=15.):
        """A camera's current frame as a Kinex reader takes it: subscribe, then the first
        image and, with depth, the equally stamped depth. Returns (image, depth or None)."""
        from sensor_msgs.msg import Image
        from rosbridge_runtime.devices import FRAME_QOS
        received = {"image": [], "depth": []}
        names = ("image", "depth") if depth else ("image",)
        subscriptions = [self.node.create_subscription(Image, f"{camera}/{name}",
                                                       lambda msg, name=name: received[name].append(msg),
                                                       FRAME_QOS) for name in names]

        def pair():
            depths = {seconds(m.header.stamp): m for m in received["depth"]}
            for image in received["image"]:
                if not depth or seconds(image.header.stamp) in depths:
                    return image, depths.get(seconds(image.header.stamp))
        try:
            self.until(lambda: pair() is not None, timeout)
            return pair()
        finally:
            for subscription in subscriptions:
                self.node.destroy_subscription(subscription)

    def wait(self, duration):
        return self.call(self.wait_client, Wait.Request(duration_s=float(duration)))

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def close(self):
        for process in reversed(self.processes):
            self.stop(process)
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.try_shutdown()
        for log in self.logs:
            log.close()
        self.lease.close()
