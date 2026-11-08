"""The world node: ROS callbacks queue work; the main thread owns the simulator and publishers."""
import faulthandler
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import rclpy
from kinex_interfaces.srv import Wait
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Trigger

from rosbridge.core import load_backend
from rosbridge.runtime.world import WorldRuntime
from .devices import device
from .publishing import ObservationPublisher


class SimNode(Node):
    def __init__(self):
        super().__init__("world")
        self.sim = None
        self.executor_error = None
        p = self._read_parameters()
        self.sim = load_backend(p["backend"])(max_steps=p["max_steps"] or None,
                                              terminate_on_success=p["terminate_on_success"])
        if p["spectator_rate_hz"]:
            # The views a recording asked for (comma-separated; empty: every view the simulator offers). The
            # simulator sets up only these, so views nobody records cost no rendering.
            selected = [name for name in p["spectator_select"].split(",") if name]
            self.sim.configure_spectators(p["spectator_width"], p["spectator_height"], selected or None)
            unknown = sorted(set(selected) - {view.name for view in self.sim.spectators})
            if unknown:
                raise ValueError(f"this simulator offers no spectator views {unknown}")
        # A run is one episode: the launch's task and seed, loaded once.
        self.sim.reset(p["task"] or self.sim.default_task, p["seed"], p["randomize"])
        self.world = WorldRuntime(self.sim, p["mode"], lambda: self.observations.publish())
        self.devices = [device(spec, self) for spec in self.sim.devices]
        self.declare_parameter("spectator_views", [v.name for v in self.sim.spectators],
                               ParameterDescriptor(read_only=True), ignore_override=True)
        self.observations = ObservationPublisher(self, p["spectator_rate_hz"])
        # World waits are sequential; stop cancels the pending one.
        self.pending_wait = None
        services = ReentrantCallbackGroup()
        self.create_service(Wait, "world/wait", self._wait, callback_group=services)
        self.create_service(Trigger, "world/stop", self._stop, callback_group=services)
        self.observations.publish()
        self.get_logger().info(f"task={self.sim.task} seed={self.sim.seed} mode={p['mode']} "
                               f"period={self.sim.period:.4f}s")

    def _read_parameters(self):
        defaults = {"backend": "", "task": "", "seed": 0, "randomize": True, "mode": "realtime",
                    "max_steps": 0, "terminate_on_success": False, "spectator_rate_hz": 0.,
                    "spectator_width": 960, "spectator_height": 720, "spectator_select": ""}
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        params = {name: self.get_parameter(name).value for name in defaults}
        if params["mode"] not in ("realtime", "stepped"):
            raise ValueError("mode must be realtime or stepped")
        if params["max_steps"] < 0:
            raise ValueError("max_steps must be nonnegative")
        if not math.isfinite(params["spectator_rate_hz"]) or params["spectator_rate_hz"] < 0:
            raise ValueError("spectator_rate_hz must be finite and nonnegative")
        if params["spectator_width"] < 16 or params["spectator_height"] < 16:
            raise ValueError("spectator_width and spectator_height must be at least 16 pixels")
        return params

    async def request(self, function):
        """Run function on the owner thread. Awaiting a ROS future releases the executor worker."""
        ready = Future(executor=self.executor)

        def complete(future):
            try:
                ready.set_result(future.result())
            except Exception as error:
                ready.set_exception(error)
        self.world.submit(function).add_done_callback(complete)
        return await ready

    async def _wait(self, request, response):
        ready = Future(executor=self.executor)

        def admit():
            if self.pending_wait is not None:
                raise ValueError("another wait is pending; world waits are sequential")
            if not math.isfinite(request.duration_s) or request.duration_s < 0:
                raise ValueError("duration_s must be finite and nonnegative")

            def complete(result):
                self.pending_wait = None
                ready.set_result(self.world.reply(result))  # with the commands refused since the previous reply
            self.pending_wait = complete
            self.world.waits.admit(request.duration_s, self.world.time_s, self.world.steps,
                                   self.sim.period, self.sim.done, complete)
        try:
            await self.request(admit)
            result = await ready
            response.success, response.message = result.success, result.message
            response.time_s, response.waited_s = result.time_s, result.waited_s
        except Exception as error:
            response.success, response.message = False, str(error)
        return response

    async def _stop(self, request, response):
        def stop():
            if self.pending_wait is not None:
                self.world.waits.cancel(self.pending_wait, self.world.time_s)
        try:
            await self.request(stop)
            response.success = True
        except Exception as error:
            response.success, response.message = False, str(error)
        return response

    def run(self):
        ready = os.environ.get("ROSBRIDGE_RECORDING_READY")
        while ready and rclpy.ok() and not Path(ready).is_file():
            if self.executor_error is not None:
                raise RuntimeError("ROS executor failed") from self.executor_error
            time.sleep(.02)
        # Startup recording discovery must not become realtime catch-up time.
        self.world.pacer.reset(self.sim.period)
        while rclpy.ok():
            if self.executor_error is not None:
                raise RuntimeError("ROS executor failed") from self.executor_error
            self.world.drain()
            if not self.world.tick():
                self.world.drain(self.world.idle_s())


def main():
    # SIGUSR1 writes every thread's Python stack to stderr (graph.log): a world that stops answering can be located.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    rclpy.init()
    node = SimNode.__new__(SimNode)
    executor = MultiThreadedExecutor(num_threads=4)
    spinner, error = None, None
    try:
        node.__init__()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass
            except Exception as error:
                node.executor_error = error
        spinner = threading.Thread(target=spin, daemon=True)
        spinner.start()
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as failure:
        # A world that stops on an error says why in its log (graph.log, an operator artifact) and in the episode's
        # record; a client only sees its pending wait end with "runtime stopped". The traceback goes out before
        # cleanup, which waits on the native worker and may be killed.
        error = f"{type(failure).__name__}: {failure}"[:500]
        print("world stopped by an error:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if hasattr(node, 'world'):
            node.world.close(error=error)
        elif getattr(node, 'sim', None) is not None:
            node.sim.close()
        executor.shutdown(timeout_sec=5.)
        if spinner is not None:
            spinner.join(timeout=5.)
        if hasattr(node, '_handle'):
            node.destroy_node()
        rclpy.try_shutdown()
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
