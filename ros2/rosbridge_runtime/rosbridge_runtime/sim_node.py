"""The world node: ROS callbacks queue work; the main thread owns the simulator and publishers."""
import faulthandler
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import rclpy
from custom_interfaces.srv import Wait
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Trigger

from rosbridge.core import load_backend
from rosbridge.runtime.description import logged, task_description
from rosbridge.runtime.world import WorldRuntime
from .devices import Goals, device
from .publishing import ObservationPublisher


class SimNode(Node):
    def __init__(self):
        super().__init__("world")
        self.sim = None
        self.executor_error = None
        p = self._read_parameters()
        # A configuration the simulator declares selects its alternative devices; other backends never see one.
        options = {"configuration": p["configuration"]} if p["configuration"] else {}
        self.sim = load_backend(p["backend"])(max_steps=p["max_steps"] or None, early_stop=p["early_stop"],
                                              **options)
        if p["spectator_rate_hz"]:
            # The views a recording asked for (comma-separated; empty: every view the simulator offers). The
            # simulator sets up only these, so views nobody records cost no rendering.
            selected = [name for name in p["spectator_select"].split(",") if name]
            self.sim.configure_spectators(p["spectator_width"], p["spectator_height"], selected or None)
            unknown = sorted(set(selected) - {view.name for view in self.sim.spectators})
            if unknown:
                raise ValueError(f"this simulator offers no spectator views {unknown}")
        # A run is one episode: the launch's task, seed and scene options, loaded once.
        self.sim.reset(p["task"] or self.sim.default_task, p["seed"], p["randomize"], **p["scene"])
        # The run log states the episode's task description, which `rosbridge describe` reads back.
        self.get_logger().info(logged(task_description(p["task"], self.sim)))
        self.world = WorldRuntime(self.sim, p["mode"], lambda: self.observations.publish(),
                                  early_stop=p["early_stop"])
        # Action goals by id, each with the future of its end (done) and end(reason); and those a server is still
        # accepting, by the id of their request, which have no goal id yet (devices.Goals).
        self.goals, self.starting = {}, {}
        self.devices = [device(spec, self) for spec in self.sim.devices]
        self.actions = [Goals(action, self) for action in getattr(self.sim, "actions", ())]
        self.declare_parameter("spectator_views", [v.name for v in self.sim.spectators],
                               ParameterDescriptor(read_only=True), ignore_override=True)
        self.observations = ObservationPublisher(self, p["spectator_rate_hz"])
        # World waits are sequential; stop cancels the pending one.
        self.pending_wait = None
        services = ReentrantCallbackGroup()
        self.create_service(Wait, "world/wait", self._wait, callback_group=services)
        self.create_service(Trigger, "world/stop", self._stop, callback_group=services)
        self.observations.publish()
        scene = "".join(f" {name}={value}" for name, value in sorted({**options, **p["scene"]}.items()))
        self.get_logger().info(f"task={self.sim.task} seed={self.sim.seed}{scene} mode={p['mode']} "
                               f"period={self.sim.period:.4f}s")

    def _read_parameters(self):
        defaults = {"backend": "", "task": "", "seed": 0, "scene": "", "configuration": "", "randomize": True,
                    "mode": "realtime", "max_steps": 0, "early_stop": False, "spectator_rate_hz": 0.,
                    "spectator_width": 960, "spectator_height": 720, "spectator_select": ""}
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        params = {name: self.get_parameter(name).value for name in defaults}
        # The scene options the launch gave, as one JSON object; reset receives them as keyword arguments.
        params["scene"] = json.loads(params["scene"]) if params["scene"] else {}
        if not isinstance(params["scene"], dict):
            raise ValueError("scene must be a JSON object of scene options")
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
            if not all(math.isfinite(s) and s >= 0 for s in (request.duration_s, request.goal_timeout_s)):
                raise ValueError("duration_s and goal_timeout_s must be finite and nonnegative")

            goals = [self.goals.get(bytes(goal.uuid)) for goal in request.goals]
            if None in goals:  # unless it is one the server has just accepted, which gets its id at once
                raise (LookupError if self.starting else ValueError)("goals must be goals this world accepted")
            until = [goal.done.done for goal in goals]

            def expire():
                for goal in goals:
                    goal.end("not finished within the wait's goal_timeout_s")

            def complete(result):
                self.pending_wait = None
                # With the commands refused since the previous reply.
                ready.set_result((self.world.reply(result), [done() for done in until]))
            self.pending_wait = complete
            self.world.wait(request.duration_s, complete, until, request.goal_timeout_s, expire)
        try:
            for _ in range(100):
                try:
                    await self.request(admit)
                    break
                except LookupError as unnamed:
                    error = ValueError(unnamed)
            else:
                raise error
            result, response.finished = await ready
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
        if not hasattr(node, 'world') and os.environ.get("ROSBRIDGE_RUN_DIR"):
            # The launch names why the world could not start (docs/README.md), such as a seed out of range.
            (Path(os.environ["ROSBRIDGE_RUN_DIR"]) / "launch-failure.json").write_text(json.dumps({"error": error}))
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
