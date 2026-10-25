"""World-time waits for ROS clients, independent of simulator or execution mode."""
import math
import time


class WorldClockClient:
    """Sequential waits on world/wait; the world rejects a wait while another is pending."""

    def __init__(self, node, service="world/wait"):
        from rclpy.callback_groups import ReentrantCallbackGroup
        from kinex_interfaces.srv import Wait
        self.node, self._type = node, Wait
        self.client = node.create_client(Wait, service, callback_group=ReentrantCallbackGroup())

    def wait_async(self, duration_s):
        """Return a ROS future. The caller's executor must spin to receive completion."""
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        return self.client.call_async(self._type.Request(duration_s=float(duration_s)))

    def wait(self, duration_s, *, timeout_s=300.):
        """Spin a standalone node until complete, with a separate wall-time timeout.

        Use wait_async from callbacks or nodes already spinning in an executor.
        A timeout does not cancel a wait already admitted; world/stop does.
        """
        from rclpy.executors import SingleThreadedExecutor
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if self.node.executor is not None and self.node in self.node.executor.get_nodes():
            raise RuntimeError("use wait_async for a node already attached to an executor")
        deadline = time.monotonic() + timeout_s
        if not self.client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError(f"world clock unavailable: {self.client.srv_name}")
        future = self.wait_async(duration_s)
        executor = SingleThreadedExecutor(context=self.node.context)
        executor.add_node(self.node)
        try:
            executor.spin_until_future_complete(future, timeout_sec=max(0., deadline - time.monotonic()))
        finally:
            executor.remove_node(self.node)
            executor.shutdown()
        if not future.done():
            self.client.remove_pending_request(future)
            raise TimeoutError("world wait wall-time timeout; the admitted wait remains pending")
        result = future.result()
        if result is None:
            raise RuntimeError("world wait interrupted by ROS shutdown")
        return result
