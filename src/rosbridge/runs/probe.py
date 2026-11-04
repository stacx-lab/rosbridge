"""Environment-local graph readiness: the world node, its latched clock and wait service."""
import argparse
import json
import time


def probe(timeout=10.):
    import rclpy
    from rclpy.node import Node
    from kinex_interfaces.srv import Wait
    from rosgraph_msgs.msg import Clock
    from rosbridge_runtime.devices import STREAM_QOS
    rclpy.init(args=[])
    node = Node("rosbridge_probe")
    clock = []
    node.create_subscription(Clock, "/clock", clock.append, STREAM_QOS)
    wait = node.create_client(Wait, "world/wait")
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            names = {name for name, namespace in node.get_node_names_and_namespaces() if namespace == "/"}
            if "world" in names and wait.service_is_ready() and clock:
                stamp = clock[-1].clock
                return {"ready": True, "time_s": stamp.sec + stamp.nanosec * 1e-9}
        return {"ready": False, "error": "the world node, clock or wait service did not become ready"}
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10.)
    result = probe(parser.parse_args().timeout)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
