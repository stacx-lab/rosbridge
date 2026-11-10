"""The R1Pro base's odometry on TF (each R1Pro rig's devices/base/odom.py, which Kinex runs when the base device
loads): base/odom -> base/base equals the /base/odometry message, its stamp included."""
import asyncio
import copy
import math
import os
from pathlib import Path
import sys

import pytest
import yaml

if os.environ.get("KINEX_PYTHON"):
    sys.path.insert(0, os.environ["KINEX_PYTHON"])
pytest.importorskip("kinex.host.device")
from kinex.communication import communication, qos  # noqa: E402
from kinex.host.device import DeviceHost  # noqa: E402

from rosbridge.runs.domains import DomainLease  # noqa: E402

CATALOG = Path(__file__).resolve().parents[4] / "catalog/robots"


def odometry(sec, nanosec, x, y, yaw):
    """A planar message as rosbridge_runtime.conversions.odometry writes it."""
    from nav_msgs.msg import Odometry
    message = Odometry(child_frame_id="base/base")
    message.header.frame_id = "base/odom"
    message.header.stamp.sec, message.header.stamp.nanosec = sec, nanosec
    message.pose.pose.position.x, message.pose.pose.position.y = x, y
    message.pose.pose.orientation.z, message.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return message


@pytest.mark.parametrize("rig", ["behavior/behavior-r1pro", "behavior-dex/behavior-r1pro-sharpa"])
def test_the_base_odometry_is_on_tf_with_the_messages_stamp(rig):
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_srvs.srv import Trigger
    from tf2_msgs.msg import TFMessage

    base = CATALOG / rig / "devices/base"
    definition = yaml.safe_load((base / "profile.yaml").read_text())
    assert definition["runtime"]["nodes"] == [{"entry": "odom.py", "factory": "create"}]
    lease = DomainLease()
    definition = copy.deepcopy(definition)
    definition["runtime"].update(domain_id=lease.domain, ready_timeout_s=10)
    definition["runtime"]["nodes"] = [dict(node, entry=str(base / node["entry"]))
                                      for node in definition["runtime"]["nodes"]]
    operations = definition["operations"]

    async def run():
        async with communication({}, domain_id=lease.domain) as backend:
            node = backend.node
            publisher = node.create_publisher(Odometry, operations["odometry"]["endpoint"],
                                              qos(operations["odometry"]))
            node.create_subscription(Twist, operations["velocity"]["endpoint"], lambda message: None,
                                     qos(operations["velocity"]))

            def stop(request, response):  # unloading the device stops it
                response.success = True
                return response
            node.create_service(Trigger, operations["stop"]["endpoint"], stop)
            seen = []
            node.create_subscription(TFMessage, "/tf", lambda message: seen.extend(message.transforms), 100)
            publisher.publish(odometry(3, 250_000_000, 0., 0., 0.))
            host = DeviceHost()
            try:
                await host.load(definition)  # ready once its node has seen an odometry message
                assert [type(n).__name__ for n in host.io.extra_nodes] == ["OdometryPublisher"]
                sample = odometry(12, 345_000_000, 1.5, -.25, .7)
                async with asyncio.timeout(10):
                    while not any(t.header.stamp.sec == 12 for t in seen):
                        publisher.publish(sample)
                        await asyncio.sleep(.05)
                (tf,) = {(t.header.stamp.sec, t.header.stamp.nanosec): t for t in seen
                         if t.header.stamp.sec == 12}.values()
                assert (tf.header.frame_id, tf.child_frame_id) == ("base/odom", "base/base")
                assert (tf.header.stamp.sec, tf.header.stamp.nanosec) == (12, 345_000_000)
                p, q = sample.pose.pose.position, sample.pose.pose.orientation
                t, r = tf.transform.translation, tf.transform.rotation
                assert (t.x, t.y, t.z) == pytest.approx((p.x, p.y, p.z))
                assert (r.x, r.y, r.z, r.w) == pytest.approx((q.x, q.y, q.z, q.w))
            finally:
                await host.run("unload", {})
            assert backend.context.ok()
    try:
        asyncio.run(run())
    finally:
        lease.close()
