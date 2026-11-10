"""Publish the base's odometry on TF as <base>/odom -> <base>/base with each message's stamp, so frames
resolve against the episode-start pose (the odometry topic alone left <base>/odom off the TF tree)."""
import asyncio

from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from kinex.communication import qos
from kinex.lib.frames import dynamic_qos, transform
from kinex.node import node_name


class OdometryPublisher(Node):
    def __init__(self, profile, context):
        super().__init__(node_name(profile, "odom"), context=context)
        self.broadcaster = TransformBroadcaster(self, qos=dynamic_qos())
        self.observed = False
        odometry = profile["operations"]["odometry"]
        self.create_subscription(Odometry, odometry["endpoint"], self.odometry, qos(odometry))

    def odometry(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        self.broadcaster.sendTransform(transform(message.header.frame_id, message.child_frame_id,
                                                 [p.x, p.y, p.z, q.x, q.y, q.z, q.w], message.header.stamp))
        self.observed = True

    async def ready(self):
        while not self.observed:
            await asyncio.sleep(.02)


def create(profile, context):
    return OdometryPublisher(profile, context)
