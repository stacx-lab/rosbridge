"""Device endpoints on the world node; commands and renders run on the simulator owner thread."""
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.event_handler import PublisherEventCallbacks
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from . import conversions as conv

# Retain the latest sample even when DDS discovery completes after publication.
STREAM_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        reliability=ReliabilityPolicy.RELIABLE)
# Camera frames: the one rendered for a new subscriber must arrive, and none goes stale.
FRAME_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE,
                       reliability=ReliabilityPolicy.RELIABLE)


class Actuator:
    """An arm (`joint_target`) or gripper (`target`) with `joint_state` and `stop`."""
    stopped = 'holding the measured position'

    def __init__(self, spec, node):
        self.spec, self.node = spec, node
        self.group = next(g for g in node.sim.layout.groups if g.name == spec.source)
        self.indices = list(self.group.indices)
        self.names = [node.sim.layout.joint_names[i] for i in self.indices]
        callbacks = ReentrantCallbackGroup()
        self._endpoints(callbacks)
        # Kinex discovers stop before commanding, so it must follow the command subscription.
        node.create_service(Trigger, f'{spec.name}/stop', self.stop, callback_group=callbacks)

    def _endpoints(self, callbacks):
        node, name = self.node, self.spec.name
        self.state = node.create_publisher(JointState, f'{name}/joint_state', STREAM_QOS)
        if self.group.scalar:
            self.command = node.create_subscription(Float64, f'{name}/target', self.target, 10,
                                                    callback_group=callbacks)
        else:
            self.command = node.create_subscription(JointTrajectory, f'{name}/joint_target', self.joint_target, 10,
                                                    callback_group=callbacks)

    def publish(self, stamp, positions, velocities):
        self.state.publish(conv.joint_state(stamp, self.names, positions[self.indices], velocities[self.indices]))

    def _admit(self, apply):
        def run():
            try:
                apply()
            except ValueError as error:
                # A topic publisher gets no reply; the next world wait's reply names the refusal.
                self.node.world.refuse(self.command.topic_name, error)
                self.node.get_logger().error(f'{self.spec.name} rejected a command: {error}',
                                             throttle_duration_sec=1.)
        self.node.world.submit(run)

    def target(self, msg):
        self._admit(lambda: self.node.world.commands.set_group(self.group.name, [msg.data]))

    def joint_target(self, msg):
        def apply():
            world = self.node.world
            if sorted(msg.joint_names) != sorted(self.names) or len(set(msg.joint_names)) != len(self.names):
                raise ValueError(f'trajectory must name each of {self.names} once')
            if any(p.velocities or p.accelerations or p.effort for p in msg.points):
                raise ValueError('only position trajectories are supported')
            start, times, points = conv.trajectory(msg, self.names)
            if not times or times[-1] > 120:
                raise ValueError('trajectory must end within 120 seconds')
            if start is not None and start < world.time_s:
                raise ValueError('trajectory start is in the past')
            world.commands.set_trajectory(self.group.name, world.time_s if start is None else start, times, points)
        self._admit(apply)

    def _hold(self):
        return self.node.sim.state()[0][self.indices]

    async def stop(self, request, response):
        try:
            await self.node.request(lambda: self.node.world.commands.stop(self.group.name, self._hold()))
            response.success, response.message = True, self.stopped
        except Exception as error:
            response.success, response.message = False, str(error)
        return response


class Base(Actuator):
    """A mobile base: body `velocity` (Twist) with a world-time watchdog, `odometry` and `stop`."""
    stopped = 'commanding zero velocity'

    def _endpoints(self, callbacks):
        node, name = self.node, self.spec.name
        self.odometry = node.create_publisher(Odometry, f'{name}/odometry', STREAM_QOS)
        self.command = node.create_subscription(Twist, f'{name}/velocity', self.velocity, 10,
                                                callback_group=callbacks)

    def publish(self, stamp, positions, velocities):
        self.odometry.publish(conv.odometry(stamp, self.spec.name, positions[self.indices],
                                            velocities[self.indices]))

    def velocity(self, msg):
        def apply():
            if msg.linear.z or msg.angular.x or msg.angular.y:
                raise ValueError('only linear.x, linear.y and angular.z are commandable')
            world = self.node.world
            world.commands.set_group(self.group.name, [msg.linear.x, msg.linear.y, msg.angular.z],
                                     world.time_s)
        self._admit(apply)

    def _hold(self):
        return [0.] * len(self.indices)


class Camera:
    """`image` and `camera_info` in frame `<name>/base`; RGB-D cameras add registered `depth`
    (16UC1 mm) with the same stamp.

    Frames render only for subscribers: at each camera sample, and at once for a new
    subscriber, so a reader receives the current frame even while the world is paused.
    """

    def __init__(self, spec, node):
        self.spec, self.node = spec, node
        self.frame = f'{spec.name}/base'
        self.latest = self.latest_depth = None
        events = PublisherEventCallbacks(matched=self.matched)
        self.image = node.create_publisher(Image, f'{spec.name}/image', FRAME_QOS,
                                           event_callbacks=events)
        self.depth = (node.create_publisher(Image, f'{spec.name}/depth', FRAME_QOS,
                                            event_callbacks=events)
                      if spec.kind == 'rgbd' else None)
        self.info = node.create_publisher(CameraInfo, f'{spec.name}/camera_info', STREAM_QOS)

    def render(self, stamp):
        if self.latest is None or self.latest.header.stamp != stamp:
            self.latest = conv.image(stamp, self.frame, self.node.sim.render(self.spec.source))
        return self.latest

    def render_depth(self, stamp):
        if self.latest_depth is None or self.latest_depth.header.stamp != stamp:
            self.latest_depth = conv.depth_image(stamp, self.frame,
                                                 self.node.sim.render_depth(self.spec.source))
        return self.latest_depth

    def send(self, stamp):
        """Render and publish the subscribed streams; image and depth share one stamp."""
        if self.image.get_subscription_count():
            self.image.publish(self.render(stamp))
        if self.depth is not None and self.depth.get_subscription_count():
            self.depth.publish(self.render_depth(stamp))

    def publish(self, stamp):
        self.info.publish(conv.camera_info(stamp, self.frame, *self.node.sim.intrinsics(self.spec.source)))
        self.send(stamp)

    def matched(self, status):
        if status.current_count_change > 0:
            self.node.world.submit(self.send_current)

    def send_current(self):
        try:
            self.send(conv.stamp(self.node.world.time_s))
        except Exception as error:
            self.node.get_logger().error(f'{self.spec.name} render failed: {error}')


def device(spec, node):
    kind = Camera if spec.kind in ('rgb', 'rgbd') else Base if spec.kind == 'locomotion' else Actuator
    return kind(spec, node)
