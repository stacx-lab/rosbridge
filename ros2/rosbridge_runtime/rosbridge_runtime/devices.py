"""Device endpoints on the world node; commands and captures run on the simulator owner thread."""
from geometry_msgs.msg import Twist
from kinex_interfaces.srv import Capture, CaptureRGBD
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from . import conversions as conv

# Retain the latest sample even when DDS discovery completes after publication.
STREAM_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
            node.create_subscription(Float64, f'{name}/target', self.target, 10, callback_group=callbacks)
        else:
            node.create_subscription(JointTrajectory, f'{name}/joint_target', self.joint_target, 10,
                                     callback_group=callbacks)

    def publish(self, stamp, positions, velocities):
        self.state.publish(conv.joint_state(stamp, self.names, positions[self.indices], velocities[self.indices]))

    def _admit(self, apply):
        def run():
            try:
                apply()
            except ValueError as error:
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
        node.create_subscription(Twist, f'{name}/velocity', self.velocity, 10, callback_group=callbacks)

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
    """`image` while subscribed, `camera_info` and on-demand `capture`, in frame `<name>/base`.

    RGB-D cameras add registered `depth` (16UC1 mm) and capture both with CaptureRGBD.
    """

    def __init__(self, spec, node):
        self.spec, self.node = spec, node
        self.frame = f'{spec.name}/base'
        self.latest = self.latest_depth = None
        self.image = node.create_publisher(Image, f'{spec.name}/image', qos_profile_sensor_data)
        self.depth = (node.create_publisher(Image, f'{spec.name}/depth', qos_profile_sensor_data)
                      if spec.kind == 'rgbd' else None)
        self.info = node.create_publisher(CameraInfo, f'{spec.name}/camera_info', STREAM_QOS)
        node.create_service(CaptureRGBD if self.depth else Capture, f'{spec.name}/capture',
                            self.capture, callback_group=ReentrantCallbackGroup())

    def reset(self):
        self.latest = self.latest_depth = None

    def render(self, stamp):
        if self.latest is None or self.latest.header.stamp != stamp:
            self.latest = conv.image(stamp, self.frame, self.node.sim.render(self.spec.source))
        return self.latest

    def render_depth(self, stamp):
        if self.latest_depth is None or self.latest_depth.header.stamp != stamp:
            self.latest_depth = conv.depth_image(stamp, self.frame,
                                                 self.node.sim.render_depth(self.spec.source))
        return self.latest_depth

    def publish(self, stamp):
        sim = self.node.sim
        self.info.publish(conv.camera_info(stamp, self.frame, sim.width, sim.height,
                                           *sim.intrinsics(self.spec.source)))
        if self.image.get_subscription_count():
            self.image.publish(self.render(stamp))
        if self.depth is not None and self.depth.get_subscription_count():
            self.depth.publish(self.render_depth(stamp))

    async def capture(self, request, response):
        def capture():
            stamp = conv.stamp(self.node.world.time_s)
            image = self.render(stamp)
            self.image.publish(image)
            if self.depth is None:
                return image, None
            depth = self.render_depth(stamp)
            self.depth.publish(depth)
            return image, depth
        try:
            response.image, depth = await self.node.request(capture)
            if depth is not None:
                response.depth_image = depth
        except Exception as error:
            self.node.get_logger().error(f'{self.spec.name} capture failed: {error}')
        return response


def device(spec, node):
    kind = Camera if spec.kind in ('rgb', 'rgbd') else Base if spec.kind == 'locomotion' else Actuator
    return kind(spec, node)
