"""Device endpoints on the world node; commands and renders run on the simulator owner thread."""
import json
import time
from types import SimpleNamespace

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.event_handler import PublisherEventCallbacks
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from rosbridge.core.lens import LensGeometry
from rosbridge.core.motions import INVALID_GOAL, SUCCESSFUL
from custom_interfaces.srv import Json
from sensor_msgs.msg import CameraInfo, Image, JointState, MultiDOFJointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from . import conversions as conv

# Retain the latest sample even when DDS discovery completes after publication.
STREAM_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        reliability=ReliabilityPolicy.RELIABLE)
# Camera frames: the one rendered for a new subscriber must arrive, and none goes stale.
FRAME_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE,
                       reliability=ReliabilityPolicy.RELIABLE)
# Wall time after which a subscribed camera of a resting world sends its frame again.
KEEPALIVE_S = .2


class Actuator:
    """An arm (`joint_target`, or `pose_target` of its tip link with the measured `pose_state`) or gripper
    (`target`) with `joint_state` and `stop`. A joint group with a tip takes joint targets and reports `pose_state`,
    or, with move, serves the `move` action in their place: goals of tip poses it reaches at its own pace."""
    stopped = 'holding the measured position'

    def __init__(self, spec, node):
        self.spec, self.node = spec, node
        self.group = next(g for g in node.sim.layout.groups if g.name == spec.source)
        if self.group.scalar:
            self.stopped = 'keeping the grip target'  # the command buffer keeps a gripper's applied target
        self.indices = list(self.group.indices)
        self.names = [node.sim.layout.joint_names[i] for i in self.indices]
        callbacks = ReentrantCallbackGroup()
        self._endpoints(callbacks)
        # A client may discover stop before commanding, so it must follow the command subscription.
        node.create_service(Trigger, f'{spec.name}/stop', self.stop, callback_group=callbacks)

    def _endpoints(self, callbacks):
        node, name = self.node, self.spec.name
        self.state = node.create_publisher(JointState, f'{name}/joint_state', STREAM_QOS)
        if self.group.scalar:
            self.command = node.create_subscription(Float64, f'{name}/target', self.target, 10,
                                                    callback_group=callbacks)
            return
        if self.group.tip:
            self.pose_state = node.create_publisher(MultiDOFJointState, f'{name}/pose_state', STREAM_QOS)
        if self.group.tip and not self.group.joint_targets:
            self.command = node.create_subscription(MultiDOFJointTrajectory, f'{name}/pose_target',
                                                    self.pose_target, 10, callback_group=callbacks)
            return
        if self.group.move:
            self.command = ActionServer(node, FollowJointTrajectory, f'{name}/move', self.execute,
                                        goal_callback=self.accept, handle_accepted_callback=self.accepted,
                                        cancel_callback=self.cancel, callback_group=callbacks)
            return
        self.command = node.create_subscription(JointTrajectory, f'{name}/joint_target', self.joint_target, 10,
                                                callback_group=callbacks)

    def publish(self, stamp, positions, velocities):
        self.state.publish(conv.joint_state(stamp, self.names, positions[self.indices], velocities[self.indices]))
        if self.group.tip:
            self.pose_state.publish(conv.pose_state(stamp, f'{self.spec.name}/base', self.group.tip,
                                                    self.node.sim.tip_pose(self.group.name)))

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
            if times and times[-1] > 120:  # the command buffer refuses one without points
                raise ValueError('trajectory must end within 120 seconds')
            if start is not None and start < world.time_s:
                raise ValueError('trajectory start is in the past')
            world.commands.set_trajectory(self.group.name, world.time_s if start is None else start, times, points)
        self._admit(apply)

    def pose_target(self, msg):
        def apply():
            world, frame, tip = self.node.world, f'{self.spec.name}/base', self.group.tip
            if msg.header.frame_id not in ('', frame):
                raise ValueError(f'pose targets must be expressed in {frame}')
            if list(msg.joint_names) != [tip]:
                raise ValueError(f'pose targets must name the tip link {tip} once')
            if any(p.velocities or p.accelerations for p in msg.points):
                raise ValueError('only pose trajectories are supported')
            start, times, poses = conv.pose_trajectory(msg)
            if times and times[-1] > 120:  # the command buffer refuses one without points
                raise ValueError('trajectory must end within 120 seconds')
            if start is not None and start < world.time_s:
                raise ValueError('trajectory start is in the past')
            world.commands.set_pose_trajectory(self.group.name, world.time_s if start is None else start, times, poses,
                                               lambda: self.node.sim.tip_pose(self.group.name))
        self._admit(apply)

    def path(self, request):
        """A goal's tip poses, their nominal times, its tolerance (m, rad) and seconds to settle; a ValueError says
        why it is none."""
        msg, frame, tip = request.multi_dof_trajectory, f'{self.spec.name}/base', self.group.tip
        if msg.header.frame_id not in ('', frame) or list(msg.joint_names) != [tip] or not msg.points:
            raise ValueError(f'a goal is a multi_dof_trajectory of {tip} poses in {frame}')
        _, times, poses = conv.pose_trajectory(msg)
        tolerance = [0., 0.]
        for item in request.component_goal_tolerance:
            if item.component in (item.TRANSLATION, item.ROTATION):
                tolerance[item.component - item.TRANSLATION] = item.position
        return poses, times, tolerance, conv.seconds(request.goal_time_tolerance)

    async def accept(self, request):
        """Start a goal in the world before the server accepts it, so the world has every goal a client holds."""
        frame, tip = f'{self.spec.name}/base', self.group.tip

        def feedback(time_s, target, pose):
            goal.sample = FollowJointTrajectory.Feedback(
                multi_dof_joint_names=[tip],
                multi_dof_desired=MultiDOFJointTrajectoryPoint(transforms=[conv.transform(target)]),
                multi_dof_actual=MultiDOFJointTrajectoryPoint(transforms=[conv.transform(pose)]))
            goal.sample.header.stamp, goal.sample.header.frame_id = conv.stamp(time_s), frame
            if goal.handle is not None:
                goal.handle.publish_feedback(goal.sample)

        def end(reason, code=INVALID_GOAL):
            if not goal.done.done():  # still this arm's goal
                self.node.world.end(self.group.name, reason, code, hold=True)

        def start():
            try:
                self.node.world.move(self.group.name, lambda *end: goal.done.set_result(end), feedback,
                                     *self.path(request))
            except ValueError as error:
                goal.done.set_result((INVALID_GOAL, str(error)))
        goal = SimpleNamespace(done=Future(executor=self.node.executor), end=end, handle=None, sample=None)
        await self.node.request(start)
        self.node.starting[id(request)] = goal
        return GoalResponse.ACCEPT

    def accepted(self, handle):
        """Name the started goal by the id the server gave it; the server passes one request to both callbacks."""
        goal = self.node.goals[bytes(handle.goal_id.uuid)] = self.node.starting[id(handle.request)]
        del self.node.starting[id(handle.request)]
        goal.handle = handle
        if goal.sample is not None:  # its first step, commanded before it had a handle
            handle.publish_feedback(goal.sample)
        handle.execute()

    async def execute(self, handle):
        code, text = await self.node.goals[bytes(handle.goal_id.uuid)].done
        (handle.succeed if code == SUCCESSFUL else handle.canceled if handle.is_cancel_requested else handle.abort)()
        return FollowJointTrajectory.Result(error_code=code, error_string=text)

    def cancel(self, handle):
        self.node.world.submit(lambda: self.node.goals[bytes(handle.goal_id.uuid)].end('canceled'))
        return CancelResponse.ACCEPT

    async def stop(self, request, response):
        try:
            await self.node.request(lambda: self.node.world.hold(self.group.name))
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


class Camera:
    """`image` and `camera_info` in frame `<name>/base`; RGB-D cameras add registered `depth`
    (16UC1 mm) with the same stamp. A camera its backend renders through a lens also serves `geometry`,
    `unproject` and `project` (rosbridge.core.lens), for the lens of the episode the world loaded.

    Frames render only for subscribers: at each camera sample, and at once for a new
    subscriber, so a reader receives the current frame even while the world is paused.
    DDS can deliver that frame before the reader has discovered the camera, which drops
    it, so while the world rests a subscribed camera sends its frame again every
    KEEPALIVE_S, as a live camera keeps streaming a still scene.
    """

    def __init__(self, spec, node):
        self.spec, self.node = spec, node
        self.frame = f'{spec.name}/base'
        self.latest = self.latest_depth = None
        self.sent_s, self.sent_stamp, self.queued = time.monotonic(), None, False
        events = PublisherEventCallbacks(matched=self.matched)
        self.image = node.create_publisher(Image, f'{spec.name}/image', FRAME_QOS,
                                           event_callbacks=events)
        self.depth = (node.create_publisher(Image, f'{spec.name}/depth', FRAME_QOS,
                                            event_callbacks=events)
                      if spec.kind == 'rgbd' else None)
        self.info = node.create_publisher(CameraInfo, f'{spec.name}/camera_info', STREAM_QOS)
        lens = getattr(node.sim, 'distortion', {}).get(spec.source)
        if lens is not None:
            geometry = LensGeometry(self.frame, *node.sim.intrinsics(spec.source), lens)
            callbacks = ReentrantCallbackGroup()  # pure arithmetic, off the simulator owner thread
            for operation in ('geometry', 'unproject', 'project'):
                node.create_service(Json, f'{spec.name}/{operation}', json_handler(getattr(geometry, operation)),
                                    callback_group=callbacks)
        node.create_timer(KEEPALIVE_S / 2, self.keepalive)

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
        self.sent_s, self.sent_stamp = time.monotonic(), stamp

    def publish(self, stamp):
        sim = self.node.sim
        lens = getattr(sim, 'distortion', {}).get(self.spec.source)
        self.info.publish(conv.camera_info(stamp, self.frame, *sim.intrinsics(self.spec.source), lens))
        self.send(stamp)

    def matched(self, status):
        if status.current_count_change > 0:
            self.node.world.submit(self.send_current)

    def keepalive(self):
        subscribed = self.image.get_subscription_count() or (
            self.depth is not None and self.depth.get_subscription_count())
        if subscribed and not self.queued and time.monotonic() - self.sent_s >= KEEPALIVE_S:
            self.queued = True
            self.node.world.submit(self.resend)

    def resend(self):
        """Send the last frame again while the world rests at its stamp; a running world's
        camera samples keep the backend's camera rate."""
        self.queued = False
        if self.sent_stamp == conv.stamp(self.node.world.time_s):
            self.send_current()

    def send_current(self):
        try:
            self.send(conv.stamp(self.node.world.time_s))
        except Exception as error:
            self.node.get_logger().error(f'{self.spec.name} render failed: {error}')


def json_handler(operation):
    """A Json service callback: operation(**request) -> reply, both JSON objects. A failure is a reply."""
    def call(request, response):
        try:
            response.response = json.dumps(operation(**json.loads(request.request)), allow_nan=False)
            response.success = True
        except Exception as error:
            response.message = f'{type(error).__name__}: {error}'
        return response
    return call


def device(spec, node):
    kind = Camera if spec.kind in ('rgb', 'rgbd') else Base if spec.kind == 'locomotion' else Actuator
    return kind(spec, node)
