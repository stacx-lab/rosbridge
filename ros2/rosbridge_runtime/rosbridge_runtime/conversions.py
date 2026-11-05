"""numpy <-> ROS message conversion."""


def stamp(seconds):
    from builtin_interfaces.msg import Time
    ns = round(seconds * 1e9)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def seconds(t):
    return t.sec + t.nanosec * 1e-9


def clock(t):
    from rosgraph_msgs.msg import Clock
    return Clock(clock=t)


def joint_state(t, names, positions, velocities):
    from sensor_msgs.msg import JointState
    msg = JointState(name=list(names), position=positions.tolist(), velocity=velocities.tolist())
    msg.header.stamp = t
    return msg


def odometry(t, name, pose, twist):
    """Planar pose (x, y, yaw) in <name>/odom and body twist (vx, vy, wz) in <name>/base."""
    import math
    from nav_msgs.msg import Odometry
    msg = Odometry(child_frame_id=f"{name}/base")
    msg.header.stamp, msg.header.frame_id = t, f"{name}/odom"
    x, y, yaw = map(float, pose)
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    twist_msg = msg.twist.twist
    twist_msg.linear.x, twist_msg.linear.y, twist_msg.angular.z = map(float, twist)
    return msg


def image(t, frame, rgb):
    from sensor_msgs.msg import Image
    h, w, _ = rgb.shape
    msg = Image(height=h, width=w, encoding="rgb8", is_bigendian=0, step=w * 3, data=rgb.tobytes())
    msg.header.stamp, msg.header.frame_id = t, frame
    return msg


def depth_image(t, frame, metres):
    """16UC1 millimetres along the optical axis; 0 marks missing or out-of-range samples."""
    import numpy as np
    from sensor_msgs.msg import Image
    h, w = metres.shape
    valid = np.isfinite(metres) & (metres > 0)
    mm = np.where(valid, np.rint(np.where(valid, metres, 0) * 1000), 0)
    mm = np.where(mm <= 65535, mm, 0).astype(np.uint16)
    msg = Image(height=h, width=w, encoding="16UC1", is_bigendian=0, step=w * 2, data=mm.tobytes())
    msg.header.stamp, msg.header.frame_id = t, frame
    return msg


def camera_info(t, frame, width, height, fx, fy, cx, cy):
    from sensor_msgs.msg import CameraInfo
    msg = CameraInfo(width=width, height=height, distortion_model="plumb_bob")
    msg.header.stamp, msg.header.frame_id = t, frame
    msg.d = [0.0] * 5
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def trajectory(msg, joint_names, ignored=()):
    """JointTrajectory -> (start_s or None, times, points) ordered as joint_names; the positions of named
    joints in ignored are dropped."""
    names = list(msg.joint_names)
    if names:
        if sorted(n for n in names if n not in ignored) != sorted(joint_names) or len(set(names)) != len(names):
            raise ValueError(f"trajectory joints must be {joint_names}, got {names}")
        order = [names.index(n) for n in joint_names]
    else:
        order = list(range(len(joint_names)))
    size = len(names) or len(joint_names)
    points = []
    for p in msg.points:
        if len(p.positions) != size:
            raise ValueError(f"trajectory point needs {size} positions")
        points.append([p.positions[i] for i in order])
    times = [seconds(p.time_from_start) for p in msg.points]
    start = seconds(msg.header.stamp)
    return (start if start > 0 else None), times, points
