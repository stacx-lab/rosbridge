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


def odometry(t, name, pose, twist, attitude=None):
    """Planar pose (x, y, yaw) in <name>/odom and body twist (vx, vy, wz) in <name>/base. With a tilting base's attitude
    (the backend's base_attitude: z, roll, pitch, vz, wx, wy), the pose and twist in full: the orientation is
    yaw * pitch * roll (ZYX), as an IMU reports it."""
    import math
    from nav_msgs.msg import Odometry
    msg = Odometry(child_frame_id=f"{name}/base")
    msg.header.stamp, msg.header.frame_id = t, f"{name}/odom"
    x, y, yaw = map(float, pose)
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    twist_msg = msg.twist.twist
    twist_msg.linear.x, twist_msg.linear.y, twist_msg.angular.z = map(float, twist)
    orientation = msg.pose.pose.orientation
    if attitude is None:
        orientation.z, orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return msg
    msg.pose.pose.position.z = float(attitude["z"])
    orientation.x, orientation.y, orientation.z, orientation.w = quaternion(attitude["roll"], attitude["pitch"], yaw)
    twist_msg.linear.z = float(attitude["vz"])
    twist_msg.angular.x, twist_msg.angular.y = float(attitude["wx"]), float(attitude["wy"])
    return msg


def quaternion(roll, pitch, yaw):
    """(x, y, z, w) of the rotation yaw about z, then pitch about the new y, then roll about the newest x (ZYX)."""
    import math
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


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


def trajectory(msg, joint_names):
    """JointTrajectory -> (start_s or None, times, points) ordered as joint_names."""
    if msg.joint_names:
        if sorted(msg.joint_names) != sorted(joint_names):
            raise ValueError(f"trajectory joints must be {joint_names}, got {list(msg.joint_names)}")
        order = [list(msg.joint_names).index(n) for n in joint_names]
    else:
        order = list(range(len(joint_names)))
    points = []
    for p in msg.points:
        if len(p.positions) != len(joint_names):
            raise ValueError(f"trajectory point needs {len(joint_names)} positions")
        points.append([p.positions[i] for i in order])
    times = [seconds(p.time_from_start) for p in msg.points]
    start = seconds(msg.header.stamp)
    return (start if start > 0 else None), times, points
