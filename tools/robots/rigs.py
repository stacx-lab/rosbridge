"""Write Kinex workspace rigs: profiles and kinematic-only URDFs.

Builders (build_<simulator>.py) measure frames from the native simulator and
call these templates, so every rig shares one endpoint layout and wording.
Native geometry tests check the written files against the simulator.
"""
from pathlib import Path
import math

import numpy as np
import yaml

from rosbridge.catalog import cameras

CREATED = "2026-10-24T00:00:00Z"


def qos(reliability="reliable", durability="volatile"):
    return {"depth": 1, "reliability": reliability, "durability": durability}


def operation(description, transport, direction, endpoint, kind, quality=None, timeout_s=10):
    return {"description": description, "transport": transport, "direction": direction,
            "endpoint": endpoint, "type": kind, "qos": quality or qos(), "timeout_s": timeout_s}


def frame(parent, translation, rotation_xyzw):
    return {"translation_m": [clean(v) for v in translation],
            "rotation_xyzw": [clean(v) for v in rotation_xyzw], "parent": parent}


def clean(value, digits=12):
    value = round(float(value), digits)
    return 0.0 if value == 0 else value


def device(name, kind, description, operations, *, observation=None, layout=None, **facts):
    """A device profile; layout is an rgb camera's (mono or stereo)."""
    runtime = {"domain_id": 0}
    if observation:
        runtime["observation"] = observation
    # Every operation is an endpoint of the world node, so a device declares no nodes of its own.
    runtime.update({"nodes": [], "ready_timeout_s": 30})
    return {"name": name, "runtime": runtime, "operations": operations, "loaded": False,
            "meta": {"created_at": CREATED, "updated_at": CREATED}, "type": kind,
            **({"layout": layout} if layout else {}), "description": description,
            "source": "introspected", "calibrated_at": None, "method": "introspection", **facts}


def world(description):
    ops = {
        "time": operation("Authoritative world clock", "topic", "output", "/clock",
                          "rosgraph_msgs/msg/Clock", qos(durability="transient_local")),
        "wait": operation(
            "Advance or await duration_s of world time (finite and nonnegative). Zero is an "
            "observation barrier. Waits are sequential; one requested while another is pending "
            "is rejected. A terminal episode rejects further progression. The reply's message "
            "also names the device commands refused since the previous reply, with their "
            "reasons.", "service", "input", "/world/wait",
            "custom_interfaces/srv/Wait",
            timeout_s=120),
        "stop": operation("Cancel the pending wait.", "service", "input", "/world/stop",
                          "std_srvs/srv/Trigger"),
    }
    value = device("world", "world", description, ops, observation="time")
    value.update(source="default", method="unknown", bounds=None, surfaces=[],
                 frames={"base": {"parent": None, "translation_m": None, "rotation_xyzw": None}})
    return value


def joint(name, description, joints, home, base, *, units="radians", part="arm", tip="flange",
          model="arm.urdf", pose=None):
    """A joint device; tip is its FK/IK link, None for trees such as hands. With pose, a sentence on how the
    simulator executes them, the device takes pose targets of its tip instead of joint targets."""
    count = len(joints)
    over = f"all {count} {part} joints" if count > 1 else f"the {part} joint"
    ops = {
        "joint_state": operation(
            f"Measured positions of {', '.join(joints)} in {units} with velocities and the "
            "world-clock acquisition stamp; do not infer motion completion from a sent command.",
            "topic", "output", f"/{name}/joint_state", "sensor_msgs/msg/JointState",
            qos(durability="transient_local")),
    }
    if pose:
        ops["pose_target"] = operation(
            f"Trajectory of absolute {tip} poses in {name}/base: joint_names [{tip}], one transform per "
            "point (metres, unit quaternion xyzw), time_from_start strictly increasing and within 120 "
            "seconds; a zero header stamp starts it now. It is interpolated from the current target. "
            f"{pose} A new trajectory replaces the active one and an invalid one is ignored; verify "
            f"the {tip} with pose_state.",
            "topic", "input", f"/{name}/pose_target", "trajectory_msgs/msg/MultiDOFJointTrajectory")
        ops["pose_state"] = operation(
            f"Measured {tip} pose in {name}/base, stamped with joint_state.",
            "topic", "output", f"/{name}/pose_state", "sensor_msgs/msg/MultiDOFJointState",
            qos(durability="transient_local"))
    else:
        ops["joint_target"] = operation(
            f"Joint trajectory over {over} in {units}, timed from receipt by "
            "strictly increasing time_from_start and finishing within 120 seconds; velocities, "
            "accelerations and effort must be empty. A zero header stamp selects now; a nonzero "
            "stamp must not start in the past. Points are absolute native joint positions, "
            "linearly interpolated at the native control period. Replacement clears pending "
            "references; invalid trajectories are ignored, so verify measured joint_state.",
            "topic", "input", f"/{name}/joint_target", "trajectory_msgs/msg/JointTrajectory")
    ops["stop"] = operation(f"Stop {part} motion and hold the measured posture.", "service", "input",
                            f"/{name}/stop", "std_srvs/srv/Trigger")
    return device(name, "joint", description, ops,
                  model={"format": "urdf", "path": model, "base_link": "base_link",
                         "tip_links": [tip] if tip else []},
                  limits="native", home={joint: clean(q) for joint, q in zip(joints, home)},
                  stop={"behavior": "hold", "affected_devices": [name]},
                  frames={"base": base})


def gripper(name, description, joint, target, base, tcp, *, state=None, hold=None, opened=1.0,
            closed=0.0):
    ops = {
        "joint_state": operation(
            state or f"Measured {joint}, the native normalized gripper opening (0 closed, 1 open), "
            "stamped at acquisition; the URDF finger links follow it.",
            "topic", "output", f"/{name}/joint_state", "sensor_msgs/msg/JointState",
            qos(durability="transient_local")),
        "target": operation(target, "topic", "input", f"/{name}/target", "std_msgs/msg/Float64"),
        "stop": operation(hold or "Hold the measured opening; fingers may settle briefly after "
                          "the stop.", "service", "input", f"/{name}/stop", "std_srvs/srv/Trigger"),
    }
    return device(name, "gripper", description, ops, observation="joint_state",
                  stop={"behavior": "hold", "affected_devices": [name]},
                  frames={"base": base, "tcp": tcp},
                  model={"format": "urdf", "path": "gripper.urdf", "base_link": "base_link"},
                  grip={"joint": joint, "open": opened, "closed": closed})


def locomotion(name, description, axes, base, *, timeout_s, units="normalized"):
    """A velocity-driven base: axes maps vx, vy, ... to native (min, max) bounds, in [-1, 1] of
    the controller range (normalized) or m/s and rad/s (si)."""
    named = ", ".join(f"{field} ({axis})" for axis, field in (
        ("vx", "linear.x"), ("vy", "linear.y"), ("vz", "linear.z"), ("wx", "angular.x"),
        ("wy", "angular.y"), ("wz", "angular.z")) if axis in axes)
    scale = ("each normalized to [-1, 1] of the native controller range" if units == "normalized"
             else "in m/s and rad/s within the declared bounds")
    ops = {
        "velocity": operation(
            f"Body velocity in {name}/base axes as {named}, {scale}; other fields must be "
            "zero. The body follows the latest "
            f"twist and stops {timeout_s:g} s of world time after the last one, so re-send to "
            "keep moving; out-of-range twists are ignored, so verify with odometry.",
            "topic", "input", f"/{name}/velocity", "geometry_msgs/msg/Twist"),
        "odometry": operation(
            f"Measured body twist in {name}/base (m/s and rad/s at the body origin) and the "
            f"planar pose in {name}/odom, the body frame at episode start; never a global pose.",
            "topic", "output", f"/{name}/odometry", "nav_msgs/msg/Odometry",
            qos(durability="transient_local")),
        "stop": operation("Command zero velocity.", "service", "input", f"/{name}/stop",
                          "std_srvs/srv/Trigger"),
    }
    return device(name, "locomotion", description, ops, observation="odometry",
                  velocity={"units": units,
                            "axes": {axis: {"min": clean(low), "max": clean(high)}
                                     for axis, (low, high) in axes.items()}},
                  posture=None,
                  modes={"drive": {"description": "Drives on velocity commands.",
                                   "accepts": ["velocity"], "transitions": []}},
                  limits="strict", stop={"behavior": "halt", "affected_devices": [name]},
                  frames={"base": base})


def camera(name, description, intrinsics, base, depth_range=None, distortion=None):
    """An RGB or RGB-D camera; distortion is its lens's OpenCV Brown-Conrady (k1, k2, p1, p2, k3), or None for
    an undistorted pinhole. An undistorted pinhole's FOV and rays follow from the profile; the world node serves
    a distorted camera's geometry, unproject and project (rosbridge.core.lens)."""
    width, height = intrinsics[:2]
    color = cameras.color(*intrinsics, distortion)
    rgbd = depth_range is not None
    ops = {
        "image": operation(
            "RGB image measurements with encoding, dimensions, row stride and acquisition "
            "timestamp in the Image header. The frame_id identifies the calibrated optical frame; "
            "use its stamp for tf2 lookup. A new subscriber receives the current frame at once, "
            "also while the world is paused, then one per camera sample; reading never "
            "advances world time.", "topic", "output", f"/{name}/image", "sensor_msgs/msg/Image"),
    }
    if rgbd:
        ops["depth"] = operation(
            "16UC1 depth registered to image in millimetres along the optical axis; 0 marks "
            "missing or out-of-range samples. Rendered with its image: same stamp and frame_id, "
            "so pair the two by stamp.", "topic", "output", f"/{name}/depth",
            "sensor_msgs/msg/Image")
    if distortion is None:
        info = ("CameraInfo mirroring the profile, with K and P from color.projection, zero plumb_bob "
                "distortion, image size and optical frame. The camera's geometry is the profile's, "
                "not this topic's; intrinsics are not TF transforms.")
    else:
        info = ("CameraInfo mirroring the profile, with K and P from color.projection, plumb_bob "
                "distortion from color.distortion, image size and optical frame. The camera's geometry "
                "is the profile's and the geometry services', not this topic's; intrinsics are not TF "
                "transforms.")
        for operation_name, text in (
                ("geometry", "The active lens geometry: {} returns {streams: {color: descriptor}}, the "
                             "profile's width, height, projection and distortion with the optical frame and "
                             "a calibration_id."),
                ("unproject", "Pixels to optical rays through the lens: {stream, calibration_id, pixels: "
                              "[[u, v], ...]} returns {rays: [[x, y, 1], ...]}, null where invalid; at most "
                              "8192 pixels."),
                ("project", "Optical-frame points to pixels through the lens: {stream, calibration_id, "
                            "points: [[X, Y, Z], ...]} with Z > 0 returns {pixels: [[u, v], ...]}, null "
                            "where invalid; at most 8192 points.")):
            ops[operation_name] = operation(text, "service", "input", f"/{name}/{operation_name}",
                                            "custom_interfaces/srv/Json")
    ops["camera_info"] = operation(info, "topic", "output", f"/{name}/camera_info",
                                   "sensor_msgs/msg/CameraInfo", qos(durability="transient_local"))
    facts = {"color": color}
    if rgbd:
        near, far = depth_range
        span = {"min": clean(near, 6), "max": clean(min(far, 65.535), 6)}
        facts["depth"] = {"width": width, "height": height, "fps": None, "quantity": "z",
                          "scale_m_per_unit": 0.001, "invalid_value": 0, "valid_range_m": span,
                          "reliable_range_m": dict(span), "registration": "color"}
    facts["frames"] = {"base": base}
    return device(name, "rgbd" if rgbd else "rgb", description, ops, layout=None if rgbd else "mono", **facts)


def workspace(identifier, name, kind, model):
    return {"identity": {"id": identifier, "name": name, "kind": kind, "model": model,
                         "source": "user"},
            "devices": {"source": "user"},
            "safety": {"source": "user", "keep_out_zones": []},
            "meta": {"created_at": CREATED, "updated_at": CREATED}}


# Geometry ---------------------------------------------------------------------------------

def quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_xyzw(matrix):
    m = np.asarray(matrix, float)
    trace = np.trace(m)
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = [0., 0., 0., 0.]
        q[i], q[j], q[k] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.asarray(q)
    return q if q[3] >= 0 else -q


def matrix_rpy(matrix):
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    m = np.asarray(matrix, float)
    pitch = math.atan2(-m[2, 0], math.hypot(m[0, 0], m[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        return 0.0, pitch, math.atan2(-m[0, 1], m[1, 1])
    return math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])


def transform(position, rotation):
    value = np.eye(4)
    value[:3, :3], value[:3, 3] = rotation, position
    return value


def relative(parent, child):
    return np.linalg.inv(parent) @ child


def pose_frame(parent, matrix):
    return frame(parent, matrix[:3, 3], matrix_xyzw(matrix[:3, :3]))


# MuJoCo -----------------------------------------------------------------------------------

def body_local(model, body):
    """A body's fixed pose in its parent body frame."""
    return transform(model.body_pos[body], quat_matrix(model.body_quat[body]))


def body_world(data, body):
    return transform(data.xpos[body], data.xmat[body].reshape(3, 3))


def body_joint(model, body):
    jid = model.body_jntadr[body]
    if model.body_jntnum[body] != 1 or np.any(model.jnt_pos[jid]):
        raise ValueError(f"unexpected joint layout on {model.body(body).name}")
    return jid


def _mujoco_joint(model, body, parent, child):
    """The URDF joint into a body: its one hinge or slide joint at the body origin as a revolute
    or prismatic joint, or a fixed joint for a body without joints."""
    entry = {"parent": parent, "child": child, "origin": body_local(model, body)}
    if not model.body_jntnum[body]:
        return {**entry, "name": f"{child}_joint", "type": "fixed"}
    jid = body_joint(model, body)
    kind = {2: "prismatic", 3: "revolute"}[int(model.jnt_type[jid])]
    return {**entry, "name": model.joint(jid).name, "type": kind, "axis": model.jnt_axis[jid],
            "limits": model.jnt_range[jid]}


def mujoco_chain(model, bodies, name, comment, tip="flange"):
    """A kinematic-only serial chain through MuJoCo bodies: base_link is the first body's
    parent and the last body is the tip link."""
    links, joints = ["base_link"], []
    for index, body in enumerate(bodies, 1):
        child = tip if index == len(bodies) else f"link{index}"
        joints.append(_mujoco_joint(model, model.body(body).id, links[-1], child))
        links.append(child)
    return urdf(name, comment, links, joints)


def mujoco_tree(model, root, name, comment):
    """A kinematic-only copy of the MuJoCo subtree below body root, renamed base_link; the other
    links keep their body names."""
    root = model.body(root).id
    names, joints = {root: "base_link"}, []
    for body in range(root + 1, model.nbody):  # MuJoCo numbers a body after its parent
        parent = int(model.body_parentid[body])
        if parent in names:
            names[body] = model.body(body).name
            joints.append(_mujoco_joint(model, body, names[parent], names[body]))
    return urdf(name, comment, list(names.values()), joints)


# URDF -------------------------------------------------------------------------------------

def _number(value):
    return f"{clean(value):.12g}"


def urdf(name, comment, links, joints):
    """joints: dicts with name, type, parent, child, origin (4x4), axis, limits, mimic."""
    lines = ['<?xml version="1.0"?>', f"<!-- {comment} -->", f'<robot name="{name}">']
    lines += [f'  <link name="{link}"/>' for link in links]
    for joint in joints:
        lines.append(f'  <joint name="{joint["name"]}" type="{joint["type"]}">')
        lines.append(f'    <parent link="{joint["parent"]}"/>')
        lines.append(f'    <child link="{joint["child"]}"/>')
        origin = joint.get("origin", np.eye(4))
        xyz = " ".join(_number(v) for v in origin[:3, 3])
        rpy = " ".join(_number(v) for v in matrix_rpy(origin[:3, :3]))
        lines.append(f'    <origin xyz="{xyz}" rpy="{rpy}"/>')
        if joint["type"] != "fixed":
            lines.append(f'    <axis xyz="{" ".join(_number(v) for v in joint["axis"])}"/>')
            lower, upper = joint["limits"]
            lines.append(f'    <limit lower="{_number(lower)}" upper="{_number(upper)}" '
                         f'effort="{_number(joint.get("effort", 100))}" '
                         f'velocity="{_number(joint.get("velocity", 10))}"/>')
        if joint.get("mimic"):
            source, multiplier, offset = joint["mimic"]
            lines.append(f'    <mimic joint="{source}" multiplier="{_number(multiplier)}" '
                         f'offset="{_number(offset)}"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def urdf_joints(path):
    import xml.etree.ElementTree as ET
    return {joint.get("name"): joint for joint in ET.parse(path).getroot().findall("joint")}


def urdf_origin(joint):
    node = joint.find("origin")
    xyz = np.fromstring(node.get("xyz", "0 0 0") if node is not None else "0 0 0", sep=" ")
    roll, pitch, yaw = np.fromstring(node.get("rpy", "0 0 0") if node is not None else "0 0 0", sep=" ")
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return transform(xyz, rz @ ry @ rx)


def _motion(joint):
    limit = joint.find("limit")
    return {"axis": np.fromstring(joint.find("axis").get("xyz"), sep=" "),
            "limits": (float(limit.get("lower")), float(limit.get("upper")))}


def source_arm(source, joint_names, name, comment, tip="flange"):
    """A kinematic-only serial chain from a native URDF: base_link, link1.., tip."""
    links, chain, previous = ["base_link"], [], "base_link"
    for index, joint_name in enumerate(joint_names, 1):
        joint = source[joint_name]
        child = tip if index == len(joint_names) else f"link{index}"
        links.append(child)
        chain.append({"name": joint_name, "type": "revolute", "parent": previous, "child": child,
                      "origin": urdf_origin(joint), **_motion(joint)})
        previous = child
    return urdf(name, comment, links, chain)


def source_gripper(source, drive, fingers, lower, upper, name, comment, limits=(0, 1)):
    """A virtual drive joint over limits (open..closed command range) whose native finger joints
    mimic it linearly, from lower at limits[0] to upper at limits[1]."""
    low, high = limits
    scale = (upper - lower) / (high - low)
    chain = [{"name": drive, "type": "revolute", "parent": "base_link", "child": "drive",
              "axis": (0, 0, 1), "limits": limits}]
    links = ["base_link", "drive"]
    for finger, link in zip(fingers, ("finger_a", "finger_b")):
        joint = source[finger]
        links.append(link)
        chain.append({"name": finger, "type": "prismatic", "parent": "base_link", "child": link,
                      "origin": urdf_origin(joint), **_motion(joint),
                      "mimic": (drive, scale, lower - low * scale)})
    return urdf(name, comment, links, chain)


def source_tree(source, root, name, comment):
    """A kinematic-only copy of the native URDF subtree below link root, renamed base_link, with
    the native joint order."""
    subtree, grown = {root}, True
    while grown:
        children = {joint.find("child").get("link") for joint in source.values()
                    if joint.find("parent").get("link") in subtree}
        grown = not children <= subtree
        subtree |= children
    links, joints = ["base_link"], []
    for joint in source.values():
        parent = joint.find("parent").get("link")
        if parent not in subtree:
            continue
        child = joint.find("child").get("link")
        entry = {"name": joint.get("name"), "type": joint.get("type"), "child": child,
                 "parent": "base_link" if parent == root else parent, "origin": urdf_origin(joint)}
        if entry["type"] != "fixed":
            entry.update(_motion(joint))
        if (mimic := joint.find("mimic")) is not None:
            entry["mimic"] = (mimic.get("joint"), float(mimic.get("multiplier", 1)),
                              float(mimic.get("offset", 0)))
        links.append(child)
        joints.append(entry)
    return urdf(name, comment, links, joints)


def write_rig(directory, manifest, devices, files=None):
    """devices: name -> profile; files: relative path -> text, for URDFs."""
    directory = Path(directory)
    (directory / "devices").mkdir(parents=True, exist_ok=True)
    (directory / "workspace.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    for name, profile in devices.items():
        folder = directory / "devices" / name
        folder.mkdir(exist_ok=True)
        (folder / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False, width=88))
    for path, text in (files or {}).items():
        (directory / path).write_text(text)
