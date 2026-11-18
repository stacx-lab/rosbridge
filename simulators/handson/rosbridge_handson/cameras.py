"""HandsOn's cameras for a task and seed, as upstream's evaluator renders them, without Isaac.

The task's camera profile (evaluation/camera_contracts.py: select) gives each camera's intrinsics and nominal
mount; t05 and t15 have a profile of their own. Each episode draws a small mount offset per camera
(evaluation/camera_randomization.py: sample, at most 2 mm and 0.3 deg per axis) from its camera seed: the
setting's camera seed for the 16 tasks with a setting, else the native seed (evaluation/native.py: run_episode).
The head camera renders through its calibrated lens, OpenCV's Brown-Conrady model, which its rig profile states and
the world node serves; the wrist cameras are undistorted pinholes.
"""
import numpy as np

CAMERAS = {"head_camera": "main", "left_camera": "wrist_left", "right_camera": "wrist_right"}
OPENGL_TO_OPTICAL = np.diag([1., -1., -1., 1.])
# The storage id of upstream's render_probe draw.
PROBE = "independent-camera-probe"


def pose_matrix(position, quaternion_wxyz):
    w, x, y, z = np.asarray(quaternion_wxyz, float) / np.linalg.norm(quaternion_wxyz)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    matrix[:3, 3] = position
    return matrix


def lens(spec):
    """A contract camera's width, height, fx, fy, cx, cy in ROS pixel coordinates, as it renders. Upstream authors
    its K on Isaac's OpenCV lens (calibrated_cameras.apply_lens), which places the principal point with pixel
    corners at integers: rendered against an undistorted pinhole at the same pose, every camera's image lies half
    a pixel off its K in ROS (and OpenCV) pixel centers. So its principal point is K's less half a pixel."""
    (fx, _, cx), (_, fy, cy), _ = spec["K_effective"]
    width, height = spec["native_width_height"]
    return width, height, float(fx), float(fy), float(cx) - .5, float(cy) - .5


def distortion(spec):
    """A contract camera's lens as OpenCV Brown-Conrady (k1, k2, p1, p2, k3), or None for an undistorted pinhole."""
    d = [float(v) for v in spec["D_effective"]]
    if any(d[5:]):
        raise ValueError(f"lens {d} has rational terms; the lens geometry takes k1, k2, p1, p2, k3")
    return tuple(d[:5]) if any(d) else None


def contract(task):
    """The task's camera contract."""
    from handson.evaluation import camera_contracts
    value, _ = camera_contracts.select(task)
    return value


def camera_seed(task, native_seed):
    """The seed of an episode's camera draw."""
    from handson.evaluation import settings
    if task not in settings.task_ids():
        return native_seed
    return settings.select_setting(native_seed, task, 0)["seeds"]["camera"]


def offsets(task, seed):
    """Each camera's drawn mount offset, an OpenGL camera-local 4x4 (identity with seed None: the nominal mount)."""
    if seed is None:
        return {device: np.eye(4) for device in CAMERAS}
    from handson.evaluation.camera_randomization import sample
    draw = sample(task, PROBE, seed)["cameras"]
    return {device: pose_matrix(draw[role]["translation_m"], draw[role]["quaternion_wxyz"])
            for device, role in CAMERAS.items()}


def mounts(value, offset):
    """Each camera's optical-frame mount: the head camera in the world, a wrist camera on its palm."""
    result = {}
    for device, role in CAMERAS.items():
        spec = value["cameras"][role]
        nominal = np.asarray(spec["nominal_world_pose_opengl" if role == "main" else "nominal_camera_to_palm_opengl"])
        result[device] = nominal @ offset[device] @ OPENGL_TO_OPTICAL
    return result


def matrix_xyzw(matrix):
    """A rotation matrix's unit quaternion (x, y, z, w) with w >= 0."""
    m = np.asarray(matrix, float)
    trace = np.trace(m)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = [0., 0., 0., 0.]
        q[i], q[j], q[k] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.asarray(q)
    return q if q[3] >= 0 else -q
