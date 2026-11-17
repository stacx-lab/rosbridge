"""A camera profile's color geometry: its pinhole and, for a distorted lens, OpenCV's Brown-Conrady model.

Kinex derives an undistorted pinhole's field of view and rays from its profile. A distorted camera's profile
states its field of view, and the camera's node serves its rays through the same model
(kinex.common.camera_geometry.OpenCVGeometry).
"""
import math

import numpy as np

BROWN_CONRADY = ("k1", "k2", "p1", "p2", "k3")


def clean(value, digits=12):
    value = round(float(value), digits)
    return 0.0 if value == 0 else value


def distort(x, y, coefficients):
    """Normalized image coordinates through the lens (k1, k2, p1, p2, k3), as OpenCV projects them."""
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort(xd, yd, coefficients, iterations=100):
    """The normalized coordinates the lens takes to (xd, yd), by fixed-point iteration."""
    x, y = xd, yd
    for _ in range(iterations):
        u, v = distort(x, y, coefficients)
        x, y = x + xd - u, y + yd - v
    return x, y


def fov(width, height, fx, fy, cx, cy, coefficients):
    """Horizontal and vertical field of view in degrees, between the rays through the image's edge midpoints,
    as Kinex derives it for an undistorted pinhole."""
    def ray(u, v):
        return np.array([*undistort((u - cx) / fx, (v - cy) / fy, coefficients), 1.])

    def angle(a, b):
        return math.degrees(math.acos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    middle_u, middle_v = (width - 1) / 2, (height - 1) / 2
    return {"horizontal_deg": clean(angle(ray(-.5, middle_v), ray(width - .5, middle_v))),
            "vertical_deg": clean(angle(ray(middle_u, -.5), ray(middle_u, height - .5)))}


def color(width, height, fx, fy, cx, cy, distortion=None):
    """Profile color geometry (ROS pixel centers); distortion is the lens's (k1, k2, p1, p2, k3), or None for an
    undistorted pinhole. fps is null: simulated frames render on demand at camera samples."""
    value = {"width": width, "height": height, "fps": None,
             "projection": {"model": "pinhole", "fx": clean(fx), "fy": clean(fy), "cx": clean(cx),
                            "cy": clean(cy), "skew": 0},
             "distortion": {"model": "none"}}
    if distortion is not None:
        value["distortion"] = {"model": "opencv_brown_conrady",
                               **{key: clean(v) for key, v in zip(BROWN_CONRADY, distortion, strict=True)}}
        value["fov"] = fov(width, height, fx, fy, cx, cy, distortion)
    return value
