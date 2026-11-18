"""A lensed camera's geometry operations: its descriptor, and pixels to rays and points to pixels through the lens.

The world node serves them for each camera its backend renders through a lens (SimulatorBackend.distortion), as
free-form JSON services. `geometry` takes {} and returns {streams: {color: descriptor}}: the profile's width,
height, projection and distortion with the optical frame and a calibration_id. `unproject` and `project` name
the stream and that calibration_id, so a reader holding another calibration is refused, and convert at most
LIMIT coordinates; an entry that cannot be converted comes back as null.
"""
import copy
import uuid

import numpy as np

from rosbridge.catalog.cameras import BROWN_CONRADY, color, distort, undistort

LIMIT = 8192


class LensGeometry:
    """The color stream of a camera with an OpenCV Brown-Conrady lens (k1, k2, p1, p2, k3), in ROS pixel centers."""

    def __init__(self, frame, width, height, fx, fy, cx, cy, distortion):
        stated = color(width, height, fx, fy, cx, cy, distortion)
        self.descriptor = {**{key: stated[key] for key in ("width", "height", "projection", "distortion")},
                           "frame": frame, "calibration_id": uuid.uuid4().hex}
        # Convert with the stated values, so a reader computing from the descriptor agrees.
        projection = stated["projection"]
        self.pinhole = tuple(projection[key] for key in ("fx", "fy", "cx", "cy"))
        self.lens = tuple(stated["distortion"][key] for key in BROWN_CONRADY)

    def geometry(self):
        return {"streams": {"color": copy.deepcopy(self.descriptor)}}

    def unproject(self, stream, calibration_id, pixels):
        """Pixels [[u, v], ...] to optical rays [[x, y, 1], ...]."""
        return {"rays": self._convert(stream, calibration_id, pixels, 2, self.rays)}

    def project(self, stream, calibration_id, points):
        """Optical-frame points [[X, Y, Z], ...] with Z > 0 to pixels [[u, v], ...]."""
        return {"pixels": self._convert(stream, calibration_id, points, 3, self.pixels)}

    def rays(self, pixels):
        fx, fy, cx, cy = self.pinhole
        u, v = np.asarray(pixels, float).T
        x, y = undistort((u - cx) / fx, (v - cy) / fy, self.lens)
        return np.column_stack((x, y, np.ones_like(x)))

    def pixels(self, points):
        fx, fy, cx, cy = self.pinhole
        x, y, z = np.asarray(points, float).T
        xd, yd = distort(x / z, y / z, self.lens)
        return np.column_stack((fx * xd + cx, fy * yd + cy))

    def _convert(self, stream, calibration_id, values, size, function):
        if stream != "color" or calibration_id != self.descriptor["calibration_id"]:
            raise ValueError("stale or unknown camera calibration; refresh geometry")
        if not isinstance(values, list) or len(values) > LIMIT:
            raise ValueError(f"geometry batch must contain at most {LIMIT} coordinates")
        output, valid, indices = [], [], []
        for index, value in enumerate(values):
            if not isinstance(value, list) or len(value) != size:
                raise ValueError("invalid coordinate shape")
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in value):
                raise ValueError("coordinates must be numbers")
            output.append(None)
            if np.isfinite(value).all() and (size == 2 or value[2] > 0):
                indices.append(index)
                valid.append(value)
        if valid:
            with np.errstate(all="ignore"):  # a ray far outside the lens diverges, and comes back as null
                converted = function(valid)
            for index, value in zip(indices, converted):
                if np.isfinite(value).all():
                    output[index] = value.tolist()
        return output
