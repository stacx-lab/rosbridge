"""Serve the camera's lens geometry from its profile; Kinex computes undistorted pinholes' rays itself."""
from rclpy.node import Node

from kinex.common.camera_geometry import OpenCVGeometry, serve_geometry
from kinex.node import node_name


def create(profile, context):
    node = Node(node_name(profile, "geometry"), context=context)
    color = {key: profile["color"][key] for key in ("width", "height", "projection", "distortion")}
    serve_geometry(node, profile, {"color": OpenCVGeometry({**color, "frame": f"{profile['name']}/base"})})
    return node
