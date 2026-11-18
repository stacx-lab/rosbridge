"""Run the shared ROS node against the synthetic backend in a separate process."""
from rosbridge_runtime import sim_node
from tests.toy_backend import CameraToy, LensToy, PoseToy, ToyBackend

sim_node.load_backend = lambda name: {"pose_toy": PoseToy, "camera_toy": CameraToy,
                                      "lens_toy": LensToy}.get(name, ToyBackend)
sim_node.main()
