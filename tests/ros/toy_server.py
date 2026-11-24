"""Run the shared ROS node against the synthetic backend in a separate process."""
from rosbridge_runtime import sim_node
from tests.toy_backend import CameraToy, DepthToy, LensToy, PairPoseToy, PoseToy, SceneToy, ToyBackend

sim_node.load_backend = lambda name: {"pose_toy": PoseToy, "pair_pose_toy": PairPoseToy, "camera_toy": CameraToy,
                                      "depth_toy": DepthToy, "lens_toy": LensToy,
                                      "scene_toy": SceneToy}.get(name, ToyBackend)
sim_node.main()
