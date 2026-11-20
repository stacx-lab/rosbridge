"""Run the shared ROS node against the synthetic backend in a separate process."""
from rosbridge_runtime import sim_node
from tests.toy_backend import CameraToy, DepthToy, JointPoseToy, LensToy, PoseToy, SceneToy, ToyBackend

sim_node.load_backend = lambda name: {"pose_toy": PoseToy, "joint_pose_toy": JointPoseToy, "camera_toy": CameraToy,
                                      "depth_toy": DepthToy, "lens_toy": LensToy,
                                      "scene_toy": SceneToy}.get(name, ToyBackend)
sim_node.main()
