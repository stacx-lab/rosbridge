"""Run the shared ROS node against the synthetic backend in a separate process."""
from rosbridge_runtime import sim_node
from tests.toy_backend import ToyBackend

sim_node.load_backend = lambda name: ToyBackend
sim_node.main()
