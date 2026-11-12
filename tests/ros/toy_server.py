"""Run the shared ROS node against the synthetic backend in a separate process."""
from rosbridge_runtime import sim_node
from tests.toy_backend import StumblingToy

sim_node.load_backend = lambda name: StumblingToy
sim_node.main()
