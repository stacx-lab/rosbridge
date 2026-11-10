"""Run the shared ROS node against the synthetic backend in a separate process."""
from rosbridge_runtime import sim_node
from tests.toy_backend import GoalToyBackend, ToyBackend

sim_node.load_backend = lambda name: GoalToyBackend if name == "goal" else ToyBackend
sim_node.main()
