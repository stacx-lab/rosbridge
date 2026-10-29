"""The world node's BEHAVIOR backend with the G1 and Sharpa hands: a proxy for its OmniGibson worker."""
from rosbridge.core.worker import WorkerBackend


class BehaviorG1Worker(WorkerBackend):
    simulator = "behavior-g1"
    # Isaac Sim shutdown can take a while; the proxy kills the worker after this.
    close_timeout_s = 120.
