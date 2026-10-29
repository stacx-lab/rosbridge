"""The world node's BEHAVIOR backend: a proxy for OmniGibson in its Python 3.11 worker."""
from rosbridge.core.worker import WorkerBackend


class BehaviorWorker(WorkerBackend):
    simulator = "behavior"
    # Isaac Sim shutdown can take a while; the proxy kills the worker after this.
    close_timeout_s = 120.
