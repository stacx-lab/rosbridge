"""The world node's RoboDojo backend: a proxy for Isaac Sim 5.1 in its Python 3.11 worker."""
from rosbridge.core.worker import WorkerBackend


class RoboDojoWorker(WorkerBackend):
    simulator = "robodojo"
    # Isaac Sim shutdown can take a while; the proxy kills the worker after this.
    close_timeout_s = 120.
