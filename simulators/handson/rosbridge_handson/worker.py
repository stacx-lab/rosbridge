"""The world node's HandsOn backend: a proxy for Isaac Sim 5.1 in its Python 3.11 worker."""
from rosbridge.core.worker import WorkerBackend


class HandsOnWorker(WorkerBackend):
    simulator = "handson"
    # Isaac Sim shutdown can take a while; the proxy kills the worker after this.
    close_timeout_s = 120.

    def measures(self):
        """The native failure that ended the trial, if one did (HandsOnSim.measures)."""
        return self._call("measures")
