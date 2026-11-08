"""The world node's BEHAVIOR backend with the R1Pro and Sharpa hands: a proxy for its OmniGibson worker."""
from rosbridge.core.worker import WorkerBackend


class BehaviorDexWorker(WorkerBackend):
    simulator = "behavior-dex"
    # Isaac Sim shutdown can take a while; the proxy kills the worker after this.
    close_timeout_s = 120.

    def measures(self):
        """The episode's q_score, as BEHAVIOR's evaluator computes it (BehaviorSim.measures)."""
        return self._call("measures")
