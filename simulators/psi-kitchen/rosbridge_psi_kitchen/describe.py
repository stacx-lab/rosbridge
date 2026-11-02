"""Print a task variant's public facts as one JSON line (last line of output), without a ROS graph.

    python -m rosbridge_psi_kitchen.describe TASK SEED
"""
import json
import sys

from .backend import PsiKitchenSim


def main():
    task, seed = sys.argv[1], int(sys.argv[2])
    sim = PsiKitchenSim(cameras=())
    try:
        sim.reset(task, seed)
        print(json.dumps({'task': sim.task, 'seed': seed, 'max_steps': sim.max_steps,
                          'instruction': sim.env.get_ep_meta().get('lang', '')}), flush=True)
    finally:
        sim.close()


if __name__ == '__main__':
    main()
