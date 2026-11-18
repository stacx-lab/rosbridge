"""Print scenes' public facts as JSON lines, without a ROS graph: one line per scene, in order, with an "error"
instead of the time limit and instruction when the scene cannot be built.

    [PSI_TIME_LIMIT_S=T] python -m rosbridge_psi_kitchen.describe TASK SEED SCENE [SCENE ...]
"""
import json
import os
import sys

from .backend import PsiKitchenSim


def describe(task, seed, scene):
    os.environ['PSI_SCENE'] = str(scene)
    sim = PsiKitchenSim(cameras=())
    try:
        sim.reset(task, seed)
        return {'task': task, 'scene': scene, 'seed': seed, 'time_limit_s': sim.max_steps * sim.period,
                'instruction': sim.env.get_ep_meta().get('lang', '')}
    except Exception as error:  # a scene the task cannot be built in is a result here
        return {'task': task, 'scene': scene, 'seed': seed, 'error': f'{type(error).__name__}: {error}'}
    finally:
        sim.close()


def main():
    task, seed = sys.argv[1], int(sys.argv[2])
    for scene in sys.argv[3:]:
        print(json.dumps(describe(task, seed, int(scene))), flush=True)


if __name__ == '__main__':
    main()
