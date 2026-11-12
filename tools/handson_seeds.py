"""Pre-screen HandsOn seeds: which pass upstream's settle gate and leave the scene in place, booting each task and
seed once without ROS.

    pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \\
        python tools/handson_seeds.py --tasks t27_place_largest_fitting_ball t23-photograph-logo \\
        --seeds 0-9 --gpus 0,1 --output seeds.json

Each task and seed resets HandsOnSim in a process of its own, as a launch does (Isaac hosts one episode per
process), on one of --gpus (--per-gpu processes each), then closes it; each costs about a launch's startup. A seed
is accepted when upstream's settle gate accepts it and every task body the native environment exposes ends the gate
within DRIFT_M of where the reset put it. The gate checks only that the scene comes to rest, so it accepts a scene
whose object was thrown off the table and lies still on the floor, as t09's socket tray does at seed 0. Both checks
are deterministic, so the result holds for later launches of the same task and seed. Writes (or prints)
{task: {seed: {"accepted": true|false, "seconds": s, "prompt" or "error": ...}}}, with the number of task bodies and
their largest displacement across the gate (drift_m) where the gate accepts; a rejected seed also names what rejected
it, "rejected": "settle_gate" or "scene" (with the bodies that moved); accepted is null when the reset failed for
another reason.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

MARK = "handson-seed: "
# A task body that moves farther than this while the settle gate settles the scene has left its place. Across the gate,
# bodies that stay in place move at most a few millimetres (3.7 mm in all 66 tasks at seed 0).
DRIFT_M = .02


def seeds(text):
    """0-9 or 1,4,7."""
    if "-" in text:
        first, last = map(int, text.split("-"))
        return list(range(first, last + 1))
    return [int(value) for value in text.split(",")]


def positions(env):
    """Every task body's world position: the native environment's PhysX body views, else its object poses."""
    import numpy as np
    views = getattr(env, "_task_body_views", None)
    if views:
        poses = {name: view.get_transforms()[0] for name, view in views.items()}
    else:
        env._refresh_object_poses()
        poses = getattr(env, "_poses", None) or {}
    return {name: np.asarray(pose.cpu() if hasattr(pose, "cpu") else pose, dtype=float).reshape(-1)[:3]
            for name, pose in poses.items()}


def scene(before, after):
    """The bodies' largest displacement between two position readings, and the bodies that moved more than DRIFT_M."""
    import numpy as np
    drift = {name: float(np.linalg.norm(after[name] - before[name])) for name in before if name in after}
    moved = {name: dict(reset=before[name].round(4).tolist(), settled=after[name].round(4).tolist(),
                        moved_m=round(distance, 4)) for name, distance in drift.items() if distance > DRIFT_M}
    return round(max(drift.values(), default=0.), 4), moved


def one(task, seed):
    """Reset one episode and report on stdout before closing, since closing Isaac may end this process."""
    import torch  # noqa: F401  first, as the backend imports it
    from handson.evaluation import task_contracts
    from rosbridge.core import SeedRejected
    from rosbridge_handson.backend import HandsOnSim
    os.environ.pop("ROSBRIDGE_RUN_DIR", None)  # receipts go to a temporary directory
    readings, initialize = [], task_contracts.initialize

    def gate(env, *args):
        """Upstream's settle gate, with the task bodies' positions read before and after it."""
        readings.append(positions(env))
        settled = initialize(env, *args)
        readings.append(positions(env))
        return settled
    task_contracts.initialize = gate
    sim, started = HandsOnSim(), time.monotonic()
    try:
        sim.reset(task, seed)
        drift, moved = scene(*readings)
        result = dict(accepted=not moved, bodies=len(readings[0]), drift_m=drift,
                      prompt=json.loads((sim._out / "PROMPT.json").read_text())["prompt"])
        if moved:
            result.update(rejected="scene", moved=moved, error=f"{task} seed {seed}: the settle gate moved "
                          + ", ".join(f"{name} {body['moved_m']:.3f} m" for name, body in moved.items()))
    except SeedRejected as error:
        result = dict(accepted=False, rejected="settle_gate", error=str(error))
    except Exception as error:  # noqa: BLE001  not a seed's verdict
        result = dict(accepted=None, error=f"{type(error).__name__}: {error}"[:500])
    print(MARK + json.dumps(dict(result, seconds=round(time.monotonic() - started, 1))), flush=True)
    sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", help="native task names or HandsOn task IDs (required)")
    parser.add_argument("--seeds", type=seeds, help="a range such as 0-9, or a list such as 1,4,7 (required)")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices (CUDA_VISIBLE_DEVICES of each run)")
    parser.add_argument("--per-gpu", type=int, default=1, help="concurrent resets per GPU (about 14 GB each)")
    parser.add_argument("--timeout", type=float, default=1800., help="wall seconds per reset")
    parser.add_argument("--output", help="JSON file to write; default: stdout")
    parser.add_argument("--one", nargs=2, metavar=("TASK", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:  # a worker of this tool
        return one(args.one[0], int(args.one[1]))
    if not args.tasks or args.seeds is None:
        parser.error("--tasks and --seeds are required")
    jobs = queue.SimpleQueue()
    for task in args.tasks:
        for seed in args.seeds:
            jobs.put((task, seed))
    results, lock = {task: {} for task in args.tasks}, threading.Lock()

    def slot(gpu):
        while True:
            try:
                task, seed = jobs.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            try:
                # Kit's log may hold bytes that are not UTF-8.
                run = subprocess.run([sys.executable, __file__, "--one", task, str(seed)], env=env, text=True,
                                     errors="replace", stdin=subprocess.DEVNULL, capture_output=True,
                                     timeout=args.timeout)
                lines = [line for line in run.stdout.splitlines() if line.startswith(MARK)]
                result = (json.loads(lines[-1][len(MARK):]) if lines else
                          dict(accepted=None, error=f"no result (exit {run.returncode}): {run.stderr[-300:]}"))
            except subprocess.TimeoutExpired:
                result = dict(accepted=None, error=f"timed out after {args.timeout:g} s")
            except Exception as error:  # noqa: BLE001  a slot keeps serving the queue whatever one job raised
                result = dict(accepted=None, error=f"{type(error).__name__}: {error}"[:300])
            with lock:
                results[task][seed] = dict(result, gpu=gpu)
                verdict = {True: "accepted", False: "rejected", None: "error"}[result["accepted"]]
                print(f"{task} seed {seed}: {verdict}" + (f" ({result['rejected']})" if "rejected" in result else ""),
                      file=sys.stderr, flush=True)

    threads = [threading.Thread(target=slot, args=(gpu,)) for gpu in args.gpus.split(",") for _ in range(args.per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    text = json.dumps({task: dict(sorted(entries.items())) for task, entries in results.items()}, indent=1)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
