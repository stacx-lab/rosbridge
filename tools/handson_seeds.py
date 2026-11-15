"""Screen HandsOn's native seeds and write the table a launch's seed reads (simulators/handson/seeds.json).

    pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \\
        python tools/handson_seeds.py --gpus 0,1,2,3,4,5,6,7 --per-gpu 2

For each task, the screen tries native seeds in the order 0, 1, 2, ... and keeps the first --count that it accepts;
a launch's seed k names the k-th. A seed is accepted when upstream's settle gate accepts it and every task body the
native environment exposes ends the gate within DRIFT_M of where the reset put it. The gate checks only that the
scene comes to rest, so it accepts a scene whose object slid out of place and stopped, as t09's socket tray does by
6 cm at seed 0. Both checks are deterministic: that seed's tray moves 0.0607 m on every GPU.

Each candidate resets HandsOnSim at its native seed in a process of its own, as a launch does (Isaac hosts one
episode per process), on one of --gpus (--per-gpu processes each), then closes it; each costs about a launch's
startup. A slot starts a reset only while its GPU has --min-free-gb free, and one start at a time per GPU, so other
jobs keep their memory. A task takes candidates in order while its accepted and running candidates are fewer than
--count (rosbridge_handson.seeds.Screen). A candidate whose reset fails for another reason runs once more; a second
error, or --max-candidates seeds without --count accepted, leaves the task incomplete, and the screen exits 1.

The table (--output) is rewritten after every verdict and read back on a rerun, which screens only what is
missing: an interrupted screen resumes, and a larger --count extends it. Every run's detail (seconds, drift,
error, GPU) is appended to <output>.log.jsonl, which is not committed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulators/handson"))
from rosbridge_handson.seeds import SEEDS, Screen  # noqa: E402

MARK = "handson-seed: "
# A task body that moves farther than this while the settle gate settles the scene has left its place. Across the gate,
# bodies that stay in place move at most a few millimetres (3.7 mm in all 66 tasks at seed 0).
DRIFT_M = .02
# A reset's memory grows for about this long after its start (a whole reset takes about a minute); a GPU starts its
# next reset after it.
START_S = 30.


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
    """Reset one episode at a native seed and report on stdout before closing, since closing Isaac may end this
    process."""
    import torch  # noqa: F401  first, as the backend imports it
    from handson.evaluation import task_contracts
    from rosbridge_handson.backend import HandsOnSim, SettleGateRejected
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
        sim._reset(task, seed, seed)
        drift, moved = scene(*readings)
        result = dict(accepted=not moved, bodies=len(readings[0]), drift_m=drift)
        if moved:
            result.update(rejected="scene", moved=moved)
    except SettleGateRejected as error:
        result = dict(accepted=False, rejected="settle_gate", error=str(error)[:500])
    except Exception as error:  # noqa: BLE001  not a seed's verdict
        result = dict(accepted=None, error=f"{type(error).__name__}: {error}"[:500])
    print(MARK + json.dumps(dict(result, seconds=round(time.monotonic() - started, 1))), flush=True)
    sim.close()


def free_gb(gpu):
    run = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    return float(run.stdout.strip()) / 1024


def gpu_name(gpu):
    run = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return run.stdout.strip() or None


def write(path, table):
    """Replace the table atomically, so an interrupted screen leaves the previous one."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(table, indent=1) + "\n")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=10, help="seeds each task keeps (default: 10)")
    parser.add_argument("--tasks", nargs="+", help="native task names or HandsOn task IDs (default: every task)")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices (CUDA_VISIBLE_DEVICES of each run)")
    parser.add_argument("--per-gpu", type=int, default=1, help="concurrent resets per GPU (about 14 GB each)")
    parser.add_argument("--min-free-gb", type=float, default=22., help="free GPU memory a reset needs to start")
    parser.add_argument("--max-candidates", type=int, default=100, help="native seeds a task may try")
    parser.add_argument("--timeout", type=float, default=1800., help="wall seconds per reset")
    parser.add_argument("--output", type=Path, default=SEEDS, help=f"the table (default: {SEEDS.relative_to(ROOT)})")
    parser.add_argument("--one", nargs=2, metavar=("TASK", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:  # a worker of this tool
        return one(args.one[0], int(args.one[1]))
    from rosbridge.catalog import resolve, tasks as catalog
    names = ([resolve("handson", task=task)[2]["native_task"] for task in args.tasks] if args.tasks else
             [task["native_task"] for task in catalog(name="handson").values()])
    revision = tomllib.loads((ROOT / "simulators/handson/simulator.toml").read_text())["source_revision"]
    gpus = args.gpus.split(",")
    facts = dict(drift_m=DRIFT_M, source_revision=revision, gpu=gpu_name(gpus[0]))
    previous = json.loads(args.output.read_text()) if args.output.is_file() else {}
    for key in ("drift_m", "source_revision"):
        if previous and previous.get(key) != facts[key]:
            parser.error(f"{args.output} was screened with {key} {previous.get(key)!r}, not {facts[key]!r}; "
                         "move it away to screen afresh")
    # Tasks the table holds but this screen does not select stay as they are.
    kept = {task: entry for task, entry in previous.get("tasks", {}).items() if task not in names}
    screen = Screen(names, args.count, args.max_candidates, previous)
    lock, log = threading.Lock(), args.output.with_name(args.output.name + ".log.jsonl")
    starts = {gpu: threading.Lock() for gpu in gpus}

    def save():
        table = screen.table(**facts)
        table["tasks"] = dict(sorted({**kept, **table["tasks"]}.items()))
        write(args.output, table)

    def run(task, seed, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        with starts[gpu]:
            while free_gb(gpu) < args.min_free_gb:
                time.sleep(30)
            # Kit's log may hold bytes that are not UTF-8.
            process = subprocess.Popen([sys.executable, __file__, "--one", task, str(seed)], env=env, text=True,
                                       errors="replace", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            output = {}
            reader = threading.Thread(target=lambda: output.update(zip(("stdout", "stderr"), process.communicate())))
            reader.start()
            reader.join(START_S)
        reader.join(args.timeout)
        if reader.is_alive():
            process.kill()
            reader.join()
            return dict(accepted=None, error=f"timed out after {args.timeout:g} s")
        lines = [line for line in output["stdout"].splitlines() if line.startswith(MARK)]
        return (json.loads(lines[-1][len(MARK):]) if lines else
                dict(accepted=None, error=f"no result (exit {process.returncode}): {output['stderr'][-300:]}"))

    def slot(gpu):
        while True:
            with lock:
                job = screen.next()
                if job is None and screen.done():
                    return
            if job is None:
                time.sleep(5)  # a running candidate's verdict may open another
                continue
            task, seed = job
            try:
                result = run(task, seed, gpu)
            except Exception as error:  # noqa: BLE001  a slot keeps serving the queue whatever one job raised
                result = dict(accepted=None, error=f"{type(error).__name__}: {error}"[:300])
            with lock:
                screen.record(task, seed, result)
                save()
                with log.open("a") as stream:
                    stream.write(json.dumps(dict(result, task=task, seed=seed, gpu=gpu)) + "\n")
                verdict = {True: "accepted", False: "rejected", None: "error"}[result["accepted"]]
                print(f"{task} seed {seed}: {verdict}" + (f" ({result['rejected']})" if "rejected" in result else "")
                      + f"; {len(screen.accepted(task))}/{args.count} accepted", file=sys.stderr, flush=True)

    threads = [threading.Thread(target=slot, args=(gpu,)) for gpu in gpus for _ in range(args.per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    save()
    for task, why in sorted(screen.incomplete.items()):
        print(f"incomplete: {task}: {why}", file=sys.stderr)
    return 1 if screen.incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
