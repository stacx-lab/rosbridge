"""Pre-screen HandsOn seeds: which pass upstream's settle gate, booting each task and seed once without ROS.

    pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \\
        python tools/handson_seeds.py --tasks t27_place_largest_fitting_ball t23-photograph-logo \\
        --seeds 0-9 --gpus 0,1 --output seeds.json

Each task and seed resets HandsOnSim in a process of its own, as a launch does (Isaac hosts one episode per
process), on one of --gpus (--per-gpu processes each), then closes it; each costs about a launch's startup. The
gate is deterministic, so the result holds for later launches of the same task and seed. Writes (or prints)
{task: {seed: {"accepted": true|false, "seconds": s, "prompt" or "error": ...}}}; accepted is null when the
reset failed for another reason than a rejected seed.
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


def seeds(text):
    """0-9 or 1,4,7."""
    if "-" in text:
        first, last = map(int, text.split("-"))
        return list(range(first, last + 1))
    return [int(value) for value in text.split(",")]


def one(task, seed):
    """Reset one episode and report on stdout before closing, since closing Isaac may end this process."""
    from rosbridge.core import SeedRejected
    from rosbridge_handson.backend import HandsOnSim
    os.environ.pop("ROSBRIDGE_RUN_DIR", None)  # receipts go to a temporary directory
    sim, started = HandsOnSim(), time.monotonic()
    try:
        sim.reset(task, seed)
        result = dict(accepted=True, prompt=json.loads((sim._out / "PROMPT.json").read_text())["prompt"])
    except SeedRejected as error:
        result = dict(accepted=False, error=str(error))
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
                run = subprocess.run([sys.executable, __file__, "--one", task, str(seed)], env=env, text=True,
                                     stdin=subprocess.DEVNULL, capture_output=True, timeout=args.timeout)
                lines = [line for line in run.stdout.splitlines() if line.startswith(MARK)]
                result = (json.loads(lines[-1][len(MARK):]) if lines else
                          dict(accepted=None, error=f"no result (exit {run.returncode}): {run.stderr[-300:]}"))
            except subprocess.TimeoutExpired:
                result = dict(accepted=None, error=f"timed out after {args.timeout:g} s")
            with lock:
                results[task][seed] = dict(result, gpu=gpu)
                print(f"{task} seed {seed}: " + {True: "accepted", False: "rejected", None: "error"}[result["accepted"]],
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
