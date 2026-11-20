"""Screen RoboTwin's seeds as upstream's evaluation does and write the table a launch's seed reads
(simulators/robotwin/seeds.json).

    pixi run --frozen -e robotwin-screen install-curobo   # once
    pixi run --frozen -e robotwin-screen python tools/robotwin_seeds.py --gpus 0,1,2,4

Upstream's evaluation (scripts/eval_policy_xpolicylab.py: main, eval_remote_policy) tries the seeds
100000 · (1 + eval seed) and up, eval seed 0 here, in order, and evaluates a seed only once its expert
check has passed:
  - setup_demo, the scripted expert (play_once) and close_env, without UnStableError or any other error;
  - then the expert's plan_success and check_success;
  - then the evaluation's own setup_demo, without error.
The k-th seed it evaluates is a launch's seed k.

The screen runs that check with upstream's own code: the pinned revision exported unpatched to
.runtime/robotwin-screen/tree, this checkout's prepared assets, upstream's task arguments (load_task_args),
its planner (cuRobo for aloha-agilex, from the robotwin-screen environment) and its edits to mplib and
SAPIEN (scripts/_install.sh).

Each task screens in a process of its own on one of --gpus (--per-gpu processes each). Within that process
one task environment serves every candidate in order, as upstream keeps one. Screen on idle GPUs, one task
each: upstream plans with cuRobo under a 10 s wall-clock budget, so on a loaded GPU verdicts shift both ways
(a screen at two tasks per busy GPU differed from an idle one in 18 of 50 tasks). Even on idle GPUs, a borderline seed's verdict can flip between runs, about one seed in twenty to
thirty in the hardest tasks, as it would in upstream's own evaluation: the table is one run of
upstream's check. The table (--output) is
rewritten after every verdict and read back on a rerun, so an interrupted screen resumes and a larger
--count extends it. Every verdict is appended to <output>.log.jsonl and each task's stderr to
<output>.<task>.stderr; neither is committed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "simulators/robotwin/seeds.json"
TREE = ROOT / ".runtime/robotwin-screen/tree"
MARK = "robotwin-seed: "
EVAL_SEED = 0
# upstream's scripts/_install.sh edits, as (installed file, pattern, replacement)
EDITS = (("mplib/planner.py", r"(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)",
          r"\1\3"),
         ("sapien/wrapper/urdf_loader.py", r'("r")(\))( as)', r'\1, encoding="utf-8") as'))


def manifest():
    return tomllib.loads((ROOT / "simulators/robotwin/simulator.toml").read_text())


def export(revision):
    """The pinned upstream revision, unpatched, with this checkout's prepared assets: objects linked, and the
    embodiments mirrored file by file, so that upstream's own path step (scripts/_download_assets.sh:
    update_embodiment_config_path.py) writes the cuRobo configurations next to them, not into the asset store."""
    marker = TREE / ".revision"
    if marker.is_file() and marker.read_text() == revision:
        return
    shutil.rmtree(TREE, ignore_errors=True)
    TREE.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(ROOT / manifest()["source"]), "archive", revision],
                             capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(TREE)], input=archive, check=True)
    assets = ROOT / ".runtime/robotwin/assets"  # rosbridge_robotwin.assets.ARCHIVES
    (TREE / "assets/objects").symlink_to((assets / "objects").resolve())
    shutil.copytree((assets / "embodiments").resolve(), TREE / "assets/embodiments", copy_function=os.symlink)
    subprocess.run([sys.executable, "scripts/update_embodiment_config_path.py"], cwd=TREE, check=True,
                   stdin=subprocess.DEVNULL, capture_output=True)
    marker.write_text(revision)


def upstream_edits():
    """Apply upstream's edits to this environment's mplib and SAPIEN, once."""
    import importlib.util
    for relative, pattern, replacement in EDITS:
        package, _, rest = relative.partition("/")
        path = Path(importlib.util.find_spec(package).submodule_search_locations[0]) / rest
        text = path.read_text()
        edited = re.sub(pattern, replacement, text)
        if edited != text:
            path.write_text(edited)


def one(task, seed, accepted, count):
    """Screen one task's candidates from seed on, reporting each verdict on stdout."""
    import importlib.util
    os.chdir(TREE)  # upstream runs from its tree; tasks read assets by relative path
    spec = importlib.util.spec_from_file_location("eval_policy", TREE / "scripts/eval_policy_xpolicylab.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    from envs.utils.create_actor import UnStableError
    args, _ = upstream.load_task_args({"task_name": task, "policy_name": "seed-screen", "task_config": "demo_clean"})
    args["eval_mode"], args["render_freq"] = True, 0  # as eval_remote_policy runs the expert check
    task_env = upstream.class_decorator(task)
    while accepted < count:
        started, verdict = time.monotonic(), {"seed": seed}
        try:
            task_env.setup_demo(now_ep_num=accepted, seed=seed, is_test=True, **args)
            task_env.play_once()
            task_env.close_env()
            if not (task_env.plan_success and task_env.check_success()):
                verdict["rejected"] = "the expert did not solve it"
            else:
                try:
                    task_env.setup_demo(now_ep_num=accepted, seed=seed, is_test=True, **args)
                except Exception as error:  # noqa: BLE001  upstream skips the seed on any setup error
                    upstream.safe_close_env(task_env)
                    verdict["rejected"] = f"evaluation setup: {type(error).__name__}: {error}"[:300]
                else:
                    accepted += 1  # upstream evaluates the episode here, then closes as follows
                    task_env.close_env(clear_cache=((accepted + 1) % args["clear_cache_freq"] == 0))
        except UnStableError:
            upstream.safe_close_env(task_env)
            verdict["rejected"] = "unstable"
        except Exception as error:  # noqa: BLE001  upstream skips the seed on any expert-check error
            upstream.safe_close_env(task_env)
            verdict["rejected"] = f"expert check: {type(error).__name__}: {error}"[:300]
        print(MARK + json.dumps(dict(verdict, accepted="rejected" not in verdict,
                                     seconds=round(time.monotonic() - started, 1))), flush=True)
        seed += 1


def write(path, table):
    """Replace the table atomically, so an interrupted screen leaves the previous one."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(table, indent=1) + "\n")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=10, help="seeds each task keeps (default: 10)")
    parser.add_argument("--tasks", nargs="+", help="native task names (default: every cataloged task)")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices (CUDA_VISIBLE_DEVICES of each task)")
    parser.add_argument("--per-gpu", type=int, default=1, help="tasks screened at once per GPU")
    parser.add_argument("--max-candidates", type=int, default=200, help="seeds a task may try")
    parser.add_argument("--output", type=Path, default=SEEDS, help=f"the table (default: {SEEDS.relative_to(ROOT)})")
    parser.add_argument("--one", nargs=4, metavar=("TASK", "SEED", "ACCEPTED", "COUNT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:  # a worker of this tool
        task, *numbers = args.one
        return one(task, *map(int, numbers))
    import curobo  # noqa: F401  run install-curobo first: upstream's evaluation plans with cuRobo
    revision = manifest()["source_revision"]
    export(revision)
    upstream_edits()
    catalog = tomllib.loads((ROOT / "simulators/robotwin/tasks.toml").read_text())["tasks"]
    names = args.tasks or list(dict.fromkeys(task["native_task"] for task in catalog))
    first = 100000 * (1 + EVAL_SEED)
    facts = {"source_revision": revision, "eval_seed": EVAL_SEED, "planner": "curobo v0.7.8"}
    table = json.loads(args.output.read_text()) if args.output.is_file() else {**facts, "tasks": {}}
    for key, value in facts.items():
        if table.get(key) != value:
            parser.error(f"{args.output} was screened with {key} {table.get(key)!r}, not {value!r}; "
                         "move it away to screen afresh")
    lock, log, incomplete = threading.Lock(), args.output.with_name(args.output.name + ".log.jsonl"), []
    pending = [task for task in names
               if len(table["tasks"].setdefault(task, {"seeds": [], "next": first})["seeds"]) < args.count]
    slots = [gpu for gpu in args.gpus.split(",") for _ in range(args.per_gpu)]

    def screen(task, gpu):
        entry = table["tasks"][task]
        command = [sys.executable, __file__, "--one", task, str(entry["next"]), str(len(entry["seeds"])),
                   str(args.count)]
        errors = args.output.with_name(f"{args.output.name}.{task}.stderr")
        with errors.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=stream, text=True,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
        for line in process.stdout:
            if not line.startswith(MARK):
                continue
            verdict = json.loads(line[len(MARK):])
            with lock:
                if verdict["accepted"]:
                    entry["seeds"].append(verdict["seed"])
                entry["next"] = verdict["seed"] + 1
                write(args.output, table)
                with log.open("a") as stream:
                    stream.write(json.dumps(dict(verdict, task=task, gpu=gpu)) + "\n")
            print(f"{task} {verdict['seed']}: {'accepted' if verdict['accepted'] else verdict['rejected']} "
                  f"({len(entry['seeds'])}/{args.count}, {verdict['seconds']} s)", flush=True)
            if entry["next"] - first >= args.max_candidates and len(entry["seeds"]) < args.count:
                process.kill()
                break
        process.wait()
        if len(entry["seeds"]) < args.count:
            incomplete.append(task)

    def worker(gpu):
        while True:
            with lock:
                if not pending:
                    return
                task = pending.pop(0)
            screen(task, gpu)
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if incomplete:
        print(f"incomplete: {', '.join(sorted(incomplete))} (rerun to resume)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
