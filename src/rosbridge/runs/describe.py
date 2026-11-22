"""`rosbridge describe`: an episode's task description, read from its backend in the simulator's own environment.

The backend opens the episode once, as the world node opens a run's (rosbridge.core.episode), in a disposable run
folder, so describing takes as long as a launch's start and needs the same preparation.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rosbridge.catalog import check_configuration, check_scene, resolve
from rosbridge.provisioning.environment import overlay_command, run_activated
from rosbridge.provisioning.setup import doctor, workspace_lock
from rosbridge.storage import write_json


def describe(selection, *, seed=None, project=None, configuration=None, scene=None):
    """The task description of the episode a launch of the task, seed, configuration and scene options runs. The
    seed defaults as a launch's does."""
    root, config, task = resolve(selection, project)
    options = {"task": task["native_task"], "seed": seed if seed is not None else task.get("seed", 0),
               "configuration": check_configuration(config, configuration),
               "scene": check_scene(config, scene or {})}
    report = doctor(task["id"], project=root)
    if not report["ready"]:
        raise ValueError("; ".join(report["issues"] + report["next_steps"]))
    parent = root / ".cache/rosbridge/describe"
    parent.mkdir(parents=True, exist_ok=True)
    with workspace_lock(root, config["name"], shared=True), \
            tempfile.TemporaryDirectory(prefix=f"{config['name']}-", dir=parent) as temporary:
        folder = Path(temporary)
        python, env = run_activated(root, config, folder)
        output, log = folder / "description.json", folder / "describe.log"
        command = [python, "-m", "rosbridge.runs.describe", output, config["name"], json.dumps(options)]
        with log.open("w") as stream:
            code = subprocess.run(overlay_command(root, config, command), cwd=root, env=env,
                                  stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
        reply = json.loads(output.read_text()) if output.is_file() else {}
        if "error" in reply:
            raise ValueError(reply["error"])
        if code or "description" not in reply:
            raise RuntimeError(f"{task['id']} could not be opened to describe it (exit {code}):\n"
                               + log.read_text(errors="replace")[-4000:])
        return reply["description"]


def main():
    from rosbridge.core.episode import open_episode, task_description
    output, simulator, options = sys.argv[1:]
    options = json.loads(options)
    try:
        sim = open_episode(simulator, options["task"], options["seed"], configuration=options["configuration"],
                           scene=options["scene"])
    except ValueError as error:
        # A refusal the user acts on, such as a seed out of range or a scene option's value.
        write_json(output, {"error": str(error)})
        raise SystemExit(1)
    try:
        write_json(output, {"description": task_description(options["task"], sim)})
    finally:
        sim.close()


if __name__ == "__main__":
    main()
