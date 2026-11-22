"""`rosbridge setup` in the simulator's own environment, where its workspace hook and backend run.

The workspace export (rosbridge.catalog.robots.export) runs there with the episode's task description: the
backend opens the episode once, as the world node will open it (rosbridge.core.episode), in a disposable run
folder, and its description goes to the workspace's .kinex/task_description.json. The simulator boots and
resets, so setup takes as long as a launch's start and needs the same preparation.
"""
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rosbridge.catalog import check_configuration, check_scene, resolve
from rosbridge.catalog.robots import export
from rosbridge.provisioning.environment import overlay_command, run_activated
from rosbridge.provisioning.setup import doctor, workspace_lock
from rosbridge.storage import write_json


def setup(selection, directory=None, *, seed=None, project=None, domain_id=0, configuration=None, scene=None):
    root, config, task = resolve(selection, project)
    check_configuration(config, configuration)
    options = {"selection": task["id"], "directory": str(Path(directory or Path.cwd()).resolve()), "seed": seed,
               "domain_id": domain_id, "configuration": configuration, "scene": check_scene(config, scene or {})}
    report = doctor(task["id"], project=root)
    if not report["ready"]:
        raise ValueError("; ".join(report["issues"] + report["next_steps"]))
    parent = root / ".cache/rosbridge/setup"
    parent.mkdir(parents=True, exist_ok=True)
    with workspace_lock(root, config["name"], shared=True), \
            tempfile.TemporaryDirectory(prefix=f"{config['name']}-", dir=parent) as temporary:
        folder = Path(temporary)
        python, env = run_activated(root, config, folder)
        output, log = folder / "setup.json", folder / "setup.log"
        command = [python, "-m", "rosbridge.runs.setup", output, json.dumps(options)]
        with log.open("w") as stream:
            code = subprocess.run(overlay_command(root, config, command), cwd=root, env=env,
                                  stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
        reply = json.loads(output.read_text()) if output.is_file() else {}
        if "error" in reply:
            raise ValueError(reply["error"])
        if code or "result" not in reply:
            raise RuntimeError(f"setup of {task['id']} failed in its environment (exit {code}):\n"
                               + log.read_text(errors="replace")[-4000:])
        return reply["result"]


def described(root, config, task, seed, *, configuration, scene):
    """The episode's task description, from its backend opened here."""
    from rosbridge.core.episode import open_episode, task_description
    sim = open_episode(config["name"], task["native_task"], seed, configuration=configuration, scene=scene)
    try:
        return task_description(task["native_task"], sim)
    finally:
        sim.close()


def main():
    output, options = sys.argv[1:]
    options = json.loads(options)
    describe = partial(described, configuration=options["configuration"], scene=options.pop("scene"))
    try:
        result = export(options.pop("selection"), options.pop("directory"), describe=describe, **options)
    except ValueError as error:
        # A refusal the user acts on, such as an existing workspace or a seed out of range.
        write_json(output, {"error": str(error)})
        raise SystemExit(1)
    write_json(output, {"result": result})


if __name__ == "__main__":
    main()
