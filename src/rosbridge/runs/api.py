"""CLI-facing run operations; native libraries remain in the selected environment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from rosbridge.catalog import resolve, simulator
from rosbridge.project import project_root
from rosbridge.recording import destination, reserve
from rosbridge.storage import run_environment, write_json
from rosbridge.provisioning.environment import activated, overlay_command
from rosbridge.provisioning.setup import doctor
from .records import SCHEMA, alive, directory, read, select, wait_stopped
from .processes import signal_owner

SPECTATOR_RATE_HZ = 10.


def launch(selection, *, project=None, options=None, demo=False):
    options = options or {}
    root, config, task = resolve(selection, project)
    report = doctor(task["id"], project=root)
    if not report["ready"]:
        raise ValueError("; ".join(report["issues"] + report["next_steps"]))
    recording_dir = destination(options["record"]) if options.get("record") is not None else None
    recording_type = options.get("record_type") or "mp4"
    mode = options.get("execution_mode") or ("stepped" if demo else None)
    arguments = [f"backend:={config['name']}", f"task:={task['native_task']}"]
    # A catalog task may name a default seed, e.g. a scene that is stable from seed 0.
    seed = options.get("seed") if options.get("seed") is not None else task.get("seed")
    for name, value in (("mode", mode), ("seed", seed),
                        ("terminate_on_success", options.get("stop_on_success"))):
        if value is not None:
            arguments.append(f"{name}:={str(value).lower() if isinstance(value, bool) else value}")
    if recording_dir is not None:
        arguments.append(f"spectator_rate_hz:={SPECTATOR_RATE_HZ}")
    if demo:
        arguments.append(f"demo:={config['ros_package']}")
    run_id = uuid.uuid4().hex[:12]
    folder = directory(root) / run_id
    folder.mkdir(parents=True, mode=0o700)
    record = {"schema": SCHEMA, "id": run_id, "uid": os.getuid(), "project": str(root),
              "simulator": config["name"], "environment": config["environment"],
              "task": task["id"], "native_task": task["native_task"], "state": "starting",
              "requested_domain": options.get("domain_id"), "timeout": options.get("timeout", 900.),
              "demo": demo, "launch_arguments": arguments, "created_at": time.time(),
              "recording_dir": str(recording_dir) if recording_dir is not None else None,
              "recording_type": recording_type if recording_dir is not None else None,
              "log": str(folder / "graph.log"), "run_file": str(folder / "run.json")}
    path = folder / "run.json"
    supervisor_env = dict(os.environ, **run_environment(folder))
    if recording_dir is not None:
        reserve(recording_dir, run_id, task["id"])
    write_json(path, record)
    with (folder / "supervisor.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-m", "rosbridge.runs.supervisor", str(path)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=supervisor_env)
    try:
        deadline = time.monotonic() + record["timeout"] + 15.
        while time.monotonic() < deadline:
            current = read(path)
            if current.get("ready_at"):
                return current, process
            if process.poll() is not None or current["state"] == "failed":
                raise RuntimeError(current.get("error", "Supervisor failed") + f"; logs: {folder}")
            time.sleep(.05)
        raise TimeoutError(f"Supervisor startup timed out; logs: {folder}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=45)
        raise


def probe_command(python, timeout):
    return [str(python), "-m", "rosbridge.runs.probe", "--timeout", str(timeout)]


def _settled(record):
    """A dead supervisor's unfinished state is lost."""
    state = "lost" if record["state"] in ("starting", "ready", "stopping") else record["state"]
    return dict(record, state=state, running=False, ready=False)


def _joined(record):
    """The run's simulator environment and ROS domain."""
    root = Path(record["project"])
    _, _, config = simulator(record["simulator"], root)
    python, env = activated(root, config)
    env.update(record["ros_environment"])
    return root, config, python, env


def status(run=None, *, project=None):
    _, record = select(project_root(project), run)
    live = alive(record)
    result = dict(record, running=True, ready=False) if live else _settled(record)
    if record.get("recording_dir"):
        metadata = Path(record["recording_dir"]) / "recording.json"
        if metadata.is_file():
            result["recording"] = json.loads(metadata.read_text())
    if not live or record["state"] != "ready":
        return result
    root, config, python, env = _joined(record)
    try:
        probe = subprocess.run(overlay_command(root, config, probe_command(python, 3)),
                               env=env, capture_output=True, text=True, timeout=6)
        result["ready"] = probe.returncode == 0
        if result["ready"]:
            result["time_s"] = json.loads(probe.stdout)["time_s"]
    except subprocess.TimeoutExpired:
        pass
    return result


def stop(run=None, *, project=None):
    path, record = select(project_root(project), run)
    if not alive(record):
        return _settled(record)
    signal_owner(record["owner"], signal.SIGTERM)
    return dict(wait_stopped(path), running=False, ready=False)


def execute(run, command, *, project=None):
    _, record = select(project_root(project), run)
    if not alive(record) or record["state"] != "ready":
        raise ValueError("Run is not active")
    root, config, _, env = _joined(record)
    # Preserve the caller's cwd: controller paths belong to their workspace.
    return subprocess.call(overlay_command(root, config, command), env=env)


def follow(record, process):
    """Stream retained graph output in the foreground; Ctrl-C closes the graph."""
    path = Path(record["log"])
    try:
        with path.open() as log:
            while process.poll() is None:
                output = log.read()
                if output:
                    print(output, end="", file=sys.stderr, flush=True)
                time.sleep(.05)
            print(log.read(), end="", file=sys.stderr, flush=True)
        return process.returncode
    except KeyboardInterrupt:
        stop(record["run_file"], project=record["project"])
        process.wait(timeout=5)
        return 130


def judge(run=None, *, project=None):
    """Read native outcomes without connecting to the agent's ROS graph."""
    root = project_root(project)
    path, record = select(root, run)
    results = path.parent / "episode-results.jsonl"
    outcomes = [json.loads(line) for line in results.read_text().splitlines() if line.strip()] if results.exists() else []
    return {"id": record["id"], "state": "judged" if outcomes else "pending", "episodes": outcomes}
