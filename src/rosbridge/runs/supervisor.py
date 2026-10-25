"""Own one launch graph and its domain lease, independent of the invoking terminal."""
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rosbridge.catalog import simulator
from rosbridge.provisioning.environment import activated, native_activated, overlay_command
from rosbridge.provisioning.setup import workspace_lock
from rosbridge.storage import run_environment, write_json
from .api import probe_command
from .domains import DomainLease
from .records import identity, read
from .processes import parent_death_signal


def terminate(process, timeout=8):
    if process is None or process.poll() is not None:
        return
    def send(sig, *, group=False):
        try:
            if group:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass
    # ROS launch forwards SIGINT to its nodes. Signalling the whole group here
    # would interrupt node teardown a second time.
    send(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        send(signal.SIGTERM, group=True)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL, group=True)
            process.wait()


def main():
    path = Path(sys.argv[1])
    record = read(path)
    record["owner"] = identity(os.getpid())
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    write_json(path, record)
    lease = graph = probe = recorder = None
    locks = ExitStack()
    exit_code = 1
    try:
        root, _, config = simulator(record["simulator"], record["project"])
        preparation = locks.enter_context(workspace_lock(root, config["name"], shared=True))
        lease = DomainLease(record.get("requested_domain"))
        locks.callback(lease.close)
        record["domain_id"] = lease.domain
        python, env = activated(root, config)
        env.update(run_environment(path.parent))
        env["ROS_DOMAIN_ID"] = str(lease.domain)
        if config.get("native_environment"):
            # The world node proxies a backend that runs in the simulator's own interpreter.
            native_python, native_env = native_activated(root, config)
            native_env.update(run_environment(path.parent))
            native_env["ROS_DOMAIN_ID"] = env["ROS_DOMAIN_ID"]
            write_json(path.parent / "native-environment.json",
                       {"python": str(native_python), "env": native_env})
            env["ROSBRIDGE_NATIVE_ENVIRONMENT"] = str(path.parent / "native-environment.json")
        env.pop("ROSBRIDGE_RECORDING_READY", None)
        if record.get("recording_dir"):
            env["ROSBRIDGE_RECORDING_READY"] = str(path.parent / "recording.ready")
        record["ros_environment"] = {key: env[key] for key in
                                     ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "ROS_HOME",
                                      "ROS_LOG_DIR", "TMPDIR") if key in env}
        command = ["ros2", "launch", "rosbridge_runtime", "demo.launch.py" if record["demo"] else "sim.launch.py",
                   *record["launch_arguments"]]
        with (path.parent / "graph.log").open("a") as log:
            if stop[0]:
                exit_code = 0
                return
            parent_pid = os.getpid()
            if record.get("recording_dir"):
                recorder_log = locks.enter_context((path.parent / "recorder.log").open("a"))
                recorder = subprocess.Popen(
                    overlay_command(root, config, [python, "-m", "rosbridge.recording.node", str(path)]),
                    cwd=root, env=env, stdout=recorder_log, stderr=subprocess.STDOUT,
                    start_new_session=True, preexec_fn=lambda: parent_death_signal(parent_pid),
                    pass_fds=(preparation.fileno(),))
                record["recorder"] = identity(recorder.pid)
                record["recorder_log"] = str(path.parent / "recorder.log")
                write_json(path, record)
            graph = subprocess.Popen(overlay_command(root, config, command), cwd=root, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     preexec_fn=lambda: parent_death_signal(parent_pid),
                                     pass_fds=(lease.handle.fileno(), preparation.fileno()))
            record["graph"] = identity(graph.pid)
            write_json(path, record)
            probe = subprocess.Popen(overlay_command(root, config, probe_command(python, record["timeout"])),
                                     cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=log,
                                     start_new_session=True)
            recording_ready = path.parent / "recording.ready"
            deadline = time.monotonic() + record["timeout"]
            while not stop[0]:
                if recorder is not None and recorder.poll() is not None:
                    raise RuntimeError(f"Recorder exited unexpectedly (exit {recorder.returncode}); see recorder.log")
                if record["state"] == "starting" and probe.poll() is not None:
                    if probe.returncode:
                        raise RuntimeError("ROS readiness check failed; see graph.log")
                    # A recording run is ready once the recorder holds every initial frame.
                    if not record.get("recording_dir") or recording_ready.is_file():
                        record["state"] = "ready"
                        record["ready_at"] = time.time()
                        write_json(path, record)
                code = graph.poll()
                if code is not None:
                    if record["demo"] and code == 0 and record.get("ready_at"):
                        exit_code = 0
                        break
                    raise RuntimeError(f"Required graph exited unexpectedly (exit {code}); see graph.log")
                if record["state"] == "starting" and time.monotonic() >= deadline:
                    raise TimeoutError("Launch readiness timeout; see graph.log")
                time.sleep(.05)
            if stop[0]:
                exit_code = 0
                record["state"] = "stopping"
                write_json(path, record)
    except BaseException as error:
        record["error"] = str(error) or type(error).__name__
        print(record["error"], file=sys.stderr, flush=True)
    finally:
        def cleanup(action, *args, **kwargs):
            nonlocal exit_code
            try:
                action(*args, **kwargs)
            except Exception as error:
                exit_code = 1
                record.setdefault("error", str(error) or type(error).__name__)
                print(f"Cleanup failed: {error}", file=sys.stderr, flush=True)

        cleanup(terminate, probe)
        cleanup(terminate, graph)
        cleanup(terminate, recorder, timeout=15)
        if record.get("recording_dir"):
            from rosbridge.recording import finalize
            cleanup(finalize, record, recorder.returncode if recorder is not None else None)
        cleanup(locks.close)
        if record.get("error"):
            exit_code = 1
        record["state"] = "stopped" if exit_code == 0 else "failed"
        record["exit_code"] = exit_code
        record["ended_at"] = time.time()
        cleanup(write_json, path, record)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
