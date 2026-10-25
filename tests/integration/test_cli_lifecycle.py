"""End-to-end CLI checks. Requires bootstrap and `rosbridge prepare abc`."""
import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

from rosbridge.runs.records import alive, directory, read

from tests.integration.support import CLI, ROOT, invoke, start


def test_two_domains_exec_and_explicit_conflict(runs):
    first = start(runs)
    second = start(runs)
    assert first["domain_id"] != second["domain_id"]
    result = invoke("launch", "abc", "--domain-id", first["domain_id"], "--detach", "--json", expected=1)
    assert "already managed" in json.loads(result.stdout)["error"]
    # The lightweight CLI activates ROS only in the child command. Shell syntax
    # in arguments must arrive verbatim, and a controller's exit code survives.
    result = invoke("exec", first["id"], "--", "python", "-c",
                    "import os,sys,json; print(json.dumps([os.environ['ROS_DOMAIN_ID'],sys.argv[1]])); sys.exit(7)",
                    "space and $(literal)", expected=7)
    assert json.loads(result.stdout) == [str(first["domain_id"]), "space and $(literal)"]
    controller = '''import json, rclpy
from rclpy.node import Node
from rosbridge.clients import WorldClockClient
rclpy.init()
node = Node("cli_control_test")
result = WorldClockClient(node).wait(.102, timeout_s=10.)
print(json.dumps({"success":result.success,"waited_s":result.waited_s}))
node.destroy_node()
rclpy.shutdown()
'''
    result = invoke("exec", first["id"], "--", "python", "-c", controller)
    assert json.loads(result.stdout)["waited_s"] == pytest.approx(.102)
    status1 = json.loads(invoke("status", first["id"], "--json").stdout)
    status2 = json.loads(invoke("status", second["id"], "--json").stdout)
    assert status1["ready"] and status2["ready"]
    assert status1["time_s"] == pytest.approx(.102) and status2["time_s"] == 0.
    # The default is deliberately ambiguous while two local runs are live.
    assert "live runs" in invoke("status", "--json", expected=1).stdout
    result = json.loads(invoke("stop", first["id"], "--json").stdout)
    assert result["state"] == "stopped" and not result["running"]
    assert json.loads(invoke("status", second["id"], "--json").stdout)["ready"]


def test_startup_timeout_cleans_graph():
    before = set(directory(ROOT).glob("*/run.json"))
    result = invoke("launch", "abc", "--timeout", ".01", "--detach", "--json", expected=1)
    assert "timeout" in json.loads(result.stdout)["error"].lower() or "readiness" in result.stdout
    after = set(directory(ROOT).glob("*/run.json")) - before
    assert len(after) == 1
    record = read(after.pop())
    assert record["state"] == "failed" and not alive(record)
    assert not alive({"owner": record.get("graph")})


def test_foreground_ctrl_c_stops_graph(tmp_path):
    with (tmp_path / "output").open("w+") as output, (tmp_path / "log").open("w") as log:
        process = subprocess.Popen([*CLI, "launch", "abc", "--execution-mode", "stepped", "--json",
                                    "--record", str(tmp_path / "recording")],
                                   cwd=ROOT, stdout=output, stderr=log, start_new_session=True)
        record = None
        try:
            import time
            deadline = time.monotonic() + 60.
            while time.monotonic() < deadline:
                output.seek(0)
                line = output.readline()
                if line:
                    record = json.loads(line)
                    break
                assert process.poll() is None
                time.sleep(.1)
            assert record is not None
            os.kill(process.pid, signal.SIGINT)
            assert process.wait(timeout=25) == 130
            final = read(Path(record["run_file"]))
            assert final["state"] == "stopped" and not alive(final)
            assert not alive({"owner": final["graph"]})
            assert final["recording_state"] == "complete"
            assert (tmp_path / "recording/third_person.mp4").exists()
        finally:
            if record and alive(read(Path(record["run_file"]))):
                invoke("stop", record["id"], "--json")
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=25)


def test_invalid_graph_reports_failure_and_cleans_up():
    before = set(directory(ROOT).glob("*/run.json"))
    # A seed beyond ROS's int64 parameter range makes the world fail to start.
    result = invoke("launch", "abc", "--seed", str(2 ** 64), "--detach", "--json", expected=1)
    assert "graph exited" in json.loads(result.stdout)["error"]
    path, = set(directory(ROOT).glob("*/run.json")) - before
    final = read(path)
    assert final["state"] == "failed" and not alive(final)
    assert not alive({"owner": final["graph"]})


def test_supervisor_loss_terminates_graph(runs):
    import time
    record = start(runs)
    os.kill(record["owner"]["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 15.
    while time.monotonic() < deadline and alive({"owner": record["graph"]}):
        time.sleep(.1)
    assert not alive({"owner": record["graph"]})
    result = json.loads(invoke("status", record["id"], "--json").stdout)
    assert result["state"] == "lost" and not result["running"]


def test_launch_arguments_reach_the_world_node(runs):
    record = json.loads(invoke("launch", "abc", "--execution-mode", "stepped", "--seed", "21",
                               "--detach", "--json").stdout)
    runs.append(record)
    code = '''import json, rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
rclpy.init()
node=Node("parameters_probe")
client=node.create_client(GetParameters,"/world/get_parameters")
assert client.wait_for_service(timeout_sec=5.)
future=client.call_async(GetParameters.Request(names=["mode","seed","spectator_rate_hz"]))
rclpy.spin_until_future_complete(node,future,timeout_sec=5.)
values=future.result().values
print(json.dumps([values[0].string_value,values[1].integer_value,values[2].double_value]))
node.destroy_node()
rclpy.shutdown()
'''
    result = invoke("exec", record["id"], "--", "python", "-c", code)
    assert json.loads(result.stdout) == ["stepped", 21, 0.]
    assert json.loads(invoke("status", record["id"], "--json").stdout)["time_s"] == 0.
    error = json.loads(invoke("prepare", "abc", "--json", expected=1).stdout)["error"]
    assert "stop its runs" in error
