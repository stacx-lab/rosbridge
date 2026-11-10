"""A broken attachment joint detaches the object and the simulation goes on (patches/behavior/0006). OmniGibson ran
the break's handler, which deletes the joint's prim, inside Kit's physics step callback, and on a step that does not
render, as rosbridge steps an episode, Kit aborted the process there. Isaac hosts one app per process and
OmniGibson's shutdown ends it, so joint_break.py runs the break in a process of its own."""
import json
import subprocess
import sys
from pathlib import Path


def test_a_broken_attachment_joint_detaches_and_the_simulation_goes_on(tmp_path):
    path = tmp_path / "report.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("joint_break.py")), str(path)],
                         capture_output=True, text=True, timeout=1200)
    errors = [line for line in (run.stdout + run.stderr).splitlines() if "[py stderr]" in line]  # as Kit logs them
    assert path.is_file(), "\n".join(errors[-40:]) or run.stderr[-4000:]
    report, end = json.loads(path.read_text()), (run.returncode, "\n".join(errors[-40:]) or run.stderr[-4000:])
    assert report["attached"] == {"attached": True, "joint": True}
    # The pull broke the joint and carried the shelf off; the break detached it and removed the joint's prim
    assert report.get("broken") == {"attached": False, "joint": False, "child": False, "rose": True}, end
    # The process lived: 30 more steps ran, and the shelf, put back in its slot, attached again
    assert report.get("later") == {"steps": 30, "attached": False}, end
    assert report.get("again") == {"attached": True, "joint": True} and run.returncode == 0, end
