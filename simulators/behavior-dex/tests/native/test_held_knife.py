"""The slicing rule of patches/behavior/0005 with a knife that a Sharpa hand holds by its fingers alone
(simulators/behavior/tests/native/test_slicing.py checks the rule itself). Isaac hosts one app per process and
OmniGibson's shutdown ends it, so held_knife.py runs in a process of its own."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_a_knife_held_in_a_sharpa_hand_cuts_at_a_slow_press_and_leaves_the_halves_whole(tmp_path):
    path = tmp_path / "report.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("held_knife.py")), str(path)],
                         capture_output=True, text=True, timeout=2400)
    errors = [line for line in (run.stdout + run.stderr).splitlines() if "[py stderr]" in line]  # as Kit logs them
    assert path.is_file(), "\n".join(errors[-40:]) or run.stderr[-4000:]
    report = json.loads(path.read_text())
    grip, press = report["grip"], report["press"]
    # Fingers alone held the knife, edge down, for 1.5 s and through the press
    assert grip["slip mm"] < 30 and grip["turn deg"] < 25 and grip["edge down deg"] < 30, grip
    assert len(grip["links"]) >= 2, grip
    assert press["slip mm"] < 60, press
    # The hand came down at 0.02 m/s and the blade cut the egg
    assert press["active before"] and press["whole before"] and press["cut"], press
    # The blade went on down between the halves and stayed 2.5 s: inactive while it touched them, active again no
    # sooner than 2 s after it last did, and the halves whole
    assert press["seconds after the cut"] > 2.5 and press["seconds touching the halves"] > .2, press
    assert not press["active while touching"] and press["steps from the last touch to active"] in (None, 60, 61), press
    assert press["halves"] == 2 and press["q_score"] == pytest.approx(.4), press
