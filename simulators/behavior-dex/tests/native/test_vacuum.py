"""OmniGibson's particle remover by projection as patches/behavior-dex/0002 makes it for the R1Pro with Sharpa hands:
it takes a visual particle whose centre lies within 1 cm of its volume. OmniGibson hangs that volume from its meta
link, so the vacuum's box (30 x 12 x 2 cm) ends 1.1 mm below the floor it stands on and unpatched it removes floor
dust only while lifted 1.2 to 21.5 mm. The box follows the vacuum on steps that do not render by
patches/behavior/0005. Isaac hosts one app per process and OmniGibson's shutdown ends it, so vacuum_floor.py runs in a
process of its own."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def vacuum(tmp_path_factory):
    path = tmp_path_factory.mktemp("vacuum") / "report.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("vacuum_floor.py")), str(path)],
                         capture_output=True, text=True, timeout=1200)
    errors = [line for line in (run.stdout + run.stderr).splitlines() if "[py stderr]" in line]  # as Kit logs them
    assert path.is_file(), "\n".join(errors[-40:]) or run.stderr[-4000:]
    report = json.loads(path.read_text())
    assert not report["errors"], report["errors"]
    # The vacuum stands upright with its removal box under the floor's surface, where the dust is not
    standing = report["standing"]
    assert standing["tilt deg"] < .5 and standing["box mm"] == pytest.approx([302.2, 119.2, 20.3], abs=.2)
    assert standing["box top above the floor mm"] == pytest.approx(-1.1, abs=.3)
    assert report["particles"] == 22
    return report


def test_a_vacuum_that_is_off_removes_nothing(vacuum):
    assert vacuum["off"]["left"] and vacuum["off"]["tilt deg"] < .5


def test_a_standing_vacuum_removes_the_dust_under_it(vacuum):
    assert not vacuum["on"]["left"]


def test_the_margin_reaches_one_centimetre_beside_the_volume_and_not_through_a_floor(vacuum):
    beside, plate = vacuum["beside"], vacuum["plate"]
    assert beside["box off the station mm"] == pytest.approx([0., 0.], abs=1.)
    # 5 mm beside the box's long side: removed; 15 mm beside it and 15 mm beyond its end: not
    assert (beside["near left"], beside["side left"], beside["end left"]) == (False, True, True)
    # Standing on a 5 cm plate over a particle of the floor: the box ends 2.9 cm above it and the particle stays
    assert plate["box bottom above the floor mm"] == pytest.approx(28.5, abs=1.) and plate["left"]


def test_a_vacuum_pushed_along_the_floor_removes_the_dust_it_passes(vacuum):
    pushed = vacuum["pushed"]
    # A block pushed it 0.87 m in 4.5 s; it stayed standing, on the floor and on its course
    assert pushed["travel"] > .8 and pushed["tilt deg"] < 2. and pushed["lift mm"] < 2., pushed
    assert abs(pushed["drift"]) < .02 and abs(pushed["yaw"]) < 3., pushed
    # Each of the lane's 8 particles went at the step that brought it inside the box and its margin, which the
    # remover read where the vacuum was; the 2 particles 3 cm beside the lane stayed
    assert (pushed["particles"], pushed["left"], pushed["late"], pushed["outside"]) == (8, 0, 0, 2), pushed
    assert len(pushed["removed at"]) == 8 and pushed["lag mm"] < 1., pushed


def test_a_lifted_vacuum_removes_floor_dust_up_to_three_centimetres(vacuum):
    lifted = vacuum["lifted"]
    # Unpatched it removes only between 1.2 and 21.5 mm of lift, where the box holds the dust's centres
    assert [lifted[f"{lift} mm"]["left"] for lift in (1, 2, 20, 23, 30, 33)] == [False] * 5 + [True], lifted
    assert lifted["33 mm"]["box bottom above the floor mm"] == pytest.approx(11.5, abs=.5)
    # Of 22 particles: the lane's 8, the one 5 mm beside the box, the one under it and 5 under the lifted vacuum went
    assert vacuum["left"] == 7
