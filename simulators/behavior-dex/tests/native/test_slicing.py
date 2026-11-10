"""OmniGibson's slicing and dicing rules as patches/behavior-dex/0001 makes them for the R1Pro with Sharpa hands: a
sliceable or diceable object is cut by an active slicer that touches it and whose blade, the box of its slicer meta
link grown by 5 mm, reaches it; the slicer that cut becomes inactive, and active again 2 s after it last touched whole
or cut food. Unpatched, any part of any slicer cuts while any slicer of the scene is active, and a slicer is active
again 2 s after it last touched whole food, lying on the halves it cut or not. Isaac hosts one app per process and
OmniGibson's shutdown ends it, so slicer_cuts.py and held_knife.py each run in a process of their own."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = ("axe", "carver", "knife")  # on a log, a hard-boiled egg and a cupcake
SPEEDS = (.02, .3, 1.)  # m/s


def scenario(script, path):
    run = subprocess.run([sys.executable, str(Path(__file__).with_name(script)), str(path)],
                         capture_output=True, text=True, timeout=2400)
    errors = [line for line in (run.stdout + run.stderr).splitlines() if "[py stderr]" in line]  # as Kit logs them
    assert path.is_file(), "\n".join(errors[-40:]) or run.stderr[-4000:]
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def cuts(tmp_path_factory):
    report = scenario("slicer_cuts.py", tmp_path_factory.mktemp("slicer") / "report.json")
    assert not report["errors"], report["errors"]
    assert set(report["start"]["active"].values()) == {True}  # every slicer starts active
    return report


def test_a_slicers_handle_does_not_cut(cuts):
    for tool in TOOLS:
        handle = cuts[f"{tool} handle"]
        # The handle came down on the food and pressed on it for 20 steps: the food is whole, the slicer still active
        assert handle["touched at"] is not None and not handle["cut"] and handle["halves"] == 0, (tool, handle)
        assert handle["active before"] and handle["active after"], (tool, handle)
        assert cuts["blades"][tool]["blade_from_handle_end"] > .1  # the blade starts well away from where it touched


def test_a_blade_cuts_at_any_speed_down_to_a_slow_press(cuts):
    for tool in TOOLS:
        for speed in SPEEDS:
            blade = cuts[f"{tool} blade {speed}"]
            assert blade["active before"] and blade["cut"] and blade["halves"] == 2, (tool, blade)
            # The slicer that cut is inactive, and active again 2 s after the cut that removed what it touched
            assert not blade["active after"] and 58 <= blade["rearm steps"] <= 61, (tool, blade)


def test_a_slicer_that_cut_does_not_cut_while_another_is_active(cuts):
    first, again = cuts["second: first cupcake"], cuts["second: disarmed knife"]
    assert first["cut"] and not first["active after"]
    # Within its 2 s the same knife pressed its blade on the next cupcake for 20 steps while the other knife floated
    # active: nothing was cut. The other knife's blade then cut it
    assert not again["active before"] and again["other active"] and not again["cut"], again
    other = cuts["second: active knife"]
    assert again["touched at"] is not None and other["active before"] and other["cut"] and other["halves"] == 2, other


def test_a_slicer_lying_on_the_halves_it_cut_stays_inactive_and_leaves_them_whole(cuts):
    resting = cuts["resting"]
    # The knife lay on the halves for 5 s, touching them for most of it and never out of touch for its 2 s
    assert resting["seconds touching"] > 2.5 and resting["longest gap s"] < 1.5, resting
    assert not resting["active while lying"] and (resting["halves"], resting["diced"]) == (2, 0), resting


def test_a_slicer_is_active_again_two_seconds_after_it_is_clear(cuts):
    resting = cuts["resting"]
    # Lifted off the halves, the knife was active again 2 s after it last touched one (the delay sums 60 float32
    # steps), and the halves were still whole
    assert resting["steps from the last touch to active"] in (60, 61) and resting["halves after"] == 2, resting


def test_an_articulated_slicers_blade_cuts_where_its_link_has_moved_to(cuts):
    pruner = cuts["pruner"]
    # The open pruner's blade box, even grown by the rule's 5 mm, did not reach the stalk; the blade link closed on
    # it by its joint's velocity and cut it where the box had followed the link to
    assert not pruner["cut while set"] and pruner["active before"] and pruner["open box to stalk mm"] > 6., pruner
    assert pruner["cut"] and pruner["halves"] == 2 and not pruner["active after"], pruner
    assert pruner["box moved mm"] > 15., pruner
    assert pruner["joint"] < pruner["opened"] - .3


def test_a_slicer_without_a_blade_meta_link_cuts_with_its_whole_body(cuts):
    plain = cuts["plain"]
    assert plain["active before"] and plain["cut"] and plain["halves"] == 2 and not plain["active after"], plain


def test_a_knife_held_in_a_sharpa_hand_cuts_at_a_slow_press_and_leaves_the_halves_whole(tmp_path):
    report = scenario("held_knife.py", tmp_path / "report.json")
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
