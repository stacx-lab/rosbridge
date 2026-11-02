"""The values that decide a worker's outcomes are stated in the texts it reads (g1_rules)."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest
import yaml

from rosbridge_behavior import g1_rules
from rosbridge_behavior.g1_rules import assisted_note, physical_note, rules

PROJECT = Path(__file__).resolve().parents[4]
PROFILES = PROJECT / "catalog/robots/behavior-g1/behavior-g1-sharpa/devices"
RUNTIME = PROJECT / ".runtime/behavior-g1"
MEASUREMENTS = PROJECT / "simulators/behavior-g1/measurements"


def words(text):
    return re.sub(r"\s+", " ", text).strip()


def text(place):
    device, *field = place.split(".")
    if device == "instruction":
        return {"assisted": assisted_note(), "physical": physical_note()}[field[0]]
    value = yaml.safe_load((PROFILES / device / "profile.yaml").read_text())
    for key in field:
        value = value[key]
    return value


def missing(entries):
    return [rule for rule in entries if words(rule.phrase) not in words(text(rule.place))]


def test_every_rule_bearing_value_is_stated_in_the_text_the_worker_reads():
    assert missing(rules()) == []


def test_a_value_changed_without_its_text_fails(monkeypatch):
    for name, value in (("VELOCITY_LIMIT", .6), ("FALL_HOLD_S", 2.), ("TOGGLE_HOLD_S", .2), ("WALK_DEAD_ZONE", .05)):
        with monkeypatch.context() as patch:
            patch.setattr(g1_rules, name, value)
            assert name in {rule.name for rule in missing(rules())}
    # The note is rendered from the values, so it follows them.
    monkeypatch.setattr(g1_rules, "GRASP_CLOSURE", .25)
    assert missing(rules()) == [] and "above 25% of their closing range" in words(assisted_note())


def test_the_code_and_the_structured_profiles_act_on_the_same_values():
    from rosbridge_behavior import g1_sharpa
    base = yaml.safe_load((PROFILES / "base/profile.yaml").read_text())
    assert base["velocity"]["axes"] == {axis: {"min": -g1_rules.VELOCITY_LIMIT, "max": g1_rules.VELOCITY_LIMIT}
                                        for axis in ("vx", "vy", "wz")}
    limit = ET.parse(PROFILES / "pelvis/pelvis.urdf").getroot().find("joint[@name='pelvis_height']/limit")
    assert (float(limit.get("lower")), float(limit.get("upper"))) == g1_rules.PELVIS_LIMITS
    pelvis = yaml.safe_load((PROFILES / "pelvis/profile.yaml").read_text())
    assert pelvis["home"] == {"pelvis_height": g1_rules.STANDING_PELVIS}
    for side in ("left", "right"):
        hand = yaml.safe_load((PROFILES / f"{side}_hand/profile.yaml").read_text())
        assert sum("_AA" not in joint for joint in hand["home"]) == g1_rules.FLEXION_JOINTS
    assert (g1_sharpa.GRASP_CLOSURE, g1_sharpa.VELOCITY_LIMIT, g1_sharpa.PELVIS_LIMITS) == (
        g1_rules.GRASP_CLOSURE, g1_rules.VELOCITY_LIMIT, g1_rules.PELVIS_LIMITS)
    assert g1_sharpa.FALL_TILT == pytest.approx(g1_rules.FALL_TILT_DEG * 3.141592653589793 / 180)
    assert G1Sharpa_config()["self_collisions"] is g1_rules.SELF_COLLISIONS


def G1Sharpa_config():
    from rosbridge_behavior.g1_sharpa import G1Sharpa
    embodiment = G1Sharpa.__new__(G1Sharpa)
    from collections import defaultdict
    embodiment.drives = defaultdict(lambda: {"stiffness": 1., "damping": 1.})
    embodiment.hands = {"left": [], "right": []}
    return embodiment.robot_config()


def source(relative):
    path = RUNTIME / relative
    if not path.is_file():
        pytest.skip(f"needs the prepared behavior-g1 runtime ({relative})")
    return path.read_text()


def test_upstream_values_match_their_sources():
    """OmniGibson and GR00T act on these values; the texts state them."""
    robot = source("OmniGibson/omnigibson/robots/robot.py")
    assert float(re.search(r"^m\.GRASP_WINDOW = ([\d.]+)", robot, re.M)[1]) == g1_rules.GRASP_WINDOW_S
    assert float(re.search(r"^m\.ASSIST_GRASP_MASS_THRESHOLD = ([\d.]+)", robot, re.M)[1]) == g1_rules.ASSIST_MASS_KG
    attached = source("OmniGibson/omnigibson/object_states/attached_to.py")
    assert float(re.search(r"^m\.DEFAULT_POSITION_THRESHOLD = ([\d.]+)", attached, re.M)[1]) == g1_rules.ATTACH_DISTANCE_M
    assert float(re.search(r"^m\.DEFAULT_ORIENTATION_THRESHOLD = th\.deg2rad\(th\.tensor\(\[([\d.]+)\]\)\)",
                           attached, re.M)[1]) == g1_rules.ATTACH_ANGLE_DEG
    toggle = source("OmniGibson/omnigibson/object_states/toggle.py")
    assert float(re.search(r"^m\.CAN_TOGGLE_SECONDS = ([\d.]+)", toggle, re.M)[1]) == g1_rules.TOGGLE_HOLD_S
    assert "COLOR_ON = th.tensor([0, 1.0, 0])  # green" in toggle and "COLOR_OFF = th.tensor([1.0, 0, 0])  # red" in toggle
    joints = source("OmniGibson/omnigibson/objects/dataset_object.py")
    assert [float(v) for v in re.findall(r'drive:(?:linear|angular):physics:stiffness"\)\.Set\(([\d.]+)\)', joints)] == [
        g1_rules.FURNITURE_JOINT_STIFFNESS] * 2
    policy = source("third_party/decoupled_wbc/control/policy/g1_gear_wbc_policy.py")
    assert float(re.search(r"switch_cond = np\.linalg\.norm\(np\.concatenate\(\[synced_navigate_cmd\[:2\], "
                           r"\[vyaw_flag\]\]\)\) < ([\d.]+)", policy)[1]) == g1_rules.WALK_DEAD_ZONE
    wbc = yaml.safe_load(source("third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml"))
    assert wbc["height_cmd"] == g1_rules.STANDING_PELVIS


def measured(name):
    files, derive = g1_rules.MEASURED[name]
    return derive(*(json.loads((MEASUREMENTS / file).read_text()) for file in files))


def test_measured_values_are_what_their_stored_measurements_show():
    """A stated measured value must be the one its measurement file gives, not only the constant the text follows."""
    assert {name: measured(name) for name in g1_rules.MEASURED} == {
        name: getattr(g1_rules, name) for name in g1_rules.MEASURED}
    stated = {rule.name for rule in rules()}
    assert set(g1_rules.MEASURED) <= stated  # each is stated somewhere the worker reads


def test_a_measured_value_its_file_does_not_support_fails(monkeypatch):
    # An earlier text said the waist holds its target in a squat; the measurement shows it settles behind it.
    monkeypatch.setattr(g1_rules, "WAIST_SQUAT", {0.: (0., 0.), .3: (.3, .3), .52: (.52, .52)})
    assert measured("WAIST_SQUAT") != g1_rules.WAIST_SQUAT
    assert "WAIST_SQUAT" in {rule.name for rule in missing(rules())}
    monkeypatch.setattr(g1_rules, "FLOOR_WRIST_M", .1)
    assert measured("FLOOR_WRIST_M") != g1_rules.FLOOR_WRIST_M
    # "Leaning adds little reach" is wrong: standing, the settled waist adds 0.1-0.16 m.
    monkeypatch.setattr(g1_rules, "WAIST_REACH_M", {0.: .50, .3: .52, .52: .53})
    assert measured("WAIST_REACH_M") != g1_rules.WAIST_REACH_M
    assert "WAIST_REACH_M" in {rule.name for rule in missing(rules())}


def test_the_depth_text_states_the_profiles_own_intrinsics():
    """Depth is planar z; the unprojection the text gives must use the projection the profile declares."""
    camera = yaml.safe_load((PROFILES / "head_camera/profile.yaml").read_text())
    k = camera["color"]["projection"] if "color" in camera else camera["projection"]
    assert k["fx"] == k["fy"] and k["cx"] == k["cy"]
    said = words(camera["operations"]["depth"]["description"])
    assert f"fx = fy = {k['fx']:g} and cx = cy = {k['cx']:g}" in said
    assert "((u - cx) / fx * z, (v - cy) / fy * z, z)" in said


def test_the_world_ends_at_the_loaded_rooms_walls():
    """The episode loads only the activity's rooms (BEHAVIOR's evaluator: TASK_NAMES_TO_ROOMS), and doors in them open
    onto empty space: the stored scan of the 100 activities' scenes (scene_edges.json) finds doors with floor on one side
    and none on the other (a worker walked through one and fell to -27 m)."""
    environment = (PROJECT / "simulators/behavior/rosbridge_behavior/environment.py").read_text()
    assert 'cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS[task]' in environment
    edges = json.loads((MEASUREMENTS / "scene_edges.json").read_text())
    doors = [door for activity in edges["activities"].values() for door in activity["no_floor_beyond"]]
    assert doors and all(min(s["floor"] for s in door["sides"].values()) == 0 and
                         max(s["floor"] for s in door["sides"].values()) > 0 for door in doors)
    assert edges["summary"]["with_a_door_onto_no_floor"] == sum(1 for a in edges["activities"].values() if a["no_floor_beyond"])
    assert {"PARTIAL_SCENE", "SCENE_EDGE"} <= {rule.name for rule in rules()} and not missing(
        [rule for rule in rules() if rule.name in ("PARTIAL_SCENE", "SCENE_EDGE")])

