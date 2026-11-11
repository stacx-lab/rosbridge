import re

from rosbridge.catalog import resolve, simulator, tasks


def test_catalog_holds_core_50_and_transfer_16_on_one_rig():
    entries = list(tasks(name="handson").values())
    prefixes = [re.match(r"t\d\d|eval_(comb|gen|flex)\d\d", t["native_task"]).group() for t in entries]
    assert prefixes == [f"t{i:02d}" for i in range(1, 51)] + [
        f"eval_{group}{i:02d}" for group, count in (("comb", 6), ("flex", 3), ("gen", 7))
        for i in range(1, count + 1)]
    for task in entries:
        assert task["robot"] == "handson/handson-rokae-wuji"
        assert task["id"] == "handson/" + task["native_task"].replace("_", "-")
        assert resolve(task["id"])[2] == task
    assert resolve("handson")[2]["native_task"] == "t24_place_ball_in_bowl"


def test_manifest_runs_isaac_lab_in_a_native_worker():
    _, _, config = simulator("handson")
    assert (config["environment"], config["native_environment"]) == ("handson", "handson-native")
    assert [(n["path"], n["install"]) for n in config["nested_sources"]] == [
        ("third_party/IsaacLab", ["source/isaaclab"])]
    assert set(config["devices"]) == {f"{side}_{part}" for side in ("left", "right")
                                      for part in ("arm", "hand", "camera")} | {"head_camera"}
