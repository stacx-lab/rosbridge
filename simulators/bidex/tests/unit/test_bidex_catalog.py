from collections import Counter

from rosbridge.catalog import resolve, simulator, tasks

TABLETOP, DEXJOCO, ASTRIBOT = (f"bidex/bidex-{rig}" for rig in (
    "fr3v2-sharpa-tabletop", "fr3v2-sharpa-dexjoco", "astribot-s1-sharpa"))


def test_catalog_pairs_every_task_with_its_ood_twin_on_one_rig():
    entries = list(tasks(name="bidex").values())
    tasks_, twins = entries[::2], entries[1::2]
    assert len(tasks_) == len(twins) == 106
    for task, twin in zip(tasks_, twins):
        assert task["native_task"] == f'{task["suite"]}/{task["id"][6:].replace("-", "_")}'
        assert (twin["id"], twin["native_task"]) == (task["id"] + "-ood", task["native_task"] + "__ood")
        assert (twin["robot"], twin["suite"]) == (task["robot"], task["suite"])
        assert resolve(task["id"])[2] == task and resolve(twin["id"])[2] == twin
    assert Counter(task["suite"] for task in tasks_) == {
        "Memory": 10, "Long_Horizon": 10, "Generalization": 40, "Instruction_Following": 20,
        "Limited_Demonstrations": 26}
    assert Counter(task["robot"] for task in tasks_) == {TABLETOP: 76, DEXJOCO: 20, ASTRIBOT: 10}
    assert resolve("bidex")[2]["id"] == "bidex/put-apple-in-white-bowl"


def test_manifest_runs_in_process_with_hands_and_rgbd_cameras():
    _, _, config = simulator("bidex")
    assert config["environment"] == "bidex" and "native_environment" not in config
    assert config["devices"] == {
        **{f"{side}_{part}": "joint" for part in ("arm", "hand") for side in ("right", "left")},
        **{f"{camera}_camera": "rgbd" for camera in ("head", "left", "right")}}
