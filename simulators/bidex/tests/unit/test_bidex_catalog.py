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


def test_an_out_of_domain_workspace_moves_the_cameras_its_episode_moves(tmp_path):
    from pathlib import Path
    import numpy as np
    import pytest
    import yaml
    from rosbridge.catalog.robots import export
    root = Path(__file__).resolve().parents[4]
    if not (root / ".runtime/bidex/bidex/eval/ood.py").is_file():
        pytest.skip("BiDex-100 is not prepared")
    nominal = root / "catalog/robots/bidex/bidex-fr3v2-sharpa-tabletop/devices"

    def cameras(task, seed):
        directory = tmp_path / f"{task}-{seed}"
        export(f"bidex/{task}", directory, seed=seed, project=root)
        return {name: yaml.safe_load((directory / ".kinex/devices" / name / "profile.yaml").read_text())["frames"]
                for name in ("head_camera", "left_camera", "right_camera")}
    catalog = {name: yaml.safe_load((nominal / name / "profile.yaml").read_text())["frames"]
               for name in ("head_camera", "left_camera", "right_camera")}
    # Seed 0 of the OOD twin is the evaluator's eval seed 2, episode 0: its plan moves the head camera only.
    assert cameras("put-apple-in-white-bowl", 0) == catalog
    moved = cameras("put-apple-in-white-bowl-ood", 0)
    assert moved["left_camera"] == catalog["left_camera"] and moved["right_camera"] == catalog["right_camera"]
    head, before = moved["head_camera"]["base"], catalog["head_camera"]["base"]
    assert head["parent"] == before["parent"] == "right_arm:base"
    shift = np.linalg.norm(np.subtract(head["translation_m"], before["translation_m"]))
    turn = 2 * np.degrees(np.arccos(min(1., abs(np.dot(head["rotation_xyzw"], before["rotation_xyzw"])))))
    # The pose bank's limits are 1.8 cm and 6 degrees per axis.
    assert 1e-3 < shift < .018 * 3 ** .5 and 0 < turn < 6 * 3 ** .5
