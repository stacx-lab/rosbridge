"""Measure the R1Pro with Sharpa hands and write catalog/robots/behavior-dex/behavior-r1pro-sharpa.

The body is the R1Pro rig's (build_behavior.body): chains from the composed URDF
(rosbridge_behavior.dex_model), the camera mounts measured in the live OmniGibson scene, with
the wrist cameras on the backs of the hands. Each hand is a Sharpa joint device on its arm's
flange. Runs in the native environment after
`rosbridge prepare behavior-dex`:
    pixi run -e behavior-dex-native env PYTHONPATH=src:simulators/behavior:.runtime/behavior-dex \
        python tools/robots/build_behavior_dex.py
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulators/behavior"), str(Path(__file__).parent)]

from build_behavior import body  # noqa: E402
import rigs  # noqa: E402
import sharpa_rig  # noqa: E402
from rosbridge_behavior import dex_model  # noqa: E402
from rosbridge_behavior.backend import R1ProSharpaSim  # noqa: E402


def main():
    sim = R1ProSharpaSim()
    sim.reset(sim.default_task, 0)
    composed = ET.fromstring(dex_model.compose(ROOT / ".runtime/behavior-dex"))
    source = {joint.get("name"): joint for joint in composed.findall("joint")}
    devices, files = body(sim, source, wrist="hand's back, looking along the fingers past the fingertips")
    for side in dex_model.SIDES:
        devices[f"{side}_hand"], files[f"devices/{side}_hand/hand.urdf"] = sharpa_rig.hand(
            source, side, sim.embodiment.hands[side], {},
            "on the R1Pro's arm_link7, at its gripper's mount face")
    order = ["world", "base", "torso", "left_arm", "left_hand", "right_arm", "right_hand",
             "head_camera", "left_camera", "right_camera"]
    rigs.write_rig(ROOT / "catalog/robots/behavior-dex/behavior-r1pro-sharpa",
                   rigs.workspace("behavior-r1pro-sharpa", "BEHAVIOR R1Pro with Sharpa hands", "mobile",
                                  "Galaxea R1 Pro (wheeled dual-arm mobile manipulator) with SharpaWave hands"),
                   {name: devices[name] for name in order}, files)
    print("rig written", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
