"""BEHAVIOR's R1Pro with SharpaWave hands as OmniGibson robot model r1pro_sharpa.

Each hand replaces a parallel gripper at the gripper's mount face on arm_link7: fingers continue
the gripper's approach axis, the palm faces the robot's midline while the arms hang at rest
(their zero pose) and the thumb leads forward. Each wrist camera, the R1Pro's RealSense with its
housing, optics and sensor name, moves from the gripper body to the back of the hand, where the
fingers do not cover it: at the R1Pro's pose a Sharpa hand fills its view from 2 cm.

Run as a module, this boots Isaac Sim and grafts the hands onto the R1Pro's shipped USD, so the
body, its collision hulls and its holonomic base stay exactly the R1Pro's: OmniGibson's URDF
importer converts both hands at their final poses with Sharpa's drive calibration and textured
visuals, and Sdf copies them into the R1Pro layer in place of the grippers. compose() writes the
kinematics as a URDF for the Kinex rig.
"""
import json
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .sharpa import (SIDES, absolute_meshes, apply_drives, attach, colors, filtered_pairs, finger_links,
                     hand_drives, hand_joints, high_fidelity_hands, origin, paint, set_origin)

MODEL = "r1pro_sharpa"
TITLE = "the R1Pro with Sharpa hands"
R1PRO = Path("datasets/omnigibson-robot-assets/models/r1pro")
ROOT = "dex_mount"  # the hands-only import's root, dropped by the graft
# Each hand's flange frame (fingers along +z, palm toward +x) in its arm_link7 frame, which is the
# R1Pro gripper's frame turned by nothing: with the arms at zero they hang at the sides, the
# gripper approaching along -z, so the palm faces the midline and the thumb leads forward (+x).
MOUNT_ROTATIONS = {
    "left": np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]),
    "right": np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]),
}
GRIPPER = ("gripper_link", "gripper_finger_link1", "gripper_finger_link2", "gripper_joint",
           "gripper_finger_joint1", "gripper_finger_joint2")
# The wrist camera's fixed joint, re-attached from the gripper body to arm_link7.
CAMERA_JOINT = "realsense_joint"
# The USD camera (-z forward, +y up) in its RealSense link (r1pro_source_cfg camera_links).
CAMERA_IN_LINK = np.array([[0., -1., 0., 0.], [-1., 0., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]])
# The wrist camera on the back of the hand, in the hand's flange frame (fingers +z, palm +x):
# 100 mm behind the palm's axis at the flange, looking along the fingers turned 10 degrees toward
# the palm side, the image's top away from the hand. The open hand then fills the bottom sixth
# of the view, fingertips included (75 mm and 22 degrees: half of it).
CAMERA_POSITION, CAMERA_TILT_DEG = (-0.100, 0., -0.014), 10.
# The importer's mesh prototypes, which the hands' links reference.
PROTOTYPES = ("visuals", "colliders")


def mount(source, side):
    """The flange's 4x4 pose in the side's arm_link7: at the gripper's mount face, the top of
    the gripper body's mesh above its fixed joint."""
    urdf = Path(source) / R1PRO / "urdf"
    robot = ET.parse(urdf / "r1pro.urdf").getroot()
    gripper = origin(robot.find(f"joint[@name='{side}_gripper_joint']"))
    mesh = robot.find(f"link[@name='{side}_gripper_link']/visual/geometry/mesh").get("filename")
    top = max(float(line.split()[3]) for line in (urdf / mesh).read_text().splitlines() if line.startswith("v "))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = MOUNT_ROTATIONS[side], gripper[:3, 3] + gripper[:3, :3] @ [0., 0., top]
    return matrix


def camera_pose():
    """The USD camera's pose (4x4) in the flange frame."""
    tilt = np.deg2rad(CAMERA_TILT_DEG)
    forward, up = np.array([np.sin(tilt), 0., np.cos(tilt)]), np.array([-np.cos(tilt), 0., np.sin(tilt)])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((np.cross(up, -forward), up, -forward))
    matrix[:3, 3] = CAMERA_POSITION
    return matrix


def camera_link(flange):
    """The RealSense link's pose (4x4) in arm_link7, given the flange's there."""
    return flange @ camera_pose() @ np.linalg.inv(CAMERA_IN_LINK)


def compose(source):
    """The R1Pro with Sharpa hands as URDF text, for kinematics: grippers removed, each hand
    on its mount and its wrist camera on arm_link7 behind the back of the hand."""
    source = Path(source)
    robot = ET.parse(source / R1PRO / "urdf/r1pro.urdf").getroot()
    absolute_meshes(robot, lambda name: source / R1PRO / "urdf" / name)
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    for side in SIDES:
        matrix = mount(source, side)
        camera = joints[f"{side}_{CAMERA_JOINT}"]
        set_origin(camera, camera_link(matrix))
        camera.find("parent").set("link", f"{side}_arm_link7")
        for element in list(robot):
            if element.get("name") in {f"{side}_{name}" for name in GRIPPER}:
                robot.remove(element)
        attach(robot, source, side, f"{side}_arm_link7", matrix[:3, 3], matrix[:3, :3])
    robot.set("name", MODEL)
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode") + "\n"


def hands(source, link7):
    """Both hands alone as URDF text, at their poses in the R1Pro USD's root frame with the arms
    at zero; link7 maps each side to its arm_link7 pose (4x4) there."""
    robot = ET.Element("robot", name=MODEL)
    ET.SubElement(robot, "link", name=ROOT)
    for side in SIDES:
        matrix = link7[side] @ mount(source, side)
        attach(robot, source, side, ROOT, matrix[:3, 3], matrix[:3, :3])
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode") + "\n"


def definition(source):
    """The R1Pro's OmniGibson definition with joint-controlled hands in place of the grippers."""
    source = Path(source)
    robot = yaml.safe_load((source / R1PRO / "r1pro.yaml").read_text())
    hands = {side: hand_joints(source, side) for side in SIDES}
    for side in SIDES:
        robot["default_controllers"][f"gripper_{side}"] = "JointController"
    pairs = [pair for pair in robot["disabled_collision_pairs"] if not any("gripper" in link for link in pair)]
    for side in SIDES:
        # The wrist camera sits beside the thumb's base, as it sat beside the gripper's fingers.
        thumb = [link for joint, link in zip(*hands[side]) if "thumb" in joint]
        near = [f"{side}_hand_flange", f"{side}_hand_wrist", f"{side}_hand_C_MC", *thumb]
        pairs += [[f"{side}_realsense_link", link] for link in near]
        pairs += [[f"{side}_arm_link7", link] for link in (f"{side}_hand_wrist", f"{side}_hand_C_MC")]
        # The R1Pro collides with itself, as the evaluator configures it; within each hand only
        # the pairs Sharpa's own model filters are exempt (the G1 runs without self-collisions).
        pairs += filtered_pairs(source, side)
    robot["disabled_collision_pairs"] = pairs
    # Tuck poses are lists in the native joint order, unknown before the import; zero lies within
    # every joint's limits, and R1ProSharpa sets the evaluator's posture by joint name.
    del robot["mobile_manipulation"]
    dofs = 6 + len(robot["articulated_trunk"]["trunk_joint_names"]) + sum(
        len(robot["manipulation"]["arm_joint_names"][side]) + len(hands[side][0]) for side in SIDES)
    robot["default_joint_pos"] = [0.] * dofs
    manipulation = robot["manipulation"]
    manipulation["finger_link_names"] = {side: finger_links(source, side) for side in SIDES}
    manipulation["finger_joint_names"] = {side: hands[side][0] for side in SIDES}
    manipulation["gripper_link_names"] = {side: [f"{side}_hand_flange", f"{side}_hand_wrist", f"{side}_hand_C_MC",
                                                 f"{side}_realsense_link"] for side in SIDES}
    return robot


def link_poses(usda, links):
    """Links' poses (4x4) in a flat robot USDA's root frame, read from its text before Kit
    boots: each link Xform's translate and orient, the first after its definition."""
    text, poses = Path(usda).read_text(), {}
    for link in links:
        start = text.index(f'def Xform "{link}"')
        translate = re.search(r"double3 xformOp:translate = \(([^)]*)\)", text[start:])
        orient = re.search(r"quatd xformOp:orient = \(([^)]*)\)", text[start:])
        w, x, y, z = (float(v) for v in orient.group(1).split(","))
        matrix = np.eye(4)
        matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        matrix[:3, 3] = [float(v) for v in translate.group(1).split(",")]
        poses[link] = matrix
    return poses


# The graft, inside Isaac Sim ----------------------------------------------------------------

def xform(spec):
    """A flat robot layer's link pose (4x4) in its root frame, from its translate and orient."""
    translate, orient = (spec.attributes[name].default for name in ("xformOp:translate", "xformOp:orient"))
    return pose(translate, orient)


def pose(position, quaternion):
    w, (x, y, z) = quaternion.GetReal(), quaternion.GetImaginary()
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    matrix[:3, 3] = list(position)
    return matrix


def joint_frame(spec, body):
    return pose(spec.attributes[f"physics:localPos{body}"].default, spec.attributes[f"physics:localRot{body}"].default)


def set_joint_frame(spec, body, matrix):
    from pxr import Gf

    rotation = Gf.Matrix3d(*matrix[:3, :3].T.flatten().tolist()).ExtractRotation().GetQuat()
    spec.attributes[f"physics:localPos{body}"].default = Gf.Vec3f(*matrix[:3, 3].tolist())
    spec.attributes[f"physics:localRot{body}"].default = Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary()))


def set_xform(spec, matrix):
    """Set a flat robot layer's link pose (4x4) in its root frame."""
    from pxr import Gf

    rotation = Gf.Matrix3d(*matrix[:3, :3].T.flatten().tolist()).ExtractRotation().GetQuat()
    spec.attributes["xformOp:translate"].default = Gf.Vec3d(*matrix[:3, 3].tolist())
    spec.attributes["xformOp:orient"].default = Gf.Quatd(rotation)


def set_target(spec, name, path):
    rel = spec.relationships[name]
    rel.targetPathList.ClearEdits()
    rel.targetPathList.explicitItems.append(path)


def retarget(layer, old, new):
    """Point relationships and connections under old at new: what CopySpec leaves outside the
    copied subtree."""
    from pxr import Sdf

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        lists = []
        if isinstance(spec, Sdf.RelationshipSpec):
            lists.append(spec.targetPathList)
        elif isinstance(spec, Sdf.AttributeSpec):
            lists.append(spec.connectionPathList)
        for edits in lists:
            for item in list(edits.GetAddedOrExplicitItems()):
                if item.HasPrefix(old):
                    edits.ReplaceItemEdits(item, item.ReplacePrefix(old, new))
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)


def dangling(layer):
    """Relationship targets that name no prim in layer."""
    from pxr import Sdf

    missing = []

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec):
            missing.extend(f"{path} -> {item}" for item in spec.targetPathList.GetAddedOrExplicitItems()
                           if not layer.GetPrimAtPath(item.GetPrimPath()))
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return missing


def graft(r1pro, hands_layer, output, flanges):
    """Write output: the R1Pro layer as model r1pro_sharpa with hands_layer's hands, eef links
    and materials in place of its grippers, and each wrist camera behind its hand; flanges maps
    each side to its flange's pose (4x4) in arm_link7."""
    from pxr import Sdf

    source, hands = Sdf.Layer.FindOrOpen(str(r1pro)), Sdf.Layer.FindOrOpen(str(hands_layer))
    out = Sdf.Layer.CreateNew(str(output))
    old, root = Sdf.Path("/r1pro"), Sdf.Path(f"/{MODEL}")
    for key in ("upAxis", "metersPerUnit", "kilogramsPerUnit", "timeCodesPerSecond"):
        if source.pseudoRoot.HasInfo(key):
            out.pseudoRoot.SetInfo(key, source.pseudoRoot.GetInfo(key))
    for prim in source.rootPrims:
        Sdf.CopySpec(source, prim.path, out, root if prim.path == old else prim.path)
    out.defaultPrim = MODEL
    retarget(out, old, root)
    robot = out.GetPrimAtPath(root)
    robot.attributes["ig:model"].default = MODEL
    links = {spec.name: spec for spec in robot.nameChildren.values()}
    for side in SIDES:
        link7 = links[f"{side}_arm_link7"]
        gripper, housing = links[f"{side}_gripper_link"], links[f"{side}_realsense_link"]
        if not np.allclose(xform(housing.nameChildren["Camera"]), CAMERA_IN_LINK, atol=1e-4):
            raise ValueError(f"the {side} wrist camera is not where CAMERA_IN_LINK puts it")
        # The camera's fixed joint moves from the gripper body to arm_link7, and the RealSense
        # with it (a flat layer holds each link's pose with the joints at zero).
        path = link7.path.AppendChild(f"{side}_{CAMERA_JOINT}")
        Sdf.CopySpec(out, gripper.nameChildren[f"{side}_{CAMERA_JOINT}"].path, out, path)
        moved = out.GetPrimAtPath(path)
        set_target(moved, "physics:body0", link7.path)
        frame = camera_link(flanges[side])
        set_joint_frame(moved, 0, frame)
        set_joint_frame(moved, 1, np.eye(4))
        set_xform(housing, xform(link7) @ frame)
        del link7.nameChildren[f"{side}_gripper_joint"]
        for name in ("gripper_link", "gripper_finger_link1", "gripper_finger_link2", "eef_link"):
            del robot.nameChildren[f"{side}_{name}"]
    # The hands, their eef links, joints and materials, and the prototypes they reference.
    hand_root = hands.GetPrimAtPath(root)
    for spec in hand_root.nameChildren.values():
        if spec.name == ROOT:
            continue
        if spec.name == "Looks":
            for material in spec.nameChildren.values():
                target = root.AppendChild("Looks").AppendChild(material.name)
                if out.GetPrimAtPath(target):
                    raise ValueError(f"material {material.name} exists in both layers")
                Sdf.CopySpec(hands, material.path, out, target)
            continue
        if out.GetPrimAtPath(spec.path):
            raise ValueError(f"{spec.name} exists in both layers")
        Sdf.CopySpec(hands, spec.path, out, spec.path)
    for scope in PROTOTYPES:  # the R1Pro's meshes are inline; the import's physics scene is not needed
        path = Sdf.Path.absoluteRootPath.AppendChild(scope)
        if hands.GetPrimAtPath(path):
            if out.GetPrimAtPath(path):
                raise ValueError(f"{path} exists in both layers")
            Sdf.CopySpec(hands, path, out, path)
            if out.GetPrimAtPath(path.AppendChild(ROOT)):
                del out.GetPrimAtPath(path).nameChildren[ROOT]  # the dropped root's empty prototype
    for side in SIDES:
        joint = find(out, root, f"{side}_hand_mount_joint")
        link7 = out.GetPrimAtPath(root.AppendChild(f"{side}_arm_link7"))
        # The import's root sits at the origin, so the joint's frame there is its world pose.
        frame = np.linalg.inv(xform(link7)) @ joint_frame(joint, 0)
        set_target(joint, "physics:body0", link7.path)
        set_joint_frame(joint, 0, frame)
    missing = dangling(out)
    if missing:
        raise ValueError("the graft left dangling targets: " + "; ".join(missing[:5]))
    out.Save()


def find(layer, root, name):
    found = []
    layer.Traverse(root, lambda path: found.append(path) if path.IsPrimPath() and path.name == name else None)
    if len(found) != 1:
        raise ValueError(f"expected one {name}, found {found}")
    return layer.GetPrimAtPath(found[0])


def main():
    """source (the prepared runtime), staging data root, output model directory."""
    source, data, output = (Path(argument).resolve() for argument in sys.argv[1:4])
    from rosbridge.core.gpu import single_gpu

    os.environ.update(OMNIGIBSON_DATA_PATH=str(data), OMNIGIBSON_HEADLESS="1", OMNIGIBSON_GPU_ID="0",
                      OMNIGIBSON_RENDER_GPU=str(single_gpu()))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.examples.robots.import_custom_robot import add_sensor
    from omnigibson.utils.asset_conversion_utils import import_og_asset_from_urdf

    r1pro = source / R1PRO
    poses = link_poses(r1pro / "usd/r1pro.usda", [f"{side}_arm_link7" for side in SIDES])
    link7 = {side: poses[f"{side}_arm_link7"] for side in SIDES}
    staging = data / "staging"
    staging.mkdir(parents=True)
    urdf = staging / f"{MODEL}_hands.urdf"
    urdf.write_text(hands(source, link7))
    _, usd, robot = import_og_asset_from_urdf(
        category="robot", model=MODEL, dataset_name="robot", urdf_path=str(urdf),
        collision_method="convex", overwrite=True, use_usda=True)
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()
    for side in SIDES:
        # z out of the fingertips, y across the fingers, between the palm and the fingers.
        add_sensor(stage, robot, "VisualSphere", f"{side}_eef_link", f"{side}_hand_C_MC",
                   [0.02, 0., 0.07], [0., 0., 0., 1.])
    gains = hand_drives(source)
    apply_drives(stage, gains)  # Sharpa's gains are torques per radian
    high_fidelity_hands(stage, robot, source)
    paint(stage, robot, colors(ET.fromstring(urdf.read_text())))
    stage.Save()
    model = data / "robot/objects/robot" / MODEL
    # The stage keeps the hands' layer open, so the graft is written beside it and replaces it.
    usd = Path(stage.GetRootLayer().realPath)
    grafted = usd.with_name(f"{MODEL}_graft.usda")
    graft(r1pro / "usd/r1pro.usda", usd, grafted, {side: mount(source, side) for side in SIDES})
    os.replace(grafted, usd)
    shutil.copytree(r1pro / "usd/materials", usd.parent / "materials", dirs_exist_ok=True)
    (model / "misc").mkdir(exist_ok=True)
    shutil.copy(r1pro / "misc/metadata.json", model / "misc/metadata.json")
    (model / "urdf").mkdir(exist_ok=True)
    (model / "urdf" / f"{MODEL}.urdf").write_text(compose(source))
    (model / f"{MODEL}.yaml").write_text(yaml.safe_dump(definition(source), sort_keys=False))
    (model / "drives.json").write_text(json.dumps(gains, indent=2) + "\n")
    shutil.move(model, output)
    og.shutdown()


if __name__ == "__main__":
    main()
