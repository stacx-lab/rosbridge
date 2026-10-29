"""The Unitree G1 with SharpaWave hands as OmniGibson robot model g1_sharpa.

compose() replaces the G1 29-DoF rev 1.0 rubber hands with Sharpa's flange-mounted Wave hands
at the hands' palm mounts. Run as a module, this boots Isaac Sim, converts the composition with
OmniGibson's own robot import functions and writes the model directory: its USD with joint
armature, the hands' textured high-fidelity visuals and the URDF's colours elsewhere, the robot
definition and drives.json, the position-drive gains of every joint.
"""
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

MODEL = "g1_sharpa"
G1 = Path("third_party/unitree_ros/robots/g1_description")
SHARPA = Path("third_party/sharpa/wave_01")
# SharpaWave's textured visuals of each hand, on the URDF's links but for the flange and wrist.
HIGH_FIDELITY = {
    "left": "sharpa_wave_high_fidelity/left_sharpa_wave_high_fidelity/src/left_sharpa_wave_hiRes.usd",
    "right": "sharpa_wave_high_fidelity/right_sharpa_wave_high_fidelity/source/right_hand_HB_HiRes.usd",
}
WBC = Path("third_party/decoupled_wbc")
SIDES = ("left", "right")
LEGS = [f"{side}_{joint}_joint" for side in SIDES
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARMS = {side: [f"{side}_{joint}_joint" for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                                      "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        for side in SIDES}
# GR00T WholeBodyControl's body order: its lower body (legs, waist), then the arms.
BODY = LEGS + WAIST + ARMS["left"] + ARMS["right"]
# The armature of every body joint in GR00T's MuJoCo G1 (model_data/g1/g1_29dof_with_hand.xml).
BODY_ARMATURE = 0.01
CAMERA = "d435_link"
# The USD camera (-z forward, +y up) on the D435 mount (+x forward, +z up), as x, y, z, w.
CAMERA_ORIENTATION = [0.5, -0.5, -0.5, 0.5]
# Each hand's flange frame (fingers along +z, palm toward +x) in its wrist_yaw_link (forearm
# along +x) at the G1 hand mount: the palm faces the body's midline as the G1's own hands do,
# and the index finger leads on the thumb side.
MOUNTS = {
    "left": ((0.0415, 0.003, 0.), np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])),
    "right": ((0.0415, -0.003, 0.), np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])),
}


def rpy(rotation):
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    return np.arctan2(rotation[2, 1], rotation[2, 2]), pitch, np.arctan2(rotation[1, 0], rotation[0, 0])


def absolute_meshes(robot, resolve):
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(resolve(mesh.get("filename"))))


def compose(source):
    """The composed URDF text; source is the generated tree holding both robot descriptions."""
    source = Path(source)
    robot = ET.parse(source / G1 / "g1_29dof_rev_1_0.urdf").getroot()
    absolute_meshes(robot, lambda name: source / G1 / name)
    for element in list(robot):
        if element.tag == "mujoco" or element.get("name") in (
                *(f"{side}_hand_palm_joint" for side in SIDES), *(f"{side}_rubber_hand" for side in SIDES)):
            robot.remove(element)
    robot.set("name", MODEL)
    for side in SIDES:
        package = f"{side}_sharpa_wave"
        hand = ET.parse(source / SHARPA / package / f"{package}_with_flange.urdf").getroot()
        folder = source / SHARPA / package
        absolute_meshes(hand, lambda name: folder / name.removeprefix(f"package://{package}/"))
        robot.extend(element for element in hand if element.tag in ("link", "joint"))
        position, rotation = MOUNTS[side]
        mount = ET.SubElement(robot, "joint", name=f"{side}_hand_mount_joint", type="fixed")
        ET.SubElement(mount, "parent", link=f"{side}_wrist_yaw_link")
        ET.SubElement(mount, "child", link=f"{side}_hand_flange")
        ET.SubElement(mount, "origin", xyz=" ".join(map(repr, position)),
                      rpy=" ".join(repr(float(v)) for v in rpy(rotation)))
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode") + "\n"


def hand(source, side):
    package = f"{side}_sharpa_wave"
    return ET.parse(Path(source) / SHARPA / package / f"{package}_with_flange.urdf").getroot()


def hand_joints(source, side):
    """The Sharpa hand's actuated joints and finger links, in URDF order."""
    joints = [joint for joint in hand(source, side).findall("joint") if joint.get("type") != "fixed"]
    return [joint.get("name") for joint in joints], [joint.find("child").get("link") for joint in joints]


def colors(source):
    """Each link's URDF visual colour (rgb) in the composition, resolving named materials."""
    robot = ET.fromstring(compose(source))
    named = {material.get("name"): material.find("color") for material in robot.findall("material")}
    result = {}
    for link in robot.findall("link"):
        material = link.find("visual/material")
        if material is None:
            continue
        color = material.find("color")
        color = named.get(material.get("name")) if color is None else color
        if color is not None:
            result[link.get("name")] = tuple(float(value) for value in color.get("rgba").split()[:3])
    return result


def drives(source):
    """Position-drive stiffness and damping per radian, armature and friction of every joint:
    GR00T's G1 motor gains, and Sharpa's Isaac calibration of each hand (Nm/deg in its USD)."""
    source = Path(source)
    config = yaml.safe_load((source / WBC / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml").read_text())
    result = {joint: {"stiffness": float(kp), "damping": float(kd), "armature": BODY_ARMATURE}
              for joint, kp, kd in zip(BODY, config["MOTOR_KP"], config["MOTOR_KD"], strict=True)}
    for side in SIDES:
        package = f"{side}_sharpa_wave"
        text = (source / SHARPA / package / f"{package}_with_flange.usda").read_text()
        for name in hand_joints(source, side)[0]:
            block = re.search(rf'over "{name}"\s*{{(.*?)}}', text, re.S).group(1)
            value = {key: float(re.search(rf"{attribute} = ([-+.\deE]+)", block).group(1))
                     for key, attribute in (("stiffness", "drive:angular:physics:stiffness"),
                                            ("damping", "drive:angular:physics:damping"),
                                            ("armature", "physxJoint:armature"),
                                            ("friction", "physxJoint:jointFriction"))}
            value["stiffness"] *= 180 / math.pi
            value["damping"] *= 180 / math.pi
            result[name] = value
    return result


def arm_links(source):
    """The links of both arms and hands: everything below the shoulder pitch joints."""
    robot = ET.fromstring(compose(source))
    children = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint.find("child").get("link"))
    links, pending = set(), [f"{side}_shoulder_pitch_link" for side in SIDES]
    while pending:
        link = pending.pop()
        links.add(link)
        pending += children.get(link, [])
    return links | {f"{side}_eef_link" for side in SIDES}


def standing_height(source, angles):
    """The pelvis height at which the G1's lowest foot collision sphere touches the floor, with
    the leg joints at angles (name: radians)."""
    robot = ET.fromstring(compose(source))
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    lowest = np.inf
    for link in (f"{side}_ankle_roll_link" for side in SIDES):
        pose, child = np.eye(4), link
        while child in joints:
            joint = joints[child]
            local = origin(joint)
            if joint.get("type") == "revolute":
                axis = np.array(joint.find("axis").get("xyz").split(), dtype=float)
                local = local @ axis_angle(axis, angles.get(joint.get("name"), 0.))
            pose, child = local @ pose, joint.find("parent").get("link")
        for collision in robot.find(f"link[@name='{link}']").findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is not None:
                center = pose @ origin(collision)[:, 3]
                lowest = min(lowest, center[2] - float(sphere.get("radius")))
    return -lowest


def origin(element):
    node = element.find("origin")
    xyz, rpy = ((node.get(key) if node is not None else None) or "0 0 0" for key in ("xyz", "rpy"))
    xyz = np.array(xyz.split(), dtype=float)
    roll, pitch, yaw = np.array(rpy.split(), dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = axis_angle([0, 0, 1], yaw)[:3, :3] @ axis_angle([0, 1, 0], pitch)[:3, :3] @ \
        axis_angle([1, 0, 0], roll)[:3, :3]
    matrix[:3, 3] = xyz
    return matrix


def axis_angle(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    return matrix


def definition(source):
    """The OmniGibson robot definition: GR00T's lower body as the trunk, two arms and hands."""
    hands = {side: hand_joints(source, side) for side in SIDES}
    return {
        "raw_controller_order": ["trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"],
        # Zero lies within every joint's limits, whatever the native joint order; G1Sharpa sets
        # the standing posture by joint name.
        "default_joint_pos": [0.] * (len(BODY) + sum(len(hands[side][0]) for side in SIDES)),
        "default_controllers": {"trunk": "JointController", **{
            f"{part}_{side}": "JointController" for part in ("arm", "gripper") for side in SIDES}},
        "base_footprint_link_name": "pelvis",
        # A floating base: the legs walk, so there are no base joints.
        "locomotion": {"base_joint_names": [],
                       "floor_touching_base_link_names": [f"{side}_ankle_roll_link" for side in SIDES]},
        "articulated_trunk": {"trunk_joint_names": LEGS + WAIST},
        "manipulation": {
            "n_arms": 2, "arm_names": list(SIDES),
            "arm_link_names": {side: [joint.replace("_joint", "_link") for joint in ARMS[side]]
                               for side in SIDES},
            "arm_joint_names": ARMS,
            "eef_link_names": {side: f"{side}_eef_link" for side in SIDES},
            "finger_link_names": {side: hands[side][1] for side in SIDES},
            "finger_joint_names": {side: hands[side][0] for side in SIDES},
        },
    }


def high_fidelity_hands(stage, robot, source):
    """Replace each hand link's URDF visuals with SharpaWave's textured ones and their materials."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    layer, looks = stage.GetRootLayer(), robot.GetPath().AppendChild("Looks")
    for side, file in HIGH_FIDELITY.items():
        path = Path(source) / SHARPA / file
        # Texture paths are relative, so the textures sit beside the model's USD as beside Sharpa's.
        shutil.copytree(path.parent / "textures", Path(layer.realPath).parent / "textures", dirs_exist_ok=True)
        hires = Usd.Stage.Open(str(path))
        root, copied = hires.GetDefaultPrim(), {}
        for link in (element.get("name") for element in hand(source, side).findall("link")):
            if not root.GetChild(link):
                continue
            visuals = root.GetChild(link).GetPath().AppendChild("visuals")
            ours = robot.GetPath().AppendPath(f"{link}/visuals")
            stage.RemovePrim(ours)
            stage.RemovePrim(f"/visuals/{link}")
            if not hires.GetPrimAtPath(visuals):
                continue  # the elastomers: the distal phalanges' visuals include them
            Sdf.CopySpec(hires.GetRootLayer(), visuals, layer, ours)

            def target(path):
                """The robot's copy of a bound material; OmniGibson needs ASCII-led prim names."""
                path = path.ReplacePrefix(ours, visuals)
                if not hires.GetPrimAtPath(path):
                    # Sharpa binds some thumb faces to a material it keeps under the collisions.
                    path = next((prim.GetPath() for prim in Usd.PrimRange(root.GetChild(link))
                                 if prim.IsA(UsdShade.Material) and prim.GetName() == path.name), path)
                if path not in copied:
                    name = f"{side}_sharpa_{re.sub(r'[^0-9A-Za-z_]', '', path.name)}"
                    while stage.GetPrimAtPath(looks.AppendChild(name)):
                        name += "_"
                    copied[path] = looks.AppendChild(name)
                    Sdf.CopySpec(hires.GetRootLayer(), path, layer, copied[path])
                return copied[path]
            prims = list(Usd.PrimRange(stage.GetPrimAtPath(ours)))
            for prim in prims:
                binding = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
                if binding:
                    binding.SetTargets([target(path) for path in binding.GetTargets()])
            for prim in prims:
                if not prim.IsValid():
                    continue
                # Sharpa keeps superseded meshes hidden; OmniGibson would show them.
                if prim.IsA(UsdShade.Material) or prim.IsA(UsdGeom.Mesh) and \
                        UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                    stage.RemovePrim(prim.GetPath())
                elif prim.IsA(UsdGeom.Xformable):
                    standard_xform(prim)


def standard_xform(prim):
    """Author prim's local transform as OmniGibson's translate, orient and scale ops."""
    from pxr import Gf, UsdGeom

    xformable, transform = UsdGeom.Xformable(prim), Gf.Transform()
    transform.SetMatrix(xformable.GetLocalTransformation())
    for name in prim.GetPropertyNames():
        if name.startswith("xformOp:"):
            prim.RemoveProperty(name)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(transform.GetTranslation())
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(transform.GetRotation().GetQuat())
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(transform.GetScale()))


def paint(stage, robot, source):
    """Bind the remaining URDF visuals to their links' colours; the importer binds white."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    materials = {}
    for link, color in colors(source).items():
        prototype = stage.GetPrimAtPath(f"/visuals/{link}")
        if not prototype:
            continue
        if color not in materials:
            path = robot.GetPath().AppendPath("Looks/urdf_%02x%02x%02x" % tuple(round(255 * c) for c in color))
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
            shader.SetSourceAsset("OmniPBR.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            materials[color] = material
        for prim in Usd.PrimRange(prototype):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[color])


def main():
    """source, staging data root, output model directory: import inside Isaac Sim."""
    source, data, output = map(Path, sys.argv[1:4])
    from rosbridge.core.gpu import single_gpu

    os.environ.update(OMNIGIBSON_DATA_PATH=str(data), OMNIGIBSON_HEADLESS="1", OMNIGIBSON_GPU_ID="0",
                      OMNIGIBSON_RENDER_GPU=str(single_gpu()))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    staging = data / "staging"
    staging.mkdir(parents=True)
    urdf = staging / f"{MODEL}.urdf"
    urdf.write_text(compose(source))
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.examples.robots.import_custom_robot import add_sensor
    from omnigibson.utils.asset_conversion_utils import import_og_asset_from_urdf

    # Upstream import_custom_robot without its canonical-frame shift: Isaac Sim 5.1's URDF
    # importer keeps joints outside the link prims, so that shift would move the pelvis geometry
    # but not its joints. The pelvis stays the root frame; G1Sharpa stands it at its height.
    # Hulls of the visual meshes serve as collision where the URDF has no dedicated proxy.
    _, _, robot = import_og_asset_from_urdf(
        category="robot", model=MODEL, dataset_name="robot", urdf_path=str(urdf),
        collision_method="convex", overwrite=True, use_usda=True)
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()
    for side in SIDES:
        # z out of the fingertips, y across the fingers, between the palm and the fingers.
        add_sensor(stage, robot, "VisualSphere", f"{side}_eef_link", f"{side}_hand_C_MC",
                   [0.02, 0., 0.07], [0., 0., 0., 1.])
    add_sensor(stage, robot, "Camera", CAMERA, None, [0., 0., 0.], CAMERA_ORIENTATION)
    from pxr import PhysxSchema, UsdPhysics

    gains = drives(source)
    for prim in stage.Traverse():
        if prim.GetName() in gains:
            # Force drives: stiffness and damping are torques per radian, as GR00T's and Sharpa's
            # gains assume; the importer's acceleration drives would scale them by inertia.
            UsdPhysics.DriveAPI.Get(prim, "angular").GetTypeAttr().Set("force")
            joint = PhysxSchema.PhysxJointAPI.Apply(prim)
            joint.CreateArmatureAttr(gains[prim.GetName()]["armature"])
            if "friction" in gains[prim.GetName()]:
                joint.CreateJointFrictionAttr(gains[prim.GetName()]["friction"])
    high_fidelity_hands(stage, robot, source)
    paint(stage, robot, source)
    stage.Save()
    model = data / "robot/objects/robot" / MODEL
    (model / f"{MODEL}.yaml").write_text(yaml.safe_dump(definition(source), sort_keys=False))
    (model / "drives.json").write_text(json.dumps(gains, indent=2) + "\n")
    shutil.move(model, output)
    og.shutdown()


if __name__ == "__main__":
    main()
