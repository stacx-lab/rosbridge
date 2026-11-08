"""SharpaWave hands on BEHAVIOR robot models (g1_model, dex_model).

attach() mounts Sharpa's flange-mounted Wave hand on a URDF link; the import helpers give its
joints Sharpa's Isaac drive calibration, its links Sharpa's textured high-fidelity visuals, and
the remaining URDF visuals their colours. The hand repository is a nested source of the
simulators that use it, generated at third_party/sharpa.
"""
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

SHARPA = Path("third_party/sharpa/wave_01")
# SharpaWave's textured visuals of each hand, on the URDF's links but for the flange and wrist.
HIGH_FIDELITY = {
    "left": "sharpa_wave_high_fidelity/left_sharpa_wave_high_fidelity/src/left_sharpa_wave_hiRes.usd",
    "right": "sharpa_wave_high_fidelity/right_sharpa_wave_high_fidelity/source/right_hand_HB_HiRes.usd",
}
SIDES = ("left", "right")


# URDF frames --------------------------------------------------------------------------------

def rpy(rotation):
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    return np.arctan2(rotation[2, 1], rotation[2, 2]), pitch, np.arctan2(rotation[1, 0], rotation[0, 0])


def origin(element):
    """The 4x4 transform of an element's URDF origin."""
    node = element.find("origin")
    xyz, rpy = ((node.get(key) if node is not None else None) or "0 0 0" for key in ("xyz", "rpy"))
    xyz = np.array(xyz.split(), dtype=float)
    roll, pitch, yaw = np.array(rpy.split(), dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = axis_angle([0, 0, 1], yaw)[:3, :3] @ axis_angle([0, 1, 0], pitch)[:3, :3] @ \
        axis_angle([1, 0, 0], roll)[:3, :3]
    matrix[:3, 3] = xyz
    return matrix


def set_origin(element, matrix):
    node = element.find("origin")
    if node is None:
        node = ET.SubElement(element, "origin")
    node.set("xyz", " ".join(repr(float(v)) for v in matrix[:3, 3]))
    node.set("rpy", " ".join(repr(float(v)) for v in rpy(matrix[:3, :3])))


def axis_angle(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    return matrix


def absolute_meshes(robot, resolve):
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(resolve(mesh.get("filename"))))


# The hand -----------------------------------------------------------------------------------

def hand(source, side):
    package = f"{side}_sharpa_wave"
    return ET.parse(Path(source) / SHARPA / package / f"{package}_with_flange.urdf").getroot()


def hand_joints(source, side):
    """The Sharpa hand's actuated joints and finger links, in URDF order."""
    joints = [joint for joint in hand(source, side).findall("joint") if joint.get("type") != "fixed"]
    return [joint.get("name") for joint in joints], [joint.find("child").get("link") for joint in joints]


def finger_links(source, side):
    """The links the finger joints move that can touch: each actuated joint's child, then the links fixed to them
    with collision geometry. These are the fingertips' elastomer pads, which reach past the distal phalanges, so a
    fingertip press or pinch touches through a pad first; OmniGibson counts only finger links' contacts for toggles
    and assisted grasps."""
    links = hand_joints(source, side)[1]
    robot = hand(source, side)
    touching = {link.get("name") for link in robot.findall("link") if link.find("collision") is not None}
    for joint in robot.findall("joint"):  # in tree order: a fixed link follows the link it is fixed to
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        if joint.get("type") == "fixed" and parent in links and child in touching:
            links.append(child)
    return links


def filtered_pairs(source, side):
    """Link pairs that Sharpa's own USD keeps from colliding (PhysicsFilteredPairsAPI), such as
    the palm and the thumb's metacarpal; the URDF import drops them."""
    package = f"{side}_sharpa_wave"
    text = (Path(source) / SHARPA / package / f"{package}_with_flange.usda").read_text()
    pairs = []
    for match in re.finditer(r'"([^"]+)"\s*\(\s*prepend apiSchemas = \[[^\]]*"PhysicsFilteredPairsAPI"[^\]]*\]\s*\)'
                             r'\s*\{[^{}]*?rel physics:filteredPairs = (\[[^\]]*\]|<[^>]*>)', text, re.S):
        pairs += [[match.group(1), target.rsplit("/", 1)[-1]] for target in re.findall(r"<([^>]*)>", match.group(2))]
    return pairs


def attach(robot, source, side, parent, position, rotation):
    """Append the side's flange-mounted hand to robot (a URDF root element), its flange fixed
    to parent at position and rotation (3x3, the flange axes in parent's frame)."""
    package = f"{side}_sharpa_wave"
    hand = ET.parse(Path(source) / SHARPA / package / f"{package}_with_flange.urdf").getroot()
    folder = Path(source) / SHARPA / package
    absolute_meshes(hand, lambda name: folder / name.removeprefix(f"package://{package}/"))
    robot.extend(element for element in hand if element.tag in ("link", "joint"))
    mount = ET.SubElement(robot, "joint", name=f"{side}_hand_mount_joint", type="fixed")
    ET.SubElement(mount, "parent", link=parent)
    ET.SubElement(mount, "child", link=f"{side}_hand_flange")
    ET.SubElement(mount, "origin", xyz=" ".join(repr(float(v)) for v in position),
                  rpy=" ".join(repr(float(v)) for v in rpy(rotation)))


def colors(robot):
    """Each link's URDF visual colour (rgb) in robot (a URDF root element), resolving named
    materials."""
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


def hand_drives(source):
    """Position-drive stiffness and damping per radian, armature and friction of both hands'
    joints: Sharpa's Isaac calibration (Nm/deg in its USD)."""
    result = {}
    for side in SIDES:
        package = f"{side}_sharpa_wave"
        text = (Path(source) / SHARPA / package / f"{package}_with_flange.usda").read_text()
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


# USD, inside Isaac Sim ----------------------------------------------------------------------

def apply_drives(stage, gains):
    """Force drives with armature, and friction where given, on the joints named in gains."""
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if prim.GetName() in gains:
            # Force drives: stiffness and damping are torques per radian, as the gains assume;
            # the importer's acceleration drives would scale them by inertia.
            UsdPhysics.DriveAPI.Get(prim, "angular").GetTypeAttr().Set("force")
            joint = PhysxSchema.PhysxJointAPI.Apply(prim)
            joint.CreateArmatureAttr(gains[prim.GetName()]["armature"])
            if "friction" in gains[prim.GetName()]:
                joint.CreateJointFrictionAttr(gains[prim.GetName()]["friction"])


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


def paint(stage, robot, colors):
    """Bind the remaining URDF visuals to their links' colours (link: rgb); the importer binds white."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    materials = {}
    for link, color in colors.items():
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
