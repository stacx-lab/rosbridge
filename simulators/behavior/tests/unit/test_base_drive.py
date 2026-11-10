"""The base drive's effort limit on frictionless wheels: on for the R1Pro with Sharpa hands, off for the evaluator's
R1Pro. OmniGibson, Isaac's physics material and USD's drive and material schemas are faked; configure's calls are
recorded in order."""
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from rosbridge_behavior import r1pro
from rosbridge_behavior.r1pro import BASE_DRIVES, R1Pro
from rosbridge_behavior.r1pro_sharpa import R1ProSharpa

ROBOT = "/World/scene_0/robot_r1"
WHEELS = ("wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3")  # the R1Pro's three wheel spheres


def test_only_the_dexterous_r1pro_limits_its_base_drive():
    assert R1Pro.drive_effort is None  # the evaluator's robot keeps the evaluator's physics (behavior)
    force, torque = R1ProSharpa.drive_effort
    assert (force, torque) == (500., 150.)


class Recorder:
    def __init__(self):
        self.calls, self.editing = [], False

    def __call__(self, *call):
        self.calls.append(call)


def fake_world(monkeypatch):
    """The omnigibson, omnigibson.lazy modules and a robot; every effect lands in one ordered log."""
    log = Recorder()

    @contextmanager
    def editing_usd():
        assert not log.editing, "editing_usd() must not nest"
        log.editing = True
        yield
        log.editing = False

    sim = SimpleNamespace(stop=lambda: log("stop"), play=lambda: log("play"), editing_usd=editing_usd)

    class Attribute:
        def __init__(self, owner, name):
            self.owner, self.name = owner, name

        def Set(self, value):
            assert log.editing, f"{self.name} written outside editing_usd()"
            log("set", self.owner, self.name, value)

    class Material:
        def __init__(self, prim_path, name, static_friction, dynamic_friction, restitution):
            assert log.editing
            self.prim = SimpleNamespace(path=prim_path)
            log("material", prim_path, static_friction, dynamic_friction, restitution)

    lazy = ModuleType("omnigibson.lazy")
    lazy.isaacsim = SimpleNamespace(core=SimpleNamespace(api=SimpleNamespace(materials=SimpleNamespace(
        physics_material=SimpleNamespace(PhysicsMaterial=Material)))))
    lazy.pxr = SimpleNamespace(
        PhysxSchema=SimpleNamespace(PhysxMaterialAPI=SimpleNamespace(Apply=lambda prim: SimpleNamespace(
            CreateFrictionCombineModeAttr=lambda: Attribute(prim.path, "frictionCombineMode")))),
        UsdPhysics=SimpleNamespace(DriveAPI=SimpleNamespace(Get=lambda prim, kind: SimpleNamespace(
            GetMaxForceAttr=lambda: Attribute(prim, f"drive:{kind}:maxForce")))))
    omnigibson = ModuleType("omnigibson")
    omnigibson.sim, omnigibson.lazy = sim, lazy
    monkeypatch.setitem(sys.modules, "omnigibson", omnigibson)
    monkeypatch.setitem(sys.modules, "omnigibson.lazy", lazy)
    torch = ModuleType("torch")  # R1ProSharpa.configure sets the start posture as a tensor
    torch.tensor = list
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Mesh:
        def __init__(self, path):
            self.path = path

        def apply_physics_material(self, material):
            assert not log.editing  # it opens its own editing_usd()
            log("bind", self.path, material.prim.path)

    class Link:
        def __init__(self, name, meshes):
            self.name, self.collision_meshes = name, {mesh: Mesh(f"{name}/{mesh}") for mesh in meshes}

        @property
        def mass(self):
            raise AssertionError("not read")

        @mass.setter
        def mass(self, value):
            log("mass", self.name, value)

    links = {name: Link(name, ["collisions"]) for name in WHEELS}
    links["base_link"] = Link("base_link", [f"base_link_col_{i}" for i in range(8)])
    joints = {name: SimpleNamespace(prim=name) for name in (*BASE_DRIVES, "base_footprint_z_joint",
                                                            "base_footprint_rx_joint", "base_footprint_ry_joint")}
    camera = SimpleNamespace(horizontal_aperture=20.)
    robot = SimpleNamespace(prim_path=ROBOT, base_footprint_link=Link("base_footprint", []), links=links,
                            joints=joints, sensors={"robot_r1:zed_link:Camera:0": camera}, dof_names_ordered=[])
    return log, robot, camera


def embodiment(cls):
    value = cls.__new__(cls)
    value.cameras, value.pose = {"head_camera": "robot_r1:zed_link:Camera:0"}, {}
    return value


def test_the_evaluators_r1pro_configures_only_the_base_mass_and_head_optics(monkeypatch):
    log, robot, camera = fake_world(monkeypatch)
    embodiment(R1Pro).configure(robot)
    assert log.calls == [("stop",), ("mass", "base_footprint", 250.), ("play",)]
    assert camera.horizontal_aperture == 40.


@pytest.mark.parametrize("cls", [R1ProSharpa, type("Limited", (R1Pro,), {"drive_effort": (320., 90.)})])
def test_a_limited_drive_is_written_to_usd_between_the_stop_and_the_play(monkeypatch, cls):
    """The material is bound and the limits authored while the simulator is stopped: PhysX reads both at every later
    play. A material bound while it plays does not reach PhysX."""
    log, robot, camera = fake_world(monkeypatch)
    embodiment(cls).configure(robot)
    force, torque = cls.drive_effort
    calls = log.calls
    assert calls[0] == ("stop",) and calls[1] == ("mass", "base_footprint", 250.) and calls[-1] == ("play",)
    material = f"{ROBOT}/{r1pro.WHEEL_MATERIAL}"
    body = calls[2:-1]
    assert body[0] == ("material", material, 0., 0., 0.)  # frictionless, without bounce
    assert ("set", material, "frictionCombineMode", "min") in body  # the pair's friction is the wheel's zero
    limits = {(call[1], call[2]): call[3] for call in body if call[0] == "set" and call[2].endswith("maxForce")}
    assert limits == {("base_footprint_x_joint", "drive:linear:maxForce"): force,
                      ("base_footprint_y_joint", "drive:linear:maxForce"): force,
                      ("base_footprint_rz_joint", "drive:angular:maxForce"): torque}
    bound = {call[1] for call in body if call[0] == "bind"}
    assert bound == {f"{name}/collisions" for name in WHEELS}  # the three wheel spheres, nothing of the base itself
    assert all(call[2] == material for call in body if call[0] == "bind")
    assert camera.horizontal_aperture == 40.
