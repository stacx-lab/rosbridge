"""Publish the pelvis device's URDF links on TF from measured joint states.

pelvis_pitch and pelvis_roll are measured, not commanded. pelvis.urdf declares them fixed, so the device commands
pelvis_height alone; this publisher turns each about its URDF axis by its measured angle (zero until one is measured),
so the pelvis link, where the torso mounts, has the real pelvis orientation.
"""
from pathlib import Path
from xml.etree import ElementTree

from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from kinex.communication import qos
from kinex.lib.frames import dynamic_qos, transform
from kinex.lib.geometry import compose_rt, flat
from kinex.lib.urdf import displacement, joint_transform, positions, read_tree
from kinex.node import LinkPublisher, node_name


class PelvisLinks(LinkPublisher):
    def __init__(self, profile, context, directory):
        # LinkPublisher's setup, except that the fixed joints are the measured ones, published with each joint state.
        Node.__init__(self, node_name(profile, "links"), context=context)
        model, path = profile["model"], Path(directory) / profile["model"]["path"]
        self.tree = read_tree(path, model["base_link"])
        self.device, self.base = profile["name"], model["base_link"]
        self.axes = {joint.get("name"): [float(v) for v in joint.find("axis").get("xyz").split()]
                     for joint in ElementTree.parse(path).getroot().findall("joint")
                     if joint.get("type") == "fixed" and joint.find("axis") is not None}
        self.broadcaster = TransformBroadcaster(self, qos=dynamic_qos())
        self.observed = False
        state = profile["operations"]["joint_state"]
        self.create_subscription(JointState, state["endpoint"], self.state, qos(state))

    def state(self, message):
        measured = dict(zip(message.name, message.position))
        values = positions(self.tree, measured)
        transforms = []
        for joint in self.tree["joints"]:
            if joint["type"] != "fixed" and joint["name"] in values:
                pose = joint_transform(joint, values[joint["name"]])
            elif joint["name"] in self.axes:
                turn = displacement({"type": "revolute", "axis": self.axes[joint["name"]]}, measured.get(joint["name"], 0.))
                pose = compose_rt(joint_transform(joint), turn)
            else:
                continue
            transforms.append(transform(self.frame(joint["parent"]), self.frame(joint["child"]), flat(pose),
                                        message.header.stamp))
        self.broadcaster.sendTransform(transforms)
        self.observed = True


def create(profile, context):
    return PelvisLinks(profile, context, Path(__file__).parent)
