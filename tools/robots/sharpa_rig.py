"""The Kinex joint device of a SharpaWave hand on an arm flange (build_behavior_g1, build_behavior_dex)."""
import rigs


def hand(source, side, joints, home, mount):
    """The side's hand device and its hand.urdf text. source maps joint names to the composed
    URDF's joint elements; mount says where the flange sits, for the URDF's description."""
    device = rigs.joint(
        f"{side}_hand",
        f"Sharpa Wave 22-joint {side} hand on the {side}_arm flange, palm toward the body's "
        "midline. Geometry is the sibling hand.urdf from the flange to the fingertips, with no "
        "single tip link; commands are absolute native joint positions on Sharpa's calibrated "
        "position drives, and contact is physical.",
        joints, [home.get(joint, 0.) for joint in joints],
        rigs.pose_frame(f"{side}_arm:flange", rigs.urdf_origin(source[f"{side}_hand_mount_joint"])),
        part="hand", tip=None, model="hand.urdf")
    urdf = rigs.source_tree(
        source, f"{side}_hand_flange", f"sharpa_wave_{side}",
        f"Sharpa Wave {side} hand from {side}_sharpa_wave_with_flange.urdf: base_link is the "
        f"{side}_hand_flange frame {mount}.")
    return device, urdf
