"""The rules that decide a worker's outcomes on the BEHAVIOR G1, in one place.

Every value here is one of three kinds:
- a constant the code acts on (g1_sharpa and the world's command buffer use these names);
- a value upstream code acts on (OmniGibson, GR00T), stated here and checked against the upstream source by the unit
  tests;
- a measured property of the robot, with the probe that measured it.
The worker reads them in the device profiles (catalog/robots/behavior-g1/behavior-g1-sharpa, exported into the Kinex
workspace) and, under assisted grasping, in the task instruction's note (assisted_note(), which an external task
generator renders). rules() lists where each value is stated and in which words, and a unit test fails when a value changes
without its text. No third-party imports: a task generator imports this module with the host's Python.
"""
from typing import NamedTuple

# Constants the code acts on -----------------------------------------------------------------------------------------
VELOCITY_LIMIT = .5  # m/s and rad/s: GR00T's keyboard navigation limits for vx, vy and wz
# The pelvis height command (m above the floor) GR00T's balance policy takes. Its own teleoperation clamps it to
# 0.20-0.74; the policy squats to 0.20 without falling and stands back up in one command.
PELVIS_LIMITS = (.20, .78)
CLIP = True  # targets beyond a limit act at the limit (the G1 layout clips; a refusal is silent to a topic publisher)
SELF_COLLISIONS = False  # the robot's links do not collide with each other (OmniGibson robot config)
# A hand applies a grasp while its flexion joints (all but the abduction joints, _AA) are commanded past this fraction
# of their closing range on average; it releases as soon as the average is back at or below it.
GRASP_CLOSURE = .3
# A fall: FALL_HOLD_S of world time in all with the pelvis tilted more than FALL_TILT_DEG from upright or lower than
# FALL_HEIGHT (m) above the floor, counted since it was last within FALL_UPRIGHT_DEG of upright (and higher); from then
# on the legs hold still. Measured (native probes): squats and walking stay within 17 degrees and
# above 0.248 m (arm sweeps in a deep squat reach 32 degrees), pushes the body recovers from within 24 degrees (one
# stumble touched 52 degrees for under 0.2 s), and
# fallen bodies pass 45 degrees within 0.1-0.35 s of starting to tip and lie at 50-160 degrees, 0.09-0.48 m.
FALL_TILT_DEG, FALL_UPRIGHT_DEG, FALL_HEIGHT, FALL_HOLD_S = 45., 30., .15, 1.

# Upstream values (checked against the sources by the unit tests) ----------------------------------------------------
STANDING_PELVIS = .74  # GR00T's height_cmd (sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml): the home height
WALK_DEAD_ZONE = .1  # GR00T runs its standing policy for a twist norm sqrt(vx^2 + vy^2 + wz^2) below this
GRASP_WINDOW_S = .3  # OmniGibson robot.py m.GRASP_WINDOW: contact and closure held this long before a grasp sticks
ASSIST_MASS_KG = 10.  # m.ASSIST_GRASP_MASS_THRESHOLD: heavier parts stick only as fixed furniture's doors and drawers
ATTACH_DISTANCE_M, ATTACH_ANGLE_DEG = .05, 15.  # object_states/attached_to.py: AttachedTo's automatic attachment
TOGGLE_HOLD_S = .15  # object_states/toggle.py m.CAN_TOGGLE_SECONDS: a finger on a toggle marker this long flips it
# dataset_object.py: every revolute and prismatic joint of a dataset object (doors, drawers, lids) gets a drive of
# stiffness 0 and damping only, so nothing springs back (a native probe: released fridge doors stay put)
FURNITURE_JOINT_STIFFNESS = 0.

# Measured (native probes and the measurement files) -------------------------------------------------------------------
WALK_MIN_PELVIS = .70  # no gait below a measured pelvis of ~0.68 m (native probes)
PUSH_STEP_N = (20, 30)  # 30 N on the torso for 1 s: the body steps 0.36-0.56 m and turns (environment-limits)
FLEXION_JOINTS = 16  # the Sharpa hand's 22 joints less its 6 abduction joints (_AA)
# Native probes (turning_on_radio, collision geometry at the live physics poses): floor reach, reach envelope and
# stand-off, squat shift and heading, postures.
FLOOR_PELVIS_COMMAND = .30  # the fingertips reach the floor only at this pelvis command or lower
FLOOR_FORWARD_M, FLOOR_WRIST_M = .35, .16  # farthest floor touch ahead of the pelvis (waist 0); lowest wrist flange
REACH_FORWARD_M = (.63, .67)  # the fingertips' farthest forward reach from the pelvis, near shoulder height
TOES_M, KNEES_SQUAT_M = (.15, .20), (.29, .36)  # how far the toes and (squatting) the knees stick out ahead
SQUAT_SHIFT_M = .045  # the pelvis moves back this far in a squat to 0.30 (the feet stay)
HEADING_PRECISION_RAD = .04  # what a turn on odometry yaw can reach (smallest lasting turn, residual after a loop)
NORMAL_TILT_DEG = 33  # the largest pelvis tilt of normal postures (deep-squat arm sweeps 32.3)


class Rule(NamedTuple):
    """A value and where the worker reads it: place is "<device>.<dotted profile field>" or "instruction.assisted";
    phrase is the value in the words that text must contain."""
    name: str
    place: str
    phrase: str


def percent(fraction):
    return f"{fraction:.0%}"


def assisted_note():
    """The instruction's note for a task set with assisted grasping (physical grasping says nothing)."""
    closure = percent(GRASP_CLOSURE)
    mass = f"{ASSIST_MASS_KG:g} kg"
    return f"""
Grasping is assisted: an object sticks to a hand when two of its finger links touch it and it lies between the
thumb tip or the palm and the fingertips, for {GRASP_WINDOW_S:g} s while the hand is closed. Closed means that the
average commanded flexion of the hand's {FLEXION_JOINTS} flexion joints (every joint but the abduction joints, _AA) is
above {closure} of their closing range: every finger counts, touching or not, so curl the fingers that do not touch
the object too (a thumb-index pinch with the other fingers straight averages about 20% and never sticks), and know
that a pre-shape with curled fingers can pick up whatever it touches. The object stays attached until that average
falls back to {closure} or below, and only then: to let go, open all the fingers, the curled ones included.
<side>_hand/attachment reports what each hand holds: none, held (attached to the hand) or tethered (on a pivot).
Parts that turn on a hinge or slide (doors, drawers, lids) stick through a pivot, so the hand stays tethered to them
and can swing or pull them; anything else lighter than {mass} sticks rigidly (a grasp on the frame of a light cabinet
or appliance welds the hand to it). A part heavier than {mass} does not stick, except a door, drawer or lid of a fixed
piece of furniture. Cabinet, drawer, microwave and fridge handles are solid lips or bars with less than 1 cm of room
behind them, so a hooked hand does not hold one: pinch the handle or the panel's edge between the thumb tip and the
fingertips, the other fingers curled, and keep it there while the hand closes; a light door that the closing hand
pushes away does not stick."""


def rules():
    """Every rule-bearing value with the words its text must contain (imports the world's command buffer). Texts are
    compared with their whitespace collapsed."""
    from rosbridge.core.commands import TRAJECTORY_MAX_S, VELOCITY_TIMEOUT_S

    low, high = (f"{v:.2f}" for v in PELVIS_LIMITS)
    clipped = "A target beyond a joint's limits is clipped to the limit" if CLIP else "A target beyond a joint's limits"
    component = "A component beyond its bound is clipped to it" if CLIP else "A component beyond its bound"
    out = [
        Rule("VELOCITY_LIMIT", "base.description", f"within GR00T's navigation limits of {VELOCITY_LIMIT:g}"),
        Rule("VELOCITY_TIMEOUT_S", "base.operations.velocity.description",
             f"stops {VELOCITY_TIMEOUT_S:g} s of world time after the last one"),
        Rule("CLIP", "base.operations.velocity.description", component),
        Rule("WALK_DEAD_ZONE", "base.description", f"is below {WALK_DEAD_ZONE:g} takes no step"),
        Rule("WALK_MIN_PELVIS", "base.description", f"pelvis at {WALK_MIN_PELVIS:.2f} m or higher"),
        Rule("WALK_MIN_PELVIS", "pelvis.description", f"walk only with the pelvis at {WALK_MIN_PELVIS:.2f} or higher"),
        Rule("PUSH_STEP_N", "base.description", f"more than about {PUSH_STEP_N[0]}-{PUSH_STEP_N[1]} N"),
        Rule("FALL_TILT_DEG", "base.description", f"tilted more than {FALL_TILT_DEG:g} degrees from upright"),
        Rule("FALL_UPRIGHT_DEG", "base.description", f"back within {FALL_UPRIGHT_DEG:g} degrees of upright"),
        Rule("FALL_HEIGHT", "base.description", f"below {FALL_HEIGHT:g} m above the floor"),
        Rule("FALL_HOLD_S", "base.description", f"spent {FALL_HOLD_S:g} s in all"),
        Rule("PELVIS_LIMITS", "pelvis.operations.joint_target.description", f"in metres ({low} to {high})"),
        Rule("PELVIS_LIMITS", "pelvis.operations.joint_target.description", f"beyond {low} to {high} is clipped"),
        Rule("PELVIS_LIMITS", "pelvis.description", f"squats down to {low}"),
        Rule("STANDING_PELVIS", "pelvis.description", f"standing is {STANDING_PELVIS:.2f}"),
        Rule("ATTACH_DISTANCE_M", "world.description", f"within {ATTACH_DISTANCE_M * 100:g} cm"),
        Rule("ATTACH_ANGLE_DEG", "world.description", f"and {ATTACH_ANGLE_DEG:g} degrees"),
        Rule("TOGGLE_HOLD_S", "world.description", f"held on the marker for {TOGGLE_HOLD_S:g} s toggles it"),
        Rule("FURNITURE_JOINT_STIFFNESS", "world.description", "have no spring"),
        Rule("DOOR_MOTION", "world.description", "move only along their joint, in one direction from closed"),
        Rule("HEIGHTS", "pelvis.description", "pelvis/floor, the floor point below the pelvis, gives heights above"),
        Rule("FLOOR_PELVIS_COMMAND", "pelvis.description", f"pelvis command {FLOOR_PELVIS_COMMAND:.2f} or lower"),
        Rule("FLOOR_FORWARD_M", "pelvis.description", f"out to about {FLOOR_FORWARD_M:g} m in front of the pelvis"),
        Rule("FLOOR_WRIST_M", "pelvis.description", f"flange stays at least {FLOOR_WRIST_M:g} m above the floor"),
        Rule("SQUAT_SHIFT_M", "pelvis.description", f"{SQUAT_SHIFT_M:g} m at 0.30"),
        Rule("HEADING_PRECISION_RAD", "base.description", f"Heading can be set to about {HEADING_PRECISION_RAD:g} rad"),
        Rule("NORMAL_TILT_DEG", "base.description", f"Normal postures stay within about {NORMAL_TILT_DEG:g} degrees"),
        Rule("HEIGHTS", "head_camera.description", "pelvis/floor gives heights above the floor"),
        Rule("GRASP_CLOSURE", "instruction.assisted", f"above {percent(GRASP_CLOSURE)} of their closing range"),
        Rule("GRASP_CLOSURE", "instruction.assisted", f"falls back to {percent(GRASP_CLOSURE)} or below"),
        Rule("GRASP_WINDOW_S", "instruction.assisted", f"for {GRASP_WINDOW_S:g} s while the hand is closed"),
        Rule("ASSIST_MASS_KG", "instruction.assisted", f"heavier than {ASSIST_MASS_KG:g} kg does not stick"),
        Rule("FLEXION_JOINTS", "instruction.assisted", f"the hand's {FLEXION_JOINTS} flexion joints"),
    ]
    for side in ("left", "right"):
        out += [Rule("REACH_FORWARD_M", f"{side}_arm.description",
                     f"fingertips get about {REACH_FORWARD_M[0]:g}-{REACH_FORWARD_M[1]:g} m forward"),
                Rule("TOES_M", f"{side}_arm.description", f"toes stick out {TOES_M[0]:.2f}-{TOES_M[1]:.2f} m"),
                Rule("KNEES_SQUAT_M", f"{side}_arm.description",
                     f"to {KNEES_SQUAT_M[0]:.2f}-{KNEES_SQUAT_M[1]:.2f} m (squatting)"),
                Rule("SELF_COLLISIONS", f"{side}_arm.description",
                     "links do not collide with each other" if not SELF_COLLISIONS else "links collide with each other")]
    for device in ("torso", "pelvis", "left_arm", "right_arm", "left_hand", "right_hand"):
        out.append(Rule("TRAJECTORY_MAX_S", f"{device}.operations.joint_target.description",
                        f"finishing within {TRAJECTORY_MAX_S:g} seconds"))
        if device != "pelvis":
            out.append(Rule("CLIP", f"{device}.operations.joint_target.description", clipped))
    return out
