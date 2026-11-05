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
CLIP = True  # targets beyond a limit act at the limit (the G1 layout clips; a refusal is reported only in the next wait's reply)
SELF_COLLISIONS = False  # the robot's links do not collide with each other (OmniGibson robot config)
# A hand applies a grasp while its flexion joints (all but the abduction joints, _AA) are commanded past this fraction
# of their closing range on average; it releases as soon as the average is back at or below it.
GRASP_CLOSURE = .3
# An assisted grasp needs contact on this many distinct digits; a finger link belongs to the digit its name gives
# (<side>_<digit>_..., the fingertip pads included), and the palm is no digit. A thumb with the palm, or one fingertip's
# pad and phalanx, is one digit (such contacts used to attach as "two finger links").
DIGITS, GRASP_DIGITS = ("thumb", "index", "middle", "ring", "pinky"), 2
# PhysX breaks an assisted grasp's joint beyond this force (N) and torque (GRASP_BREAK_N * GRASP_BREAK_TORQUE_M, N m),
# and the hand lets go; None: unbreakable, as OmniGibson's. ROSBRIDGE_GRASP_BREAK_N overrides it (g1_sharpa).
GRASP_BREAK_N, GRASP_BREAK_TORQUE_M = None, .1
# Assisted grasping by opposition (g1_grasp; BEHAVIOR's R1Pro attaches an object both jaws touch while the gripper
# closes).
# A digit squeezes while one of its flexion joints (all but the abduction joints, _AA) is commanded at least
# SQUEEZE_ON_RAD beyond its measured position; a held object stays attached while a gripping digit is commanded at least
# SQUEEZE_OFF_RAD beyond (the lower value keeps a grip from chattering).
SQUEEZE_ON_RAD, SQUEEZE_OFF_RAD = .15, .05
# A touching digit also grips while it is commanded at least CLOSED_FRACTION of its flexion range closed (the mean over
# its flexion joints) and, while a grasp forms, the hand has stopped closing and the digit has reached its targets
# (within SQUEEZE_OFF_RAD): a hand closed round a thin handle reaches its targets and squeezes nothing, yet holds it, as
# the R1Pro's jaws commanded closed on an object do. Counting closed digits while the hand still closed made grasps form
# earlier than squeezing ones (the native grasp suite: a glass attached before its 12 kg mass was set).
CLOSED_FRACTION = .5
# A grasp forms when, over OmniGibson's window (GRASP_WINDOW_S), the object lay on a ray, touched the thumb or palm,
# touched a finger and was squeezed, each on at least GRASP_EVIDENCE of the window's steps and within the last
# GRASP_RECENT_S (contacts on thin lips and handles flicker from step to step).
GRASP_EVIDENCE, GRASP_RECENT_S = .5, .05
# A held object is also let go once the gripping digits are commanded open: their joints' mean command OPEN_RAD below
# the tightest it was while held (an object can hold a finger out beyond a partly opened command, still a squeeze).
OPEN_RAD = .2
# An opened hand lets go of the object once its fingers have left it (no phalanx touches it), LET_GO_S after it opened
# at the latest, so that the opening fingers do not push it out (the R1Pro's jaws are clear when it lets go). Measured
# (native grasp suite): holding on until the fingers had left (0.5 s) let the opening fingers press into the still
# attached object and fling it when let go (mug wraps 1.3-3.7 m/s); 0.08 s lets go while they only begin to part.
LET_GO_S = .08
# While a hand holds an object, each flexion joint's position target is at most GRIP_CLAMP_RAD beyond its measured
# position, so the fingers press lightly instead of driving into the held object (which flung it when let go).
GRIP_CLAMP_RAD = .05
# A held object (not a tether) is let go when the load the arm passes to the hand (the palm link's incoming joint force,
# in which the fingers' squeeze cancels), low-pass filtered with LOAD_TAU_S, stays above LOAD_LIMIT_N for LOAD_HOLD_S;
# None: no load limit. Tethers to doors, drawers and lids are never let go by load, as OmniGibson's joints never break:
# the 10.8 kg room door door_tonlzz pressed by the closing hand passed the limit within its close and lost its tether in
# 11 of 12 trials (native grasp suite), while the legged-robot reasons for limiting tethers are gone (a body
# held up by a hand on a handle is physics; two hands never hold one object, HAND_OVER).
LOAD_LIMIT_N, LOAD_TAU_S, LOAD_HOLD_S = 150., .1, .3
# A released object leaves at the hand's velocity; while it still touches the hand (RELEASE_FOLLOW_S at most) it moves
# from the hand no faster than RELEASE_MARGIN_MPS sideways or up, and no faster down than falling since the release gives
# plus that margin, so the contacts it was squeezed with (fingers, palm) separate it from the hand without flinging it.
RELEASE_FOLLOW_S, RELEASE_MARGIN_MPS = .6, .1
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
# dataset_object.py also gives each of those joints PhysX's joint friction coefficient (constants.py: revolute 0.2,
# prismatic 0.3), which PhysX applies to the joint's load, so the force that starts a door, drawer or lid moving grows
# with the part's weight (the measured forces are below).
FURNITURE_JOINT_FRICTION = (.2, .3)  # revolute, prismatic

# Measured (native probes and the measurement files) -------------------------------------------------------------------
WALK_MIN_PELVIS = .70  # no gait below a measured pelvis of ~0.68 m (native probes)
# A steady horizontal force at the hands (arms forward at about 1.07 m, held 2 s, standing;
# furniture_forces.json "hands"): PUSH_STEP_N[0] pushing them back or [1] pulling them forward makes the body step away
# (the next lower force tried did not, pulling); a steady PULL_FALL_N pull made it fall. (30 N on the torso for 1 s:
# steps of 0.36-0.56 m.)
PUSH_STEP_N, PULL_FALL_N = (10, 20), 30
FLEXION_JOINTS = 16  # the Sharpa hand's 22 joints less its 6 abduction joints (_AA)
# A held object's weight on the arm (assisted grasping; the arms' drives carry it, their own weight compensated): a 2 kg
# part sagged the hand ARM_PAYLOAD[1]-[2] m in 2 s in four holds, and 9.9 kg made the robot stagger 0.7-0.8 m (native
# probes).
ARM_PAYLOAD = (2, .14, .33)
# Native probes (turning_on_radio, collision geometry at the live physics poses): floor reach, reach envelope and
# stand-off, squat shift and heading, postures.
FLOOR_PELVIS_COMMAND = .30  # the fingertips reach the floor only at this pelvis command or lower
FLOOR_FORWARD_M, FLOOR_WRIST_M = .35, .16  # farthest floor touch ahead of the pelvis (waist 0); lowest wrist flange
REACH_FORWARD_M = (.63, .67)  # the fingertips' farthest forward reach from the pelvis, near shoulder height
TOES_M, KNEES_SQUAT_M = (.15, .20), (.29, .36)  # how far the toes and (squatting) the knees stick out ahead
SQUAT_SHIFT_M = .045  # the pelvis moves back this far in a squat to 0.30 (the feet stay)
HEADING_PRECISION_RAD = .04  # what a turn on odometry yaw can reach (smallest lasting turn, residual after a loop)
NORMAL_TILT_DEG = 33  # the largest pelvis tilt of normal postures (deep-squat arm sweeps 32.3)
# The waist pitch GR00T settles at (rad, steady over 1-5 s) for a target: with the pelvis at 0.60, a 0.52 target; in a
# squat (pelvis command 0.45 and below, the arm not raised overhead), targets 0, 0.3 and 0.52. The pelvis pitches
# forward instead (reach2_postures.json, reach_conditions.json).
WAIST_AT_060 = (.20, .34)
WAIST_SQUAT = {0.: (-.39, -.16), .3: (.08, .18), .52: (.15, .25)}
# Standing (pelvis command 0.74), the forward reach of the palm (the grasp point between thumb and fingers) from the
# point midway between the ankles, per waist pitch target once the waist has settled, at hand heights WAIST_REACH_AT_M
# (reach_envelope.json: IK targets 0.95 and 1.1 m, measured in the simulator).
WAIST_REACH_M = {0.: .50, .3: .60, .52: .66}
WAIST_REACH_AT_M = (.95, 1.15)
# Walking up to furniture and leaning at it (native device-path attempts, approach_reach.json;
# positions in the foot frame). In the arms' home pose (the profile's home, held at episode start) the fingertips are
# HOME_TIP_M[0] ahead of the ankle midpoint at HOME_TIP_M[1] above the floor. Walking up in that pose to the places where
# the object rests at APPROACH_PLACE_MIN_Z or higher (furniture tops HOME_WALK_TOPS_M), the walk stopped more than
# APPROACH_STALL_M short at HOME_WALK_STALL[0] of [1] places, by [2]-[3]; at each of those the best walk with the palms
# raised (0.20 m ahead of and 0.05 m above the shoulders) stopped within RAISED_WALK_M. Standing there (pelvis 0.74) with a
# 0.52 waist target: of the attempts whose measured waist pitch stayed at WAIST_HELD or more through every reach stage,
# WAIST_HELD_STEP_BACK[0] of [1] ended more than APPROACH_STALL_M farther from the target ([2]-[3] m), against [4] of [5]
# of the attempts whose waist settled lower.
APPROACH_PLACE_MIN_Z, APPROACH_STALL_M, WAIST_HELD = .6, .1, .45  # the selection, not measured values
HOME_TIP_M = (.45, .89)
HOME_WALK_TOPS_M = (.69, 1.08)
HOME_WALK_STALL = (11, 15, .14, .29)
RAISED_WALK_M = .03
WAIST_HELD_STEP_BACK = (21, 28, .14, .74, 3, 21)
# Furniture under OmniGibson's friction (furniture_forces.json; constant horizontal forces held 1 s at the handle with
# the robot held still; force ramps along a sliding door's track;
# hand pulls with GR00T). Held for 1 s at the handle, a hinged furniture door of DOOR_TURN_N[i][0]
# to [i][1] kg turns 0.3 rad with [i][2] to [i][3] N (every measured door in a band; a handle point within 0.15 m of an
# upright hinge, where the force grows as the lever shrinks, is left out); a drawer comes out 5 cm with DRAWER_PULL_N; a room
# door of ROOM_DOOR_TURN_N[i][:2] kg needs [i][2:] N; an 18 kg sliding door starts along its track at SLIDING_DOOR_N[0]
# and slides 0.1 m in 1 s with [1].
DOOR_TURN_N = ((0, 1, 2, 5), (1, 4, 7, 20), (4, 7, 30, 50))
DRAWER_PULL_N = (7, 15)
ROOM_DOOR_TURN_N = ((7, 8.5, 30, 30), (9.5, 10.5, 30, 50), (11, 15, 80, 80))
SLIDING_DOOR_N = (23, 80)
# A heavy room door pulled toward the robot by a held handle (physical pinch or assisted attachment): the most it turned
# while the robot walked backward, and while the arm pulled in (degrees).
DOOR_PULL_DEG = (.4, 19)
# Walking chest-first (forearms tucked, 0.25 m/s commanded, heading held) into a closed room door that swings away from
# the robot (30 s): doors of CHEST_PUSH[0]-[1] kg reached 30 degrees in [2]-[3] s.
CHEST_PUSH = (10, 15, 15, 29)
# Parts on level hinges whose weight turns them: released part-way open, they fell open or shut within 3 s.
LEVEL_HINGE_FALLS = ("oven", "dishwasher", "grill", "car")


def _handle_rows(forces):
    """One row per door joint: where two probes measured the same joint, the one with the finer force grid."""
    best = {}
    for row in forces["handle"]:
        key = (row["task"], row["object"], row["joint"])
        if key not in best or len(row["grid_N"]) > len(best[key]["grid_N"]):
            best[key] = row
    return list(best.values())


def _span(values):
    values = list(values)
    return (min(values), max(values))


def _furniture_doors(forces):
    """Per stated weight band: the forces that turned the hinged furniture doors (not room doors) measured in it."""
    rows = [(r["mass_kg"], r["turn_0.3_1s_N"]) for r in _handle_rows(forces) if r["type"].startswith("Revolute")
            and r["kind"] not in ("room door", "garage door") and r["turn_0.3_1s_N"]
            and (r["lever_m"] >= .15 or r["kind"].endswith("(level hinge)"))]
    bands = tuple((lo, hi, *_span(f for m, f in rows if lo <= m < hi)) for lo, hi, _, _ in DOOR_TURN_N)
    uncovered = [m for m, _ in rows if not any(lo <= m < hi for lo, hi, _, _ in DOOR_TURN_N)]
    return bands + ((("not covered by a stated band", uncovered),) if uncovered else ())


def _room_doors(forces):
    """Per stated weight band (kg, to the nearest 0.5): the weights and forces of the room doors measured in it (handle
    rows; the edge rows, whose free edge is within a few cm of a room door's handle)."""
    rows = [(r["mass_kg"], r["turn_0.3_1s_N"]) for r in _handle_rows(forces) if r["kind"] == "room door" and r["turn_0.3_1s_N"]]
    rows += [(r["mass"], r["force_0.3rad_1s_N"]) for r in forces["edge"] if "mass" in r]
    bands = []
    for lo, hi, _, _ in ROOM_DOOR_TURN_N:
        seen = [(round(mass * 2) / 2, force) for mass, force in rows if lo <= round(mass * 2) / 2 <= hi]
        bands.append((*_span(m for m, _ in seen), *_span(f for _, f in seen)))
    uncovered = [m for m, _ in rows if not any(lo <= round(m * 2) / 2 <= hi for lo, hi, _, _ in ROOM_DOOR_TURN_N)]
    return tuple(bands) + ((("not covered by a stated band", uncovered),) if uncovered else ())


def _sliding(forces):
    row = next(r for r in forces["edge"] if r["object"].startswith("sliding_door") and "force_0.1m_1s_N" in r)
    return (int(row["breakaway_N"]), row["force_0.1m_1s_N"])


def _hand_step(forces):
    rows = forces["hands"]
    push = min(-r["force_N"] for r in rows if r["force_N"] < 0 and r["stepped"])
    pull = min(r["force_N"] for r in rows if r["force_N"] > 0 and r["stepped"])
    held = max((r["force_N"] for r in rows if 0 < r["force_N"] and not r["stepped"]), default=0.)
    return (round(push), round(pull)) if held < pull else ("a pull of", held, "did not step")


def _chest_push(forces):
    rows = [r for r in forces["push"] if r["probe"].startswith("pushdoor_long") and r["posture"] == "chest"]
    reached = [r for r in rows if r["seconds_to_30deg"] is not None]
    if len(reached) < len(rows) or not rows:
        return ("did not reach 30 degrees", [r["door"] for r in rows if r not in reached])
    return (round(min(r["mass_kg"] for r in rows)), round(max(r["mass_kg"] for r in rows)),
            round(min(r["seconds_to_30deg"] for r in rows)), round(max(r["seconds_to_30deg"] for r in rows)))


def _door_pull(forces):
    rows = [r for r in forces["pulls"] if r["object"].startswith("door_")]
    walk = [r["opened_deg"] for r in rows if r["pull"] in ("walk_back",)]
    arm = [r["opened_deg"] for r in rows if r["pull"] in ("arm_retract", "arm_pull")]
    return (round(max(walk), 1), round(max(arm)))


def _nearest(value, step):
    return round(round(value / step) * step, 6)


def _standing_reach(envelope):
    """Per waist pitch target, standing: the palm points measured for the 0.95 and 1.1 m targets."""
    return {p["waist_pitch_cmd"]: [r["grasp"] for r in p["reach"] if r["z_target"] in (.95, 1.1)]
            for p in envelope["postures"] if p["pelvis_cmd"] == STANDING_PELVIS and p["waist_pitch_cmd"] in WAIST_REACH_M}


def _approach_places(approach):
    return [p for p in approach["places"] if p["rests_at_z"] >= APPROACH_PLACE_MIN_Z]


def _home_tip(approach):
    home = approach["arm_poses_foot_frame"]["home"]
    return tuple(_nearest((home["left_tip"][i] + home["right_tip"][i]) / 2, .01) for i in (0, 2))


def _home_stall(approach):
    """Places where every home-arm walk stopped more than APPROACH_STALL_M short: how many, and how short."""
    places = _approach_places(approach)
    stalled = [p for p in places if p["home_arm_walks_short_m"] and min(p["home_arm_walks_short_m"]) > APPROACH_STALL_M]
    shorts = [s for p in stalled for s in p["home_arm_walks_short_m"]]
    return (len(stalled), len(places), _nearest(min(shorts), .01), _nearest(max(shorts), .01))


def _raised_walk(approach):
    """At the places where the home-arm walk stalled, how close the best raised-arm walk came (the farthest of them)."""
    places = _approach_places(approach)
    stalled = [p for p in places if p["home_arm_walks_short_m"] and min(p["home_arm_walks_short_m"]) > APPROACH_STALL_M]
    return _nearest(max(min(abs(s) for s in p["raised_arm_walks_short_m"]) for p in stalled), .01)


def _waist_step_back(approach):
    """Standing 0.52 waist attempts at those places: stepped back (more than APPROACH_STALL_M) with the waist held at
    WAIST_HELD or more through the reach, and with it settled lower."""
    rows = [r for r in approach["standing_waist_0.52"] if r["rests_at_z"] >= APPROACH_PLACE_MIN_Z and r["waist_pitch_during_reach"]]
    held = [r for r in rows if min(r["waist_pitch_during_reach"]) >= WAIST_HELD]
    settled = [r for r in rows if min(r["waist_pitch_during_reach"]) < WAIST_HELD]
    back = [r["moved_back_m"] for r in held if r["moved_back_m"] > APPROACH_STALL_M]
    return (len(back), len(held), _nearest(min(back), .01), _nearest(max(back), .01),
            sum(r["moved_back_m"] > APPROACH_STALL_M for r in settled), len(settled))


def _waist_squat(postures, conditions):
    out = {}
    for target in WAIST_SQUAT:
        seen = [p["waist"] for c in postures["conditions"] if c["pelvis_cmd"] <= .45 and c["waist_cmd"] == target
                for p in c["postures"] if p["shoulder_pitch"] > -2.6]
        seen += [c["waist_pitch_measured"] for c in conditions["conditions"]
                 if c["pelvis_cmd"] <= .45 and c["waist_pitch_cmd"] == target]
        out[target] = (round(min(seen), 2), round(max(seen), 2))
    return out


# Each measured value with the stored measurement it comes from (simulators/behavior-g1/measurements/, copies of the
# probes' outputs cited above) and how it is read from it; the unit tests recompute every value from its files, so a
# text cannot state a value its measurement does not support.
MEASURED = {
    "FLOOR_PELVIS_COMMAND": (("floor_band.json",), lambda band: max(
        c["pelvis_cmd"] for c in band if c["lowest_tip_z"] <= .005)),
    "FLOOR_FORWARD_M": (("floor_band.json",), lambda band: _nearest(max(
        c["tip_within_2cm_max_x"] or 0. for c in band if c["waist_cmd"] == 0.), .05)),
    "FLOOR_WRIST_M": (("floor_band.json",), lambda band: round(min(c["lowest_flange_z"] for c in band), 2)),
    "SQUAT_SHIFT_M": (("locomotion_summary.json",), lambda loco: _nearest(-sum(
        r["squat"][0] for r in loco["m3"] if r["pelvis_cmd"] == .3) / sum(r["pelvis_cmd"] == .3 for r in loco["m3"]), .005)),
    "NORMAL_TILT_DEG": (("reach_conditions.json",), lambda reach: -int(-max(
        c["sweep_tilt_max_deg"] for c in reach["conditions"]) // 1)),
    "WAIST_AT_060": (("reach2_postures.json",), lambda postures: (lambda seen: (round(min(seen), 2), round(max(seen), 2)))(
        [p["waist"] for c in postures["conditions"] if c["pelvis_cmd"] == .6 and c["waist_cmd"] == .52
         for p in c["postures"]])),
    "WAIST_SQUAT": (("reach2_postures.json", "reach_conditions.json"), _waist_squat),
    "WAIST_REACH_M": (("reach_envelope.json",), lambda envelope: {
        waist: _nearest(max(g[0] for g in points), .01) for waist, points in _standing_reach(envelope).items()}),
    "WAIST_REACH_AT_M": (("reach_envelope.json",), lambda envelope: (lambda z: (_nearest(min(z), .05), _nearest(max(z), .05)))(
        [g[2] for points in _standing_reach(envelope).values() for g in points])),
    "HOME_TIP_M": (("approach_reach.json",), _home_tip),
    "HOME_WALK_TOPS_M": (("approach_reach.json",), lambda approach: (lambda z: (_nearest(min(z), .01), _nearest(max(z), .01)))(
        [p["rests_at_z"] for p in _approach_places(approach)])),
    "HOME_WALK_STALL": (("approach_reach.json",), _home_stall),
    "RAISED_WALK_M": (("approach_reach.json",), _raised_walk),
    "WAIST_HELD_STEP_BACK": (("approach_reach.json",), _waist_step_back),
    "DOOR_TURN_N": (("furniture_forces.json",), _furniture_doors),
    "DRAWER_PULL_N": (("furniture_forces.json",), lambda f: _span(
        r["move_0.05_1s_N"] for r in _handle_rows(f) if r["kind"] == "drawer")),
    "ROOM_DOOR_TURN_N": (("furniture_forces.json",), _room_doors),
    "SLIDING_DOOR_N": (("furniture_forces.json",), _sliding),
    "DOOR_PULL_DEG": (("furniture_forces.json",), _door_pull),
    "PUSH_STEP_N": (("furniture_forces.json",), _hand_step),
    "CHEST_PUSH": (("furniture_forces.json",), _chest_push),
    "PULL_FALL_N": (("furniture_forces.json",), lambda f: round(min(r["force_N"] for r in f["hands"] if r["force_N"] > 0 and r["fell"]))),
    "LEVEL_HINGE_FALLS": (("furniture_forces.json",), lambda f: tuple(
        part for part in LEVEL_HINGE_FALLS if any(part in r["object"] and max(map(abs, r["drift_after_3s_deg_from"].values()))
                                                  >= 10. for r in f["level"]))),
}


class Rule(NamedTuple):
    """A value and where the worker reads it: place is "<device>.<dotted profile field>" or "instruction.assisted";
    phrase is the value in the words that text must contain."""
    name: str
    place: str
    phrase: str


def _bands(bands):
    """'30 N at 7-8.5 kg, 30-50 N at 9.5-10.5 kg and 80 N at 11-15 kg'"""
    said = [(f"{f0:g}" if f0 == f1 else f"{f0:g}-{f1:g}") + f" N at {m0:g}-{m1:g} kg" for m0, m1, f0, f1 in bands]
    return ", ".join(said[:-1]) + " and " + said[-1]


def percent(fraction):
    return f"{fraction:.0%}"


def assisted_note():
    """The instruction's note for a task set with assisted grasping (physical grasping says nothing)."""
    closure = percent(GRASP_CLOSURE)
    mass = f"{ASSIST_MASS_KG:g} kg"
    if GRASP_BREAK_N is None:
        load = ("An attachment does not break: a held object pressed into a wall, or held by both hands at once, carries "
                "the arms' full force.")
    else:
        load = (f"An attachment breaks, and the hand lets go, when its load passes {GRASP_BREAK_N:g} N (a hang on it, a "
                f"press into a wall, two hands pulling one object apart).")
    return f"""
Grasping is assisted: an object sticks to a hand when at least {_number(GRASP_DIGITS)} digits touch it (the thumb, index,
middle, ring and pinky; the palm is no digit, and a fingertip's pad and bone are one digit) and it lies between the
thumb tip or the palm and the fingertips, for {GRASP_WINDOW_S:g} s while the hand is closed and not opening. Closed
means that the average commanded flexion of the hand's {FLEXION_JOINTS} flexion joints (every joint but the abduction
joints, _AA) is above {closure} of their closing range: every finger counts, touching or not, so curl the fingers that
do not touch the object too (a thumb-index pinch with the other fingers straight averages about 20% and never sticks),
and know that a pre-shape with curled fingers can pick up whatever two of its digits touch. Contact is checked only as
the grasp forms. The object stays attached until that average falls back to {closure} or below, and only then: to let
go, open all the fingers, the curled ones included; the object leaves with the hand's own motion. {load}
<side>_hand/attachment reports what each hand holds: none, held (fixed to the hand) or tethered (on a pivot), and
<side>_hand/attached_object names it as <object>/<link>. A part on a hinge or slide of its object (a door, drawer or
lid, also a jar's lid or an appliance's window) sticks through a pivot, so the hand stays tethered to it and can swing
or pull it; a loose object lighter than {mass} sticks rigidly. Nothing fixed in place sticks: walls, floors, counters,
shelves and the bodies of furniture and appliances (a cabinet, a microwave, a door frame, a switch) never attach. A
part heavier than {mass} does not stick, except a door, drawer or lid of a fixed piece of furniture. The arm carries a
held object's weight: {ARM_PAYLOAD[0]:g} kg sagged the hand {ARM_PAYLOAD[1]:g}-{ARM_PAYLOAD[2]:g} m, and 9.9 kg made the robot stagger.
Cabinet, drawer, microwave and fridge handles are solid lips or bars with less than 1 cm of room behind them, so a
hooked hand does not hold one: pinch the handle or the panel's edge between the thumb tip and the fingertips, the other
fingers curled, and keep it there while the hand closes; a light door that the closing hand pushes away does not
stick."""


def _number(n):
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))


def physical_note():
    """The instruction's note for a task set with physical grasping."""
    return """
Grasping is physical: an object is held only by the contact and friction of the fingers and palm, as a real hand holds
it; nothing attaches to a hand, and <side>_hand/attachment always reports none."""


def rules():
    """Every rule-bearing value with the words its text must contain (imports the world's command buffer). Texts are
    compared with their whitespace collapsed."""
    from rosbridge.core.commands import TRAJECTORY_MAX_S, VELOCITY_TIMEOUT_S
    from rosbridge.runtime.waits import ended
    from rosbridge.runtime.world import END_CAUSES, REFUSED
    from . import scene_doors

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
        Rule("PUSH_STEP_N", "base.description",
             f"a steady force at the hands of about {PUSH_STEP_N[0]} N pushing them back (pressing on furniture, a door or "
             f"a wall) or {PUSH_STEP_N[1]} N pulling them forward makes the body step away"),
        Rule("PULL_FALL_N", "base.description", f"a steady {PULL_FALL_N} N pull made it fall"),
        Rule("HOME_TIP_M", "base.description",
             f"In the arms' home pose, which an episode starts in, the fingertips are {HOME_TIP_M[0]:.2f} m ahead of the "
             f"point midway between the ankles, {HOME_TIP_M[1]:.2f} m above the floor"),
        Rule("HOME_WALK_TOPS_M", "base.description",
             f"walking up to furniture with tops {HOME_WALK_TOPS_M[0]:.2f}-{HOME_WALK_TOPS_M[1]:.2f} m high in that pose"),
        Rule("HOME_WALK_STALL", "base.description",
             f"the body stopped {HOME_WALK_STALL[2]:.2f}-{HOME_WALK_STALL[3]:.2f} m short of its goal at "
             f"{HOME_WALK_STALL[0]} of {HOME_WALK_STALL[1]} places"),
        Rule("RAISED_WALK_M", "base.description",
             f"the same walk with the palms raised above the top got within {RAISED_WALK_M:.2f} m of it"),
        Rule("WAIST_HELD_STEP_BACK", "base.description",
             "Standing at such furniture with a 0.52 waist target, the body stepped back {2:.2f}-{3:.2f} m in {0} of {1} "
             "attempts in which the waist held at {held:.2f} or more through the arm's forward reach, against {4} of {5} "
             "in which it settled back".format(*WAIST_HELD_STEP_BACK, held=WAIST_HELD)),
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
        Rule("DOOR_RELEASE", "world.description", "if it moves, part of the robot is touching it"),
        Rule("DOOR_PINCH", "world.description", "a pinch catches only a closed or held door"),
        Rule("DOOR_MOTION", "world.description", "move only along their joint, in one direction from closed"),
        Rule("PARTIAL_SCENE", "world.description",
             "Only the rooms the activity uses are loaded, as in BEHAVIOR's own evaluation, and past them there may be no floor"),
        Rule("SCENE_EDGE", "world.description",
             f"a door may open onto empty space (a room that is not loaded, or the outside), and then {scene_doors.BLOCKED}"),
        Rule("SCENE_DOORS", "world.description",
             "this description ends with the scene's room doors, where each is, which way it swings and which lead out of "
             "the loaded rooms"),
        Rule("FURNITURE_JOINT_FRICTION", "world.description", "resist with a friction that grows with the part's weight"),
        Rule("DOOR_TURN_N", "world.description",
             "Held for 1 s at the handle, a door of under {1} kg (most cabinet doors) turns 0.3 rad with about {f0}, one of "
             "{2}-{3} kg (a fridge or microwave door, a large cabinet door) with {f1} and one of {4}-{5} kg (an oven's "
             "drop-down door, a large fridge door) with {f2}".format(
                 *(f"{x:g}" for x in (DOOR_TURN_N[0][0], DOOR_TURN_N[0][1], DOOR_TURN_N[1][0], DOOR_TURN_N[1][1],
                                       DOOR_TURN_N[2][0], DOOR_TURN_N[2][1])),
                 **{f"f{i}": f"{b[2]:g}-{b[3]:g} N" for i, b in enumerate(DOOR_TURN_N)})),
        Rule("DRAWER_PULL_N", "world.description",
             f"a drawer comes out 5 cm with about {DRAWER_PULL_N[0]:g}-{DRAWER_PULL_N[1]:g} N"),
        Rule("ROOM_DOOR_TURN_N", "world.description", f"a room door needs about {_bands(ROOM_DOOR_TURN_N)}"),
        Rule("SLIDING_DOOR_N", "world.description",
             f"an 18 kg sliding door needs about {SLIDING_DOOR_N[0]:g} N along its track before it moves at all and "
             f"{SLIDING_DOOR_N[1]:g} N to slide 0.1 m in 1 s"),
        Rule("PUSH_STEP_N", "world.description",
             f"a steady {PUSH_STEP_N[0]} N pushing back on its hands or {PUSH_STEP_N[1]} N pulling them forward makes it "
             f"step away"),
        Rule("DOOR_PULL_DEG", "world.description",
             f"turned one at most {DOOR_PULL_DEG[0]:g} degrees as the robot walked backward and {DOOR_PULL_DEG[1]:g} "
             f"degrees as the arm pulled in"),
        Rule("CHEST_PUSH", "world.description",
             f"while one that swings away from the robot opens slowly to a walk into it chest first, forearms tucked (doors "
             f"of {CHEST_PUSH[0]}-{CHEST_PUSH[1]} kg turned 30 degrees in {CHEST_PUSH[2]}-{CHEST_PUSH[3]} s)"),
        Rule("LEVEL_HINGE_FALLS", "world.description",
             "(an oven's or a dishwasher's door, a grill lid, a car's trunk) carry their own weight, so released part-way "
             "open they can fall open or shut"),
        Rule("HEIGHTS", "pelvis.description", "pelvis/floor, the floor point below the pelvis, gives heights above"),
        Rule("FLOOR_PELVIS_COMMAND", "pelvis.description", f"pelvis command {FLOOR_PELVIS_COMMAND:.2f} or lower"),
        Rule("FLOOR_FORWARD_M", "pelvis.description", f"out to about {FLOOR_FORWARD_M:g} m in front of the pelvis"),
        Rule("FLOOR_WRIST_M", "pelvis.description", f"flange stays at least {FLOOR_WRIST_M:g} m above the floor"),
        Rule("SQUAT_SHIFT_M", "pelvis.description", f"{SQUAT_SHIFT_M:g} m at 0.30"),
        Rule("HEADING_PRECISION_RAD", "base.description", f"Heading can be set to about {HEADING_PRECISION_RAD:g} rad"),
        Rule("NORMAL_TILT_DEG", "base.description", f"Normal postures stay within about {NORMAL_TILT_DEG:g} degrees"),
        Rule("HEIGHTS", "head_camera.description", "pelvis/floor gives heights above the floor"),
        Rule("WAIT_END", "world.operations.wait.description", ended(0.).replace("0.00", "T")),
        Rule("END_CAUSE", "world.operations.wait.description", f"world time T s because {END_CAUSES['success']}"),
        Rule("END_CAUSE", "world.operations.wait.description", f"because {END_CAUSES['max_steps']}"),
        Rule("REFUSED", "world.operations.wait.description", f'after "{REFUSED}"'),
        Rule("REFUSED", "world.description", "a refused one is named, with its reason, in the reply to the next world wait"),
        Rule("REFUSED", "base.operations.velocity.description",
             "the reply to the next world wait names the device and the reason"),
        Rule("WAIST_AT_060", "torso.description",
             f"with the pelvis at 0.60 a 0.52 target reaches only {WAIST_AT_060[0]:.2f}-{WAIST_AT_060[1]:.2f}"),
        *(Rule("WAIST_SQUAT", "torso.description", f"{target:g} at {low:.2f} to {high:.2f}")
          for target, (low, high) in WAIST_SQUAT.items()),
        Rule("WAIST_REACH_M", "torso.description",
             f"the palm reaches {WAIST_REACH_M[0.]:.2f} m ahead of the point midway between the ankles with the waist at 0, "
             f"{WAIST_REACH_M[.3]:.2f} m at 0.3 and {WAIST_REACH_M[.52]:.2f} m at 0.52"),
        Rule("WAIST_REACH_AT_M", "torso.description",
             f"at hand heights of {WAIST_REACH_AT_M[0]:.2f}-{WAIST_REACH_AT_M[1]:.2f} m"),
        Rule("WAIST_SETTLED", "torso.description", "once the measured waist pitch has settled at its target"),
        Rule("GRASP_CLOSURE", "instruction.assisted", f"above {percent(GRASP_CLOSURE)} of their closing range"),
        Rule("GRASP_CLOSURE", "instruction.assisted", f"falls back to {percent(GRASP_CLOSURE)} or below"),
        Rule("GRASP_WINDOW_S", "instruction.assisted", f"for {GRASP_WINDOW_S:g} s while the hand is closed"),
        Rule("ASSIST_MASS_KG", "instruction.assisted", f"heavier than {ASSIST_MASS_KG:g} kg does not stick"),
        Rule("FLEXION_JOINTS", "instruction.assisted", f"the hand's {FLEXION_JOINTS} flexion joints"),
        Rule("GRASP_DIGITS", "instruction.assisted", f"at least {_number(GRASP_DIGITS)} digits touch it"),
        Rule("DIGITS", "instruction.assisted", "(the thumb, index, middle, ring and pinky; the palm is no digit"),
        Rule("GRASP_BREAK_N", "instruction.assisted", "An attachment does not break" if GRASP_BREAK_N is None else
             f"when its load passes {GRASP_BREAK_N:g} N"),
        Rule("ARM_PAYLOAD", "instruction.assisted", f"{ARM_PAYLOAD[0]:g} kg sagged the hand {ARM_PAYLOAD[1]:g}-{ARM_PAYLOAD[2]:g} m"),
        Rule("ATTACHED_OBJECT", "instruction.assisted", "<side>_hand/attached_object names it as <object>/<link>"),
        Rule("ATTACH_EVERY_STEP", "world.description", "at every step in which the object touches the other"),
        Rule("ATTACH_SNAP", "world.description", "the object snaps into exact alignment"),
        Rule("PHYSICAL_GRASPING", "instruction.physical", "nothing attaches to a hand"),
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
        out.append(Rule("REFUSED", f"{device}.operations.joint_target.description",
                        "(one without points, say) is refused (nothing of it is executed), and the reply to the next "
                        "world wait names the device and the reason"))
        if device != "pelvis":
            out.append(Rule("CLIP", f"{device}.operations.joint_target.description", clipped))
    return out
