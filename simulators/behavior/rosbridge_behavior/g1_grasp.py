"""The G1's assisted grasp as plain rules, free of the simulator so that unit tests can drive them.

BEHAVIOR's reference robot, the R1Pro, grasps with two opposing jaws: an object is attached when both jaws touch it,
it lies between them, and the gripper is commanded to close, for 0.3 s; it stays attached until the gripper is
commanded open. On the Sharpa hand the same idea reads:
- opposition: the object touches the thumb or the palm on one side and one of the four fingers on the other
  (two fingers side by side, or a finger's pad and bone, oppose nothing);
- grip: a digit touching it is commanded to flex further than it is (position drives turn that excess into grip
  force, as the R1Pro's closing command does on its blocked jaws), or is commanded at least half closed and has reached
  its targets (a hand closed round a thin handle presses nothing); a little-curled hand that only touches a thing, a
  hand still closing, or an opening hand, does not grip;
- both hold over OmniGibson's 0.3 s window, judged on most of its steps so that flickering contacts on a thin lip or
  handle still count.
"""
from collections import deque

from .g1_rules import (CLOSED_FRACTION, DIGITS, GRASP_EVIDENCE, GRASP_RECENT_S, GRASP_WINDOW_S, SQUEEZE_OFF_RAD,
                       SQUEEZE_ON_RAD)

# The palm (name after "<side>_"): it opposes the fingers as the thumb does, but has no joint to squeeze with. The
# thumb's metacarpal (the ball of the thumb) is a thumb link; the pinky's metacarpal is a pinky link, so a pinky curled
# onto its own base opposes nothing.
PALM = ("hand_C_MC",)


def link_name(path):
    return path.rsplit("/", 1)[-1]


def digit(link):
    """The digit a hand link (prim path or name) belongs to ("right_index_DP" -> "index"; a pad and its phalanx are
    one digit), or None for the palm and any other link."""
    parts = link_name(link).split("_")
    return parts[1] if len(parts) > 2 and parts[1] in DIGITS else None


def digits(links):
    """The distinct digits among hand links."""
    return {digit(link) for link in links} - {None}


def surface(link):
    """Which side of a grasp a hand link is on: "thumb" (the thumb or the palm) or "finger" (index, middle, ring,
    pinky), None for any other link."""
    name = link_name(link).split("_", 1)[-1]
    if name in PALM or digit(link) == "thumb":
        return "thumb"
    return "finger" if digit(link) else None


def opposed(links):
    """Whether hand links touch an object from opposite sides: the thumb or palm, and a finger."""
    sides = {surface(link) for link in links}
    return "thumb" in sides and "finger" in sides


def squeezing(links, squeeze, margin=SQUEEZE_ON_RAD, closed=None):
    """Whether a digit among the touching links grips: it squeezes by at least margin (rad), or it is commanded at
    least CLOSED_FRACTION closed (a hand closed round a thin handle reaches its targets and presses nothing, as the
    R1Pro's jaws closed on an object are still commanded closed). squeeze maps a digit to the most one of its flexion
    joints is commanded beyond its reference position, closed to its commanded closure (see Grip)."""
    closed = closed or {}
    return any(squeeze.get(d, float("-inf")) >= margin or closed.get(d, 0.) >= CLOSED_FRACTION for d in digits(links))


def closure(joint_digits, command, lower, upper):
    """Per digit, the mean commanded flexion of its joints as a fraction of their ranges (0 open, 1 fully flexed)."""
    sums = {}
    for d, c, lo, hi in zip(joint_digits, command, lower, upper):
        sums.setdefault(d, []).append(min(1., max(0., (c - lo) / max(hi - lo, 1e-6))))
    return {d: sum(v) / len(v) for d, v in sums.items()}


def excess(joint_digits, command, reference):
    """Per digit, the most one of its flexion joints is commanded beyond a reference position (rad)."""
    out = {}
    for d, c, r in zip(joint_digits, command, reference):
        out[d] = max(out.get(d, float("-inf")), c - r)
    return out


class Grip:
    """One hand's assisted grasp while it forms: the evidence window.

    dt is the physics step; GRASP_WINDOW_S of steps make the window. Each step records whether the target lies on a
    ray between the thumb or palm and the fingertips, touches the thumb side, touches the finger side, and is squeezed
    by a touching digit. A grasp forms once the window is full, each of the four held on GRASP_EVIDENCE of its steps,
    and all four held together on one of the last GRASP_RECENT_S (the grasp is complete now, not only on average).
    joint_digits names the digit of each flexion joint, in the order of the command and measured sequences the methods
    take."""

    def __init__(self, dt, joint_digits, limits=None):
        self.dt, self.joint_digits = dt, tuple(joint_digits)
        self.limits = limits  # the flexion joints' (lower, upper) limits, for the commanded closure; None: squeeze only
        self.steps = max(1, round(GRASP_WINDOW_S / dt))
        self.recent = max(1, round(GRASP_RECENT_S / dt))
        self.window = deque(maxlen=self.steps)
        self.target = None  # the link the window is about
        self.touched = deque(maxlen=self.recent)  # the digits that touched it on the last steps

    def reset(self):
        self.window.clear()
        self.touched.clear()
        self.target = None

    def observe(self, candidate, touching, on_ray, command, measured, closing=False):
        """One physics step without a grasp. candidate: the link OmniGibson's search picks (touched, on a ray, closest
        to the hand), or None; touching: the hand links (finger links and palm) touching the window's target, after it
        is chosen (a callable from link to links); on_ray: whether a ray hits that target (a callable); command and
        measured: the flexion joints' targets and positions (rad); closing: whether the commanded closure is still
        rising (a closing hand grips only by squeezing; closed digits count once it has stopped closing). Returns the
        target once the window shows a grasp, else None."""
        if candidate is not None and candidate != self.target:
            self.reset()
            self.target = candidate
        if self.target is None:
            return None
        squeeze = excess(self.joint_digits, command, measured)
        links = touching(self.target)
        sides = {surface(link) for link in links}
        self.touched.append(digits(links))
        self.window.append((on_ray(self.target), "thumb" in sides, "finger" in sides,
                            squeezing(links, squeeze, closed={} if closing else self.closed(command, squeeze))))
        if len(self.window) < self.steps:
            return None
        seen = list(self.window)
        mostly = all(sum(step[i] for step in seen) >= GRASP_EVIDENCE * self.steps for i in range(4))
        return self.target if mostly and any(all(step) for step in seen[-self.recent:]) else None

    def closed(self, command, squeeze=None):
        """Per digit, its commanded closure; with squeeze (a grasp forming), only the digits that have reached their
        targets (within SQUEEZE_OFF_RAD): a closing hand's moving fingers, or fingers the object blocks (a squeeze),
        do not count as closed round it until they have settled."""
        if self.limits is None:
            return {}
        closed = closure(self.joint_digits, command, *self.limits)
        return closed if squeeze is None else {d: c for d, c in closed.items() if squeeze.get(d, 0.) <= SQUEEZE_OFF_RAD}
