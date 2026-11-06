"""The G1's assisted grasp as plain rules, free of the simulator so that unit tests can drive them.

BEHAVIOR's reference robot, the R1Pro, grasps with two opposing jaws: an object is attached when both jaws touch it,
it lies between them, and the gripper is commanded to close, for 0.3 s; it stays attached until the gripper is
commanded open. On the Sharpa hand the same idea reads:
- opposition: the object touches the thumb or the palm on one side and one of the four fingers on the other
  (two fingers side by side, or a finger's pad and bone, oppose nothing);
- grip: a digit touching it is commanded to flex further than it is, or than its open pose where the object pushes it
  back past open (position drives turn that excess into grip force, as the R1Pro's closing command does on its blocked
  jaws; a digit commanded open never grips), or is commanded at least half closed and has reached
  its targets (a hand closed round a thin handle presses nothing); a little-curled hand that only touches a thing, a
  hand still closing, or an opening hand, does not grip;
- both hold over OmniGibson's 0.3 s window, judged on most of its steps so that flickering contacts on a thin lip or
  handle still count;
- held, it stays attached until the fingers that grip it stop gripping or are commanded open, and then until they have
  left it (so that they do not push it as they open), or until the load on the grasp stays above what a hand can grip
  for a sustained time.
g1_sharpa feeds these functions from OmniGibson's contacts, rays and joint states once per physics step.
"""
from collections import deque

from .g1_rules import (CLOSED_FRACTION, DIGITS, GRASP_EVIDENCE, GRASP_RECENT_S, GRASP_WINDOW_S, GRIP_CLAMP_RAD,
                       LET_GO_S, LOAD_HOLD_S, LOAD_LIMIT_N, LOAD_TAU_S, OPEN_RAD, RELEASE_MARGIN_MPS, SQUEEZE_OFF_RAD,
                       SQUEEZE_ON_RAD)

GRAVITY = 9.81  # m/s^2

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


def phalanx(link):
    """Whether a hand link is a moving phalanx of a digit (proximal, middle or distal: what pushes an object when the
    digit moves), not a metacarpal, the palm or a virtual link."""
    return digit(link) is not None and link_name(link).rsplit("_", 1)[-1] in ("PP", "MP", "DP")


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


def excess(joint_digits, command, reference, open_pose=None):
    """Per digit, the most one of its flexion joints is commanded beyond a reference position (rad); with open_pose
    (each joint's open position), beyond the reference or the open pose, whichever is more flexed, so that a joint an
    object pushes back past open counts only what its command flexes it past open (SQUEEZE_FROM_OPEN_POSE)."""
    out = {}
    for i, (d, c, r) in enumerate(zip(joint_digits, command, reference)):
        if open_pose is not None:
            r = max(r, open_pose[i])
        out[d] = max(out.get(d, float("-inf")), c - r)
    return out


def clamp(command, measured):
    """The position target a flexion joint gets while its hand holds an object: no more than GRIP_CLAMP_RAD beyond
    where it is, so a finger blocked by the held object presses with a bounded force instead of driving into it (and
    no depenetration flings the object when it is let go); opening commands pass unchanged."""
    return min(command, measured + GRIP_CLAMP_RAD)


def leaving_velocity(relative, seconds):
    """A released object's velocity relative to the hand (x, y, z; z up), limited while it still touches the hand:
    sideways and upward no faster than RELEASE_MARGIN_MPS, downward no faster than falling for `seconds` gives plus
    that margin. Falling out of the hand, or resting in it, is unchanged; a push from the contacts it was squeezed with
    (depenetration, fingers that keep closing) is cut, so it does not fly."""
    x, y, z = relative
    sideways = (x * x + y * y) ** .5
    scale = min(1., RELEASE_MARGIN_MPS / sideways) if sideways > 0 else 1.
    return x * scale, y * scale, max(-(GRAVITY * seconds + RELEASE_MARGIN_MPS), min(z, RELEASE_MARGIN_MPS))


class Grip:
    """One hand's assisted grasp: the evidence window while a grasp forms, then the hold until it is let go.

    dt is the physics step; GRASP_WINDOW_S of steps make the window. Each step records whether the target lies on a
    ray between the thumb or palm and the fingertips, touches the thumb side, touches the finger side, and is squeezed
    by a touching digit. A grasp forms once the window is full, each of the four held on GRASP_EVIDENCE of its steps,
    and all four held together on one of the last GRASP_RECENT_S (the grasp is complete now, not only on average).
    joint_digits names the digit of each flexion joint, in the order of the command and measured sequences the methods
    take."""

    def __init__(self, dt, joint_digits, load_limit=LOAD_LIMIT_N, limits=None, open_pose=None):
        self.dt, self.joint_digits, self.load_limit = dt, tuple(joint_digits), load_limit
        self.limits = limits  # the flexion joints' (lower, upper) limits, for the commanded closure; None: squeeze only
        # the flexion joints' open positions: a squeeze counts from the open pose when an object holds a joint back past
        # it (SQUEEZE_FROM_OPEN_POSE); None: from where the joint is
        self.open_pose = tuple(open_pose) if open_pose is not None else None
        self.steps = max(1, round(GRASP_WINDOW_S / dt))
        self.recent = max(1, round(GRASP_RECENT_S / dt))
        self.window = deque(maxlen=self.steps)
        self.target = None  # the link the window is about
        self.touched = deque(maxlen=self.recent)  # the digits that touched it on the last steps
        self.held = None  # the digits gripping a held object, None while nothing is held
        self.load = self.over_s = 0.
        self.why = None  # why the last grasp ended: "squeeze", "open" or "load"
        self.tightest = None  # the gripping digits' tightest mean command while held
        self.yielded = False  # it handed its object to the other hand and grasps again only after opening
        self.letting_s = 0.  # how long the hand has been opened while a finger still touches the held object

    def squeeze(self, command, measured):
        """Per digit, its squeeze (excess) beyond where it is, or beyond its open pose where an object holds it back
        past open."""
        return excess(self.joint_digits, command, measured, getattr(self, "open_pose", None))

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
        if self.yielded:
            return None
        if candidate is not None and candidate != self.target:
            self.reset()
            self.target = candidate
        if self.target is None:
            return None
        squeeze = self.squeeze(command, measured)
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

    def grips(self, links, command, measured):
        """Whether the touching hand links grip an object now: opposition, and a touching digit squeezes or is closed
        (the window's conditions on one step, less the ray)."""
        squeeze = self.squeeze(command, measured)
        return opposed(links) and squeezing(links, squeeze, closed=self.closed(command, squeeze))

    def hold(self, gripping=None):
        """The grasp was made: the given digits grip it, else those that touched the target on the last steps."""
        self.held = set(gripping) if gripping else set().union(*self.touched) if self.touched else set()
        self.load = self.over_s = 0.
        self.tightest = None
        self.reset()

    def keep(self, command, measured, load_n, fingers_on=lambda: False):
        """One physics step while holding. Returns why the grasp ends now, else None (keep holding): "squeeze" when
        no gripping digit is commanded SQUEEZE_OFF_RAD beyond its measured flexion nor CLOSED_FRACTION closed (the
        hand opened), "open" when they are commanded OPEN_RAD open from their tightest, "load" when the load on the
        grasp (N; None for a tether, which has no load limit), low-pass filtered with LOAD_TAU_S, has stayed above the
        load limit for LOAD_HOLD_S. An opened hand
        keeps the object while fingers_on() says a phalanx still touches it, for LET_GO_S at most: it is let go once
        the fingers have left it (or just begin to), so they do not push it as they open. A grasp OmniGibson restored
        from a saved state is gripped by every digit."""
        held = DIGITS if self.held is None else self.held
        squeeze = self.squeeze(command, measured)
        gripping = [c for d, c in zip(self.joint_digits, command) if d in held] or list(command)
        mean = sum(gripping) / len(gripping)
        self.tightest = mean if getattr(self, "tightest", None) is None else max(self.tightest, mean)
        self.why = None
        closed = self.closed(command)
        if not any(squeeze.get(d, float("-inf")) >= SQUEEZE_OFF_RAD or closed.get(d, 0.) >= CLOSED_FRACTION
                   for d in held):
            self.why = "squeeze"
        elif mean <= self.tightest - OPEN_RAD:
            self.why = "open"
        elif self.load_limit is not None and load_n is not None and load_n == load_n:  # None (a tether), NaN: no limit
            self.load += (load_n - self.load) * self.dt / (LOAD_TAU_S + self.dt)
            self.over_s = self.over_s + self.dt if self.load > self.load_limit else 0.
            if self.over_s >= LOAD_HOLD_S - 1e-9:
                self.why = "load"
        if self.why is None:
            self.letting_s = 0.
        elif self.why != "load" and self.letting_s < LET_GO_S - 1e-9 and fingers_on():
            self.letting_s += self.dt
            return None  # opened, and a finger still touches it
        return self.why

    def let_go(self):
        self.held = self.tightest = None
        self.let_after_s, self.letting_s = self.letting_s, 0.  # (how long the fingers took to leave it)
        self.load = self.over_s = 0.
        self.reset()
