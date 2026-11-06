"""g1_grasp: the assisted grasp's rules without a simulator (opposition, squeeze, the window, the hold, the load)."""
import pytest

from rosbridge_behavior import g1_grasp
from rosbridge_behavior.g1_grasp import (Grip, clamp, closure, digit, digits, excess, leaving_velocity, opposed,
                                        phalanx, squeezing, surface)
from rosbridge_behavior.g1_rules import (CLOSED_FRACTION, GRASP_EVIDENCE, GRASP_WINDOW_S, GRIP_CLAMP_RAD, LET_GO_S,
                                         LOAD_HOLD_S, SQUEEZE_OFF_RAD, SQUEEZE_ON_RAD)

DT = .005  # the G1's physics step
TARGET = "/World/scene_0/glass/base_link"


def test_a_digit_is_the_thumb_or_a_finger_and_its_pad_belongs_to_it():
    assert digit("/World/scene_0/robot_g1/right_index_elastomer") == digit("right_index_DP") == "index"
    assert digit("right_hand_C_MC") is None and digit("right_eef_link") is None
    assert digits(["right_thumb_MC", "right_thumb_PP", "right_thumb_elastomer"]) == {"thumb"}


def test_the_thumb_and_the_palm_oppose_the_fingers():
    assert [surface(link) for link in ("right_thumb_DP", "right_thumb_MC", "left_hand_C_MC")] == ["thumb"] * 3
    assert [surface(link) for link in ("right_index_DP", "right_middle_elastomer", "left_pinky_PP", "left_pinky_MC")] == [
        "finger"] * 4
    assert surface("right_eef_link") is None and surface("/World/scene_0/glass/base_link") is None


@pytest.mark.parametrize("links, expected", [
    (["right_thumb_elastomer", "right_index_elastomer"], True),  # a pinch
    (["right_hand_C_MC", "right_middle_PP", "right_ring_PP"], True),  # a power grasp against the palm
    (["right_thumb_PP", "right_thumb_MC", "right_hand_C_MC"], False),  # thumb and palm are one side
    (["right_index_DP", "right_middle_DP"], False),  # scissoring
    (["right_ring_DP", "right_pinky_DP", "right_pinky_MP"], False),  # a ring and pinky brush
    (["right_index_DP", "right_index_elastomer"], False),  # one fingertip, pad and bone
    (["right_pinky_MC", "right_pinky_PP", "right_ring_DP"], False),  # ring and pinky curled onto the pinky's base
    ([], False),
])
def test_opposition_needs_both_sides(links, expected):
    assert opposed(links) is expected


def test_a_touching_digit_squeezes_when_commanded_beyond_where_it_is():
    links = ["right_thumb_DP", "right_index_DP"]
    assert squeezing(links, {"index": SQUEEZE_ON_RAD, "thumb": 0.})
    assert not squeezing(links, {"index": SQUEEZE_ON_RAD - .01, "thumb": .0})
    assert not squeezing(links, {"middle": 1.})  # the squeezing digit does not touch
    assert not squeezing(["right_hand_C_MC"], {"index": 1.})  # the palm has no joint to squeeze with


def test_a_closed_touching_digit_grips_without_squeezing():
    links = ["right_thumb_DP", "right_index_DP"]
    assert closure(("thumb", "thumb", "index"), [.2, .4, 1.], [0., 0., 0.], [1., 1., 2.]) == pytest.approx(
        {"thumb": .3, "index": .5})
    assert squeezing(links, {}, closed={"index": CLOSED_FRACTION})
    assert not squeezing(links, {}, closed={"index": CLOSED_FRACTION - .01, "middle": 1.})


def test_a_held_hand_presses_lightly_and_opens_freely():
    assert clamp(1.2, .4) == pytest.approx(.4 + GRIP_CLAMP_RAD)
    assert clamp(.42, .4) == .42
    assert clamp(0., .4) == 0.


JOINTS = ("thumb", "thumb", "index", "index", "middle", "ring", "pinky")  # the digit of each flexion joint
REST = [.5] * len(JOINTS)


def command(squeeze, measured=REST):
    """Targets that command each digit's joints squeeze[digit] beyond measured (its last joint)."""
    return [m + (squeeze.get(d, 0.) if i + 1 == len(JOINTS) or JOINTS[i + 1] != d else 0.)
            for i, (d, m) in enumerate(zip(JOINTS, measured))]


def grip(load_limit=None):
    return Grip(DT, JOINTS, load_limit)


def run(grip, steps, links, on_ray=True, squeeze=None, candidate=TARGET, measured=None):
    """Steps of the window with the same observation; returns the step (from 1) that grasped, or None."""
    squeeze = {"thumb": .5, "index": .5} if squeeze is None else squeeze
    for k in range(steps):
        links_k = links(k) if callable(links) else links
        q = REST if measured is None else measured(k)
        if grip.observe(candidate if links_k else None, lambda target: links_k, lambda target: on_ray,
                        command(squeeze, REST), q):
            return k + 1
    return None


PINCH = ["right_thumb_elastomer", "right_index_elastomer"]
WINDOW = round(GRASP_WINDOW_S / DT)


def test_a_steady_squeezed_pinch_grasps_after_the_window():
    assert run(grip(), 200, PINCH) == WINDOW


@pytest.mark.parametrize("links, squeeze, on_ray", [
    (["right_thumb_PP", "right_thumb_MC"], None, True),  # thumb only
    (["right_index_DP", "right_middle_DP"], None, True),  # scissoring
    (PINCH, {"thumb": 0., "index": 0.}, True),  # shaped, not squeezing (a curled pre-shape that touches)
    (PINCH, None, False),  # not between the thumb or palm and the fingertips
])
def test_no_grasp_without_opposition_squeeze_and_ray(links, squeeze, on_ray):
    assert run(grip(), 400, links, on_ray, squeeze) is None


def test_a_flickering_finger_still_grasps():
    # the index touches on 2 of every 3 steps (a thin lip); the thumb steadily
    flicker = lambda k: PINCH if k % 3 else PINCH[:1]
    assert run(grip(), 400, flicker) == WINDOW


def test_a_finger_touching_less_than_the_evidence_share_does_not_grasp():
    sparse = lambda k: PINCH if k % 4 == 0 else PINCH[:1]  # 25% of the steps
    assert GRASP_EVIDENCE > .25 and run(grip(), 400, sparse) is None


def test_sides_that_touch_one_after_the_other_do_not_grasp():
    # thumb alone for the first half of the window, then the index alone: never both within GRASP_RECENT_S
    alternate = lambda k: PINCH[:1] if (k // (WINDOW // 2)) % 2 == 0 else PINCH[1:]
    assert run(grip(), 400, alternate) is None


def test_the_window_restarts_on_a_new_target_and_on_reset():
    g = grip()
    assert run(g, WINDOW - 1, PINCH) is None
    assert run(g, WINDOW - 1, PINCH, candidate="/World/scene_0/cup/base_link") is None  # restarted
    g.reset()  # e.g. the hand opening
    assert run(g, WINDOW - 1, PINCH) is None and run(g, 1, PINCH) == 1


def keep(g, squeeze, load=0., fingers_on=lambda: False):
    return g.keep(command(squeeze), REST, load, fingers_on)


def test_a_shape_the_hand_has_reached_does_not_squeeze_until_pressed_or_closed():
    # the command stays where the fingers are: touching only
    assert run(grip(), 400, PINCH, squeeze={}) is None
    # the object bends the index back SQUEEZE_ON_RAD (the R1Pro's blocked jaws under a closing command): a squeeze
    pushed = lambda k: [m - (SQUEEZE_ON_RAD if d == "index" else 0.) for d, m in zip(JOINTS, REST)]
    assert run(grip(), 400, PINCH, squeeze={}, measured=pushed) == WINDOW


def test_a_hand_that_handed_its_object_over_grasps_again_only_after_opening():
    g = grip()
    g.yielded = True
    assert run(g, 400, PINCH) is None
    g.yielded = False  # the handler clears it when the hand opens
    assert run(g, WINDOW, PINCH) == WINDOW


def test_a_held_grasp_lets_go_when_the_squeeze_ends_with_hysteresis():
    g = grip()
    assert run(g, WINDOW, PINCH) == WINDOW
    g.hold()
    assert g.held == {"thumb", "index"}
    assert keep(g, {"thumb": SQUEEZE_OFF_RAD + 1e-6, "index": -.2}) is None  # one gripping digit still squeezes
    assert keep(g, {"thumb": (SQUEEZE_ON_RAD + SQUEEZE_OFF_RAD) / 2}) is None  # between OFF and ON
    assert keep(g, {"thumb": SQUEEZE_OFF_RAD - .01, "middle": 1.}) == "squeeze"  # the middle does not grip it
    assert g.why == "squeeze"


def test_a_hand_closed_round_a_thin_handle_grasps_and_holds_it_until_opened():
    n = len(JOINTS)
    half = Grip(DT, JOINTS, None, ([0.] * n, [REST[0] / CLOSED_FRACTION] * n))  # REST is CLOSED_FRACTION closed
    assert run(half, 200, PINCH, squeeze={}) == WINDOW  # the targets are reached: no squeeze, but closed
    less = Grip(DT, JOINTS, None, ([0.] * n, [2.] * n))  # REST is a quarter closed: a shape that only touches
    assert run(less, 200, PINCH, squeeze={}) is None
    moving = Grip(DT, JOINTS, None, ([0.] * n, [REST[0] / CLOSED_FRACTION] * n))
    lag = (SQUEEZE_OFF_RAD + SQUEEZE_ON_RAD) / 2  # closed, but still closing (neither settled nor squeezing)
    assert run(moving, 200, PINCH, squeeze={"thumb": lag, "index": lag}) is None
    still_closing = Grip(DT, JOINTS, None, ([0.] * n, [REST[0] / CLOSED_FRACTION] * n))
    assert not any(still_closing.observe(TARGET, lambda t: PINCH, lambda t: True, REST, REST, closing=True)
                   for _ in range(200))  # the command still rising: only a squeeze grips
    half.hold()
    assert half.keep(REST, REST, 0.) is None
    assert half.keep([r - .1 for r in REST], REST, 0.) == "squeeze"  # opened below CLOSED_FRACTION, no squeeze


def test_a_hand_that_grips_an_object_now_can_take_it_over_at_once():
    n = len(JOINTS)
    g = Grip(DT, JOINTS, None, ([0.] * n, [REST[0] / CLOSED_FRACTION] * n))  # REST is closed
    assert g.grips(PINCH, REST, REST)
    assert not g.grips(["right_index_DP", "right_middle_DP"], REST, REST)  # one side only
    assert not Grip(DT, JOINTS, None, ([0.] * n, [2.] * n)).grips(PINCH, REST, REST)  # a quarter closed, no squeeze
    g.hold({"thumb", "index"})
    assert g.held == {"thumb", "index"} and g.keep(REST, REST, 0.) is None


def test_a_held_grasp_lets_go_when_the_gripping_digits_are_commanded_open():
    from rosbridge_behavior.g1_rules import OPEN_RAD
    g = grip()
    run(g, WINDOW, PINCH)
    g.hold()
    tight = [m + .5 for m in REST]
    assert g.keep(tight, REST, 0.) is None
    # opened by less than OPEN_RAD while the object holds the fingers out: still held
    assert g.keep([c - OPEN_RAD / 2 for c in tight], REST, 0.) is None
    # opened by OPEN_RAD (the object still holds a finger out beyond the new command): let go
    assert g.keep([c - OPEN_RAD for c in tight], REST, 0.) == "open"


def test_the_moving_phalanges_are_what_pushes_an_object():
    assert all(phalanx(f"right_{name}") for name in ("thumb_PP", "thumb_DP", "index_MP", "pinky_DP"))
    assert not any(phalanx(f"right_{name}") for name in ("thumb_MC", "pinky_MC", "index_MCP_VL", "hand_C_MC", "eef"))


def test_an_opened_hand_lets_go_once_its_fingers_have_left_the_object():
    g = grip()
    run(g, WINDOW, PINCH)
    g.hold()
    opened = {"thumb": 0., "index": 0.}  # the squeeze has ended
    on = [True] * 5 + [False]
    assert [keep(g, opened, 0., fingers_on=lambda: on.pop(0)) for _ in range(6)] == [None] * 5 + ["squeeze"]
    g.let_go()
    assert g.let_after_s == pytest.approx(5 * DT)
    run(g, WINDOW, PINCH)
    g.hold()
    steps = [keep(g, opened, 0., fingers_on=lambda: True) for _ in range(round(2 * LET_GO_S / DT))]
    assert steps.index("squeeze") == round(LET_GO_S / DT)  # fingers that stay on it: let go after LET_GO_S
    g.let_go()
    run(g, WINDOW, PINCH)
    g.hold()
    assert keep(g, opened, 0., fingers_on=lambda: True) is None
    assert keep(g, {"thumb": .5}, 0., fingers_on=lambda: True) is None and g.letting_s == 0.  # squeezed again


def test_a_grasp_restored_from_a_saved_state_is_held_by_any_squeezing_digit():
    g = grip()
    assert keep(g, {"ring": .3}) is None and keep(g, {"ring": 0.}) == "squeeze"


def test_the_load_limit_ignores_spikes_and_ends_a_sustained_overload():
    limit, squeeze = 100., {"thumb": .5, "index": .5}
    g = grip(limit)
    run(g, WINDOW, PINCH)
    g.hold()
    for k in range(400):  # 4 steps of 2000 N every 0.5 s over a 60 N pull
        assert keep(g, squeeze, 2000. if k % 100 < 4 else 60.) is None
    for _ in range(round(2 / DT)):
        assert keep(g, squeeze, .99 * limit) is None  # just under the limit, held for 2 s
    ended = next(k for k in range(1000) if keep(g, squeeze, 1.5 * limit))
    assert g.why == "load" and LOAD_HOLD_S / DT <= ended + 1 <= (LOAD_HOLD_S + .1) / DT
    g.let_go()
    run(g, WINDOW, PINCH)
    g.hold()
    assert all(keep(g, squeeze, float("nan")) is None for _ in range(200))  # an unreadable load never counts
    assert all(keep(g, squeeze, None) is None for _ in range(200))  # a tether: no load limit


def test_no_load_limit_keeps_any_load():
    g = grip()
    run(g, WINDOW, PINCH)
    g.hold()
    assert all(keep(g, {"thumb": .5}, 1e6) is None for _ in range(1000))


def test_the_rules_module_is_free_of_the_simulator():
    assert "omnigibson" not in g1_grasp.__dict__ and "torch" not in g1_grasp.__dict__


def test_a_released_object_falls_or_rests_but_is_not_flung():
    from rosbridge_behavior.g1_rules import RELEASE_MARGIN_MPS as m
    assert leaving_velocity((0., 0., -.98), .1) == (0., 0., -.98)  # falling out of the hand: unchanged
    assert leaving_velocity((0., 0., 0.), .2) == (0., 0., 0.)  # resting in it: unchanged
    x, y, z = leaving_velocity((2., 0., 1.5), .1)  # pushed sideways and up by the fingers: cut to the margin
    assert (x, y, z) == pytest.approx((m, 0., m))
    assert leaving_velocity((0., 0., -5.), .1)[2] == pytest.approx(-(9.81 * .1 + m))  # shot downward: cut to falling


# A digit an object pushes back past its open pose. The Sharpa fingers' MCP joints bend 0.17 rad backward
# (the thumb's 0.52): measured from where the joint is, a finger pushed back to -0.17 and commanded open (0) read as a
# 0.17 rad squeeze, so an open hand kept holding a light pinch (its tightest command below OPEN_RAD, so "open" never
# came) and an open hand pressed onto a handle grabbed it. From the open pose it reads none.
OPEN = [0.] * len(JOINTS)
PUSHED_BACK = -.17  # a finger MCP's lower limit


def light_pinch(open_pose):
    """A light pinch held (the gripping digits commanded 0.15 rad, below OPEN_RAD), the index pushed back to its limit."""
    from rosbridge_behavior.g1_rules import OPEN_RAD
    g = Grip(DT, JOINTS, None, None, open_pose)
    tight = [.15 if d in ("thumb", "index") else 0. for d in JOINTS]
    measured = [PUSHED_BACK if d == "index" else .1 if d == "thumb" else 0. for d in JOINTS]
    assert .15 < OPEN_RAD
    g.hold({"thumb", "index"})
    assert g.keep(tight, measured, 0.) is None
    return g, measured


def test_a_pushed_back_finger_commanded_open_kept_the_grasp_when_measured_from_where_it_is():
    g, measured = light_pinch(None)  # the previous measure: the defect
    assert all(g.keep(OPEN, measured, 0.) is None for _ in range(round(1. / DT)))  # opened, held for 1 s and on


def test_a_pushed_back_finger_commanded_open_lets_go_within_let_go_s():
    g, measured = light_pinch(OPEN)
    steps = [g.keep(OPEN, measured, 0., fingers_on=lambda: True) for _ in range(round(2 * LET_GO_S / DT))]
    assert "squeeze" in steps and steps.index("squeeze") <= round(LET_GO_S / DT)
    g, measured = light_pinch(OPEN)  # opened below the open pose (a client's "fingers to the lower limit")
    assert g.keep([-.17] * len(JOINTS), measured, 0.) == "squeeze"


def test_an_open_hand_pressed_onto_a_handle_does_not_grab_it():
    # every digit commanded open, the handle pushing the index back to its limit, thumb and index touching it
    pressed = lambda k: [PUSHED_BACK if d == "index" else 0. for d in JOINTS]
    for open_pose, grabs in ((None, True), (OPEN, False)):  # measured from where it is (the defect), from the open pose
        g = Grip(DT, JOINTS, None, None, open_pose)
        grasped = any(g.observe(TARGET, lambda t: PINCH, lambda t: True, OPEN, pressed(k)) for k in range(400))
        assert grasped is grabs


def test_a_finger_pushed_back_but_commanded_closed_still_grips():
    g = Grip(DT, JOINTS, None, None, OPEN)
    closing = [.3] * len(JOINTS)
    measured = [PUSHED_BACK if d == "index" else .3 for d in JOINTS]  # the object holds the index back past open
    assert g.squeeze(closing, measured)["index"] == pytest.approx(.3)  # counted from the open pose, not from -0.17
    assert any(g.observe(TARGET, lambda t: PINCH, lambda t: True, closing, measured) for _ in range(400))
    g.hold({"thumb", "index"})
    assert g.keep(closing, measured, 0.) is None
    # a joint short of its open pose (flexed by the object) counts from where it is, as before
    assert excess(JOINTS, [.5] * len(JOINTS), [.2] * len(JOINTS), OPEN)["index"] == pytest.approx(.3)


def test_a_tether_to_a_loose_objects_part_is_let_go_by_load_and_one_to_fixed_furniture_is_not():
    from rosbridge_behavior.g1_grasp import load_limited
    assert load_limited(tethered=False, fixed_base=False) and load_limited(tethered=False, fixed_base=True)  # held
    assert load_limited(tethered=True, fixed_base=False)  # a bin's or a toolbox's lid
    assert not load_limited(tethered=True, fixed_base=True)  # a cabinet's door, a room door: never by load
    assert not load_limited(tethered=True, fixed_base=False, loose_tether_load=False)  # the previous rule
    # the handler passes the load only where it applies (None otherwise): a loose lid pulled at 1.5x the limit lets go
    # after LOAD_HOLD_S, as a held object does; a fixed door's tether keeps any load
    limit = 150.
    for fixed, ends in ((False, True), (True, False)):
        g = grip(limit)
        run(g, WINDOW, PINCH)
        g.hold()
        load = 1.5 * limit if load_limited(True, fixed) else None
        steps = [keep(g, {"thumb": .5, "index": .5}, load) for _ in range(round(1. / DT))]
        assert ("load" in steps) is ends
        if ends:  # the filter (LOAD_TAU_S) passes the limit after ~0.11 s, then LOAD_HOLD_S
            assert LOAD_HOLD_S / DT <= steps.index("load") + 1 <= (LOAD_HOLD_S + .2) / DT
