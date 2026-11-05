"""g1_grasp: the assisted grasp's rules without a simulator (opposition, squeeze, the window)."""
import pytest

from rosbridge_behavior import g1_grasp
from rosbridge_behavior.g1_grasp import Grip, closure, digit, digits, opposed, squeezing, surface
from rosbridge_behavior.g1_rules import CLOSED_FRACTION, GRASP_EVIDENCE, GRASP_WINDOW_S, SQUEEZE_OFF_RAD, SQUEEZE_ON_RAD

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


JOINTS = ("thumb", "thumb", "index", "index", "middle", "ring", "pinky")  # the digit of each flexion joint
REST = [.5] * len(JOINTS)


def command(squeeze, measured=REST):
    """Targets that command each digit's joints squeeze[digit] beyond measured (its last joint)."""
    return [m + (squeeze.get(d, 0.) if i + 1 == len(JOINTS) or JOINTS[i + 1] != d else 0.)
            for i, (d, m) in enumerate(zip(JOINTS, measured))]


def grip():
    return Grip(DT, JOINTS)


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


def test_a_shape_the_hand_has_reached_does_not_squeeze_until_pressed_or_closed():
    # the command stays where the fingers are: touching only
    assert run(grip(), 400, PINCH, squeeze={}) is None
    # the object bends the index back SQUEEZE_ON_RAD (the R1Pro's blocked jaws under a closing command): a squeeze
    pushed = lambda k: [m - (SQUEEZE_ON_RAD if d == "index" else 0.) for d, m in zip(JOINTS, REST)]
    assert run(grip(), 400, PINCH, squeeze={}, measured=pushed) == WINDOW


def test_a_hand_closed_round_a_thin_handle_grasps_it():
    n = len(JOINTS)
    half = Grip(DT, JOINTS, limits=([0.] * n, [REST[0] / CLOSED_FRACTION] * n))  # REST is CLOSED_FRACTION closed
    assert run(half, 200, PINCH, squeeze={}) == WINDOW  # the targets are reached: no squeeze, but closed
    less = Grip(DT, JOINTS, limits=([0.] * n, [2.] * n))  # REST is a quarter closed: a shape that only touches
    assert run(less, 200, PINCH, squeeze={}) is None
    moving = Grip(DT, JOINTS, limits=([0.] * n, [REST[0] / CLOSED_FRACTION] * n))
    lag = (SQUEEZE_OFF_RAD + SQUEEZE_ON_RAD) / 2  # closed, but still closing (neither settled nor squeezing)
    assert run(moving, 200, PINCH, squeeze={"thumb": lag, "index": lag}) is None
    still_closing = Grip(DT, JOINTS, limits=([0.] * n, [REST[0] / CLOSED_FRACTION] * n))
    assert not any(still_closing.observe(TARGET, lambda t: PINCH, lambda t: True, REST, REST, closing=True)
                   for _ in range(200))  # the command still rising: only a squeeze grips


def test_the_rules_module_is_free_of_the_simulator():
    assert "omnigibson" not in g1_grasp.__dict__ and "torch" not in g1_grasp.__dict__
