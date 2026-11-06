"""scene_heat: a scene's heat sources as OmniGibson heats with them (one element each), in the world text."""
import sys
import types

import numpy as np
import pytest

from rosbridge_behavior import scene_heat

ORIGIN = (np.array([10., 5., .8]), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))  # start facing +y
FLOOR = 0.


class Link:
    def __init__(self, name, position):
        self.name, self.position = name, position

    def get_position_orientation(self):
        return np.asarray(self.position, float), np.array([0., 0., 0., 1.])


class HeatSourceOrSink:  # stand-ins for OmniGibson's state classes (the module looks them up by class)
    pass


class ToggledOn:
    pass


def state(cls, links, **fields):
    s = cls()
    s.links = {link.name: link for link in links}
    s.link = links[0] if links else None  # OmniGibson: next(iter(links)), the first meta link
    s.__dict__.update(fields)
    return s


def obj(name, category, heat=None, toggle=None, fixed=True):
    o = types.SimpleNamespace(name=name, category=category, fixed_base=fixed, states={})
    if heat is not None:
        o.states[HeatSourceOrSink] = heat
    if toggle is not None:
        o.states[ToggledOn] = toggle
    return o


@pytest.fixture
def scene(monkeypatch):
    states = types.ModuleType("omnigibson.object_states")
    states.HeatSourceOrSink, states.ToggledOn = HeatSourceOrSink, ToggledOn
    monkeypatch.setitem(sys.modules, "omnigibson", types.ModuleType("omnigibson"))
    monkeypatch.setitem(sys.modules, "omnigibson.object_states", states)
    burners = [Link("heatsource_0_link", (10.3, 6.4, .95)), Link("heatsource_1_link", (10.18, 6.4, .95)),
               Link("heatsource_2_link", (10.3, 6.0, .95)), Link("heatsource_3_link", (9.7, 6.0, .95))]
    common = dict(temperature=200., distance_threshold=.2, requires_on_fire=False)
    stove = obj("stove_abc_0", "stove", state(HeatSourceOrSink, burners, requires_toggled_on=True, requires_closed=False,
                                              requires_inside=False, **common),
                state(ToggledOn, [Link("togglebutton_0_link", (10., 5.9, 1.0))]))
    oven = obj("oven_def_0", "oven", state(HeatSourceOrSink, [], requires_toggled_on=True, requires_closed=True,
                                           requires_inside=True, **common))
    fridge = obj("fridge_ghi_0", "fridge", state(HeatSourceOrSink, [], temperature=-5., distance_threshold=.2,
                                                 requires_on_fire=False, requires_toggled_on=False,
                                                 requires_closed=True, requires_inside=True))
    log = obj("firewood_0", "firewood", state(HeatSourceOrSink, [], requires_toggled_on=False, requires_closed=False,
                                               requires_inside=False, temperature=1000., distance_threshold=.2,
                                               requires_on_fire=True))
    return types.SimpleNamespace(objects=[stove, oven, fridge, log, obj("table_0", "table")])


def test_only_the_first_element_of_a_multi_burner_stove_is_the_live_one(scene):
    found = {s["object"]: s for s in scene_heat.find(scene)}
    assert set(found) == {"stove_abc_0", "oven_def_0", "fridge_ghi_0"}  # a flammable object's own fire is not listed
    stove = found["stove_abc_0"]
    assert stove["element"] == "heatsource_0_link" and stove["element_position"] == [10.3, 6.4, .95]
    assert len(stove["other_positions"]) == 3 and stove["toggle_position"] == [10., 5.9, 1.]
    assert "element" not in found["oven_def_0"] and found["oven_def_0"]["requires_inside"]


def test_the_paragraph_names_the_live_element_where_it_is_and_what_it_needs(scene):
    text = scene_heat.describe(scene_heat.find(scene), ORIGIN, FLOOR)
    assert text.startswith(scene_heat.HEADER)
    # the live burner at world (10.3, 6.4) is 1.4 m ahead and 0.3 m to the right of a robot at (10, 5) facing +y
    # a dead burner 0.12 m from the live one is within its range (OmniGibson: the distance to the thing itself)
    assert ("stove_abc_0 (stove) heats within 0.2 m of one point, at (1.4, -0.3), 0.95 m up; its model has four such "
            "elements (burners) and the others, at (1.4, -0.2) (0.12 m from the live one, so within its range), "
            "(1.0, -0.3), (1.0, 0.3), do not heat by themselves, only while it is switched on (its switch marker at "
            "(0.9, 0.0), 1.00 m up)") in text
    assert "oven_def_0 (oven) heats only what is inside it, only while it is switched on and closed" in text
    assert "fridge_ghi_0 (fridge) cools only what is inside it, only while it is closed" in text
    assert "firewood" not in text and "table" not in text


def test_a_movable_source_says_its_element_moves_and_an_unreadable_one_is_left_out(scene):
    source = dict(scene_heat.find(scene)[0], fixed_base=False)
    assert "(it can be moved, and its element moves with it)" in scene_heat.describe([source], ORIGIN, FLOOR)
    assert scene_heat.describe([{"object": "x", "category": "stove", "error": "KeyError"}], ORIGIN, FLOOR) == ""
    assert scene_heat.describe([], ORIGIN, FLOOR) == ""


def test_a_scene_the_code_cannot_read_gives_no_paragraph(capsys):
    assert scene_heat.text(None, ORIGIN, FLOOR) == ""
    assert "scene heat sources not read" in capsys.readouterr().err
