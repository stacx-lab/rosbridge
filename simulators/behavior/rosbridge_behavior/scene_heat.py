"""The heat sources and sinks of a loaded BEHAVIOR scene, read from OmniGibson's own object states, for the world text.

OmniGibson's HeatSourceOrSink keeps ONE heating element per object: the first of the object's heat-source meta links
(LinkBasedStateMixin.link is next(iter(links)); heat_source_or_sink.py stores that link's pose index alone). A source
heats what comes within its distance threshold (m.DEFAULT_DISTANCE_THRESHOLD, 0.2 m, unless the object's model sets
another) of that point, measured to the nearest point of the target's own geometry; a source that requires it heats only
what is inside it (an oven, a microwave), and a cold source (a fridge, a freezer) cools in the same way. Heat passes
only from a source to a target: a pot or pan on a burner does not heat the food in it. A stove or cooktop with several
burners therefore heats at one burner alone, and nothing a client sees says which. This module states it per scene;
the physics is unchanged (it only reads the states and link poses).

Per source (find): its name and category, whether it heats or cools, its range, whether it needs to be switched on,
closed or the target inside it, the live element's link and position, how many heat-source elements its model has, and
the toggle marker's position. describe() writes the paragraph in the robot's episode-start frame (base/odom), as the
scene's room doors are written (scene_doors); backend writes it after the doors into the run directory's scene text,
which a task runner may add to the world device's description."""
import math
import sys
import traceback

import numpy as np

AMBIENT_C = 23.  # OmniGibson's ambient temperature (temperature.py m.DEFAULT_TEMPERATURE): hotter heats, colder cools
HEADER = ("Heat sources of this scene, from OmniGibson's own heating rule: each appliance heats at ONE element only, "
          "the first in its model, so a stove or cooktop with several burners heats at one burner alone; a source heats "
          "what comes within its range of that element (measured to the nearest point of the thing itself), or, where "
          "stated, only what is inside it, and a pot or pan does not pass heat on, so the food itself must come that "
          "close. Positions in base/odom as above, heights above the floor the robot starts on")


def _values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _position(link):
    return _values(link.get_position_orientation()[0])


def find(scene):
    """Every heat source or sink of the loaded scene that is an appliance (not a flammable object's own fire): dicts
    with lists, for JSON."""
    from omnigibson.object_states import HeatSourceOrSink, ToggledOn

    out = []
    for obj in scene.objects:
        state = obj.states.get(HeatSourceOrSink)
        if state is None or getattr(state, "requires_on_fire", False):
            continue
        try:
            out.append(_source(obj, state, obj.states.get(ToggledOn)))
        except Exception as error:  # a source the code cannot read is reported, not described
            out.append({"object": obj.name, "category": obj.category, "error": f"{type(error).__name__}: {error}"})
    return out


def _source(obj, state, toggle):
    links = list(state.links.values()) if state.links else []
    entry = {"object": obj.name, "category": obj.category, "temperature_c": float(state.temperature),
             "distance_m": float(state.distance_threshold), "requires_toggled_on": bool(state.requires_toggled_on),
             "requires_closed": bool(state.requires_closed), "requires_inside": bool(state.requires_inside),
             "elements": [link.name for link in links], "fixed_base": bool(obj.fixed_base)}
    if not state.requires_inside and links:
        live = state.link  # OmniGibson's choice: the first meta link
        entry["element"] = live.name
        entry["element_position"] = _position(live).tolist()
        entry["other_positions"] = [_position(link).tolist() for link in links if link is not live]
    if toggle is not None and toggle.links:
        entry["toggle_position"] = _position(toggle.link).tolist()
    return entry


def _xy(point):
    """(x, y) in metres to 0.1 m, never -0.0."""
    return f"({round(float(point[0]), 1) + 0.:.1f}, {round(float(point[1]), 1) + 0.:.1f})"


def _height(z):
    return f"{max(0., round(float(z), 2)) + 0.:.2f} m up"


def _number(n):
    return {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))


def describe(sources, origin, floor=None):
    """The world text's paragraph on this scene's heat sources and sinks, positions in base/odom; origin: the
    episode-start base pose (position, rotation matrix) the odometry starts from; floor: the height (m, world) of the
    floor the robot starts on (heights are above it), or None (heights above the base frame's origin)."""
    start, rotation = (np.asarray(x, float) for x in origin)
    ground = float(start[2]) if floor is None else float(floor)

    def local(point):
        return rotation.T @ (np.asarray(point, float) - start)

    shown = [s for s in sources if "error" not in s]
    if not shown:
        return ""
    lines = []
    for source in sorted(shown, key=lambda s: s["object"]):
        cools = source["temperature_c"] < AMBIENT_C
        word = "cools" if cools else "heats"
        text = f"{source['object']} ({source['category'].replace('_', ' ')})"
        if source["requires_inside"] or "element_position" not in source:
            text += f" {word} only what is inside it"
        else:
            element = source["element_position"]
            text += (f" {word} within {source['distance_m']:g} m of one point, at {_xy(local(element))}, "
                     f"{_height(element[2] - ground)}")
            others = source.get("other_positions") or []

            def other(point):  # a dead element within the live one's range still has things on it heated (by the live one)
                gap = math.dist(point, element)
                return _xy(local(point)) + (f" ({gap:.2f} m from the live one, so within its range)"
                                            if gap < source["distance_m"] else "")
            if others:
                text += (f"; its model has {_number(len(others) + 1)} such elements (burners) and the others, at "
                         + ", ".join(other(p) for p in others) + f", do not {word[:-1]} by themselves")
        needs = [need for need, flag in (("switched on", source["requires_toggled_on"]),
                                          ("closed", source["requires_closed"])) if flag]
        if needs:
            text += ", only while it is " + " and ".join(needs)
        if source["requires_toggled_on"] and "toggle_position" in source:
            toggle = source["toggle_position"]
            text += f" (its switch marker at {_xy(local(toggle))}, {_height(toggle[2] - ground)})"
        if not source["fixed_base"]:
            text += " (it can be moved, and its element moves with it)"
        lines.append(text)
    return HEADER + ": " + "; ".join(f"({i}) {line}" for i, line in enumerate(lines, 1)) + "."


def text(scene, origin, floor=None):
    """The paragraph for a loaded scene, or "" when it has no appliance heat source or cannot be read (the log says
    why): a scene the code cannot read keeps the world text without it."""
    try:
        return describe(find(scene), origin, floor)
    except Exception:
        print(f"scene heat sources not read: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return ""
