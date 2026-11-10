"""The goal's particles in rooms that are not loaded are removed when an episode loads (environment.particles_out,
Episode.load) and named in its record (BehaviorSim.measures). OmniGibson is faked: a room map that splits the world at
x = 0 and y = 0, particle systems with attachment groups, and the modules Episode.load imports."""
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rosbridge_behavior.backend import BehaviorSim
from rosbridge_behavior.environment import SETTLE_STEPS, Episode, not_covered_goals, particles_out

NOT_DUST = ["not", ["covered", "floor.n.01_1", "dust.n.01_1"]]


class Rooms:
    """garage_0 at x > 0, garden_0 at x < 0 and y > 0, no room at x < 0 and y < 0 (under a wall)."""
    def get_room_instance_by_point(self, xy):
        x, y = xy
        return "garage_0" if x > 0 else "garden_0" if y > 0 else None


class System:
    """A macro particle system's attachment groups, {object name: {particle name: place}}, as saved; a load restores
    them. Removals and loads are logged."""
    def __init__(self, name, groups, log):
        self.name, self.saved, self.log = name, groups, log
        self.load_state(None, serialized=False)

    def load_state(self, state, serialized):
        self._group_particles = {group: dict(places) for group, places in self.saved.items()}
        self.log.append(("load", self.name))

    @property
    def groups(self):
        return set(self._group_particles)

    @property
    def particles(self):
        return {name: None for places in self._group_particles.values() for name in places}

    @staticmethod
    def get_group_name(obj):
        return obj.name

    def get_group_particles_position_orientation(self, group):
        return np.array(list(self._group_particles[group].values()), float).reshape(-1, 3), None

    def remove_particles(self, idxs):
        removed = [list(self.particles)[i] for i in idxs]
        for places in self._group_particles.values():
            for name in removed:
                places.pop(name, None)
        self.log.append(("remove", self.name, removed))


def condition(*options):
    """A compiled top-level goal condition: its options, each a list of grounded literals."""
    return SimpleNamespace(flattened_condition_options=[list(option) for option in options])


def world(*conditions, log=None):
    log = [] if log is None else log
    floor = SimpleNamespace(name="floors_0", fixed_base=True, keep_still=lambda: None)
    rag = SimpleNamespace(name="rag_0", fixed_base=False, keep_still=lambda: None)
    dust = System("dust", {"floors_0": {"dustParticle0": (1., 1., 0.), "dustParticle1": (-.3, 2., 0.),
                                        "dustParticle2": (-.3, -1., 0.), "dustParticle3": (2., -1., 0.)},
                           "rag_0": {"dustParticle4": (-.5, 3., .4)}}, log)
    water = System("water", {"floors_0": {"waterParticle0": (-.3, 2., 0.)}}, log)
    sand = System("sand", {}, log)
    scope = {"agent.n.01_1": None, "floor.n.01_1": floor, "rag.n.01_1": rag,
             "dust.n.01_1": dust, "water.n.06_1": water, "sand.n.04_1": sand}
    task = SimpleNamespace(object_scopes=[scope], activity_goal_conditions=list(conditions))
    scene = SimpleNamespace(seg_map=Rooms(), is_visual_particle_system=lambda name: name in ("dust", "sand"))
    return scene, task, dust


def test_the_goals_not_covered_literals_are_found_in_each_conditions_options():
    goal = [condition([NOT_DUST, ["ontop", "rag.n.01_1", "floor.n.01_1"]]),
            condition([NOT_DUST], [["not", ["covered", "floor.n.01_1", "sand.n.04_1"]]]),  # (or ...): two options
            condition([["covered", "rag.n.01_1", "dust.n.01_1"],
                       ["not", ["not", ["covered", "rag.n.01_1", "water.n.06_1"]]]])]  # an implication's antecedent
    assert not_covered_goals(goal) == [("floor.n.01_1", "dust.n.01_1"), ("floor.n.01_1", "sand.n.04_1")]
    assert not_covered_goals(None) == []


def test_only_particles_in_a_room_that_is_not_loaded_are_out():
    """Not one in a loaded room, and not one where the map names no room."""
    scene, task, dust = world(condition([NOT_DUST]))
    assert particles_out(scene, task, ["garage_0"]) == [(dust, ["dustParticle1"])]
    assert particles_out(scene, task, ["garden_0"]) == [(dust, ["dustParticle0", "dustParticle3"])]
    assert particles_out(scene, task, ["garage_0", "garden_0"]) == []


def test_a_movable_objects_particles_a_physical_system_and_a_goal_without_particles_are_left_alone():
    scene, task, _ = world(condition([["not", ["covered", "rag.n.01_1", "dust.n.01_1"]],
                                      ["not", ["covered", "floor.n.01_1", "water.n.06_1"]],
                                      ["not", ["covered", "floor.n.01_1", "sand.n.04_1"]],
                                      ["not", ["covered", "agent.n.01_1", "dust.n.01_1"]],
                                      ["covered", "floor.n.01_1", "dust.n.01_1"]]))
    assert particles_out(scene, task, ["garage_0"]) == []


@pytest.fixture
def loading(monkeypatch, tmp_path):
    """Episode.load on a faked OmniGibson: returns the episode and the log of what happened, in order."""
    log = []
    instance = tmp_path / "instance.json"
    instance.write_text(json.dumps({"dust.n.01_1": {}}))
    modules = {
        "omnigibson": {"sim": SimpleNamespace(update_handles=lambda: log.append("update handles"),
                                              step_physics=lambda: log.append("settle step"))},
        "omnigibson.eval.utils.eval_utils": {"NUM_PUBLIC_TEST_INSTANCES": 20, "TEST_INSTANCE_IDS": range(301, 321),
                                             "seed_everything": lambda seed: None},
        "omnigibson.eval.utils.light_utils": {"LightToggleSynchronizer": None, "set_light_control_toggles": None},
        "omnigibson.tasks.behavior_task": {"get_presampled_robot_poses": None},
        "omnigibson.utils.asset_utils": {"get_task_instance_path": lambda *args, **kwargs: str(instance)},
        "omnigibson.utils.bddl_utils": {"is_system_bddl_inst": lambda name: name.split(".")[0] in (
            "dust", "water", "sand")},
        "omnigibson.utils.python_utils": {"recursively_convert_to_torch": lambda value: value},
        "omnigibson.metrics.task_metric": {"compute_q_score": lambda success, now, initial: float(success)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    def episode(rooms):
        scene, task, dust = world(condition([NOT_DUST]), log=log)

        def saved():
            log.append(("initial state saved", sorted(dust.particles)))
        scene.update_initial_file = saved
        task.__dict__.update(scene_name="garage", activity_name="sweeping_garage", activity_definition_id=0,
                             get_cached_activity_scene_filename=lambda **kwargs: "instance",
                             get_goal_option_satisfaction=lambda env: [[False]])
        loaded = Episode.__new__(Episode)
        loaded.task, loaded.rooms, loaded.embodiment = "sweeping_garage", rooms, SimpleNamespace(pose_key="r1pro")
        loaded.env = SimpleNamespace(task=task, scene=scene, reset=lambda: log.append("reset"))
        loaded.robot = SimpleNamespace(reset=lambda: None)
        return loaded
    return episode, log


def test_load_removes_them_before_the_initial_state_is_saved_and_the_record_names_them(loading):
    make, log = loading
    episode = make(["garage_0"])
    sim = BehaviorSim(SimpleNamespace(frequencies={"action_frequency": 30}, rendering="evaluator"))
    sim.episode, sim.success, sim.initial_goal_options = episode, False, [[False]]
    for _ in range(2):  # a reset loads the instance again, and the same particle goes
        log.clear()
        episode.load(0)
        assert log == [("load", "dust"), "update handles", *["settle step"] * SETTLE_STEPS,
                       ("remove", "dust", ["dustParticle1"]),
                       ("initial state saved", ["dustParticle0", "dustParticle2", "dustParticle3", "dustParticle4"]),
                       "reset"]
        assert episode.particles_removed == ["dustParticle1"] and episode.instance == 301
        assert sim.measures() == {"q_score": 0., "particles_removed_out_of_loaded_rooms": ["dustParticle1"]}


def test_a_load_that_removes_nothing_records_only_the_q_score(loading):
    make, log = loading
    episode = make(["garage_0", "garden_0"])
    episode.load(0)
    assert not any(entry[0] == "remove" for entry in log if isinstance(entry, tuple))
    sim = BehaviorSim(SimpleNamespace(frequencies={"action_frequency": 30}, rendering="evaluator"))
    sim.episode, sim.success, sim.initial_goal_options = episode, False, [[False]]
    assert episode.particles_removed == [] and sim.measures() == {"q_score": 0.}
