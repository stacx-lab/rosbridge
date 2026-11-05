"""environment.batch_particle_view_rebuilds against stand-ins for OmniGibson's macro particle systems (no simulator).

OmniGibson rebuilds every physics view after each particle it removes (og.sim.remove_prim -> update_handles). The
batching keeps the removals and their order and runs one rebuild where the last per-particle one ran.
"""
import sys
import types

import pytest

from rosbridge_behavior.environment import batch_particle_view_rebuilds


@pytest.fixture
def og(monkeypatch):
    """Stand-in omnigibson modules: a simulator that logs removals and rebuilds, and the three system classes with
    OmniGibson's removal structure (remove_particles / remove_all_group_particles loop over remove_particle_by_name,
    which removes a prim and rebuilds; _sync_particle_groups removes stale groups, then adds the desired ones)."""
    log = []

    class Sim:
        def update_handles(self):
            log.append("rebuild")

        def remove_prim(self, prim):
            log.append(f"remove {prim}")
            self.update_handles()

    module = types.ModuleType("omnigibson")
    module.sim = Sim()

    class MacroParticleSystem:
        def __init__(self, groups):
            self.groups = {g: list(p) for g, p in groups.items()}

        @property
        def names(self):
            return [p for g in self.groups.values() for p in g]

        def remove_particle_by_name(self, name):
            for particles in self.groups.values():
                if name in particles:
                    particles.remove(name)
            module.sim.remove_prim(name)

        def remove_particles(self, idxs, **kwargs):
            names = self.names
            for idx in idxs:
                self.remove_particle_by_name(names[idx])

        def add_particle(self, group, name):
            self.groups[group].append(name)
            log.append(f"add {name}")

        def generate_particles(self, positions, orientations=None, scales=None, **kwargs):
            raise AssertionError("not used here")

    class MacroPhysicalParticleSystem(MacroParticleSystem):
        def _sync_particles(self, n_particles):
            raise AssertionError("not used here")

    class MacroVisualParticleSystem(MacroParticleSystem):
        def remove_all_group_particles(self, group):
            for name in tuple(self.groups[group]):
                self.remove_particle_by_name(name)

        def remove_attachment_group(self, group):
            self.remove_all_group_particles(group)
            del self.groups[group]

        def _sync_particle_groups(self, desired):
            for group in [g for g in self.groups if g not in desired]:
                self.remove_attachment_group(group)
            for group, names in desired.items():
                if group not in self.groups:
                    self.groups[group] = []
                    for name in names:
                        self.add_particle(group, name)

    systems = types.ModuleType("omnigibson.systems.macro_particle_system")
    systems.MacroParticleSystem = MacroParticleSystem
    systems.MacroPhysicalParticleSystem = MacroPhysicalParticleSystem
    systems.MacroVisualParticleSystem = MacroVisualParticleSystem
    stubs = {"omnigibson": module, "omnigibson.utils": types.ModuleType("omnigibson.utils"),
             "omnigibson.utils.transform_utils": types.ModuleType("omnigibson.utils.transform_utils"),
             "omnigibson.systems": types.ModuleType("omnigibson.systems"),
             "omnigibson.systems.macro_particle_system": systems}
    if "torch" not in sys.modules:
        stubs["torch"] = types.ModuleType("torch")
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    module.utils, module.systems = stubs["omnigibson.utils"], stubs["omnigibson.systems"]
    module.utils.transform_utils, module.systems.macro_particle_system = stubs["omnigibson.utils.transform_utils"], systems
    return types.SimpleNamespace(log=log, sim=module.sim, Visual=MacroVisualParticleSystem,
                                 Physical=MacroPhysicalParticleSystem)


def removals_and_rebuilds(run, og):
    del og.log[:]
    run()
    return list(og.log)


def test_a_reset_restore_rebuilds_once_per_stale_group_not_once_per_particle(og):
    """The spraying reset: _sync_particle_groups removing two sprayed groups of 3 and 2 particles."""
    def restore():
        og.Visual({"tree_1": ["p0", "p1", "p2"], "tree_2": ["p3", "p4"]})._sync_particle_groups({})

    before = removals_and_rebuilds(restore, og)
    batch_particle_view_rebuilds()
    after = removals_and_rebuilds(restore, og)
    assert before.count("rebuild") == 5
    assert after == ["remove p0", "remove p1", "remove p2", "rebuild", "remove p3", "remove p4", "rebuild"]
    assert [x for x in after if x != "rebuild"] == [x for x in before if x != "rebuild"]  # same removals, same order


def test_visual_remove_particles_rebuilds_once_at_its_end_and_the_state_is_the_same(og):
    """A ParticleRemover or washing rule removing several visual particles in one call."""
    batch_particle_view_rebuilds()
    system = og.Visual({"floor": ["d0", "d1", "d2", "d3"]})
    assert removals_and_rebuilds(lambda: system.remove_particles([0, 1, 2]), og) == [
        "remove d0", "remove d1", "remove d2", "rebuild"]
    assert system.groups == {"floor": ["d3"]}


def test_a_restore_that_also_adds_particles_keeps_its_order(og):
    batch_particle_view_rebuilds()
    system = og.Visual({"old": ["p0", "p1"]})
    log = removals_and_rebuilds(lambda: system._sync_particle_groups({"new": ["q0"]}), og)
    assert log == ["remove p0", "remove p1", "rebuild", "add q0"]  # the rebuild before the additions, as before
    assert "update_handles" not in vars(og.sim)  # the deferral is gone afterwards


def test_a_removal_that_fails_midway_still_rebuilds_and_restores_the_simulator(og):
    batch_particle_view_rebuilds()

    class Failing(og.Visual):
        def remove_particle_by_name(self, name):
            if name == "bad":
                raise KeyError(name)
            super().remove_particle_by_name(name)

    system = Failing({"a": ["p0", "bad"]})
    del og.log[:]
    with pytest.raises(KeyError):
        system.remove_all_group_particles("a")
    assert og.log == ["remove p0", "rebuild"]
    assert "update_handles" not in vars(og.sim)


def test_installing_twice_wraps_once(og):
    batch_particle_view_rebuilds()
    wrapped = og.Visual.remove_all_group_particles
    batch_particle_view_rebuilds()
    assert og.Visual.remove_all_group_particles is wrapped
