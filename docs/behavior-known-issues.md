# BEHAVIOR: known environment issues

This page lists every problem known in the BEHAVIOR activities as rosbridge runs them: the 100 activities of the 2026 challenge in the `behavior` simulator (the evaluator's R1Pro with grippers and assisted grasping) and in `behavior-dex` (the same R1Pro with two Sharpa Wave hands, which grasp by finger contact only). It covers what is fixed by an open pull request, what is still open, what only BEHAVIOR-1K upstream can fix, and what nobody has investigated. Problems of a harness that runs agents, outside this repository, are not listed.

## How to read this

**Evidence.** Each entry says how it is known:

- *measured*: a native run in Isaac Sim through rosbridge's own backend;
- *read*: source code or data files;
- *inferred*: reasoning from measured or read facts, not itself run;
- *recorded*: seen in an agent episode.

Unless stated otherwise, native runs used seed 0, which loads public test instance 301. `behavior-g1` was not prepared and nothing here was run on it.

**Status.**

| Status | Meaning |
|---|---|
| #16 … #20, #22, #23 | Fixed or mitigated by that open pull request. Section 9 says what each was not validated on |
| upstream | Needs a change in BEHAVIOR-1K or OmniGibson; rosbridge may carry a workaround |
| open | No fix exists |
| unknown | The cause is not established |
| left as is | Known and deliberately unchanged, with the reason |

Nothing in the "upstream" rows has been reported to BEHAVIOR-1K yet. Section 10 lists the reports with their reproductions.

## 1. All issues at a glance

| # | Issue | Robots | Status |
|---|---|---|---|
| 2.1 | The base drive has no effort limit and tips the robot over | both | observation in #19; #22 for behavior-dex; upstream |
| 2.2 | Joints thrown tens of radians out of range, or NaN | both | unknown; #19 ends the episode |
| 2.3 | One-step velocity spikes of the base's virtual joints | both | harmless as measured; upstream |
| 2.4 | Doors get stiffer with mass, and creep | both, decisive for hands | #23 for behavior-dex; upstream |
| 2.5 | Lids and flaps fall shut or fall open | both | #23 for behavior-dex (fixed objects only); upstream |
| 2.6 | Many handles cannot be gripped by a hand | behavior-dex | open; upstream (assets) |
| 2.7 | Torso targets that stall | both | cause found (collision); device text not written |
| 2.8 | "Car windows have no collision" | – | not reproduced |
| 3.1 | The first removed visual particle crashed the world, and start poses were wrong | both | #16 |
| 3.2 | Particle removers followed a tool only when a camera was read | both | #16; upstream |
| 3.3 | A broken attachment joint aborted the process | both | #16; upstream |
| 3.4 | A physics failure stopped the world with an error | both | #19 |
| 3.5 | Native worker segmentation fault | ? | unknown |
| 4.1 | Wait replies did not say that or why the episode ended | all simulators | #17 |
| 4.2 | A tilted base looked upright in odometry and TF | both | #19 |
| 4.3 | A client cannot read the episode's step limit | all BEHAVIOR | open |
| 4.4 | The agent is not told how cutting, toggling and removal work | both | open |
| 5.1 | Any part of any slicer cuts; slicers cut again on their halves | both | #20 for behavior-dex; upstream |
| 5.2 | A vacuum standing flat removes nothing | both | #20 for behavior-dex (optional); upstream |
| 5.3 | Oven racks count for `open` | both | upstream |
| 6.1 | Goal particles beyond a locked door | both | #18; upstream |
| 6.2 | The only route to the sink passes a counter flap | both | open |
| 6.3 | A toy at the back of a low cabinet | behavior-dex | open, low confidence |
| 7.1 | No official instruction for half the activities | both | upstream |
| 7.2 | Four goals disagree with their instructions | both | upstream |
| 7.3 | Two goals bind one of two identical fixtures | both | upstream |
| 7.4 | Wording errors in official instructions | both | upstream |
| 8.1 | Success fires at first contact | both | left as is |
| 8.2 | `OnTop` depends on yaw under an overhang | both | upstream |
| 8.3 | `nextto` is brittle | both | upstream |
| 8.4 | `inside` and `filled` are tight in two activities | both | upstream |

Section 11 lists what has not been investigated at all. The largest item there is whether each activity can be solved by the Sharpa hands.

## 2. Physics fidelity

### 2.1 The base drive has no effort limit and tips the robot over

- **Symptom.** The R1Pro tilts and often falls over when its base drives against furniture, a door or a wall.
- **Cause** (*measured*). OmniGibson writes 1000 N to the base's three virtual drive joints at initialization, through the articulation view. The stop and play that the evaluator uses to set the 250 kg base mass (`eval/evaluator.py`, copied in `R1Pro.configure`) discards it: PhysX then holds 3.4e38, and `JointPrim.max_effort` hides that by answering 100. The drive pushes with 100 to 540 kN.
- A second defect behind it: with the intended 1000 N restored, the robot reaches 0.45 m/s of a 0.75 m/s command, because its wheels are fixed spheres that slide with friction 0.5 and need 1248 N to move. The evaluator's robot drives only because the limit is lost.
- **Evidence** (*measured*). Backing into a fridge, both robots fall over and come to rest at 82.0° (Sharpa hands) and 82.6° (grippers). Across 123 drives into furniture, walls and doors in four scene and seed combinations, the unlimited drive tilted the base more than 10° in 90 and left it lying beyond 45° in 46. Open-floor driving stays within 4.1°.
- **Status.**
  - #19 makes the tilt visible (4.2) and states it in wait replies. It does not prevent it.
  - #22, for behavior-dex: frictionless wheels and a drive limit of 500 N and 150 N m, written to USD between the stop and the play. The same 123 drives stayed at or below 3.9°. The gripper robot's trajectory is bit-identical with and without it.
  - `behavior` keeps the defect, so that its physics stays the evaluator's.
- **What the limit of #22 costs.** Pushing a door open with the base is weaker: a door that resists can yaw the base 35 to 43° or slide it 0.2 to 0.4 m aside, and a wedged chair moved 0.04 m against 0.38 m. Not validated: driving while a hand holds a drawer or door, thresholds and floor edges, any value between 500 and 700 N (700 N tilts 20.5°), and an agent episode.

### 2.2 Joints thrown tens of radians out of range, or NaN

- **Symptom.** In one control step a joint jumps far past its limit, or the base's orientation becomes NaN. The episode ran on with a broken robot, or the world stopped.
- **Evidence.** *Recorded* by the author of #19: `torso_joint2` at 24 rad after base bursts on a tipped robot; arm joints at −61 and −1334 rad in agent episodes.
- **Not reproduced.** No reachable state diverged in 1,234 native scenarios (falls from 30° to 150° in 8 directions and 7 postures, drives into furniture, 20 burst sequences on a tipped or lying robot, pushes). All 15 runaways produced started with the robot placed inside furniture, which no client can do.
- **Suspected cause**, untested: an ill-conditioned contact island (a 250 kg base on virtual links of 1e-6 kg, gravity-free arm links wedged under it, one velocity iteration). The candidate solver settings were never tried, because there is no reproduction to try them on. If the trigger is the unlimited base drive, the limit of 2.1 removes it for behavior-dex; that is plausible and unproven.
- **Status.** Unknown. #19 ends the episode cleanly with reason `invalid_state` (3.4); nothing prevents the divergence.

### 2.3 One-step velocity spikes of the base's virtual joints

- *Measured.* The base's six virtual joints are an x, y, z, rx, ry, rz chain, singular where ry = ±90°. When a fall carries the base through that pose, rx and rz spike for one step, up to 1690 rad/s. This happened in 15 reachable scenarios, with no lasting effect.
- Consequence: joint velocity cannot serve as a divergence detector, and the base's virtual joints are checked for finite values only.

### 2.4 Doors get stiffer with mass, and creep

- **Cause** (*read and measured*). OmniGibson gives every revolute joint friction 0.2 and every prismatic joint 0.3 (`dataset_object.py`), which PhysX multiplies by the load the joint carries. The effort needed to start a part grows with its mass, and below it the part creeps instead of holding.
- **Evidence.** Effort at which a part has moved 0.02 rad or 5 mm, on 55 fixed models:

  | Part | Evaluator | With #23 |
  |---|---|---|
  | Cabinet doors | 0.6 to 2.1 N m | 0.5 to 1.6 |
  | Microwaves | 1.2 to 2.1 | 1.6 |
  | Fridges (`dszchb`, `petcxr`, `juwaoh`, `jtqazu`) | 2.6 to 6.6 | 1.6 to 1.9 |
  | Washer, dryer | 3.7, 4.5 | 1.7 |
  | Oven `ffitak` | 5.0 | 1.7 |
  | Room doors | 9.7 to 14.6 | 1.8 to 2.1 |
  | Sliding door | 33.7 N | 7.8 N |
  | Drawers | 5.7 to 16 N | 4.7 to 7.0 N |

- **Why it matters for hands.** The gripper robot's assisted grasp is a rigid constraint (it held 200 N). A Sharpa wrap grasp on a 5.5 cm object starts slipping at 12 N and loses it at 32 to 47 N.
- **Status.** #23, for behavior-dex: for every single-axis joint of a fixed object, a load-independent static friction effort (OmniGibson's coefficient times the load the joint carries at rest, capped at 1.5 N m or 6 N) and 0.7 of it as dynamic. It names no object and measures nothing per joint. Locked doors stay locked (0.0011 rad under 200 N m), `Open` and `ToggledOn` read the same, and step time is unchanged.
- **Does it help a hand?** *Measured* with scripted grasps on the real handles:

  | Case | Evaluator | With the rule |
  |---|---|---|
  | Fridge `petcxr` left door (bar with a 2.8 cm gap) | handle lost after 0.14 rad | opened 0.39 rad |
  | Room door with a lever handle (13 kg) | 0.04 rad | 0.42 rad |
  | Light doors: `petcxr` right, ovens `ffitak` and `amblrk`, washer by its door ring | open | open |
  | Room doors pushed with the open hand | open 0.46 to 0.59 rad | the same |
  | Handles of 2.6 | not gripped | not gripped |

- **Side effects.** Doors coast: pushed at 0.5 rad/s a door travels 0.04 to 0.12 rad after release (0.004 to 0.02 before).

### 2.5 Lids and flaps fall shut or fall open

- *Measured.* A car trunk set 25% or 50% open shuts within 1 s. Oven `ffitak` and both dishwasher doors fall open from 50%. The bar counter's flap and the toilet seat and lid shut. The assets have no spring or detent.
- **Affects.** `loading_the_car`, `unloading_the_car`, `carrying_in_groceries`; `make_pizza`, `cook_a_brisket`; `putting_dirty_dishes_in_sink` (6.2); `tidying_bathroom`.
- **Status.** #23, for behavior-dex: gravity is switched off on a hinged part of a fixed object when gravity would turn it with more than half its static friction. Those parts then hold at 25, 50, 75 and 90% open, can be pushed shut, and `Open` reads correctly.
- **Not covered.** Movable objects are untouched, so the toolbox, a hinged jar, laptops, recycling bins and the trash can still fall. Relative to the evaluator, lids now stay where they are left, so a trunk or oven door must be pushed shut.

### 2.6 Many handles cannot be gripped by a hand

- *Measured* with the Sharpa hand placed on the real handles, and by a ray census of collision against visual geometry:

  | Why | Models |
  |---|---|
  | The collision hull fills the gap behind the handle | fridges `dszchb` and `jtqazu`, dishwasher `ynrhuk`, car trunks (0.4 cm gap), cabinet `fancyy` (held, then lost while opening) |
  | Small handle, or none that protrudes | fridge `juwaoh`, both microwaves, dryer `xsuyua`, cabinets `lkxmne`, `gjeoer`, `rkgjer`, garden doors `vudhlc`, restaurant door `mkcndz` |
  | A 1.6 cm gap the hand got no hold on | cabinet `gjrero` |

- **Affects.** Fridge `dszchb` alone is used by 12 activities: `boxing_food_after_dinner`, `bringing_water`, `canning_food`, `cook_bacon`, `cook_cabbage`, `cook_hot_dogs`, `freeze_fruit`, `freeze_pies`, `halve_an_egg`, `preparing_lunch_box`, `setting_the_table`, `slicing_vegetables`. Fridge `jtqazu`: `cook_a_frozen_pie`, `cook_brussels_sprouts`, `store_produce`. Cabinet `gjrero` holds the toys of `wash_dog_toys`. 22 activities start with a goal object inside a closed container (11.2).
- **Status.** Open. Lighter doors do not help a hand that has nothing to hold; the fix is re-decomposed collision meshes, which is asset data.
- **Not known.** Whether these can be opened without the handle, by hooking the door's edge with a fingertip or pushing from the side. The tests gripped handles only, were scripted and open-loop, and used the left hand; several grasp variants missed their target by 4 to 100 cm and were excluded. So "not gripped" means "these scripted grasps got no hold", which is weaker than "no grasp exists".

### 2.7 Torso targets that stall

- *Measured.* Every stall is a collision with the 100 N m drive saturated against it, not a weak drive. From the start posture, with the arms hanging, the torso follows the evaluator's squat only halfway: the hanging hands meet the base. With both arms raised forward every posture tracks exactly with efforts below 5 N m. The gripper robot stalls at the same postures.
- **Status.** The device text should say that a torso target that stalls is a collision, and to raise both arms before lowering the torso. Not written.

### 2.8 "Car windows have no collision": not reproduced

All six glass links of both car models have a collision mesh, and 34 to 42 of 54 rays through each glass link hit it. The body's hulls also lie in the window boxes, so a ray that reports only the closest hit names the body, which may explain the earlier reading. Not tested: whether a fast arm can tunnel through the thin shell.

## 3. Simulation robustness

### 3.1 The first removed visual particle crashed the world, and start poses were wrong

- **Cause** (*measured*). `patches/behavior/0004` stopped updating the physics handles after a visual particle is removed. Deleting any prim drops the handle of every articulation view, so the next joint read raised and joint or pose writes were silently discarded.
- **Symptoms.** The first particle a cleaning tool removed ended the episode, in the 11 activities whose goals remove visual particles. Restoring an instance lost the robot's start pose: `sweeping_garage` started 1.04 m and 2.02 rad off, `clean_a_patio` 4.8 m, `putting_dirty_dishes_in_sink` 1.4 m, `scrubbing_bathroom_floor` 0.10 m.
- **Status.** #16 corrects the patch itself and batches the updates.

### 3.2 Particle removers followed a tool only when a camera was read

- **Cause** (*measured*). Particle removers and `link.aabb`, `link.visual_aabb`, `obj.aabb` read poses from Fabric, which Kit updates when it renders. The evaluator renders inside every step; rosbridge steps without rendering. A broom driven 1.03 m over three particles removed none; its removal volume read 1028 mm behind.
- **Status.** #16 (its second patch) updates Fabric on steps that do not render. It changes what `behavior` computes: removals happen one step earlier, which is what the evaluator's own step computes. A step costs a median 0.6 to 3.8 ms more.
- **Not measured.** `contains` reads `check_points_in_volume`, which is stale by code reading.

### 3.3 A broken attachment joint aborted the process

- **Cause** (*measured*). `on_joint_break` deletes the joint prim inside Kit's physics-step callback; the process aborts (`Mutex.cpp(103): Recursion not allowed`), with both robots. It needs a step that does not render: under the evaluator's own step the same break is handled.
- **Affects.** `make_cabinet_doors`, and by the same code `hanging_pictures`, `attach_a_camera_to_a_tripod`, `installing_smoke_detectors`.
- **Status.** #16 (its third patch) runs the callbacks after the physics step.

### 3.4 A physics failure stopped the world with an error

- **Symptom.** PhysX reported `Illegal BroadPhaseUpdateData`, OmniGibson's own step asserted on the base's NaN orientation, and the world stopped: the record said "stopped" with an error and no score.
- **Status.** #19 checks the robot's state after every step at two levels. A joint more than 0.5 rad beyond its range is stated in every wait reply and the episode goes on. A non-finite value, a revolute joint more than 2π beyond its range, or a prismatic joint more than its own range width beyond its range ends the episode with reason `invalid_state`, a `fault` text and its score.
- **How the levels were chosen** (*measured*). The largest excess in a reachable scenario was 4.97 rad: a squatting robot fell on its side with a wrist wedged under the base, which wound the forearm roll joint past its stop for 3 s before it unwound by itself. Every runaway read 10.1 rad or more. So 2π ends no reachable scenario and every runaway, with little room: any level from 5 to 10 rad does the same on this data, and the runaway side comes only from starts placed inside furniture.
- **Left as is.** A fall does not end the episode. A fallen robot runs on until the step limit.

### 3.5 Native worker segmentation fault

One recorded `canning_food` episode ended with signal 11. It was never reproduced and the cause is unknown.

## 4. What the client sees

### 4.1 Wait replies did not say that or why the episode ended

On `main` a wait whose last step is the terminal one, and every zero wait after the end, answered only `ok`. #17 states in every wait reply at or after the end that the episode ended and what ended it: the time limit, the simulator, or (with #19) an invalid robot state. It names the goal only for a run launched with `--stop-on-success`, as a separate commit that can be declined. The other six simulators share the runtime and change the same way; that is read from their backends, not run.

### 4.2 A tilted base looked upright

The base's odometry was planar and nothing put `base/odom` on TF. #19 publishes the base's full attitude on its odometry and on TF for both R1Pro rigs, and states a tilt of more than 15° in every wait reply. An earlier refusal of base motion beyond 30° is dropped: propped at 30.9°, the robot was refused 44 times while driving back out would have recovered it.

### 4.3 A client cannot read the episode's step limit

The limit is 1.5 times the mean human demonstration length (`task.jsonl`). The world device gives elapsed time only, and the limit appears nowhere a client can read before the episode ends. It cannot be computed from `task_data.json` either: 1.5 times its `duration` matches the real limit for none of the 100 activities, is within 1% for 53 and off by more than 10% for 30 (`tidying_living_room` 14220 against 18848 steps; `cook_brussels_sprouts` 30915 against 23492). `docs/behavior.md` also says the world node's `max_steps` parameter overrides the limit, but the launch file does not pass one. Status: open.

### 4.4 The agent is not told how the rules work

Nothing in the device texts says that only a blade cuts and a slicer must be lifted clear for 2 s before it cuts again; that a washer, dryer, dishwasher or microwave refuses to switch on while open; that one remaining particle keeps a surface "covered"; or, with the counterbalance of #23, that a lid stays where it is left. Status: open.

## 5. Transition rules

### 5.1 Any part of any slicer cuts, and slicers cut again on their halves

- *Measured* with OmniGibson's rule: the handles of an axe, a carving knife and a table knife cut a log, an egg and a cupcake at first contact. A knife that had just cut, and was inactive, cut the next cupcake while another knife was active. A knife laid on the egg halves it had cut diced them, taking `halve_an_egg` from `q_score` 0.4 to 0. In `make_pizza` the knife's handle halved the dough and a pepperoni, which the recipe takes whole.
- **Affects.** Nine activities with a slicer and cuttable food: `chopping_wood`, `halve_an_egg`, `cook_brussels_sprouts`, `chop_an_onion`, `slicing_vegetables`, `canning_food`, `cook_cabbage`, `make_pizza`, `setting_the_table`.
- **Status.** #20 for behavior-dex, as a source patch: an object is cut by a slicer that is active, in contact with it, and whose blade box reaches it; the slicer that cut is inactive until 2 s after it last touched food. There is no speed condition. `behavior` keeps OmniGibson's rule.
- **Left as it is under both rules.** Food set down on a blade at rest is cut. `halve_an_egg`'s knife lies on the chopping board at load.

### 5.2 A vacuum standing flat removes nothing

- **Cause** (*read and measured*). The asset places the vacuum's removal box above its meta link; `ParticleModifier._initialize` re-places it below, 1.1 mm under the floor it stands on. The rule removes floor dust only when the vacuum is lifted 1.2 to 21.5 mm, or leans by a degree or more.
- **How much it matters.** When the robot's base pushes the vacuum, the vacuum leans 5 to 10° on the robot and OmniGibson's own rule removes the dust (2 of 2 runs). Pushed flat by a block, it removes 0 of 8 particles.
- **Status.** #20 for behavior-dex adds a 1 cm margin as a separate commit, which makes removal independent of lean. It is optional: it is not what makes `vacuuming_floors` possible.
- The Sharpa hand can switch the vacuum on with a pointing index fingertip, and the base can push it 0.5 to 0.7 m.

### 5.3 Oven racks count for `open`

*Measured* in `make_pizza`: with the door at 0 and a rack 0.021 m out, the oven reads open and its heat is inactive. Whether a rack moves that far during real manipulation was never tested. Affects `make_pizza`, `cook_a_brisket`, `cook_a_frozen_pie`, `cook_brussels_sprouts`.

## 6. Task instance data

### 6.1 Goal particles beyond a locked door

- *Measured and read.* OmniGibson's ground sampler tests that a pose is traversable, not that it lies in the covered object's room. `sweeping_garage` at seed 0 has a sand and a dust particle on the slab beyond the garage door, which rosbridge locks because it opens onto a room that is not loaded. `covered` holds while one particle is left, so the goal cannot be reached. Instance 313 has one such particle, and so do 24 of 300 training instances.
- **Status.** #18 removes at load the goal particles that the room map places in a room that is not loaded, for every embodiment, and names them in the episode record.
- **Limit of the rule.** It is correct for every public instance, checked on all 894 public particle places, but not by construction. It holds because the sampler's 7×7 erosion empties the three map cells between the door and the room boundary, and because the 0.1 m room map takes each cell's label from its first pixel. Instances resampled for a smaller robot, or a room map at another resolution, would break it. The robust rule, not built: flood the traversability map from the robot's start with the locked doors closed, and remove what the flood does not reach.

### 6.2 The only route to the sink passes a counter flap

`putting_dirty_dishes_in_sink`: from the start in the dining room, the route to the kitchen sink (11.7 m, passage 0.85 to 0.95 m) passes a flap that is a jointed link of the bar counter, not a door object. OmniGibson's traversability map does not know it and reads "no route". The flap needs 13 N m and shuts by itself; with #23 it needs 1.8 N m and stays up. Nobody has lifted it with a hand or driven through. Status: open.

### 6.3 A toy at the back of a low cabinet

`wash_dog_toys`: one of the four toys lies in the far corner of the cabinet it starts in. An earlier posture search found no collision-free reach from 40 stances, with low confidence. The cabinet's handle is also one the hand got no hold on (2.6). Status: open, not re-examined.

## 7. Goal definitions and instructions

### 7.1 No official instruction for half the activities

*Read.* `docs/challenge/task_data.json` has an instruction for activities 0 to 49 only. Texts for all 100 exist in the Hugging Face dataset `behavior-1k/2026-challenge-demos`, but for activities 50 to 99 they contradict the goal in 4 (`tidying_living_room`, `unloading_the_car`, `organizing_school_stuff`, `carrying_out_garden_furniture`), omit judged conditions in 15, give wrong counts in 2 (`make_rose_centerpieces`, `installing_smoke_detectors`) and name an ambiguous target in 4 (`store_honey`, `store_batteries`, `installing_a_modem`, `tidying_bathroom`).

### 7.2 Four goals disagree with their instructions

| Activity | Defect | Evidence |
|---|---|---|
| `loading_the_car` | The BDDL has a second top-level `:goal` expression, which the parser drops, so closing the trunk is not judged. It is the only one of 1016 definitions like this | measured: success with the trunk open |
| `setting_mousetraps` | `forn (2)` means exactly two; the instruction says "at least two". Only "under" the sink works in this instance | read and measured |
| `collecting_childrens_toys` | The goal also requires the train set, which the instruction omits | measured: the six instructed items give `q_score` 0.857 and no success |
| `make_microwave_popcorn` | The goal names no microwave | read |

### 7.3 Two goals bind one of two identical fixtures

- `thawing_frozen_food`: the bread's plate must end on one particular kitchen countertop of two.
- `carrying_out_garden_furniture`: the lawn chair must end on one particular garden floor of two.

No start fact points at the bound one, so no instruction derived from the BDDL can name it. That a chair on the other lawn fails is read, not measured.

### 7.4 Wording errors in official instructions

`spraying_for_bugs` and `putting_dishes_away_after_cleaning` (six cabinets are bound); `spraying_fruit_trees` ("both trees" are two specific trees of 26); `bringing_in_wood` and `putting_shoes_on_rack` (the goal names one of two same-kind fixtures).

## 8. Judging

### 8.1 Success fires at first contact

The task ends at the first step all literals hold, for example with a cup still moving at 0.29 m/s. This is the evaluator's semantics and rosbridge keeps it. A stricter dwell measure belongs in a harness, as an extra recorded metric.

### 8.2 `OnTop` depends on yaw under an overhang

`OnTop` tests "nothing above" with one ray. A filter on the coffee maker's plate reads false at 3 of 8 yaws, because the maker's head overhangs the plate. On the maker's top surface, 24 of 24 placements read true, so `set_up_a_coffee_station_in_your_kitchen` stays satisfiable.

### 8.3 `nextto` is brittle

It needs a box gap of at most (sum of both boxes' sides)/18 and one of 16 horizontal rays to hit. In `polishing_shoes`, 22 of 96 side-by-side placements satisfied all three literals. In `setting_mousetraps`, a trap on the floor beside the wall-hung sink reads false. Measured in these two only; `tidying_bedroom`, `putting_shoes_on_rack`, `setup_a_bar_for_a_cocktail_party`, `laying_tile_floors` and `setting_the_table` rely on the same predicate.

### 8.4 `inside` and `filled` are tight in two activities

- `inside` tests only the inner object's box centre. Roses dropped into the vase of `make_rose_centerpieces` read inside for at most 2 of 3 at once.
- `filled` is a 20% volume threshold. In `canning_food` the bowl needs 7 of the steak's 8 cubes.
- Both logs of `setting_the_fire` must rest on one newspaper barely larger than they are. Neither this nor the three roses was shown satisfiable in one configuration.

### 8.5 Semantics worth knowing

- `covered` is true with one attached particle: "clean" means every particle gone, and one spray hit "covers".
- `q_score` counts only literals newly true, so literals true at load give no credit but can block success.
- Any heat source satisfies a cooking goal, and any removal method a washing goal.
- Every one of the 100 goals is expressible and judged by a working mechanic. 24 were driven to success natively by setting state, 5 fell short because of the probe's own placement, and 71 rest on reading or on a sibling activity.

## 9. What the open pull requests were not validated on

Each passes its unit suites at every commit, and its native tests fail without the change.

| Pull request | Validated natively | Not run |
|---|---|---|
| #16, three OmniGibson patches | Runtimes prepared from the series; `behavior` and `behavior-dex` suites; `sweeping_garage` and three other particle activities; a forced joint break in `make_cabinet_doors` | Natural joint breaks and the other three attach activities; the ROS suites; an agent episode |
| #17, end notice | Unit and ROS tests with a toy backend | The other six simulators; a native world through the ROS graph |
| #18, particles in rooms not loaded | `sweeping_garage` at seeds 0 and 12 on both robots; three other activities unchanged | Success at seed 0 on the gripper robot; the ROS suites; an agent episode |
| #19, attitude, tilt notice, state check | One kitchen at seed 0 through the world runtime, with a NaN written into the base | The end level of 2π through the world runtime (unit tests only; the native runs used π); other scenes; an agent episode |
| #20, slicer and vacuum rules | 21 native tests; a knife held in a Sharpa hand; the vacuum switched on by a fingertip and pushed by the base | A held axe (the 1.4 kg axe slips out of the fist that holds the knife); cutting inside `cook_cabbage` and `slicing_vegetables`; the ROS suites; an agent episode |
| #22, base drive limit | 123 contact drives in four scene and seed combinations | With a hand holding a drawer; thresholds; an agent episode |
| #23, door friction and counterbalance | 346 joint measurements on 55 fixed models in ten worlds | Together with the drive limit; objects resting on a counterbalanced part; an agent episode |

Across all of them:

- **They were never run together.** #16, #18 and #20 form one line, and #17, #19, #22 and #23 another. #22 and #23 are siblings on #19 and were never combined with each other or with #16.
- **No agent has used any of them.** Every validation is a scripted native probe or a test.
- **One instance.** Almost everything is seed 0, instance 301. Other public instances place objects elsewhere.
- **A run does not record which rule patches were active.** Until it does, results on `behavior` and `behavior-dex` cannot be told apart from the record alone.

## 10. Reports to file with BEHAVIOR-1K

None has been filed.

| # | Report | Reproduction |
|---|---|---|
| U1 | The base joints' effort limit is lost after stop and play, and the getter hides it (2.1) | Write 1000 N, `og.sim.stop(); og.sim.play()`, read 3.4e38; `JointPrim.max_effort` answers 100 |
| U2 | Stepping without rendering leaves Fabric stale for particle removers and link boxes (3.2) | A broom moved by velocity over floor particles with `render_on_step(False)` removes none |
| U3 | A joint-break callback deletes a prim inside the step callback (3.3) | Attach in `make_cabinet_doors`, break the joint on a step that does not render |
| U4 | Slicing: any link cuts, any active slicer suffices, a slicer is active again on its halves (5.1) | Handle onto the egg cuts; an inactive knife cuts while another is active; active again at 2.03 s on its halves |
| U5 | The vacuum's removal box is re-placed below its meta link (5.2) | A vacuum standing flat removes nothing; it works lifted 1.2 to 21.5 mm |
| U6 | The ground sampler accepts particles outside the covered object's room (6.1) | `sweeping_garage` instances 301 and 313 |
| U7 | Joint friction is a coefficient on the joint's load, and lids have no counterbalance (2.4, 2.5) | The table in 2.4; the trunk shuts from 50%; the oven door falls open |
| U8 | Handle collision hulls fill the finger gap (2.6) | Fridge `dszchb`: 8 collision against 2 visual ray hits behind the bar |
| U9 | Oven rack joints count for `Open` (5.3) | `make_pizza`: door at 0, rack at 0.021 m, oven reads open, heat inactive |
| U10 | `OnTop`'s single "nothing above" ray (8.2) | Filter on the coffee maker's plate: false at 3 of 8 yaws |
| U11 | BDDL and parser (7.2) | The four activities of 7.2 |
| U12 | Instruction data (7.1, 7.3, 7.4) | `task_data.json` has no instruction for activities 50 to 99 |
| U13 | The base's virtual joints spike at ±90° pitch (2.3) | A fall through that pose |

## 11. Not investigated

### 11.1 Whether each activity can be solved by the Sharpa hands

This is the largest open question. Every goal is satisfiable by the judge (8.5), but almost no manipulation was measured: the only physical grasps are a kitchen knife, a 5.5 cm flashlight, and scripted grips on about twenty door and appliance handles.

| Question | Activities | What is known |
|---|---|---|
| Can a container be opened when its handle cannot be gripped? | 12 with fridge `dszchb`, 3 with `jtqazu`, `make_microwave_popcorn`, `wash_dog_toys`, and more (2.6) | Opening by the door's edge was not tried |
| Can the hand open the heavy fridge `petcxr`? | `carrying_in_groceries`, `cleaning_up_plates_and_food`, `clearing_food_from_table_into_fridge`, `cook_a_brisket`, `make_pizza`, `thawing_frozen_food` | Yes with #23 in a scripted grasp; without it the hand loses the handle |
| Can the hand slide a sliding door? | `sorting_household_items`, `turning_out_all_lights_before_sleep` | It starts at 33.7 N (7.8 N with #23). No hand test |
| Can the hand hold and use each tool? | broom, scrub brush, steel wool, atomizer, lighter, and so on | A kitchen knife is held and cuts. The 1.4 kg axe slips out of the same fist. Nothing else was held |
| Can the hand switch on small toggle markers? | radio, modem, fax, scanner, table lamp, lighter, stove | Not measured natively for these seven. An offline check leaves the modem and the table lamp uncertain |
| Do objects attach by physical contact? | `hanging_pictures`, `attach_a_camera_to_a_tripod`, `installing_smoke_detectors` | Placing the poster at the nail produced no contact and no attachment; set by state it holds. `make_cabinet_doors` attaches as designed |
| Objects on the floor | 22 activities have a goal object within reach only with a deep forward lean | Ordinary postures stop the fingertips 0.07 m above the floor |
| Heavy or large objects to move | for example `carrying_out_garden_furniture` (barrow 11.3 kg, lawn chair 5.0 kg) | No carry or two-handed grasp was measured |

### 11.2 Geometric screening, not followed up

From a native survey of all 100 activities at seed 0. These are 2-D results with a palm-reach envelope; none was confirmed by manipulation.

- **A goal object starts inside a closed container (22):** `boxing_food_after_dinner`, `bringing_water`, `can_meat`, `canning_food`, `clean_your_rusty_garden_tools`, `cook_a_brisket`, `cook_a_frozen_pie`, `cook_bacon`, `cook_brussels_sprouts`, `cook_cabbage`, `cook_hot_dogs`, `freeze_fruit`, `freeze_pies`, `halve_an_egg`, `make_gift_bags_for_baby_showers`, `make_pizza`, `preparing_lunch_box`, `setting_the_table`, `slicing_vegetables`, `thawing_frozen_food`, `unloading_the_car`, `wash_dog_toys`.
- **A door or jointed part on the shortest route to a goal object (11):** `bringing_paper_to_recycling`, `carrying_in_groceries`, `dispose_of_batteries`, `hiding_Easter_eggs`, `loading_the_car`, `moving_boxes_to_storage`, `putting_dirty_dishes_in_sink`, `sorting_household_items`, `tidying_bathroom`, `turning_out_all_lights_before_sleep`, `unloading_the_car`. Every door starts closed.
- **No standing point within ordinary reach (3):** `clean_up_your_desk` (one pen), `cook_bacon` (the bacon, inside the closed fridge), `hiding_Easter_eggs` (four trees; the nearest five satisfy the goal).
- **Something goal-relevant moved during two seconds of idling (5):** `composting_waste`, `dispose_of_batteries`, `sorting_household_items`, `store_batteries`, `tidying_bathroom`. No goal literal flipped.

### 11.3 Agent episodes

25 episodes were run, one activity each, all on the pull requests as first opened. Three succeeded (`turning_on_radio`, `chopping_wood`, `moving_boxes_to_storage`), two reached `q_score` 0.33 (`loading_the_car`, `set_up_a_coffee_station_in_your_kitchen`), 20 scored 0. Thirteen logs contain a tilt notice. One episode was ended by the first version of the state check at 0.549 rad beyond a wrist joint's range, a level that is now only stated. Nobody has read the logs for why the others failed, and 75 activities were never run.

### 11.4 Also not examined

- Other instances and seeds.
- Perception: whether the cameras show what each activity needs, objects in unlit closed containers, depth quality.
- Whether any upstream item is already fixed after the pinned BEHAVIOR-1K revision. Only the joint friction lines and the missing instructions were checked against upstream main.
