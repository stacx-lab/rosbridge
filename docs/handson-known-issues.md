# HandsOn: known environment issues

This page lists the problems known in HandsOn's 66 dexterous-manipulation tasks as rosbridge runs them: upstream `TsinghuaC3I/HandsOn` pinned at `c9598b53fc2a01355e74837de397bd7b2b1ab602`, in Isaac Sim 5.1 and Isaac Lab v2.3.2, with Core-50 (`t01`–`t50`) and Transfer-16 (`eval_comb01..06`, `eval_gen01..07`, `eval_flex01..03`). Upstream's only commit after `fee28452`, where these were found, publishes its leaderboard and changes no simulation code, so every upstream entry still holds. [HandsOn](handson.md) describes how rosbridge runs the simulator, including the corrections preparation applies. Problems of a harness that runs agents, outside this repository, are not listed.

## What rosbridge fixes

rosbridge reproduces the pinned evaluator: its physics, success and failure, budgets and instructions, so that results compare with upstream's. By that rule:

1. **rosbridge's own protocol** (how it drives, ends and reports an episode) is fixed wherever it departs from the evaluator's.
2. **An upstream defect** is corrected at preparation only when the correction leaves the scored task bit-identical (physics, success, failure and simulation time), restores what upstream evidently intends (its code, comments or published figures) or protects the host, and admits no other reading. Three do: Kit's present thread, the archived materials and `eval_gen06`'s spray cone (11.1–11.3).
3. **Any other upstream defect** is left as the evaluator has it, documented here and worth reporting upstream, even when it makes a task impossible: correcting one such task would make its scores incomparable while the others stay as they are. `eval_comb03`'s lid (8.1) is one.

## How to read this

**Evidence.** Each entry says how it is known:

- *measured*: a native run in Isaac Sim through rosbridge's own backend, driving the robot only through the commands a client sends (arm joint targets before the arms took palm poses; see 15);
- *read*: source code or data files;
- *inferred*: reasoning from measured or read facts, not itself run;
- *recorded*: seen in recorded agent episodes, with counts.

Unless stated otherwise, native runs used seed 0. A *positive control* is a scripted native test that reached the task's own success through client commands only, with no state written.

**Paths.** Upstream paths are relative to `.runtime/handson/handson/handson/`, the prepared upstream package (for example `sim/isaaclab/p002_contract.py:64`). Upstream's repository-level `configs/` and `docs/` are relative to `.runtime/handson/`. Isaac Lab paths are relative to `.runtime/handson/third_party/IsaacLab/source/isaaclab/isaaclab/`. *The instruction bank* is upstream's `tasks/assets/language_bank.json`, from which each episode's instruction is drawn. *The config prompt* is the `prompt` of upstream's `configs/multi_task/<task>.yaml`, a separate text; where it states a rule the instruction bank omits, the entry says so.

**Status.**

| Status | Meaning |
|---|---|
| rosbridge | Fixed or documented in rosbridge's protocol (rule 1) |
| prepared | Corrected at preparation (rule 2) |
| upstream | Needs a change in HandsOn (or Isaac Lab); rosbridge leaves it as the evaluator has it (rule 3) |
| open | No fix exists or is proposed |
| reported | Evidence only; a decision on changing it is pending |
| left as is | Known and deliberately unchanged, with the reason |
| property | Not a defect: how the environment works, worth knowing |

Nothing has been reported to HandsOn yet. Section 14 lists the reports worth filing.

## 1. All issues at a glance

| # | Task(s) | Issue | Status |
|---|---|---|---|
| 2.1 | all that can fail | A latched native failure was silent; the episode ran to its horizon | rosbridge |
| 2.2 | all without a success latch | A run judged on its final state loses a success reached earlier | rosbridge |
| 2.3 | 27 tasks; `t02`, `t20`, `t31`, `t45`, `eval_comb06` | No native failure; states where success is impossible without failing | rosbridge (documents) |
| 2.4 | `eval_comb01`, `eval_gen01/03/06/07` | Commands to the left side upstream holds were dropped silently | rosbridge |
| 2.5 | `t16`, `eval_comb02` | The hinge brake needs the commanded palm to rise; no commanded palm reached it | rosbridge |
| 2.6 | 11 tasks | Upstream's palm limits and guards were skipped on arm joint targets | rosbridge |
| 2.7 | `t22` | Upstream's IK joint limits did not bound joint targets | rosbridge (no longer applies) |
| 2.8 | 15 Core tasks | Tasks declared single-arm move the left arm natively | left as is |
| 3.1 | 14 Core tasks reject seeds | A seed the settle gate rejects failed the launch like any error | rosbridge |
| 3.2 | `t09` | The gate accepts a scene whose socket tray slid while settling | rosbridge (pre-screen) |
| 3.3 | 13 tasks; Transfer-16 | No settle gate; default gate parameters | property |
| 3.4 | `t24`, `t26` | Seed-only mode always asks for the tennis ball and the apple | left as is |
| 3.5 | all | Receipts beside the instruction are for the judge | rosbridge (documents) |
| 4.1 | all with free task bodies | Task bodies spin at most 30°/s: a deg/s value meant as rad/s | reported |
| 5.1 | `t07` | The skillet collides as a flat floor inside vertical walls, unlike its visual bowl | upstream |
| 5.2 | `t36` | The spoon's marble-only proxies collide with a 43 mm golf ball | upstream |
| 5.3 | `eval_flex01` | The mug's floor does not collide with the sponge | upstream |
| 5.4 | `eval_comb03` | A legacy contact profile, not its parent `t21`'s; misses blow up the solver | upstream |
| 5.5 | `eval_comb03` | The box lid's friction tuning is never applied | upstream |
| 5.6 | `eval_comb05` | The mug collides as a square under a round visual | upstream |
| 5.7 | `t32` | The neck collider and the checker's outlet are not where they look | upstream |
| 5.8 | `t31` | Visual-only parts, a mug that tips before it slides | property |
| 5.9 | `eval_comb01` | The box lid sticks to a light box | property |
| 5.10 | `eval_flex02` | Thick, sticky cloth, and a hand floor at its top surface | upstream |
| 6.1 | `t23`, `t47`, `t50`, `eval_comb04`, `eval_gen06` | Only right-hand contacts count where the instruction leaves the hand open | upstream |
| 6.2 | `t05`, `eval_comb05` | The fill counts only while both hands stay on | upstream |
| 6.3 | `t37`, `eval_comb05` | How the scrub is judged: tip point, bottle displacement, path, order | upstream |
| 6.4 | `t45`, `eval_comb06` | The stir is judged at a hidden point with a minimum radius | upstream |
| 6.5 | `eval_comb06` | The seasoning disk and a 4 s pour clock | upstream |
| 6.6 | `t08` | A valid hit is face-only, fast and aligned; the block must be lifted 3 cm | upstream |
| 6.7 | `t32` | Five conditions, three unstated | upstream |
| 6.8 | `t36` | Hand rules that fail the episode; lenient latches | upstream |
| 6.9 | `t38`, `eval_gen05` | Knob rules: speed, cross-hand cancel, 30° re-arm | upstream |
| 6.10 | `t07` | "Lift" counts only with the sponge on the middle of the blade | upstream |
| 6.11 | `t02` | Each connector seats at two rolls; one counts | upstream |
| 6.12 | `eval_comb03` | Hand-off and lid credits | upstream |
| 7.1 | `t17` | Instruction: left lever past 45°; checker: right lever at 25° | upstream (disclosed) |
| 7.2 | `eval_comb02` | Instruction allows either hand; only the right hand is wired | upstream |
| 7.3 | `t10` | Instruction's 12 N limit is waived for scoring | left as is (disclosed) |
| 7.4 | `t37` | Checklist label 6 cm, code 8 cm | upstream |
| 8.1 | `eval_comb03` | The box lid has no hand-contact view; the task can never succeed | upstream |
| 8.2 | `t24` | No release or speed check | upstream |
| 8.3 | `t19` | "Hand clear" is 2 cm from the cup's centre, inside the cup | upstream |
| 8.4 | `t32` | The height is measured from one sponge face; both look alike | upstream |
| 8.5 | `eval_comb01` | The pitcher-lid tolerance is centred off the physical mouth | upstream |
| 8.6 | `eval_flex01` | The packing threshold cannot be reached | upstream |
| 8.7 | `t45`, `eval_comb06` | One 0 N contact sample latches a permanent loss | upstream |
| 8.8 | `t20` | The pen-table force always reads 0 N | upstream |
| 8.9 | `t22` | Two failure guards are never called | upstream |
| 8.10 | `t35` | Only two of nine failure codes can be set | upstream |
| 8.11 | `t01` | Provisional calibration: the success latch is never armed | upstream |
| 9.1 | all | The record held no checklist and no failure cause | rosbridge (the cause) |
| 9.2 | `eval_comb06` | The stream is drawn hit or miss; seasoning progress is shown nowhere | upstream |
| 9.3 | all | Wrist camera mounts collide but do not count as hand contact | property |
| 10.1 | `eval_gen06` | A budget borrowed from `t50`, and a trigger out of reach in place | upstream |
| 10.2 | `eval_flex02` | Three quarters of the towel lies beyond pinch reach | upstream |
| 10.3 | `eval_comb03`, `eval_comb05` | Little slack in the budget | property |
| 11.1 | all | Kit's present thread stayed on in headless worlds | prepared |
| 11.2 | `t15`, `t27`, `t29`, `t30`, `t37` | Glass and some plastics rendered opaque red | prepared |
| 11.3 | `eval_gen06` | The spray cone never rendered | prepared |
| 11.4 | all | A three-view capture costs seconds; cameras sampled at 30 Hz | rosbridge |
| 11.5 | `t49` | Collision-only grip boxes are invisible | open |
| 11.6 | all | The head camera rendered without its lens; every principal point was half a pixel off | rosbridge |
| 12.1 | all | No torsional friction at fingertips; default friction 1.0/0.8 | property |
| 12.2 | all | Heavily damped PD arm drives | property |
| 12.3 | all but `eval_flex01` | Stiff finger drives | property |
| 12.4 | all | Hand collision shapes differ from the hand's look | property |
| 12.5 | `t22` | A grasp-assist spring holds the screwdriver; success latches at touchdown | property |
| 12.6 | all | What the demonstrations do and do not show | property |

Section 15 lists what has not been investigated. The largest item there is whether each task can be solved through palm pose targets at all: positive controls exist for six tasks, reached with the earlier joint targets.

## 2. Control path and episode protocol

### 2.1 A latched native failure was silent, and the episode ran to its horizon

A native failure latches at its first step, and success is impossible from then on, but nothing ended the episode or told the client; upstream's evaluator ends a trial at its first success or failure (`evaluation/rollout.py:119,140`). *Measured*: `t14` failed (a click before the mouse was on its pad) at step 100 of 375, and every wait reply until step 375 was `ok`. *Recorded*: rules that fail silently decided many episodes (`t36` 27 of 31, `eval_comb06` 12 of 32; 6.5, 6.8). Status: rosbridge. With `--early-stop` the episode ends where the evaluator ends a trial, at a success or a reported failure; the wait reply and the record name the end and the native cause. A step that succeeds and fails at once is a success, as upstream scores it.

### 2.2 A run judged on its final state loses a success reached earlier

Without `--early-stop`, `success` is what holds on the last step. Holds restart when a condition breaks, so a success can be lost before the horizon. Only `t07`, `t20`, `t22`, `t33`, `t34`, `t47`, `eval_comb04`, `t18`, `t49`, `eval_gen02`, `eval_flex01` and `eval_flex03` latch success. *Recorded*: one of 52 `t45` episodes met the checker at step 1224 and ended at step 1505 without success, after a regrasp rocked the pot and a single left-hand contact sample read 0 N (8.7). *Measured*: in a scripted `t32` run all five conditions held for 48 frames; lifting one finger 1 mm off the bottle's neck afterwards lost the success to the end. Status: rosbridge. The record carries `first_success_step`, and `--early-stop` scores the first success, as upstream does.

### 2.3 27 tasks cannot fail, and five states make success impossible without failing

- **What happens** (*read*). 27 tasks have no native failure and can only succeed or time out: `t02`–`t06`, `t08`, `t09`, `t12`, `t19`, `t21`, `t23`–`t26`, `t31`, `t32`, `t42`–`t45`, `eval_comb01`, `eval_comb03`, `eval_comb04`, `eval_gen01`, `eval_gen03`, `eval_gen07`, `eval_flex03`. Some states rule out success without raising a failure: in `t02` a cable seated before the plug (`sim/isaaclab/abc_collection_env.py:33449-33450`), in `t20` the left hand touching the pen first, in `t31` an interrupted brew, and in `t45` and `eval_comb06` a lost left-hand contact (`sim/isaaclab/b320_bimanual_env.py:4934-4941`, `sim/isaaclab/eval_comb06_season_and_stir_env.py:147-152`; the latch clears a success condition, `b320_bimanual_env.py:5845`, and sets no failure).
- **Effect.** An episode in such a state runs on to its horizon even with `--early-stop`.
- **Status.** rosbridge documents the 27 tasks and these five states in [HandsOn](handson.md); the states are upstream's.

### 2.4 Commands to a side that upstream holds were dropped silently

Upstream's control path holds the left arm and hand at home in `eval_comb01`, `eval_gen01`, `eval_gen03`, `eval_gen06` and `eval_gen07` and never asks for their targets (`sim/isaaclab/bimanual_assembly_env.py:3641-3662`), so left commands were accepted and had no effect. Status: rosbridge. `rosbridge setup` exports these tasks' `left_arm` and `left_hand` with their state only, so Kinex refuses a command at once; a client that publishes the topics directly is still ignored. The native suite checks the five tasks against upstream's control path. *Measured*: in `eval_gen01` a 4 cm lift commanded to both palms moved the right palm 4 cm and the left 0 mm.

### 2.5 `t16_open_laptop` and `eval_comb02`: the hinge brake needs the commanded palm to rise

- **What happens** (*read*). The display moves only through a contact-gated hinge (`sim/isaaclab/fu015_collection_env.py:389-425, 1799-1834, 1990-2022`). A brake holds it unless the right hand touches it; within 0.25° of closed, the brake also needs the commanded right palm to rise at least 0.2 mm in that step. A drive of at most 4 N·m then opens it 0.4° per step while contact and rise last. With joint targets in place of upstream's palm command, native logic never saw the commanded palm rise.
- **Status.** rosbridge. The arms take palm pose targets, which upstream's own IK executes, so the commanded palm upstream reads is the client's. *Measured*: in `t16` upstream's commanded right palm equalled the pose target exactly. Under the earlier joint-target control, with the commanded palm supplied by forward kinematics, targets that raised the palm a little each step along the hinge's arc with a fingertip on the display opened it to 57° in 6 s; that sequence has not been re-run with pose targets. `eval_comb02`'s left-hand route is still not wired (7.2).

### 2.6 Upstream's palm limits and guards were skipped on arm joint targets

- **What happened** (*read*). With joint targets in place of upstream's palm IK, the task's `_limit_arm_pose_target` and native logic that rewrites the commanded palm did not apply:
  - `eval_flex02`: the palm is lifted so that no finger collider goes below the table plus the towel's thickness, 0.90123 m (`sim/isaaclab/eval_flex02_fold_towel_twice_env.py:322-323, 330-434`);
  - `eval_flex01`: each step's palm within 4 cm and 8° of the measured palm, the hand above the table (`sim/isaaclab/ycb11_compress_sponge_env.py:376-414`);
  - `t22`: a wrist that carries the screwdriver keeps its latched attitude (`sim/isaaclab/as019_screwdriver_handover_env.py:1756-1788`);
  - the palm within 0.78 m of the arm's base in `t18`, `t21`, `t22`, `eval_comb03`, `eval_flex02`, `eval_flex03`, `eval_gen02`, `eval_gen04` (`sim/isaaclab/bimanual_assembly_env.py:128, 4450-4453`); the arm reaches 0.875 m;
  - upstream's default palm filter in the Transfer tasks without a contract, `eval_comb03`, `eval_flex03`, `eval_gen02`, `eval_gen04` (`bimanual_assembly_env.py:114-128, 4437-4498`);
  - `t10_lift_carton`: no palm rises until both hands grip with 10 N, then the palms move only together (`sim/isaaclab/b320_bimanual_env.py:3815-3851, 4134-4616`);
  - `t01`: the left hand is locked once the left block is level at height (`sim/isaaclab/b320_bc021_env.py:737-874`).
- **Evidence.** *Measured* under joint targets: in `eval_flex02`, fingers commanded 25 mm below the floor reached the table top, 10.9 mm through the cloth; in `t10`, palms commanded 5 cm up before any grasp rose 5.0 cm. *Recorded*: in `eval_flex02`, cloth caught on a hand in 11 of 11 and 5 or 6 of 10 episodes of two sets; in `t22`, every lift combined with a wrist turn while holding dropped the tool (4 of 4).
- **Status.** rosbridge. With palm pose targets, upstream's limits and guards run on the commanded palm as they do for its own policies. *Measured*: in `t10` a 4 cm lift commanded to both palms before any grasp moved neither; in ten other tasks the same lift converged within 1 s. Results under the earlier joint-target control are not comparable in these tasks. Under `eval_flex02`'s floor, flat cloth cannot be pinched (5.10).

### 2.7 `t22`: upstream's IK joint limits did not bound joint targets

Upstream's `t22` IK keeps joints within operational limits (`sim/envs/rokae_wuji_ik_replay/rokae.py:28-38, 89-107`), which client joint targets could leave. Status: no longer applies; the arms take palm pose targets through upstream's IK.

### 2.8 Core tasks declared single-arm move the left arm natively

- **What happens** (*read, measured*). Upstream's configs declare `robot_type: single_arm` for 19 tasks, and its policy wrappers hold the inactive side at its measured state (`eef52.py:74-86`). In 15 Core ones (`t03`, `t04`, `t09`, `t11`, `t14`, `t19`, `t24`, `t25`, `t26`, `t29`, `t33`, `t34`, `t44`, `t47`, `t50`) upstream's native control path drives the left arm: the frozen Core bases (`core_box_base`, `core_pitcher_base`, `core_peg_base`) have no left hold, and `sim/isaaclab/bimanual_assembly_env.py:3641-3649` exempts five tasks from it. A probe of upstream's control path in all 66 tasks agrees.
- **Evidence** (*recorded*). 11 of 25 successful episodes in these tasks manipulated with the left hand. In `t47` and `t50` left-hand presses never count (6.1).
- **Status.** Left as is: rosbridge follows the native control path. Refusing the left side would follow the policy protocol instead and would change these tasks' difficulty.

## 3. Seeds, the settle gate and receipts

### 3.1 A seed the settle gate rejects failed the launch like any other error

After the reset a stable-origin gate settles the scene and fails closed with `RecordingOriginStabilityError` if the reset drifts more than 5 mm or 3° from its authored pose or contacts penetrate (`sim/isaaclab/recording_origin.py`). Upstream's evaluator does not retry; the paper replaced a rejected candidate with the next accepted one. *Read*: upstream's archive of 45,000 accepted paper trials records 4.8 % of reset candidates rejected, all in tasks with the native recording gate (`t27` 43.0 %, `t23` 25.2 %, `t07` 19.9 %, then `t14`, `t19`, `t24`, `t04`, `t25`, `t26`, `t09`, `t48`, `t36`, `t40`, `t11`). Status: rosbridge. A launch's seed `k` names the task's `k`-th native seed that a pre-screen accepted (`simulators/handson/seeds.json`); see [HandsOn](handson.md).

### 3.2 The settle gate accepts a scene whose socket tray slid while settling: `t09_insert_peg`

At native seed 0 the gate accepts a reset in which the socket tray slides 6.1 cm while settling (*measured* by the pre-screen, identical on four GPUs). An earlier report of the tray thrown 6 m to the floor was not reproduced. Status: rosbridge. The pre-screen also rejects a native seed whose gate moved a task body more than 2 cm, which only this one did, so no launch seed names it; launches do not repeat the check. Upstream: the gate tests stillness, not that the scene is intact.

### 3.3 13 tasks have no settle gate, and Transfer-16 uses default gate parameters

`t05`, `t12`, `t13`, `t20`, `t21`, `t30`, `t31`, `t32`, `t37`, `t42`, `t43`, `t45` and `t46` have none: `evaluation/boundary.py:13` returns at once and `evaluation/task_contracts.py:47-51` returns before any gate. `t01`, `t15` and `t38` wait for the checker's readiness and stillness. Transfer-16 tasks have no contract, so their gate runs with code-default parameters (`sim/isaaclab/recording_origin.py:738-897`). *Read*. Status: property; [HandsOn](handson.md) lists the 13 tasks.

### 3.4 `t24` always asks for the tennis ball and `t26` for the apple

The target resolves to the first of the task's target objects unless a `target_object` override is given (`sim/isaaclab/collection_runtime.py:22-27`, `sim/isaaclab/core_box_env.py:2823, 9147-9151`). Only upstream's archived paper trials set the override: `t24` asks for the golf, mini soccer and tennis balls 186, 304 and 410 times of 900, and `t26` for four fruits. *Read, measured*. Status: left as is; rosbridge does not use the paper trials, and [HandsOn](handson.md) says so.

### 3.5 Receipts beside the instruction are for the judge

`PROMPT.json` holds `prompt` and other fields: `canonical_prompt` undoes the paraphrase draw, `goal` and `native_goal_readback` restate the target, `eval_gen05`'s readback gives the knobs' initial angles (`tasks/goal_readers.py:79-91`) and `eval_comb04`'s the fruit's id. `TASK_BOUNDARY_SETTLE.json` and `TASK_BOUNDARY.json` (`t01`, `t15`, `t38`) hold every object pose and joint position (`evaluation/boundary.py:39,61`, `evaluation/readback.py:55-65`), and `SETTING_SELECTION.json` holds `t38`'s knob angles. *Read and measured*. Status: rosbridge documents that only `prompt` goes to an agent.

## 4. Unit slip

### 4.1 Free task bodies spin at most 30°/s

- **What happens** (*read*). `sim/isaaclab/task_preview.py:5646-5651` gives every rigid body under `/World/Task` `RigidBodyPropertiesCfg(max_angular_velocity=30.0)`, in 65 of the 66 tasks; `t22`'s screwdriver gets 20 (`sim/isaaclab/as019_asset_support/scene_runtime.py:358`). Isaac Lab documents the field in deg/s (`sim/schemas/schemas_cfg.py:92-93`), and `modify_rigid_body_properties` (`sim/schemas/schemas.py:269`) writes it unchanged to `physxRigidBody:maxAngularVelocity`, whose PhysX unit is degrees per second (schema default 5729.58, that is 100 rad/s). So 30 means 0.524 rad/s. Upstream meant rad/s: its comment at `sim/isaaclab/e01_detached_hand_scene.py:60-65` calls the importer default "100 rad/s" and says a cap of 10 "still exceeds normal Tracker motion", which is true of 10 rad/s, not of 10°/s against human palm peaks of about 2 rad/s.
- **Evidence** (*measured*). A body given 5 rad/s and tossed: the `t19` cup reads 0.419 rad/s after one step, and 4.675 rad/s (its angular damping alone) with the cap raised to the schema default in the same process; the `t35` airplane, a floating articulation root, 0.526 rad/s. The `t36` golf ball settles at 0.52 rad/s, rolling at 1.13 cm/s, and keeps creeping at that speed after any touch. Not limited: robot links, which are joint-driven (after a 1 rad wrist jump the palm turned at 4.85 rad/s, 14 times the robot's own 20°/s cap at `sim/isaaclab/bimanual_assembly_scene.py:1142`), and articulation links that are not a root (knobs, propeller, lids, doors). In `eval_comb03` the block's cap is 4°/s, 0.07 rad/s (`sim/isaaclab/d_collection_env.py:88-90`).
- **Effect** (*inferred* beyond the bodies measured). Free objects that must roll, tip, flip or turn quickly are slowed: rolling balls in `t24`, `t27`, `t34` (at most 1.7 cm/s, computed) and `t36`; reorienting in `t19`, `eval_gen03`, `t07`, `t20`, `t25`, `t29`, `t33`, `t41`, `t42` and the blocks of `t01`, `t11`, `t44`. A knocked-over object takes at least 3 s to turn 90°. Held objects turned faster than the cap may lag or slip in the grasp (not measured).
- **Demonstrations** (*inferred*). They were recorded without an effective cap on objects. In `t19` the cup starts 84° from upright, must end within 25° and hold 0.5 s, which takes at least 2.5 s under the cap; more than half of the 500 demonstrations are shorter (median 2.2 s, shortest 1.1 s). `t34`'s demonstrations take a median 9.3 s for a roll of about 0.3 m that needs about 18 s under the cap. Budgets are three times the 75th percentile of demonstration length, so they come from motions the cap does not allow. The demonstrations' source commits differ from the pinned revision, so when the cap entered upstream cannot be told from the files.
- **Status.** Reported; a decision on changing it is pending. If 30 rad/s was meant, the value in Isaac Lab's unit is about 1719°/s; removing the cap gives PhysX's default. That is upstream's to decide; upstream's evaluator also asserts the `t21` block's 30 (`evaluation/setting_runtime.py:91-95`).

## 5. Physics and collider mismatches

### 5.1 `t07_flip_sponge`: the skillet collides as a flat floor inside vertical walls

- **What happens** (*read, measured*). The skillet's collision is a flat floor disc and 18 vertical wall slabs, 58.7 mm tall, with inner faces at a radius of 121.5–129.4 mm (`sim/envs/xmls/t07_flip_sponge_objects.xml` and its collider meshes). The visual is a filleted bowl rising from a radius of about 110 mm. `rim_clear` needs every point of the sponge's grid within 117.5 mm of the centre (`sim/isaaclab/ycb08_semantics.py:664-667`), so a sponge resting against the invisible wall fails while looking as if it lies on the floor; its end must be at least 17 mm off the wall. The blade's friction is 0.25/0.18 with combine mode `min` (`sim/isaaclab/t07_flip_sponge_env.py:93-98`), against 1.0/0.8 between sponge and pan.
- **Evidence.** *Measured*: a flipped sponge 0–16 mm from the wall gave a largest radius of 0.1275–0.1182 m against 0.1175. *Computed*: a free sponge tips onto the blade only below 28–35° of blade angle, while clearing the wall with the tip on the floor needs 34–42°. *Recorded*: 34 episodes, none successful; every end state examined failed `rim_clear` and the hidden `tool_under` (6.10).
- **Positive control.** None. With two teleports (a start at the wall and the final placement) and every history condition latched by agent actions, the checker reached success, so it is satisfiable; the missing step is physical. Human demonstrations use a wall as a backstop and nudge the flipped sponge toward the centre.
- **Fix.** Make the collision wall follow the visual bowl, or measure `rim_clear` to the collision radius.
- **Status.** Upstream.

### 5.2 `t36_scoop_marbles`: the spoon's marble-only proxies collide with a 43 mm golf ball

- **What happens** (*read, measured*). The "marble" is the YCB golf ball, radius 21.659 mm (`sim/isaaclab/abc_collection_env.py:4291-4299`). The spoon's invisible rim proxies and scanned lip hulls (`sim/envs/xmls/a_series_ycb_task_objects.xml:185-203`) collide in MuJoCo only with the 12 mm `a_marbles`; the PhysX import cannot keep that filtering (`abc_collection_env.py:1591-1596`). For the golf ball they form 36 mm side walls with a 45.0 mm slot (0.84 mm per side) and a lip edge 15–21 mm high at every pitch the face-up gate allows. All contacts use friction 1.0/0.8.
- **Evidence** (*measured*). All 10 spoon colliders are enabled, with no filtered pairs. Dropped in from above, the ball rests in the pocket; rolled at 0.05–0.8 m/s into a resting spoon, it never entered. Under the spin cap (4.1) it rolls at most 1.13 cm/s; faster pushes make it slide, and a sliding ball cannot climb the lip. In 17 scripted attempts with privileged poses, none got the ball into the pocket; one lifted it 8 mm onto the lip, where the lenient `loaded` latch fired (6.8).
- **Effect.** As built the task is unreasonably hard. There is no positive control and no proof of impossibility.
- **Fix** (upstream's to choose). Use the 12 mm marble; or put the proxies in a collision group that excludes the golf ball, after checking that the pocket still holds it; or widen the rims to about 50 mm and lower the lip hulls. The instruction should not call a golf ball a marble.
- **Status.** Upstream.

### 5.3 `eval_flex01`: the mug's floor does not collide with the sponge

- **What happens** (*measured*). Mug collider 9 is an analytic `UsdGeom.Cylinder` (`sim/envs/xmls/eval_flex01_compress_sponge_into_mug_objects.xml:23`, `sim/isaaclab/ycb11_scene.py:71-78`). The deformable sponge passes through it and rests on the table, sinking 15–25 mm into the table after a violent springback. The mug itself rests on the table through the same collider. That PhysX's GPU deformables do not collide with analytic cylinders is *inferred*.
- **Fix.** Replace collider 9 with its source 32-gon convex mesh (`mug_Collider9_m.obj`). On its own this does not make the task solvable (8.6).
- **Status.** Upstream.

### 5.4 `eval_comb03` runs a legacy contact profile, not its parent `t21`'s

- **What happens** (*read*). `t21` is a paper task: evaluator setting D03, profile `friction_4_4` or `friction_4_2p5` (`evaluation/settings.json`), block caps of 0.5 m/s depenetration, 1.0 m/s and 30°/s (`sim/isaaclab/b_eval_d03_physics.py`). `eval_comb03` has no paper setting and no contract, so the D path's defaults apply: block 0.15 m/s, 0.25 m/s and 4°/s (`sim/isaaclab/d_collection_env.py:75-96`, applied at 540–650), all 42 hand links 0.15 m/s, 2.0 m/s and 12°/s with 32/4 iterations (727–790), a 0.03 rad per step arm limiter and a free 90 g box. The block also starts at another pose than in `t21`.
- **Evidence.** *Measured*: the fingertip pinch that holds the block in `t21` under either profile held it 0 of 8 times at `eval_comb03`'s pose. Under `eval_comb03`'s profile every miss diverged the PhysX solver (4 of 4: NaN, or arm links above 500 rad/s throwing the block 1.5 m); under `t21`'s profile the misses became shoves (0 of 2 NaN). A palm rising 1 mm into a block the other hand still squeezed blew up at its first frame. *Recorded*: about 15 blow-ups in `eval_comb03` episodes, none in 8 `t21` episodes; contact-stopped grasps held the block 6 times.
- **Fix.** Give `eval_comb03` `t21`'s D03 profile. In the tests this removed the solver deaths, not the pinch's sensitivity to pose.
- **Status.** Upstream.

### 5.5 `eval_comb03`: the box lid's friction tuning is never applied

- **What happens** (*read*). The same `Box_lid` is tuned in `t03_cover_box` to 2.5/2.0 with combine mode `max` (`sim/isaaclab/abc_collection_env.py:2748-2753`, applied by `_configure_lid_grasp_physics`, 9923–9990), and `eval_comb01` applies `t03`'s tuning to its own lid (`sim/isaaclab/eval_comb01_cover_pitcher_and_box_env.py:131-138`). `eval_comb03` does neither; its world log prints "0 lid colliders friction-tuned" in every episode.
- **Evidence** (*measured*). The lid sticks against a slow push until about 5.6 N, then breaks free faster than the fingers and glides (deceleration 0.17 m/s², μ ≈ 0.02); on the rim it slid 3.3 cm when the fingers opened. Four push profiles kept right-hand-only contact for 1.9–3.4 cm of lid travel, short of the 5 cm the lid credit needs (6.12).
- **Fix.** Call `_configure_lid_grasp_physics(sim_utils, reference_task="t03_cover_box", lid_name="Box_lid")` for `eval_comb03`, as `eval_comb01` does.
- **Status.** Upstream.

### 5.6 `eval_comb05`: the mug collides as a square

The mug's collider is four wall boxes, 81 mm across the faces and 115 mm across the corners, under a round 81 mm visual (`sim/isaaclab/b320_bimanual_scene.py:2484-2509`). *Measured*: a top pinch at 45° to the mug's yaw landed a finger on a corner and stopped 7 cm high; aligned with the mug's yaw, the same pinch lifted it. Status: upstream.

### 5.7 `t32_press_soap_pump`: the neck collider and the outlet are not where they look

The bottle's neck collider is a cylinder of radius 20.8 mm from z 1.048 to 1.103, ending 1–3 mm under the pump head (`sim/isaaclab/b320_bimanual_scene.py:3667-3690`); the visible stem above the collar is about 5–6 mm wide. The checker's outlet is the far end of the head's box at mid-height, 3–6 mm above the visible spout tip (`b320_bimanual_scene.py:112, 114`). The pump head is a separate body (6.7). *Read and measured*. Status: upstream.

### 5.8 `t31_prepare_coffee`: visual-only parts, and a mug that tips before it slides

- **What happens** (*read, computed*). The mug collides as a 24-gon prism, radius 41.5 mm, 97 mm tall, with a closed top (the visual mouth has no collider); it weighs 0.22 kg with friction 1.0/0.8 (the MJCF's 1.25 is not used). A sideways push above 41.5 mm tips it at 0.9–1.8 N, less than the 2.16 N needed to slide it. The handle's window is 16.5 mm, smaller than the 17 mm fingertip sphere, so no finger threads it. The cabinet shelf, its centre post and the machine's pillars are visual only. The doors are free hinges (damping 0, `sim/envs/xmls/d_long_horizon_objects.xml:768-769, 803-804`). The button has no return spring, so a press latches, and leaving the brew early is a dead state (2.3). The table's near edge is at x −0.550.
- **Evidence** (*recorded*). 33 episodes; 7 of the 18 mugs taken out of the cabinet went off the near table edge. Upstream's 300 demonstrations follow one route: the left hand takes the mug out and sets it down, and the right hand wraps its body.
- **Status.** Property; the visual-only parts are upstream's.

### 5.9 `eval_comb01`: the box lid sticks to a light box

The 40 g lid has friction 2.5/2.0 with combine mode `max`; the 90 g box and the table have 1.0/0.8 (`sim/isaaclab/abc_collection_env.py:2240, 2748-2753`). A press-drag on the lid's top moves box and lid together once it presses more than about 0.85 N. The lid's collider is a solid 292 × 429 × 20 mm slab, the convex hull of the visual, with no skirt; it can rest only on the box's 19 mm rim. *Computed and recorded*: 22 pushing or sweeping episodes, none within coverage; a right-hand edge clamp, as in parent `t03`'s demonstrations, placed the lid within coverage 4 of 4 times. Status: property.

### 5.10 `eval_flex02`: thick, sticky cloth, and a hand floor at its top surface

- **Cloth** (*read, measured*). PhysX particle cloth, 41 × 41 particles, with a particle rest offset of 5.62 mm, so the sheet is 11.2 mm thick (a towel is 3–5 mm); friction is 1.6 for every contact: hand, table, layer on layer (`sim/isaaclab/marc_sawyer_towel.py:128, 193-218`). The hand collides as convex hulls of each finger part; self-collision is off, so curled finger hulls overlap 9–14 mm, and finger gaps of 9.0–10.4 mm are narrower than the 12 mm one layer needs. Cloth pressed into these crevices is clamped, and cloth draped over a digit holds e^(1.6θ) times the load.
- **Floor** (*read*). Upstream keeps every finger hull at or above the table plus the towel's thickness, 0.90123 m (`sim/isaaclab/eval_flex02_fold_towel_twice_env.py:385-434`): the top surface of flat cloth. A fingertip there touches only the top 0.5 mm of a flat layer and cannot pull it up. Upstream's own note on `flexible_hand_colliders.towel_hand_diagnostic` says closed fingers can stay above a flat towel.
- **Evidence** (*measured*). With the floor, 0 of 14 scripted pinches (one- and two-handed, placed with privileged poses) lifted cloth. Without it, 2 of 4 caught cloth, each with its lowest hull 4–10 mm into the cloth. The floor is soft: hand-to-hand contact and fast closes put hulls 3.7–4.5 mm below it.
- **Effect.** Under upstream's floor, which applies now that the arms take palm poses (2.6), flat cloth cannot be pinched. The only grasp that held, a two-hand squeeze of a whole end, gathers the edge so much that a perfect fold misses the 22 mm shape tolerance (28 mm, computed with upstream's checker). There is no positive control and no proof. Reach adds to it (10.2).
- **Fix** (upstream's to choose). Put the floor at the table plus the hand's rest offset, or at the particles' centre plane; exempt the fingertip hulls; or thin the cloth.
- **Status.** Upstream.

## 6. Checker rules a policy cannot infer from the instruction

### 6.1 Only right-hand contacts count where the instruction leaves the hand open

- **Tasks.** `t23` (shutter), `t47` (drill trigger), `t50` (spray trigger), `eval_comb04` (shutter), `eval_gen06` (spray trigger).
- **What happens** (*read*). `sim/isaaclab/collection_runtime.py:113-127` collects only bodies named `right_palm_link` or `right_finger*`, and the trigger, grasp and shutter contact views filter to them (`sim/isaaclab/core_pitcher_env.py:5180-5244`). A left finger moves the trigger and reads 0 N. The instruction bank names no hand for `t23`, `t50`, `eval_comb04` and `eval_gen06`; for `t47` it gives the drill to the right hand but not the press, and the config prompt says "Any grasp and any finger may be used" (`configs/multi_task/t47_trigger_drill_on_block.yaml:6`).
- **Evidence** (*recorded*). In `t50`, 3 of 3 episodes of one set gripped and aimed the bottle and pressed the trigger with the left index; no spray latched. In `t47`, 3 left-index presses never counted. Later episodes solved `t23`, `t47` and `t50` with the right hand.
- **Status.** Upstream. In `eval_gen06` upstream holds the left side, and its workspace has no left commands (2.4); in `t47` and `t50` the native path drives the left arm (2.8).

### 6.2 `t05_fill_cup` and `eval_comb05`: the fill counts only while both hands stay on

- **What happens** (*read*). A fill frame needs the left hand on the mug and the right hand on the lever, at least 0.10 N each, with the lever at least 70 % open and the mug aligned under the outlet, for 15 consecutive frames (`sim/isaaclab/core_faucet_env.py:5672-5710` in `t05`, `sim/isaaclab/b320_bimanual_env.py:6002-6039` in `eval_comb05`). The stream is a geometric test, not simulated water. The lever has no spring and stays open by itself, so letting go looks harmless. Upstream's own comment says success needs continuous right-hand contact with the handle (`sim/isaaclab/core_faucet_scene.py:3716-3717`); neither the instruction bank nor the config prompt says so.
- **Evidence.** *Recorded*: the fill condition was never met in 7 `t05` episodes; a mug held under the running stream for 1.1 s without the lever contact did not count. Later episodes solved `t05`. *Measured*: `eval_comb05`'s positive control kept a hooked index on the lever through the 15 frames.
- **Status.** Upstream.

### 6.3 `t37_scrub_bottle` and `eval_comb05`: how the scrub is judged

- **What happens** (*read*; `sim/isaaclab/abc_collection_env.py:1305-1312, 24815-25091`).
  - Bands are judged at a point 5.6 mm inside the bristle tip (brush-local (−0.088, 0, 0)), not where the bristles touch. They are thirds of the bottle's height with boundaries at bottle-local 0.0925 and 0.1575 m (world z 0.920, 0.9875, 1.0525, 1.120 at seed 0), within 69.7 mm of the bottle's axis. The label's three colour stripes all lie in the middle band.
  - A sample counts only with the left hand on the bottle, the right hand on the brush, bristles on the bottle, a safe grasp seen once, no right fingertip within 11.5 mm of the bristles, and the bottle within 40 mm of its reset pose and tilted at most 15° (25 mm at the end).
  - The counted path must exceed 8 cm. It adds the tip point's travel, at most 2 cm per substep, only while every contact holds.
  - Order: the first valid sample on an end band fixes the direction; every later one must be in the expected band or the one just finished, or a failure latches (`sim/isaaclab/xy4_env.py:138-142` in `eval_comb05`). A first sample in the middle band, a skipped band or a return fails at once, and a regrasp never resets progress.
  - Release needs no bristle contact and both hands at least 6 cm away, then 0.5 s still.
- **Evidence.** *Recorded*: in `t37` every agent moved the bottle 9–13 cm, voiding every sample; in `eval_comb05`, 37 episodes, 3 order failures, and the scrub never latched. *Measured*: `eval_comb05`'s positive control succeeded at step 2031 (67.7 s of 78.9 s), twice identically: a lower, middle and upper touch, each with a short stroke at constant radius.
- **Status.** Upstream.

### 6.4 `t45_stir_pot` and `eval_comb06`: the stir is judged at a hidden point

- **What happens** (*read*). The tracked point is the spoon's origin plus R·(0, 0, −0.10) (`sim/isaaclab/b320_bimanual_env.py:5802`): on the back face of the bowl, 2.5 cm past its centre and 2 cm from the tip. A sample is valid when that point is 2.5–12 cm from the pot body's origin (its bottom centre, with no tilt correction) horizontally and within 0.16 m of it vertically (`:5806`). Steps of at most 35° are summed: more than 650° and all six 60° world sectors are needed, in any order, and reversals subtract. The pot must stay within 25 mm. Nothing requires the spoon inside the pot. The instruction bank says "six ordered checkpoints" or "all six angular sectors"; nothing marks them, and `t45`'s scene contract text says "two full ordered circles" (`sim/isaaclab/b320_bimanual_scene.py:829-833`).
- **Effect.** A stir centred on the bowl at a 2–2.6 cm radius leaves the point under 2.5 cm on the far side. *Recorded*: four clean `t45` stirs, replayed, reached 3–5 of the 6 sectors with the left contact intact.
- **Positive controls** (*measured*). `t45` succeeded with the spoon's tip 3 cm deep in the pot and the point circling 3.1 cm about the axis (success at 37.7 s of 54.7 s), and also with the spoon circling 1.5 cm above the rim. `eval_comb06` succeeded with the point circling 2–3 cm above the rim (step 977, 32.6 s of 78.3 s).
- **Status.** Upstream.

### 6.5 `eval_comb06`: the seasoning disk and the pour clock

- **What happens** (*read*). A jar hits when it is held and tilted at least 100° and its cap's axis crosses the pot's opening plane and a plane 7.6 mm below it within 58 mm of the pot's origin, unobstructed (`sim/isaaclab/eval_comb06_season_and_stir_scene.py:36-39`, `sim/isaaclab/eval_comb06_season_and_stir_env.py:159-181`). Each jar needs 0.6 s of hits and both 0.4 s together. Each jar's tilted time is clocked, aimed or not, and a failure latches past 4.0 s before the hits are complete (`b320_bimanual_env.py:236-238`, `eval_comb06_season_and_stir_env.py:196-202`). `t30`'s instruction states its 4 s limit (`b320_bimanual_scene.py:700`); `eval_comb06`'s does not.
- **Evidence.** *Recorded*: the pour clock ended 12 of 32 episodes. Six chains that looked complete never left the seasoning phase: replaying their commands (*measured*), the salt's axis crossed 8.6 cm from the pot's axis, while the drawn stream entered the pot (9.2).
- **Status.** Upstream.

### 6.6 `t08_hammer_nail`: a valid hit, and a lifted block

- **What happens** (*read*; `sim/isaaclab/abc_collection_env.py:3709-3727, 40104-40236`). Success needs a valid hit latched at some earlier 120 Hz substep: the face point (not the butt) within 18 mm of the nail head, radial error at most 12 mm, axial offset −3 to +8 mm, the face normal within 25° of the nail's axis, and approach speed at least 0.20 m/s within 30° of the axis; no contact force or grip is needed. At the scored step the nail must be at least 6 mm deep and the block's bottom at least 30 mm above the table. The nail travels only 30 mm from 55.4 mm proud (`sim/envs/xmls/c01_wood_nail.xml:43`), so it is never flush; with the block on the table it stops after about 20 mm. Upstream's grasp and strike-align assists void success (`:40235`); both are off. The instruction bank says to hold the block and strike the nail "so it sinks into the block"; the config prompt says "Pick up the wood block … so the nail sinks slightly" (`configs/multi_task/t08_hammer_nail.yaml:6`). Neither states the lift height or the hit rule.
- **Evidence.** *Recorded*: 36 episodes, 0 valid hits; depth and lift held together in about 17. Butt presses and slow face presses move the nail and never count. *Measured* positive control: success at 21.9 s of 52.6 s, with the palm normal turned straight down (face alignment 0.9994), one control step to an arm target 6.5 cm below the hover (0.36 m/s at the head) and the block lifted 54 mm. The same chain with the face tilted 26° did not latch.
- **Status.** Upstream.

### 6.7 `t32_press_soap_pump`: five conditions, three unstated

- **What happens** (*read*; `sim/isaaclab/b320_bimanual_env.py:5885-5975`). For 12 consecutive frames: (a) the right hand on the bottle's body or neck with at least 0.10 N, the pump head being a separate body that does not count; (b) the right hand on the pump head; (c) the outlet over the sponge's face, 3 mm inside its edges; (d) the outlet 20–80 mm above the sponge, measured along the sponge's authored top face (8.4); (e) pump travel at least 75 %. There is no failure rule, and the last step decides. The instruction bank says to align the nozzle above the sponge with the right hand and depress the dispenser; the config prompt adds "hold the bottle with the right hand" (`configs/multi_task/t32_press_soap_pump.yaml:3`). Neither states (d) or the face.
- **Evidence.** *Recorded*: none of 40 episodes touched the bottle's body; all pressed only the pump head. *Measured* positive control: success from step 904 (30.1 s) to the end of the 41.1 s budget, with a flexed middle finger pressing the head and the straight index resting on the neck (0.48 N).
- **Status.** Upstream.

### 6.8 `t36_scoop_marbles`: hand rules that fail the episode, and lenient latches

- **What happens** (*read*; `sim/isaaclab/abc_collection_env.py:31897-31936, 40645-40707`). Any right-hand contact with the ball latches a failure, and so does left-hand contact while the ball is above 0.9467 m and outside the spoon's box. That threshold assumes a table at 0.89 m; it is at 0.8955 m, so the left hand may raise the ball 29.5 mm. The instruction bank states neither rule; the config prompt states both (`configs/multi_task/t36_scoop_marbles.yaml:6`). Lenient parts: `spoon_loaded_seen` fires once the ball has climbed 4–5 mm up the lip at 43–45°; neither the over-bowl latch nor the bowl check needs the ball carried in the spoon; the bowl check uses the bowl's own frame.
- **Evidence** (*recorded*). 27 of 31 episodes ended on the hand rules.
- **Status.** Upstream. See also 5.2.

### 6.9 `t38_set_dual_knobs` and `eval_gen05`: knob rules

- **What happens** (*read*; `sim/isaaclab/b320_ka005_2_contract.py:408-542`, which `eval_gen05` reuses). An attempt arms on the first contact if both knobs are at least 30° from their targets after 0.25 s without contact. Any cross-hand touch (a hand on the other side's knob, at least 0.15 N) cancels it, and re-arming needs both knobs 30° from their targets again, so a cross touch after one knob is set is a dead state until it is turned back. Each knob must first turn its way faster than 0.12 rad/s for 3 substeps with its own hand on it. Targets need ±5° to enter and ±7° to hold, then release and 0.5 s still. Only palm and finger links count. The instruction's angles are the knob joints' signed native angles (`tasks/goal_readers.py:81-92`), which no view displays.
- **Evidence.** *Measured*: at `eval_gen05` seed 26, where both knobs start at 0°, a scripted native test succeeded in 2 of 4 runs (at 19.6 s and 41.6 s of 43.1 s). A grasped knob's reported velocity is noisy (median 1.8–3.3 rad/s while it turned at 0.2 rad/s), so the speed rule rarely binds; *computed* with upstream's state machine, a smooth 6.8°/s turn exactly onto both targets never succeeds. A friction pinch passes on 10–83 % of the hand's turn, and opening the hand kicks the knob up to 23°.
- **Status.** Upstream.

### 6.10 `t07_flip_sponge`: "lift" counts only with the sponge on the middle of the blade

`tool_under` needs, in one physics substep, contact between spatula and sponge, at least 20 % of the sponge's 25 grid points over the blade, and the blade's mid-point (55 mm behind the tip) at least 2 mm below the sponge's centre (`sim/isaaclab/ycb08_semantics.py:684-704, 874-882`). Only after it does a 2 cm rise count as the lift, and only then can the flip count. A sponge raised 5 cm on the blade's tip counts for nothing; for a flat sponge the blade must be within about 5° of flat, possible only with the sponge stopped by a wall. *Read and measured*: `tool_under` never latched in any agent end state examined. Status: upstream. See 5.1.

### 6.11 `t02_connect_charger_and_cable`: each connector seats at two rolls, and one counts

The plug's prongs and the cable's tip are symmetric under a 180° roll and seat physically either way with about 4 mm of clearance (`sim/envs/xmls/c_tool_interaction_objects.xml:202-252`), but the checker also requires each connector's secondary axis within 15° (`sim/isaaclab/abc_collection_env.py:32822-32834, 32847-32859`). Cues: a blue accent on the plug's +z face, and the strip's indicator lights when the plug counts; the cable's indicator stays hidden until it connects. No instruction mentions orientation. *Read*; that about half of physically seated connectors do not count is *inferred*. Status: upstream.

### 6.12 `eval_comb03`: hand-off and lid credits

- **What happens** (*read*; `sim/isaaclab/d_collection_env.py:2249-2301`, `sim/isaaclab/xy1_composition.py:79-155`). The hand-off needs at least 2 consecutive frames with both hands at 0.15 N or more on the 20 g block; the block's own weight is 0.196 N, so a block resting in a palm reads barely above the threshold. The lid credit needs the lid moved at least 5 cm (or raised 1.5 cm) during uninterrupted right-hand-only contact, and a single left-hand-only lid frame revokes it even after it latched. Success holds 15 frames.
- **Evidence** (*measured*). A light press of the left middle fingertip on the block resting in the right palm completed the hand-off; squeezing it between both hands blew up the solver (5.4). A fingertip hook pull on the lid's far edge latched the lid credit at 5.1 cm with 0.5–1 N. No run combined both with a placed lid inside the 40.5 s budget (10.3).
- **Status.** Upstream. The right-hand rule matches the instruction; the revocation and the thresholds are unstated.

## 7. Instruction against checker

### 7.1 `t17_open_latched_door`: lever hand and angle

All 41 texts of the instruction bank say to press the near lever past 45° with the left hand and push the door with the right. The checker releases the latch only when the right hand touches the lever with it at 25° or more for 5 steps (`sim/isaaclab/p002_contract.py:64, 286-294`), and the checklist label says 45°. The config prompt states the checker's version (`configs/multi_task/t17_open_latched_door.yaml:7`). Upstream discloses this as a source-alignment waiver (`docs/custom_policy_integration.md:168`). *Recorded*: agents that followed the instruction turned the lever to 42° with the left hand, and the latch never released. Status: upstream (disclosed).

### 7.2 `eval_comb02_open_laptop_and_click_mouse`: either hand, but only the right is wired

The instruction allows either hand to open the laptop. The left-hand hooks in `sim/isaaclab/xy2_collection_env.py:70-86` are never called by the hinge code, so only the right hand's route can release the brake (2.5). *Read*. Status: upstream.

### 7.3 `t10_lift_carton`: the 12 N limit is waived

The instruction says neither side may exceed 12 N. Upstream's evaluator protocol waives force for scoring and keeps support, lift, a 3 s hold and a drop check (`evaluation/carton.py:13-81`, disclosed in `docs/custom_policy_integration.md:165-167`); the native 12 N check could not fire anyway, because forces are capped at 10 N before the peak is taken (`sim/isaaclab/b320_bimanual_env.py:3716-3725`). *Read*. Status: left as is: rosbridge runs the evaluator's protocol.

### 7.4 `t37_scrub_bottle`: checklist label 6 cm, code 8 cm

The checklist row says the scrub path must exceed 6 cm (`sim/isaaclab/abc_collection_env.py:38047`); the code requires 8 cm (`:1310`). *Read*. Status: upstream (label only).

## 8. Checker defects: lenient, dead or unreachable

### 8.1 `eval_comb03`: the box lid has no hand-contact view, so the task can never succeed

The D runtime builds hand-contact views only for the bodies `_D_CONTACT_TARGETS` lists, and `eval_comb03`'s entry lists the block alone (`sim/isaaclab/d_collection_env.py:196`). The lid's reading failed closed (`:1780-1781`), so the right-hand lid credit (`sim/isaaclab/xy1_composition.py:111-145`) could never latch. *Measured*: a right-hand hook pull moved the lid 9 cm with 0.00 N read; with the lid added to that entry, a fingertip press read 19 N and, with the block stage set complete, a 5 cm hook pull latched the lid credit. Status: upstream. Adding the lid makes a task that the evaluator scores 0 succeedable, which changes its scoring, so rosbridge does not (rule 3).

### 8.2 `t24_place_ball_in_bowl`: no release or speed check

Success needs the ball within 9.5 cm of the bowl's axis and 1.0–11.5 cm above its bottom for 6 steps (`sim/isaaclab/collection_runtime.py:1706-1709`, spec at `sim/isaaclab/core_box_env.py:2084-2091`), even while it is held and moving. *Recorded*: one success fired while the ball was in the closed hand, being lowered about 1.5 cm above the rim. Fix: require no hand contact and a resting ball. Status: upstream.

### 8.3 `t19_orient_cup_upright`: "hand clear" lies inside the cup

"Hand clear" means every palm and finger link origin at least 2 cm from the cup's centre (`sim/isaaclab/core_pitcher_env.py:1289`), less than the cup's 2.5–3.4 cm radius, so a hand wrapped around the cup passes; nothing checks contact or speed. *Recorded*: one success fired with the cup still in the closed hand, mid-motion. Fix: test contact, or distance from the cup's surface. Status: upstream.

### 8.4 `t32_press_soap_pump`: the height is measured from one face, and both faces look alike

Condition (d) of 6.7 measures height along the sponge's authored top face only (`sim/isaaclab/b320_bimanual_env.py:5900-5912`). A sponge held upside down, with the outlet 2–8 cm above its upper face, reads negative (−54 mm in a reproduced agent end state, *measured*), and both large faces render yellow. A sideways pinch followed by turning the hand palm-up flips the sponge. *Recorded*: this was the main reason ten end states that met every visible condition failed. Fix: measure along whichever face points up. Status: upstream.

### 8.5 `eval_comb01`: the pitcher-lid tolerance is centred off the physical mouth

- **What happens.** The pitcher check needs the lid's insert axis within 11 mm of the pitcher's base axis, its top −12 to +6 mm, and tilt at most 12° (`sim/isaaclab/abc_collection_env.py:2896-2912, 36172-36290`). In PhysX the seat disc is disabled and the lid's plug (radius 43 mm) sits among 17 wall colliders whose free play is not centred on that axis. Until the check holds, phase 1 never starts and nothing the box lid does counts (`sim/isaaclab/eval_comb01_cover_pitcher_and_box_env.py:140-165`).
- **Evidence.** *Measured*: lids dropped at several offsets settled at 1.0, 5.1 and 7.7–8.2 mm (pass), and at 11.2 and 13.1 mm toward the spout, flush and still (fail). *Recorded*: four episodes placed the box lid within coverage (0.949–0.991) and still failed; that the pitcher check was the cause is *inferred*.
- **Fix.** Centre the tolerance on the mouth's free play, or widen it.
- **Status.** Upstream.

### 8.6 `eval_flex01_compress_sponge_into_mug`: the packing threshold cannot be reached

- **What happens** (*read*). Success needs at least 85 % of the sponge's 268 collision nodes inside a cylinder of radius 30 mm (smaller than the physical cavity, as its docstring says), at least 80 % below the rim and the centre of mass 12 mm below it, held 1.0 s after both hands are 27.5 mm away (`sim/isaaclab/ycb11_semantics.py:37-73, 169-211`; criteria in `configs/multi_task/eval_flex01_compress_sponge_into_mug.yaml:32-41`). Once the packed stage latched, an inside fraction below 0.70 latches a failure at once. The sponge is purely elastic (E 5 kPa, no plasticity) and springs back within 2–4 control steps.
- **Evidence.** *Computed*: no placement of a sponge at or near its rest shape puts more than 0.526 of its nodes inside (20,000 orientations). *Measured*: 104 end states (wedged, folded, twisted, slowly withdrawn, 36 random) settled at 0.325–0.526 and latched the ejection failure. Upstream has no demonstration, scripted success or reported success for this task. 5.3 adds to it.
- **Fix** (upstream's to choose). Thresholds near 0.45 inside, 0.75 below the rim and 0.35 for ejection; or score "inside" against the physical cavity, with about 0.60.
- **Status.** Upstream.

### 8.7 `t45_stir_pot` and `eval_comb06`: one 0 N contact sample latches a permanent loss

- **What happens** (*read*). Once the left hand has touched the pot, any control sample below 0.05 N latches a loss of left contact for good (`sim/isaaclab/b320_bimanual_env.py:4934-4941`, `sim/isaaclab/eval_comb06_season_and_stir_env.py:147-152`), a dead state (2.3).
- **Evidence.** *Measured*: while a hand pressed steadily (about 3.5 N on both neighbouring samples, palm and pot still to 0.1 mm), the left-hand reading dropped to 0.0 N for exactly one sample, 5 times in two `eval_comb06` runs, during a hard right-hand squeeze on the spoon in the pot or an in-pot stir; none occurred in 1286 steps in which nothing pushed on the pot. *Recorded*: a `t45` episode lost a success it had reached this way (2.2).
- **Fix.** Require the loss on several consecutive samples.
- **Status.** Upstream.

### 8.8 `t20_pass_and_orient_pen`: the pen-table force always reads 0 N

The pen-table contact view filters on the work table (`sim/isaaclab/b320_bimanual_env.py:1850-1873`), which is a static collider in `t20` (a kinematic body only in `t10`); PhysX's GPU contact filter does not support it, and the world log warns so. *Measured*: the filtered force read 0.0 N for 30 steps with the pen resting on the table (net force 0.29 N). So the drop failure (`:3612-3625`) never fires and "table clear" (`:5484-5489`) is always true: lenient, not blocking. The checker's "tip" is the end with the cap and clip (*inferred* from the asset). Status: upstream.

### 8.9 `t22_pass_screwdriver`: two failure guards are never called

The drop-rollback and tip-inward guards (`sim/isaaclab/as019_screwdriver_handover_env.py:4327, 4474`) have no caller, so those failures never end an episode. *Read*. Status: upstream.

### 8.10 `t35_rotate_propeller`: only two of nine failure codes can be set

Of nine failure codes (`tasks/t35_rotate_propeller/semantics.py:22-34`), only non-finite state and propeller overspeed are ever set; displacement, lift, tilt, reverse turn and the others never fail. *Read*. Status: upstream.

### 8.11 `t01_align_blocks_in_midair`: the success latch is never armed

The task runs on a provisional acceptance calibration (`sim/isaaclab/b320_bc021_env.py:103-104`), so its success latch is never armed and its conditions are re-checked every step (`sim/isaaclab/b320_bc021_contract.py:1426`): a success can be lost after it is reached. *Read*. Status: upstream.

## 9. Feedback gaps: state that is judged but not visible

### 9.1 The record held no checklist and no failure cause

- **What happened** (*read*). Upstream's evaluator records binary success only (`evaluation/rollout.py:154-171`). Every environment does have a task-specific success checklist, `_success_condition_rows()`, which headless runs never compute (`sim/isaaclab/abc_collection_env.py:37512-37519`). A failure was recorded as `native_failed`; its cause existed only in per-task info entries such as `ycb01_failure_reason` and `b320_failure_reason`.
- **Status.** rosbridge, for the cause: with `--early-stop` a failure ends the episode, and the wait reply and the record give the native info that explains it, such as `native_failed: ycb01_failure_reason=ball_or_hand_touched_gate`. The checklist and a partial score are not recorded.

### 9.2 `eval_comb06`: the stream is drawn hit or miss

The cyan stream is drawn whenever a jar is held and tilted at least 100° (`sim/isaaclab/b320_bimanual_env.py:1086, 5555, 5590`), ending where its axis meets the receiver plane, wherever that is; the 58 mm disk and the raycast of 6.5 never affect it, and the seasoning's progress is shown nowhere. *Measured*: in the replay of one agent episode both streams enter the pot while the salt scored no hit. Fix: draw only hits, or colour them. Status: upstream.

### 9.3 Wrist camera mounts collide but do not count as hand contact

The hand-contact checks examined take only bodies named `<side>_palm*` and `<side>_finger*` (`sim/isaaclab/b320_bimanual_env.py:1698-1714`, `sim/isaaclab/collection_runtime.py:113-127`). The wrist camera mount sits 4.2 cm out along the palm normal and collides, as do the arm links. *Measured*: in `eval_comb06` a palm-side press moved the pot 1.2 cm while the left-hand reading stayed 0.0 N (the mount pushing is *inferred*); in `eval_flex02` the mount meets the cloth first and stalls the palm 1.7–2.3 cm above its target; in `eval_comb03` the arm's link 5 and wrist attachment pushed the lid with no force read. Status: property.

## 10. Budget and reach limits

### 10.1 `eval_gen06_spray_shoe`: a borrowed budget, and a trigger out of reach in place

- **What happens.** The budget, 13.9 s, is `t50`'s copied over (`evaluation/budgets.json:309-313` against 116–121, mode `semantic_3xp75`); the config allows 70 s. The bottle stands at the right arm's reach edge (its neck and trigger 0.72 m from the right shoulder, beyond the palm's 0.70 m reach from it) with its trigger facing sideways: every grip reachable in place leaves the trigger at least 15 mm beyond any fingertip, and grips that hook it need the bottle moved to x −0.30 or nearer (*computed*, IK on upstream's MJCF).
- **Evidence.** *Computed*: moving the bottle, regripping, aiming and squeezing takes 12.3–13.5 s at the speeds agents used. *Recorded*: 38 episodes, 0 successes, 13 failures for a spray latched off target; no agent saw a spray (11.3).
- **Status.** Upstream.

### 10.2 `eval_flex02_fold_towel_twice`: three quarters of the towel lies beyond pinch reach

In a pinch posture no fingertip reaches past x −0.10 (upstream's MJCF, joint limits and the 0.78 m palm radius), and the towel spans x −0.198 to +0.165 at seed 0, so 74 % of it lies beyond. It must first be dragged at least 26.5 cm toward the robot, which leaves its near edge 6 cm from the x −0.52 workspace bound. *Computed*. *Recorded* under the earlier joint-target control, without the floor (2.6): 36 episodes often formed the first fold and never left a released double fold. Fix: move the reset 20–25 cm toward the robot. Status: upstream.

### 10.3 `eval_comb03` and `eval_comb05`: little slack in the budget

`eval_comb05`'s 78.9 s is the sum of `t37`'s and `t05`'s budgets; its positive control, with no time spent deciding, used 67.7 s. `eval_comb03` has 40.5 s; the hand-off chain of 6.12 takes 21.1 s, leaving about 12 s for the lid credit, grip, carry and release. *Measured*. Status: property.

## 11. Rendering, assets and runtime

### 11.1 Kit's present thread stayed on in headless worlds

Isaac Lab's headless rendering app (`.runtime/handson/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit`) means to turn the present thread off, but names the extension `omni.renderer.core`; in Kit 107 (Isaac Sim 5.1) it is `omni.kit.renderer.core`, so the line had no effect. A headless world then presented about 300 frames a second, each raising a GPU interrupt, and with about five worlds per GPU a host locked up. *Measured*. Status: prepared (`patches/handson/0001` corrects the name; see [HandsOn](handson.md)). An Isaac Lab report.

### 11.2 Glass and some plastics rendered opaque red

Three of upstream's archived MDL files import Isaac Sim modules by relative name that upstream did not bundle beside them, so in `t15`, `t27`, `t29`, `t30` and `t37` the wine glass, the glass marble, the salt and pepper shakers, the water bottle and its label and `t15`'s window failed to compile and rendered opaque red, on any machine. *Measured*. Status: prepared (preparation links the three modules from the pinned Isaac Sim; see [HandsOn](handson.md)).

### 11.3 `eval_gen06_spray_shoe`: the spray cone never rendered

The cone imports as an Xform with a same-named child mesh. Setup hides every prim of that name (`sim/isaaclab/eval_gen06_spray_shoe_scene.py:239-242`), but the spray's show call un-hides only the first match, the Xform (`:198-203`, `sim/isaaclab/abc_collection_env.py:15511-15534`), so a latched spray, aimed or not, was invisible. *Measured*. The cone appears on the step a spray latches, which is also the step a miss fails, so it confirms a press but leaves no time to correct the aim. Status: prepared (`patches/handson/0002`; see [HandsOn](handson.md)).

### 11.4 A three-view capture costs seconds

Each camera view is path-traced; a three-view capture costs about 4 s (3.7–7 s measured) against 0.05–0.14 s per step, and at the shared 30 Hz camera rate a stepped run with a subscriber spent its time rendering. Status: rosbridge: HandsOn's cameras sample at 2 Hz of simulation time, and a new subscriber still gets a frame at once. *Measured*: in `t24` a persistent head-camera subscriber got frames at 0.5 s and 1.0 s over a 1 s wait, which took 15.5 s; one-shot reads took about 6 s.

### 11.5 `t49`: collision-only grip boxes are invisible

`sim/isaaclab/lw_native_articulation_scene.py:1416-1427` makes the grasp proxies collision-only and invisible, so the hand meets boxes it cannot see. *Read*. Status: open (minor).

### 11.6 The head camera rendered without its lens; every principal point was half a pixel off

Upstream renders the head camera through the Gemini 336's calibrated lens. Isaac's OpenCV lens schema carries the Brown-Conrady coefficients k1, k2, p1, p2 and k3; k4 to k6 are 0. rosbridge zeroed the lens so that Kinex could compute the camera's rays from its profile, which moved the image's edges by up to 4.3 px against the evaluator's. The lens schema also places every camera's principal point (upstream's K, authored by `sim/isaaclab/calibrated_cameras.py:apply_lens` on all three cameras) with pixel corners at integers, half a pixel from ROS and OpenCV pixel centers.

*Measured* against an undistorted spectator at each camera's pose, by phase correlation of textured patches:

- with the profiles' K as stated, the wrist images lay 0.37 px off in both axes, and so did the head image;
- with half a pixel added to (cx, cy), the wrists' mean offset is below 0.04 px;
- the head image's edges miss a pinhole by about 2 px, but line up with the calibrated lens within 0.3 px; scaling the lens by 0.8 or 1.2 fits worse.

Status: rosbridge.
- The head camera renders through its lens, as the evaluator does.
- Its profile states the lens (`opencv_brown_conrady`) and its field of view, and the world node serves `geometry`, `unproject` and `project` through that lens.
- `camera_info` carries the same plumb_bob D.
- Every profile's principal point is K's less half a pixel.
- HandsOn's native suite checks all three cameras this way.

## 12. Properties to know

### 12.1 No torsional friction at fingertips, and default friction

The hand's colliders and most task bodies bind no material, so the scene's default applies: 1.0 static, 0.8 dynamic, combine mode `average` (`sim/isaaclab/abc_collection_env.py:6923`). The MJCF friction values (hammer handle 5.0, hand 3.0) are not imported; task materials exist where code binds them (`t07` blade 0.25/0.18 `min`, `eval_gen06` bottle 0.9, `eval_comb05` bottle 0.75, `t03` and `eval_comb01` lid 2.5/2.0 `max`, the D03 block 4.0/2.5 `max`). No torsional patch radius is authored, so fingertip contacts carry no torsional friction (*inferred* from the attributes, consistent with the pivoting measured): a two-fingertip pinch is a free hinge about the line through its contacts. Only multi-finger wraps hold a tool against a load; 54 of 78 human `t08` demonstrations use a power grasp.

### 12.2 Heavily damped arm drives

The arms are implicit PD drives, 380/300/105 N·m/rad with damping 52/44/18 (joints 1–2 / 3–4 / 5–7), on gravity-free links (`sim/isaaclab/bimanual_assembly_scene.py:1030-1069, 1166-1176`): each joint is a first-order lag with a time constant of 0.14–0.17 s, and peak speed is about the distance still ahead divided by 0.18 s. *Measured*: one-step jumps of 3, 5 and 8 cm peak at 0.17, 0.28 and 0.45 m/s; a one-step 0.5 rad jump of joint 6 or 7 gives 0.56–0.59 m/s 15 cm along the fingers. So a motion aimed at a target arrives slowly, and a strike needs a target beyond the contact in one step, or a wrist flick. Joints 6 and 7 are limited to ±1.047 rad. Since the arms take palm poses, upstream's IK sets these joint targets each step.

### 12.3 Stiff finger drives

The fingers are implicit drives of 8 N·m/rad, damping 0.12, effort 6 N·m (`sim/isaaclab/bimanual_assembly_scene.py:1177-1182`); the source MJCF's actuators have kp 0.18–0.69 and 0.29–1.30 N·m (`sim/envs/xmls/rokae_wuji1_left.xml:291-310`). Closing past contact squeezes at tens of newtons (0.1 rad of error is about 9 N at a tip): it flings a 20 g block, tips a 0.22 kg mug and clamps cloth. `eval_flex01` replaces the hand drive with the source gains (`sim/isaaclab/ycb11_compress_sponge_env.py:360-372`).

### 12.4 Hand collision shapes

*Read and measured on all 66 tasks.* The palm is a 70 × 52 × 110 mm box whose face is 35 mm in front of the palm frame (`sim/envs/xmls/wuji_hand1/mjcf/right_primitive.xml:12`); links 2, 3 and 4 are capsules centred on their joints (radius 7, 6.5, 6 mm), with fingertip spheres of radius 8.5 mm. Link 2 collides with task objects, although upstream's comment says it sits in a non-contacting mask (`sim/isaaclab/abc_collection_env.py:7103-7104`). 16 tasks add link 1 capsules (radius 8 mm): `eval_comb01`, `eval_comb05`, `eval_comb06`, `eval_flex03`, `eval_gen02`, `eval_gen04`, `t02`, `t11`, `t12`, `t18`, `t29`, `t31`, `t34`, `t42`, `t43`, `t49`. `eval_flex01` and `eval_flex02` use convex hulls of the finger meshes instead (`sim/isaaclab/flexible_hand_colliders.py:85-138`). The arms collide as convex hulls of their link meshes. Self-collision is off on both robots, so one hand's colliders overlap freely; every robot shape collides with the other arm and hand.

### 12.5 `t22`: a grasp-assist spring, and success at touchdown

A contact-owned spring holds the screwdriver whenever a closed hand touches it, at 850 N/m, at most 45 N and 2.5 N·m, with an orientation guide (`sim/isaaclab/as019_screwdriver_handover_env.py:620-667`), for any policy. Success latches at the first substep the tool rests in the zone; a release, stillness or hold are "deliberately not success gates" (`:4386-4393`). *Recorded*: one success latched at touchdown with the left fingers still around the shaft.

### 12.6 What the demonstrations do and do not show

Transfer-16 tasks have no demonstrations (`docs/tasks.md:66-68`); upstream records no confirmed teleoperated or successful episode for `eval_comb01` and marks `eval_flex02`'s collection unconfirmed. Core demonstrations were recorded under collection-time physics from source commits that differ from the pinned revision (see 4.1). A replayed `t37` demonstration passes through `eval_comb05`'s sink walls and counter, so `t37`'s demonstrations do not transfer there. Budgets are three times the 75th percentile of demonstration length (`evaluation/budgets.json`); the `eval_flex` tasks keep their native durations.

## 13. How the corrections at preparation were checked

At the pin, on L40S GPUs, the prepared tree with the corrections was compared with the same tree without them (*measured*):

- **Physics.** `t24` at seed 0, 30 idle steps and then 60 lifting both palms 4 cm: every step's joint positions and velocities and the poses of its five task bodies are bit-identical with and without the corrections, and between runs on two GPUs.
- **Present thread (11.1).** An idle world raised 269 interrupts a second on its GPU and used 0.21 cores; corrected, none and 0.10 cores. The host lockups at about five worlds per GPU were seen on the development line.
- **Materials (11.2).** `t29` logs no MDL compile error, and its wine glass renders clear instead of opaque red: 4,396 pixels of the head view change, all on the glass.
- **Spray cone (11.3).** After upstream's own show call in `eval_gen06`, the cone's child mesh is visible and 1,318 pixels of the head view change; uncorrected, it stays invisible and none change.

The unit suites, the native suite and the asset checks of all 66 tasks pass with the corrections.

## 14. Reports worth filing with HandsOn

None has been filed.

| # | Report | Reproduction |
|---|---|---|
| U1 | `max_angular_velocity=30.0` is in deg/s (4.1) | Toss the `t19` cup at 5 rad/s: 0.419 rad/s after one step; 4.675 with the cap at the schema default |
| U2 | Isaac Lab's headless kit names the present-thread extension `omni.renderer.core`, not Kit 107's `omni.kit.renderer.core` (11.1); for Isaac Lab | One headless world: about 300 presents a second |
| U3 | Archived MDL files without the modules they import (11.2) | `t29`: the wine glass renders opaque red |
| U4 | `eval_gen06`'s name-based hide and first-match show disagree (11.3) | Latch a spray: the child mesh stays invisible |
| U5 | `eval_comb03`: lid contact view (8.1), lid friction tuning (5.5), legacy contact profile (5.4) | A hook pull moves the lid 9 cm with 0 N read; "0 lid colliders friction-tuned"; `t21`'s pinch at `eval_comb03`'s pose diverges |
| U6 | `eval_flex01`: the floor cylinder does not collide with the sponge (5.3); the threshold is unreachable (8.6) | Any released sponge settles at 0.33–0.53 inside |
| U7 | `eval_flex02`: floor at the cloth's top surface, cloth thickness and friction, reset beyond reach (5.10, 10.2) | Pinch flat cloth with the floor on |
| U8 | `t07`: collision wall against visual bowl and `rim_clear` (5.1, 6.10) | A green-up sponge touching the wall reads 0.1275 m against 0.1175 |
| U9 | `t36`: marble-only proxies collide with the golf ball (5.2) | Roll the ball into a resting spoon |
| U10 | Lenient or dead checks: `t24` (8.2), `t19` (8.3), `t20` (8.8), `t22` (8.9), `t35` (8.10), `t01` (8.11) | Read; `t20`'s filtered force is 0 N with the pen on the table |
| U11 | `t32`'s one-face height (8.4); `eval_comb01`'s off-centre pitcher tolerance (8.5) | A flipped sponge reads −54 mm; a flush lid at +11 mm fails |
| U12 | A single 0 N contact sample latches a permanent loss (8.7) | Press the pot steadily while squeezing the spoon in it |
| U13 | Rules no instruction states (6.1–6.12) | The entries' file:line references |
| U14 | Instruction against checker: `t17` (disclosed), `eval_comb02`, `t37`'s label (7.1, 7.2, 7.4) | Read |
| U15 | `eval_gen06`'s budget copied from `t50` (10.1) | `evaluation/budgets.json:309-313` |
| U16 | `eval_comb06`'s stream drawn on a miss (9.2); `t49`'s invisible grip boxes (11.5) | A salt pour 8.6 cm off the axis draws into the pot |
| U17 | `apply_lens` authors OpenCV's K on Isaac's lens schema, which places the principal point with pixel corners at integers: the cameras render half a pixel off their calibration (11.6) | Compare a camera with an undistorted pinhole at its pose |

## 15. Not investigated

- **Whether each task can be solved through palm pose targets.** Positive controls exist for `t08`, `t32`, `t45`, `eval_comb05`, `eval_comb06` and `eval_gen05` (at seed 26), all reached with the earlier arm joint targets and not re-run since the arms take palm poses; `t08`'s, for one, used a one-step joint jump for its strike. `t07`, `t36`, `eval_comb03` and `eval_flex02` are undetermined after scripted attempts, `eval_flex01` is shown impossible, and `t31` and `eval_comb01` were analysed against demonstrations and one scripted drop test. The other 53 tasks were not examined this way.
- **Other seeds.** Almost everything here is seed 0. Other seeds change layouts and goals.
- **The spin cap's effects** beyond the bodies measured: grasp slip of held tools turned faster than the cap, and how success rates change with it.
- **`eval_comb06`'s in-pot stir**, the instruction's natural reading: three scripted attempts failed at the handle regrasp.
- **Perception**: whether the cameras show what each task needs.
- **Whether any upstream item is fixed after the pinned revision.** Nothing here was compared with upstream's current branch.
