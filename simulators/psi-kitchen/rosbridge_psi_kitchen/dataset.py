"""RoboCasa365's human demonstrations of a task, in the robot's own terms: what its devices report and see at each
control step, and the command that reproduces the recording.

    python -m rosbridge_psi_kitchen.dataset TASK --out DIR [--kitchens 1-5] [--size 128] [--cache DIR] [--limit N]
                                               [--jobs N]

TASK is a psi-kitchen task id (open-drawer). Its target-split human dataset (500 teleoperated demonstrations over
target kitchens 1-10, the kitchens PSI_SCENE builds) is downloaded into --cache once. Each episode from --kitchens is
restored from its recorded states into a psi-kitchen robot (backend.PsiKitchenSim, the same controllers and device
readings as a live run):

- state: the robot's device readings, as backend state() gives them;
- images: the three cameras at the robot's 256 x 256, resized to --size with cv2.INTER_AREA (resize());
- action: the recorded command in the devices' terms (the backend's step() vector): the arm target is the joints for
  the hand pose upstream's Cartesian arm controller aimed at (aim()), the torso target its position plus the recorded
  torso command, the gripper target and base velocity the recorded commands.

An episode is kept only if its commands, replayed from its first state, do the task again. DIR gets RLE-style shards
(no simulator states, models or object names): camera_{left,right,wrist}.u8 (N, S, S, 3) uint8, state.f32 (N, 12),
action.f32 (N, 12), episode_end.i32 (N,) (each frame's exclusive episode end), episodes.jsonl and meta.json.
"""
import argparse
from contextlib import ExitStack
from functools import cache
import gzip
import json
import os
from pathlib import Path

import cv2
import numpy as np

from rosbridge.catalog import resolve
from .backend import CAMERAS, ROBOT, PsiKitchenSim

FPS = 20
STATE = ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6', 'arm_joint7',
         'gripper', 'torso', 'base_x', 'base_y', 'base_yaw']
ACTION = ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6', 'arm_joint7',
          'gripper', 'torso', 'base_vx', 'base_vy', 'base_wz']
UNITS = {
    'state': 'arm joint positions (rad, /arm/joint_state), gripper command (-1 open .. +1 closed, '
             '/gripper/joint_state), torso height (m, /torso/joint_state), base pose relative to the episode start '
             '(x, y in m, yaw in rad, /base/odometry)',
    'action': 'arm joint targets (rad, /arm/joint_target: the joints for the hand pose the operator commanded), '
              'gripper target (-1 .. +1, /gripper/target), torso target (m, /torso/joint_target), base velocity (body '
              'frame, each in -1 .. 1, /base/velocity linear.x, linear.y, angular.z), held for one 0.05 s control '
              'step',
}


def resize(image, size):
    """A 256 x 256 camera frame as the shards hold it."""
    return image if image.shape[0] == size else cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def source(task, cache):
    """The task's target human dataset (LeRobot layout) under cache, downloaded on first use."""
    import robocasa.macros as macros
    macros.DATASET_BASE_PATH = str(cache)
    from robocasa.utils.dataset_registry_utils import get_ds_meta
    meta = get_ds_meta(task=task, split='target', source='human')
    if meta is None:
        raise ValueError(f'RoboCasa has no target human demonstrations of {task}')
    path = Path(meta['path'])
    if not (path / 'meta').is_dir():
        from robocasa.scripts.download_datasets import download_and_extract_from_box
        download_and_extract_from_box(destination=str(path))
        if not (path / 'meta').is_dir():
            raise RuntimeError(f'download of {task} failed')
    return path


def selected(folder, kitchens, limit=None):
    """Indices of the episodes recorded in `kitchens` (at most `limit` per kitchen), in order."""
    counts = dict.fromkeys(kitchens, 0)
    for extras in sorted((folder / 'extras').glob('episode_*')):
        meta = json.loads((extras / 'ep_meta.json').read_text())
        kitchen = meta['layout_id']
        if kitchen in counts and not (limit and counts[kitchen] >= limit):  # target kitchens: style = layout
            counts[kitchen] += 1
            yield int(extras.name.split('_')[1])


def load(folder, index):
    """An episode: (ep_meta, model xml, states, recorded commands {hand (N, 6): the Cartesian arm command, mode (N,):
    > 0 in base mode, grip (N,): the gripper direction, base (N, 3): velocity, torso (N,)})."""
    import pyarrow.parquet as pq
    modality = json.loads((folder / 'meta/modality.json').read_text())['action']
    extras = folder / 'extras' / f'episode_{index:06d}'
    data = next(folder.glob(f'data/*/episode_{index:06d}.parquet'))
    actions = np.stack(pq.read_table(data, columns=['action']).column('action').to_pylist())
    part = {key: actions[:, modality[key]['start']:modality[key]['end']] for key in modality}
    commands = {'hand': np.hstack([part['end_effector_position'], part['end_effector_rotation']]),
                'mode': part['control_mode'][:, 0], 'grip': part['gripper_close'][:, 0],
                'base': part['base_motion'][:, :3], 'torso': part['base_motion'][:, 3]}
    return (json.loads((extras / 'ep_meta.json').read_text()),
            gzip.decompress((extras / 'model.xml.gz').read_bytes()).decode(),
            np.load(extras / 'states.npz')['states'], commands)


def instruction(ep_meta):
    """The episode's task sentence as a live robot states it (upstream fills in object names some recordings keep as
    placeholders)."""
    from robocasa.scripts.dataset_scripts.playback_utils import resolve_instruction_from_ep_meta
    return resolve_instruction_from_ep_meta(ep_meta) or ''


def restore(sim, ep_meta, xml, state):
    """Put the episode's kitchen and first state into sim's env, as upstream playback does, and start an episode."""
    env = sim.env
    env.set_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env.update_state()
    sim.settle()
    sim.max_steps = None


def joints_for(sim, scratch, pos, ori):
    """Arm joints that put the hand at (pos, ori) in the world, within the joint limits: mass-weighted Newton steps
    (as the Cartesian controller moves the arm) from the current joints, on a scratch copy of the state."""
    import mujoco as mj
    from robosuite.utils.control_utils import orientation_error
    arm, model = sim.robot.part_controllers['right'], sim.model
    address, site = model.jnt_qposadr[arm.joint_index], model.site(arm.ref_name).id
    inverse = np.linalg.inv(arm.mass_matrix)
    jacobian = np.zeros((6, model.nv))
    scratch.qpos[:] = sim.data.qpos
    q = arm.joint_pos.copy()
    for _ in range(10):
        scratch.qpos[address] = q
        mj.mj_kinematics(model, scratch)
        mj.mj_comPos(model, scratch)
        error = np.r_[pos - scratch.site_xpos[site], orientation_error(ori, scratch.site_xmat[site].reshape(3, 3))]
        if np.abs(error).max() < 1e-6:
            break
        mj.mj_jacSite(model, scratch, jacobian[:3], jacobian[3:], site)
        J = jacobian[:, arm.qvel_index]
        q = q + inverse @ J.T @ np.linalg.solve(J @ inverse @ J.T, error)
    return np.clip(q, *model.jnt_range[arm.joint_index].T)


def aim(sim, scratch, hand, base_mode, goal, scale):
    """The arm's joint target for one recorded Cartesian command at the current (restored) state, and the hand goal
    it aims at, in the arm base's frame. As upstream's OSC_POSE controller sets its goal: the current hand pose moved by
    the command (scaled by `scale`), or, in base mode, the previous goal moved by it."""
    import robosuite.utils.transform_utils as T
    arm = sim.robot.part_controllers['right']
    arm.update(force=True)
    site = sim.model.site(f'{arm.naming_prefix}{arm.part_name}_center').id
    origin, frame = sim.data.site_xpos[site], sim.data.site_xmat[site].reshape(3, 3)
    if goal is None or base_mode <= 0:
        goal = frame.T @ (arm.ref_pos - origin), frame.T @ arm.ref_ori_mat
    move = np.clip(hand, -1, 1) * scale
    goal = goal[0] + move[:3], T.quat2mat(T.axisangle2quat(move[3:])) @ goal[1]
    return joints_for(sim, scratch, origin + frame @ goal[0], frame @ goal[1]), goal


def labels(positions, arm, torso, grip, base):
    """Commands in the devices' terms: the arm and torso targets, the recorded gripper direction as a target (0 holds
    the command), the recorded base velocity."""
    action = np.empty((len(positions), 12))
    action[:, :7], action[:, 8] = arm, torso
    action[:, 7] = np.where(grip > 0, 1., np.where(grip < 0, -1., positions[:, 7]))
    action[:, 9:] = base
    return action


def grip_commands(start, grip, speed):
    """The robot's integrated gripper command over the episode, as the native gripper integrates the recorded
    directions: one value per recorded state."""
    command = [start]
    for direction in grip[:-1]:
        command.append(float(np.clip(command[-1] + speed * np.sign(direction), -1, 1)))
    return np.array(command)


@cache
def arm_scale():
    """Upstream's Cartesian arm command scale (OSC_POSE output_max), under which the demonstrations were recorded."""
    from robosuite.controllers import load_composite_controller_config
    return np.array(load_composite_controller_config(robot=ROBOT)['body_parts']['right']['output_max'])


def convert(sim, ep_meta, xml, states, commands, size):
    """One episode in the robot's terms: (state, images {camera: frames}, action, replayed successfully)."""
    import mujoco as mj
    restore(sim, ep_meta, xml, states[0])
    start, speed = sim.state()[0][7], sim.robot.gripper['right'].speed
    torso, scratch = sim.robot.part_controllers['torso'], mj.MjData(sim.model)
    positions, arm, lift, goal, images = [], [], [], None, {camera: [] for camera in CAMERAS.values()}
    for state, hand, base_mode, up in zip(states, commands['hand'], commands['mode'], commands['torso']):
        sim.env.sim.set_state_from_flattened(state)
        sim.env.sim.forward()
        positions.append(sim.state()[0])
        target, goal = aim(sim, scratch, hand, base_mode, goal, arm_scale())
        arm.append(target)
        lift.append(positions[-1][8] + torso.scale_action(np.array([up]))[0])
        for native, camera in CAMERAS.items():
            images[camera].append(resize(sim.render(native), size))
    positions = np.array(positions)
    positions[:, 7] = grip_commands(start, commands['grip'], speed)
    action = labels(positions, np.array(arm), np.array(lift), commands['grip'], commands['base'])
    restore(sim, ep_meta, xml, states[0])
    success = False
    for command in action:
        sim.step(command)
        success = success or sim.success
    return positions, {camera: np.array(frames) for camera, frames in images.items()}, action, success


SIM = None  # each conversion process's robot


def start(task, kitchen):
    global SIM
    os.environ['PSI_SCENE'] = str(kitchen)
    SIM = PsiKitchenSim()
    SIM.reset(task, 0)


def job(args):
    folder, index, size = args
    ep_meta, xml, states, commands = load(folder, index)
    return index, ep_meta, convert(SIM, ep_meta, xml, states, commands, size)


def build(task, out, kitchens, size=128, cache=None, limit=None, jobs=1):
    """Convert the task's demonstrations from `kitchens` into shards in `out`, with `jobs` processes; returns
    meta.json's content."""
    from concurrent.futures import ProcessPoolExecutor  # a crashed process fails the build; a Pool would hang
    from multiprocessing import get_context
    native = resolve('psi-kitchen', task=task)[2]['native_task']
    out = Path(out)
    folder = source(native, Path(cache or out.parent / 'raw'))
    out.mkdir(parents=True, exist_ok=True)
    counts = {k: {'found': 0, 'kept': 0} for k in kitchens}
    frames = 0
    jobs_in = ((folder, index, size) for index in selected(folder, kitchens, limit))
    with ProcessPoolExecutor(jobs, get_context('spawn'), initializer=start, initargs=(task, kitchens[0])) as pool, \
            ExitStack() as stack:
        files = {name: stack.enter_context((out / name).open('wb')) for name in
                 (*(f'{camera}.u8' for camera in CAMERAS.values()), 'state.f32', 'action.f32', 'episode_end.i32')}
        listing = stack.enter_context((out / 'episodes.jsonl').open('w'))
        for index, ep_meta, (state, images, action, success) in pool.map(job, jobs_in):
            kitchen = ep_meta['layout_id']
            counts[kitchen]['found'] += 1
            print(f'[dataset] {task} episode {index} kitchen {kitchen}: {len(state)} steps, '
                  f'{"kept" if success else "dropped (the replay failed)"}', flush=True)
            if not success:
                continue
            counts[kitchen]['kept'] += 1
            end = frames + len(state)
            for camera, array in images.items():
                files[f'{camera}.u8'].write(array.tobytes())
            files['state.f32'].write(state.astype(np.float32).tobytes())
            files['action.f32'].write(action.astype(np.float32).tobytes())
            files['episode_end.i32'].write(np.full(len(state), end, np.int32).tobytes())
            listing.write(json.dumps({'start': frames, 'end': end, 'kitchen': kitchen,
                                      'instruction': instruction(ep_meta)}) + '\n')
            frames = end
    meta = {'task': task, 'fps': FPS, 'n_frames': frames,
            'n_episodes': sum(c['kept'] for c in counts.values()), 'image_size': size,
            'cameras': list(CAMERAS.values()), 'state': STATE, 'action': ACTION, 'units': UNITS,
            'episodes_per_kitchen': {str(k): c['kept'] for k, c in counts.items()},
            'dropped_per_kitchen': {str(k): c['found'] - c['kept'] for k, c in counts.items()}}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def kitchen_list(text):
    """'1-5' or '1,3,5'."""
    if '-' in text:
        first, last = text.split('-')
        return list(range(int(first), int(last) + 1))
    return [int(k) for k in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('task')
    parser.add_argument('--out', required=True)
    parser.add_argument('--kitchens', type=kitchen_list, default=kitchen_list('1-5'))
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--cache', help='download folder (default: OUT/../raw)')
    parser.add_argument('--limit', type=int, help='at most this many episodes per kitchen')
    parser.add_argument('--jobs', type=int, default=1, help='conversion processes')
    args = parser.parse_args()
    meta = build(args.task, args.out, args.kitchens, args.size, args.cache, args.limit, args.jobs)
    print(json.dumps({key: meta[key] for key in ('task', 'n_episodes', 'n_frames', 'episodes_per_kitchen',
                                                  'dropped_per_kitchen')}), flush=True)


if __name__ == '__main__':
    main()
