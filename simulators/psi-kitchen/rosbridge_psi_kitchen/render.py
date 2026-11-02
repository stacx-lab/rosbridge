"""Re-render an episode offline from its evidence (model.xml + steps.f64); nothing is simulated.

    python -m rosbridge_psi_kitchen.render EPISODE_DIR [--camera NAME] [--out FILE]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402


def frames(folder, camera='robot0_agentview_left', width=480, height=360, max_frames=300):
    """Yield RGB frames of the logged states, at most max_frames evenly spaced ones."""
    folder = Path(folder)
    meta = json.loads((folder / 'meta.json').read_text())
    rows = np.fromfile(folder / 'steps.f64').reshape(-1, len(meta['row']))
    model = mj.MjModel.from_xml_path(str(folder / 'model.xml'))
    data = mj.MjData(model)
    option = mj.MjvOption()
    option.geomgroup[0] = 0  # visual geoms only, as the robot's cameras show them
    with mj.Renderer(model, height, width) as renderer:
        for row in rows[np.unique(np.linspace(0, len(rows) - 1, max_frames).astype(int))]:
            data.qpos[:] = row[-meta['nq']:]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            yield renderer.render()


def render(folder, out=None, fps=10, **options):
    out = Path(out or Path(folder) / 'replay.mp4')
    stream = frames(folder, **options)
    first = next(stream)
    height, width = first.shape[:2]
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-pix_fmt', 'yuv420p',
               '-c:v', 'libx264', str(out)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for frame in (first, *stream):
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
    if encoder.returncode:
        raise RuntimeError(f'ffmpeg failed with {encoder.returncode}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folder')
    parser.add_argument('--camera', default='robot0_agentview_left')
    parser.add_argument('--out')
    args = parser.parse_args()
    print(render(args.folder, args.out, camera=args.camera))


if __name__ == '__main__':
    main()
