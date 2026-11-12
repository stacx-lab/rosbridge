"""Independent MuJoCo spectator framebuffer and depth; capture never advances physics."""
from contextlib import contextmanager

import numpy as np


def refresh(model, data):
    """mj_step integrates qpos after computing positions: refresh frames for cameras and
    judging without touching dynamics or the solver's warm start."""
    import mujoco
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    mujoco.mj_camlight(model, data)


def render_depth(renderer, model):
    """(H, W) float32 metres along the optical axis from a mujoco.Renderer with an updated
    scene; 0 where nothing is within the clip range."""
    try:
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    near, far = model.vis.map.znear * model.stat.extent, model.vis.map.zfar * model.stat.extent
    # The far plane is reconstructed through float32 projection terms; keep a 0.1% margin.
    valid = np.isfinite(depth) & (depth > near) & (depth < far * (1 - 1e-3))
    return np.where(valid, depth, 0).astype(np.float32)


class SpectatorRenderer:
    def __init__(self, model, data, views, *, restore_context=None, options=None):
        import mujoco as mj
        self.model, self.data, self.views = model, data, {v.name: v for v in views}
        self.gl = self.context = None
        self.restore_context = restore_context
        try:
            self.camera_ids = {}
            for view in views:
                camera = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, view.source)
                if camera < 0:
                    raise ValueError(f'unknown spectator source: {view.source}')
                self.camera_ids[view.name] = camera
            if not views:
                return
            self.scene = mj.MjvScene(model, maxgeom=10000)
            self.camera, self.options, self.perturb = mj.MjvCamera(), mj.MjvOption(), mj.MjvPerturb()
            width, height = max(v.width for v in views), max(v.height for v in views)
            self.gl = (type(restore_context)(width, height) if restore_context is not None
                       else mj.GLContext(width, height))
            self.gl.make_current()
            self.context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100)
            mj.mjr_resizeOffscreen(width, height, self.context)
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            if options is not None:
                self.options.geomgroup[:] = options.geomgroup
                self.options.sitegroup[:] = options.sitegroup
                self.options.flags[:] = options.flags
        except BaseException:
            self.close()
            raise
        finally:
            self._restore()

    def _restore(self):
        if self.restore_context is not None:
            self.restore_context.make_current()

    @contextmanager
    def _current(self):
        self.gl.make_current()
        try:
            yield
        finally:
            self._restore()

    def capture(self, name):
        import mujoco as mj
        view = self.views[name]
        with self._current():
            self.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = self.camera_ids[name]
            mj.mjv_updateScene(self.model, self.data, self.options, self.perturb,
                              self.camera, mj.mjtCatBit.mjCAT_ALL, self.scene)
            rgb = np.empty((view.height, view.width, 3), dtype=np.uint8)
            viewport = mj.MjrRect(0, 0, view.width, view.height)
            mj.mjr_render(viewport, self.scene, self.context)
            mj.mjr_readPixels(rgb, None, viewport, self.context)
            return np.ascontiguousarray(rgb[::-1])

    def intrinsics(self, name):
        view, cid = self.views[name], self.camera_ids[name]
        focal = .5 * view.height / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        return float(focal), float(focal), (view.width - 1) / 2., (view.height - 1) / 2.

    def close(self):
        if self.gl is not None:
            self.gl.make_current()
            if self.context is not None:
                self.context.free()
                self.context = None
            self.gl.free()
            self.gl = None
            self._restore()
