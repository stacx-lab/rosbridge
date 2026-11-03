"""Fresh camera frames on demand, without stepping physics. Ported from RoboEnv's BEHAVIOR adapter.

Zero-delay update ordering makes each render deliver the state it rendered, so frames show the
current state. The R1Pro embodiments keep OmniGibson's anti-aliasing, as the evaluator renders;
the G1 renders on demand with DLAA (RENDER_SETTINGS).
"""
from contextlib import contextmanager

ZERO_DELAY_SETTINGS = {
    "/app/renderer/waitIdle": True,
    "/app/hydraEngine/waitIdle": True,
    "/app/updateOrder/checkForHydraRenderComplete": 1000,
}
# On-demand rendering: DLAA instead of OmniGibson's DLSS Performance: on-demand captures
# accumulate few renders, and anti-aliasing at native resolution cuts their frame-to-frame noise
# by a third (DLSS upscales from half). OmniGibson's renderer setup resets it, so it applies once
# the environment exists.
RENDER_SETTINGS = {"/rtx/post/aa/op": 4}
DISPATCHER = "/Render/PostProcess/SDGPipeline/PostProcessDispatcher"


@contextmanager
def capture_launch_settings():
    """Supply Kit startup arguments through Isaac's launcher defaults.

    OmniGibson owns SimulationApp construction and clears sys.argv, so scope these
    defaults to its boot.
    """
    from isaacsim import SimulationApp

    previous = SimulationApp.DEFAULT_LAUNCHER_CONFIG
    arguments = [f"--{key}={str(value).lower()}" for key, value in ZERO_DELAY_SETTINGS.items()]
    SimulationApp.DEFAULT_LAUNCHER_CONFIG = {
        **previous, "extra_args": [*previous.get("extra_args", []), *arguments]}
    try:
        yield
    finally:
        SimulationApp.DEFAULT_LAUNCHER_CONFIG = previous


def apply_render_settings():
    import carb.settings

    for key, value in RENDER_SETTINGS.items():
        carb.settings.get_settings().set(key, value)


def _stamp_reader():
    import omni.graph.core as graph

    node = graph.get_node_by_path(DISPATCHER)
    if node is None:
        raise RuntimeError("BEHAVIOR camera dispatcher is unavailable")
    numerator = node.get_attribute("outputs:referenceTimeNumerator")
    denominator = node.get_attribute("outputs:referenceTimeDenominator")

    def stamp():
        value, divisor = numerator.get(), denominator.get()
        return (value, divisor) if value is not None and divisor else None
    return stamp


def delivered_time():
    """The reference time, in seconds, of the frame the camera dispatcher last delivered."""
    stamp = _stamp_reader()()
    return None if stamp is None else stamp[0] / stamp[1]


def render_settings():
    import carb.settings

    settings = carb.settings.get_settings()
    return {key: settings.get(key) for key in (*ZERO_DELAY_SETTINGS, *RENDER_SETTINGS, "/rtx/post/dlss/execMode")}


def render_for_capture(deliveries=2, dlaa=True):
    """Render until the camera dispatcher delivers `deliveries` frames of the current state;
    returns the number of renders.

    With zero-delay update ordering a new delivery contains this render's state; a render
    call alone does not prove that the dispatcher delivered it. Physics steps without
    rendering, so the first frame after a change carries the renderer's temporal history
    (denoising and anti-aliasing) of an older state, as each step's render does in OmniGibson.
    On demand, the second is clean while that history is recent; without one, as after a
    camera cut, the path tracer needs more frames.
    """
    import carb.settings
    import omnigibson as og

    settings = carb.settings.get_settings()
    required = {**ZERO_DELAY_SETTINGS, **(RENDER_SETTINGS if dlaa else {})}
    if any(settings.get(key) != value for key, value in required.items()):
        raise RuntimeError("BEHAVIOR camera capture requires zero-delay startup"
                           + (" and DLAA render settings" if dlaa else ""))
    stamp = _stamp_reader()
    previous, delivered = stamp(), 0
    for renders in range(1, 16 * deliveries + 1):
        og.sim.render()
        current = stamp()
        if current is not None and current != previous:
            previous, delivered = current, delivered + 1
            if delivered == deliveries:
                return renders
    raise RuntimeError(f"BEHAVIOR cameras did not deliver {deliveries} fresh frames within {16 * deliveries} renders")
