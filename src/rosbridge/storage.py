"""Persistent installation paths, disposable caches, and atomic JSON files."""
import json
import os
from pathlib import Path
import tempfile


def write_json(path, value):
    """Publish a complete, private JSON file, including with concurrent writers."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix=f".{path.name}-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run_environment(folder):
    """Keep all managed ROS processes for one run in the same disposable directory."""
    folder = Path(folder).resolve()
    paths = {"ROS_HOME": folder / "ros", "ROS_LOG_DIR": folder / "ros/log",
             "TMPDIR": folder / "tmp"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {**{key: str(path) for key, path in paths.items()}, "ROSBRIDGE_RUN_DIR": str(folder)}


def local_environment(root, base=None):
    env = dict(os.environ if base is None else base)
    cache = root.resolve() / ".cache/rosbridge"
    paths = {
        "XDG_CACHE_HOME": cache / "xdg",
        "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": root.resolve() / ".runtime/python",
        "PIXI_CACHE_DIR": cache / "pixi",
        "PIP_CACHE_DIR": cache / "pip",
        "ROS_HOME": cache / "ros",
        "ROS_LOG_DIR": cache / "ros/log",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "MPLCONFIGDIR": cache / "matplotlib",
        "NUMBA_CACHE_DIR": cache / "numba",
        "CUDA_CACHE_PATH": cache / "cuda",
        "__GL_SHADER_DISK_CACHE_PATH": cache / "shaders",
        "TMPDIR": cache / "tmp",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["ROSBRIDGE_PROJECT"] = str(root.resolve())
    return env
