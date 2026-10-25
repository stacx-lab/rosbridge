"""Materialize a pinned upstream plus ordered patches outside the source checkout."""
import argparse
import hashlib
import importlib
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from rosbridge.project import project_root
from rosbridge.catalog import preparation_tasks, simulator
from rosbridge.storage import local_environment


def run(*args, cwd):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def nested_sources(config):
    """Pinned upstreams exported into the generated tree, such as forks that are git submodules."""
    return config.get("nested_sources", [])


def sync_source(root, config):
    """Clone the pinned checkout and any nested pinned checkouts."""
    for nested in nested_sources(config):
        _sync(root, root / nested["source"], nested["source_url"], nested["source_revision"])
    return _sync(root, root / config["source"], config["source_url"], config["source_revision"])


def _sync(root, source, url, revision):
    """Clone a pinned checkout without relying on a parent git repository."""
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        temporary = source.with_name(source.name + ".clone")
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            run("git", "clone", "--no-checkout", url, str(temporary), cwd=root)
            run("git", "checkout", "--detach", revision, cwd=temporary)
            temporary.rename(source)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    if run("git", "status", "--porcelain", "--untracked-files=no", cwd=source):
        raise ValueError("upstream has local changes; preserve them before preparing")
    if run("git", "rev-parse", "HEAD", cwd=source) != revision:
        run("git", "fetch", "origin", revision, cwd=source)
        run("git", "checkout", "--detach", revision, cwd=source)
    return source


def fingerprint(root, config, revision):
    patch_dir = root / config["patches"]
    series = patch_dir / "series"
    names = [line.strip() for line in series.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    patches = [patch_dir / name for name in names]
    result = {"revision": revision, "patches": [
        {"file": name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
        for name, patch in zip(names, patches)
    ]}
    if nested_sources(config):
        result["nested"] = [{"path": nested["path"], "revision": nested["source_revision"]}
                            for nested in nested_sources(config)]
    return result


def _export(source, revision, destination):
    archive = subprocess.check_output(["git", "archive", revision], cwd=source)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        contents.extractall(destination, filter="data")


def materialize(root, name):
    _, _, config = simulator(name, root)
    source = root / config["source"]
    revision = run("git", "rev-parse", "HEAD", cwd=source)
    if config["source_revision"] != revision:
        raise ValueError("upstream must be initialized at the pinned revision")
    if run("git", "status", "--porcelain", "--untracked-files=no", cwd=source):
        raise ValueError("upstream has tracked changes; store them as patches, then restore it")
    for nested in nested_sources(config):
        checkout = root / nested["source"]
        if run("git", "rev-parse", "HEAD", cwd=checkout) != nested["source_revision"]:
            raise ValueError(f"{nested['source']} must be initialized at its pinned revision")
        if run("git", "status", "--porcelain", "--untracked-files=no", cwd=checkout):
            raise ValueError(f"{nested['source']} has tracked changes; store them as patches")
    patch_dir = root / config["patches"]
    identity = fingerprint(root, config, revision)
    patches = [patch_dir / entry["file"] for entry in identity["patches"]]
    destination = root / ".runtime" / config["name"]
    marker = destination / ".prepared.json"
    if destination.exists():
        if marker.exists() and json.loads(marker.read_text()) == identity:
            return destination, config
        raise ValueError(f"stale or incomplete generated source: remove {destination} and retry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        _export(source, revision, staging)
        for nested in nested_sources(config):
            target = staging / nested["path"]
            if target.exists() and any(target.iterdir()):
                raise ValueError(f"nested source path is occupied: {nested['path']}")
            target.mkdir(parents=True, exist_ok=True)
            _export(root / nested["source"], nested["source_revision"], target)
        # Patches apply to the assembled tree, including nested sources.
        # Prevent git apply from discovering an enclosing repository.
        run("git", "init", "-q", cwd=staging)
        for patch in patches:
            run("git", "apply", "--check", str(patch.resolve()), cwd=staging)
            run("git", "apply", str(patch.resolve()), cwd=staging)
        (staging / ".prepared.json").write_text(json.dumps(identity, indent=2) + "\n")
        staging.rename(destination)
    return destination, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--task", help="download assets only for this task; default: all cataloged tasks")
    args = parser.parse_args()
    root = project_root(args.project)
    os.environ.update(local_environment(root))
    from rosbridge.assets import file_lock

    _, _, selected = preparation_tasks(args.simulator, root, args.task)
    with file_lock(root / ".runtime" / f"{args.simulator}.lock"):
        destination, config = materialize(root, args.simulator)
        # Exclude upstream policy/training extras; the environment owns dependencies.
        editables = [(destination, {})] if config.get("install_source", True) else []
        # Exported trees carry no git metadata; a nested source may pin its build version.
        editables += [(destination / nested["path"] / package, nested.get("build_environment", {}))
                      for nested in nested_sources(config) for package in nested.get("install", [])]
        for editable, environment in editables:
            subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                            "--no-deps", "-e", str(editable)], check=True,
                           env={**os.environ, **environment})
        # A newly installed editable .pth is only loaded by the next interpreter.
        sys.path.insert(0, str(destination))
        sys.path.insert(0, str(root / config["python_path"]))
        module, function = config["asset_preparer"].split(":")
        preparer = getattr(importlib.import_module(module), function)
        for index, task in enumerate(selected, 1):
            print(f"[prepare] task {index}/{len(selected)}: {task['id']}", flush=True)
            try:
                preparer(destination, task["native_task"])
            except Exception as error:
                raise RuntimeError(f"Asset preparation failed for {task['id']}: {error}") from error


if __name__ == "__main__":
    main()
