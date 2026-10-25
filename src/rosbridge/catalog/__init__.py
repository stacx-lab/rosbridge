"""Declarative simulator and task discovery; never import native backends."""
from pathlib import Path
import tomllib

from rosbridge.project import project_root


def simulator(name, project=None):
    root = project_root(project)
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid simulator: {name}")
    directory = root / "simulators" / name
    manifest = directory / "simulator.toml"
    if not manifest.is_file():
        raise ValueError(f"Unknown simulator: {name}")
    config = tomllib.loads(manifest.read_text())
    return root, directory, config


def tasks(project=None, name=None):
    root = project_root(project)
    names = [name] if name else sorted(p.parent.name for p in (root / "simulators").glob("*/simulator.toml"))
    result = {}
    for item in names:
        _, directory, config = simulator(item, root)
        for task in tomllib.loads((directory / config["task_catalog"]).read_text())["tasks"]:
            if task["id"] in result or not task["id"].startswith(item + "/"):
                raise ValueError(f"Invalid or duplicate task ID: {task['id']}")
            result[task["id"]] = dict(task, simulator=item)
    return result


def resolve(selection, project=None, task=None):
    name, separator, slug = selection.partition("/")
    if not separator and task is None:
        root = project_root(project)
        if not (root / "simulators" / name / "simulator.toml").is_file():
            matches = [entry for entry in tasks(root).values()
                       if selection in (entry["native_task"], entry["id"].split("/", 1)[1])]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous task {selection!r}; use simulator/task")
            if matches:
                return resolve(matches[0]["id"], root)
    root, _, config = simulator(name, project)
    if task is not None and separator:
        raise ValueError("Select a task either in the ID or with --task")
    target = task if task is not None else (slug if separator else config["default_task"])
    if target.startswith(name + "/"):
        target = target[len(name) + 1:]
    entries = tasks(root, name)
    for entry in entries.values():
        if target in (entry["id"].split("/", 1)[1], entry["native_task"]):
            return root, config, entry
    raise ValueError(f"Unknown task {target!r}; run rosbridge tasks list --simulator {name}")


def preparation_tasks(selection, project=None, task=None):
    """A bare simulator prepares its entire catalog; explicit selections narrow it."""
    root, config, selected = resolve(selection, project, task)
    entries = ([selected] if task is not None or selection != config["name"] else
               list(tasks(root, config["name"]).values()))
    return root, config, entries
