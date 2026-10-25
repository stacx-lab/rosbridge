"""Installed adapters register a lazy `rosbridge.backends` entry point."""

from importlib.metadata import entry_points


def load_backend(name):
    plugins = list(entry_points(group="rosbridge.backends"))
    matches = [plugin for plugin in plugins if plugin.name == name]
    if not matches:
        available = ", ".join(sorted(plugin.name for plugin in plugins)) or "none"
        raise ValueError(f"unknown backend {name!r}; installed backends: {available}")
    if len(matches) != 1:
        raise ValueError(f"duplicate backend registration: {name}")
    return matches[0].load()
