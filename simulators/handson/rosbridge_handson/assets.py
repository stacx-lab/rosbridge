"""HandsOn bundles its robot, task and rendering assets in the pinned source; nothing is downloaded."""


def prepare(runtime, task):
    check_task(runtime, task)


def check_task(runtime, task):
    """The task has an official evaluation budget and the bundled assets match upstream's SHA-256 pins."""
    import json
    from pathlib import Path
    import sys

    try:
        source = Path(runtime) / "handson"
        if str(source) not in sys.path:  # its editable install loads only in the next interpreter
            sys.path.append(str(source))
        budgets = json.loads((source / "handson/evaluation/budgets.json").read_text())
        if task not in budgets:
            raise ValueError(f"{task} has no evaluation budget")
        from handson.robots.assets import require_benchmark_assets
        require_benchmark_assets()
    except Exception as error:
        raise RuntimeError(f"HandsOn assets unavailable; run rosbridge prepare handson --task {task}") from error
