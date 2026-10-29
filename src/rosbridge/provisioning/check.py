"""Environment-local import and task asset checks; no simulator construction."""
import argparse
import importlib

from rosbridge.catalog import resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator")
    parser.add_argument("--task", action="append")
    parser.add_argument("--native", action="store_true",
                        help="check the native worker interpreter instead of the ROS environment")
    args = parser.parse_args()
    selected = [resolve(args.simulator, task=task) for task in (args.task or [None])]
    root, config, _ = selected[0]
    if args.native:
        module, name = config["native_backend"].split(":")
        getattr(importlib.import_module(module), name)
    else:
        import rclpy  # noqa: F401
        from kinex_interfaces.srv import Wait  # noqa: F401
        from rosbridge.core import load_backend
        load_backend(config["name"])
        if config.get("native_environment"):
            return  # the native worker reads the assets, so its interpreter checks them
    module, function = config["asset_checker"].split(":")
    checker = getattr(importlib.import_module(module), function)
    for _, _, task in selected:
        checker(root / ".runtime" / config["name"], task["native_task"])


if __name__ == "__main__":
    main()
