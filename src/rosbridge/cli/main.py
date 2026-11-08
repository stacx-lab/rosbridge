"""Prepare public simulators and manage their complete ROS 2 graphs."""
import argparse
import json
import math
from pathlib import Path
import sys


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def domain(value):
    from rosbridge.runs.domains import check_domain
    try:
        return check_domain(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def parser():
    root = argparse.ArgumentParser(prog="rosbridge", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    def common(command):
        command.add_argument("--project", type=Path)
        command.add_argument("--json", action="store_true", help="Machine-readable output")
    for name in ("prepare", "doctor"):
        command = commands.add_parser(name)
        common(command)
        command.add_argument("simulator")
        default = "all cataloged tasks" if name == "prepare" else "simulator default task"
        command.add_argument("--task", help=f"Task slug or native task name; default: {default}")
    for name in ("launch", "demo"):
        command = commands.add_parser(name)
        common(command)
        command.add_argument("selection", metavar="SIMULATOR[/TASK]", help="Simulator or task ID")
        command.add_argument("--execution-mode", choices=("realtime", "stepped"))
        command.add_argument("--domain-id", type=domain, help="Explicit ROS domain; default: automatically leased")
        command.add_argument("--timeout", type=positive, default=900., help="Startup timeout in wall seconds")
        command.add_argument("--seed", type=int)
        command.add_argument("--record", type=Path, metavar="DIR",
                             help="Save spectator recordings to a new or empty directory")
        command.add_argument("--record-type", choices=("mp4", "bag", "both"),
                             help="Recording format (default: mp4); requires --record")
        command.add_argument("--record-rate", type=float, metavar="HZ",
                             help="Spectator frames per world second (default: 10); requires --record")
        command.add_argument("--record-size", metavar="WxH",
                             help="Spectator image size in pixels (default: 960x720); requires --record")
        command.add_argument("--record-views", metavar="NAME,...",
                             help="The spectator views to record (default: all the simulator offers); requires --record")
        command.add_argument("--record-quality", type=int, metavar="CRF",
                             help="H.264 constant rate factor, lower is sharper (default: 20); requires --record")
        success = command.add_mutually_exclusive_group()
        success.add_argument("--stop-on-success", dest="stop_on_success", action="store_true")
        success.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
        command.set_defaults(stop_on_success=None)
        if name == "launch":
            command.add_argument("--detach", action="store_true")
    for name in ("status", "stop", "judge"):
        command = commands.add_parser(name)
        common(command)
        command.add_argument("run", nargs="?", help="Run ID or run.json; default: the only live local run")
    robots = commands.add_parser("robots")
    listing = robots.add_subparsers(dest="robots_command", required=True).add_parser("list")
    common(listing)
    setup = commands.add_parser("setup", help="Export unloaded Kinex device profiles")
    common(setup)
    setup.add_argument("robot")
    setup.add_argument("--directory", type=Path)
    setup.add_argument("--domain-id", type=domain, default=0)
    task = commands.add_parser("tasks")
    listing = task.add_subparsers(dest="tasks_command", required=True).add_parser("list")
    common(listing)
    listing.add_argument("--simulator")
    listing.add_argument("--robot", help="Filter tasks by robot catalog ID")
    command = commands.add_parser("exec", help="Join a run's environment and ROS domain")
    command.add_argument("--project", type=Path)
    command.add_argument("run")
    command.add_argument("argv", nargs=argparse.REMAINDER)
    return root


def emit(value, machine=False):
    if machine:
        print(json.dumps(value, default=str, allow_nan=False), flush=True)
    elif "issues" in value:
        print(f"{value['simulator']}: {'ready' if value['ready'] else 'not ready'}")
        if value.get("tasks"):
            print(f"  Tasks checked: {len(value['tasks'])}")
        for item in value["issues"]:
            print(f"  {item}")
        for item in value["next_steps"]:
            print(f"Next: {item}")
    elif "domain_id" in value:
        print(f"Run: {value['id']}\nTask: {value['task']}\nState: {value['state']}\n"
              f"ROS domain: {value['domain_id']}\nLog: {value['log']}\n"
              f"Join: rosbridge exec {value['id']} -- COMMAND", flush=True)
        if value.get("recording_dir"):
            print(f"Recording: {value['recording_dir']}", flush=True)
    else:
        print(json.dumps(value, indent=2, default=str), flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("prepare", "doctor"):
            from rosbridge.provisioning.setup import doctor, prepare
            result = (prepare if args.command == "prepare" else doctor)(
                args.simulator, project=args.project, task=args.task)
            emit(result, args.json)
            return 0 if result["ready"] else 1
        if args.command == "tasks":
            from rosbridge.catalog import tasks
            entries = tasks(args.project, args.simulator)
            if args.robot:
                from rosbridge.catalog.robots import list_robots
                if args.robot not in list_robots(args.project):
                    raise ValueError(f"unknown robot {args.robot!r}")
                entries = {key: value for key, value in entries.items() if value.get("robot") == args.robot}
            if args.json:
                emit({"tasks": len(entries), "ids": sorted(entries),
                      "robots": {key: value.get("robot") for key, value in sorted(entries.items())}}, True)
            else:
                print("\n".join(sorted(entries)))
            return 0
        if args.command in ("robots", "setup"):
            from rosbridge.catalog.robots import list_robots, setup
            result = ({"robots": list_robots(args.project)} if args.command == "robots" else
                      setup(args.robot, args.directory, project=args.project, domain_id=args.domain_id))
            emit(result, args.json)
            return 0
        from rosbridge.runs import api
        if args.command in ("launch", "demo"):
            for flag in ("record_type", "record_rate", "record_size", "record_views", "record_quality"):
                if getattr(args, flag, None) is not None and args.record is None:
                    raise ValueError(f"--{flag.replace('_', '-')} requires --record DIR")
            record, process = api.launch(args.selection, project=args.project, options=vars(args),
                                         demo=args.command == "demo")
            emit(record, args.json)
            if getattr(args, "detach", False):
                return 0
            return api.follow(record, process)
        if args.command in ("status", "stop", "judge"):
            result = {"status": api.status, "stop": api.stop, "judge": api.judge}[args.command](args.run, project=args.project)
            emit(result, args.json)
            return 0
        if args.command == "exec":
            command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            if not command:
                raise ValueError("exec requires a command after --")
            code = api.execute(args.run, command, project=args.project)
            return code if code >= 0 else 128 - code
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        if getattr(args, "json", False):
            emit({"error": f"{type(error).__name__}: {error}"}, True)
        else:
            print(f"rosbridge: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
