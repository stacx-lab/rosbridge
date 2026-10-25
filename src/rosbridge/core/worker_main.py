"""Native worker: construct one backend and serve its calls on this main thread. No ROS."""
import ctypes
import importlib
import os
import signal
import sys
import traceback
from multiprocessing.connection import Connection

from rosbridge.core.worker import DYNAMIC, STATIC, decode, encode

STATIC_CHANGES = ("__init__", "__setattr__", "configure_spectators", "reset")


def snapshot(backend, names):
    """Attributes that exist; properties of an uninitialized episode may raise."""
    values = {}
    for name in names:
        try:
            values[name] = getattr(backend, name)
        except Exception:
            pass
    return values


def serve(connection, factory):
    backend = None
    while True:
        try:
            method, args, kwargs = decode(connection.recv())
        except EOFError:
            break
        try:
            if method == "__init__":
                backend, result = factory(**kwargs), None
            elif method == "__setattr__":
                result = setattr(backend, *args)
            else:
                result = getattr(backend, method)(*args, **kwargs)
            reply = ("ok", result)
        except Exception as error:
            reply = ("error", (type(error).__name__, str(error), traceback.format_exc()))
        names = DYNAMIC + (STATIC if method in STATIC_CHANGES else ())
        connection.send(encode((*reply, snapshot(backend, names) if backend is not None else None)))
        if method == "close":
            return
    if backend is not None:
        backend.close()


def main():
    descriptor, target = int(sys.argv[1]), sys.argv[2]
    # The proxy owns shutdown; a terminal interrupt must not tear down native state first.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    # Native runtimes such as Kit handle SIGTERM; an orphaned worker has nothing to save.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot arm worker parent-death signal")
    module, name = target.split(":")
    factory = getattr(importlib.import_module(module), name)
    connection = Connection(descriptor)
    try:
        serve(connection, factory)
    finally:
        connection.close()
        # Native runtimes such as Isaac Sim may hang in interpreter teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
