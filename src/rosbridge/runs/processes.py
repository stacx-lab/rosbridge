"""Linux process ownership primitives for managed graphs."""
import ctypes
import errno
import os
import signal


def parent_death_signal(parent_pid):
    """Called in the single-threaded supervisor's child immediately before exec."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGINT, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot arm graph parent-death signal")
    if os.getppid() != parent_pid:
        os._exit(1)


# pidfd_send_signal and pidfd_open share one number on every Linux architecture (unified syscall table, 5.1+).
SYS_PIDFD_SEND_SIGNAL, SYS_PIDFD_OPEN = 424, 434


def _pidfd():
    """open(pid) and send(fd, sig) on pidfds: Python's own when the build has them, else glibc's wrappers (2.36+),
    else the raw system calls, so an older glibc does not fall back to a racy kill()."""
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        def send(fd, sig):
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
        return lambda pid: os.pidfd_open(pid, 0), send
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if hasattr(libc, "pidfd_open"):
        libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        raw_open = lambda pid: libc.pidfd_open(pid, 0)
        raw_send = lambda fd, sig: libc.pidfd_send_signal(fd, sig, None, 0)
    else:
        libc.syscall.restype = ctypes.c_long
        raw_open = lambda pid: libc.syscall(SYS_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
        raw_send = lambda fd, sig: libc.syscall(SYS_PIDFD_SEND_SIGNAL, ctypes.c_int(fd), ctypes.c_int(sig), None, ctypes.c_uint(0))

    def open_(pid):
        fd = raw_open(pid)
        if fd < 0:
            error = ctypes.get_errno()
            raise ProcessLookupError(error, "no such process") if error == errno.ESRCH else OSError(error, "Cannot open run process handle")
        return fd

    def send(fd, sig):
        if raw_send(fd, sig) < 0:
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, "Cannot signal run process handle")
    return open_, send


def signal_owner(owner, sig):
    """A pidfd prevents PID reuse between identity validation and signal delivery."""
    from .records import identity
    open_, send = _pidfd()
    try:
        fd = open_(owner["pid"])
    except ProcessLookupError:
        return
    try:
        if identity(owner["pid"]) != owner:
            raise ValueError("Run process identity changed; refusing to signal it")
        send(fd, sig)
    finally:
        os.close(fd)
