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


# Unified system call numbers, the same on every Linux architecture (pidfd_open needs Linux 5.3).
SYS_PIDFD_SEND_SIGNAL, SYS_PIDFD_OPEN = 424, 434


def signal_owner(owner, sig):
    """A pidfd prevents PID reuse between identity validation and signal delivery."""
    from .records import identity
    # Raw system calls: some uv-provided CPython builds omit os.pidfd_open, and glibc only wraps
    # them since 2.36 (Ubuntu 22.04 has 2.35). Never fall back to a racy kill().
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(SYS_PIDFD_OPEN), ctypes.c_long(owner["pid"]), ctypes.c_long(0))
    if fd < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return
        raise OSError(error, "Cannot open run process handle")
    try:
        if identity(owner["pid"]) != owner:
            raise ValueError("Run process identity changed; refusing to signal it")
        if libc.syscall(ctypes.c_long(SYS_PIDFD_SEND_SIGNAL), ctypes.c_long(fd), ctypes.c_long(sig),
                        None, ctypes.c_long(0)) < 0:
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, "Cannot signal run process handle")
    finally:
        os.close(fd)
