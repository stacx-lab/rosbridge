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


def signal_owner(owner, sig):
    """A pidfd prevents PID reuse between identity validation and signal delivery."""
    from .records import identity
    # Some uv-provided CPython builds omit os.pidfd_open even on a capable
    # Linux host. Use glibc's wrappers rather than falling back to a racy kill().
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    fd = libc.pidfd_open(owner["pid"], 0)
    if fd < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return
        raise OSError(error, "Cannot open run process handle")
    try:
        if identity(owner["pid"]) != owner:
            raise ValueError("Run process identity changed; refusing to signal it")
        if libc.pidfd_send_signal(fd, sig, None, 0) < 0:
            error = ctypes.get_errno()
            if error != errno.ESRCH:
                raise OSError(error, "Cannot signal run process handle")
    finally:
        os.close(fd)
