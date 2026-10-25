"""Per-user domain leases shared by all rosbridge checkouts."""
import errno
import os
import socket


def check_domain(domain):
    """ROS domain IDs whose default DDS ports fit below the ephemeral range."""
    if type(domain) is not int or not 0 <= domain <= 101:
        raise ValueError("ROS domain ID must be an integer between 0 and 101")
    return domain


class DomainLease:
    def __init__(self, requested=None):
        if requested is not None:
            check_domain(requested)
        for domain in ([requested] if requested is not None else range(1, 100)):
            # Linux abstract sockets coordinate checkouts without persistent files.
            # The inherited descriptor keeps the lease until the graph exits.
            handle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                handle.bind(f"\0rosbridge-{os.getuid()}-domain-{domain}")
            except OSError as error:
                handle.close()
                if error.errno != errno.EADDRINUSE:
                    raise
                continue
            self.domain, self.handle = domain, handle
            return
        raise ValueError("Requested ROS domain is already managed" if requested is not None
                         else "No available ROS domains in automatic range 1–99")

    def close(self):
        self.handle.close()
