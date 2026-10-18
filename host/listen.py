"""UDP listener shared by demo_ui.py and measure.py (bridge.py publishes, these only read)."""

from __future__ import annotations

import json
import socket
from typing import Iterator


class Listener:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):  # demo_ui and measure can listen at the same time
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        self.sock.bind((host, port))

    def poll(self, timeout: float = 0.0) -> Iterator[dict]:
        """Yields every datagram already queued (waits up to ``timeout`` for the first)."""
        self.sock.settimeout(timeout if timeout > 0 else None)
        self.sock.setblocking(timeout > 0)
        first = True
        while True:
            try:
                data = self.sock.recv(65536)
            except (BlockingIOError, socket.timeout):
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                pass
            if first:
                first = False
                self.sock.setblocking(False)

    def close(self) -> None:
        self.sock.close()


def parse_listen(s: str) -> tuple[str, int]:
    if ":" in s:
        h, p = s.rsplit(":", 1)
        return h or "127.0.0.1", int(p)
    return "127.0.0.1", int(s)
