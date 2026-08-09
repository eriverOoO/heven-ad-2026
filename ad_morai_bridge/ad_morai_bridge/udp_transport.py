"""Bounded UDP sender and receiver transport primitives."""

from collections.abc import Callable
from dataclasses import dataclass
import socket
import threading
import time


@dataclass(frozen=True)
class UdpEndpoint:
    host: str
    port: int

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("UDP host must not be empty")
        if not 0 < int(self.port) < 65_536:
            raise ValueError(f"invalid UDP port {self.port}")


class UdpReceiver:
    def __init__(
        self,
        endpoint: UdpEndpoint,
        callback: Callable[[bytes, tuple[str, int], float], None],
        *,
        receive_buffer_bytes: int = 4 * 1024 * 1024,
        socket_factory=socket.socket,
    ):
        self.endpoint = endpoint
        self.callback = callback
        self._stopping = threading.Event()
        self._socket = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(receive_buffer_bytes))
        self._socket.settimeout(0.1)
        self._socket.bind((endpoint.host, endpoint.port))
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.is_alive:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"ad-morai-udp-{self.endpoint.port}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                data, address = self._socket.recvfrom(65_535)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stopping.is_set():
                    break
                self.last_error = exc
                break
            try:
                self.callback(data, address, time.monotonic())
            except Exception as exc:
                self.last_error = exc

    def close(self) -> None:
        self._stopping.set()
        try:
            self._socket.close()
        finally:
            if self._thread and self._thread is not threading.current_thread():
                self._thread.join(timeout=1.0)


class UdpSender:
    def __init__(self, endpoint: UdpEndpoint, *, socket_factory=socket.socket):
        self.endpoint = endpoint
        self._socket = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()

    def send(self, packet: bytes) -> int:
        with self._lock:
            return self._socket.sendto(packet, (self.endpoint.host, self.endpoint.port))

    def close(self) -> None:
        self._socket.close()
