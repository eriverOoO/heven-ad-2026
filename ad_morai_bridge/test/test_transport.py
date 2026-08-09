import socket
import threading
import time

from ad_morai_bridge.udp_transport import UdpEndpoint, UdpReceiver, UdpSender


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_real_loopback_receive_and_sender_roundtrip():
    port = _free_port()
    received = []
    event = threading.Event()

    def callback(data, address, received_at):
        received.append((data, address, received_at))
        event.set()

    receiver = UdpReceiver(UdpEndpoint("127.0.0.1", port), callback)
    receiver.start()
    sender = UdpSender(UdpEndpoint("127.0.0.1", port))
    try:
        assert sender.send(b"smoke") == 5
        assert event.wait(1.0)
        assert received[0][0] == b"smoke"
        assert received[0][1][0] == "127.0.0.1"
        assert received[0][2] > 0.0
    finally:
        sender.close()
        receiver.close()
    assert not receiver.is_alive


def test_close_unblocks_idle_receiver():
    receiver = UdpReceiver(UdpEndpoint("127.0.0.1", _free_port()), lambda *_: None)
    receiver.start()

    receiver.close()

    assert not receiver.is_alive


def test_callback_exception_is_recorded_without_dropping_later_datagrams():
    port = _free_port()
    bad_seen = threading.Event()
    good_seen = threading.Event()

    def callback(data, _address, _received_at):
        if data == b"bad":
            bad_seen.set()
            raise RuntimeError("simulated callback failure")
        if data == b"good":
            good_seen.set()

    receiver = UdpReceiver(UdpEndpoint("127.0.0.1", port), callback)
    receiver.start()
    sender = UdpSender(UdpEndpoint("127.0.0.1", port))
    try:
        assert sender.send(b"bad") == 3
        assert bad_seen.wait(1.0)
        assert sender.send(b"good") == 4
        assert good_seen.wait(1.0)
        assert receiver.is_alive
        assert isinstance(receiver.last_error, RuntimeError)
    finally:
        sender.close()
        receiver.close()


def test_close_from_callback_does_not_join_receiver_thread_itself():
    port = _free_port()
    close_returned = threading.Event()

    def callback(_data, _address, _received_at):
        receiver.close()
        close_returned.set()

    receiver = UdpReceiver(UdpEndpoint("127.0.0.1", port), callback)
    receiver.start()
    sender = UdpSender(UdpEndpoint("127.0.0.1", port))
    try:
        assert sender.send(b"stop") == 4
        assert close_returned.wait(1.0)
        for _ in range(50):
            if not receiver.is_alive:
                break
            time.sleep(0.01)
        assert not receiver.is_alive
        assert receiver.last_error is None
    finally:
        sender.close()
        receiver.close()


def test_receiver_does_not_enable_port_reuse_and_exposes_runtime_socket_error():
    class FailingSocket:
        def __init__(self):
            self.options = []

        def setsockopt(self, level, option, value):
            self.options.append((level, option, value))

        def settimeout(self, _timeout):
            pass

        def bind(self, _address):
            pass

        def recvfrom(self, _size):
            raise OSError("simulated receiver failure")

        def close(self):
            pass

    fake = FailingSocket()
    receiver = UdpReceiver(
        UdpEndpoint("127.0.0.1", 19_000),
        lambda *_: None,
        socket_factory=lambda *_: fake,
    )
    assert not any(
        level == socket.SOL_SOCKET and option == socket.SO_REUSEADDR
        for level, option, _ in fake.options
    )

    receiver.start()
    for _ in range(50):
        if not receiver.is_alive:
            break
        time.sleep(0.01)

    assert not receiver.is_alive
    assert isinstance(receiver.last_error, OSError)
    receiver.close()
