"""Tests for OSC status sender in harness -- no CUDA required."""
import socket
import threading
import time
import pytest
from cuda_ipc_transport.harness import _OSCStatus


def _has_pythonosc():
    try:
        import pythonosc
        return True
    except ImportError:
        return False


class TestOSCStatusDisabled:
    """Test _OSCStatus when disabled or unavailable."""

    def test_disabled_when_port_zero(self):
        osc = _OSCStatus(0)
        assert osc._client is None
        osc.connected(1)
        osc.frame(42)
        osc.close()

    def test_disabled_when_port_negative(self):
        osc = _OSCStatus(-1)
        assert osc._client is None

    def test_graceful_without_pythonosc(self, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if "pythonosc" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        osc = _OSCStatus(7099)
        assert osc._client is None


@pytest.mark.skipif(not _has_pythonosc(), reason="python-osc not installed")
class TestOSCStatusSends:

    def test_sends_connected_and_frame(self):
        port = 17199
        received = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        sock.settimeout(2.0)

        def listen():
            try:
                while True:
                    data, _ = sock.recvfrom(1024)
                    received.append(data)
            except socket.timeout:
                pass

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        osc = _OSCStatus(port)
        osc.connected(1)
        osc.frame(10)
        osc.close()
        time.sleep(0.5)
        sock.close()
        t.join(timeout=3)
        # Decode OSC packets and verify addresses + values
        from pythonosc.osc_message import OscMessage
        messages = []
        for raw in received:
            try:
                msg = OscMessage(raw)
                messages.append((msg.address, msg.params))
            except Exception:
                pass
        addrs = [m[0] for m in messages]
        assert "/transport/connected" in addrs, f"Missing /transport/connected in {addrs}"
        assert "/transport/frame" in addrs, f"Missing /transport/frame in {addrs}"
        connected_msgs = [m for m in messages if m[0] == "/transport/connected"]
        assert any(m[1] == [1] for m in connected_msgs), "connected(1) not found"
        assert any(m[1] == [0] for m in connected_msgs), "connected(0) not found (shutdown)"
        frame_msgs = [m for m in messages if m[0] == "/transport/frame"]
        assert any(m[1] == [10] for m in frame_msgs), "frame(10) not found"
