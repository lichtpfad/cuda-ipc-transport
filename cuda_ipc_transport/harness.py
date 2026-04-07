"""CLI harness: run a source and send frames via CUDA IPC."""
import argparse
import signal
import sys
import time

from .channel import CUDAIPCChannel
from .sender import CUDAIPCSender
from .sources.test_pattern import TestPatternSource
from .sources.file import FileSource
from .sources.camera import CameraSource


class _OSCStatus:
    """Lightweight OSC status sender. No-ops gracefully if python-osc not installed."""

    def __init__(self, port: int):
        self._client = None
        if port <= 0:
            return
        try:
            from pythonosc.udp_client import SimpleUDPClient
            self._client = SimpleUDPClient("127.0.0.1", port)
            print(f"[harness] OSC status -> 127.0.0.1:{port}")
        except ImportError:
            print("[harness] WARNING: python-osc not installed. "
                  "Install with: pip install cuda_ipc_transport[osc]",
                  file=sys.stderr)

    def connected(self, value: int = 1):
        if self._client:
            self._client.send_message("/transport/connected", value)

    def frame(self, n: int):
        if self._client:
            self._client.send_message("/transport/frame", n)

    def close(self):
        if self._client:
            try:
                self._client.send_message("/transport/connected", 0)
            except Exception:
                pass


def _resolve_channel(channel: str, channel_prefix: str = None) -> str:
    """Resolve effective channel name from args.

    --channel-prefix takes priority: creates {prefix}_result.
    --channel is backward-compatible literal name.
    """
    if channel_prefix:
        return f"{channel_prefix}_result"
    return channel


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cuda_ipc_transport",
        description="Send GPU frames from a source to TouchDesigner via CUDA IPC",
    )
    parser.add_argument("--source", choices=["test", "file", "camera"], default="test",
                        help="Frame source (default: test)")
    parser.add_argument("--channel", default="cuda_ipc_test",
                        help="SharedMemory channel name (default: cuda_ipc_test)")
    parser.add_argument("--channel-prefix", default=None,
                        help="Channel prefix. Creates {prefix}_result as send channel. "
                             "Overrides --channel when set.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0,
                        help="Number of frames to send (0 = infinite)")
    parser.add_argument("--file", default=None,
                        help="Path for --source file")
    parser.add_argument("--osc-status-port", type=int, default=0,
                        help="UDP port for OSC transport status (0 = disabled)")
    args = parser.parse_args(argv)

    if args.channel_prefix and args.channel != "cuda_ipc_test":
        print(f"[harness] WARNING: --channel-prefix overrides --channel", file=sys.stderr)
    effective_channel = _resolve_channel(args.channel, args.channel_prefix)

    if args.source == "test":
        source = TestPatternSource(args.width, args.height, args.fps)
    elif args.source == "file":
        if not args.file:
            print("ERROR: --file required with --source file", file=sys.stderr)
            sys.exit(1)
        source = FileSource(args.file)
    else:
        source = CameraSource(0, args.width, args.height)

    channel = CUDAIPCChannel(effective_channel, args.width, args.height)
    sender = CUDAIPCSender(channel)

    if not sender.initialize():
        print("ERROR: sender.initialize() failed — is CUDA available?", file=sys.stderr)
        sys.exit(1)

    print(f"[harness] Sending '{args.source}' -> channel '{effective_channel}' at {args.fps} fps")
    print(f"[harness] {args.width}x{args.height} | Press Ctrl+C to stop")

    osc = _OSCStatus(args.osc_status_port)
    osc.connected(1)

    running = [True]

    def _stop(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT, _stop)

    frame_time = 1.0 / args.fps
    sent = 0
    try:
        while running[0]:
            t0 = time.perf_counter()
            frame = source.get_frame()
            sender.send_numpy(frame)
            sent += 1
            osc.frame(sent)

            if sent % 100 == 0:
                print(f"[harness] sent {sent} frames")

            if args.frames > 0 and sent >= args.frames:
                break

            elapsed = time.perf_counter() - t0
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        osc.close()
        sender.close()
        source.close()
        print(f"[harness] Done. Sent {sent} frames.")
