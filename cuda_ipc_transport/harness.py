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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cuda_ipc_transport",
        description="Send GPU frames from a source to TouchDesigner via CUDA IPC",
    )
    parser.add_argument("--source", choices=["test", "file", "camera"], default="test",
                        help="Frame source (default: test)")
    parser.add_argument("--channel", default="cuda_ipc_test",
                        help="SharedMemory channel name (default: cuda_ipc_test)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0,
                        help="Number of frames to send (0 = infinite)")
    parser.add_argument("--file", default=None,
                        help="Path for --source file")
    args = parser.parse_args(argv)

    if args.source == "test":
        source = TestPatternSource(args.width, args.height, args.fps)
    elif args.source == "file":
        if not args.file:
            print("ERROR: --file required with --source file", file=sys.stderr)
            sys.exit(1)
        source = FileSource(args.file)
    else:
        source = CameraSource(0, args.width, args.height)

    channel = CUDAIPCChannel(args.channel, args.width, args.height)
    sender = CUDAIPCSender(channel)

    if not sender.initialize():
        print("ERROR: sender.initialize() failed — is CUDA available?", file=sys.stderr)
        sys.exit(1)

    print(f"[harness] Sending '{args.source}' -> channel '{args.channel}' at {args.fps} fps")
    print(f"[harness] {args.width}x{args.height} | Press Ctrl+C to stop")

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

            if sent % 100 == 0:
                print(f"[harness] sent {sent} frames")

            if args.frames > 0 and sent >= args.frames:
                break

            elapsed = time.perf_counter() - t0
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        sender.close()
        source.close()
        print(f"[harness] Done. Sent {sent} frames.")
