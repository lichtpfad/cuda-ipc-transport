"""End-to-end smoke test for cuda_ipc_transport harness + OSC.

Usage:
    .venv/Scripts/python scripts/test_smoke.py
    .venv/Scripts/python scripts/test_smoke.py --osc-port 17201 --prefix smoketest
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time


def wait_for_osc(port: int, timeout: float = 2.0) -> list:
    """Listen on UDP port, collect raw packets until timeout."""
    received = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(timeout)

    def listen():
        try:
            while True:
                data, _ = sock.recvfrom(1024)
                received.append(data)
        except socket.timeout:
            pass
        except OSError:
            pass  # socket closed

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return received, sock, t


def main():
    parser = argparse.ArgumentParser(description="Smoke test for harness + OSC")
    parser.add_argument("--osc-port", type=int, default=17201)
    parser.add_argument("--prefix", default="smoketest")
    parser.add_argument("--venv", default=None,
                        help="Path to venv (default: C:/dev/cuda/.venv)")
    args = parser.parse_args()

    venv = args.venv or os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv")
    # On Windows the executable has a .exe extension; on Unix it does not.
    python = os.path.join(venv, "Scripts", "python")
    if not os.path.exists(python):
        python = python + ".exe"
    if not os.path.exists(python):
        print(f"FAIL: Python not found at {python}")
        sys.exit(1)

    checks_passed = 0
    checks_total = 3

    # 1. Start OSC listener
    print(f"[smoke] Listening OSC on 127.0.0.1:{args.osc_port}")
    received, sock, listener = wait_for_osc(args.osc_port, timeout=10.0)

    # 2. Start harness
    cmd = [
        python, "-m", "cuda_ipc_transport",
        "--channel-prefix", args.prefix,
        "--osc-status-port", str(args.osc_port),
        "--source", "test",
        "--width", "64",
        "--height", "64",
        "--fps", "10",
        "--frames", "30",
    ]
    print(f"[smoke] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 3. Wait for harness to finish
    try:
        stdout, stderr = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        print("FAIL: harness timed out (15s)")
        sys.exit(1)

    # Close listener
    time.sleep(0.5)
    sock.close()
    listener.join(timeout=3)

    # 4. Check results
    print(f"\n[smoke] Harness exit code: {proc.returncode}")
    if proc.returncode == 0:
        print("  [PASS] harness exited cleanly")
        checks_passed += 1
    else:
        print(f"  [FAIL] harness exit code {proc.returncode}")
        print(f"  stderr: {stderr.decode('utf-8', errors='replace')[:500]}")

    # Check OSC received
    print(f"[smoke] OSC packets received: {len(received)}")
    if len(received) > 0:
        try:
            from pythonosc.osc_message import OscMessage
            messages = []
            for raw in received:
                try:
                    msg = OscMessage(raw)
                    messages.append((msg.address, msg.params))
                except Exception:
                    pass

            addrs = [m[0] for m in messages]
            has_connected = "/transport/connected" in addrs
            has_frame = "/transport/frame" in addrs

            if has_connected:
                print("  [PASS] /transport/connected received")
                checks_passed += 1
            else:
                print(f"  [FAIL] /transport/connected not in {addrs[:5]}")

            if has_frame:
                frame_vals = [m[1][0] for m in messages if m[0] == "/transport/frame"]
                max_frame = max(frame_vals) if frame_vals else 0
                print(f"  [PASS] /transport/frame received (max={max_frame})")
                checks_passed += 1
            else:
                print(f"  [FAIL] /transport/frame not in {addrs[:5]}")

        except ImportError:
            print("  [SKIP] python-osc not installed, counting raw packets only")
            if len(received) >= 3:
                checks_passed += 2
                print(f"  [PASS] {len(received)} raw packets (>= 3)")
    else:
        print("  [FAIL] no OSC packets received")

    # Summary
    print(f"\n[smoke] Result: {checks_passed}/{checks_total} checks passed")
    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()
