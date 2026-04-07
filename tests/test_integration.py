"""Integration test: CUDAIPCSender → SharedMemory → CUDAIPCReceiver.

Requires CUDA GPU. Run with:
    pytest tests/test_integration.py -v -m cuda
"""
import time
import pytest
import numpy as np


def cuda_available():
    """Check if CUDA runtime is available."""
    try:
        from cuda_ipc_transport.wrapper import get_cuda_runtime
        get_cuda_runtime()
        return True
    except RuntimeError:
        return False


@pytest.mark.cuda
def test_sender_receiver_roundtrip():
    """Sender initializes, sends 10 frames, receiver reads all 10."""
    pytest.importorskip("cuda_ipc_transport", reason="CUDA not available")

    if not cuda_available():
        pytest.skip("CUDA runtime not available")

    from cuda_ipc_transport.channel import CUDAIPCChannel
    from cuda_ipc_transport.sender import CUDAIPCSender
    from cuda_ipc_transport.receiver import CUDAIPCReceiver

    ch = CUDAIPCChannel("_test_roundtrip", 64, 64, channels=4, dtype="uint8")
    sender = CUDAIPCSender(ch)
    assert sender.initialize(), "Sender init failed"

    receiver = CUDAIPCReceiver("_test_roundtrip")
    assert receiver.connect(), "Receiver connect failed"

    received = 0
    frame = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)

    for _ in range(10):
        sender.send_numpy(frame)
        time.sleep(0.005)  # allow async copy to complete
        ptr, size, shape = receiver.get_frame()
        if ptr is not None:
            received += 1
            assert size == ch.data_size
            assert shape == (64, 64, 4)

    sender.close()
    receiver.close()
    assert received >= 9, f"Only received {received}/10 frames"


@pytest.mark.cuda
def test_sender_no_leak():
    """GPU VRAM delta < 10MB after 100 frames."""
    pytest.importorskip("cuda_ipc_transport", reason="CUDA not available")

    if not cuda_available():
        pytest.skip("CUDA runtime not available")

    from cuda_ipc_transport.wrapper import get_cuda_runtime
    from cuda_ipc_transport.channel import CUDAIPCChannel
    from cuda_ipc_transport.sender import CUDAIPCSender
    from cuda_ipc_transport.receiver import CUDAIPCReceiver

    cuda = get_cuda_runtime()
    free_before, _ = cuda.mem_get_info()

    ch = CUDAIPCChannel("_test_leak", 512, 512, channels=4)
    sender = CUDAIPCSender(ch)
    sender.initialize()
    receiver = CUDAIPCReceiver("_test_leak")
    receiver.connect()

    frame = np.zeros((512, 512, 4), dtype=np.uint8)
    for _ in range(100):
        sender.send_numpy(frame)
        time.sleep(0.001)
        receiver.get_frame()

    sender.close()
    receiver.close()

    free_after, _ = cuda.mem_get_info()
    leak_mb = (free_before - free_after) / 1024 / 1024
    assert leak_mb < 10, f"VRAM leak: {leak_mb:.1f} MB"
