from cuda_ipc_transport.channel import CUDAIPCChannel

_2MiB = 2 * 1024 * 1024


def test_data_size():
    ch = CUDAIPCChannel("test", 512, 512, channels=4, dtype="uint8")
    assert ch.data_size == 512 * 512 * 4 * 1  # uint8 = 1 byte/channel


def test_buffer_size_aligned():
    ch = CUDAIPCChannel("test", 512, 512, channels=4, dtype="uint8")
    # data_size = 1048576 = exactly 1 MiB — rounds up to 2 MiB
    assert ch.buffer_size == _2MiB


def test_buffer_size_small():
    ch = CUDAIPCChannel("test", 64, 64, channels=4, dtype="uint8")
    # data_size = 16384 — rounds up to 2 MiB
    assert ch.buffer_size == _2MiB


def test_name():
    ch = CUDAIPCChannel("hagar_out", 512, 512)
    assert ch.name == "hagar_out"


def test_float32_data_size():
    ch = CUDAIPCChannel("test", 512, 512, channels=4, dtype="float32")
    assert ch.data_size == 512 * 512 * 4 * 4  # 4 bytes/channel


def test_dtype_code_uint8():
    ch = CUDAIPCChannel("test", 512, 512, dtype="uint8")
    assert ch.dtype_code == 2


def test_dtype_code_float32():
    ch = CUDAIPCChannel("test", 512, 512, dtype="float32")
    assert ch.dtype_code == 0
