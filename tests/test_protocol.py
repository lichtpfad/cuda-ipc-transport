import struct
from cuda_ipc_transport.protocol import (
    MAGIC, NUM_SLOTS, SLOT_SIZE, SHM_HEADER_SIZE,
    shm_size, meta_offset, SHMLayout,
)


def test_magic():
    assert MAGIC == 0x43504943
    assert struct.pack("<I", MAGIC) == b"CIPC"


def test_shm_size_2slots():
    # header(20) + 2*128 + 1 + 5*4 + 8 = 20+256+1+20+8 = 305
    assert shm_size(2) == 305


def test_shm_size_3slots():
    # header(20) + 3*128 + 1 + 5*4 + 8 = 20+384+1+20+8 = 433
    assert shm_size(3) == 433


def test_meta_offset_3slots():
    # 20 + 3*128 = 20 + 384 = 404
    assert meta_offset(3) == 404


def test_slot_offset():
    layout = SHMLayout(num_slots=3)
    assert layout.slot_offset(0) == 20
    assert layout.slot_offset(1) == 20 + 128
    assert layout.slot_offset(2) == 20 + 256


def test_pack_unpack_header():
    layout = SHMLayout(num_slots=3)
    buf = bytearray(shm_size(3))
    layout.pack_header(buf, version=1, write_idx=0)
    magic, version, num_slots, write_idx = layout.unpack_header(buf)
    assert magic == MAGIC
    assert version == 1
    assert num_slots == 3
    assert write_idx == 0


def test_pack_unpack_metadata():
    layout = SHMLayout(num_slots=3)
    buf = bytearray(shm_size(3))
    layout.pack_metadata(buf, width=512, height=512, channels=4, dtype_code=2, data_size=1048576)
    w, h, c, dt, sz = layout.unpack_metadata(buf)
    assert w == 512 and h == 512 and c == 4 and dt == 2 and sz == 1048576


def test_set_write_idx():
    layout = SHMLayout(num_slots=3)
    buf = bytearray(shm_size(3))
    layout.set_write_idx(buf, 42)
    assert layout.get_write_idx(buf) == 42


def test_shutdown_flag():
    layout = SHMLayout(num_slots=3)
    buf = bytearray(shm_size(3))
    assert not layout.is_shutdown(buf)
    layout.set_shutdown(buf)
    assert layout.is_shutdown(buf)
