"""SharedMemory protocol layout for cuda_ipc_transport v1.0."""
import struct
from dataclasses import dataclass

MAGIC = 0x43504943          # "CIPC" in little-endian
NUM_SLOTS = 3               # default for new writers
SLOT_SIZE = 128             # 64B mem_handle + 64B event_handle
SHM_HEADER_SIZE = 20        # magic(4) + version(8) + num_slots(4) + write_idx(4)

DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_UINT8 = 2

_META_FIELDS = 5  # width, height, channels, dtype_code, data_size
_META_SIZE = _META_FIELDS * 4   # 20 bytes
_TIMESTAMP_SIZE = 8             # float64


def shm_size(num_slots: int) -> int:
    """Total SharedMemory size in bytes for given number of slots."""
    return SHM_HEADER_SIZE + num_slots * SLOT_SIZE + 1 + _META_SIZE + _TIMESTAMP_SIZE


def meta_offset(num_slots: int) -> int:
    """Byte offset of shutdown flag (metadata starts at +1)."""
    return SHM_HEADER_SIZE + num_slots * SLOT_SIZE


@dataclass
class SHMLayout:
    num_slots: int

    def slot_offset(self, slot: int) -> int:
        return SHM_HEADER_SIZE + slot * SLOT_SIZE

    def meta_offset(self) -> int:
        """Byte offset of shutdown flag for this layout's num_slots."""
        return meta_offset(self.num_slots)

    def pack_header(self, buf, version: int, write_idx: int) -> None:
        struct.pack_into("<I", buf, 0, MAGIC)
        struct.pack_into("<Q", buf, 4, version)
        struct.pack_into("<I", buf, 12, self.num_slots)
        struct.pack_into("<I", buf, 16, write_idx)

    def unpack_header(self, buf):
        magic = struct.unpack_from("<I", buf, 0)[0]
        version = struct.unpack_from("<Q", buf, 4)[0]
        num_slots = struct.unpack_from("<I", buf, 12)[0]
        write_idx = struct.unpack_from("<I", buf, 16)[0]
        return magic, version, num_slots, write_idx

    def pack_metadata(self, buf, width, height, channels, dtype_code, data_size) -> None:
        mo = meta_offset(self.num_slots)
        buf[mo] = 0  # shutdown flag
        struct.pack_into("<I", buf, mo + 1, width)
        struct.pack_into("<I", buf, mo + 5, height)
        struct.pack_into("<I", buf, mo + 9, channels)
        struct.pack_into("<I", buf, mo + 13, dtype_code)
        struct.pack_into("<I", buf, mo + 17, data_size)

    def unpack_metadata(self, buf):
        mo = meta_offset(self.num_slots)
        width = struct.unpack_from("<I", buf, mo + 1)[0]
        height = struct.unpack_from("<I", buf, mo + 5)[0]
        channels = struct.unpack_from("<I", buf, mo + 9)[0]
        dtype_code = struct.unpack_from("<I", buf, mo + 13)[0]
        data_size = struct.unpack_from("<I", buf, mo + 17)[0]
        return width, height, channels, dtype_code, data_size

    def get_write_idx(self, buf) -> int:
        return struct.unpack_from("<I", buf, 16)[0]

    def set_write_idx(self, buf, idx: int) -> None:
        struct.pack_into("<I", buf, 16, idx)

    def get_version(self, buf) -> int:
        return struct.unpack_from("<Q", buf, 4)[0]

    def is_shutdown(self, buf) -> bool:
        return buf[meta_offset(self.num_slots)] != 0

    def set_shutdown(self, buf) -> None:
        buf[meta_offset(self.num_slots)] = 1
