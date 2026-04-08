"""CUDAIPCReceiver — read GPU frames from CUDA IPC ring buffer."""
import ctypes
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

from .protocol import MAGIC, SHMLayout
from .wrapper import get_cuda_runtime, cudaIpcMemHandle_t


class CUDAIPCReceiver:
    """Opens SharedMemory, imports IPC handles, returns GPU frame pointers.

    Reads num_slots from the SharedMemory header — compatible with both
    2-slot (StreamDiffusion) and 3-slot (new senders) writers.

    Usage:
        r = CUDAIPCReceiver("hagar_out")
        r.connect()
        ptr, size, shape = r.get_frame()  # (int, int, (H,W,C)) or (None,0,())
        r.close()
    """

    def __init__(self, channel_name: str):
        self.channel_name = channel_name
        self._shm: Optional[SharedMemory] = None
        self._buf = None
        self._cuda = None
        self._layout: Optional[SHMLayout] = None
        self._num_slots = 0
        self._opened_ptrs: dict = {}
        self._version = 0
        self._last_write_idx = -1
        self._width = self._height = self._channels = self._buffer_size = 0
        self._ready = False

    def connect(self) -> bool:
        """Open SharedMemory, validate protocol, import IPC handles."""
        self.close()
        try:
            self._shm = SharedMemory(name=self.channel_name, create=False)
            self._buf = self._shm.buf
        except FileNotFoundError:
            print(f"[CUDAIPCReceiver] SharedMemory '{self.channel_name}' not found")
            return False
        except Exception as e:
            print(f"[CUDAIPCReceiver] SharedMemory open error: {e}")
            return False

        # Validate magic
        magic = struct.unpack_from("<I", self._buf, 0)[0]
        if magic != MAGIC:
            print(f"[CUDAIPCReceiver] Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")
            self._close_shm()
            return False

        # Read num_slots from header (not hardcoded!)
        self._num_slots = struct.unpack_from("<I", self._buf, 12)[0]
        self._version = struct.unpack_from("<Q", self._buf, 4)[0]
        self._layout = SHMLayout(num_slots=self._num_slots)

        # Read metadata
        mo = self._layout.meta_offset()
        self._width = struct.unpack_from("<I", self._buf, mo + 1)[0]
        self._height = struct.unpack_from("<I", self._buf, mo + 5)[0]
        self._channels = struct.unpack_from("<I", self._buf, mo + 9)[0]
        self._buffer_size = struct.unpack_from("<I", self._buf, mo + 17)[0]

        if not (self._width and self._height and self._buffer_size):
            print(f"[CUDAIPCReceiver] Metadata not ready: {self._width}x{self._height}")
            self._close_shm()
            return False

        try:
            self._cuda = get_cuda_runtime()
        except Exception as e:
            print(f"[CUDAIPCReceiver] CUDA init failed: {e}")
            return False

        self._open_ipc_handles()
        self._ready = True
        print(f"[CUDAIPCReceiver] Connected: {self.channel_name} "
              f"{self._width}x{self._height}x{self._channels} slots={self._num_slots}")
        return True

    def _open_ipc_handles(self):
        self._close_ipc_handles()
        for slot in range(self._num_slots):
            offset = 20 + slot * 128  # SHM_HEADER_SIZE + slot * SLOT_SIZE
            mem_handle = cudaIpcMemHandle_t()
            ctypes.memmove(ctypes.addressof(mem_handle), bytes(self._buf[offset:offset + 64]), 64)
            try:
                ptr = self._cuda.ipc_open_mem_handle(mem_handle)
                self._opened_ptrs[slot] = ptr
            except Exception as e:
                print(f"[CUDAIPCReceiver] slot {slot} open failed: {type(e).__name__}: {e}")
                self._last_ipc_error = str(e)

    def _close_ipc_handles(self):
        if self._cuda:
            for ptr in self._opened_ptrs.values():
                try:
                    self._cuda.ipc_close_mem_handle(ptr)
                except Exception:
                    pass
        self._opened_ptrs.clear()

    def _close_shm(self):
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
            self._buf = None

    def is_ready(self) -> bool:
        return self._ready and bool(self._opened_ptrs)

    def get_frame(self) -> Tuple:
        """Return (cuda_ptr_int, data_size, (height, width, channels)) or (None, 0, ())."""
        if not self.is_ready():
            return (None, 0, ())

        # Auto-reconnect on version change (sender restarted)
        cur_version = struct.unpack_from("<Q", self._buf, 4)[0]
        if cur_version != self._version:
            print(f"[CUDAIPCReceiver] Version changed, reconnecting")
            self.reconnect()
            return (None, 0, ())

        # Check shutdown flag
        if self._layout.is_shutdown(self._buf):
            print("[CUDAIPCReceiver] Shutdown detected")
            self.close()
            return (None, 0, ())

        write_idx = struct.unpack_from("<I", self._buf, 16)[0]
        if write_idx == self._last_write_idx:
            return (None, 0, ())  # No new frame

        # Ring buffer ordering: read slot written at write_idx-1
        # Sender already wrote to slot N-1 and moved to slot N, so slot N-1 is safe
        slot = (write_idx - 1) % self._num_slots
        if slot not in self._opened_ptrs:
            return (None, 0, ())

        self._last_write_idx = write_idx
        ptr = self._opened_ptrs[slot]
        ptr_int = ptr.value if hasattr(ptr, "value") else int(ptr)
        return (ptr_int, self._buffer_size, (self._height, self._width, self._channels))

    def reconnect(self):
        self.close()
        self.connect()

    def close(self):
        self._ready = False
        self._close_ipc_handles()
        self._close_shm()
        self._last_write_idx = -1


# Module-level singleton for backward compatibility with old cuda_ipc_reader.py
_default_reader: Optional[CUDAIPCReceiver] = None


def get_reader(channel_name: str = "sd_to_td_ipc") -> CUDAIPCReceiver:
    """Drop-in replacement for old cuda_ipc_reader.get_reader().

    Old callers (sd_output_reader_ipc_callbacks.py): get_reader() → "sd_to_td_ipc"
    New callers (td/importer.py): get_reader("my_channel") → explicit name
    """
    global _default_reader
    if _default_reader is None:
        _default_reader = CUDAIPCReceiver(channel_name)
        _default_reader.connect()
    return _default_reader


def cleanup():
    """Release module-level singleton."""
    global _default_reader
    if _default_reader is not None:
        _default_reader.close()
        _default_reader = None
