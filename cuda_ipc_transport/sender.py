"""CUDAIPCSender — Python process → TouchDesigner via CUDA IPC ring buffer."""
import struct
import time
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from .channel import CUDAIPCChannel
from .protocol import NUM_SLOTS, SHMLayout, shm_size
from .wrapper import get_cuda_runtime


class CUDAIPCSender:
    """Allocates GPU ring buffer, exports IPC handles, sends frames via CUDA IPC.

    Usage:
        ch = CUDAIPCChannel("hagar_out", 512, 512)
        sender = CUDAIPCSender(ch)
        sender.initialize()
        sender.send_numpy(frame)   # numpy (H,W,C) uint8
        sender.send_cuda(ptr, sz)  # raw CUDA device pointer + size
        sender.close()
    """

    def __init__(self, channel: CUDAIPCChannel):
        self.channel = channel
        self._cuda = None
        self._stream = None
        self._dev_ptrs: list = [None] * NUM_SLOTS
        self._ipc_handles: list = []
        self._ipc_events: list = []
        self._ipc_event_handles: list = []
        self._shm: Optional[SharedMemory] = None
        self._layout: Optional[SHMLayout] = None
        self._write_idx = 0
        self._pinned: Optional[np.ndarray] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Allocate GPU buffers, create IPC handles, open SharedMemory."""
        try:
            self._cuda = get_cuda_runtime()
            self._stream = self._cuda.create_stream(0x01)  # cudaStreamNonBlocking

            ch = self.channel
            for slot in range(NUM_SLOTS):
                ptr = self._cuda.malloc(ch.buffer_size)
                self._dev_ptrs[slot] = ptr
                self._ipc_handles.append(self._cuda.ipc_get_mem_handle(ptr))
                evt = self._cuda.create_ipc_event()
                self._ipc_events.append(evt)
                self._ipc_event_handles.append(self._cuda.ipc_get_event_handle(evt))

            # Pinned memory for fast H2D transfer
            self._pinned = np.empty(
                (ch.height, ch.width, ch.channels),
                dtype=np.uint8 if ch.dtype == "uint8" else np.float32,
            )

            # Create or open SharedMemory
            size = shm_size(NUM_SLOTS)
            try:
                self._shm = SharedMemory(name=ch.name, create=True, size=size)
            except FileExistsError:
                self._shm = SharedMemory(name=ch.name, create=False)

            self._layout = SHMLayout(num_slots=NUM_SLOTS)
            self._layout.pack_header(self._shm.buf, version=int(time.time()), write_idx=0)
            self._layout.pack_metadata(
                self._shm.buf,
                width=ch.width, height=ch.height, channels=ch.channels,
                dtype_code=ch.dtype_code, data_size=ch.data_size,
            )

            # Write IPC handles for all slots
            for slot in range(NUM_SLOTS):
                offset = self._layout.slot_offset(slot)
                self._shm.buf[offset:offset + 64] = bytes(self._ipc_handles[slot].internal)
                self._shm.buf[offset + 64:offset + 128] = bytes(self._ipc_event_handles[slot].reserved)

            self._initialized = True
            return True
        except Exception as e:
            print(f"[CUDAIPCSender] init failed: {e}")
            return False

    def send_numpy(self, frame: np.ndarray) -> bool:
        """Copy numpy frame (H,W,C) to GPU ring buffer and signal reader."""
        if not self._initialized:
            return False
        slot = self._write_idx % NUM_SLOTS
        np.copyto(self._pinned, frame)
        src_ptr = self._pinned.ctypes.data_as(c_void_p)
        self._cuda.memcpy_async(
            self._dev_ptrs[slot], src_ptr, self.channel.data_size, 1, self._stream
        )  # kind=1: H2D
        return self._signal(slot)

    def send_cuda(self, ptr: int, size: int) -> bool:
        """Copy from existing CUDA device pointer (D2D) and signal reader."""
        if not self._initialized:
            return False
        slot = self._write_idx % NUM_SLOTS
        self._cuda.memcpy_async(
            self._dev_ptrs[slot], c_void_p(ptr), size, 3, self._stream
        )  # kind=3: D2D
        return self._signal(slot)

    def _signal(self, slot: int) -> bool:
        """Record event THEN increment write_idx (ordering guarantee for reader)."""
        try:
            self._cuda.record_event(self._ipc_events[slot], stream=self._stream)
            self._write_idx += 1
            self._layout.set_write_idx(self._shm.buf, self._write_idx)
            return True
        except Exception as e:
            print(f"[CUDAIPCSender] signal failed: {e}")
            return False

    def is_ready(self) -> bool:
        return self._initialized

    def close(self):
        """Signal shutdown, free GPU resources, unlink SharedMemory."""
        if self._shm and self._shm.buf:
            try:
                self._layout.set_shutdown(self._shm.buf)
            except Exception:
                pass

        if self._cuda:
            for evt in self._ipc_events:
                try:
                    self._cuda.destroy_event(evt)
                except Exception:
                    pass
            if self._stream:
                try:
                    self._cuda.destroy_stream(self._stream)
                except Exception:
                    pass
            for ptr in self._dev_ptrs:
                if ptr:
                    try:
                        self._cuda.free(ptr)
                    except Exception:
                        pass

        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
        self._initialized = False
