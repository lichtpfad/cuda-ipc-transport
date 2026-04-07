"""TDCUDAIPCExporter — TouchDesigner Extension.

Exports TD TOP texture via CUDA IPC to a named channel.
Uses top_op.cudaMemory(stream) for GPU-direct access (no CPU roundtrip).

Deploy in TD:
    1. Create a Base COMP
    2. Add this class as an Extension (DAT → Extensions tab)
    3. Add custom String parameter 'Channelname' = channel name (e.g. "hagar_in")
    4. In a Script DAT onFrameStart: ext.TDCUDAIPCExporter.ExportFrame(op('my_top'))
"""
import struct
import time
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

from ..protocol import NUM_SLOTS, SHMLayout, shm_size
from ..wrapper import get_cuda_runtime


class TDCUDAIPCExporter:
    """Exports TD TOP.cudaMemory() to a CUDA IPC ring buffer.

    Per-frame overhead: ~1µs (async GPU copy + event record).
    No CPU memory copies — GPU-direct.
    """

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self._cuda = None
        self._stream = None
        self._dev_ptrs: list = [None] * NUM_SLOTS
        self._ipc_handles: list = [None] * NUM_SLOTS
        self._ipc_events: list = [None] * NUM_SLOTS
        self._ipc_event_handles: list = [None] * NUM_SLOTS
        self._shm: Optional[SharedMemory] = None
        self._layout: Optional[SHMLayout] = None
        self._write_idx = 0
        self._data_size = 0
        self._buffer_size = 0
        self._width = self._height = self._channels = 0
        self._initialized = False
        self._cuda_mem_ref = None  # prevent GC of cudaMemory object

        # Channel name from component parameter or fallback
        try:
            self._channel_name: str = ownerComp.par.Channelname.eval()
        except AttributeError:
            self._channel_name = f"td_{ownerComp.name}"

    def Initialize(self, width: int, height: int, channels: int,
                   buffer_size: int = None) -> bool:
        """Allocate GPU ring buffer and SharedMemory. Called once at first frame."""
        try:
            self._cuda = get_cuda_runtime()

            if self._stream is None:
                self._stream = self._cuda.create_stream(0x01)  # cudaStreamNonBlocking

            _2MiB = 2 * 1024 * 1024
            raw_size = buffer_size if buffer_size is not None else width * height * channels
            self._data_size = raw_size
            self._buffer_size = (raw_size + _2MiB - 1) // _2MiB * _2MiB
            self._width, self._height, self._channels = width, height, channels

            for slot in range(NUM_SLOTS):
                self._dev_ptrs[slot] = self._cuda.malloc(self._buffer_size)
                self._ipc_handles[slot] = self._cuda.ipc_get_mem_handle(self._dev_ptrs[slot])
                evt = self._cuda.create_ipc_event()
                self._ipc_events[slot] = evt
                self._ipc_event_handles[slot] = self._cuda.ipc_get_event_handle(evt)

            size = shm_size(NUM_SLOTS)
            try:
                self._shm = SharedMemory(name=self._channel_name, create=True, size=size)
            except FileExistsError:
                self._shm = SharedMemory(name=self._channel_name, create=False)

            self._layout = SHMLayout(num_slots=NUM_SLOTS)
            self._layout.pack_header(self._shm.buf, version=int(time.time()), write_idx=0)
            self._layout.pack_metadata(
                self._shm.buf, width=width, height=height, channels=channels,
                dtype_code=2, data_size=raw_size,  # 2 = uint8
            )
            for slot in range(NUM_SLOTS):
                offset = self._layout.slot_offset(slot)
                self._shm.buf[offset:offset + 64] = bytes(self._ipc_handles[slot].internal)
                self._shm.buf[offset + 64:offset + 128] = bytes(
                    self._ipc_event_handles[slot].reserved
                )

            self._initialized = True
            debug(  # noqa: F821 — TD built-in
                f"[TDCUDAIPCExporter] Ready: '{self._channel_name}' {width}x{height}x{channels}"
            )
            return True

        except Exception as e:
            debug(f"[TDCUDAIPCExporter] Initialize failed: {e}")  # noqa: F821
            import traceback
            debug(traceback.format_exc())  # noqa: F821
            return False

    def ExportFrame(self, top_op) -> bool:
        """Export TOP texture to CUDA IPC ring buffer. Call each frame.

        Args:
            top_op: TouchDesigner TOP operator (e.g. op('camera_top'))
        """
        try:
            # Get GPU pointer from TOP (GPU-direct, no CPU copy)
            cuda_mem = top_op.cudaMemory(
                stream=int(self._stream.value) if self._initialized else None
            )
            if cuda_mem is None:
                return False

            # Keep reference to prevent GC during async copy
            self._cuda_mem_ref = cuda_mem

            w = cuda_mem.shape.width
            h = cuda_mem.shape.height
            c = cuda_mem.shape.numComps

            # (Re)initialize if first frame or resolution changed
            if not self._initialized or cuda_mem.size != self._data_size:
                if self._initialized:
                    debug(  # noqa: F821
                        f"[TDCUDAIPCExporter] Resolution changed: "
                        f"{self._width}x{self._height} → {w}x{h}"
                    )
                    self.Cleanup()
                if not self.Initialize(w, h, c, cuda_mem.size):
                    return False

            slot = self._write_idx % NUM_SLOTS

            # Async D2D copy: TOP GPU buffer → our persistent ring buffer slot
            self._cuda.memcpy_async(
                dst=self._dev_ptrs[slot],
                src=c_void_p(cuda_mem.ptr),
                count=self._data_size,
                kind=3,  # cudaMemcpyDeviceToDevice
                stream=self._stream,
            )

            # Record event THEN increment write_idx (sync ordering guarantee)
            self._cuda.record_event(self._ipc_events[slot], stream=self._stream)
            self._write_idx += 1
            self._layout.set_write_idx(self._shm.buf, self._write_idx)

            return True

        except Exception as e:
            debug(f"[TDCUDAIPCExporter] ExportFrame error: {e}")  # noqa: F821
            return False

    def Cleanup(self):
        """Release GPU and SharedMemory resources.

        Call before TD project closes or when restarting the extension.
        Order: signal shutdown → destroy events → destroy stream → free GPU → close SHM.
        """
        if not self._initialized and self._shm is None:
            return

        # Step 1: Signal shutdown to consumer
        if self._shm and self._shm.buf:
            try:
                self._layout.set_shutdown(self._shm.buf)
            except Exception:
                pass

        if self._cuda:
            # Step 2: Destroy IPC events
            for evt in self._ipc_events:
                if evt:
                    try:
                        self._cuda.destroy_event(evt)
                    except Exception:
                        pass

            # Step 3: Destroy stream
            if self._stream:
                try:
                    self._cuda.destroy_stream(self._stream)
                    self._stream = None
                except Exception:
                    pass

            # Step 4: Free GPU buffers
            for ptr in self._dev_ptrs:
                if ptr:
                    try:
                        self._cuda.free(ptr)
                    except Exception:
                        pass

        # Step 5: Close SharedMemory
        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

        self._initialized = False
        self._dev_ptrs = [None] * NUM_SLOTS
        self._ipc_events = [None] * NUM_SLOTS
        self._ipc_handles = [None] * NUM_SLOTS
        self._ipc_event_handles = [None] * NUM_SLOTS

    def IsReady(self) -> bool:
        return self._initialized

    def __delTD__(self):
        """Called when TD reinitializes the extension."""
        self.Cleanup()
