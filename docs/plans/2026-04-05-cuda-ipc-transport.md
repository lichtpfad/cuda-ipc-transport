# cuda_ipc_transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Python package for bidirectional GPU-direct CUDA IPC frame transfer between any Python process and TouchDesigner, with pluggable sources and named channels.

**Architecture:** `CUDAIPCSender` allocates persistent GPU ring buffers and exports IPC handles via SharedMemory. `CUDAIPCReceiver` imports handles and returns raw CUDA pointers for zero-copy access. TD extensions (`TDCUDAIPCExporter`, `TDCUDAIPCImporter`) wrap these for use inside TouchDesigner. Sync policy: ring buffer ordering only (Windows IPC events are cross-process broken).

**Tech Stack:** Python 3.11, ctypes (cudart64_12.dll), numpy, opencv-python, multiprocessing.shared_memory. No torch dependency.

**Spec:** `C:/work/ANNIEQ/docs/superpowers/specs/2026-04-05-cuda-ipc-transport-design.md`

---

## File Map

```
C:/dev/cuda/
  cuda_ipc_transport/
    __init__.py          exports: CUDAIPCChannel, CUDAIPCSender, CUDAIPCReceiver, get_reader
    __main__.py          python -m cuda_ipc_transport → harness.main()
    wrapper.py           CUDARuntimeAPI, get_cuda_runtime() — from CUDAIPCWrapper.py
    protocol.py          MAGIC, NUM_SLOTS, SHMLayout, shm_size(), pack/unpack helpers
    channel.py           CUDAIPCChannel(name, width, height, channels, dtype)
    sender.py            CUDAIPCSender(channel) — send_numpy(), send_cuda(), close()
    receiver.py          CUDAIPCReceiver(channel_name) + get_reader() compat shim
    td/
      __init__.py
      exporter.py        TDCUDAIPCExporter(ownerComp) — ExportFrame(top_op)
      importer.py        onCook(scriptOp), onSetupParameters, onPulse
    sources/
      __init__.py
      base.py            Source ABC
      test_pattern.py    TestPatternSource(width, height, fps)
      file.py            FileSource(path)
      camera.py          CameraSource(device_id, width, height)
    harness.py           main(args) — source loop
  tests/
    test_protocol.py     pure Python, no GPU required
    test_channel.py      pure Python, no GPU required
    test_sources.py      numpy only, no GPU required
    test_integration.py  requires CUDA GPU — sender→receiver roundtrip
  pyproject.toml
  .venv/                 Python 3.11 venv
```

---

## Task 1: Repo setup

**Files:**
- Create: `C:/dev/cuda/pyproject.toml`
- Create: `C:/dev/cuda/cuda_ipc_transport/__init__.py`

- [ ] **Step 1: Create venv**

```bash
cd C:/dev/cuda
python -m venv .venv
.venv/Scripts/activate
pip install numpy opencv-python pytest
```

Expected: `.venv/Scripts/python.exe` exists, `pip list` shows numpy, opencv-python, pytest.

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "cuda_ipc_transport"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy", "opencv-python"]

[tool.setuptools.packages.find]
where = ["."]
include = ["cuda_ipc_transport*"]
```

- [ ] **Step 3: Install package in editable mode**

```bash
cd C:/dev/cuda
.venv/Scripts/pip install -e .
```

Expected: `pip show cuda_ipc_transport` shows `Location: c:\dev\cuda`.

- [ ] **Step 4: Create package skeleton**

`cuda_ipc_transport/__init__.py`:
```python
from .channel import CUDAIPCChannel
from .sender import CUDAIPCSender
from .receiver import CUDAIPCReceiver, get_reader

__all__ = ["CUDAIPCChannel", "CUDAIPCSender", "CUDAIPCReceiver", "get_reader"]
```

`cuda_ipc_transport/td/__init__.py`: empty file.
`cuda_ipc_transport/sources/__init__.py`: empty file.

- [ ] **Step 5: Verify import**

```bash
.venv/Scripts/python -c "import cuda_ipc_transport; print('ok')"
```

Expected: `ok`

- [ ] **Step 6: Commit**

```bash
cd C:/dev/cuda
git add pyproject.toml cuda_ipc_transport/
git commit -m "feat: package skeleton with pyproject.toml"
```

---

## Task 2: wrapper.py

Copy and adapt `CUDAIPCWrapper.py` from ANNIEQ — no logic changes, only move to package.

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/wrapper.py`

- [ ] **Step 1: Copy wrapper**

```bash
cp C:/work/ANNIEQ/scripts/CUDAIPCWrapper.py C:/dev/cuda/cuda_ipc_transport/wrapper.py
```

- [ ] **Step 2: Verify it loads**

```bash
cd C:/dev/cuda
.venv/Scripts/python -c "from cuda_ipc_transport.wrapper import get_cuda_runtime; print('ok')"
```

Expected: `ok` (does NOT require GPU for import, only for `get_cuda_runtime()` call).

- [ ] **Step 3: Commit**

```bash
git add cuda_ipc_transport/wrapper.py
git commit -m "feat: add wrapper.py (CUDARuntimeAPI ctypes wrapper)"
```

---

## Task 3: protocol.py

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/protocol.py`
- Create: `C:/dev/cuda/tests/test_protocol.py`

- [ ] **Step 1: Write failing tests**

`tests/test_protocol.py`:
```python
import struct
from cuda_ipc_transport.protocol import (
    MAGIC, NUM_SLOTS, SLOT_SIZE, SHM_HEADER_SIZE,
    shm_size, meta_offset, SHMLayout,
)


def test_magic():
    assert MAGIC == 0x43495043
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
```

- [ ] **Step 2: Run to verify fail**

```bash
cd C:/dev/cuda
.venv/Scripts/pytest tests/test_protocol.py -v
```

Expected: `ModuleNotFoundError: No module named 'cuda_ipc_transport.protocol'`

- [ ] **Step 3: Implement protocol.py**

```python
"""SharedMemory protocol layout for cuda_ipc_transport v1.0."""
import struct
from dataclasses import dataclass

MAGIC = 0x43495043          # "CIPC"
NUM_SLOTS = 3               # default for new writers
SLOT_SIZE = 128             # 64B mem_handle + 64B event_handle
SHM_HEADER_SIZE = 20        # magic(4) + version(8) + num_slots(4) + write_idx(4)

DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_UINT8 = 2

# Metadata layout (after shutdown flag)
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/pytest tests/test_protocol.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cuda_ipc_transport/protocol.py tests/test_protocol.py
git commit -m "feat: protocol.py — SharedMemory layout v1.0"
```

---

## Task 4: channel.py

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/channel.py`
- Create: `C:/dev/cuda/tests/test_channel.py`

- [ ] **Step 1: Write failing tests**

`tests/test_channel.py`:
```python
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
```

- [ ] **Step 2: Run to verify fail**

```bash
.venv/Scripts/pytest tests/test_channel.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement channel.py**

```python
"""Channel configuration — no CUDA, no SharedMemory."""
from dataclasses import dataclass

_DTYPE_BYTES = {"uint8": 1, "float16": 2, "float32": 4}
_2MiB = 2 * 1024 * 1024


@dataclass
class CUDAIPCChannel:
    name: str
    width: int
    height: int
    channels: int = 4
    dtype: str = "uint8"   # "uint8" | "float16" | "float32"

    @property
    def data_size(self) -> int:
        return self.width * self.height * self.channels * _DTYPE_BYTES[self.dtype]

    @property
    def buffer_size(self) -> int:
        """data_size rounded up to 2 MiB (NVIDIA IPC requirement)."""
        return (self.data_size + _2MiB - 1) // _2MiB * _2MiB

    @property
    def dtype_code(self) -> int:
        return {"float32": 0, "float16": 1, "uint8": 2}[self.dtype]
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/pytest tests/test_channel.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cuda_ipc_transport/channel.py tests/test_channel.py
git commit -m "feat: channel.py — CUDAIPCChannel config dataclass"
```

---

## Task 5: sender.py

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/sender.py`

No automated unit tests — `CUDAIPCSender` requires a live CUDA GPU and creates real SharedMemory. Integration test is in Task 11.

- [ ] **Step 1: Implement sender.py**

```python
"""CUDAIPCSender — Python process → TouchDesigner via CUDA IPC ring buffer."""
import ctypes
import struct
import time
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from .channel import CUDAIPCChannel
from .protocol import DTYPE_UINT8, NUM_SLOTS, SHMLayout, shm_size
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
        self._dev_ptrs = [None] * NUM_SLOTS
        self._ipc_handles = []
        self._ipc_events = []
        self._ipc_event_handles = []
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

            # SharedMemory
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
        """Copy numpy frame (H,W,C) uint8 to GPU and signal ring buffer."""
        if not self._initialized:
            return False
        slot = self._write_idx % NUM_SLOTS
        np.copyto(self._pinned, frame)
        src_ptr = self._pinned.ctypes.data_as(c_void_p)
        self._cuda.memcpy_async(self._dev_ptrs[slot], src_ptr, self.channel.data_size, 1, self._stream)
        return self._signal(slot)

    def send_cuda(self, ptr: int, size: int) -> bool:
        """Copy from existing CUDA device pointer (D2D) and signal ring buffer."""
        if not self._initialized:
            return False
        slot = self._write_idx % NUM_SLOTS
        self._cuda.memcpy_async(self._dev_ptrs[slot], c_void_p(ptr), size, 3, self._stream)
        return self._signal(slot)

    def _signal(self, slot: int) -> bool:
        """Record event then increment write_idx (ordering guarantee)."""
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
            self._layout.set_shutdown(self._shm.buf)

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
```

- [ ] **Step 2: Verify import**

```bash
.venv/Scripts/python -c "from cuda_ipc_transport.sender import CUDAIPCSender; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add cuda_ipc_transport/sender.py
git commit -m "feat: sender.py — CUDAIPCSender with send_numpy/send_cuda"
```

---

## Task 6: receiver.py

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/receiver.py`

Integration test in Task 11. Pure unit test for compat shim only.

- [ ] **Step 1: Write compat shim test**

`tests/test_receiver_compat.py`:
```python
from unittest.mock import patch, MagicMock

def test_get_reader_default_channel():
    """get_reader() without args uses 'sd_to_td_ipc' channel name."""
    mock_receiver = MagicMock()
    mock_receiver.connect = MagicMock()

    with patch("cuda_ipc_transport.receiver.CUDAIPCReceiver", return_value=mock_receiver) as MockClass:
        # Reset module-level singleton
        import cuda_ipc_transport.receiver as mod
        mod._default_reader = None

        from cuda_ipc_transport.receiver import get_reader
        reader = get_reader()
        MockClass.assert_called_once_with("sd_to_td_ipc")
        mock_receiver.connect.assert_called_once()


def test_get_reader_custom_channel():
    """get_reader('my_channel') uses custom channel name."""
    mock_receiver = MagicMock()

    with patch("cuda_ipc_transport.receiver.CUDAIPCReceiver", return_value=mock_receiver) as MockClass:
        import cuda_ipc_transport.receiver as mod
        mod._default_reader = None

        from cuda_ipc_transport.receiver import get_reader
        get_reader("my_channel")
        MockClass.assert_called_once_with("my_channel")


def test_get_reader_singleton():
    """Second call to get_reader() returns same instance."""
    mock_receiver = MagicMock()

    with patch("cuda_ipc_transport.receiver.CUDAIPCReceiver", return_value=mock_receiver):
        import cuda_ipc_transport.receiver as mod
        mod._default_reader = None

        from cuda_ipc_transport.receiver import get_reader
        r1 = get_reader()
        r2 = get_reader()
        assert r1 is r2
```

- [ ] **Step 2: Run to verify fail**

```bash
.venv/Scripts/pytest tests/test_receiver_compat.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement receiver.py**

```python
"""CUDAIPCReceiver — read GPU frames from CUDA IPC ring buffer."""
import ctypes
import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

from .protocol import MAGIC, SHMLayout, shm_size
from .wrapper import get_cuda_runtime, cudaIpcMemHandle_t, cudaIpcEventHandle_t


class CUDAIPCReceiver:
    """Opens SharedMemory, imports IPC handles, returns GPU frame pointers.

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
        self._opened_ptrs = {}   # slot -> c_void_p
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
            return False

        magic = struct.unpack_from("<I", self._buf, 0)[0]
        if magic != MAGIC:
            self._close_shm()
            return False

        # Read num_slots from header (not hardcoded)
        self._num_slots = struct.unpack_from("<I", self._buf, 12)[0]
        self._version = struct.unpack_from("<Q", self._buf, 4)[0]
        self._layout = SHMLayout(num_slots=self._num_slots)

        # Read metadata
        mo = self._layout.meta_offset() if hasattr(self._layout, 'meta_offset') else \
             20 + self._num_slots * 128
        self._width = struct.unpack_from("<I", self._buf, mo + 1)[0]
        self._height = struct.unpack_from("<I", self._buf, mo + 5)[0]
        self._channels = struct.unpack_from("<I", self._buf, mo + 9)[0]
        self._buffer_size = struct.unpack_from("<I", self._buf, mo + 17)[0]

        if not (self._width and self._height and self._buffer_size):
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
            offset = 20 + slot * 128
            mem_handle = cudaIpcMemHandle_t()
            ctypes.memmove(ctypes.addressof(mem_handle), bytes(self._buf[offset:offset + 64]), 64)
            try:
                ptr = self._cuda.ipc_open_mem_handle(mem_handle)
                self._opened_ptrs[slot] = ptr
            except RuntimeError as e:
                print(f"[CUDAIPCReceiver] slot {slot} open failed: {e}")

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

        # Auto-reconnect on version change
        cur_version = struct.unpack_from("<Q", self._buf, 4)[0]
        if cur_version != self._version:
            self.reconnect()
            return (None, 0, ())

        # Check shutdown flag
        mo = 20 + self._num_slots * 128
        if self._buf[mo]:
            self.close()
            return (None, 0, ())

        write_idx = struct.unpack_from("<I", self._buf, 16)[0]
        if write_idx == self._last_write_idx:
            return (None, 0, ())

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


# Module-level singleton (backward compat with old cuda_ipc_reader.get_reader())
_default_reader: Optional[CUDAIPCReceiver] = None


def get_reader(channel_name: str = "sd_to_td_ipc") -> CUDAIPCReceiver:
    """Drop-in replacement for old cuda_ipc_reader.get_reader().

    Old callers: get_reader()            → uses "sd_to_td_ipc"
    New callers: get_reader("my_channel") → explicit channel
    """
    global _default_reader
    if _default_reader is None:
        _default_reader = CUDAIPCReceiver(channel_name)
        _default_reader.connect()
    return _default_reader


def cleanup():
    """Release singleton (call on shutdown)."""
    global _default_reader
    if _default_reader is not None:
        _default_reader.close()
        _default_reader = None
```

Also add `meta_offset` method to `SHMLayout` in `protocol.py`:

```python
# In SHMLayout class, add:
def meta_offset(self) -> int:
    from .protocol import meta_offset as _mo
    return _mo(self.num_slots)
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/pytest tests/test_receiver_compat.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cuda_ipc_transport/receiver.py tests/test_receiver_compat.py
git commit -m "feat: receiver.py — CUDAIPCReceiver + get_reader() compat shim"
```

---

## Task 7: sources/

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/sources/base.py`
- Create: `C:/dev/cuda/cuda_ipc_transport/sources/test_pattern.py`
- Create: `C:/dev/cuda/cuda_ipc_transport/sources/file.py`
- Create: `C:/dev/cuda/cuda_ipc_transport/sources/camera.py`
- Create: `C:/dev/cuda/tests/test_sources.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sources.py`:
```python
import numpy as np
from cuda_ipc_transport.sources.test_pattern import TestPatternSource


def test_test_pattern_shape():
    src = TestPatternSource(width=512, height=512)
    frame = src.get_frame()
    assert frame.shape == (512, 512, 4)
    assert frame.dtype == np.uint8


def test_test_pattern_increments():
    src = TestPatternSource(width=64, height=64)
    f1 = src.get_frame()
    f2 = src.get_frame()
    # frame counter increments — frames differ
    assert not np.array_equal(f1, f2)


def test_test_pattern_close():
    src = TestPatternSource(width=64, height=64)
    src.close()  # should not raise
```

- [ ] **Step 2: Run to verify fail**

```bash
.venv/Scripts/pytest tests/test_sources.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement sources/**

`sources/base.py`:
```python
from abc import ABC, abstractmethod
import numpy as np


class Source(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Return (H, W, C) uint8 numpy array."""
        ...

    def close(self):
        pass
```

`sources/test_pattern.py`:
```python
import numpy as np
import cv2
from .base import Source


class TestPatternSource(Source):
    """Generates color bars with frame counter overlay."""

    def __init__(self, width: int = 512, height: int = 512, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._frame = 0
        self._colors = [
            (255, 255, 255), (255, 255, 0), (0, 255, 255),
            (0, 255, 0),     (255, 0, 255), (255, 0, 0),
            (0, 0, 255),     (0, 0, 0),
        ]

    def get_frame(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        bar_w = self.width // len(self._colors)
        for i, (b, g, r) in enumerate(self._colors):
            x0, x1 = i * bar_w, (i + 1) * bar_w
            img[:, x0:x1, 0] = b
            img[:, x0:x1, 1] = g
            img[:, x0:x1, 2] = r
            img[:, x0:x1, 3] = 255  # alpha
        # Frame counter
        cv2.putText(img, f"frame {self._frame}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 255), 2)
        self._frame += 1
        return img
```

`sources/file.py`:
```python
import cv2
import numpy as np
from pathlib import Path
from .base import Source


class FileSource(Source):
    """Reads image or video file, loops automatically."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._cap = None
        self._single_frame = None
        if self._path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(self._path), cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self._single_frame = img
        else:
            self._cap = cv2.VideoCapture(str(self._path))

    def get_frame(self) -> np.ndarray:
        if self._single_frame is not None:
            return self._single_frame
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return frame.astype(np.uint8)

    def close(self):
        if self._cap:
            self._cap.release()
```

`sources/camera.py`:
```python
import cv2
import numpy as np
from .base import Source


class CameraSource(Source):
    """OpenCV webcam capture."""

    def __init__(self, device_id: int = 0, width: int = 512, height: int = 512):
        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            return np.zeros((512, 512, 4), dtype=np.uint8)
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return bgra.astype(np.uint8)

    def close(self):
        self._cap.release()
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/pytest tests/test_sources.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cuda_ipc_transport/sources/ tests/test_sources.py
git commit -m "feat: sources — base, test_pattern, file, camera"
```

---

## Task 8: harness.py + __main__.py

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/harness.py`
- Create: `C:/dev/cuda/cuda_ipc_transport/__main__.py`

- [ ] **Step 1: Implement harness.py**

```python
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
    parser = argparse.ArgumentParser(prog="cuda_ipc_transport")
    parser.add_argument("--source", choices=["test", "file", "camera"], default="test")
    parser.add_argument("--channel", default="cuda_ipc_test")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0,
                        help="Number of frames to send (0 = infinite)")
    parser.add_argument("--file", default=None, help="Path for --source file")
    args = parser.parse_args(argv)

    if args.source == "test":
        source = TestPatternSource(args.width, args.height, args.fps)
    elif args.source == "file":
        if not args.file:
            print("ERROR: --file required with --source file")
            sys.exit(1)
        source = FileSource(args.file)
    else:
        source = CameraSource(0, args.width, args.height)

    channel = CUDAIPCChannel(args.channel, args.width, args.height)
    sender = CUDAIPCSender(channel)

    if not sender.initialize():
        print("ERROR: sender.initialize() failed")
        sys.exit(1)

    print(f"[harness] Sending {args.source} → '{args.channel}' at {args.fps} fps")
    print(f"[harness] Press Ctrl+C to stop")

    # Graceful shutdown on Ctrl+C
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
```

- [ ] **Step 2: Implement __main__.py**

```python
from cuda_ipc_transport.harness import main
main()
```

- [ ] **Step 3: Verify CLI**

```bash
.venv/Scripts/python -m cuda_ipc_transport --help
```

Expected: prints argparse help with `--source`, `--channel`, `--width`, `--height`, `--fps`, `--frames`.

- [ ] **Step 4: Commit**

```bash
git add cuda_ipc_transport/harness.py cuda_ipc_transport/__main__.py
git commit -m "feat: harness + __main__ — CLI entry point"
```

---

## Task 9: td/exporter.py

Port of Alex's `CUDAIPCExporter.py` using `protocol.py`. This is a TD Extension — no automated tests, manual verification in Task 12.

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/td/exporter.py`

- [ ] **Step 1: Implement td/exporter.py**

```python
"""TDCUDAIPCExporter — TouchDesigner Extension.

Exports TD TOP texture via CUDA IPC to a named channel.

Usage in TD:
    1. Create COMP, add this file as an Extension
    2. In Script DAT (onFrameStart or onCook):
       ext.TDCUDAIPCExporter.ExportFrame(op('my_top'))
"""
import struct
import time
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

from ..protocol import NUM_SLOTS, SHMLayout, shm_size
from ..wrapper import get_cuda_runtime


class TDCUDAIPCExporter:
    """TD Extension: exports TOP.cudaMemory() → CUDA IPC ring buffer."""

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self._cuda = None
        self._stream = None
        self._dev_ptrs = [None] * NUM_SLOTS
        self._ipc_handles = [None] * NUM_SLOTS
        self._ipc_events = [None] * NUM_SLOTS
        self._ipc_event_handles = [None] * NUM_SLOTS
        self._shm: Optional[SharedMemory] = None
        self._layout: Optional[SHMLayout] = None
        self._write_idx = 0
        self._data_size = 0
        self._buffer_size = 0
        self._width = self._height = self._channels = 0
        self._initialized = False
        self._cuda_mem_ref = None  # prevent GC

        # Channel name from component parameter or fallback to comp name
        try:
            self._channel_name = ownerComp.par.Channelname.eval()
        except AttributeError:
            self._channel_name = f"td_{ownerComp.name}"

    def Initialize(self, width: int, height: int, channels: int, buffer_size: int = None) -> bool:
        """Allocate GPU buffers and SharedMemory. Called once at first frame."""
        try:
            self._cuda = get_cuda_runtime()

            if self._stream is None:
                self._stream = self._cuda.create_stream(0x01)

            _2MiB = 2 * 1024 * 1024
            raw_size = buffer_size if buffer_size else width * height * channels
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
                self._shm.buf[offset + 64:offset + 128] = bytes(self._ipc_event_handles[slot].reserved)

            self._initialized = True
            debug(f"[TDCUDAIPCExporter] Ready: {self._channel_name} {width}x{height}x{channels}")  # noqa
            return True
        except Exception as e:
            debug(f"[TDCUDAIPCExporter] Initialize failed: {e}")  # noqa
            return False

    def ExportFrame(self, top_op) -> bool:
        """Export TOP texture to CUDA IPC ring buffer. Call each frame."""
        try:
            cuda_mem = top_op.cudaMemory(stream=int(self._stream.value) if self._initialized else None)
            if cuda_mem is None:
                return False
            self._cuda_mem_ref = cuda_mem  # prevent GC

            w, h, c = cuda_mem.shape.width, cuda_mem.shape.height, cuda_mem.shape.numComps
            if not self._initialized or cuda_mem.size != self._data_size:
                if self._initialized:
                    self.Cleanup()
                if not self.Initialize(w, h, c, cuda_mem.size):
                    return False

            slot = self._write_idx % NUM_SLOTS
            self._cuda.memcpy_async(
                dst=self._dev_ptrs[slot],
                src=c_void_p(cuda_mem.ptr),
                count=self._data_size,
                kind=3,  # D2D
                stream=self._stream,
            )
            self._cuda.record_event(self._ipc_events[slot], stream=self._stream)
            self._write_idx += 1
            self._layout.set_write_idx(self._shm.buf, self._write_idx)
            return True
        except Exception as e:
            debug(f"[TDCUDAIPCExporter] ExportFrame error: {e}")  # noqa
            return False

    def Cleanup(self):
        """Release GPU and SharedMemory resources."""
        if not self._initialized and self._shm is None:
            return
        if self._shm and self._shm.buf:
            self._layout.set_shutdown(self._shm.buf)

        if self._cuda:
            for evt in self._ipc_events:
                if evt:
                    try:
                        self._cuda.destroy_event(evt)
                    except Exception:
                        pass
            if self._stream:
                try:
                    self._cuda.destroy_stream(self._stream)
                    self._stream = None
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
            self._shm = None

        self._initialized = False

    def IsReady(self) -> bool:
        return self._initialized

    def __delTD__(self):
        self.Cleanup()
```

- [ ] **Step 2: Verify import outside TD**

```bash
.venv/Scripts/python -c "from cuda_ipc_transport.td.exporter import TDCUDAIPCExporter; print('ok')"
```

Expected: `ok` (no TD runtime needed for import)

- [ ] **Step 3: Commit**

```bash
git add cuda_ipc_transport/td/exporter.py
git commit -m "feat: td/exporter.py — TDCUDAIPCExporter (port of CUDAIPCExporter)"
```

---

## Task 10: td/importer.py

Refactor of `sd_output_reader_ipc_callbacks.py` using `CUDAIPCReceiver`.

**Files:**
- Create: `C:/dev/cuda/cuda_ipc_transport/td/importer.py`

- [ ] **Step 1: Implement td/importer.py**

```python
"""TD Script TOP callbacks — receive CUDA IPC frames into TouchDesigner.

Deploy:
    1. Add C:/dev/cuda to sys.path in a startup Script DAT
    2. Create Script TOP, assign this file as its script
    3. Connect an input TOP to Script TOP (triggers cooking)
    4. Set component parameter `Channelname` to SharedMemory channel name

Usage in TD:
    The Script TOP cooks each frame and calls onCook(scriptOp).
    onCook reads the next CUDA IPC frame and copies it to the Script TOP output.
"""
import struct
import time

import numpy as np

_channel_name = None
_receiver = None
_reconnect_cooldown = 0.0
_stream = None
_mem_shape = None
_frame_count = 0
_LOG_INTERVAL = 60


def _get_receiver():
    global _receiver, _channel_name
    if _receiver is None:
        try:
            from cuda_ipc_transport.receiver import CUDAIPCReceiver
            # Read channel name from Script TOP's parent component parameter
            try:
                _channel_name = scriptOp.parent().par.Channelname.eval()  # noqa
            except Exception:
                _channel_name = "sd_to_td_ipc"
            _receiver = CUDAIPCReceiver(_channel_name)
            _receiver.connect()
        except Exception as e:
            debug(f"[importer] init failed: {e}")  # noqa
    return _receiver


def onCook(scriptOp):
    global _reconnect_cooldown, _stream, _mem_shape, _frame_count

    receiver = _get_receiver()
    if receiver is None:
        return

    if not receiver.is_ready():
        now = time.time()
        if now > _reconnect_cooldown:
            receiver.reconnect()
            _reconnect_cooldown = now + 2.0
        return

    if _stream is None:
        try:
            from cuda_ipc_transport.wrapper import get_cuda_runtime
            cuda = get_cuda_runtime()
            _stream = cuda.create_stream(0x01)
            s = _stream.value
            debug(f"[importer] CUDA stream: 0x{s:016x}")  # noqa
        except Exception as e:
            debug(f"[importer] stream creation failed: {e}")  # noqa

    t0 = time.perf_counter()
    ptr, size, shape = receiver.get_frame()
    t1 = time.perf_counter()

    if ptr is None:
        return

    if _mem_shape is None:
        _mem_shape = CUDAMemoryShape()  # noqa -- TD built-in
        _mem_shape.height = shape[0]
        _mem_shape.width = shape[1]
        _mem_shape.numComps = shape[2]
        _mem_shape.dataType = np.uint8

    try:
        import struct as _struct
        if _stream is not None:
            s = _struct.unpack("<Q", bytes(_stream))[0]
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape, stream=s)
        else:
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape)
    except Exception as e:
        debug(f"[importer] copyCUDAMemory error: {e}")  # noqa
        try:
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape)
        except Exception:
            pass

    t2 = time.perf_counter()
    _frame_count += 1
    if _frame_count % _LOG_INTERVAL == 0:
        get_ms = (t1 - t0) * 1000
        copy_ms = (t2 - t1) * 1000
        debug(f"[importer] #{_frame_count} get={get_ms:.1f}ms copy={copy_ms:.1f}ms")  # noqa


def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return
```

- [ ] **Step 2: Verify import**

```bash
.venv/Scripts/python -c "import ast; ast.parse(open('cuda_ipc_transport/td/importer.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add cuda_ipc_transport/td/importer.py
git commit -m "feat: td/importer.py — Script TOP callbacks using CUDAIPCReceiver"
```

---

## Task 11: Integration test (sender → receiver, no TD)

Requires real CUDA GPU. Tests the full protocol roundtrip in a single process.

**Files:**
- Create: `C:/dev/cuda/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""Integration test: CUDAIPCSender → SharedMemory → CUDAIPCReceiver.

Requires CUDA GPU. Run with:
    pytest tests/test_integration.py -v -m cuda
"""
import time
import pytest
import numpy as np

pytest.importorskip("cuda_ipc_transport.wrapper", reason="No CUDA runtime")


@pytest.mark.cuda
def test_sender_receiver_roundtrip():
    """Sender initializes, sends 10 frames, receiver reads all 10."""
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
```

- [ ] **Step 2: Run integration tests**

```bash
.venv/Scripts/pytest tests/test_integration.py -v -m cuda
```

Expected: 2 tests PASS. If GPU not available, tests are skipped.

- [ ] **Step 3: Run all tests (full suite)**

```bash
.venv/Scripts/pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: all non-GPU tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test — sender→receiver roundtrip + VRAM leak check"
```

---

## Task 12: TD deployment + manual test

Deploy to ANNIEQ project and verify full pipeline: TD test pattern → exporter → channel → Python harness reads.

**Files to modify in ANNIEQ:**
- `C:/work/ANNIEQ/scripts/` — add sys.path startup DAT

- [ ] **Step 1: Add sys.path in TD startup DAT**

In TD, create a Script DAT at `/project1/startup` with:
```python
import sys
if "C:/dev/cuda" not in sys.path:
    sys.path.insert(0, "C:/dev/cuda")
```

Cook it once (right-click → Cook).

- [ ] **Step 2: Verify import in TD**

In TD Python console:
```python
from cuda_ipc_transport.receiver import CUDAIPCReceiver
print("ok")
```

Expected: `ok`

- [ ] **Step 3: Create TDCUDAIPCExporter component in TD**

In TD Network at `/project1/sd_bridge/`:
1. Create a Base COMP named `cuda_exporter`
2. Add `TDCUDAIPCExporter` as extension:
   - Create textDAT inside, paste content of `cuda_ipc_transport/td/exporter.py`
   - Name it `TDCUDAIPCExporter`
   - In COMP Extensions → add `TDCUDAIPCExporter`
3. Add custom parameter `Channelname` (String) = `"td_test_out"`
4. Create Script DAT with `onFrameStart`:
```python
def onFrameStart(frame):
    ext.TDCUDAIPCExporter.ExportFrame(op('noise1'))
```
   where `noise1` is any test pattern TOP (Noise TOP, Pattern TOP, etc.)

- [ ] **Step 4: Run Python harness as receiver**

```bash
cd C:/dev/cuda
.venv/Scripts/python -c "
from cuda_ipc_transport.receiver import CUDAIPCReceiver
import time

r = CUDAIPCReceiver('td_test_out')
r.connect()
print('connected:', r.is_ready())
for i in range(100):
    ptr, size, shape = r.get_frame()
    if ptr is not None:
        print(f'frame {i}: ptr=0x{ptr:016x} size={size} shape={shape}')
        break
    time.sleep(0.01)
r.close()
"
```

Expected: prints at least one frame with non-zero ptr.

- [ ] **Step 5: Test Python → TD direction**

Run harness in terminal:
```bash
.venv/Scripts/python -m cuda_ipc_transport --source test --channel test_in --width 512 --height 512 --fps 30
```

In TD, create Script TOP at `/project1/sd_bridge/test_in_top`:
- Set script to content of `cuda_ipc_transport/td/importer.py`
- Add COMP parameter `Channelname` = `"test_in"`
- Connect any TOP as input (triggers cooking)

Expected: Script TOP shows color bars with frame counter.

- [ ] **Step 6: Measure performance**

Check `C:/tmp/sd_timing.txt` (importer logs every 60 frames):
- `get` should be < 5ms
- `copy` should be < 5ms

- [ ] **Step 7: Update ANNIEQ importer DAT to use new package**

Replace content of `scripts/sd_output_reader_ipc_callbacks.py` with import from package:

```python
# Redirect to package implementation
import sys
if "C:/dev/cuda" not in sys.path:
    sys.path.insert(0, "C:/dev/cuda")

from cuda_ipc_transport.td.importer import onCook, onSetupParameters, onPulse
```

Verify TD still shows SD output (if SD is running), or test_in signal.

- [ ] **Step 8: Commit both repos**

```bash
# cuda repo
cd C:/dev/cuda
git add -A
git commit -m "feat: Phase 1 complete — full bidirectional CUDA IPC transport"
git push origin master

# ANNIEQ
cd C:/work/ANNIEQ
git add scripts/sd_output_reader_ipc_callbacks.py
git commit -m "refactor: sd_output_reader → delegate to cuda_ipc_transport package"
```

---

## Self-Review Checklist

- [x] `protocol.py` — pack/unpack, `meta_offset` method in `SHMLayout` (referenced in receiver.py)
- [x] `shm_size(num_slots)` used in sender.py and exporter.py — consistent
- [x] `get_reader()` compat shim — tested in test_receiver_compat.py
- [x] `write_idx++` happens after `record_event` in both sender.py and exporter.py
- [x] `num_slots` read from header in receiver.py (not hardcoded)
- [x] `__main__.py` → `harness.main()` mapping implemented
- [x] `meta_offset` method: added to `SHMLayout` in Task 6 step 3 — also needs to be added in Task 3 step 3

**Fix:** In `protocol.py` (`SHMLayout`), add the `meta_offset` instance method directly:

```python
# Add to SHMLayout class in protocol.py:
def meta_offset(self) -> int:
    """Byte offset of shutdown flag."""
    return SHM_HEADER_SIZE + self.num_slots * SLOT_SIZE
```

This should be included in Task 3 (protocol.py implementation), not Task 6.
