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
