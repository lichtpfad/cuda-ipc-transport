from abc import ABC, abstractmethod
import numpy as np


class Source(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Return (H, W, C) uint8 numpy array."""
        ...

    def close(self):
        pass
