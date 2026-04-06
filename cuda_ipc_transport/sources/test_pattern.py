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
        # 8 color bars: white, yellow, cyan, green, magenta, red, blue, black
        self._colors = [
            (255, 255, 255), (255, 255, 0), (0, 255, 255),
            (0, 255, 0),     (255, 0, 255), (255, 0, 0),
            (0, 0, 255),     (0, 0, 0),
        ]

    def get_frame(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        bar_w = self.width // len(self._colors)
        for i, (b, g, r) in enumerate(self._colors):
            x0 = i * bar_w
            x1 = (i + 1) * bar_w
            img[:, x0:x1, 0] = b
            img[:, x0:x1, 1] = g
            img[:, x0:x1, 2] = r
            img[:, x0:x1, 3] = 255

        # Add changing pixel in top-left corner (counter encodes frame number)
        counter_color = (self._frame % 256)
        img[0, 0, 0] = counter_color  # Blue channel changes with frame

        # Add text overlay for larger images
        if self.height >= 100 and self.width >= 100:
            cv2.putText(img, f"frame {self._frame}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 255), 2)

        self._frame += 1
        return img
