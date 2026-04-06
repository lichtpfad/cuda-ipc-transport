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
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA).astype(np.uint8)

    def close(self):
        self._cap.release()
