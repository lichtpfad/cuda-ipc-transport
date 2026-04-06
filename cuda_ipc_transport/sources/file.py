from pathlib import Path
import cv2
import numpy as np
from .base import Source


class FileSource(Source):
    """Reads image or video file; loops automatically."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._cap = None
        self._single_frame = None
        if self._path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(self._path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 3:
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
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return (frame if frame is not None else np.zeros((512, 512, 4), dtype=np.uint8)).astype(np.uint8)

    def close(self):
        if self._cap:
            self._cap.release()
