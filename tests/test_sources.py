import numpy as np
from cuda_ipc_transport.sources.test_pattern import TestPatternSource


def test_test_pattern_shape_rgba():
    src = TestPatternSource(width=512, height=512)
    frame = src.get_frame()
    assert frame.shape == (512, 512, 4)
    assert frame.dtype == np.uint8


def test_test_pattern_shape_small():
    src = TestPatternSource(width=64, height=64)
    frame = src.get_frame()
    assert frame.shape == (64, 64, 4)


def test_test_pattern_increments():
    src = TestPatternSource(width=64, height=64)
    f1 = src.get_frame()
    f2 = src.get_frame()
    # frames differ (counter increments)
    assert not np.array_equal(f1, f2)


def test_test_pattern_close():
    src = TestPatternSource(width=64, height=64)
    src.close()  # must not raise


def test_test_pattern_not_black():
    src = TestPatternSource(width=512, height=512)
    frame = src.get_frame()
    assert frame.max() > 0  # color bars must not be all-zero
