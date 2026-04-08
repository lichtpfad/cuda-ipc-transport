# IPC Telemetry — Execute DAT onFrameStart
# Reads SharedMemory headers for ml_ipc and ml_out_ipc every frame.
# Writes 6 channels to ipc_chop1 Constant CHOP.
# Uses print() not debug() (Execute DAT context).

import struct
import time

_shm_in = None
_shm_out = None
_prev_in_idx = 0
_prev_out_idx = 0
_prev_time = 0.0
_fps_in = 0.0
_fps_out = 0.0
_fps_window = 1.0  # seconds


def _open_shm(name):
    try:
        from multiprocessing.shared_memory import SharedMemory
        return SharedMemory(name=name, create=False)
    except Exception:
        return None


def onFrameStart(frame):
    global _shm_in, _shm_out, _prev_in_idx, _prev_out_idx, _prev_time, _fps_in, _fps_out

    chop = op('ipc_chop1')
    if chop is None:
        return

    # Open SharedMemory handles (cache them)
    if _shm_in is None:
        _shm_in = _open_shm('ml_ipc')
    if _shm_out is None:
        _shm_out = _open_shm('ml_out_ipc')

    connected = 0
    in_idx = 0
    out_idx = 0

    try:
        if _shm_in is not None:
            in_idx = struct.unpack_from('<I', _shm_in.buf, 16)[0]
            connected += 1
    except Exception:
        _shm_in = None

    try:
        if _shm_out is not None:
            out_idx = struct.unpack_from('<I', _shm_out.buf, 16)[0]
            connected += 1
    except Exception:
        _shm_out = None

    # Compute FPS over window
    now = time.perf_counter()
    dt = now - _prev_time
    if dt >= _fps_window and _prev_time > 0:
        _fps_in = (in_idx - _prev_in_idx) / dt
        _fps_out = (out_idx - _prev_out_idx) / dt
        _prev_in_idx = in_idx
        _prev_out_idx = out_idx
        _prev_time = now
    elif _prev_time == 0:
        _prev_in_idx = in_idx
        _prev_out_idx = out_idx
        _prev_time = now

    # Latency from importer cookTime
    latency = 0.0
    try:
        imp = op('import_top')
        if imp is not None:
            latency = imp.cookTime * 1000  # ms
    except Exception:
        pass

    # Write to Constant CHOP
    chop.par.value0 = in_idx
    chop.par.value1 = out_idx
    chop.par.value2 = round(_fps_in, 1)
    chop.par.value3 = round(_fps_out, 1)
    chop.par.value4 = 1 if connected == 2 else 0
    chop.par.value5 = round(latency, 2)


def onFrameEnd(frame):
    pass

def onPlayStateChange(state):
    pass

def onDeviceChange():
    pass

def onProjectPreSave():
    pass

def onProjectPostSave():
    pass
