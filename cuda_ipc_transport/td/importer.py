"""Script TOP callbacks for CUDA IPC receiver.

Displays GPU frames from a named CUDA IPC channel using zero-copy copyCUDAMemory().
Refactored from sd_output_reader_ipc_callbacks.py.

Deploy in TD:
    1. Create a Script DAT
    2. Load this module
    3. Add custom String parameter 'Channelname' = channel name (e.g. "sd_to_td_ipc")
    4. Call onCook(op()) each frame (or set DAT as Script TOP)
"""
from dataclasses import dataclass, field
import struct
import time
import numpy as np
from typing import Optional

# Try to import from package; fall back to mod() for backward compat
try:
    import sys as _sys
    _sys.path.insert(0, 'C:/dev/cuda')
    from cuda_ipc_transport.receiver import CUDAIPCReceiver as _CUDAIPCReceiver
    from cuda_ipc_transport.wrapper import get_cuda_runtime as _get_cuda_runtime
    _USE_PACKAGE = True
except ImportError:
    _USE_PACKAGE = False
    _CUDAIPCReceiver = None
    _get_cuda_runtime = None


@dataclass
class _State:
    """Per-operator state (keyed by scriptOp.id or operator name)."""
    reader: Optional[object] = None
    stream: Optional[object] = None
    mem_shape: Optional[object] = None
    cuda: Optional[object] = None
    frame_count: int = 0
    reconnect_cooldown: float = 0.0
    channel_name: str = ""


# Module-level state dict: op_id -> _State
_states: dict = {}


def _get_state(scriptOp) -> _State:
    """Get or create state for operator. Use operator id as key."""
    op_id = id(scriptOp)  # CPython id() is stable for object lifetime
    if op_id not in _states:
        _states[op_id] = _State()
    return _states[op_id]


def _get_channel_name(scriptOp) -> str:
    """Extract channel name from scriptOp.par.Channelname or return default."""
    try:
        return scriptOp.par.Channelname.eval()
    except (AttributeError, TypeError):
        return "sd_to_td_ipc"


def _get_reader(scriptOp):
    """Get or create reader for the operator's channel."""
    global _USE_PACKAGE
    state = _get_state(scriptOp)

    if state.reader is not None:
        return state.reader

    channel = _get_channel_name(scriptOp)
    state.channel_name = channel

    if _USE_PACKAGE and _CUDAIPCReceiver is not None:
        # Use package classes
        try:
            reader = _CUDAIPCReceiver(channel)
            reader.connect()
            state.reader = reader
            return state.reader
        except Exception as e:
            try:
                debug(f"[importer] Package reader failed: {e}, falling back to mod()")  # noqa: F821
            except NameError:
                pass
            _USE_PACKAGE = False

    # Fall back to TD mod() system
    try:
        reader_mod = mod('cuda_ipc_reader')  # noqa: F821
        state.reader = reader_mod.get_reader(channel)
        return state.reader
    except Exception:
        return None


def _get_cuda(scriptOp):
    """Get or create CUDA runtime wrapper."""
    state = _get_state(scriptOp)

    if state.cuda is not None:
        return state.cuda

    if _USE_PACKAGE and _get_cuda_runtime is not None:
        try:
            state.cuda = _get_cuda_runtime()
            return state.cuda
        except Exception:
            pass

    # Fall back to mod() system
    try:
        wrapper = mod('CUDAIPCWrapper')  # noqa: F821
        state.cuda = wrapper.get_cuda_runtime()
        return state.cuda
    except Exception:
        return None


def onCook(scriptOp):
    """Main frame loop. Call once per frame to receive and display GPU frames."""
    state = _get_state(scriptOp)

    reader = _get_reader(scriptOp)
    if reader is None:
        return

    if not reader.is_ready():
        now = time.time()
        if now > state.reconnect_cooldown:
            reader.reconnect()
            state.reconnect_cooldown = now + 2.0
        return

    # Create stream once (non-blocking, dedicated to IPC copy)
    if state.stream is None:
        try:
            cuda = _get_cuda(scriptOp)
            if cuda is not None:
                state.stream = cuda.create_stream(0x01)  # cudaStreamNonBlocking
                s = state.stream.value if hasattr(state.stream, 'value') else int(state.stream)
                try:
                    debug(f"[importer] Created CUDA stream handle: {s}")  # noqa: F821
                except NameError:
                    pass
        except Exception as e:
            try:
                debug(f"[importer] Stream creation failed: {e}")  # noqa: F821
            except NameError:
                pass

    t0 = time.perf_counter()
    ptr, size, shape = reader.get_frame()
    t1 = time.perf_counter()

    if ptr is None:
        return

    # Cache CUDAMemoryShape (avoid per-frame allocation)
    if state.mem_shape is None:
        try:
            state.mem_shape = CUDAMemoryShape()  # noqa: F821 -- TD built-in
            state.mem_shape.height = shape[0]
            state.mem_shape.width = shape[1]
            state.mem_shape.numComps = shape[2]
            state.mem_shape.dataType = np.uint8
        except NameError:
            # CUDAMemoryShape not available (running outside TD)
            return

    try:
        if state.stream is not None:
            # Use stream if available
            s = struct.unpack('<Q', bytes(state.stream))[0]  # bulletproof int from c_uint64
            scriptOp.copyCUDAMemory(ptr, size, state.mem_shape, stream=s)
        else:
            scriptOp.copyCUDAMemory(ptr, size, state.mem_shape)
    except Exception as e:
        try:
            debug(f"[importer] copyCUDAMemory error: {e}")  # noqa: F821
        except NameError:
            pass
        # Try without stream as fallback
        try:
            scriptOp.copyCUDAMemory(ptr, size, state.mem_shape)
        except Exception:
            pass

    t2 = time.perf_counter()
    state.frame_count += 1

    # Log every 30 frames (configurable)
    log_interval = 30
    if state.frame_count % log_interval == 0:
        get_ms = (t1 - t0) * 1000
        copy_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000
        has_stream = state.stream is not None
        s_val = struct.unpack('<Q', bytes(state.stream))[0] if state.stream else 0
        msg = (f"[importer] channel='{state.channel_name}' frame={state.frame_count} "
               f"get_frame={get_ms:.1f}ms copy={copy_ms:.1f}ms "
               f"total={total_ms:.1f}ms stream={has_stream}({s_val})")
        try:
            debug(msg)  # noqa: F821
        except NameError:
            print(msg)


def onSetupParameters(scriptOp):
    """TD callback for parameter setup (no-op)."""
    return


def onPulse(par):
    """TD callback for pulse parameters (no-op)."""
    return
