# TD ML Bridge Asset -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three nested TD COMPs (cuda_ipc_bridge, sd_controller, ml_bridge) that package bidirectional CUDA IPC transport + subprocess management + OSC status into a drag-and-drop asset. Two TOP inputs (image + depth), one TOP output (processed result). One button Start.

**Architecture:** `ml_bridge` COMP (facade) contains `cuda_ipc_bridge` COMP (generic transport: 2 exporters + 1 importer + cooker + flip + OSC status) and `sd_controller` COMP (subprocess start/stop). Harness gains `--channel-prefix` (creates `{prefix}_img`, `{prefix}_depth`, `{prefix}_result` channels) and `--osc-status-port` (sends `/transport/connected` and `/transport/frame` via UDP). Setup script `td_setup.py` gains `--mode comp` to create the nested COMP hierarchy via h2t batch commands.

**Tech Stack:** Python 3.11, ctypes (cudart64_12/13.dll), numpy, opencv-python, python-osc (optional), multiprocessing.shared_memory, h2t CLI for TD automation.

**Spec:** `C:/dev/cuda/docs/specs/2026-04-07-td-ml-bridge-asset.md`

**Critical TD Lessons (MUST follow in every step):**
1. Script TOP callbacks MUST be defined directly in the DAT, never imported from a package.
2. Script TOP does NOT cook automatically -- needs companion Execute DAT with `cook(force=True)`.
3. `copyCUDAMemory` produces flipped image -- needs Flip TOP with `flipy=1`.
4. `json.dumps()` for multi-line strings in h2t batch code.
5. `par.outputresolution = 'custom'` -- string, not integer.
6. h2t batch code must end with `_result = 'done'` (not `op.attr = val`).

---

## File Map

```
C:/dev/cuda/
  cuda_ipc_transport/
    __init__.py          (unchanged)
    __main__.py          (unchanged)
    harness.py           MODIFIED: +--channel-prefix, +--osc-status-port, OSC sender
    sender.py            (unchanged)
    receiver.py          (unchanged)
    channel.py           (unchanged)
    protocol.py          (unchanged)
    wrapper.py           (unchanged)
    td/
      __init__.py        (unchanged)
      exporter.py        (unchanged)
      importer.py        (unchanged)
    sources/             (unchanged)
  scripts/
    td_setup.py          MODIFIED: +--mode comp|flat, COMP builder functions
    test_smoke.py        NEW: end-to-end smoke test script
  tests/
    test_harness_args.py NEW: --channel-prefix arg parsing tests
    test_osc_status.py   NEW: OSC status sender tests (mock UDP)
    (existing tests unchanged)
  pyproject.toml         MODIFIED: +optional-dependencies osc
```

---
## Task 1: `--channel-prefix` arg in harness.py

**Files:**
- Modify: `C:/dev/cuda/cuda_ipc_transport/harness.py`
- Create: `C:/dev/cuda/tests/test_harness_args.py`

### Step 1.1: Extract `_resolve_channel` and add `--channel-prefix` argument

- [ ] **Step 1.1a:** Add helper function to `harness.py` (before `main`):

```python
def _resolve_channel(channel: str, channel_prefix: str = None) -> str:
    """Resolve effective channel name from args.

    --channel-prefix takes priority: creates {prefix}_result.
    --channel is backward-compatible literal name.
    """
    if channel_prefix:
        return f"{channel_prefix}_result"
    return channel
```

- [ ] **Step 1.1b:** Add `--channel-prefix` to argparse in `main()`, after existing `--channel`:

```python
    parser.add_argument("--channel-prefix", default=None,
                        help="Channel prefix. Creates {prefix}_result as send channel. "
                             "Overrides --channel when set.")
```

- [ ] **Step 1.1c:** After `args = parser.parse_args(argv)`, resolve effective channel:

```python
    if args.channel_prefix and args.channel != "cuda_ipc_test":
        print(f"[harness] WARNING: --channel-prefix overrides --channel", file=sys.stderr)
    effective_channel = _resolve_channel(args.channel, args.channel_prefix)
```

- [ ] **Step 1.1d:** Replace `args.channel` with `effective_channel` in two places:
  - `CUDAIPCChannel(effective_channel, args.width, args.height)` (was `args.channel`)
  - Print statement: use `effective_channel` in the f-string (was `args.channel`)

### Step 1.2: Write tests for arg parsing

- [ ] Create `C:/dev/cuda/tests/test_harness_args.py`:

```python
"""Tests for harness CLI argument parsing -- no CUDA required."""
from cuda_ipc_transport.harness import _resolve_channel


class TestResolveChannel:
    """Test _resolve_channel helper."""

    def test_default_no_prefix(self):
        assert _resolve_channel("cuda_ipc_test", None) == "cuda_ipc_test"

    def test_prefix_creates_result_channel(self):
        assert _resolve_channel("cuda_ipc_test", "ml") == "ml_result"

    def test_prefix_overrides_channel(self):
        assert _resolve_channel("custom_name", "ml") == "ml_result"

    def test_custom_prefix(self):
        assert _resolve_channel("x", "sd_v2") == "sd_v2_result"

    def test_empty_string_prefix_is_falsy(self):
        assert _resolve_channel("my_chan", "") == "my_chan"
```

### Step 1.3: Verify

```bash
cd C:/dev/cuda
.venv/Scripts/python -m pytest tests/test_harness_args.py -v
```

Expected: 5 tests pass.

```bash
.venv/Scripts/python -m cuda_ipc_transport --help
```

Expected: `--channel-prefix` appears in help output.

**Commit:** `feat(harness): add --channel-prefix argument for multi-instance channel naming`

---

## Task 2: OSC status in harness.py

**Files:**
- Modify: `C:/dev/cuda/pyproject.toml`
- Modify: `C:/dev/cuda/cuda_ipc_transport/harness.py`
- Create: `C:/dev/cuda/tests/test_osc_status.py`

### Step 2.1: Add python-osc optional dependency

- [ ] In `C:/dev/cuda/pyproject.toml`, add after `dependencies` line:

```toml
[project.optional-dependencies]
osc = ["python-osc"]
```

Full file becomes:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cuda_ipc_transport"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy", "opencv-python"]

[project.optional-dependencies]
osc = ["python-osc"]

[tool.setuptools.packages.find]
where = ["."]
include = ["cuda_ipc_transport*"]

[tool.pytest.ini_options]
addopts = "-m 'not cuda' --ignore=tests/test_integration.py"
markers = [
    "cuda: mark test as requiring CUDA GPU (skip if not available)",
]
```

### Step 2.2: Add `_OSCStatus` class and `--osc-status-port` argument

- [ ] **Step 2.2a:** Add `_OSCStatus` class to `harness.py` (after imports, before `_resolve_channel`):

```python
class _OSCStatus:
    """Lightweight OSC status sender. No-ops gracefully if python-osc not installed."""

    def __init__(self, port: int):
        self._client = None
        if port <= 0:
            return
        try:
            from pythonosc.udp_client import SimpleUDPClient
            self._client = SimpleUDPClient("127.0.0.1", port)
            print(f"[harness] OSC status -> 127.0.0.1:{port}")
        except ImportError:
            print("[harness] WARNING: python-osc not installed. "
                  "Install with: pip install cuda_ipc_transport[osc]",
                  file=sys.stderr)

    def connected(self, value: int = 1):
        if self._client:
            self._client.send_message("/transport/connected", value)

    def frame(self, n: int):
        if self._client:
            self._client.send_message("/transport/frame", n)

    def close(self):
        if self._client:
            try:
                self._client.send_message("/transport/connected", 0)
            except Exception:
                pass
```

- [ ] **Step 2.2b:** Add `--osc-status-port` argument in `main()`:

```python
    parser.add_argument("--osc-status-port", type=int, default=0,
                        help="UDP port for OSC transport status (0 = disabled)")
```

- [ ] **Step 2.2c:** Initialize OSC after sender init, send connected, integrate into loop.

After `print(f"[harness] ... | Press Ctrl+C to stop")`:

```python
    osc = _OSCStatus(args.osc_status_port)
    osc.connected(1)
```

Inside the frame loop, after `sent += 1`:

```python
            osc.frame(sent)
```

In the `finally` block, before `sender.close()`:

```python
        osc.close()
```

### Step 2.3: Write OSC tests with mock UDP

- [ ] Create `C:/dev/cuda/tests/test_osc_status.py`:

```python
"""Tests for OSC status sender in harness -- no CUDA required."""
import socket
import threading
import time
import pytest
from cuda_ipc_transport.harness import _OSCStatus


def _has_pythonosc():
    try:
        import pythonosc
        return True
    except ImportError:
        return False


class TestOSCStatusDisabled:
    """Test _OSCStatus when disabled or unavailable."""

    def test_disabled_when_port_zero(self):
        osc = _OSCStatus(0)
        assert osc._client is None
        osc.connected(1)
        osc.frame(42)
        osc.close()

    def test_disabled_when_port_negative(self):
        osc = _OSCStatus(-1)
        assert osc._client is None

    def test_graceful_without_pythonosc(self, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if "pythonosc" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        osc = _OSCStatus(7099)
        assert osc._client is None


@pytest.mark.skipif(not _has_pythonosc(), reason="python-osc not installed")
class TestOSCStatusSends:

    def test_sends_connected_and_frame(self):
        port = 17199
        received = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        sock.settimeout(2.0)

        def listen():
            try:
                while True:
                    data, _ = sock.recvfrom(1024)
                    received.append(data)
            except socket.timeout:
                pass

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        osc = _OSCStatus(port)
        osc.connected(1)
        osc.frame(10)
        osc.close()
        time.sleep(0.5)
        sock.close()
        t.join(timeout=3)
        # At least 3 messages: connected(1), frame(10), connected(0)
        assert len(received) >= 3
```

### Step 2.4: Verify

```bash
cd C:/dev/cuda
.venv/Scripts/pip install -e ".[osc]"
.venv/Scripts/python -m pytest tests/test_osc_status.py -v
```

Expected: 3-4 tests pass.

**Commit:** `feat(harness): add OSC transport status (connected, frame) via --osc-status-port`

---

## Task 3: td_setup.py `--mode comp` -- cuda_ipc_bridge COMP

**Files:**
- Modify: `C:/dev/cuda/scripts/td_setup.py`

### Step 3.1: Add `--mode` argument, constants, and routing

- [ ] **Step 3.1a:** Add constants at module level:

```python
# COMP mode names
COMP_BRIDGE = "cuda_ipc_bridge"
COMP_CONTROLLER = "sd_controller"
COMP_OUTER = "ml_bridge"

DEFAULT_PREFIX = "ml"
DEFAULT_OSC_PORT = 7001
```

- [ ] **Step 3.1b:** Add `--mode` argument to argparse:

```python
    parser.add_argument("--mode", choices=["comp", "flat"], default="comp",
                        help="comp = nested COMPs (new), flat = current behavior (default: comp)")
```

- [ ] **Step 3.1c:** Route in main() -- add elif branch:

```python
    if args.teardown:
        if args.mode == "comp":
            batch = build_comp_teardown_batch()
        else:
            batch = build_teardown_batch()
    elif args.mode == "flat":
        batch = build_setup_batch(...)  # existing code
    elif args.mode == "comp":
        batch = build_comp_batch(args)
```

### Step 3.2: Write `make_bridge_exporter_code` function

- [ ] Add function generating Execute DAT code for an exporter inside the bridge COMP. The exporter reads `parent().par.Channelprefix` dynamically, constructs the channel name, and calls `TDCUDAIPCExporter.ExportFrame()`.

```python
def make_bridge_exporter_code(bridge_path: str, input_name: str,
                               channel_suffix: str, pkg_path: str) -> str:
    """Execute DAT code for exporter inside bridge COMP.

    Args:
        bridge_path: absolute TD path to bridge COMP
        input_name: inTOP name (e.g. 'in_image')
        channel_suffix: suffix after prefix (e.g. 'img' -> {prefix}_img)
        pkg_path: path to cuda_ipc_transport package
    """
    return f"""# Execute DAT -- Exporter: {input_name} -> {{prefix}}_{channel_suffix}
import sys
if '{pkg_path}' not in sys.path:
    sys.path.insert(0, '{pkg_path}')

_exporter = None
_last_channel = None


def onFrameStart(frame):
    global _exporter, _last_channel
    try:
        prefix = parent().par.Channelprefix.eval()
        channel = prefix + '_{channel_suffix}'

        if _exporter is None or channel != _last_channel:
            from cuda_ipc_transport.td.exporter import TDCUDAIPCExporter
            if _exporter is not None:
                _exporter.Cleanup()

            class _FC:
                name = 'exporter_{input_name}'
                class par:
                    class Channelname:
                        _ch = channel
                        @staticmethod
                        def eval():
                            return _FC.par.Channelname._ch

            _exporter = TDCUDAIPCExporter(_FC())
            _last_channel = channel
            debug('[bridge] exporter init: ' + channel)

        top = op('{bridge_path}/{input_name}')
        if top and top.inputs:
            _exporter.ExportFrame(top)
    except Exception as e:
        debug('[bridge] export error: {{}}'.format(e))


def onFrameEnd(frame): pass
def onPlayStateChange(state): pass
def onDeviceChange(): pass
def onProjectPreSave():
    global _exporter
    if _exporter is not None:
        _exporter.Cleanup()
        _exporter = None
def onProjectPostSave(): pass
"""
```

### Step 3.3: Write `make_bridge_importer_callbacks` function

- [ ] Add function generating Script TOP callback code. Reads channel dynamically from bridge COMP Channelprefix parameter. Follows all TD lessons (inline definition, CUDAMemoryShape built-in, reconnect logic).

The function signature:

```python
def make_bridge_importer_callbacks(bridge_path: str, pkg_path: str) -> str:
```

Key differences from existing `make_importer_callbacks()`:
- Reads `op(bridge_path).par.Channelprefix.eval()` dynamically instead of hardcoded channel
- Constructs channel as `f"{prefix}_result"`
- Re-creates reader if channel name changes (supports live prefix changes)
- Resets `_mem_shape` on channel change
- Same copyCUDAMemory pattern with stream, same reconnect cooldown
- Same `CUDAMemoryShape()` built-in usage (TD lesson 1)

### Step 3.4: Write `build_bridge_batch` function

- [ ] Creates the full list of h2t batch commands for the bridge COMP:

```python
def build_bridge_batch(parent: str, pkg_path: str, prefix: str,
                        osc_port: int, width: int, height: int) -> list:
```

Batch command sequence (each is a `{"code": ...}` dict):

| Step | Operation | Node Created |
|------|-----------|--------------|
| A | Create containerCOMP + custom params (Pkgpath, Channelprefix, Statusoscport) | `cuda_ipc_bridge` |
| B | Create 2x inTOP | `in_image`, `in_depth` |
| C | Create export image Execute DAT (code from `make_bridge_exporter_code`) | `export_img_exec` |
| D | Create export depth Execute DAT | `export_depth_exec` |
| E | Create import callbacks Text DAT (code from `make_bridge_importer_callbacks`) | `import_callbacks` |
| F | Create Script TOP (importer), `par.outputresolution = 'custom'`, `par.callbacks = import_callbacks` | `import_top` |
| G | Create cooker Execute DAT (`cook(force=True)` on `import_top` each frame) | `import_cooker` |
| H | Create Flip TOP (`flipy=1`), input from `import_top` | `import_flip` |
| I | Create OSC In CHOP, port from `Statusoscport` param | `osc_status_in` |
| J | Create Out TOP, input from `import_flip` | `out_result` |

**Critical patterns for each batch command:**
- All multi-line code uses `json.dumps(code)` for proper escaping
- Execute DATs: `n.par.framestart = 1; n.par.active = 1`
- Script TOP: `par.outputresolution = 'custom'` (string, not int!)
- Every command ends with `_result = n.path` or `_result = 'done'`

### Step 3.5: Verify

```bash
cd C:/dev/cuda
python scripts/td_setup.py --mode comp --td-port 9955
```

Expected: `[OK] N commands, Xms`. In TD: `/project1/ml_bridge/cuda_ipc_bridge` COMP with all internal nodes.

**Commit:** `feat(td_setup): add --mode comp with cuda_ipc_bridge COMP (exporters, importer, cooker, flip, OSC)`

---

## Task 4: td_setup.py `--mode comp` -- sd_controller COMP

**Files:**
- Modify: `C:/dev/cuda/scripts/td_setup.py`

### Step 4.1: Write `make_controller_extension` function

- [ ] Add function generating the Extension text DAT code for subprocess management.

Signature: `make_controller_extension(bridge_path: str) -> str`

The generated `SDControllerExt` class must have:

- `__init__(self, ownerComp)`: stores ownerComp ref, `self._process = None`
- `Start(self)`: reads params, builds command, launches `subprocess.Popen`
  - Reads `self.ownerComp.par.Venvpath`, `.Module`, `.Moduleargs`
  - Gets prefix and osc_port from bridge: `op(bridge_path).par.Channelprefix/Statusoscport`
  - If Venvpath empty, uses system `python`; otherwise `{venv}/Scripts/python`
  - Command: `{python} -m {module} --channel-prefix {prefix} --osc-status-port {osc_port} {args}`
  - Uses `shell=True`, `CREATE_NEW_PROCESS_GROUP` creationflags
  - Guards against double-start: checks `self._process.poll() is None`
- `Stop(self)`: `terminate()` then `wait(timeout=5)` then `kill()` if needed
- `IsRunning(self)`: returns bool
- `Destroy(self)`: calls `Stop()`, for cleanup on project close

### Step 4.2: Write `make_controller_pulse_code` function

- [ ] Generate Par Execute DAT code for Start/Stop pulse handling.

Signature: `make_controller_pulse_code() -> str`

The generated code defines `onValueChange(par, prev)` that calls:
- `par.owner.ext.SDControllerExt.Start()` when par.name is Start
- `par.owner.ext.SDControllerExt.Stop()` when par.name is Stop

### Step 4.3: Write `build_controller_batch` function

- [ ] Build h2t batch commands for sd_controller COMP:

Signature: `build_controller_batch(parent: str, bridge_path: str) -> list`

Batch command sequence:

| Step | Operation | Node Created |
|------|-----------|--------------|
| A | Create containerCOMP + custom params (Venvpath, Module, Moduleargs, Start pulse, Stop pulse) | `sd_controller` |
| B | Create Extension text DAT with `SDControllerExt` class | `process_mgr` |
| C | Wire extension: `op(controller).par.extension1 = op(controller/process_mgr)` | -- |
| D | Create Par Execute DAT watching Start/Stop pulses | `pulse_handler` |

Default param values:
- `Module`: `cuda_ipc_transport`
- `Moduleargs`: `--source test --width 512 --height 512`

### Step 4.4: Verify

After running setup, press Start pulse on sd_controller in TD. Check TD textport for launch and PID messages. Press Stop to verify clean termination.

**Commit:** `feat(td_setup): add sd_controller COMP with subprocess start/stop`

---

## Task 5: td_setup.py `--mode comp` -- ml_bridge outer COMP

**Files:**
- Modify: `C:/dev/cuda/scripts/td_setup.py`

### Step 5.1: Write `make_facade_pulse_code` function

- [ ] Generate Par Execute DAT code that forwards Start/Stop from facade to sd_controller.

Signature: `make_facade_pulse_code() -> str`

The generated code defines `onValueChange(par, prev)` that pulses the corresponding param on `op("sd_controller")`.

### Step 5.2: Write `build_comp_batch` function (orchestrator)

- [ ] Create the top-level function that `main()` calls for `--mode comp`:

```python
def build_comp_batch(args) -> list:
    outer = f"{PARENT}/{COMP_OUTER}"
    bridge_path = f"{outer}/{COMP_BRIDGE}"
```

Batch command sequence:

| # | Operation | Notes |
|---|-----------|-------|
| 1 | Create ml_bridge containerCOMP + facade params (Channelprefix, Venvpath, Start, Stop) | Outer shell |
| 2 | `build_bridge_batch(parent=outer, ...)` | All bridge nodes inside ml_bridge |
| 3 | `build_controller_batch(parent=outer, bridge_path=...)` | All controller nodes inside ml_bridge |
| 4 | Bind facade Channelprefix to bridge Channelprefix via expr `parent().par.Channelprefix` | Param binding |
| 5 | Bind facade Venvpath to controller Venvpath via expr `parent().par.Venvpath` | Param binding |
| 6 | Create facade pulse forwarder Par Execute DAT | `facade_pulse` |
| 7 | Create 2x inTOP at ml_bridge level | `in_image`, `in_depth` |
| 8 | Wire ml_bridge inTOPs to bridge inTOPs | Input proxying |
| 9 | Create outTOP from bridge out_result | `out_result` |

### Step 5.3: Write `build_comp_teardown_batch`

- [ ] Add teardown for COMP mode (destroying outer COMP removes all children automatically).

### Step 5.4: Update main() print statements for comp mode

- [ ] Add informative output showing the COMP tree being created.

### Step 5.5: Verify

```bash
cd C:/dev/cuda
python scripts/td_setup.py --mode comp --td-port 9955
```

Expected: `/project1/ml_bridge` with facade params. Pressing Start launches harness. Pressing Stop terminates it.

```bash
python scripts/td_setup.py --mode comp --teardown --td-port 9955
```

Expected: ml_bridge removed. Re-run setup is idempotent.

**Commit:** `feat(td_setup): add ml_bridge facade COMP with param binding and input/output proxying`

---

## Task 6: End-to-end smoke test

**Files:**
- Create: `C:/dev/cuda/scripts/test_smoke.py`

### Step 6.1: Write smoke test script

- [ ] Create `C:/dev/cuda/scripts/test_smoke.py`:

The script does:
1. Start harness subprocess with `--channel-prefix smoketest --osc-status-port 17201 --source test --width 64 --height 64 --fps 10 --frames 30`
2. Listen on UDP port for `/transport/connected` (2s timeout)
3. Wait for harness to finish (15s timeout)
4. Verify: OSC connected received, harness exited cleanly, 30 frames sent

Key functions:
- `wait_for_osc_message(port, address, timeout)` -- listens on UDP, returns raw bytes or None
- `main()` -- argparse with `--td-port` (0=skip TD), `--osc-port` (default 17201), `--prefix` (default smoketest)

Usage:
```bash
python scripts/test_smoke.py                          # harness + OSC only
python scripts/test_smoke.py --td-port 9955           # with TD setup first
python scripts/test_smoke.py --prefix custom_test     # custom prefix
```

### Step 6.2: Verify

```bash
cd C:/dev/cuda
.venv/Scripts/python scripts/test_smoke.py --osc-port 17201 --prefix smoketest
```

Expected (with CUDA): all checks pass (OSC connected, 30 frames, clean exit).
Expected (without CUDA): harness exits with code 1 (sender init fails). Acceptable for CI.

**Commit:** `test: add end-to-end smoke test script for ML bridge pipeline`

---

## Execution Notes

1. **Task ordering:** Tasks 1-2 modify `harness.py` (independent, can run in parallel). Tasks 3-5 modify `td_setup.py` (sequential: bridge -> controller -> outer). Task 6 depends on Tasks 1-2.

2. **Testing without TD:** Tasks 1, 2, 6 are testable without TouchDesigner. Tasks 3-5 require a running TD instance with h2t connected.

3. **h2t batch caveats:**
   - Every batch command must end with `_result = ...` (not an assignment to `op.par`)
   - Multi-line code strings must use `json.dumps()` to escape properly
   - TD operator type names: `containerCOMP`, `inTOP`, `outTOP`, `flipTOP`, `scriptTOP`, `executeDAT`, `textDAT`, `oscinCHOP`, `parameterexecuteDAT`

4. **Scope limits (Phase 1 only):** No SD-specific params, no OSC Out control, no OSC model status In, no Par Execute for live param changes, no custom UI panel, no server mode.
