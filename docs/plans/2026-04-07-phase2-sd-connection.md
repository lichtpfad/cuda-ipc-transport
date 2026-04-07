# Phase 2: Connect StreamDiffusion to CUDA IPC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect StreamDiffusion to the CUDA IPC transport so that TD sends video frames to SD via GPU-direct IPC and receives processed frames back — zero CPU roundtrip.

**Architecture:** Bridge channel naming is updated to match SD's convention (`{prefix}_ipc`, `{prefix}-cn_ipc`, `{prefix}_out_ipc`). sd_controller generates a temporary `stream_config.json` with the correct `input_mem_name` and `output_mem_name`, then launches SD. Bridge exporters write to channels SD reads; bridge importer reads from channel SD writes.

**Tech Stack:** Python 3.11, cuda_ipc_transport package, StreamDiffusion (Alex's fork), fire CLI, stream_config.json, h2t TD daemon.

**Specs:**
- `C:/work/ANNIEQ/docs/superpowers/specs/2026-04-05-cuda-ipc-transport-design.md` (Phase 2 section)
- `C:/dev/cuda/docs/specs/2026-04-07-td-ml-bridge-asset.md`

---

## Channel Naming Convention (SD-compatible)

| Channel | SharedMemory name | SD config field |
|---------|------------------|-----------------|
| Image (TD→SD) | `{prefix}_ipc` | `input_mem_name = "{prefix}"` |
| Depth/CN (TD→SD) | `{prefix}-cn_ipc` | auto: `{input_mem_name}-cn_ipc` |
| Result (SD→TD) | `{prefix}_out_ipc` | `output_mem_name = "{prefix}_out"` |

Example with prefix `"ml"`:
- Bridge exports to `ml_ipc` and `ml-cn_ipc`
- SD reads from `ml_ipc` and `ml-cn_ipc` (auto-constructed)
- SD writes to `ml_out_ipc`
- Bridge imports from `ml_out_ipc`

---

## File Map

```
C:/dev/cuda/
  cuda_ipc_transport/
    harness.py              MODIFIED: _resolve_channel uses _ipc suffix
  scripts/
    td_setup.py             MODIFIED: channel suffixes, sd_controller config gen + SD launch
  tests/
    test_harness_args.py    MODIFIED: updated test expectations
  td/                       NEW DIR: generated SD configs go here
```

SD code is NOT modified. Only our package adapts.

---

## Task 0: Audit live TD scene + fix existing bugs

**Source of truth is the live TD project** (`setup.3.toe`). Before changing anything, audit what's actually running.

### Step 0.1: Snapshot live scene state

- [ ] Via h2t: dump all DAT contents, params, connections from live `ml_bridge` COMP. Record as reference.

### Step 0.2: Fix disconnect→close bug in td_setup.py

- [ ] In `make_bridge_importer_callbacks()`: find `_reader.disconnect()` and change to `_reader.close()`. The `CUDAIPCReceiver` class has `close()` (receiver.py:152), not `disconnect()`.

### Step 0.3: Add preflight checks to sd_controller start()

- [ ] In `make_controller_process_mgr()`: add path validation before launching subprocess:

```python
    if not _os.path.isdir(sd_dir):
        debug('[sd_controller] ERROR: SD dir not found: ' + sd_dir)
        return
    if venv and not _os.path.isfile(python):
        debug('[sd_controller] ERROR: Python not found: ' + python)
        return
    if not _os.path.isfile(base_config_path):
        debug('[sd_controller] ERROR: Config not found: ' + base_config_path)
        return
```

### Step 0.4: Verify and commit

```bash
cd C:/dev/cuda
.venv/Scripts/python -m pytest -q
```

**Commit:** `fix: disconnect→close in importer, add preflight checks to sd_controller`

---

## Task 1: Update channel naming to match SD convention

**Files:**
- Modify: `C:/dev/cuda/cuda_ipc_transport/harness.py`
- Modify: `C:/dev/cuda/tests/test_harness_args.py`
- Modify: `C:/dev/cuda/scripts/td_setup.py`

### Step 1.1: Update `_resolve_channel` in harness.py

- [ ] Harness is a **test tool simulating SD output**. It writes to the importer channel (`{prefix}_out_ipc`), NOT the exporter channel (`{prefix}_ipc` which is TD→SD, would conflict):

```python
def _resolve_channel(channel: str, channel_prefix: str = None) -> str:
    if channel_prefix:
        return f"{channel_prefix}_out_ipc"
    return channel
```

Was: `f"{channel_prefix}_result"` → Now: `f"{channel_prefix}_out_ipc"`

This means "Start Harness" → writes to `{prefix}_out_ipc` → bridge importer reads it → color bars in import_flip. No conflict with TD→SD exporter channel.

### Step 1.2: Update tests

- [ ] In `tests/test_harness_args.py`, update expectations:

```python
def test_prefix_creates_result_channel(self):
    assert _resolve_channel("cuda_ipc_test", "ml") == "ml_out_ipc"

def test_prefix_overrides_channel(self):
    assert _resolve_channel("custom_name", "ml") == "ml_out_ipc"

def test_custom_prefix(self):
    assert _resolve_channel("x", "sd_v2") == "sd_v2_out_ipc"
```

### Step 1.3: Update bridge exporter channel suffixes in td_setup.py

- [ ] In `make_bridge_exporter_code()`: the image exporter uses suffix `_ipc`, the depth exporter uses `-cn_ipc`.

Find `channel_suffix` usage in the generated code:
```python
channel = prefix + '_{channel_suffix}'
```

Change to pass the full suffix including separator:
- Image call: `channel_suffix='_ipc'`
- Depth call: `channel_suffix='-cn_ipc'`

And update the generated code line to:
```python
channel = prefix + '{channel_suffix}'
```
(remove the `_` before `{channel_suffix}` since suffix now includes separator)

### Step 1.4: Update bridge importer channel suffix

- [ ] In `make_bridge_importer_callbacks()`: change channel construction from `prefix + '_result'` to `prefix + '_out_ipc'`.

### Step 1.5: Update build_bridge_batch() calls

- [ ] Update the two calls to `make_bridge_exporter_code`:
```python
exporter_img_code = make_bridge_exporter_code(bridge_path, 'in_image', '_ipc', pkg_path)
exporter_depth_code = make_bridge_exporter_code(bridge_path, 'in_depth', '-cn_ipc', pkg_path)
```

### Step 1.6: Verify

```bash
cd C:/dev/cuda
.venv/Scripts/python -m pytest tests/test_harness_args.py -v
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -c "import ast; ast.parse(open('scripts/td_setup.py', encoding='utf-8').read()); print('ok')"
```

Expected: all tests pass, syntax ok.

**Commit:** `refactor: align channel naming with SD convention (_ipc, -cn_ipc, _out_ipc)`

---

## Task 2: sd_controller generates stream_config.json and launches SD

**Files:**
- Modify: `C:/dev/cuda/scripts/td_setup.py` — update `make_controller_process_mgr()`

### Step 2.1: Update sd_controller params

- [ ] In `build_controller_batch()`, replace Module/Moduleargs params with SD-specific:

```python
"page.appendStr('Sddir', label='SD Directory')\n"
"n.par.Sddir.val = 'C:/work/stream_alex/StreamDiffusionTD-Custom/StreamDiffusion/StreamDiffusionTD'\n"
"page.appendStr('Sdconfig', label='Base Config')\n"
"n.par.Sdconfig.val = 'stream_config.json'\n"
```

Keep existing: Venvpath, Start (pulse), Stop (pulse).
Remove or keep Module/Moduleargs as legacy (unused by SD start).

### Step 2.2: Rewrite `make_controller_process_mgr()` for SD launch

- [ ] The `start()` function now:

1. Reads `ctrl_op.par.Venvpath`, `ctrl_op.par.Sddir`, `ctrl_op.par.Sdconfig`
2. Reads bridge prefix: `op('{bridge_path}').par.Channelprefix.eval()`
3. Loads base config from `{Sddir}/{Sdconfig}`
4. Overrides `input_mem_name = prefix` and `output_mem_name = f"{prefix}_out"`
5. Saves to `C:/dev/cuda/td/sd_config_{prefix}.json`
6. Launches: `{venv}/Scripts/python {Sddir}/main_sdtd.py --config {config_path}`

```python
def start(ctrl_op):
    global _process
    if _process is not None and _process.poll() is None:
        debug('[sd_controller] already running PID={}'.format(_process.pid))
        return

    import json as _json
    import os as _os

    venv = ctrl_op.par.Venvpath.eval()
    sd_dir = ctrl_op.par.Sddir.eval()
    sd_config = ctrl_op.par.Sdconfig.eval()

    bridge = op('{bridge_path}')
    prefix = bridge.par.Channelprefix.eval() if bridge else 'ml'

    # Load base config
    base_config_path = _os.path.join(sd_dir, sd_config)
    with open(base_config_path, 'r') as f:
        config = _json.load(f)

    # Override channel names to match bridge
    config['input_mem_name'] = prefix
    config['output_mem_name'] = prefix + '_out'

    # Save generated config
    out_dir = 'C:/dev/cuda/td'
    _os.makedirs(out_dir, exist_ok=True)
    gen_config_path = _os.path.join(out_dir, 'sd_config_' + prefix + '.json')
    with open(gen_config_path, 'w') as f:
        _json.dump(config, f, indent=2)

    if venv:
        python = venv.replace(chr(92), '/') + '/Scripts/python'
    else:
        python = 'python'

    main_script = _os.path.join(sd_dir, 'main_sdtd.py')
    cmd = [python, main_script, '--config', gen_config_path]

    try:
        _process = subprocess.Popen(cmd, shell=False, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        debug('[sd_controller] SD started PID={}'.format(_process.pid))
    except Exception as e:
        debug('[sd_controller] SD start failed: {}'.format(e))
```

### Step 2.3: Create `td/` directory

- [ ] Create `C:/dev/cuda/td/.gitkeep` so the directory exists in git.

### Step 2.4: Verify syntax

```bash
cd C:/dev/cuda
.venv/Scripts/python -c "import ast; ast.parse(open('scripts/td_setup.py', encoding='utf-8').read()); print('ok')"
.venv/Scripts/python -m pytest -q
```

**Commit:** `feat(sd_controller): generate stream_config.json and launch SD with correct channel names`

---

## Task 3: Update live TD scene via h2t daemon

**No file changes** — this task updates the running TD scene to match the new code.

### Step 3.1: Update bridge exporters channel naming

- [ ] Via h2t td batch: rewrite `export_img_exec` and `export_depth_exec` DAT text in the live scene to use new suffixes (`_ipc` and `-cn_ipc`).

### Step 3.2: Update bridge importer channel naming

- [ ] Via h2t td batch: rewrite `import_callbacks` DAT text to use `_out_ipc` suffix.

### Step 3.3: Update sd_controller process_mgr

- [ ] Via h2t td batch: rewrite `process_mgr` DAT text with the new SD launch logic.

### Step 3.4: Add Sddir and Sdconfig params to sd_controller

- [ ] Via h2t td batch: append custom params, set defaults:
  - `Sddir = "C:/work/stream_alex/StreamDiffusionTD-Custom/StreamDiffusion/StreamDiffusionTD"`
  - `Sdconfig = "stream_config.json"`
  - `Venvpath = "C:/work/stream_alex/StreamDiffusionTD-Custom/StreamDiffusion/venv"`

### Step 3.5: Verify bridge harness still works

- [ ] Press Start Harness on bridge → color bars appear in import_flip
- [ ] Stop harness

---

## Task 4: End-to-end test — SD via CUDA IPC

### Step 4.1: Connect video source

- [ ] Connect `moviefilein1` (or any video TOP) to `ml_bridge` in_image input.

### Step 4.2: Start bridge exporters

- [ ] Verify bridge is exporting frames to `{prefix}_ipc` SharedMemory.

### Step 4.3: Start SD via sd_controller

- [ ] Press Start on sd_controller.
- [ ] Check TD textport for `[CUDA IPC] Initialized successfully` from SD.
- [ ] Check SD log for `input_mem_name` matching bridge prefix.

### Step 4.4: Verify result frames

- [ ] Bridge importer (`import_top`) shows processed frames (not black, not color bars).
- [ ] `import_flip` shows correctly oriented SD output.

### Step 4.5: 500-frame stability test

- [ ] Run for 500+ frames.
- [ ] No black frames in output.
- [ ] GPU VRAM stable (no leak).

**Success criteria (from spec):** 500+ frames without black input in SD.

---

## Execution Notes

1. **Source of truth = live TD project** (`setup.3.toe`). Code must match live scene, not the other way around.
2. **Task 0 first**: audit live scene, fix existing bugs before any new features.
3. **Tasks 1-2 modify code** (sequential: harness first, then td_setup.py).
4. **Task 3 is live TD work** via h2t daemon — sync live scene with code changes.
5. **Task 4 is manual verification** — requires TD open, SD venv ready, GPU available.
6. **SD code is NOT modified** — only our channel naming and config generation adapt to SD's convention.
7. **Preflight checks**: sd_controller must validate paths before launch, error in textport if invalid.
8. **If conflict between code and live scene** — live scene wins, code adapts.
