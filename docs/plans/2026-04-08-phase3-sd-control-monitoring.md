# Phase 3: SD Control, Monitoring & Optimization — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute task-by-task. Each task is independently testable. Execute in order — later tasks depend on earlier ones being live.

**Spec:** `C:/dev/cuda/docs/specs/2026-04-08-phase3-sd-control-monitoring.md`
**Live scene:** `C:/dev/cuda/td/setup.6.toe` on port 9955
**All TD work:** `h2t td eval/batch` against live scene. No scene rebuild.

---

## Constraints (always enforced)

1. Execute DATs use `print()` not `debug()` — `debug()` is undefined in Execute DATs → NameError → CUDA error 700 cascade
2. `par.ext` cannot be set via Python in TD 2025 — use `mod('dat_name').function()` pattern
3. parexec DAT: `par.op` points to watched operator + `onValueChange` for string/float/int, `onPulse` for Pulse
4. `h2t td eval` returns None for multi-line code — use exec() wrapper or single expressions
5. backslash in generated code: `chr(92)`
6. SD OSC IN = 6503 (commands to SD), OSC OUT = 7187 (telemetry from SD)

---

## Task 1: SD Telemetry — OSC In DAT + CHOP on sd_controller

**Goal:** Receive `/stream-info/fps`, `/frame_ready`, `/server_active`, `/pipeline_fps` from SD on port 7187. Map to 4 CHOP channels.

**New operators inside `/project1/ml_bridge/sd_controller/`:**
- `sd_osc_in` — OSC In CHOP, port 7187
- `sd_chop` — Constant CHOP (fallback/display reference, 4 channels, initial 0)

**Rationale:** OSC In CHOP directly creates channels from OSC addresses. The channels update live as SD sends messages. No extra DAT needed.

### Step 1.1 — Create sd_osc_in CHOP

Save to `C:/dev/cuda/td/phase3-task1-osc-in.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "n = me.create(oscInCHOP, 'sd_osc_in'); n.par.port = 7187; n.par.netadr = '127.0.0.1'; n.par.active = True"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task1-osc-in.json --port 9955
```

### Step 1.2 — Set OSC address channel names

The OSC In CHOP auto-creates channels from OSC addresses. Map address → channel via the CHOP's address table:

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller/sd_osc_in').par.addrchan.val = '/stream-info/fps sd/fps /frame_ready sd/frame_ready /server_active sd/server_active /pipeline_fps sd/pipeline_fps'" --port 9955
```

> Note: If SD is not running, channels won't exist yet. The CHOP shows them once SD sends OSC.

### Step 1.3 — Verify

```bash
h2t td eval "str(op('/project1/ml_bridge/sd_controller/sd_osc_in').par.port.eval())" --port 9955
```

Expected: `7187`

When SD is running:
```bash
h2t td eval "[c.name + '=' + str(c[0]) for c in op('/project1/ml_bridge/sd_controller/sd_osc_in').chans()]" --port 9955
```

Expected: list including `sd/fps`, `sd/frame_ready`, `sd/server_active`, `sd/pipeline_fps`.

---

## Task 2: IPC Telemetry — Execute DAT reads SharedMemory headers → Constant CHOP

**Goal:** Every frame, read `write_idx` from `ml_ipc` and `ml_out_ipc` headers. Compute FPS over 1-second window. Write 6 channels to `ipc_chop` (Constant CHOP).

**New operators inside `/project1/ml_bridge/cuda_ipc_bridge/`:**
- `ipc_chop` — Constant CHOP, 6 channels (initial 0)
- `ipc_telemetry` — Execute DAT, `onFrameStart` callback

### Step 2.1 — Create ipc_chop (Constant CHOP, 6 channels)

Save to `C:/dev/cuda/td/phase3-task2a-ipc-chop.json`:

```json
[
  {
    "op": "/project1/ml_bridge/cuda_ipc_bridge",
    "method": "run",
    "args": [
      "n = me.create(constantCHOP, 'ipc_chop'); n.par.value0 = 0; n.par.value1 = 0; n.par.value2 = 0; n.par.value3 = 0; n.par.value4 = 0; n.par.value5 = 0; n.par.name0 = 'ipc/in_write_idx'; n.par.name1 = 'ipc/out_write_idx'; n.par.name2 = 'ipc/in_fps'; n.par.name3 = 'ipc/out_fps'; n.par.name4 = 'ipc/connected'; n.par.name5 = 'ipc/latency_ms'"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task2a-ipc-chop.json --port 9955
```

Verify:
```bash
h2t td eval "str([c.name for c in op('/project1/ml_bridge/cuda_ipc_bridge/ipc_chop').chans()])" --port 9955
```

Expected: `['ipc/in_write_idx', 'ipc/out_write_idx', 'ipc/in_fps', 'ipc/out_fps', 'ipc/connected', 'ipc/latency_ms']`

### Step 2.2 — Create ipc_telemetry Execute DAT

Save to `C:/dev/cuda/td/phase3-task2b-ipc-telemetry.json`:

```json
[
  {
    "op": "/project1/ml_bridge/cuda_ipc_bridge",
    "method": "run",
    "args": [
      "n = me.create(executeDAT, 'ipc_telemetry'); n.par.framestart = True; n.par.active = True"
    ]
  },
  {
    "op": "/project1/ml_bridge/cuda_ipc_bridge/ipc_telemetry",
    "method": "run",
    "args": [
      "me.text = open('C:/dev/cuda/td/ipc_telemetry_code.py', encoding='utf-8').read()"
    ]
  }
]
```

First, write the callback code to `C:/dev/cuda/td/ipc_telemetry_code.py` (see Step 2.3), then run:

```bash
h2t td batch C:/dev/cuda/td/phase3-task2b-ipc-telemetry.json --port 9955
```

### Step 2.3 — ipc_telemetry_code.py

Save to `C:/dev/cuda/td/ipc_telemetry_code.py`:

```python
# IPC Telemetry — Execute DAT, onFrameStart
# Reads SharedMemory headers for ml_ipc and ml_out_ipc
# Writes 6 channels to ipc_chop via Constant CHOP par assignment
# IMPORTANT: use print() not debug() — debug() undefined in Execute DAT

import struct as _struct
import time as _time

_state = {'in_idx_prev': 0, 'out_idx_prev': 0, 't_prev': None, 'in_fps': 0.0, 'out_fps': 0.0}

def onFrameStart(frame):
    try:
        from multiprocessing import shared_memory as _shm
        chop = op('ipc_chop')

        in_mem = None
        out_mem = None
        connected = 0

        try:
            in_mem = _shm.SharedMemory(name='ml_ipc', create=False)
            out_mem = _shm.SharedMemory(name='ml_out_ipc', create=False)
            connected = 1
        except Exception:
            pass

        if in_mem is None or out_mem is None:
            chop.par.value4 = 0  # ipc/connected
            if in_mem:
                in_mem.close()
            if out_mem:
                out_mem.close()
            return

        in_idx = _struct.unpack_from('<I', in_mem.buf, 16)[0]
        out_idx = _struct.unpack_from('<I', out_mem.buf, 16)[0]
        in_mem.close()
        out_mem.close()

        now = _time.perf_counter()
        if _state['t_prev'] is not None:
            dt = now - _state['t_prev']
            if dt >= 1.0:
                d_in = in_idx - _state['in_idx_prev']
                d_out = out_idx - _state['out_idx_prev']
                _state['in_fps'] = d_in / dt
                _state['out_fps'] = d_out / dt
                _state['in_idx_prev'] = in_idx
                _state['out_idx_prev'] = out_idx
                _state['t_prev'] = now
        else:
            _state['in_idx_prev'] = in_idx
            _state['out_idx_prev'] = out_idx
            _state['t_prev'] = now

        latency_ms = 0.0
        try:
            import_top = op('../import_top')
            latency_ms = import_top.cookTime * 1000.0 if import_top else 0.0
        except Exception:
            pass

        chop.par.value0 = in_idx
        chop.par.value1 = out_idx
        chop.par.value2 = _state['in_fps']
        chop.par.value3 = _state['out_fps']
        chop.par.value4 = connected
        chop.par.value5 = latency_ms

    except Exception as e:
        print('[ipc_telemetry] error:', e)
```

### Step 2.4 — Verify

With SD running and exporting:
```bash
h2t td eval "str([str(c.name) + '=' + str(round(c[0],2)) for c in op('/project1/ml_bridge/cuda_ipc_bridge/ipc_chop').chans()])" --port 9955
```

Expected: `ipc/in_fps > 0`, `ipc/out_fps > 0`, `ipc/connected = 1`

---

## Task 3: Runtime OSC Control — parexec → OSC Out to SD

**Goal:** Changes to Seed, Prompt, Negativeprompt, Delta, Guidancescale, Usecontrolnet, Maxfps pars on sd_controller instantly send OSC to SD port 6503. Pause/Play pulses also send OSC.

**New operators inside `/project1/ml_bridge/sd_controller/`:**
- `sd_osc_out` — OSC Out DAT, port 6503, host 127.0.0.1
- `sd_runtime_parexec` — parexec DAT watching sd_controller

### Step 3.1 — Add runtime custom pars to sd_controller

Save to `C:/dev/cuda/td/phase3-task3a-runtime-pars.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "n = me; p = n.appendCustomPage('Runtime'); p.appendInt('Seed', label='Seed'); n.par.Seed.val = 44814; p.appendStr('Prompt', label='Prompt'); n.par.Prompt.val = ''; p.appendStr('Negativeprompt', label='Negative Prompt'); n.par.Negativeprompt.val = ''; p.appendFloat('Delta', label='Delta'); n.par.Delta.min = 0.0; n.par.Delta.max = 1.0; n.par.Delta.val = 1.0; p.appendFloat('Guidancescale', label='Guidance Scale'); n.par.Guidancescale.val = 1.0; p.appendToggle('Usecontrolnet', label='Use ControlNet'); n.par.Usecontrolnet.val = 0; p.appendInt('Maxfps', label='Max FPS'); n.par.Maxfps.val = 120"
    ]
  },
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "n = me; p2 = n.appendCustomPage('Control'); p2.appendPulse('Pause', label='Pause'); p2.appendPulse('Play', label='Play')"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task3a-runtime-pars.json --port 9955
```

Verify:
```bash
h2t td eval "str([p.name for p in op('/project1/ml_bridge/sd_controller').customPages])" --port 9955
```

Expected: includes `'Runtime'` and `'Control'`.

### Step 3.2 — Create sd_osc_out DAT

Save to `C:/dev/cuda/td/phase3-task3b-osc-out.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "n = me.create(oscoutDAT, 'sd_osc_out'); n.par.port = 6503; n.par.netadr = '127.0.0.1'"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task3b-osc-out.json --port 9955
```

Verify:
```bash
h2t td eval "str(op('/project1/ml_bridge/sd_controller/sd_osc_out').par.port.eval())" --port 9955
```

Expected: `6503`

### Step 3.3 — Create sd_runtime_parexec DAT

Write the parexec code to `C:/dev/cuda/td/sd_runtime_parexec_code.py`:

```python
# parexec DAT — watches sd_controller pars
# onChange → OSC Out to SD port 6503
# onPulse → OSC Out to SD port 6503
# IMPORTANT: use print() not debug() — debug() is undefined in Execute DAT context

import json as _json

def onValueChange(par, prev):
    try:
        osc_out = op('sd_osc_out')
        name = par.name

        if name == 'Seed':
            osc_out.sendOSC('/seed', [int(par.eval())])

        elif name == 'Prompt':
            prompt_text = par.eval()
            encoded = _json.dumps([[prompt_text, 1.0]])
            osc_out.sendOSC('/prompt_list', [encoded])

        elif name == 'Negativeprompt':
            osc_out.sendOSC('/negative_prompt', [par.eval()])

        elif name == 'Delta':
            osc_out.sendOSC('/delta', [float(par.eval())])

        elif name == 'Guidancescale':
            osc_out.sendOSC('/guidance_scale', [float(par.eval())])

        elif name == 'Usecontrolnet':
            osc_out.sendOSC('/use_controlnet', [int(par.eval())])

        elif name == 'Maxfps':
            osc_out.sendOSC('/max_fps', [int(par.eval())])

    except Exception as e:
        print('[sd_runtime_parexec] onValueChange error:', e)


def onPulse(par):
    try:
        osc_out = op('sd_osc_out')
        name = par.name

        if name == 'Pause':
            osc_out.sendOSC('/pause', [])

        elif name == 'Play':
            osc_out.sendOSC('/play', [])

    except Exception as e:
        print('[sd_runtime_parexec] onPulse error:', e)
```

Save to `C:/dev/cuda/td/phase3-task3c-parexec.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "n = me.create(parexecDAT, 'sd_runtime_parexec'); n.par.op = '/project1/ml_bridge/sd_controller'; n.par.valuechange = True; n.par.pulse = True; n.par.active = True"
    ]
  },
  {
    "op": "/project1/ml_bridge/sd_controller/sd_runtime_parexec",
    "method": "run",
    "args": [
      "me.text = open('C:/dev/cuda/td/sd_runtime_parexec_code.py', encoding='utf-8').read()"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task3c-parexec.json --port 9955
```

### Step 3.4 — Verify OSC roundtrip

With SD running:
```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').par.Seed.val = 99999" --port 9955
```

Check SD log for: `[OSC] /seed 99999`. Visually confirm output changes within ~1 second.

Verify Prompt encoding:
```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').par.Prompt.val = 'test prompt'" --port 9955
```

SD should receive `/prompt_list` with value `[["test prompt", 1.0]]` (JSON string).

---

## Task 4: Launch Improvements — env vars, Model menu, Acceleration menu

**Goal:** sd_controller `start()` sets required env vars. Add Modelsdir, Model (menu), Acceleration (menu), Tensorrtvae (toggle) pars. Model menu scans Modelsdir for `.engine` files or subdirs.

### Step 4.1 — Add launch pars to sd_controller

Save to `C:/dev/cuda/td/phase3-task4a-launch-pars.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "n = me; p = n.appendCustomPage('Launch'); p.appendFolder('Modelsdir', label='Models Dir'); p.appendMenu('Model', label='Model'); n.par.Model.menuLabels = ['stabilityai/sd-turbo']; n.par.Model.menuNames = ['stabilityai/sd-turbo']; n.par.Model.val = 'stabilityai/sd-turbo'; p.appendMenu('Acceleration', label='Acceleration'); n.par.Acceleration.menuLabels = ['None', 'torch_compile (15min)', 'max-autotune (1.5hr)']; n.par.Acceleration.menuNames = ['none', 'torch_compile', 'max-autotune']; n.par.Acceleration.val = 'none'; p.appendToggle('Tensorrtvae', label='TensorRT VAE'); n.par.Tensorrtvae.val = True"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task4a-launch-pars.json --port 9955
```

> Note: Modelsdir is required — no default. User must set it before first Start.

Verify:
```bash
h2t td eval "str([p.name for p in op('/project1/ml_bridge/sd_controller').pars('Modelsdir', 'Model', 'Acceleration', 'Tensorrtvae')])" --port 9955
```

### Step 4.2 — Write process_mgr with env vars + model menu scan

Write `C:/dev/cuda/td/process_mgr_v3.py`:

```python
# sd_controller process_mgr — Phase 3
# Launches SD with env vars, generates config from UI pars
# IMPORTANT: print() only, never debug()

import subprocess as _subprocess
import json as _json
import os as _os

_process = None

def start(ctrl_op):
    global _process
    if _process is not None and _process.poll() is None:
        print('[sd_controller] already running PID={}'.format(_process.pid))
        return

    venv = ctrl_op.par.Venvpath.eval()
    sd_dir = ctrl_op.par.Sddir.eval()
    sd_config = ctrl_op.par.Sdconfig.eval()

    if not _os.path.isdir(sd_dir):
        print('[sd_controller] ERROR: SD dir not found:', sd_dir)
        return

    base_config_path = _os.path.join(sd_dir, sd_config)
    if not _os.path.isfile(base_config_path):
        print('[sd_controller] ERROR: Config not found:', base_config_path)
        return

    if venv and not _os.path.isfile(venv.replace(chr(92), '/') + '/Scripts/python.exe'):
        print('[sd_controller] ERROR: Python not found in venv:', venv)
        return

    with open(base_config_path, 'r') as f:
        config = _json.load(f)

    # Override channel names
    config['input_mem_name'] = 'ml'
    config['output_mem_name'] = 'ml_out'

    # Apply launch pars
    model = ctrl_op.par.Model.eval()
    if model:
        config['model_id_or_path'] = model

    accel = ctrl_op.par.Acceleration.eval()
    if accel == 'none':
        config['acceleration'] = 'none'
        config['compile_mode'] = 'default'
    elif accel == 'torch_compile':
        config['acceleration'] = 'torch_compile'
        config['compile_mode'] = 'default'
    elif accel == 'max-autotune':
        config['acceleration'] = 'torch_compile'
        config['compile_mode'] = 'max-autotune'

    config['tensorrt_vae_only'] = bool(ctrl_op.par.Tensorrtvae.eval())

    out_dir = 'C:/dev/cuda/td'
    _os.makedirs(out_dir, exist_ok=True)
    gen_config_path = _os.path.join(out_dir, 'sd_config_ml.json')
    with open(gen_config_path, 'w') as f:
        _json.dump(config, f, indent=2)
    print('[sd_controller] config saved:', gen_config_path)

    env = dict(_os.environ)
    env.update({
        'SDTD_CUDA_IPC_OUTPUT': '1',
        'SDTD_CONTROLNET_UNION': '1',
        'SDTD_TORCHAO_QUANTIZATION': '1',
        'SDTD_TORCHAO_QUNTIZATION_TYPE': 'float8dq',
        'SDTD_SDPA_BACKEND': 'cudnn',
        'SDTD_TRT_CUDA_GRAPH': '1',
        'SDTD_IPC_SNAPSHOT': '1',
        'SDTD_ZERO_SNR': '1',
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUNBUFFERED': '1',
    })

    if venv:
        python = venv.replace(chr(92), '/') + '/Scripts/python'
    else:
        python = 'python'

    main_script = _os.path.join(sd_dir, 'main_sdtd.py')
    cmd = [python, main_script, '--config', gen_config_path]
    print('[sd_controller] launching:', ' '.join(cmd))

    try:
        _process = _subprocess.Popen(
            cmd,
            env=env,
            shell=False,
            creationflags=_subprocess.CREATE_NEW_PROCESS_GROUP
        )
        print('[sd_controller] SD started PID={}'.format(_process.pid))
    except Exception as e:
        print('[sd_controller] SD start failed:', e)


def stop(ctrl_op):
    global _process
    if _process is None:
        print('[sd_controller] not running')
        return
    if _process.poll() is not None:
        print('[sd_controller] already stopped')
        _process = None
        return
    try:
        _process.terminate()
        _process.wait(timeout=5)
        print('[sd_controller] SD stopped')
    except Exception as e:
        print('[sd_controller] stop error:', e)
        _process.kill()
    _process = None


def scan_models(ctrl_op):
    """Scan Modelsdir and repopulate Model menu."""
    models_dir = ctrl_op.par.Modelsdir.eval()
    if not models_dir or not _os.path.isdir(models_dir):
        print('[sd_controller] Modelsdir not set or not found:', models_dir)
        return
    entries = []
    for name in _os.listdir(models_dir):
        full = _os.path.join(models_dir, name)
        if _os.path.isdir(full) or name.endswith('.engine'):
            entries.append(name)
    if not entries:
        print('[sd_controller] no models found in:', models_dir)
        return
    ctrl_op.par.Model.menuLabels = entries
    ctrl_op.par.Model.menuNames = entries
    if entries:
        ctrl_op.par.Model.val = entries[0]
    print('[sd_controller] models scanned:', entries)
```

Save to `C:/dev/cuda/td/phase3-task4b-process-mgr.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller/process_mgr",
    "method": "run",
    "args": [
      "me.text = open('C:/dev/cuda/td/process_mgr_v3.py', encoding='utf-8').read()"
    ]
  }
]
```

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task4b-process-mgr.json --port 9955
```

### Step 4.3 — Wire Start/Stop pulses to process_mgr

The existing harness_pulse / facade_pulse pattern is already in place. Verify the sd_controller parexec (existing) calls `mod('process_mgr').start(me)` / `stop(me)`:

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').findChildren(type=parexecDAT)[0].text" --port 9955
```

If it calls `debug()`, replace with `print()`:
```bash
h2t td eval "n = op('/project1/ml_bridge/sd_controller').findChildren(type=parexecDAT)[0]; n.text = n.text.replace('debug(', 'print(')" --port 9955
```

### Step 4.4 — Verify launch pars + env vars

Dry-run verification (SD dir must exist):
```bash
h2t td eval "str(op('/project1/ml_bridge/sd_controller').par.Tensorrtvae.eval())" --port 9955
h2t td eval "str(op('/project1/ml_bridge/sd_controller').par.Acceleration.menuNames)" --port 9955
```

Expected: `True`, `['none', 'torch_compile', 'max-autotune']`

Model scan (requires Modelsdir set):
```bash
h2t td eval "mod('/project1/ml_bridge/sd_controller/process_mgr').scan_models(op('/project1/ml_bridge/sd_controller'))" --port 9955
```

---

## Task 5: ControlNet Depth Validation

**Goal:** Confirm `ml-cn_ipc` is being written by `export_depth_exec` and that SD reads it when `Usecontrolnet = 1`.

**Prerequisite:** Depth source connected to `in_depth` of cuda_ipc_bridge. `SDTD_CONTROLNET_UNION=1` is now included in env vars (Task 4).

### Step 5.1 — Verify export_depth_exec writes to ml-cn_ipc

```bash
h2t td eval "op('/project1/ml_bridge/cuda_ipc_bridge/export_depth_exec').text" --port 9955
```

Confirm the text contains `ml-cn_ipc` (not `ml_cn_ipc` or `ml-depth_ipc`). The channel name must be exactly `ml-cn_ipc`.

If it uses a wrong name, patch:
```bash
h2t td eval "n = op('/project1/ml_bridge/cuda_ipc_bridge/export_depth_exec'); n.text = n.text.replace('ml_cn_ipc', 'ml-cn_ipc')" --port 9955
```

### Step 5.2 — Verify SharedMemory is created

With bridge running (at least one export cooking):
```bash
h2t td eval "exec('from multiprocessing import shared_memory as s\ntry:\n    m=s.SharedMemory(name=chr(109)+chr(108)+chr(45)+chr(99)+chr(110)+chr(95)+chr(105)+chr(112)+chr(99), create=False)\n    print(m.size)\n    m.close()\nexcept Exception as e:\n    print(e)')" --port 9955
```

> Simpler: in TD textport manually: `from multiprocessing import shared_memory as s; m = s.SharedMemory(name='ml-cn_ipc', create=False); print(m.size); m.close()`

Expected: prints buffer size (not an error).

### Step 5.3 — Enable ControlNet and start SD

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').par.Usecontrolnet.val = 1" --port 9955
```

Then Start SD via UI (or via eval). Check SD log for:
```
[CUDA IPC] Opened SharedMemory: ml-cn_ipc
```

### Step 5.4 — Visual confirmation

With depth source connected and ControlNet enabled, SD output should change character (depth-guided geometry). Compare import_flip output with/without ControlNet toggle.

---

## Task 6: Optimization Path — TRT VAE → FP8 → torch.compile

**Goal:** Reach ≥ 35 FPS on SD output (`ipc/out_fps`). Sequential steps, each verified before next.

**Verification metric:** `ipc/out_fps` in ipc_chop (Task 2).

```bash
h2t td eval "str(op('/project1/ml_bridge/cuda_ipc_bridge/ipc_chop')['ipc/out_fps'][0])" --port 9955
```

### Step 6.1 — TensorRT VAE (baseline, ~25-30 FPS)

TRT engine files must already be compiled (Alex pre-compiled). If not — skip to Step 6.2 first.

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').par.Tensorrtvae.val = True" --port 9955
```

Restart SD (Stop → Start). Wait for SD to load TRT engines (~30s). Verify:
- SD log: `[TRT] Loaded VAE engine`
- `ipc/out_fps` ≥ 25

### Step 6.2 — FP8 quantization (significant UNet speedup)

FP8 is controlled by `SDTD_TORCHAO_QUANTIZATION=1` (already in env, Task 4). It's enabled by default in process_mgr_v3. If FPS didn't improve, confirm the env var is present in the launched process:

```bash
h2t td eval "str(op('/project1/ml_bridge/sd_controller').par.Sddir.eval())" --port 9955
```

Check SD logs for `[torchao] quantization enabled`.

### Step 6.3 — cuDNN SDPA (faster attention)

`SDTD_SDPA_BACKEND=cudnn` is already in env vars (Task 4, process_mgr_v3). Verify SD logs for `[SDPA] using cudnn backend`.

### Step 6.4 — torch.compile default (15 min first compile)

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').par.Acceleration.val = 'torch_compile'" --port 9955
```

Restart SD. **First launch takes ~15 minutes** — compiled artifacts cached in `__pycache__`. Subsequent starts are fast. Monitor SD log for `[torch.compile] compilation done`.

**OOM mitigation if compile fails:**
- Close all other GPU apps
- Set env var: add `'TORCHINDUCTOR_COMPILE_THREADS': '1'` to env dict in process_mgr_v3
- Verify MSVC `cl.exe` is in PATH (required by torch.compile on Windows). If missing, add `Msvcpath` par to sd_controller and prepend to PATH in process_mgr `start()`:
  ```python
  msvc = ctrl_op.par.Msvcpath.eval()
  if msvc:
      env['PATH'] = msvc + ';' + env.get('PATH', '')
  ```

### Step 6.5 — torch.compile max-autotune (~1.5 hr compile)

Only after Step 6.4 is verified working and ≥ 35 FPS is not yet reached.

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').par.Acceleration.val = 'max-autotune'" --port 9955
```

Restart SD. Compile time ~1.5 hours. Run overnight or on dedicated machine.

### Step 6.6 — Verify FPS target

```bash
h2t td eval "str(op('/project1/ml_bridge/cuda_ipc_bridge/ipc_chop')['ipc/out_fps'][0])" --port 9955
```

Target: ≥ 35.

---

## Task 7: Resilience — debug→print audit, clean cache, zombie kill

**Goal:** Prevent CUDA error 700 cascade. Clean exit paths. Prevent OSC port conflicts.

### Step 7.1 — Audit all Execute DATs for debug() calls

```bash
h2t td eval "[(n.path, n.text.count('debug(')) for n in op('/project1/ml_bridge').findChildren(type=executeDAT) if 'debug(' in n.text]" --port 9955
```

For each DAT with `debug(` calls, patch:

```bash
h2t td eval "n = op('/project1/ml_bridge/cuda_ipc_bridge/export_img_exec'); n.text = n.text.replace('debug(', 'print(')" --port 9955
h2t td eval "n = op('/project1/ml_bridge/cuda_ipc_bridge/export_depth_exec'); n.text = n.text.replace('debug(', 'print(')" --port 9955
h2t td eval "n = op('/project1/ml_bridge/cuda_ipc_bridge/import_cooker'); n.text = n.text.replace('debug(', 'print(')" --port 9955
```

Repeat for any other Execute DATs found in Step 7.1 audit.

### Step 7.2 — Add Clean Cache pulse par

```bash
h2t td eval "n = op('/project1/ml_bridge/sd_controller'); p = [pg for pg in n.customPages if pg.name == 'Launch'][0]; p.appendPulse('Cleancache', label='Clean Cache')" --port 9955
```

### Step 7.3 — Add zombie kill + clean cache to process_mgr

Add to `C:/dev/cuda/td/process_mgr_v3.py` (already has the base), then push updated version:

Append these functions to the existing `process_mgr_v3.py`:

```python
def kill_sd_zombies():
    """Kill Python processes holding SD OSC ports (6503, 7187) only."""
    import subprocess as _sp
    SD_PORTS = {'6503', '7187'}
    killed = []
    try:
        result = _sp.run(
            ['netstat', '-ano'],
            capture_output=True, text=True, timeout=5
        )
        pids_to_kill = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5:
                local = parts[1]
                state = parts[3] if len(parts) > 3 else ''
                pid = parts[-1]
                port = local.split(':')[-1]
                if port in SD_PORTS and state in ('LISTENING', 'ESTABLISHED'):
                    pids_to_kill.add(pid)
        for pid in pids_to_kill:
            try:
                _sp.run(['taskkill', '/PID', pid, '/F'], capture_output=True, timeout=3)
                killed.append(pid)
                print('[sd_controller] killed zombie PID={} (SD port)'.format(pid))
            except Exception as e:
                print('[sd_controller] kill failed for PID={}: {}'.format(pid, e))
    except Exception as e:
        print('[sd_controller] zombie scan error:', e)
    if not killed:
        print('[sd_controller] no zombies found on ports 6503/7187')


def clean_cache(ctrl_op):
    """Delete cuda_ipc_transport __pycache__ and flush sys.modules entries."""
    import sys as _sys
    import shutil as _shutil

    cache_dir = 'C:/dev/cuda/cuda_ipc_transport/__pycache__'
    if _os.path.isdir(cache_dir):
        _shutil.rmtree(cache_dir)
        print('[sd_controller] deleted', cache_dir)
    else:
        print('[sd_controller] no cache dir found at', cache_dir)

    flushed = [k for k in list(_sys.modules.keys()) if 'cuda_ipc' in k or 'stream_diff' in k.lower()]
    for k in flushed:
        del _sys.modules[k]
    print('[sd_controller] flushed sys.modules:', flushed)
```

Push updated process_mgr:
```bash
h2t td batch C:/dev/cuda/td/phase3-task4b-process-mgr.json --port 9955
```

(Reuse the same batch file — it just overwrites process_mgr text from the updated py file)

### Step 7.4 — Wire Cleancache pulse + zombie kill on Start

The existing sd_controller parexec (for Start/Stop) needs to also handle `Cleancache` pulse. Check current parexec:

```bash
h2t td eval "op('/project1/ml_bridge/sd_controller').findChildren(type=parexecDAT)[0].text" --port 9955
```

If it only handles Start/Stop, append Cleancache and zombie kill on Start. Write updated parexec to `C:/dev/cuda/td/sd_launch_parexec_code.py`:

```python
# parexec for sd_controller — Launch pars (Start, Stop, Cleancache)
# Separate from sd_runtime_parexec which handles Runtime/Control pars

def onPulse(par):
    mgr = mod('process_mgr')
    ctrl = op('/project1/ml_bridge/sd_controller')
    name = par.name

    if name == 'Start':
        mgr.kill_sd_zombies()
        mgr.start(ctrl)

    elif name == 'Stop':
        mgr.stop(ctrl)

    elif name == 'Cleancache':
        mgr.clean_cache(ctrl)
```

Save to `C:/dev/cuda/td/phase3-task7-launch-parexec.json`:

```json
[
  {
    "op": "/project1/ml_bridge/sd_controller",
    "method": "run",
    "args": [
      "pxs = me.findChildren(type=parexecDAT); px = pxs[0] if pxs else me.create(parexecDAT, 'sd_launch_parexec'); px.par.op = '/project1/ml_bridge/sd_controller'; px.par.pulse = True; px.par.active = True"
    ]
  },
  {
    "op": "/project1/ml_bridge/sd_controller/sd_launch_parexec",
    "method": "run",
    "args": [
      "me.text = open('C:/dev/cuda/td/sd_launch_parexec_code.py', encoding='utf-8').read()"
    ]
  }
]
```

> If the existing parexec has a different name, adjust the op path above.

Run:
```bash
h2t td batch C:/dev/cuda/td/phase3-task7-launch-parexec.json --port 9955
```

### Step 7.5 — Verify resilience

Zombie kill dry run (no SD running):
```bash
h2t td eval "mod('/project1/ml_bridge/sd_controller/process_mgr').kill_sd_zombies()" --port 9955
```

Expected: `no zombies found on ports 6503/7187` (or lists killed PIDs if stale processes exist).

Clean cache dry run:
```bash
h2t td eval "mod('/project1/ml_bridge/sd_controller/process_mgr').clean_cache(op('/project1/ml_bridge/sd_controller'))" --port 9955
```

Expected: reports deleted cache or `no cache dir found`, flushes sys.modules.

Full CUDA error 700 audit — check TD textport is clean:
```bash
h2t td eval "str('error 700' not in str(ui.status))" --port 9955
```

---

## File Map

```
C:/dev/cuda/
  docs/
    specs/2026-04-08-phase3-sd-control-monitoring.md  (source spec)
    plans/2026-04-08-phase3-sd-control-monitoring.md  (this file)
  td/
    phase3-task1-osc-in.json
    phase3-task2a-ipc-chop.json
    phase3-task2b-ipc-telemetry.json
    ipc_telemetry_code.py
    phase3-task3a-runtime-pars.json
    phase3-task3b-osc-out.json
    phase3-task3c-parexec.json
    sd_runtime_parexec_code.py
    phase3-task4a-launch-pars.json
    phase3-task4b-process-mgr.json
    process_mgr_v3.py
    phase3-task7-launch-parexec.json
    sd_launch_parexec_code.py
```

---

## Success Criteria Checklist

| Criterion | Command | Target |
|-----------|---------|--------|
| IPC telemetry | `ipc_chop['ipc/in_fps'][0]` | > 0 |
| SD telemetry | `sd_osc_in['sd/fps'][0]` | > 0 (with SD running) |
| OSC roundtrip | Change Seed par → visual change | < 1 sec |
| ControlNet | SD log: `Opened SharedMemory: ml-cn_ipc` | Present |
| FPS | `ipc_chop['ipc/out_fps'][0]` | ≥ 35 |
| Stability | `ipc/out_write_idx` grows for 30+ sec | Continuous |
| No error 700 | TD textport | Clean |

---

## Execution Order

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7
(each independently testable, no hard deps between 3 and 4)
```

Task 5 (ControlNet) requires Task 3 (Usecontrolnet OSC control) to be live.
Task 6 (Optimization) requires Task 4 (env vars in launch) to be live.
Task 7 (Resilience) can run partially in parallel with Tasks 3-6 — the debug→print audit (7.1) is safe at any point.
