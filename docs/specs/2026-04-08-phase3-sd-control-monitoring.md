# Phase 3: SD Control, Monitoring & Optimization — Design Spec

## Goal

Make the CUDA IPC pipeline production-ready: full telemetry, real-time SD parameter control via OSC, ControlNet depth integration, 40+ FPS optimizations, and minimal resilience against CUDA context corruption.

## Success Criteria

| Criterion | Target | How to verify |
|-----------|--------|---------------|
| IPC telemetry | 6 CHOP channels updating every frame | `ipc/in_fps > 0` and `ipc/out_fps > 0` in TD |
| SD telemetry | 4 CHOP channels from SD OSC | `sd/fps > 0` and `sd/frame_ready` incrementing |
| OSC roundtrip | Change Seed par → SD output changes within 1 sec | Visual confirmation in import_flip |
| ControlNet | Toggle on → SD reads `ml-cn_ipc` → output changes | SD log: `Opened SharedMemory: ml-cn_ipc` |
| FPS | ≥ 35 FPS on SD output (ml_out_ipc) | `ipc/out_fps` ≥ 35 with TRT VAE + FP8 |
| Stability | 500+ frames without freeze or CUDA error | `ipc/out_write_idx` grows continuously for 30+ sec |
| Resilience | No CUDA error 700 during normal operation | TD textport clean of `error 700` messages |

## Architecture

Two COMPs as isolated modules, each with its own responsibilities and OSC ports. Work is done in the **live TD scene** (`setup.6.toe`) via `h2t td eval/batch` — no scene recreation.

```
ml_bridge (outer COMP, facade)
├── cuda_ipc_bridge (COMP) — IPC transport layer
│   ├── IPC telemetry DAT (reads SharedMemory headers every frame)
│   │   → CHOP channels: ipc/in_write_idx, ipc/out_write_idx,
│   │     ipc/in_fps, ipc/out_fps, ipc/connected, ipc/latency_ms
│   ├── exporters: image (ml_ipc) + depth (ml-cn_ipc)
│   ├── importer: Script TOP (ml_out_ipc → copyCUDAMemory)
│   └── cooker, flip, out_result
│
└── sd_controller (COMP) — SD process + parameter control
    ├── OSC In CHOP: port 6503 ← SD telemetry (SD transmits on osc_in_port)
    │   → channels: stream-info/fps, frame_ready, server_active, pipeline_fps, framecount
    ├── OSC Out DAT: port 7187 → SD commands (SD listens on osc_out_port)
    ├── Custom Parameters (UI):
    │   [Launch] Start (Pulse), Stop (Pulse), Clean Cache (Pulse)
    │   [Launch] Venvpath, Sddir, Modelsdir, Model (Menu), Acceleration (Menu), Tensorrtvae (Toggle)
    │   [Runtime] Seed (Int), Prompt (Str), Negativeprompt (Str), Delta (Float 0-1),
    │             Guidancescale (Float), Usecontrolnet (Toggle), Maxfps (Int)
    │   [Control] Pause (Pulse), Play (Pulse)
    ├── process_mgr DAT: start/stop SD with env vars
    └── parexec DAT: onChange → OSC Out, onPulse → OSC Out
```

## IPC Telemetry (cuda_ipc_bridge)

**Source**: SharedMemory headers, read directly via Python `struct.unpack_from`. No OSC — data is local.

**Implementation**: Execute DAT (`ipc_telemetry`) runs `onFrameStart`:
- Opens `ml_ipc` and `ml_out_ipc` SharedMemory (cached handles)
- Reads `write_idx` from offset 16 (uint32 LE)
- Computes FPS as delta(write_idx) / delta(time) over 1-second window
- Reads importer `cookTime` for latency
- Writes values to a Constant CHOP (`ipc_chop`) via direct channel assignment: `op('ipc_chop').par.value0 = fps_in` etc.

**CHOP Channels**:

| Channel | Source | Description |
|---------|--------|-------------|
| `ipc/in_write_idx` | `ml_ipc` header | TD→SD frame counter |
| `ipc/out_write_idx` | `ml_out_ipc` header | SD→TD frame counter |
| `ipc/in_fps` | delta/time | TD export rate |
| `ipc/out_fps` | delta/time | SD output rate |
| `ipc/connected` | SharedMemory exists check | 1 if both channels live |
| `ipc/latency_ms` | importer cookTime | copyCUDAMemory timing |

**Important**: Use `print()` not `debug()` in Execute DAT error handling. `debug()` is undefined in Execute DATs and causes NameError which masks real errors.

## SD Telemetry (sd_controller)

**Source**: OSC messages from SD process on port 7187.

**Implementation**: OSC In DAT on sd_controller, port 7187. Maps to CHOP channels.

| CHOP Channel | OSC Address from SD | Description |
|-------------|---------------------|-------------|
| `sd/fps` | `/stream-info/fps` | SD inference FPS |
| `sd/frame_ready` | `/frame_ready` | SD frame counter |
| `sd/server_active` | `/server_active` | SD process alive (1/0) |
| `sd/pipeline_fps` | `/pipeline_fps` | Full pipeline FPS |

## SD Control Parameters

### Launch Parameters (require SD restart)

Changed in `sd_config_ml.json`, then Stop → Start SD.

| Parameter | Type | Config Key | Default |
|-----------|------|-----------|---------|
| Model | Menu (scan Modelsdir) | `model_id_or_path` | `stabilityai/sd-turbo` |
| Acceleration | Menu: none / torch_compile / max-autotune | `acceleration` + `compile_mode` | `none` |
| Tensorrtvae | Toggle | `tensorrt_vae_only` | `true` |
| Modelsdir | Folder | — (scan path) | Required: user must set before first Start |

**Model Menu population**: On sd_controller init and on Modelsdir change, scan directory for `.engine` files or model directories, populate Menu par.

### Runtime Parameters (OSC, instant)

parexec DAT on sd_controller watches these pars. On `onValueChange` → OSC Out sends to SD port 6503.

| Parameter | Type | OSC Address | Default |
|-----------|------|------------|---------|
| Seed | Int | `/seed` | 44814 |
| Prompt | String | `/prompt_list` | (from config) |

**Prompt encoding**: SD's `/prompt_list` handler expects JSON: `[[prompt, weight]]`. The parexec converts the plain string Prompt par to `json.dumps([[prompt_text, 1.0]])` before sending via OSC.
| Negativeprompt | String | `/negative_prompt` | "" |
| Delta | Float [0, 1] | `/delta` | 1.0 |
| Guidancescale | Float | `/guidance_scale` | 1.0 |
| Usecontrolnet | Toggle | `/use_controlnet` | 0 |
| Maxfps | Int | `/max_fps` | 120 |

### Control (Pulse → OSC)

| Parameter | OSC Address |
|-----------|------------|
| Pause | `/pause` |
| Play | `/play` |

## SD Launch — Environment Variables

sd_controller `start()` sets these env vars before launching SD subprocess:

```
SDTD_CUDA_IPC_OUTPUT=1
SDTD_CONTROLNET_UNION=1
SDTD_TORCHAO_QUANTIZATION=1
SDTD_TORCHAO_QUNTIZATION_TYPE=float8dq
SDTD_SDPA_BACKEND=cudnn
SDTD_TRT_CUDA_GRAPH=1
SDTD_IPC_SNAPSHOT=1
SDTD_ZERO_SNR=1
PYTHONIOENCODING=utf-8
PYTHONUNBUFFERED=1
```

**MSVC requirement**: `torch.compile` needs `cl.exe` in PATH. Alex's `start_TD.cmd` configures this. We need to either: (a) read MSVC path from Alex's cmd, or (b) add a Msvcpath par on sd_controller.

## ControlNet Depth Integration

**Channel**: `ml-cn_ipc` (already created, 60fps, 3 slots).

**SD reads it automatically**: `{input_mem_name}-cn_ipc` is auto-constructed by SD when `use_controlnet=true` (main_sdtd.py:949).

**Requirements**:
1. Depth source connected to `in_depth` input of cuda_ipc_bridge
2. `use_controlnet: true` in config OR `/use_controlnet 1` via OSC
3. `SDTD_CONTROLNET_UNION=1` env var (in launch scope)
4. ControlNet model specified in config: `controlnet_model: "thibaud/controlnet-sd21-depth-diffusers"`

**Validation**: After enabling, verify SD log shows `[CUDA IPC] Opened SharedMemory: ml-cn_ipc`.

## Optimization Path to 40+ FPS

Based on Alex's recommendations. Sequential, each step verified before next:

1. **TensorRT VAE** (`tensorrt_vae_only: true`) — engine files already compiled, ~25-30 FPS
2. **FP8 quantization** (`SDTD_TORCHAO_QUANTIZATION=1`) — significant UNet speedup
3. **cuDNN SDPA** (`SDTD_SDPA_BACKEND=cudnn`) — faster attention
4. **torch.compile default** (`acceleration: "torch_compile"`) — 15 min first compile
5. **torch.compile max-autotune** (`compile_mode: "max-autotune"`) — ~1.5 hour compile, best perf

Each step is a config/env change → restart SD → verify FPS. Menu par `Acceleration` on sd_controller controls steps 4-5.

**OOM mitigation**: If torch.compile OOMs during compilation:
- Close other GPU-consuming apps
- Study Alex's `start_TD.cmd` for MSVC and memory settings
- Consider `TORCHINDUCTOR_COMPILE_THREADS=1` to reduce memory pressure

## Resilience (Minimum)

1. **debug→print in all Execute DATs**: All error handling uses `print()`, never `debug()`. Prevents NameError → CUDA error 700 cascade.
2. **Clean Cache button**: Pulse par on sd_controller. Deletes `cuda_ipc_transport/__pycache__/` and flushes `sys.modules` in TD. Manual trigger only.
3. **Zombie kill on Start**: sd_controller `start()` checks for Python processes bound to SD's OSC ports (6503, 7187) and kills only those PIDs. Does NOT kill arbitrary Python processes — only port-matched ones. Prevents OSC port conflicts and stale SharedMemory.

## Backlog (not in this spec)

- Auto-reconnect on SD crash
- Scene restore from code (td_setup.py as generator)
- Graceful degradation (importer shows "no signal" when SD disconnected)
- LCM-LoRA support (`lcm_lora_id` par)
- LoRA + TensorRT compatibility research
- IP-Adapter integration
- VRAM monitoring display
- V2V cached attention tuning
- SageAttention optimization (+15-20%)
- `controlnet_weight` OSC endpoint (currently commented out in SD code)

## Implementation Approach

All work via `h2t td eval/batch` on live `setup.6.toe`. No td_setup.py recreation.

**Order of implementation**:
1. SD telemetry (OSC In on sd_controller, port 7187)
2. IPC telemetry (Execute DAT reading SharedMemory headers)
3. Runtime OSC control (parexec → OSC Out for seed/prompt/delta/etc.)
4. Launch improvements (env vars, model menu, acceleration menu)
5. ControlNet depth validation
6. Optimization path (TRT VAE → FP8 → torch.compile)
7. Resilience fixes (debug→print, clean cache, zombie kill)
