# Handoff — CUDA IPC Phase 3 — 2026-04-08 — OSC Control + Telemetry

## Результат сессии

**SD Runtime OSC control работает (prompt, seed). Telemetry работает (11 каналов). БЛОКЕР: exporters пишут чёрные кадры в ml_ipc — SD получает нулевой вход, delta не влияет.**

## Что сделано

### 1. Phase 2 завершён — bidirectional CUDA IPC live
- TD→SD: `ml_ipc` @ 60fps (exporters пишут)
- SD→TD: `ml_out_ipc` @ 21fps (importer читает, copyCUDAMemory)
- Картинка SD видна в TD (import_flip)
- Причина предыдущей нерабочести: stale `__pycache__` с старым MAGIC (0x43504943 vs 0x43495043)
- Причина CUDA error 700: exporters использовали `debug()` в Execute DAT (undefined → NameError → sticky CUDA error)
- Fix: `debug()` → `print()` в exporters, перезапуск TD для сброса CUDA context

### 2. SD Telemetry (Task 1) — DONE
- OSC In CHOP `sd_osc_in1` на sd_controller, порт **6503**
- 5 каналов от SD: `frame_ready`, `framecount`, `pipeline_fps`, `server_active`, `stream-info/fps`
- **Порты SD контринтуитивны**: `osc_in_port` (6503) = SD transmit, `osc_out_port` (7187) = SD receive

### 3. IPC Telemetry (Task 2) — DONE
- Execute DAT `ipc_telemetry1` в cuda_ipc_bridge, file-synced к `C:/dev/cuda/td/ipc_telemetry_code.py`
- Constant CHOP `ipc_chop1` с 6 каналами: `ipc:in_write_idx`, `ipc:out_write_idx`, `ipc:in_fps`, `ipc:out_fps`, `ipc:connected`, `ipc:latency_ms`
- Читает SharedMemory headers каждый фрейм через `struct.unpack_from`

### 4. Runtime OSC Control (Task 3) — PARTIALLY DONE
- Custom pars на sd_controller (Runtime page): Seed, Prompt, Negativeprompt, Delta, Guidancescale, Usecontrolnet, Maxfps, Pause, Play
- parameterexecuteDAT `sd_parexec1` — watches sd_controller, code in `C:/dev/cuda/td/sd_parexec_code.py`
- Отправка через `pythonosc` из project venv (`C:/dev/cuda/.venv/Lib/site-packages`)
- **Prompt работает** — визуально подтверждено (neon city → alien forest → golden sunset)
- **Seed работает** — визуально подтверждено (seed_list JSON format)
- **Delta отправляется** (textport: `[parexec] /delta = X`) но **не проверяем** — blocked by #44

### 5. Spec + Plan написаны
- Spec: `C:/dev/cuda/docs/specs/2026-04-08-phase3-sd-control-monitoring.md`
- Plan: `C:/dev/cuda/docs/plans/2026-04-08-phase3-sd-control-monitoring.md` (975 строк, 7 задач)

### 6. SD запуск
- Конфиг: `C:/dev/cuda/td/sd_config_ml.json` (acceleration=none, 20fps)
- Env: `SDTD_CUDA_IPC_OUTPUT=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1`
- SD запущен вручную из cmd, не из TD

## БЛОКЕР: Exporters пишут чёрные кадры

**Симптом:** `ml_ipc` write_idx застыл на 1957433 (0 fps). SD log: `pixel[0,0]=[0, 0, 0, 0]` — все нули.

**Что известно:**
- `export_img_exec` active=True, no errors
- `in_image` width=512 (источник подключён)
- `ExportFrame(top)` вызывается без exception
- Но GPU буферы за IPC handles пустые (нули)
- Textport не показывает `[bridge] export error` — ошибка молчаливая

**Гипотезы:**
1. `TDCUDAIPCExporter.ExportFrame()` не копирует пиксели в GPU буфер (CUDA memcpy не вызывается или failing silently)
2. Exporter инициализируется каждый фрейм заново (переменная `_exporter` не персистит)
3. CUDA context mismatch между exporter и IPC handle allocation
4. `top.cudaMemory()` возвращает пустой буфер (TOP не рендерится)

**Следующие шаги:**
1. Прочитать полный код `export_img_exec` Execute DAT
2. Добавить debug prints в каждый шаг: init, ExportFrame, cudaMemcpy
3. Проверить `top.cudaMemory()` — возвращает ли валидный pointer
4. Сравнить с рабочим шаблоном (Alex's `CUDAIPCExporter` из `numpy_share_out`)

## Операторы в live TD scene (setup.7.toe)

```
/project1/ml_bridge/
  cuda_ipc_bridge/
    in_image, in_depth (inTOP)
    export_img_exec, export_depth_exec (executeDAT) — ACTIVE but broken (black frames)
    import_top (scriptTOP) — WORKING
    import_callbacks (textDAT) — inline code, not file-synced
    import_cooker (executeDAT) — force-cook, ACTIVE
    import_flip (flipTOP)
    out_result (outTOP)
    ipc_chop1 (constantCHOP) — 6 telemetry channels, WORKING
    ipc_telemetry1 (executeDAT) — file-synced to ipc_telemetry_code.py, WORKING
    harness_mgr, harness_pulse, osc_status_in
  sd_controller/
    process_mgr (textDAT)
    pulse_handler (textDAT)
    sd_osc_in1 (oscinCHOP) — port 6503, WORKING (5 channels from SD)
    sd_osc_out1 (oscoutDAT) — port 7187 (NOT USED, parexec uses pythonosc instead)
    sd_osc_num_out1 (oscoutCHOP) — port 7187 (NOT USED, was experiment)
    sd_parexec1 (parameterexecuteDAT) — WORKING, uses pythonosc from venv
    sd_rename1 (renameCHOP) — NOT USED (was CHOP OSC experiment)
    par1 (parameterCHOP) — reads runtime pars as CHOP channels
  facade_pulse (textDAT)
```

## Файлы на диске

```
C:/dev/cuda/td/
  sd_config_ml.json         — SD config (acceleration=none, input=ml, output=ml_out)
  sd_parexec_code.py        — parexec callbacks (pythonosc, Runtime pars → OSC)
  ipc_telemetry_code.py     — SharedMemory header reader → Constant CHOP
  setup.7.toe               — current TD scene
  test_cuda.py              — debug script for SharedMemory + CUDA from TD
  phase3-task1-osc-in.json  — batch for OSC In creation (reference)
  debug_*.json              — debug batch files (can delete)
```

## Ключевые уроки этой сессии

1. **OSC порты SD перевёрнуты**: `osc_in_port` = transmit, `osc_out_port` = receive (main_sdtd.py:4327-4328)
2. **oscoutDAT.sendOSC()** не работает надёжно для float/int — используй pythonosc из venv
3. **pythonosc в TD**: добавить venv site-packages в sys.path, lazy import в callback
4. **Seed → /seed_list**: SD main loop читает `seed_list` (JSON [[seed,weight]]), не `seed`
5. **Prompt → /prompt_list**: JSON [[prompt, weight]]
6. **parameterexecuteDAT** callbacks работают только когда код в text DAT (не file-synced)
7. **h2t-snap после КАЖДОГО изменения** — никогда не просить пользователя проверять визуально
8. **CUDA error 700 sticky** — Execute DAT + debug() = NameError → CUDA context poisoned → restart TD

## Порты

| Порт | Использование |
|------|--------------|
| 6503 | SD transmit (telemetry FROM SD) → TD OSC In CHOP |
| 7187 | SD receive (commands TO SD) ← pythonosc from parexec |
| 9955 | TD WebServer |
| 9970 | h2t daemon |

## Запуск SD

```bash
cd C:/work/stream_alex/StreamDiffusionTD-Custom/StreamDiffusion/StreamDiffusionTD
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 SDTD_CUDA_IPC_OUTPUT=1 \
  ../venv/Scripts/python main_sdtd.py --config C:/dev/cuda/td/sd_config_ml.json
```

## Git

```
04855fc docs: update Phase 3 spec with correct OSC port mapping
0b524e4 feat: SD runtime OSC control + IPC telemetry
7d3d3d2 docs: Phase 3 implementation plan — 7 tasks with TD batch commands
ca1eead docs: Phase 3 spec — SD control, monitoring, optimization
30e21ba fix: improve receiver error reporting, add handoff doc and SD config
```
