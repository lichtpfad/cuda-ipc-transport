# TD ML Bridge Asset — Design Spec
**Date:** 2026-04-07
**Status:** Approved
**Project:** cuda_ipc_transport

---

## Goal

Создать три вложенных TD COMP (TouchDesigner Component), которые упаковывают двунаправленный CUDA IPC транспорт + subprocess management + OSC контроль в drag-and-drop ассет. Два TOP входа (image + depth), один TOP выход (processed result). Одна кнопка Start.

---

## Architecture

```
ml_bridge COMP (outer facade)
├── IN: image TOP, depth TOP (optional)
├── OUT: processed TOP (flipped, correct orientation)
│
├── cuda_ipc_bridge COMP           ← generic transport
│   ├── TDCUDAIPCExporter (image channel)
│   ├── TDCUDAIPCExporter (depth channel)
│   ├── Script TOP importer (result channel)
│   ├── Cooker Execute DAT (force-cook каждый кадр)
│   ├── Flip TOP (OpenGL origin fix)
│   ├── OSC In CHOP (transport status: connected, frame_number)
│   └── Params: Pkgpath, Channelprefix, Statusoscport
│
├── sd_controller COMP             ← pluggable model controller
│   ├── Subprocess management (start/stop via Python subprocess)
│   ├── OSC Out CHOP → harness (control: prompt, weight, seed)
│   ├── OSC In CHOP ← harness (model status: loaded, inference_ms, errors)
│   └── Params: Venvpath, Command, Controloscport, + model-specific
│
└── Params (facade): Channelprefix, Venvpath, Start/Stop pulse, Status indicators
```

---

## Channel Naming Convention

`{prefix}_img`, `{prefix}_depth`, `{prefix}_result`

Где `prefix` = параметр `Channelprefix` на bridge (default: `"ml"`).
Позволяет нескольким инстансам работать одновременно без конфликтов.

---

## 1. cuda_ipc_bridge COMP

### Purpose
Generic CUDA IPC транспорт. Ничего не знает про модели или процессы. Экспортирует TOP inputs в CUDA IPC каналы, импортирует результат обратно.

### Parameters (Custom Page "CUDA IPC")

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `Pkgpath` | Str | `C:/dev/cuda` | Путь к `cuda_ipc_transport` пакету |
| `Channelprefix` | Str | `ml` | Префикс каналов SharedMemory |
| `Statusoscport` | Int | `7001` | OSC порт для приёма статуса из harness |

### Internal Nodes

**Exporters** (2x Execute DAT):
- `export_img_exec` — onFrameStart: `TDCUDAIPCExporter.ExportFrame(input_image_top)`
- `export_depth_exec` — onFrameStart: `TDCUDAIPCExporter.ExportFrame(input_depth_top)`, пропускает если depth input не подключён

**Importer** (Script TOP + callbacks DAT):
- `import_callbacks` — textDAT с onCook, использует `CUDAIPCReceiver(f"{prefix}_result")`
- `import_top` — scriptTOP, `par.callbacks = import_callbacks`
- `import_cooker` — executeDAT, `cook(force=True)` каждый кадр
- `import_flip` — flipTOP, `flipy=1`

Callbacks определяются прямо в DAT (не импортируются из пакета). `CUDAMemoryShape`, `debug` — TD built-ins, доступны только в namespace DAT.

**OSC Status** (CHOP):
- `osc_status_in` — oscInCHOP на порту `Statusoscport`
- Ожидаемые адреса: `/transport/connected` (0/1), `/transport/frame` (int)
- Status CHOP выводится на внутреннюю панель для мониторинга

### Input/Output

- **In 1**: `in_image` (inTOP) — проксирует TOP input 0
- **In 2**: `in_depth` (inTOP) — проксирует TOP input 1 (optional)
- **Out**: `out_result` (outTOP) — из `import_flip`

---

## 2. sd_controller COMP

### Purpose
Управление Python subprocess + OSC контроль модели. Pluggable — можно заменить на контроллер для любой другой модели.

### Parameters (Custom Page "Model")

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `Venvpath` | Str | `` | Путь к venv Python (напр. `C:/work/stream_alex/.../venv`) |
| `Command` | Str | `python -m cuda_ipc_transport` | Команда запуска |
| `Commandargs` | Str | `--source test --channel ml --width 512 --height 512` | Аргументы |
| `Controloscport` | Int | `7002` | OSC порт для отправки контроля в harness |
| `Modeloscport` | Int | `7003` | OSC порт для приёма статуса от модели |
| `Start` | Pulse | — | Запуск subprocess |
| `Stop` | Pulse | — | Остановка subprocess |

### SD-Specific Parameters (Custom Page "StreamDiffusion")

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `Prompt` | Str | `""` | Текущий prompt |
| `Cnweight` | Float | `0.5` | ControlNet weight |
| `Seed` | Int | `-1` | Seed (-1 = random) |
| `Strength` | Float | `0.5` | img2img strength |

При изменении SD params → отправляются через OSC Out на `Controloscport`:
- `/sd/prompt "text"`
- `/sd/cn_weight 0.5`
- `/sd/seed 42`
- `/sd/strength 0.5`

### Internal Nodes

**Subprocess management**:
- `process_mgr` — textDAT (Extension), Python `subprocess.Popen` для запуска harness
- `onPulse(Start)` → запуск: `{Venvpath}/Scripts/python {Command} {Commandargs}`
- `onPulse(Stop)` → `process.terminate()`
- Auto-stop при закрытии проекта (`onProjectPreSave`)

**OSC Control Out**:
- `osc_control_out` — oscOutCHOP на порт `Controloscport`
- Par Execute DAT слушает изменения SD params → отправляет по OSC

**OSC Model Status In**:
- `osc_model_in` — oscInCHOP на порту `Modeloscport`
- Адреса: `/model/loaded` (0/1), `/model/inference_ms` (float), `/model/error` (str)

---

## 3. ml_bridge COMP (outer facade)

### Purpose
Фасад для конечного пользователя. Проксирует входы/выходы, собирает ключевые параметры.

### Parameters (Custom Page "ML Bridge")

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `Channelprefix` | Str | `ml` | → `cuda_ipc_bridge/Channelprefix` |
| `Venvpath` | Str | `` | → `sd_controller/Venvpath` |
| `Start` | Pulse | — | → `sd_controller/Start` |
| `Stop` | Pulse | — | → `sd_controller/Stop` |

Facade params связаны с внутренними через `op('cuda_ipc_bridge').par.Channelprefix` bind или export.

### Input/Output

- **In 0**: image TOP → `cuda_ipc_bridge` input 0
- **In 1**: depth TOP → `cuda_ipc_bridge` input 1
- **Out 0**: `cuda_ipc_bridge` output (processed, flipped)

### Status Display

На внутренней панели (или через custom parameter page):
- `Connected` (bool) — из bridge OSC In
- `Frame` (int) — из bridge OSC In
- `Process` (str) — "running" / "stopped" — из sd_controller

---

## Harness OSC Extension

`cuda_ipc_transport/harness.py` нужно расширить:

### Новые аргументы
- `--osc-status-port 7001` — порт для отправки transport status
- `--osc-control-port 7002` — порт для приёма control messages (future)

### OSC Status Output (transport level)
Отправляется каждые N кадров (default: каждый кадр):
- `/transport/connected 1` — при старте и каждые 60 кадров (heartbeat)
- `/transport/frame {N}` — номер кадра

При shutdown: `/transport/connected 0`

### Dependencies
- `python-osc` добавляется в `pyproject.toml` как optional dependency: `pip install cuda_ipc_transport[osc]`

---

## Setup Script Update

`scripts/td_setup.py` обновляется для создания полной структуры COMP вместо flat nodes. Новый режим: `--mode comp` (default) vs `--mode flat` (текущее поведение).

---

## Phasing

### Phase 1 (этот spec)
- `cuda_ipc_bridge` COMP с 2 exporters + 1 importer + cooker + flip
- `sd_controller` COMP с subprocess start/stop + venv path
- `ml_bridge` COMP (facade)
- Harness OSC status (connected, frame)
- `td_setup.py --mode comp`

### Phase 2
- SD-specific params → OSC Out
- OSC model status In
- Par Execute для live param changes

### Phase 3
- Production hardening: auto-reconnect, error display, multiple instances
- Custom UI panel

### Backlog
- **Server mode**: обернуть SD процессор в persistent server (HTTP/gRPC). Модель загружается один раз, команды приходят по API. Несколько клиентов. OSC остаётся для realtime control, server API — для lifecycle management (load model, switch checkpoint, health check).

---

## Success Criteria Phase 1

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | `ml_bridge` drag-and-drop: connect 2 TOPs, press Start → harness runs | works |
| 2 | OSC `/transport/connected` reaches bridge within 2s of harness start | < 2s |
| 3 | OSC `/transport/frame` increments in sync with harness | +/- 1 frame |
| 4 | Stop button terminates subprocess cleanly | exit code 0 or SIGTERM |
| 5 | Re-run `td_setup.py` after teardown restores full setup | idempotent |
| 6 | Two `ml_bridge` instances with different prefixes don't conflict | no SharedMemory collision |
