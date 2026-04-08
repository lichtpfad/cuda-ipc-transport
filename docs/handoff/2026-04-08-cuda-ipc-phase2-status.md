# Handoff — CUDA IPC Phase 2 — 2026-04-08

## Статус

**CUDA IPC bidirectional pipeline РАБОТАЕТ, но картинка замерзает после первого кадра.**

- TD → `ml_ipc` → SD: экспортеры пишут видеокадры (write_idx растёт)
- SD обрабатывает через sd-turbo, ~20 FPS (без torch.compile)
- SD → `ml_out_ipc` → TD: importer получает первый кадр, затем замерзает
- Обработанная картинка видна в TD (SD output: bone marrow organic lattice)

## Решённые проблемы

1. **Protocol magic mismatch** — MAGIC был 0x43504943, SD ожидает 0x43495043. Исправлено в protocol.py
2. **SD CUDA IPC output opt-in** — нужна `SDTD_CUDA_IPC_OUTPUT=1` env var
3. **Unicode crash на Windows** — `PYTHONIOENCODING=utf-8` при запуске SD
4. **OSC zombie processes** — порт 7187 занят после crash, нужно kill перед перезапуском
5. **Stale module cache в TD** — `__pycache__` + `sys.modules` кэшируют старый MAGIC; нужно flush + touch DAT text
6. **Execute DAT: debug() undefined** — Execute DATs не имеют `debug()`, только `print()`

## Нерешённые проблемы

### 1. Картинка замерзает после первого кадра (ВЫСОКИЙ ПРИОРИТЕТ)
- Importer подключается, получает один кадр, затем copyCUDAMemory не обновляет
- Возможные причины: CUDA stream, ring buffer sync, cooker не работает
- Связано с handoff 2026-04-05: copyCUDAMemory без stream = poor performance

### 2. Exporters ломают CUDA контекст (error 700)
- `ExportFrame()` в Execute DAT вызывает CUDA error 700 (sticky)
- Отравляет весь TD CUDA контекст, importer тоже перестаёт работать
- Реальная ошибка скрыта за `NameError: debug not defined`
- Нужно: заменить debug→print, найти реальную CUDA ошибку

### 3. torch.compile OOM
- `acceleration: "torch_compile"` с `fullgraph=True` заняло 16 GB VRAM и крашнулось
- `acceleration: "none"` работает (~20 FPS, 6 GB VRAM)
- Для production нужно либо torch.compile с кэшем, либо TensorRT

## Запуск SD

```bash
cd C:/work/stream_alex/StreamDiffusionTD-Custom/StreamDiffusion/StreamDiffusionTD
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 SDTD_CUDA_IPC_OUTPUT=1 \
  ../venv/Scripts/python main_sdtd.py --config C:/dev/cuda/td/sd_config_ml.json
```

## Конфигурация

- SD config: `C:/dev/cuda/td/sd_config_ml.json` (acceleration=none, input=ml, output=ml_out)
- TD project: `C:/dev/cuda/td/setup.5.toe`
- TD port: 9955
- SD venv: `C:/work/stream_alex/StreamDiffusionTD-Custom/StreamDiffusion/venv`

## Alex's repo — полезные файлы (НЕ ПРОЧИТАНЫ, сохранены для исследования)

```
C:/work/stream_alex/StreamDiffusionTD-Custom/
├── start_TD.cmd              # Launcher: sets PYTHONPATH, torch/CUDA DLLs, MSVC
├── run_torch_trace.bat       # Torch tracing/profiling
├── clean_pycache.cmd         # Cleanup script
├── clear_torch_cache.cmd     # Clear torch compile cache
├── tools/                    # Utility tools (не изучено)
├── StreamDiffusion/
│   ├── venv/                 # Python venv with torch+CUDA
│   └── StreamDiffusionTD/
│       ├── main_sdtd.py      # SD entry point
│       ├── CUDAIPCImporter.py
│       ├── CUDAIPCOutputExporter.py
│       ├── CUDAIPCWrapper.py
│       └── FrameQueueExt.py  # VRAM circular buffer (may help with freeze)
└── StreamDiffusion_0299_Torch_Compile.toe  # Alex's reference TD project (port 9957)
```

## Следующие шаги

1. **Исследовать freeze** — почему картинка не обновляется (cooker? stream? ring buffer?)
2. **Изучить Alex's start_TD.cmd и tools/** — могут содержать решения
3. **Починить exporters** — debug→print, найти реальную CUDA ошибку
4. **Тест 500+ кадров** — acceptance criteria Phase 2
