"""
TouchDesigner CUDA IPC Setup Script
====================================
Создаёт полный двунаправленный CUDA IPC сетап в TD проекте.

Использование:
    python td_setup.py                          # defaults
    python td_setup.py --td-port 9955           # конкретный TD
    python td_setup.py --export-top moviefilein1 --export-channel td_out
    python td_setup.py --import-channel test_in --width 512 --height 512
    python td_setup.py --teardown                # удалить всё созданное

Что создаётся в /project1:
    cuda_startup          textDAT     — добавляет C:/dev/cuda в sys.path
    cuda_export_exec      executeDAT  — TD->Python: экспортирует TOP в CUDA IPC канал
    cuda_import_callbacks textDAT     — callbacks для Script TOP (правильный синтаксис TD)
    cuda_import_top       scriptTOP   — Python->TD: получает кадры из CUDA IPC канала
    cuda_import_cooker    executeDAT  — force-cook Script TOP каждый кадр
    cuda_import_flip      flipTOP     — переворачивает изображение (GPU origin bottom-left)

Уроки (важно для будущих итераций):
    1. Script TOP callbacks ОПРЕДЕЛЯТЬ в DAT, не импортировать из пакета.
       TD built-ins (CUDAMemoryShape, debug, op) доступны только в namespace DAT.
    2. Script TOP НЕ кукает сам — нужен force-cook через Execute DAT каждый кадр.
    3. copyCUDAMemory дает перевернутый кадр (OpenGL origin внизу) — нужен Flip TOP.
    4. CUDA 13: cudart64_13.dll живёт в bin/x64/, не в bin/.
    5. h2t batch: op.attr = val нельзя использовать как _result — нужен отдельный _result = 'ok'.
    6. par.outputresolution = 'custom' (строка), не integer.
"""

import argparse
import json
import subprocess
import sys


# ── Defaults ──────────────────────────────────────────────────────────────────
CUDA_PKG_PATH = "C:/dev/cuda"
PARENT = "/project1"
TD_PORT = 9955

# Node names (все с префиксом cuda_ чтобы не конфликтовать)
NODE_STARTUP    = "cuda_startup"
NODE_EXPORT     = "cuda_export_exec"
NODE_IMPORT_CB  = "cuda_import_callbacks"
NODE_IMPORT_TOP = "cuda_import_top"
NODE_COOKER     = "cuda_import_cooker"
NODE_FLIP       = "cuda_import_flip"

ALL_NODES = [NODE_STARTUP, NODE_EXPORT, NODE_IMPORT_CB, NODE_IMPORT_TOP, NODE_COOKER, NODE_FLIP]


# ── Код для нод ───────────────────────────────────────────────────────────────

def make_startup_code(pkg_path: str) -> str:
    return f"""import sys
if '{pkg_path}' not in sys.path:
    sys.path.insert(0, '{pkg_path}')
print('[cuda_startup] sys.path ok')
"""


def make_exporter_code(export_top_path: str, channel: str) -> str:
    """Execute DAT — onFrameStart экспортирует TOP в CUDA IPC канал."""
    return f"""# Execute DAT — TD->Python CUDA IPC Exporter
# Экспортирует '{export_top_path}' в канал '{channel}'
# Урок: TDCUDAIPCExporter.__init__ читает ownerComp.par.Channelname,
#       при AttributeError fallback = 'td_{{ownerComp.name}}'

import sys
if '{CUDA_PKG_PATH}' not in sys.path:
    sys.path.insert(0, '{CUDA_PKG_PATH}')

_exporter = None


def onFrameStart(frame):
    global _exporter
    try:
        if _exporter is None:
            from cuda_ipc_transport.td.exporter import TDCUDAIPCExporter

            class _FC:
                name = 'exporter'
                class par:
                    class Channelname:
                        @staticmethod
                        def eval():
                            return '{channel}'

            _exporter = TDCUDAIPCExporter(_FC())
            debug('[cuda_export] initialized: {channel}')

        top = op('{export_top_path}')
        if top:
            _exporter.ExportFrame(top)
    except Exception as e:
        debug('[cuda_export] error: {{}}'.format(e))


def onFrameEnd(frame): pass
def onPlayStateChange(state): pass
def onDeviceChange(): pass
def onProjectPreSave(): pass
def onProjectPostSave(): pass
"""


def make_importer_callbacks(channel: str) -> str:
    """
    Script TOP callbacks — ОПРЕДЕЛЯТЬ здесь, не импортировать из пакета!
    TD built-ins (CUDAMemoryShape, debug) доступны только в namespace этого DAT.
    """
    return f"""# Script TOP Callbacks — CUDA IPC Receiver
# me - this DAT
# scriptOp - the Script TOP operator
#
# ВАЖНО: функции определены здесь, а не импортированы из пакета.
# TD built-ins (CUDAMemoryShape, debug) работают только в namespace DAT.

import sys
if '{CUDA_PKG_PATH}' not in sys.path:
    sys.path.insert(0, '{CUDA_PKG_PATH}')

import struct
import time
import numpy as np
from cuda_ipc_transport.receiver import CUDAIPCReceiver

_reader = None
_stream = None
_mem_shape = None
_cuda = None
_frame_count = 0
_reconnect_cooldown = 0.0


def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return


def onCook(scriptOp):
    global _reader, _stream, _mem_shape, _cuda, _frame_count, _reconnect_cooldown

    scriptOp.clearScriptErrors()

    # Читаем имя канала из параметра оператора (с fallback)
    if _reader is None:
        try:
            channel = scriptOp.par.Channelname.eval()
        except AttributeError:
            channel = '{channel}'
        _reader = CUDAIPCReceiver(channel)
        _reader.connect()

    if not _reader.is_ready():
        now = time.time()
        if now > _reconnect_cooldown:
            _reader.reconnect()
            _reconnect_cooldown = now + 2.0
        return

    # CUDA stream — создаём один раз (non-blocking)
    if _stream is None:
        try:
            from cuda_ipc_transport.wrapper import get_cuda_runtime
            _cuda = get_cuda_runtime()
            _stream = _cuda.create_stream(0x01)
            debug('[cuda_import] stream: 0x{{:016x}}'.format(_stream.value))
        except Exception as e:
            debug('[cuda_import] stream error: {{}}'.format(e))

    ptr, size, shape = _reader.get_frame()
    if ptr is None:
        return

    # CUDAMemoryShape — кешируем, не создаём каждый кадр
    if _mem_shape is None:
        m = CUDAMemoryShape()
        m.height = shape[0]
        m.width  = shape[1]
        m.numComps = shape[2]
        m.dataType = np.uint8
        _mem_shape = m

    # copyCUDAMemory — передаём stream как uint64 через struct.unpack
    # (bytes(c_uint64) работает через буферный протокол ctypes)
    try:
        if _stream is not None:
            s = struct.unpack('<Q', bytes(_stream))[0]
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape, stream=s)
        else:
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape)
    except Exception as e:
        debug('[cuda_import] copyCUDAMemory error: {{}}'.format(e))
        try:
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape)
        except Exception:
            pass

    _frame_count += 1
    if _frame_count % 60 == 0:
        debug('[cuda_import] frame={{}}'.format(_frame_count))
"""


def make_cooker_batch(parent: str, name: str, top: str) -> str:
    cooker_code = "\n".join([
        f"def onFrameStart(frame):",
        f"    op('{parent}/{top}').cook(force=True)",
        f"def onFrameEnd(frame): pass",
        f"def onPlayStateChange(state): pass",
        f"def onDeviceChange(): pass",
        f"def onProjectPreSave(): pass",
        f"def onProjectPostSave(): pass",
        "",
    ])
    return (
        f"code = {json.dumps(cooker_code)}\n"
        f"n = op('{parent}').create(executeDAT, '{name}')\n"
        f"n.text = code\n"
        f"n.par.framestart = 1\n"
        f"n.par.active = 1\n"
        f"_result = n.path"
    )


# ── Batch builder ─────────────────────────────────────────────────────────────

def build_setup_batch(
    export_top: str,
    export_channel: str,
    import_channel: str,
    import_width: int,
    import_height: int,
) -> list:
    """Возвращает список команд для h2t td batch."""

    startup_code   = make_startup_code(CUDA_PKG_PATH)
    exporter_code  = make_exporter_code(f"{PARENT}/{export_top}", export_channel)
    importer_code  = make_importer_callbacks(import_channel)

    return [
        # ── 0. Удалить старые ноды если есть ─────────────────────────────
        {
            "code": (
                "for name in {nodes}:\n"
                "    n = op('{parent}/' + name)\n"
                "    if n: n.destroy()\n"
                "_result = 'cleaned'"
            ).format(nodes=json.dumps(ALL_NODES), parent=PARENT)
        },

        # ── 1. Startup DAT — sys.path ─────────────────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{parent}').create(textDAT, '{name}')\n"
                "n.text = code\n"
                "n.run()\n"
                "_result = n.path"
            ).format(
                code=json.dumps(startup_code),
                parent=PARENT,
                name=NODE_STARTUP,
            )
        },

        # ── 2. Execute DAT — TD→Python exporter ──────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{parent}').create(executeDAT, '{name}')\n"
                "n.text = code\n"
                "n.par.framestart = 1\n"
                "n.par.active = 1\n"
                "_result = n.path"
            ).format(
                code=json.dumps(exporter_code),
                parent=PARENT,
                name=NODE_EXPORT,
            )
        },

        # ── 3. Text DAT — Script TOP callbacks ───────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{parent}').create(textDAT, '{name}')\n"
                "n.text = code\n"
                "_result = n.path"
            ).format(
                code=json.dumps(importer_code),
                parent=PARENT,
                name=NODE_IMPORT_CB,
            )
        },

        # ── 4. Script TOP — Python→TD importer ───────────────────────────
        {
            "code": (
                "t = op('{parent}').create(scriptTOP, '{name}')\n"
                "t.par.callbacks = op('{parent}/{cb}')\n"
                "t.par.outputresolution = 'custom'\n"   # строка, не int!
                "t.par.resolutionw = {w}\n"
                "t.par.resolutionh = {h}\n"
                "page = t.appendCustomPage('CUDA IPC')\n"
                "page.appendStr('Channelname', label='Channel Name')\n"
                "t.par.Channelname.val = '{channel}'\n"
                "_result = t.path"
            ).format(
                parent=PARENT,
                name=NODE_IMPORT_TOP,
                cb=NODE_IMPORT_CB,
                w=import_width,
                h=import_height,
                channel=import_channel,
            )
        },

        # ── 5. Cooker — force-cook Script TOP каждый кадр ────────────────
        # Script TOP сам НЕ кукает — нужен явный force-cook
        {
            "code": make_cooker_batch(PARENT, NODE_COOKER, NODE_IMPORT_TOP)
        },

        # ── 6. Flip TOP — copyCUDAMemory дает перевернутый кадр ──────────
        # OpenGL origin = bottom-left, CUDA buffer = top-left → нужен flip
        {
            "code": (
                "f = op('{parent}').create(flipTOP, '{name}')\n"
                "f.par.flipy = 1\n"
                "f.inputConnectors[0].connect(op('{parent}/{top}'))\n"
                "_result = f.path"
            ).format(
                parent=PARENT,
                name=NODE_FLIP,
                top=NODE_IMPORT_TOP,
            )
        },
    ]


def build_teardown_batch() -> list:
    return [
        {
            "code": (
                "removed = []\n"
                "for name in {nodes}:\n"
                "    n = op('{parent}/' + name)\n"
                "    if n:\n"
                "        n.destroy()\n"
                "        removed.append(name)\n"
                "_result = 'removed: ' + str(removed)"
            ).format(nodes=json.dumps(ALL_NODES), parent=PARENT)
        }
    ]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_batch(batch: list, td_port: int) -> bool:
    tmp = "/tmp/td_cuda_batch.json"
    with open(tmp, "w") as f:
        json.dump(batch, f, indent=2)

    result = subprocess.run(
        ["h2t", "td", "batch", tmp, "--td-port", str(td_port)],
        capture_output=True,
        text=True,
    )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Raw output:", result.stdout)
        return False

    errors = data.get("errors", [])
    if errors:
        print(f"[ERRORS] {len(errors)} failed:")
        for e in errors:
            print(f"  [{e['index']}] {e['error']}")
        return False

    elapsed = data.get("elapsed_ms", 0)
    print(f"[OK] {len(batch)} commands, {elapsed}ms")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Setup CUDA IPC bidirectional pipeline in TouchDesigner"
    )
    parser.add_argument("--td-port", type=int, default=TD_PORT,
                        help=f"TD WebServer port (default: {TD_PORT})")
    parser.add_argument("--export-top", default="moviefilein1",
                        help="TOP operator name to export TD->Python (default: moviefilein1)")
    parser.add_argument("--export-channel", default="td_test_out",
                        help="Channel name for TD->Python (default: td_test_out)")
    parser.add_argument("--import-channel", default="test_in",
                        help="Channel name for Python->TD (default: test_in)")
    parser.add_argument("--width", type=int, default=512,
                        help="Script TOP width (default: 512)")
    parser.add_argument("--height", type=int, default=512,
                        help="Script TOP height (default: 512)")
    parser.add_argument("--teardown", action="store_true",
                        help="Remove all created nodes")
    args = parser.parse_args()

    # Ping TD
    ping = subprocess.run(
        ["h2t", "td", "ping", "--td-port", str(args.td_port)],
        capture_output=True, text=True
    )
    if "true" not in ping.stdout.lower():
        print(f"[ERROR] TD not connected on port {args.td_port}")
        print("  Run: h2t td reconnect --td-port", args.td_port)
        sys.exit(1)

    print(f"[TD] connected on port {args.td_port}")

    if args.teardown:
        print("[teardown] removing nodes...")
        batch = build_teardown_batch()
    else:
        print(f"[setup]")
        print(f"  export: op('{PARENT}/{args.export_top}') -> '{args.export_channel}'")
        print(f"  import: '{args.import_channel}' -> Script TOP {args.width}x{args.height}")
        batch = build_setup_batch(
            export_top=args.export_top,
            export_channel=args.export_channel,
            import_channel=args.import_channel,
            import_width=args.width,
            import_height=args.height,
        )

    ok = run_batch(batch, args.td_port)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
