"""
TouchDesigner CUDA IPC Setup Script
====================================
Создаёт полный двунаправленный CUDA IPC сетап в TD проекте.

Использование:
    python td_setup.py                          # defaults (comp mode)
    python td_setup.py --mode flat              # flat mode (nodes in /project1)
    python td_setup.py --mode comp              # comp mode (cuda_ipc_bridge COMP)
    python td_setup.py --td-port 9955           # конкретный TD
    python td_setup.py --export-top moviefilein1 --export-channel td_out
    python td_setup.py --import-channel test_in --width 512 --height 512
    python td_setup.py --teardown                # удалить всё созданное

Что создаётся в /project1 (--mode flat):
    cuda_startup          textDAT     — добавляет C:/dev/cuda в sys.path
    cuda_export_exec      executeDAT  — TD->Python: экспортирует TOP в CUDA IPC канал
    cuda_import_callbacks textDAT     — callbacks для Script TOP (правильный синтаксис TD)
    cuda_import_top       scriptTOP   — Python->TD: получает кадры из CUDA IPC канала
    cuda_import_cooker    executeDAT  — force-cook Script TOP каждый кадр
    cuda_import_flip      flipTOP     — переворачивает изображение (GPU origin bottom-left)

Что создаётся (--mode comp):
    cuda_ipc_bridge  containerCOMP  — Bridge COMP с параметрами Pkgpath/Channelprefix/Statusoscport
      in_image         inTOP         — вход: изображение
      in_depth         inTOP         — вход: depth map
      export_img_exec  executeDAT    — экспорт in_image в CUDA IPC
      export_depth_exec executeDAT   — экспорт in_depth в CUDA IPC
      import_callbacks textDAT       — callbacks для Script TOP
      import_top       scriptTOP     — Python->TD importer
      import_cooker    executeDAT    — force-cook Script TOP
      import_flip      flipTOP       — flip (OpenGL bottom-left)
      osc_status_in    oscinCHOP     — статус по OSC
      out_result       outTOP        — выход

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

# COMP mode names
COMP_BRIDGE     = "cuda_ipc_bridge"
COMP_CONTROLLER = "sd_controller"
COMP_OUTER      = "ml_bridge"

DEFAULT_PREFIX   = "ml"
DEFAULT_OSC_PORT = 7001


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


# ── COMP mode code generators ─────────────────────────────────────────────────

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

            def _make_fc(ch_name):
                class _FC:
                    name = 'exporter_{input_name}'
                    class par:
                        class Channelname:
                            _val = ch_name
                            @staticmethod
                            def eval():
                                return _FC.par.Channelname._val
                return _FC()

            _exporter = TDCUDAIPCExporter(_make_fc(channel))
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


def make_bridge_importer_callbacks(bridge_path: str, pkg_path: str) -> str:
    """Script TOP callbacks for COMP mode importer.

    Reads channel dynamically from bridge COMP Channelprefix parameter.
    Re-creates receiver if channel changes. All TD built-ins used inline.
    """
    return f"""# Script TOP Callbacks -- CUDA IPC Receiver (COMP mode)
# me - this DAT
# scriptOp - the Script TOP operator
#
# IMPORTANT: functions defined here, not imported from package.
# TD built-ins (CUDAMemoryShape, debug) only work in DAT namespace.

import sys
if '{pkg_path}' not in sys.path:
    sys.path.insert(0, '{pkg_path}')

import struct
import time
import numpy as np
from cuda_ipc_transport.receiver import CUDAIPCReceiver

_reader = None
_last_channel = None
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
    global _reader, _last_channel, _stream, _mem_shape, _cuda, _frame_count, _reconnect_cooldown

    scriptOp.clearScriptErrors()

    try:
        prefix = op('{bridge_path}').par.Channelprefix.eval()
        channel = prefix + '_result'
    except Exception:
        channel = 'ml_result'

    if _reader is None or channel != _last_channel:
        if _reader is not None:
            try:
                _reader.disconnect()
            except Exception:
                pass
        _reader = CUDAIPCReceiver(channel)
        _reader.connect()
        _last_channel = channel
        _mem_shape = None
        debug('[bridge_import] init channel: ' + channel)

    if not _reader.is_ready():
        now = time.time()
        if now > _reconnect_cooldown:
            _reader.reconnect()
            _reconnect_cooldown = now + 2.0
        return

    if _stream is None:
        try:
            from cuda_ipc_transport.wrapper import get_cuda_runtime
            _cuda = get_cuda_runtime()
            _stream = _cuda.create_stream(0x01)
            debug('[bridge_import] stream: 0x{{:016x}}'.format(_stream.value))
        except Exception as e:
            debug('[bridge_import] stream error: {{}}'.format(e))

    ptr, size, shape = _reader.get_frame()
    if ptr is None:
        return

    if _mem_shape is None:
        m = CUDAMemoryShape()
        m.height = shape[0]
        m.width  = shape[1]
        m.numComps = shape[2]
        m.dataType = np.uint8
        _mem_shape = m

    try:
        if _stream is not None:
            s = struct.unpack('<Q', bytes(_stream))[0]
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape, stream=s)
        else:
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape)
    except Exception as e:
        debug('[bridge_import] copyCUDAMemory error: {{}}'.format(e))
        try:
            scriptOp.copyCUDAMemory(ptr, size, _mem_shape)
        except Exception:
            pass

    _frame_count += 1
    if _frame_count % 60 == 0:
        debug('[bridge_import] frame={{}}'.format(_frame_count))
"""


# ── COMP mode batch builders ───────────────────────────────────────────────────

def build_bridge_batch(parent: str, pkg_path: str, prefix: str,
                       osc_port: int, width: int, height: int) -> list:
    """Returns list of h2t batch command dicts to create cuda_ipc_bridge COMP."""

    bridge_path = f"{parent}/{COMP_BRIDGE}"

    exporter_img_code   = make_bridge_exporter_code(bridge_path, "in_image", "img", pkg_path)
    exporter_depth_code = make_bridge_exporter_code(bridge_path, "in_depth", "depth", pkg_path)
    importer_cb_code    = make_bridge_importer_callbacks(bridge_path, pkg_path)

    cooker_code = "\n".join([
        f"def onFrameStart(frame):",
        f"    op('{bridge_path}/import_top').cook(force=True)",
        f"def onFrameEnd(frame): pass",
        f"def onPlayStateChange(state): pass",
        f"def onDeviceChange(): pass",
        f"def onProjectPreSave(): pass",
        f"def onProjectPostSave(): pass",
        "",
    ])

    return [
        # ── A. Create containerCOMP + custom params ───────────────────────
        {
            "code": (
                "n = op('{parent}').create(containerCOMP, '{name}')\n"
                "page = n.appendCustomPage('ML Bridge')\n"
                "page.appendStr('Pkgpath', label='Package Path')\n"
                "n.par.Pkgpath.val = '{pkg_path}'\n"
                "page.appendStr('Channelprefix', label='Channel Prefix')\n"
                "n.par.Channelprefix.val = '{prefix}'\n"
                "page.appendInt('Statusoscport', label='Status OSC Port')\n"
                "n.par.Statusoscport.val = {osc_port}\n"
                "_result = n.path"
            ).format(
                parent=parent,
                name=COMP_BRIDGE,
                pkg_path=pkg_path,
                prefix=prefix,
                osc_port=osc_port,
            )
        },

        # ── B. Create 2x inTOP ────────────────────────────────────────────
        {
            "code": (
                "n1 = op('{bridge_path}').create(inTOP, 'in_image')\n"
                "n2 = op('{bridge_path}').create(inTOP, 'in_depth')\n"
                "_result = n1.path + ', ' + n2.path"
            ).format(bridge_path=bridge_path)
        },

        # ── C. Execute DAT — export image ─────────────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{bridge_path}').create(executeDAT, 'export_img_exec')\n"
                "n.text = code\n"
                "n.par.framestart = 1\n"
                "n.par.active = 1\n"
                "_result = n.path"
            ).format(
                code=json.dumps(exporter_img_code),
                bridge_path=bridge_path,
            )
        },

        # ── D. Execute DAT — export depth ─────────────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{bridge_path}').create(executeDAT, 'export_depth_exec')\n"
                "n.text = code\n"
                "n.par.framestart = 1\n"
                "n.par.active = 1\n"
                "_result = n.path"
            ).format(
                code=json.dumps(exporter_depth_code),
                bridge_path=bridge_path,
            )
        },

        # ── E. Text DAT — Script TOP callbacks ───────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{bridge_path}').create(textDAT, 'import_callbacks')\n"
                "n.text = code\n"
                "_result = n.path"
            ).format(
                code=json.dumps(importer_cb_code),
                bridge_path=bridge_path,
            )
        },

        # ── F. Script TOP — importer ──────────────────────────────────────
        {
            "code": (
                "t = op('{bridge_path}').create(scriptTOP, 'import_top')\n"
                "t.par.callbacks = op('{bridge_path}/import_callbacks')\n"
                "t.par.outputresolution = 'custom'\n"
                "t.par.resolutionw = {w}\n"
                "t.par.resolutionh = {h}\n"
                "_result = t.path"
            ).format(
                bridge_path=bridge_path,
                w=width,
                h=height,
            )
        },

        # ── G. Execute DAT — cooker ───────────────────────────────────────
        {
            "code": (
                "code = {code}\n"
                "n = op('{bridge_path}').create(executeDAT, 'import_cooker')\n"
                "n.text = code\n"
                "n.par.framestart = 1\n"
                "n.par.active = 1\n"
                "_result = n.path"
            ).format(
                code=json.dumps(cooker_code),
                bridge_path=bridge_path,
            )
        },

        # ── H. Flip TOP ───────────────────────────────────────────────────
        {
            "code": (
                "f = op('{bridge_path}').create(flipTOP, 'import_flip')\n"
                "f.par.flipy = 1\n"
                "f.inputConnectors[0].connect(op('{bridge_path}/import_top'))\n"
                "_result = f.path"
            ).format(bridge_path=bridge_path)
        },

        # ── I. OSC In CHOP ────────────────────────────────────────────────
        {
            "code": (
                "c = op('{bridge_path}').create(oscinCHOP, 'osc_status_in')\n"
                "c.par.port = op('{bridge_path}').par.Statusoscport.eval()\n"
                "_result = c.path"
            ).format(bridge_path=bridge_path)
        },

        # ── J. Out TOP ────────────────────────────────────────────────────
        {
            "code": (
                "o = op('{bridge_path}').create(outTOP, 'out_result')\n"
                "o.inputConnectors[0].connect(op('{bridge_path}/import_flip'))\n"
                "_result = o.path"
            ).format(bridge_path=bridge_path)
        },
    ]


def build_comp_teardown_batch(parent: str) -> list:
    """Removes the bridge COMP (destroying it removes all children)."""
    bridge_path = f"{parent}/{COMP_BRIDGE}"
    return [
        {
            "code": (
                "n = op('{bridge_path}')\n"
                "if n:\n"
                "    n.destroy()\n"
                "    _result = 'removed: {bridge_path}'\n"
                "else:\n"
                "    _result = 'not found: {bridge_path}'"
            ).format(bridge_path=bridge_path)
        }
    ]


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
    parser.add_argument("--mode", choices=["comp", "flat"], default="comp",
                        help="comp = nested COMPs (new), flat = current behavior (default: comp)")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"Channel prefix for comp mode (default: {DEFAULT_PREFIX})")
    parser.add_argument("--osc-port", type=int, default=DEFAULT_OSC_PORT,
                        help=f"OSC status port for comp mode (default: {DEFAULT_OSC_PORT})")
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
        if args.mode == "comp":
            print("[teardown] removing bridge COMP...")
            batch = build_comp_teardown_batch(PARENT)
        else:
            print("[teardown] removing nodes...")
            batch = build_teardown_batch()
    elif args.mode == "flat":
        print(f"[setup] flat mode")
        print(f"  export: op('{PARENT}/{args.export_top}') -> '{args.export_channel}'")
        print(f"  import: '{args.import_channel}' -> Script TOP {args.width}x{args.height}")
        batch = build_setup_batch(
            export_top=args.export_top,
            export_channel=args.export_channel,
            import_channel=args.import_channel,
            import_width=args.width,
            import_height=args.height,
        )
    elif args.mode == "comp":
        print(f"[setup] comp mode")
        print(f"  bridge: {PARENT}/{COMP_BRIDGE}")
        print(f"  prefix: '{args.prefix}', osc_port: {args.osc_port}")
        print(f"  resolution: {args.width}x{args.height}")
        batch = build_bridge_batch(
            parent=PARENT,
            pkg_path=CUDA_PKG_PATH,
            prefix=args.prefix,
            osc_port=args.osc_port,
            width=args.width,
            height=args.height,
        )

    ok = run_batch(batch, args.td_port)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
