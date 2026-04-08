# sd_controller parexec — sends OSC to SD on parameter change
# Uses print() not debug() (Execute DAT context).

import json
import sys

_client = None
_SD_OSC_PORT = 7187

_PAR_OSC_MAP = {
    'Seed': ('/seed_list', 'seed'),
    'Prompt': ('/prompt_list', 'prompt'),
    'Negativeprompt': ('/negative_prompt', 'str'),
    'Delta': ('/delta', 'float'),
    'Guidancescale': ('/guidance_scale', 'float'),
    'Usecontrolnet': ('/use_controlnet', 'int'),
    'Maxfps': ('/max_fps', 'int'),
}


def _get_client():
    global _client
    if _client is None:
        venv_sp = 'C:/dev/cuda/.venv/Lib/site-packages'
        if venv_sp not in sys.path:
            sys.path.insert(0, venv_sp)
        from pythonosc import udp_client
        _client = udp_client.SimpleUDPClient('127.0.0.1', _SD_OSC_PORT)
    return _client


def onValueChange(par, prev):
    name = par.name
    if name not in _PAR_OSC_MAP:
        return

    address, encoder = _PAR_OSC_MAP[name]
    val = par.eval()

    try:
        c = _get_client()
        if encoder == 'prompt':
            c.send_message(address, json.dumps([[str(val), 1.0]]))
        elif encoder == 'seed':
            c.send_message(address, json.dumps([[int(val), 1.0]]))
        elif encoder == 'str':
            c.send_message(address, str(val))
        elif encoder == 'float':
            c.send_message(address, float(val))
        elif encoder == 'int':
            c.send_message(address, int(val))
        print('[parexec] {} = {}'.format(address, val))
    except Exception as e:
        print('[parexec] ERROR: {}'.format(e))


def onPulse(par):
    name = par.name
    try:
        c = _get_client()
        if name == 'Pause':
            c.send_message('/pause', 1)
        elif name == 'Play':
            c.send_message('/play', 1)
        print('[parexec] /{}'.format(name.lower()))
    except Exception as e:
        print('[parexec] pulse ERROR: {}'.format(e))
