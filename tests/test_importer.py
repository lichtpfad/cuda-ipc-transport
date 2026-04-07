"""Tests for td/importer.py. No GPU required."""
from unittest.mock import MagicMock, patch
import cuda_ipc_transport.td.importer as importer


def _reset_states():
    """Clear module-level state between tests."""
    importer._states.clear()


def test_get_channel_name_default():
    """_get_channel_name returns default when par.Channelname missing."""
    scriptOp = MagicMock()
    del scriptOp.par  # Remove par attribute
    result = importer._get_channel_name(scriptOp)
    assert result == "sd_to_td_ipc"


def test_get_channel_name_custom():
    """_get_channel_name returns custom value from par.Channelname."""
    scriptOp = MagicMock()
    scriptOp.par.Channelname.eval.return_value = "my_custom_channel"
    result = importer._get_channel_name(scriptOp)
    assert result == "my_custom_channel"


def test_get_channel_name_eval_fails():
    """_get_channel_name returns default if eval() raises TypeError."""
    scriptOp = MagicMock()
    scriptOp.par.Channelname.eval.side_effect = TypeError("eval not supported")
    result = importer._get_channel_name(scriptOp)
    assert result == "sd_to_td_ipc"


def test_state_dataclass_defaults():
    """_State() has correct default values."""
    state = importer._State()
    assert state.reader is None
    assert state.stream is None
    assert state.mem_shape is None
    assert state.cuda is None
    assert state.frame_count == 0
    assert state.reconnect_cooldown == 0.0
    assert state.channel_name == ""


def test_get_state_creates_new():
    """_get_state() creates new _State for new operator."""
    _reset_states()
    scriptOp = MagicMock()
    state = importer._get_state(scriptOp)
    assert state is not None
    assert isinstance(state, importer._State)
    assert state.frame_count == 0


def test_get_state_reuses():
    """_get_state() returns same _State for same operator."""
    _reset_states()
    scriptOp = MagicMock()
    state1 = importer._get_state(scriptOp)
    state1.frame_count = 42
    state2 = importer._get_state(scriptOp)
    assert state1 is state2
    assert state2.frame_count == 42


def test_get_state_independent():
    """Different operators have independent _State instances."""
    _reset_states()
    op1 = MagicMock()
    op2 = MagicMock()
    state1 = importer._get_state(op1)
    state2 = importer._get_state(op2)
    assert state1 is not state2
    state1.frame_count = 10
    state2.frame_count = 20
    assert state1.frame_count == 10
    assert state2.frame_count == 20
