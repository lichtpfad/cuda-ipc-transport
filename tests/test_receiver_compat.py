"""Tests for get_reader() backward-compat shim. No GPU required."""
from unittest.mock import patch, MagicMock
import cuda_ipc_transport.receiver as mod


def _reset_singleton():
    mod._default_reader = None


def test_get_reader_default_channel():
    """get_reader() without args uses 'sd_to_td_ipc' channel name."""
    _reset_singleton()
    mock_receiver = MagicMock()
    with patch.object(mod, 'CUDAIPCReceiver', return_value=mock_receiver) as MockClass:
        from cuda_ipc_transport.receiver import get_reader
        get_reader()
        MockClass.assert_called_once_with("sd_to_td_ipc")
        mock_receiver.connect.assert_called_once()


def test_get_reader_custom_channel():
    """get_reader('my_channel') uses custom channel name."""
    _reset_singleton()
    mock_receiver = MagicMock()
    with patch.object(mod, 'CUDAIPCReceiver', return_value=mock_receiver) as MockClass:
        from cuda_ipc_transport.receiver import get_reader
        get_reader("my_channel")
        MockClass.assert_called_once_with("my_channel")


def test_get_reader_singleton():
    """Second call returns same instance."""
    _reset_singleton()
    mock_receiver = MagicMock()
    with patch.object(mod, 'CUDAIPCReceiver', return_value=mock_receiver):
        from cuda_ipc_transport.receiver import get_reader
        r1 = get_reader()
        r2 = get_reader()
        assert r1 is r2
        assert mock_receiver.connect.call_count == 1  # connect only called once
