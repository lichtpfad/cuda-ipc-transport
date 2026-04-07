"""Tests for harness CLI argument parsing -- no CUDA required."""
from cuda_ipc_transport.harness import _resolve_channel


class TestResolveChannel:
    """Test _resolve_channel helper."""

    def test_default_no_prefix(self):
        assert _resolve_channel("cuda_ipc_test", None) == "cuda_ipc_test"

    def test_prefix_creates_result_channel(self):
        assert _resolve_channel("cuda_ipc_test", "ml") == "ml_out_ipc"

    def test_prefix_overrides_channel(self):
        assert _resolve_channel("custom_name", "ml") == "ml_out_ipc"

    def test_custom_prefix(self):
        assert _resolve_channel("x", "sd_v2") == "sd_v2_out_ipc"

    def test_empty_string_prefix_is_falsy(self):
        assert _resolve_channel("my_chan", "") == "my_chan"
