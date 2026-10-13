"""Which NVRTC releases the CUDA backend warns about (no GPU required)."""

from tack.runtime.nvrtc_version import old_nvrtc_message


def test_nvrtc_12_is_named_with_the_remedy():
    message = old_nvrtc_message(12, 0)
    assert message.startswith("NVRTC 12.0 is older than the NVRTC 13")
    assert "nvidia-cuda-nvrtc" in message


def test_nvrtc_13_and_later_pass():
    for version in [(13, 0), (13, 2), (13, 4), (14, 0)]:
        assert old_nvrtc_message(*version) is None
