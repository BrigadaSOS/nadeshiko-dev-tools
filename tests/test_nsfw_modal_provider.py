import pytest

modal = pytest.importorskip("modal")  # the runner module imports modal at load time

from nadeshiko_dev_tools.nsfw_tagger import modal_runner  # noqa: E402


def test_require_gpu_provider_accepts_cuda_first():
    assert (
        modal_runner.require_gpu_provider(["CUDAExecutionProvider", "CPUExecutionProvider"])
        == "CUDAExecutionProvider"
    )


@pytest.mark.parametrize("providers", [["CPUExecutionProvider"], [], ["TensorrtExecutionProvider"]])
def test_require_gpu_provider_rejects_fallback(providers):
    with pytest.raises(RuntimeError, match="not on the GPU"):
        modal_runner.require_gpu_provider(providers)


def test_image_pins_a_cuda12_onnxruntime_build():
    major, minor, _ = (int(x) for x in modal_runner.ONNXRUNTIME_GPU_VERSION.split("."))
    assert (major, minor) < (1, 25), (
        "newer onnxruntime-gpu builds need CUDA 13; update the base image first"
    )
