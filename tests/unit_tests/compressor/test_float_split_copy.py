import pytest
import torch

from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not fs_sp.is_available()),
    reason="CUDA and float_split_stride_pin extension are required",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1024,), (128, 256), (8, 64, 128)])
def test_split_copy_roundtrip(dtype, shape):
    torch.manual_seed(1234)
    device = torch.device("cuda")
    source = torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
    # Some MetaX/PyTorch unit-test environments hang while creating a new
    # torch.cuda.Stream(); benchmark/training code still covers non-default streams.
    stream = torch.cuda.current_stream()
    exp_host, sm_bits, exp_gpu = fs_sp.split_copy(source, stream.cuda_stream)
    event = stream.record_event()

    event.synchronize()
    restored = fs_sp.merge(
        exp_host,
        sm_bits,
        list(source.shape),
        list(source.stride()),
        source.storage_offset(),
        source.dtype,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(restored, source)
    assert exp_host.device.type == "cpu"
    assert exp_host.is_pinned()
    assert sm_bits.device.type == "cuda"
    assert exp_gpu.device.type == "cuda"
