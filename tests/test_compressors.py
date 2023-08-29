import pytest
import torch

from MiniFL import compressors as compressors


@pytest.mark.parametrize(
    "compressor_cls_and_kwargs",
    [
        (compressors.IdentityCompressor, {}),
        (compressors.RandKUnbiasedCompressor, {"p": 0.5}),
        (compressors.PermKUnbiasedCompressor, {"rank": 0, "world_size": 5}),
        (compressors.EdenUnbiasedCompressor, {"bits": 1.2}),
        (compressors.EdenUnbiasedCompressor, {"bits": 0.9}),
    ],
)
def test_unbiased(compressor_cls_and_kwargs):
    SIZE = 10
    NUM = 10000
    torch.manual_seed(0)

    compressor_cls, kwargs = compressor_cls_and_kwargs
    c = compressor_cls(SIZE, **kwargs)
    assert isinstance(c, compressors.UnbiasedCompressor)

    expected_mean = torch.ones(SIZE) / 2
    mean = torch.zeros(SIZE)
    for _ in range(NUM):
        mean += c.compress(torch.rand(SIZE)).data
    mean = mean / NUM

    torch.testing.assert_close(mean, expected_mean, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("world_size", [2, 3])
@pytest.mark.parametrize(
    "compressor_cls_and_kwargs",
    [
        (compressors.PermKUnbiasedCompressor, {}),
        (compressors.CorrelatedQuantizer, {"num_levels": 1}),
    ],
)
def test_correlated_unbiased(world_size: int, compressor_cls_and_kwargs):
    SIZE = 10
    NUM = 10000
    torch.manual_seed(0)

    compressor_cls, kwargs = compressor_cls_and_kwargs
    assert issubclass(
        compressor_cls, compressors.InputVarianceCompressor
    ), f"{compressor_cls} is not UnbiasedCompressor"
    cs = [compressor_cls(SIZE, world_size=world_size, rank=i, **kwargs) for i in range(world_size)]

    expected_mean = torch.ones(SIZE) / 2
    mean = torch.zeros(SIZE)
    for _ in range(NUM):
        for c in cs:
            mean += c.compress(torch.rand(SIZE)).data
    mean = mean / NUM / world_size

    torch.testing.assert_close(mean, expected_mean, atol=0.05, rtol=0.05)
