import pytest
import torch

from MiniFL.compressors.basic import IdentityCompressor, PermKUnbiasedCompressor, RandPUnbiasedCompressor
from MiniFL.compressors.interfaces import UnbiasedCompressor


@pytest.mark.parametrize(
    "compressor_cls_and_kwargs",
    [
        (IdentityCompressor, {}),
        (RandPUnbiasedCompressor, {"p": 0.5}),
        (PermKUnbiasedCompressor, {"rank": 0, "world_size": 5}),
    ],
)
def test_unbiased(compressor_cls_and_kwargs):
    SIZE = 10
    NUM = 10000
    torch.manual_seed(0)

    compressor_cls, kwargs = compressor_cls_and_kwargs
    c = compressor_cls(**kwargs)
    assert isinstance(c, UnbiasedCompressor)

    expected_mean = torch.ones(SIZE) / 2
    mean = torch.zeros(SIZE)
    for _ in range(NUM):
        mean += c.decompress(c.compress(torch.rand(SIZE)))
    mean = mean / NUM

    torch.testing.assert_close(mean, expected_mean, atol=0.05, rtol=0.05)
