import torch
from torch import FloatTensor

from MiniFL.message import Message
from MiniFL.utils import get_num_bits

from .basic import RandKContractiveCompressor, TopKCompressor
from .interfaces import Compressor


def _rounding(x, stochastic=False, minimum_stochastic_distance=0.2):
    if stochastic:
        x_floor = x.floor()
        th = x - x_floor
        if minimum_stochastic_distance > 0:
            th[th < minimum_stochastic_distance] = 0.0
            th[th > 1 - minimum_stochastic_distance] = 1.0
        pr = torch.rand_like(x)
        x_floor += pr < th
        return x_floor
    else:
        return x.round()


def _compress_nbits(x, bits, scale_method="max", scale_dims=(0, 1), stochastic=False, minimum_stochastic_distance=0.2):

    fbits = bits - 1

    if scale_method == "max":
        # issue: sensitive to outlier points
        scale = x.abs().amax(scale_dims, keepdims=True)
    elif scale_method == "l2":
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean(scale_dims, keepdims=True).sqrt() * 2
    else:
        raise Exception("unkonwn scale method.")
    # fp16 should be enough
    scale = scale.half()
    x = x / (scale + 1e-6)

    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1 << fbits)
    clip_max = (1 << fbits) - 1

    x = _rounding(x, stochastic=stochastic, minimum_stochastic_distance=minimum_stochastic_distance)
    x = x.clip(clip_min, clip_max)

    x = x - clip_min
    x = x.type(torch.uint8)

    return x, scale


def _decompress_nbits(x, scale, bits):

    fbits = bits - 1

    clip_min = -(1 << fbits)
    clip_max = (1 << fbits) - 1

    x = x.float() + clip_min

    x = x / (clip_max + 1) * scale

    return x


class CocktailCompressor(Compressor):
    def __init__(self, size: int, rand_p: float, top_p: float, bits, scale_method="max", seed=0):
        super().__init__(size)
        self.rand_k = RandKContractiveCompressor(size, p=rand_p, seed=seed)
        self.top_k = TopKCompressor(size, p=top_p)
        self.bits = bits
        self.scale_method = scale_method

    def compress(self, x: FloatTensor) -> Message:
        rand_k_msg = self.rand_k.compress(x)
        rand_indexes, rand_values = rand_k_msg.data
        top_k_msg = self.top_k.compress(rand_values)
        top_indexes, top_values = top_k_msg.data
        quantized_values, scale = _compress_nbits(
            top_values, self.bits, scale_method=self.scale_method, scale_dims=(0,)
        )

        return Message(
            data=self.__decompress(rand_indexes, top_indexes, quantized_values, rand_values.shape, x.shape, scale),
            size=top_k_msg.size - top_values.numel() * get_num_bits(top_values.dtype) + top_values.numel() * self.bits,
        )

    def __decompress(self, rand_indexes, top_indexes, quantized_values, top_shape, rand_shape, scale) -> FloatTensor:
        top_values = _decompress_nbits(quantized_values, scale, self.bits)
        rand_values = self.top_k.decompress(top_indexes, top_values, top_shape)
        x = self.rand_k.decompress(rand_indexes, rand_values, rand_shape)
        return x
