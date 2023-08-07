import torch
from torch import FloatTensor

from MiniFL.message import Message
from MiniFL.utils import get_num_bits

from .basic import RandPBiasedCompressor, TopPBiasedCompressor
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
    def __init__(self, rand_p: float, top_p: float, bits, scale_method="max", seed=0):
        self.rand_p = RandPBiasedCompressor(p=rand_p, seed=seed)
        self.top_p = TopPBiasedCompressor(p=top_p)
        self.bits = bits
        self.scale_method = scale_method

    def compress(self, x: FloatTensor) -> Message:
        rand_p_msg = self.rand_p.compress(x)
        rand_indexes, rand_values = rand_p_msg.data
        top_p_msg = self.top_p.compress(rand_values)
        top_indexes, top_values = top_p_msg.data
        quantized_values, scale = _compress_nbits(
            top_values, self.bits, scale_method=self.scale_method, scale_dims=(0,)
        )

        return Message(
            data=(rand_indexes, top_indexes, quantized_values),
            size=top_p_msg.size - top_values.numel() * get_num_bits(top_values.dtype) + top_values.numel() * self.bits,
            metadata={"rand_shape": x.shape, "top_shape": rand_values.shape, "scale": scale},
        )

    def decompress(self, msg: Message) -> FloatTensor:
        rand_indexes, top_indexes, quantized_values = msg.data
        top_values = _decompress_nbits(quantized_values, msg.metadata["scale"], self.bits)
        rand_values = self.top_p.decompress(
            Message(data=(top_indexes, top_values), size=0, metadata={"shape": msg.metadata["top_shape"]})
        )
        x = self.rand_p.decompress(
            Message(data=(rand_indexes, rand_values), size=0, metadata={"shape": msg.metadata["rand_shape"]})
        )
        return x
