import math
from abc import abstractmethod
from functools import cache, lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import inv
from scipy.special import erf
from scipy.stats import ortho_group
from torch import FloatTensor

from MiniFL.message import Message

from .basic import RandKBaseCompressor
from .interfaces import Compressor, ContractiveCompressor, InputVarianceCompressor, UnbiasedCompressor

QuantizationType = Literal["max_lloyd", "ee"]


class EdenBaseCompressor(Compressor):
    def __init__(
        self,
        size: int,
        bits: float,
        real_rotation=False,
        world_size: int = None,
        q_type: QuantizationType = "max_lloyd",
        device="cpu",
        seed=0,
    ):
        super().__init__(size=size)
        self.hadamard_size = 2 ** (math.floor(math.log2(self.size)) + 1)
        self.bits = bits
        self.world_size = world_size

        # Sparcification
        if self.bits < 1:
            self.p = bits
            self.random_k = RandKBaseCompressor(self.size if real_rotation else self.hadamard_size, p=self.p, seed=seed)
            self.bits = 1
        else:
            self.p = 1

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Rotation
        self.real_rotation = real_rotation

        # Quantization
        self.centroids, self.boundaries = get_all_quantization_constants_tensors(q_type, device)
        bits_frac, bits_low = math.modf(self.bits)
        if math.isclose(bits_frac, 0):
            self.fractional_bits = False
        else:
            self.fractional_bits = True
            self.bits_low = bits_low
            self.bits_high = bits_low + 1
            self.bits_frac = bits_frac

    @abstractmethod
    def get_scale(self, x: FloatTensor, unscaled_centers_vec: FloatTensor) -> float:
        pass

    def compress(self, x: FloatTensor) -> Message:
        compression_result = self.inner_compress(x)
        if compression_result["is_zero"]:
            bits = 32
        elif self.fractional_bits:
            bits = (
                compression_result["assignments"][0].numel() * self.bits_low
                + compression_result["assignments"][1].numel() * self.bits_high
                + 32
            )
        else:
            bits = compression_result["assignments"].numel() * self.bits + 32
        return Message(
            data=self.inner_decompress(compression_result),
            size=bits,
        )

    def inner_compress(self, x: FloatTensor):
        compression_result = {}
        if torch.count_nonzero(x) == 0:
            compression_result["is_zero"] = True
            compression_result["original_shape"] = x.shape
            return compression_result
        else:
            compression_result["is_zero"] = False

        # Flatten
        original_shape = x.shape
        compression_result["original_shape"] = original_shape
        data = x.flatten()

        # Rotate
        if self.real_rotation:
            pre_rotation_size = data.shape[0]
            compression_result["pre_rotation_size"] = pre_rotation_size
            rotation_seed = self.generator.seed() % 2**32  # TODO: get_state()
            compression_result["rotation_seed"] = rotation_seed
            np.random.seed(seed=rotation_seed)
            data = torch.from_numpy(ortho_group.rvs(pre_rotation_size) @ data.numpy()).to(data.device).to(data.dtype)
        else:
            if self.size & (self.size - 1) != 0:
                data = F.pad(data, (0, self.hadamard_size - self.size))

            rotation_seed = self.generator.get_state()
            compression_result["rotation_seed"] = rotation_seed
            data = randomized_hadamard_transform_(data, self.generator)

        # Sparcify
        if self.p < 1:
            random_indexes, data, flattened_shape = self.random_k.inner_compress(x=data)
            compression_result["flattened_shape"] = flattened_shape
            compression_result["random_indexes"] = random_indexes

        # Quantize
        d = data.numel()
        normalized_data = data * math.sqrt(d) / l2(data)

        if self.fractional_bits:
            unquantized_shape = data.shape
            compression_result["unquantized_shape"] = unquantized_shape
            quantization_seed = self.generator.get_state()
            compression_result["quantization_seed"] = quantization_seed
            mask = bernoulli_mask(data.shape, data.device, self.bits_frac, self.generator)

            data_low, data_high = mask_split(normalized_data, mask)

            assignments_low = torch.bucketize(data_low, self.boundaries[self.bits_low])
            assignments_high = torch.bucketize(data_high, self.boundaries[self.bits_high])

            assignments = (assignments_low, assignments_high)

            unscaled_centers_vec_low = torch.take(self.centroids[self.bits_low], assignments_low)
            unscaled_centers_vec_high = torch.take(self.centroids[self.bits_high], assignments_high)

            unscaled_centers_vec = mask_combine(unscaled_centers_vec_low, unscaled_centers_vec_high, mask)
        else:
            assignments = torch.bucketize(normalized_data, self.boundaries[self.bits])
            unscaled_centers_vec = torch.take(self.centroids[self.bits], assignments)

        scale = self.get_scale(data, unscaled_centers_vec)
        compression_result["scale"] = scale
        compression_result["assignments"] = assignments
        return compression_result

    def inner_decompress(self, compression_result) -> FloatTensor:
        if compression_result["is_zero"]:
            return torch.zeros(compression_result["original_shape"], dtype=torch.float32)

        # Dequantize
        assignments = compression_result["assignments"]
        scale = compression_result["scale"]
        if self.fractional_bits:
            quantization_seed = compression_result["quantization_seed"]
            unquantized_shape = compression_result["unquantized_shape"]
            assignments_low, assignments_high = assignments

            unscaled_centers_vec_low = torch.take(self.centroids[self.bits_low], assignments_low)
            unscaled_centers_vec_high = torch.take(self.centroids[self.bits_high], assignments_high)

            mask = bernoulli_mask(
                unquantized_shape, assignments_low.device, self.bits_frac, self.generator.set_state(quantization_seed)
            )
            unscaled_centers_vec = mask_combine(unscaled_centers_vec_low, unscaled_centers_vec_high, mask)
        else:
            unscaled_centers_vec = torch.take(self.centroids[self.bits], assignments)
        data = scale * unscaled_centers_vec

        # Unsparsify
        if self.p < 1:
            random_indexes = compression_result["random_indexes"]
            flattened_shape = compression_result["flattened_shape"]
            data = self.random_k.inner_decompress(random_indexes, data, flattened_shape)

        # Rotate back
        if self.real_rotation:
            rotation_seed = compression_result["rotation_seed"]
            pre_rotation_size = compression_result["pre_rotation_size"]
            np.random.seed(seed=rotation_seed)
            data = torch.from_numpy(inv(ortho_group.rvs(pre_rotation_size)) @ data.numpy()).to(data.device)
        else:
            rotation_seed = compression_result["rotation_seed"]
            data = inverse_randomized_hadamard_transform_(data, self.generator.set_state(rotation_seed))[: self.size]

        # Unflatten
        original_shape = compression_result["original_shape"]
        x = data.view(original_shape)

        return x


class EdenUnbiasedCompressor(EdenBaseCompressor, UnbiasedCompressor, InputVarianceCompressor):
    def get_scale(self, x: FloatTensor, unscaled_centers_vec: FloatTensor) -> float:
        return sum_squares(x) / (unscaled_centers_vec @ x) / self.p

    def omega(self) -> float:
        if self.p < 1:
            return torch.pi / (2 * self.p * self.bits) - 1
        vars = bits_var()
        if self.fractional_bits:
            (
                1
                / (
                    (self.bits_high - self.bits) * (1 - vars[self.bits_low])
                    + (self.bits - self.bits_low) * (1 - vars[self.bits_high])
                )
                - 1
            )
        else:
            return 1 / (1 - vars[self.bits]) - 1

    def ab(self) -> (float, float):
        if self.fractional_bits or self.bits != 1:
            raise NotImplementedError("Only implemented for bits=1")

        if self.world_size is None:
            raise AttributeError("world_size must be set for Input Variance")

        return (torch.pi / 2 - 1) / self.world_size, 0


class EdenContractiveCompressor(EdenBaseCompressor, ContractiveCompressor):
    def get_scale(self, x: FloatTensor, unscaled_centers_vec: FloatTensor) -> float:
        return l2(x) / math.sqrt(x.numel())

    def alpha(self) -> float:
        vars = bits_var()
        if self.p < 1:
            return 1 - (1 - self.bits) * vars[0] - (self.bits - 0) * vars[1]
        if self.fractional_bits:
            return (
                1
                - (self.bits_high - self.bits) * vars[self.bits_low]
                - (self.bits - self.bits_low) * vars[self.bits_high]
            )
        else:
            return 1 - vars[self.bits]


### Hadamard


def hadamard_transform_(vec):
    """fast Walsh–Hadamard transform (in-place)

    :param vec: vec is expected to be a power of 2!
    :return: the Hadamard transform of vec
    """
    d = vec.numel()
    original_shape = vec.shape
    h = 2
    while h <= d:
        hf = h // 2
        vec = vec.view(d // h, h)

        ## the following is a more inplace way of doing the following:
        # half_1 = batch[:, :, :hf]
        # half_2 = batch[:, :, hf:]
        # batch = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)
        # the NOT inplace seems to be actually be slightly faster
        # (I assume for making more memory-contiguous operations. That being said,
        # it more easily throws out-of-memory and may slow things overall,
        # so using inplace version below:)

        vec[:, :hf] = vec[:, :hf] + vec[:, hf : 2 * hf]
        vec[:, hf : 2 * hf] = vec[:, :hf] - 2 * vec[:, hf : 2 * hf]
        h *= 2

    vec *= d**-0.5  # vec /= np.sqrt(d)

    return vec.view(*original_shape)


def rademacher_like(x, generator):
    """(previously random_diagonal)"""
    return 2 * torch.torch.empty_like(x).bernoulli_(generator=generator) - 1


def randomized_hadamard_transform_(x, generator):
    d = rademacher_like(x, generator)

    return hadamard_transform_(x * d)


def inverse_randomized_hadamard_transform_(tx, generator):
    d = rademacher_like(tx, generator)

    return hadamard_transform_(tx) * d


### Quantization
def gen_boundaries(centroids):
    return [(a + b) / 2 for a, b in zip(centroids[:-1], centroids[1:])]


# half-normal centroids
opt_hn_centroids = {
    1: [0.7978845608028654],
    2: [0.4527800398860679, 1.5104176087114887],
    3: [0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335],
    4: [
        0.12839501671105813,
        0.38804823445328507,
        0.6567589957631145,
        0.9423402689122875,
        1.2562309480263467,
        1.6180460517130526,
        2.069016730231837,
        2.732588804065177,
    ],
    5: [
        0.06588962234909321,
        0.1980516892038791,
        0.3313780514298761,
        0.4666991751197207,
        0.6049331689395434,
        0.7471351317890572,
        0.89456439585444,
        1.0487823813655852,
        1.2118032120324,
        1.3863389353626248,
        1.576226389073775,
        1.7872312118858462,
        2.0287259913633036,
        2.3177364021261493,
        2.69111557955431,
        3.260726295605043,
    ],
    6: [
        0.0334094558802581,
        0.1002781217139195,
        0.16729660990171974,
        0.23456656976873475,
        0.3021922894403614,
        0.37028193328115516,
        0.4389488009177737,
        0.5083127587538033,
        0.5785018460645791,
        0.6496542452315348,
        0.7219204720694183,
        0.7954660529025513,
        0.870474868055092,
        0.9471530930156288,
        1.0257343133937524,
        1.1064859596918581,
        1.1897175711327463,
        1.2757916223519965,
        1.3651378971823598,
        1.458272959944728,
        1.5558274659528346,
        1.6585847114298427,
        1.7675371481292605,
        1.8839718992293555,
        2.009604894545278,
        2.146803022259123,
        2.2989727412973995,
        2.471294740528467,
        2.6722617014102585,
        2.91739146530985,
        3.2404166403241677,
        3.7440690236964755,
    ],
    7: [
        0.016828143177728235,
        0.05049075396896167,
        0.08417241989671888,
        0.11788596825032507,
        0.1516442630131618,
        0.18546025708680833,
        0.21934708340331643,
        0.25331807190834565,
        0.2873868062260947,
        0.32156710392315796,
        0.355873075050329,
        0.39031926330596733,
        0.4249205523979007,
        0.4596922300454219,
        0.49465018161031576,
        0.5298108436256188,
        0.565191195643323,
        0.600808970989236,
        0.6366826613981411,
        0.6728315674936343,
        0.7092759460939766,
        0.746037126679468,
        0.7831375375631398,
        0.8206007832455021,
        0.858451939611374,
        0.896717615963322,
        0.9354260757626341,
        0.9746074842160436,
        1.0142940678300427,
        1.054520418037026,
        1.0953237719213182,
        1.1367442623434032,
        1.1788252655205043,
        1.2216138763870124,
        1.26516137869917,
        1.309523700469555,
        1.3547621051156036,
        1.4009441065262136,
        1.448144252238147,
        1.4964451375010575,
        1.5459387008934842,
        1.596727786313424,
        1.6489283062238074,
        1.7026711624156725,
        1.7581051606756466,
        1.8154009933798645,
        1.8747553268072956,
        1.9363967204122827,
        2.0005932433837565,
        2.0676621538384503,
        2.1379832427349696,
        2.212016460501213,
        2.2903268704925304,
        2.3736203164211713,
        2.4627959084523208,
        2.5590234991374485,
        2.663867022558051,
        2.7794919110540777,
        2.909021527386642,
        3.0572161028423737,
        3.231896182843021,
        3.4473810105937095,
        3.7348571053691555,
        4.1895219330235225,
    ],
    8: [
        0.008445974137017219,
        0.025338726226901278,
        0.042233889994651476,
        0.05913307399220878,
        0.07603788791797023,
        0.09294994306815242,
        0.10987089037069565,
        0.12680234584461386,
        0.1437459285205906,
        0.16070326074968388,
        0.1776760066764216,
        0.19466583496246115,
        0.21167441946986007,
        0.22870343946322488,
        0.24575458029044564,
        0.2628295721769575,
        0.2799301528634766,
        0.29705806782573063,
        0.3142150709211129,
        0.3314029639954903,
        0.34862355883476864,
        0.3658786774238477,
        0.3831701926964899,
        0.40049998943716425,
        0.4178699650069057,
        0.4352820704086704,
        0.45273827097956804,
        0.4702405882876,
        0.48779106011037887,
        0.505391740756901,
        0.5230447441905988,
        0.5407522460590347,
        0.558516486141511,
        0.5763396823538222,
        0.5942241184949506,
        0.6121721459546814,
        0.6301861414640443,
        0.6482685527755422,
        0.6664219019236218,
        0.684648787627676,
        0.7029517931200633,
        0.7213336286470308,
        0.7397970881081071,
        0.7583450032075904,
        0.7769802937007926,
        0.7957059197645721,
        0.8145249861674053,
        0.8334407494351099,
        0.8524564651728141,
        0.8715754936480047,
        0.8908013031010308,
        0.9101374749919184,
        0.9295877653215154,
        0.9491559977740125,
        0.9688461234581733,
        0.9886622867721733,
        1.0086087121824747,
        1.028689768268861,
        1.0489101021225093,
        1.0692743940997251,
        1.0897875553561465,
        1.1104547388972044,
        1.1312812154370708,
        1.1522725891384287,
        1.173434599389649,
        1.1947731980672593,
        1.2162947131430126,
        1.238005717146854,
        1.2599130381874064,
        1.2820237696510286,
        1.304345369166531,
        1.3268857708606756,
        1.349653145284911,
        1.3726560932224416,
        1.3959037693197867,
        1.419405726021264,
        1.4431719292973744,
        1.4672129964566984,
        1.4915401336751468,
        1.5161650628244996,
        1.541100284490976,
        1.5663591473033147,
        1.5919556551358922,
        1.6179046397057497,
        1.6442219553485078,
        1.6709244249695359,
        1.6980300628044107,
        1.7255580190748743,
        1.7535288357430767,
        1.7819645728459763,
        1.81088895442524,
        1.8403273195729115,
        1.870306964218662,
        1.9008577747790962,
        1.9320118435829472,
        1.9638039107009146,
        1.9962716117712092,
        2.0294560760505993,
        2.0634026367482017,
        2.0981611002741527,
        2.133785932225919,
        2.170336784741086,
        2.2078803102947337,
        2.2464908293749546,
        2.286250990303635,
        2.327254033532845,
        2.369604977942217,
        2.4134218838650208,
        2.458840003415269,
        2.506014300608167,
        2.5551242195294983,
        2.6063787537827645,
        2.660023038604595,
        2.716347847697055,
        2.7757011083910723,
        2.838504606698991,
        2.9052776685316117,
        2.976670770545963,
        3.0535115393558603,
        3.136880130166507,
        3.2282236667414654,
        3.3295406612081644,
        3.443713971315384,
        3.5751595986789093,
        3.7311414987004117,
        3.9249650523739246,
        4.185630113705256,
        4.601871059539151,
    ],
}


def gen_all_normal_quantization_constants():
    # add symmetric negative normal centroids
    centroids = {i: [-j for j in reversed(c)] + c for i, c in opt_hn_centroids.items()}

    # centroids to bin boundaries
    boundaries = {i: gen_boundaries(c) for i, c in centroids.items()}

    return centroids, boundaries


def gen_all_ee_centroids_and_boundaries():
    # deltas
    deltas = {
        0.1: 5.071514195296913,
        0.2: 4.45202601258643,
        0.3: 4.043459049426019,
        0.4: 3.7248574872501194,
        0.5: 3.4561779181240126,
        0.6: 3.218733147950843,
        0.7: 3.0020635211258195,
        0.8: 2.7996155444998294,
        0.9: 2.607049875659868,
        1: 2.4216194555629045,
        2: 1.0824465435871389,
        3: 0.5224332449870417,
        4: 0.25901674387569074,
        5: 0.12923770176712424,
        6: 0.06458514949372329,
        7: 0.032288366192005924,
        8: 0.01614365715340682,
    }

    ### half-normal centroids
    centroids = {}
    boundaries = {}

    for b, delta in deltas.items():
        centroids[b] = [delta * j for j in range(-1000, 1001)]

        boundaries[b] = gen_boundaries(centroids[b])

    return centroids, boundaries


QUANTIZATION_CONSTANTS = {
    "max_lloyd": gen_all_normal_quantization_constants(),
    "ee": gen_all_ee_centroids_and_boundaries(),
}


@lru_cache(maxsize=None)
def get_all_quantization_constants_tensors(q_type, device, /):
    centroids, boundaries = QUANTIZATION_CONSTANTS[q_type]

    centroids = {i: torch.tensor(c, device=device) for i, c in centroids.items()}
    boundaries = {i: torch.tensor(b, device=device) for i, b in boundaries.items()}

    return centroids, boundaries


def bernoulli_mask(shape, device, p, generator):
    return torch.empty(shape, dtype=torch.bool, device=device).bernoulli_(p=p, generator=generator)


def mask_split(x, mask):
    x0 = torch.masked_select(x, torch.logical_not(mask))
    x1 = torch.masked_select(x, mask)
    return x0, x1


def mask_combine(x0, x1, mask):
    x = torch.empty(mask.shape, dtype=x0.dtype, device=x0.device)
    x.masked_scatter_(torch.logical_not(mask), x0)
    x.masked_scatter_(mask, x1)

    return x


def sum_squares(x):
    return torch.sum(x**2)


def l2(x):
    return torch.sqrt(sum_squares(x))


def section_variance(a, b, c) -> float:
    if math.isinf(c):
        return (
            np.sqrt(2 / np.pi) * np.exp(-(a**2) / 2) * (a - 2 * b)
            - (b**2 + 1) * erf(a / np.sqrt(2))
            + (b**2 + 1) * erf(c / np.sqrt(2))
        )
    else:
        return (
            np.sqrt(2 / np.pi) * np.exp(-(a**2) / 2) * (a - 2 * b)
            - (b**2 + 1) * erf(a / np.sqrt(2))
            + (b**2 + 1) * erf(c / np.sqrt(2))
            + np.sqrt(2 / np.pi) * np.exp(-(c**2) / 2) * (2 * b - c)
        )


@cache
def bits_var():
    result = {0: 1}
    for bits, centers in opt_hn_centroids.items():
        borders = [0] + [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])] + [float("inf")]
        variance = sum(section_variance(a, b, c) for a, b, c in zip(borders[:-1], centers, borders[1:]))
        result[bits] = variance
    return result
