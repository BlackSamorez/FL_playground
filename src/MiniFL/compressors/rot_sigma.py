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

    vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
    vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
    h *= 2

  vec *= d ** -0.5  # vec /= np.sqrt(d)

  return vec.view(*original_shape)