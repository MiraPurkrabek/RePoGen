# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import numpy as np
import torch


def v2v(x, y):
    if torch.is_tensor(x):
        return (x - y).pow(2).sum(dim=-1).sqrt().mean()
    else:
        return np.sqrt(np.power(x - y, 2)).sum(axis=-1).mean()
