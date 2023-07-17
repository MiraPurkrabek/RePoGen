# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import numpy as np
import torch


def from_torch(x, dtype=np.float32):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x.astype(dtype)
