# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


def to_np(array, dtype=np.float32):
    if hasattr(array, 'todense'):
        array = array.todense()
    return np.array(array, dtype=dtype)
