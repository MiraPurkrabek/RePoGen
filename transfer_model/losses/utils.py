# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import torch

def get_reduction_method(reduction='mean'):
    if reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum
    elif reduction == 'none':
        return lambda x: x
    else:
        raise ValueError('Unknown reduction method: {}'.format(reduction))
