# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import os
import os.path as osp
import pickle

import numpy as np
import torch
from loguru import logger

from .typing import Tensor


def read_deformation_transfer(
    deformation_transfer_path: str,
    device=None,
    use_normal: bool = False,
) -> Tensor:
    ''' Reads a deformation transfer
    '''
    if device is None:
        device = torch.device('cpu')
    assert osp.exists(deformation_transfer_path), (
        'Deformation transfer path does not exist:'
        f' {deformation_transfer_path}')
    logger.info(
        f'Loading deformation transfer from: {deformation_transfer_path}')
    # Read the deformation transfer matrix
    with open(deformation_transfer_path, 'rb') as f:
        def_transfer_setup = pickle.load(f, encoding='latin1')
    if 'mtx' in def_transfer_setup:
        def_matrix = def_transfer_setup['mtx']
        if hasattr(def_matrix, 'todense'):
            def_matrix = def_matrix.todense()
        def_matrix = np.array(def_matrix, dtype=np.float32)
        if not use_normal:
            num_verts = def_matrix.shape[1] // 2
            def_matrix = def_matrix[:, :num_verts]
    elif 'matrix' in def_transfer_setup:
        def_matrix = def_transfer_setup['matrix']
    else:
        valid_keys = ['mtx', 'matrix']
        raise KeyError(f'Deformation transfer setup must contain {valid_keys}')

    def_matrix = torch.tensor(def_matrix, device=device, dtype=torch.float32)
    return def_matrix


def apply_deformation_transfer(
    def_matrix: Tensor,
    vertices: Tensor,
    faces: Tensor,
    use_normals=False
) -> Tensor:
    ''' Applies the deformation transfer on the given meshes
    '''
    if use_normals:
        raise NotImplementedError
    else:
        def_vertices = torch.einsum('mn,bni->bmi', [def_matrix, vertices])
        return def_vertices
