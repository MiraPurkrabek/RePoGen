# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import os
import os.path as osp
import sys
import pickle

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm

from repogen import build_layer

from .config import parse_args
from .data import build_dataloader
from .transfer_model import run_fitting
from .utils import read_deformation_transfer, np_mesh_to_o3d


def main() -> None:
    exp_cfg = parse_args()

    if torch.cuda.is_available() and exp_cfg["use_cuda"]:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if exp_cfg["use_cuda"]:
            if input("use_cuda=True and GPU is not available, using CPU instead,"
                     " would you like to continue? (y/n)") != "y":
                sys.exit(3)

    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    data_obj_dict = build_dataloader(exp_cfg)

    dataloader = data_obj_dict['dataloader']

    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        var_dict = run_fitting(
            exp_cfg, batch, body_model, def_matrix, mask_ids)
        paths = batch['paths']

        for ii, path in enumerate(paths):
            _, fname = osp.split(path)

            output_path = osp.join(
                output_folder, f'{osp.splitext(fname)[0]}.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(var_dict, f)

            output_path = osp.join(
                output_folder, f'{osp.splitext(fname)[0]}.obj')
            mesh = np_mesh_to_o3d(
                var_dict['vertices'][ii], var_dict['faces'])
            o3d.io.write_triangle_mesh(output_path, mesh)


if __name__ == '__main__':
    main()
