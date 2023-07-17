# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from typing import List, Tuple
import sys

import torch
import torch.utils.data as dutils
from .datasets import MeshFolder

from loguru import logger


def build_dataloader(exp_cfg):
    dset_name = exp_cfg.datasets.name
    if dset_name == "mesh-folder":
        mesh_folder_cfg = exp_cfg.datasets.mesh_folder
        key, *_ = mesh_folder_cfg.keys()
        value = mesh_folder_cfg[key]
        logger.info(f"{key}: {value}\n")
        dataset = MeshFolder(**mesh_folder_cfg)
    else:
        raise ValueError(f"Unknown dataset: {dset_name}")

    batch_size = exp_cfg.batch_size
    num_workers = exp_cfg.datasets.num_workers

    logger.info(f"Creating dataloader with B={batch_size}, workers={num_workers}")
    dataloader = dutils.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return {"dataloader": dataloader, "dataset": dataset}
