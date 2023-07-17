# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from typing import Tuple, Optional
from copy import deepcopy
#  from yacs.config import CfgNode as CN
from dataclasses import dataclass
from omegaconf import OmegaConf

from .loss_defaults import conf as loss_cfg, LossConfig
from .dataset_defaults import conf as dataset_cfg, DatasetConfig
from .optim_defaults import conf as optim_cfg, OptimConfig
from .body_model_defaults import conf as body_model_cfg, BodyModelConfig


@dataclass
class EdgeFitting:
    per_part: bool = False
    reduction: str = 'mean'


@dataclass
class VertexFitting:
    per_part: bool = False
    reduction: str = 'mean'
    type: str = 'l2'


@dataclass
class Config:
    use_cuda: bool = True
    log_file: str = '/tmp/logs'
    output_folder: str = 'output'
    save_verts: bool = True
    save_joints: bool = True
    save_mesh: bool = False
    save_img_summaries: bool = True
    summary_steps: int = 5
    degrees: Tuple[float] = (90,)
    float_type: str = 'float'
    logger_level: str = 'INFO'
    interactive: bool = True
    batch_size: Optional[int] = 1
    color_path: str = 'data/smpl_with_colors.ply'

    optim: OptimConfig = optim_cfg
    datasets: DatasetConfig = dataset_cfg
    losses: LossConfig = loss_cfg
    body_model: BodyModelConfig = body_model_cfg

    deformation_transfer_path: str = ''
    mask_ids_fname: str = ''

    per_part: bool = True
    edge_fitting: EdgeFitting = EdgeFitting()


conf = OmegaConf.structured(Config)
