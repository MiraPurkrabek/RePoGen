# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.
#  from yacs.config import CfgNode as CN

from typing import List, Tuple, Union
from omegaconf import OmegaConf
from loguru import logger
from dataclasses import dataclass, make_dataclass


@dataclass
class LossTemplate:
    type: str = 'l2'
    active: bool = False
    weight: Tuple[float] = (0.0,)
    requires_grad: bool = True
    enable: int = 0


@dataclass
class LossConfig:
    type: str = 'smplify-x'


conf = OmegaConf.structured(LossConfig)
