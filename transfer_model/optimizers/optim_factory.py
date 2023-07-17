# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import sys

from typing import NewType, List, Dict

import torch
import torch.optim as optim
from loguru import logger
from torchtrustncg import TrustRegion

Tensor = NewType("Tensor", torch.Tensor)


def build_optimizer(parameters: List[Tensor], optim_cfg: Dict) -> Dict:
    """Creates the optimizer"""
    optim_type = optim_cfg.get("type", "sgd")
    logger.info(f"Building: {optim_type.title()}")

    num_params = len(parameters)
    parameters = list(filter(lambda x: x.requires_grad, parameters))
    if num_params != len(parameters):
        logger.info(f"Some parameters have requires_grad off")

    if optim_type == "adam":
        optimizer = optim.Adam(parameters, **optim_cfg.get("adam", {}))
        create_graph = False
    elif optim_type == "lbfgs" or optim_type == "lbfgsls":
        optimizer = optim.LBFGS(parameters, **optim_cfg.get("lbfgs", {}))
        create_graph = False
    elif optim_type == "trust_ncg" or optim_type == "trust-ncg":
        optimizer = TrustRegion(parameters, **optim_cfg.get("trust_ncg", {}))
        create_graph = True
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(parameters, **optim_cfg.get("rmsprop", {}))
        create_graph = False
    elif optim_type == "sgd":
        optimizer = optim.SGD(parameters, **optim_cfg.get("sgd", {}))
        create_graph = False
    else:
        raise ValueError(f"Optimizer {optim_type} not supported!")
    return {"optimizer": optimizer, "create_graph": create_graph}


def build_scheduler(optimizer, sched_type="exp", lr_lambda=0.1, **kwargs):
    if lr_lambda <= 0.0:
        return None

    if sched_type == "exp":
        return optim.lr_scheduler.ExponentialLR(optimizer, lr_lambda)
    else:
        raise ValueError("Unknown learning rate" + " scheduler: ".format(sched_type))
