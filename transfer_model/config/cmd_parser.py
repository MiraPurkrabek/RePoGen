# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from __future__ import absolute_import
from __future__ import division

import sys
import os

import argparse
from loguru import logger

from omegaconf import OmegaConf
from .defaults import conf as default_conf


def parse_args(argv=None) -> OmegaConf:
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = "Model transfer script"
    parser = argparse.ArgumentParser(
        formatter_class=arg_formatter, description=description
    )

    parser.add_argument(
        "--exp-cfg",
        type=str,
        dest="exp_cfg",
        help="The configuration of the experiment",
    )
    parser.add_argument(
        "--exp-opts",
        default=[],
        dest="exp_opts",
        nargs="*",
        help="Command line arguments",
    )

    cmd_args = parser.parse_args()

    cfg = default_conf.copy()
    if cmd_args.exp_cfg:
        cfg.merge_with(OmegaConf.load(cmd_args.exp_cfg))
    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))

    return cfg
