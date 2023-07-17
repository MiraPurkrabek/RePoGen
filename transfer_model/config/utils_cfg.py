# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from typing import Tuple
from dataclasses import dataclass


@dataclass
class Variable:
    create: bool = True
    requires_grad: bool = True


@dataclass
class Pose(Variable):
    type: str = "aa"
