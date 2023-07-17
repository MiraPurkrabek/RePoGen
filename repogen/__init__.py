# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from .body_models import (
    create,
    SMPL,
    SMPLH,
    SMPLX,
    MANO,
    FLAME,
    build_layer,
    SMPLLayer,
    SMPLHLayer,
    SMPLXLayer,
    MANOLayer,
    FLAMELayer,
)

from .joint_names import (
    COCO_JOINTS,
    COCO_SKELETON,
)
