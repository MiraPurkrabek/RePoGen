# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

from .utils import to_tensor


class VertexJointSelector(nn.Module):
    def __init__(
        self, vertex_ids=None, use_hands=True, use_feet_keypoints=True, **kwargs
    ):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array(
            [
                vertex_ids["nose"],
                vertex_ids["reye"],
                vertex_ids["leye"],
                vertex_ids["rear"],
                vertex_ids["lear"],
            ],
            dtype=np.int64,
        )

        extra_joints_idxs = np.concatenate([extra_joints_idxs, face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array(
                [
                    vertex_ids["LBigToe"],
                    vertex_ids["LSmallToe"],
                    vertex_ids["LHeel"],
                    vertex_ids["RBigToe"],
                    vertex_ids["RSmallToe"],
                    vertex_ids["RHeel"],
                ],
                dtype=np.int32,
            )

            extra_joints_idxs = np.concatenate([extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ["thumb", "index", "middle", "ring", "pinky"]

            tips_idxs = []
            for hand_id in ["l", "r"]:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate([extra_joints_idxs, tips_idxs])

        self.register_buffer(
            "extra_joints_idxs", to_tensor(extra_joints_idxs, dtype=torch.long)
        )

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(
            vertices, 1, self.extra_joints_idxs.to(torch.long)
        )  # The '.to(torch.long)'.
        # added to make the trace work in c++,
        # otherwise you get a runtime error in c++:
        # 'index_select(): Expected dtype int32 or int64 for index'
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints
