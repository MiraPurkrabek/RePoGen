# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]


COCO_JOINTS = {
    "nose": {"idx": 75-20, "range": 300},               # 0
    "left eye": {"idx": 77-20, "range": 700},           # 1
    "right eye": {"idx": 76-20, "range": 700},          # 2
    "left ear": {"idx": 79-20, "range": 500},           # 3
    "right ear": {"idx": 78-20, "range": 500},          # 4
    "left shoulder": {"idx": 36-20, "range": 200},      # 5
    'right_shoulder': {"idx": 37-20, "range": 200},     # 6
    'left_elbow': {"idx": 38-20, "range": 150},         # 7
    'right_elbow': {"idx": 39-20, "range": 150},        # 8
    'left_wrist': {"idx": 40-20, "range": 100},         # 9
    'right_wrist': {"idx": 41-20, "range": 100},        # 10
    'left_hip': {"idx": 21-20, "range": 200},           # 11
    'right_hip': {"idx": 22-20, "range": 200},          # 12
    'left_knee': {"idx": 24-20, "range": 100},          # 13
    'right_knee': {"idx": 25-20, "range": 100},         # 14
    'left_ankle': {"idx": 27-20, "range": 50},          # 15
    'right_ankle': {"idx": 28-20, "range": 50},         # 16
}

COCO_SKELETON = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [ 6, 12],
        [ 7, 13],
        [ 6,  7],
        [ 6,  8],
        [ 7,  9],
        [ 8, 10],
        [ 9, 11],
        [ 2,  3],
        [ 1,  2],
        [ 1,  3],
        [ 2,  4],
        [ 3,  5],
        [ 4,  6],
        [ 5,  7],
]

LEFT_LEG_COLOR = [0, 255, 0]
LEFT_ARM_COLOR = [150, 255, 0]
LEFT_FACE_COLOR = [223, 255, 0]

RIGHT_LEG_COLOR = [0, 0, 255]
RIGHT_ARM_COLOR = [0, 150, 255]
RIGHT_FACE_COLOR = [0, 255, 255]

TORSO_COLOR = [255, 150, 0]

COCO_SKELETON_COLORS = [
    LEFT_LEG_COLOR,        # Left ankle - Left knee
    LEFT_LEG_COLOR,        # Left knee - Left hip
    RIGHT_LEG_COLOR,       # Right ankle - Right knee
    RIGHT_LEG_COLOR,       # Right knee - Right hip
    TORSO_COLOR,      # Left hip - Right hip
    TORSO_COLOR,      # Left hip - Left shoulder
    TORSO_COLOR,      # Right hip - Right shoulder
    TORSO_COLOR,      # Left shoulder - Right shoulder
    LEFT_ARM_COLOR,        # Left shoulder - Left elbow
    RIGHT_ARM_COLOR,      # Right shoulder - Right elbow
    LEFT_ARM_COLOR,        # Left elbow - Left wrist
    RIGHT_ARM_COLOR,      # Right elbow - Right wrist
    TORSO_COLOR,      # Left eye - Right eye
    LEFT_FACE_COLOR,      # Nose - Left eye
    RIGHT_FACE_COLOR,      # Nose - Right eye
    LEFT_FACE_COLOR,      # Left eye - Left ear
    RIGHT_FACE_COLOR,      # Right eye - Right ear
    LEFT_FACE_COLOR,      # Left ear - Left shoulder
    RIGHT_FACE_COLOR,      # Right ear - Right shoulder
]

OPENPOSE_SKELETON = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18]]#, [3, 17], [6, 18]]

OPENPOSE_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

SMPLH_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


class Body:
    """
    Class for storing a single body pose.
    """

    def __init__(self, joints, joint_names):
        assert joints.ndim > 1
        assert joints.shape[0] == len(joint_names)
        self.joints = {}
        for i, j in enumerate(joint_names):
            self.joints[j] = joints[i]

    @staticmethod
    def from_smpl(joints):
        """
        Create a Body object from SMPL joints.
        """
        return Body(joints, SMPL_JOINT_NAMES)

    @staticmethod
    def from_smplh(joints):
        """
        Create a Body object from SMPLH joints.
        """
        return Body(joints, SMPLH_JOINT_NAMES)

    def _as(self, joint_names):
        """
        Return a Body object with the specified joint names.
        """
        joint_list = []
        for j in joint_names:
            if j not in self.joints:
                joint_list.append(np.zeros_like(self.joints["spine1"]))
            else:
                joint_list.append(self.joints[j])
        return np.stack(joint_list, axis=0)

    def as_smpl(self):
        """
        Convert the body to SMPL joints.
        """
        return self._as(SMPL_JOINT_NAMES)

    def as_smplh(self):
        """
        Convert the body to SMPLH joints.
        """
        return self._as(SMPLH_JOINT_NAMES)
