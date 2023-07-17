import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch

from repogen.view_regressor.data_processing import (
    process_keypoints,
    load_data_from_coco_file,
    occlude_random_keypoints,
    occlude_keypoints_with_rectangle,
    randomly_occlude_keypoints,
)
from visualizations import visualize_pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "coco_filepath", type=str, help="Filename of the coco annotations file"
    )

    return parser.parse_args()


def main(args):
    # Load the data
    keypoints, bboxes_xywh, image_ids = load_data_from_coco_file(args.coco_filepath)
    keypoints = process_keypoints(keypoints, bboxes_xywh)

    idx = 0
    pose = visualize_pose(keypoints[idx, :], has_bbox=True)
    cv2.imshow("Pose {:d}".format(idx), pose[:, :, ::-1])
    while True:
        if idx >= keypoints.shape[0]:
            idx = keypoints.shape[0] - 1

        if idx < 0:
            idx = 0

        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = cv2.waitKey(100)
        if k == ord("m") or k == 83:  # toggle current image
            idx += 1
            cv2.destroyAllWindows()
            pose = visualize_pose(keypoints[idx, :], has_bbox=True)
            cv2.imshow("Pose {:d}".format(idx), pose[:, :, ::-1])
        elif k == ord("n") or k == 81:
            idx -= 1
            cv2.destroyAllWindows()
            pose = visualize_pose(keypoints[idx, :], has_bbox=True)
            cv2.imshow("Pose {:d}".format(idx), pose[:, :, ::-1])
        elif k == ord("o"):
            cv2.destroyAllWindows()
            pose = visualize_pose(
                occlude_random_keypoints(keypoints[idx, :].squeeze()), has_bbox=True
            )
            cv2.imshow("Pose {:d}".format(idx), pose[:, :, ::-1])
        elif k == ord("p"):
            cv2.destroyAllWindows()
            pose = visualize_pose(
                occlude_keypoints_with_rectangle(keypoints[idx, :].squeeze()),
                has_bbox=True,
            )
            cv2.imshow("Pose {:d}".format(idx), pose[:, :, ::-1])
        elif k == ord("r"):
            cv2.destroyAllWindows()
            pose = visualize_pose(
                randomly_occlude_keypoints(keypoints[idx, :].squeeze()), has_bbox=True
            )
            cv2.imshow("Pose {:d}".format(idx), pose[:, :, ::-1])
        elif k == ord("e") or k == ord("q"):  # escape key
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
