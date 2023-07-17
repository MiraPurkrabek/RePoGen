import os
import argparse
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

from repogen.random_pose.pose_and_view_generation import random_camera_pose
from repogen.random_pose.visualizations import draw_points_on_sphere

from repogen.view_regressor.data_processing import c2s
from repogen.view_regressor.visualizations import plot_heatmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument(
        "--distance",
        action="store_true",
        default=False,
        help="If True, will plot distance from origin instead of position on sphere",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        default=False,
        help="If True, will plot distance from origin instead of position on sphere",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=False,
        help="If True, will plot 2D heatmap instead of 3D scatter plot",
    )
    return parser.parse_args()


def main(args):
    oks_score = []
    areas = []
    vis_kpts = []
    bbox_sizes = []
    bboxes = []
    image_ids = []

    input_dict = json.load(open(args.filepath, "r"))
    for annotation in input_dict["annotations"]:
        kpts = np.array(annotation["keypoints"]).reshape(-1, 3)
        counts = np.zeros(3)
        for i in range(3):
            counts[i] = np.sum(kpts[:, 2] == i)
        if np.sum(counts[1:]) == 0:
            continue

        # if counts[2] < 10:
        #     continue

        vis_kpts.append(counts.flatten())

        oks_score.append(annotation["oks"])

        areas.append(annotation["area"])

        bbox = np.array(annotation["bbox"])
        bboxes.append(bbox)
        bbox_sizes.append(bbox[2] * bbox[3])
        image_ids.append(annotation["image_id"])

    areas = np.array(areas)
    vis_kpts = np.array(vis_kpts).squeeze().astype(int)
    bbox_sizes = np.array(bbox_sizes)
    bboxes = np.array(bboxes)
    oks_score = np.array(oks_score)
    image_ids = np.array(image_ids)

    # Filter out the NaN ones
    nan_idx = np.isnan(oks_score)
    areas = areas[~nan_idx]
    vis_kpts = vis_kpts[~nan_idx, :]
    bbox_sizes = bbox_sizes[~nan_idx]
    bboxes = bboxes[~nan_idx, :]
    oks_score = oks_score[~nan_idx]
    image_ids = image_ids[~nan_idx]

    argsorted_bbox_sizes = np.argsort(bbox_sizes)
    for idx in argsorted_bbox_sizes[:50]:
        print(
            "bbox size: {:.2f},\tbbox: {},\timage id: {},\toks: {:.4f}".format(
                bbox_sizes[idx],
                bboxes[idx, :],
                image_ids[idx],
                oks_score[idx],
            )
        )

    dist = areas
    xlabel = "Area of the segmentaion mask [{:d}, {:d}]".format(
        int(np.max(dist)), int(np.min(dist))
    )

    # dist = vis_kpts[:, 2].flatten()
    # xlabel = "Number of visible keypoints"

    # dist = bbox_sizes.flatten()
    # xlabel = "Size of the bounding box [{:d}, {:d}]".format(int(np.max(dist)), int(np.min(dist)))

    plt.xlim([1.05 * np.max(dist), -0.05 * np.max(dist)])

    sort_idx = np.argsort(dist)
    sorted_score = oks_score[sort_idx]
    window_size = len(oks_score) // 100
    tmp = np.convolve(sorted_score, np.ones(window_size) / window_size, mode="valid")
    tmp_x = np.linspace(np.min(dist), np.max(dist), len(tmp))
    plt.scatter(dist, oks_score)
    plt.xlabel(xlabel)
    plt.ylabel("OKS score")
    plt.plot(tmp_x, tmp, "r-")
    plt.legend(["OKS score", "Moving average"])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
