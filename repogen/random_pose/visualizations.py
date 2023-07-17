import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from repogen.joint_names import COCO_SKELETON, OPENPOSE_SKELETON, OPENPOSE_COLORS


def draw_depth(img, depthmap):
    overlay_img = img.copy()
    depthmap = depthmap > 0
    overlay_img[depthmap, :] = [0, 0, 255]
    alpha = 0.6
    return cv2.addWeighted(img, alpha, overlay_img, 1-alpha, 0)


def draw_pose(img, kpts, joints_vis, draw_style="custom"):
    img = img.copy()

    assert draw_style in [
        "custom",
        "openpose",
        "openpose_vis",
    ]

    joints_vis = joints_vis.astype(bool)
    skeleton = COCO_SKELETON

    if draw_style.startswith("openpose"):
        # Reorder kpts to OpenPose order
        kpts = kpts.copy()
        kpts = kpts[[0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], :]
        joints_vis = joints_vis[[0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
        
        # Compute pelvis as mean of shoulders
        kpts[1, :] = np.mean(kpts[[2, 5], :], axis=1)
        joints_vis[1] = np.all(joints_vis[[2, 5]])

        skeleton = OPENPOSE_SKELETON

    for pi, pt in enumerate(kpts):
        
        if draw_style == "openpose":
            img = cv2.circle(
                img,
                tuple(pt.tolist()),
                radius=4,
                color=OPENPOSE_COLORS[pi],
                thickness=-1
            )
        elif draw_style == "openpose_vis":
            if joints_vis[pi]:
                img = cv2.circle(
                    img,
                    tuple(pt.tolist()),
                    radius=4,
                    color=OPENPOSE_COLORS[pi],
                    thickness=-1
                )
        else:
            marker_color = (0, 0, 255) if joints_vis[pi] else (40, 40, 40)
            thickness = 2 if joints_vis[pi] else 1
            marker_type = cv2.MARKER_CROSS
        
            img = cv2.drawMarker(
                img,
                tuple(pt.tolist()),
                color=marker_color,
                markerType=marker_type,
                thickness=thickness
            )

    for bi, bone in enumerate(skeleton):
        b = np.array(bone) - 1 # COCO_SKELETON is 1-indexed
        start = kpts[b[0], :]
        end = kpts[b[1], :]
        if draw_style.startswith("openpose"):
            
            if draw_style == "openpose_vis" and not (joints_vis[b[0]] and joints_vis[b[1]]):
                continue
            
            stickwidth = 4
            current_img = img.copy()
            mX = np.mean(np.array([start[0], end[0]]))
            mY = np.mean(np.array([start[1], end[1]]))
            length = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(start[1] - end[1], start[0] - end[0]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(current_img, polygon, OPENPOSE_COLORS[bi])
            img = cv2.addWeighted(img, 0.4, current_img, 0.6, 0)

        else:
            if not (joints_vis[b[0]] and joints_vis[b[1]]):
                continue
            
            img = cv2.line(
                img,
                start,
                end,
                thickness=1,
                color=(0, 0, 255)
            )

    return img


def draw_points_on_sphere(pts, score=None, show_axes=True):
    assert len(pts.shape) == 2
    assert pts.shape[1] == 3

    if score is not None:
        assert len(score) == pts.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    max_value = np.max(np.abs(pts))
    max_value = max(1, max_value)

    if show_axes:
        axis_size = 1.5*max_value
        x_line = np.array([[0, axis_size], [0, 0], [0, 0]])
        ax.plot(x_line[0, :], x_line[1, :], x_line[2, :], c='r', linewidth=5)
        y_line = np.array([[0, 0], [0, axis_size], [0, 0]])
        ax.plot(y_line[0, :], y_line[1, :], y_line[2, :], c='g', linewidth=5)
        z_line = np.array([[0, 0], [0, 0], [0, axis_size]])
        ax.plot(z_line[0, :], z_line[1, :], z_line[2, :], c='b', linewidth=5)

    if score is not None:
        score = np.clip(score, 0, 1)
        # mask = score >= 0
        # score = score[mask]
        # pts = pts[mask, :]
        # print("Score: min={}, max={}".format(np.min(score), np.max(score)))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=1-score, marker='o', cmap='rainbow')
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='c', marker='o')

    ax.set_xlim(-max_value, max_value)
    ax.set_ylim(-max_value, max_value)
    ax.set_zlim(-max_value, max_value)

    plt.show()
