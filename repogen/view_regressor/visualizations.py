import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from repogen.joint_names import COCO_JOINTS, COCO_SKELETON, COCO_SKELETON_COLORS

from repogen.view_regressor.data_processing import s2c, c2s

def plot_testing_data(y_test_pred, is_spherical=False):
    
    if is_spherical:
        radiuses = np.random.uniform(1, 5, size=y_test_pred.shape[0])
        y_test_pred = np.concatenate([radiuses[:, None], y_test_pred], axis=1)
        y_test_pred = s2c(y_test_pred)

    # Plot the predicted positions
    fig = plt.figure(figsize=(6.4*2, 4.8*2))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], label="Predicted positions (n={:d})".format(len(y_test_pred)))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    max_value = np.max(np.abs(y_test_pred))
    axis_size = 1.*max_value
    x_line = np.array([[0, axis_size], [0, 0], [0, 0]])
    ax.plot(x_line[0, :], x_line[1, :], x_line[2, :], c='r', linewidth=5)
    y_line = np.array([[0, 0], [0, axis_size], [0, 0]])
    ax.plot(y_line[0, :], y_line[1, :], y_line[2, :], c='g', linewidth=5)
    z_line = np.array([[0, 0], [0, 0], [0, axis_size]])
    ax.plot(z_line[0, :], z_line[1, :], z_line[2, :], c='b', linewidth=5)

    plt.show()


def plot_heatmap(pts, is_spherical=False, return_img=False, return_angles=False):

    if is_spherical:
        if pts.shape[1] == 2:
            data_theta = pts[:, 0].squeeze()
            data_phi = pts[:, 1].squeeze()
            radiuses = np.ones(data_theta.shape)
        else:
            radiuses = pts[:, 0].squeeze()
            data_theta = pts[:, 1].squeeze()
            data_phi = pts[:, 2].squeeze()
    else:
        spherical = c2s(pts)
        radiuses = spherical[:, 0]
        data_theta = spherical[:, 1].squeeze()
        data_phi = spherical[:, 2].squeeze()
    radius = np.mean(radiuses)

    # print("Theta: {:.3f}\t{:.3f}\t{:.3f}".format(np.min(data_theta), np.mean(data_theta), np.max(data_theta)))
    # print("Phi: {:.3f}\t{:.3f}\t{:.3f}".format(np.min(data_phi), np.mean(data_phi), np.max(data_phi)))

    # Print theta and phi shape

    significant_points = {
        # Format: (theta, phi, marker)
        "TOP": (np.pi/2, np.pi/2, "ro"),        # [0, 1, 0] in cartesian
        "BOTTOM": (np.pi/2, -np.pi/2, "rx"),    # [0, -1, 0] in cartesian
        "FRONT": (0, 0, "bo"),                  # [0, 0, 1] in cartesian
        "BACK": (np.pi, 0, "bx"),               # [0, 0, -1] in cartesian
        "LEFT": (np.pi/2, 0, "co"),             # [1, 0, 0] in cartesian
        "RIGHT": (np.pi/2, np.pi, "cx"),        # [-1, 0, 0] in cartesian
    }

    if return_img:
        fig = plt.figure(figsize=(6.4*2, 4.8*2))
        plt.hist2d(data_theta, data_phi, bins=100)
        for key, sp in significant_points.items():
            mkr = sp[2]
            plt.plot(sp[0], sp[1], mkr, label=key)
        plt.axis("equal")
        plt.axis('off')

        # Remove the huge white borders
        plt.margins(0)
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        heatmap_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_img = heatmap_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        if return_angles:
            return heatmap_img, data_theta, data_phi
        else:
            return heatmap_img

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4*2, 4.8*2))

        # ax1.hexbin(data_theta, data_phi, gridsize=100)
        ax1.hist2d(data_theta, data_phi, bins=100)
        
        for key, sp in significant_points.items():
            mkr = sp[2]
            ax1.plot(sp[0], sp[1], mkr, label=key)
        # plt.colorbar()
        
        ax1.legend()
        ax1.axis("equal")
        ax1.set_xlabel("theta")
        ax1.set_ylabel("phi")
        plt.suptitle("Distribution of samples, average distance = {:.2f}".format(radius))
        
        ax2.hist(radiuses, bins=100)
        ax2.grid()
        ax2.set_xlabel("radius")
        ax2.set_ylabel("count")
        
        plt.savefig(os.path.join(
            "images",
            "heatmaps",
            "heatmap_distance_{:.1f}.png".format(radius)
        ))
        plt.show()

        if return_angles:
            return data_theta, data_phi


def plot_training_data(epochs, lr, train_loss_log, test_loss_log, test_positions, y_test_pred, spherical=False):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4*2, 4.8*2))
    
    # Plot the training and test loss
    if not train_loss_log == [] and not test_loss_log == []:
        ax1.plot(np.arange(epochs), train_loss_log, label="Train loss")
        test_epochs = np.linspace(0, epochs-1, len(test_loss_log))
        # test_epochs += test_epochs[1]
        ax1.plot(test_epochs, test_loss_log, label="Test loss")
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

    # Plot the distance of test samples
    if test_positions.shape[1] == 2:
        theta_pos = test_positions[:, 0] - np.pi/2
        phi_pos = test_positions[:, 1]
        theta_y = y_test_pred[:, 0] - np.pi/2
        phi_y = y_test_pred[:, 1]
        dlon = theta_pos - theta_y
        dlat = phi_pos - phi_y
        a = np.sin(dlat/2.0)**2 + np.cos(phi_pos) * np.cos(phi_y) * np.sin(dlon/2.0)**2
        test_dist = 2 * np.arcsin(np.sqrt(a))
    elif spherical:
        test_dist = np.linalg.norm(s2c(test_positions) - s2c(y_test_pred), axis=1)
    else:
        test_dist = np.linalg.norm(test_positions - y_test_pred, axis=1)
    sorted_test_dist = np.sort(test_dist)
    ax2.plot(np.arange(len(sorted_test_dist)), sorted_test_dist, label="Distance of test samples")
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel("L2 Distance")
    
    fig.suptitle("Training data (lr={:.4f}, epochs={:d})".format(lr, epochs))
    plt.savefig("training_data.png")
    plt.show()


    # If the predicted positions are not in spherical coordinates, plot them in the 3D
    if test_positions.shape[1] > 2:

        # Plot the predicted positions
        fig = plt.figure(figsize=(6.4*2, 4.8*2))
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(test_positions[:, 0], test_positions[:, 1], test_positions[:, 2], label="True positions")
        # ax.scatter(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], label="Predicted positions")

        # Draw arrows
        for start, end, dist in zip(test_positions, y_test_pred, test_dist):
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="black", 
                    linestyle="dashed", linewidth=dist)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Predicted positions (lr={:.4f})".format(lr))
        ax.legend()

        max_value = np.max(np.abs(np.stack([test_positions, y_test_pred], axis=0)))
        axis_size = 1.5*max_value
        x_line = np.array([[0, axis_size], [0, 0], [0, 0]])
        ax.plot(x_line[0, :], x_line[1, :], x_line[2, :], c='r', linewidth=5)
        y_line = np.array([[0, 0], [0, axis_size], [0, 0]])
        ax.plot(y_line[0, :], y_line[1, :], y_line[2, :], c='g', linewidth=5)
        z_line = np.array([[0, 0], [0, 0], [0, axis_size]])
        ax.plot(z_line[0, :], z_line[1, :], z_line[2, :], c='b', linewidth=5)

        plt.show()


def visualize_pose(keypoints, has_bbox=False):
    keypoints = keypoints.copy()
    
    if keypoints.ndim < 2:
        keypoints = keypoints.reshape(-1, 2)

    if has_bbox:
        keypoints = keypoints[:-1, :]

    if keypoints.shape[0] > 17:
        keypoints = keypoints[:17, :]
    
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    
    assert np.all(x >= 0)
    assert np.all(y >= 0)

    if np.max(keypoints[:, :2]) <= 1:
        keypoints[:, :2] *= 512
        max_w = 512
        max_h = 512
    else:
        max_w = np.max(x)
        max_h = np.max(y)

    img = np.zeros((int(max_h)+1, int(max_w)+1, 3), dtype=np.uint8)

    for kpt in keypoints:
        if kpt[0] == 0 and kpt[1] == 0:
            continue
        
        img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), 4, (255, 0, 0), -1)
        
    for bone, color in zip(COCO_SKELETON, COCO_SKELETON_COLORS):
        bone = np.array(bone) - 1

        if ((keypoints[bone[0], 0] == 0 and keypoints[bone[0], 1] == 0) or
            (keypoints[bone[1], 0] == 0 and keypoints[bone[1], 1] == 0)):
            continue
        
        img = cv2.line(
            img,
            (int(keypoints[bone[0], 0]), int(keypoints[bone[0], 1])),
            (int(keypoints[bone[1], 0]), int(keypoints[bone[1], 1])),
            color,
            2
        )

    return img


if __name__ == "__main__":
    kpts = np.random.uniform(size=(17, 3)) * 640
    kpts[:, 2] = 2
    visualize_pose(kpts)