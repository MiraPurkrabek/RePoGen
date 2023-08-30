# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scipy.spatial import distance_matrix

import pyrender
import trimesh
import repogen
import cv2

from repogen.random_pose.pose_and_view_generation import (
    generate_pose,
    random_camera_pose,
    JOINTS_LIMS_DEGS,
)
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

DO_TSNE = True
DO_UMAP = False
DO_CLOSEST_POSE = False
DO_KMEANS = False
DO_JOINTS_RANGES = False
DO_SMPL_ERROR = False

def main():
    
    AMASS_PATH = "../AMASS/data/"

    print("Generating poses and views...")

    all_body_poses = []
    filepaths = []
    datasets = []

    all_body_poses = np.load("AMASS_stats/all_body_poses.npy").astype(np.float32).reshape((-1, 63))
    filepaths = np.load("AMASS_stats/filepaths.npy")
    datasets = np.load("AMASS_stats/datasets.npy")

    if len(all_body_poses) == 0:
        for dir in os.listdir(AMASS_PATH):
            if not os.path.isdir(os.path.join(AMASS_PATH, dir)):
                continue

            # if dir == "CNRS":
            #     continue

            amass_subdir = os.path.join(AMASS_PATH, dir)

            for subdir in os.listdir(amass_subdir):
                if not os.path.isdir(os.path.join(amass_subdir, subdir)):
                    continue

                amass_subdir = os.path.join(amass_subdir, subdir)

                for filename in os.listdir(amass_subdir):

                    if not filename.endswith(".npz"):
                        continue

                    print("Processing {:s}, ({:d})".format(
                        os.path.join(amass_subdir, filename), len(all_body_poses)+1,
                    ))

                    amass_poses = np.load(os.path.join(amass_subdir, filename))

                    try:
                        n_poses = len(amass_poses["trans"])
                    except KeyError:
                        continue

                    body_poses = []

                    # Take 100 poses from each sequence
                    fps = n_poses // 20
                    with tqdm(total=n_poses//fps, ascii=True) as progress_bar:
                        for i in range(0, n_poses, fps):                       
                            body_pose = amass_poses["poses"][i, 3:66]#.reshape((-1, 1))
                            body_poses.append(body_pose)
                            datasets.append(dir)
                            filepaths.append(os.path.join(amass_subdir, filename))

                            progress_bar.update()

                    all_body_poses.append(body_poses)

                    # if len(all_body_poses) > 4:
                    #     break
        all_body_poses = np.concatenate(all_body_poses, axis=0)
        filepaths = np.array(filepaths)
        datasets = np.array(datasets)
        np.save("AMASS_stats/all_body_poses.npy", all_body_poses)
        np.save("AMASS_stats/filepaths.npy", np.array(filepaths))
        np.save("AMASS_stats/datasets.npy", np.array(datasets))

    print("All body poses: {}".format(all_body_poses.shape))
    all_body_poses_deg = all_body_poses / np.pi * 180

    # Randomly sample poses from the AMASS dataset for finetuning
    # for n in [500, 1000, 3000]:
    #     sampled_poses = all_body_poses_deg[np.random.choice(len(all_body_poses_deg), n, replace=False), :]
    #     print("Saving randomly sampled {:d} poses...".format(n))
    #     np.save("AMASS_stats/sampled_poses_{:d}.npy".format(n), sampled_poses)

    try:
        synth_poses = np.load("AMASS_stats/synth_poses.npy")
        synth_poses_deg = synth_poses / np.pi * 180
    except FileNotFoundError:
        synth_poses = np.concatenate([
            np.array(generate_pose(
                simplicity=1.0,
                extreme_poses=False,
                # typical_pose="stand",
            )) for _ in range(len(all_body_poses_deg)//3)
        ], axis=0)
        np.save("AMASS_stats/synth_poses.npy", synth_poses)
        synth_poses_deg = synth_poses / np.pi * 180
    synth_poses_deg = synth_poses_deg[:len(all_body_poses_deg)//7]
    
    print(all_body_poses_deg.shape, synth_poses_deg.shape)
    
    poses_to_draw = np.concatenate([all_body_poses_deg, synth_poses_deg], axis=0)

    os.makedirs("AMASS_stats", exist_ok=True)

    if DO_TSNE:
        X_embedded = None

        # try:
        #     X_embedded = np.load("AMASS_stats/tsne.npy")
        #     synth_X_embedded = np.load("AMASS_stats/synth_tsne.npy")
        # except FileNotFoundError:
        #     X_embedded = None
        #     synth_X_embedded = None

        # t-SNE projection
        if X_embedded is None:
            print("Projecting poses to 2D space using the t-SNE...")
            tsne = TSNE(
                n_components=2,
                learning_rate='auto',
                init='random',
                perplexity=3,
            )
            X_embedded = tsne.fit_transform(poses_to_draw)
            synth_X_embedded = X_embedded[-len(synth_poses_deg):]
            X_embedded = X_embedded[:-len(synth_poses_deg)]

            # np.save("AMASS_stats/tsne.npy", X_embedded)
            # np.save("AMASS_stats/synth_tsne.npy", synth_X_embedded)

        filepaths = np.array(filepaths)
        if HAS_PLOTLY:
            traces = []

            datasets = np.array(["AMASS" for _ in range(len(datasets))])

            for dataset_name in np.unique(datasets):
                idx = datasets == dataset_name
                trace_amass = go.Scatter(
                    x=X_embedded[idx, 0],
                    y=X_embedded[idx, 1],
                    name=dataset_name.upper(),
                    mode="markers",
                    marker=dict(
                        size=5,
                        opacity=0.5,
                        color="blue",
                    ),
                    text=filepaths,
                )
                traces.append(trace_amass)
            trace_synth = go.Scatter(
                x=synth_X_embedded[:, 0],
                y=synth_X_embedded[:, 1],
                name="RePoGen",
                mode="markers",
                marker=dict(
                    size=7,
                    opacity=0.5,
                    color="red",
                ),
                marker_symbol="cross",
            )
            traces.insert(0, trace_synth)

            fig = go.Figure(data=traces)
            fig.write_html("AMASS_stats/tsne.html")
            print("Poses projected.")
        else:
            for filepath in np.unique(filepaths):
                idx = filepaths == filepath
                plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], marker="o", alpha=0.3)#, color="blue")
            # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="o", color="blue", alpha=0.5)
            plt.scatter(synth_X_embedded[:, 0], synth_X_embedded[:, 1], color="red", marker="x", alpha=0.3)
            plt.grid(True)
            plt.legend(np.unique(filepaths).tolist() + ["RePoGen"])
            plt.title("t-SNE projection of the poses")    
            plt.savefig("AMASS_stats/tsne.png")
            print("Poses projected.")

    # Umap projection
    if HAS_UMAP and DO_UMAP:
        print("Projecting poses to 2D space using the UMAP...")
        reducer = umap.UMAP()
        X_embedded = reducer.fit_transform(poses_to_draw)
        synth_X_embedded = X_embedded[-len(synth_poses_deg):]
        X_embedded = X_embedded[:-len(synth_poses_deg)]

        if HAS_PLOTLY:
            traces = []
            for dataset_name in np.unique(datasets):
                idx = datasets == dataset_name
                trace_amass = go.Scatter(
                    x=X_embedded[idx, 0],
                    y=X_embedded[idx, 1],
                    name=dataset_name.upper(),
                    mode="markers",
                    marker=dict(
                        size=5,
                        opacity=0.5,
                    ),
                    text=filepaths,
                )
                traces.append(trace_amass)
            trace_synth = go.Scatter(
                x=synth_X_embedded[:, 0],
                y=synth_X_embedded[:, 1],
                name="RePoGen",
                mode="markers",
                marker=dict(
                    size=7,
                    opacity=0.5,
                    color="red",
                ),
                marker_symbol="cross",
            )
            traces.insert(0, trace_synth)

            fig = go.Figure(data=traces)
            fig.write_html("AMASS_stats/umap.html")
            print("Poses projected.")


    if DO_KMEANS:
        # K-means clustering
        print("Clustering poses using the K-means...")
        kmeans = KMeans(n_clusters=10, random_state=0).fit(all_body_poses_deg.reshape((-1, 63)).astype(np.float32))
        centers = kmeans.cluster_centers_
        distances = np.linalg.norm(poses_to_draw.reshape((-1, 63)).astype(np.float32)[:, None, :] - centers[None, :, :], axis=2)
        distances = np.min(distances, axis=1)
        amass_distances = distances[:-len(synth_poses_deg)]
        synth_distances = distances[-len(synth_poses_deg):]
        dist_threshold = np.mean(amass_distances) + 2 * np.std(amass_distances)
        print("Poses clustered.")

        print("K-means of RePoGen poses: {:5.1f}, {:5.1f}, {:5.1f}\t{:4.1f}% of all poses is rare".format(
            np.min(synth_distances), np.mean(synth_distances), np.max(synth_distances),
            100*np.sum(synth_distances > dist_threshold)/len(synth_distances),
        ))
        print("K-means of AMASS poses  : {:5.1f}, {:5.1f}, {:5.1f}\t{:4.1f}% of all poses is rare".format(
            np.min(amass_distances), np.mean(amass_distances), np.max(amass_distances),
            100*np.sum(amass_distances > dist_threshold)/len(amass_distances),
        ))

    if DO_CLOSEST_POSE or DO_SMPL_ERROR:
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ])
        model = repogen.create(
            "models",
            model_type="smplx",
            gender="male",
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
        )
    
    if DO_CLOSEST_POSE:
        # Find the RepoGen pose that has the highest distance to the closest AMASS pose
        try:
            m = np.load("AMASS_stats/distance_matrix.npy")
        except FileNotFoundError:
            print("Comptuing the distance matrix...")
            m = distance_matrix(synth_poses_deg, all_body_poses_deg)
            np.save("AMASS_stats/distance_matrix.npy", m)
        print("Distance matrix shape:", m.shape)
        min_distances_idx = np.argmin(m, axis=1)
        min_distances = np.min(m, axis=1)
        max_distance_idx = np.argmax(min_distances)
        max_distance = np.max(min_distances)
        print("The RepoGen pose that has the highest distance to the closest AMASS pose is pose {:d} with distance {:5.1f} to {:d} from {:s} dataset".format(
            max_distance_idx,
            max_distance,
            min_distances_idx[max_distance_idx],
            datasets[min_distances_idx[max_distance_idx]],
        ))

        # Render and save the two poses
        max_distance_idx_sorted = np.argsort(min_distances)[::-1]
        os.makedirs("AMASS_stats/pose_comparison", exist_ok=True)
        print("Rendering the closest poses...")
        for i in tqdm(range(20)):
            max_distance_idx = max_distance_idx_sorted[i]

            max_distance_pose = synth_poses_deg[max_distance_idx]#.reshape((-1, 3))
            max_distance_amass_pose = all_body_poses_deg[min_distances_idx[max_distance_idx]]#.reshape((-1, 3))
            
            max_distance_pose = max_distance_pose / 180 * np.pi
            max_distance_amass_pose = max_distance_amass_pose / 180 * np.pi

            # np.save("AMASS_stats/max_distance_pose.npy", max_distance_pose)
            # np.save("AMASS_stats/max_distance_amass_pose.npy", max_distance_amass_pose)

            # Render the poses
            # camera_pose = random_camera_pose(view_preference="FRONT", distance=2.0)
            out_img = show_closest_pose(
                max_distance_pose,
                max_distance_amass_pose,
                camera_pose,
                model,
                pose1_source="RePoGen",
                pose2_source=datasets[min_distances_idx[max_distance_idx]],
            )
            out_img = cv2.putText(
                    out_img,
                    "{:.1f}".format(min_distances[max_distance_idx]),
                    (1950, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite("AMASS_stats/pose_comparison/max_distance_poses_{:02d}.png".format(i), out_img)
        for i in tqdm(range(20)):
            max_distance_idx = max_distance_idx_sorted[-i-1]

            max_distance_pose = synth_poses_deg[max_distance_idx]#.reshape((-1, 3))
            max_distance_amass_pose = all_body_poses_deg[min_distances_idx[max_distance_idx]]#.reshape((-1, 3))
            
            max_distance_pose = max_distance_pose / 180 * np.pi
            max_distance_amass_pose = max_distance_amass_pose / 180 * np.pi

            # np.save("AMASS_stats/max_distance_pose.npy", max_distance_pose)
            # np.save("AMASS_stats/max_distance_amass_pose.npy", max_distance_amass_pose)

            # Render the poses
            # camera_pose = random_camera_pose(view_preference="FRONT", distance=2.0)
            out_img = show_closest_pose(
                max_distance_pose,
                max_distance_amass_pose,
                camera_pose,
                model,
                pose1_source="RePoGen",
                pose2_source=datasets[min_distances_idx[max_distance_idx]],
            )
            out_img = cv2.putText(
                    out_img,
                    "{:.1f}".format(min_distances[max_distance_idx]),
                    (1950, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite("AMASS_stats/pose_comparison/min_distance_poses_{:02d}.png".format(i), out_img)
    
    if DO_SMPL_ERROR:
        # Render and save some AMASS poses with the weiredt left knee angle
        joint_idx = 17
        left_knee_idxs = np.argsort(all_body_poses_deg[:, joint_idx*3+1])
        os.makedirs("AMASS_stats/left_knee", exist_ok=True)
        print("Rendering the weirdest left knee poses...")
        for i in tqdm(range(5), ascii=True):
            idx = left_knee_idxs[i]
            pose = all_body_poses_deg[idx]
            pose = pose / 180 * np.pi
            out_img = show_closest_pose(
                pose,
                pose,
                camera_pose,
                model,
                pose1_source=datasets[idx],
                pose2_source=str(all_body_poses_deg[idx, joint_idx*3+1]),
            )
            cv2.imwrite("AMASS_stats/left_knee/left_knee_min_{:02d}.png".format(i), out_img)
        for i in tqdm(range(5), ascii=True):
            idx = left_knee_idxs[-i-1]
            pose = all_body_poses_deg[idx]
            pose = pose / 180 * np.pi
            out_img = show_closest_pose(
                pose,
                pose,
                camera_pose,
                model,
                pose1_source=datasets[idx],
                pose2_source=str(all_body_poses_deg[idx, joint_idx*3+1]),
            )
            cv2.imwrite("AMASS_stats/left_knee/left_knee_max_{:02d}.png".format(i), out_img)

    if DO_JOINTS_RANGES:
        print("Saving histograms for each joint...")
        print_joints_ranges(all_body_poses_deg, save_dir="AMASS_stats/joints_ranges_amass")
        print_joints_ranges(synth_poses_deg, save_dir="AMASS_stats/joints_ranges_synth")
        print("Histograms saved.")


def show_closest_pose(pose1, pose2, camera_pose, model, pose1_source=None, pose2_source=None):
    pose1_torch = torch.from_numpy(pose1).unsqueeze(0)
    pose2_torch = torch.from_numpy(pose2).unsqueeze(0)
    poses_torch = torch.cat([pose1_torch, pose2_torch], dim=0)
    
    betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    expression = torch.randn([1, model.num_betas], dtype=torch.float32)
    output1 = model(
        betas=betas, expression=expression, return_verts=True,
        body_pose=pose1_torch,
    )
    output2 = model(
        betas=betas, expression=expression, return_verts=True,
        body_pose=pose2_torch,
    )

    vertices1 = output1.vertices.detach().cpu().numpy().squeeze()
    vertices2 = output2.vertices.detach().cpu().numpy().squeeze()
    
    imgs = []
    for v, src in zip([vertices1, vertices2], [pose1_source, pose2_source]):
        # scene = pyrender.Scene(bg_color=(100, 100, 100))
        scene = pyrender.Scene(bg_color=(255, 255, 255))
        tri_mesh = trimesh.Trimesh(v, model.faces, vertex_colors=(100, 100, 230))
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh)
        light = pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=5)
        # scene.add(light, pose=random_camera_pose(distance=20.0))#, view_preference="BOTTOM"))
        # scene.add(light, pose=random_camera_pose(distance=20.0))#, view_preference="BOTTOM"))
        # scene.add(light, pose=random_camera_pose(distance=20.0))#, view_preference="TOP"))
        scene.add(light, pose=camera_pose)#, view_preference="TOP"))
        # scene.add(light, pose=random_camera_pose(distance=20.0, view_preference="FRONT"))
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)
        scene.add(camera, pose=camera_pose)
        r = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024)
        color, _ = r.render(scene)
        color = color.astype(np.uint8)

        if src is not None:
            color = cv2.putText(
                color,
                src,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

        imgs.append(color[:, :, ::-1])

    out_img = cv2.hconcat(imgs)
    return out_img


def print_joints_ranges(poses, save_dir="joints_ranges", silent=True):
    mean_body_pose = np.mean(poses, axis=0)
    max_body_pose = np.max(poses, axis=0)
    min_body_pose = np.min(poses, axis=0)

    joints = JOINTS_LIMS_DEGS

    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(joints)):
        joint_name = list(joints.keys())[i]
        if not silent:
            print("{:s}:".format(joint_name))
        for j in range(3):
            my_lims = joints[joint_name][j]

            if not silent:
                print("\t({:6.2f}, {:6.2f}, {:6.2f})  --  {}".format(
                    min_body_pose[i*3+j],
                    mean_body_pose[i*3+j],
                    max_body_pose[i*3+j],
                    my_lims,
                ))
            
            # Plot histogram
            max_lims = (
                min(my_lims[0], min_body_pose[i*3+j]),
                max(my_lims[1], max_body_pose[i*3+j]),
            )
            plt.gca().cla()
            plt.hist(
                poses[:, i*3+j],
                bins=100,
                density=True,
                range=max_lims,
            )
            
            # Plot left side of the skewed Gaussian
            std_dev = (0 - my_lims[0]) / 2
            if std_dev == 0:
                plt.plot(my_lims[0], 0.01, color="red", marker="o")
            else:
                x = np.linspace(my_lims[0], 0, 50)
                y = 1/(std_dev * 2 * np.pi) * np.exp(-1/2*(x/std_dev)**2)
                plt.plot(x, y, color="red")

            # Plot right side of the skewed Gaussian
            std_dev = (my_lims[1]) / 2
            if std_dev == 0:
                plt.plot(my_lims[1], 0.01, color="red", marker="o")
            else:
                x = np.linspace(0, my_lims[1], 50)
                y = 1/(std_dev * 2 * np.pi) * np.exp(-1/2*(x/std_dev)**2)
                plt.plot(x, y, color="red")

            # plt.plot(my_lims[1], 1, color="red", marker="o")
            plt.xlim([
                min(my_lims[0], min(my_lims[1], min_body_pose[i*3+j])) -10,
                max(my_lims[0], max(my_lims[1], max_body_pose[i*3+j])) + 10,
            ])
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "{:s}_{:d}.png".format(joint_name, j)))


if __name__ == "__main__":
    
    main()
